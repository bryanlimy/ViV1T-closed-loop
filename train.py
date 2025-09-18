import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from shutil import rmtree
from time import time

import numpy as np
import torch
import wandb
from ray.air import session
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t import Checkpoint
from viv1t import data
from viv1t import metrics
from viv1t.criterions import Criterion
from viv1t.criterions import get_criterion
from viv1t.data import CycleDataloaders
from viv1t.model import Model
from viv1t.model import get_model
from viv1t.optimizer import get_optimizer
from viv1t.utils import utils
from viv1t.utils import yaml
from viv1t.utils.estimate_batch_size import estimate_batch_size

SKIP = 50  # skip the first 50 frames in metric calculation


def train_step(
    mouse_id: str,
    batch: dict[str, torch.Tensor],
    model: Model,
    optimizer: Optimizer,
    criterion: Criterion,
    update: bool,
    micro_batch_size: int,
    total_batch_size: int | torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result, responses = {}, {"y_true": [], "y_pred": []}
    batch_loss = torch.tensor(0.0, device=device, requires_grad=False)
    for micro_batch in data.micro_batching(batch, micro_batch_size):
        y_pred, _ = model(
            inputs=micro_batch["video"].to(device),
            mouse_id=mouse_id,
            behaviors=micro_batch["behavior"].to(device),
            pupil_centers=micro_batch["pupil_center"].to(device),
        )
        y_true = micro_batch["response"][..., -y_pred.size(2) :].to(device)
        loss = criterion(y_true=y_true, y_pred=y_pred, mouse_id=mouse_id)
        loss = loss / total_batch_size
        loss.backward()
        batch_loss += loss.detach()
        responses["y_true"].append(y_true.detach())
        responses["y_pred"].append(y_pred.detach())
    if update:
        result["grad_norm"] = model.clip_grad_norm()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    result["loss/loss"] = batch_loss.cpu()
    result["metrics/correlation"] = metrics.challenge_correlation(
        y_true=torch.vstack(responses["y_true"]),
        y_pred=torch.vstack(responses["y_pred"]),
    ).cpu()
    del responses
    return result


def train(
    args,
    ds: CycleDataloaders,
    model: Model,
    optimizer: Optimizer,
    criterion: Criterion,
    epoch: int,
):
    results = {mouse_id: {} for mouse_id in args.mouse_ids}
    grad_norms = []
    num_mouse = len(args.mouse_ids)
    # the total batch size for one gradient update step
    total_batch_size = torch.tensor(
        num_mouse * args.batch_size, dtype=torch.float32, device=args.device
    )
    model = model.to(args.device)
    model.train(True)
    if hasattr(optimizer, "train"):
        optimizer.train()
    optimizer.zero_grad(set_to_none=True)
    for i, (mouse_id, mouse_batch) in enumerate(
        tqdm(ds, desc="Train", disable=args.verbose < 2)
    ):
        result = train_step(
            mouse_id=mouse_id,
            batch=mouse_batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            update=(i + 1) % num_mouse == 0,
            micro_batch_size=args.micro_batch_size,
            total_batch_size=total_batch_size,
            device=args.device,
        )
        if "grad_norm" in result:
            grad_norms.append(result.pop("grad_norm"))
        utils.update_dict(results[mouse_id], result)
    if wandb.run is not None and grad_norms:
        wandb.log({"grad_norm": np.mean(grad_norms).item()}, step=epoch)
    return utils.log_metrics(results)


@torch.inference_mode()
def validation_step(
    mouse_id: str,
    batch: dict[str, torch.Tensor],
    model: Model,
    criterion: Criterion,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    result = {}
    t = batch["response"].shape[-1] - SKIP
    y_pred, _ = model(
        inputs=batch["video"].to(device),
        mouse_id=mouse_id,
        behaviors=batch["behavior"].to(device),
        pupil_centers=batch["pupil_center"].to(device),
    )
    y_true, y_pred = batch["response"][..., -t:].to(device), y_pred[..., -t:]
    result["loss/loss"] = criterion(y_true=y_true, y_pred=y_pred, mouse_id=mouse_id)
    return result, {"y_true": y_true, "y_pred": y_pred, "video_ids": batch["video_id"]}


def validate(
    args,
    ds: dict[str, DataLoader],
    model: Model,
    optimizer: Optimizer,
    criterion: Criterion,
):
    model = model.to(args.device)
    model.train(False)
    if hasattr(optimizer, "eval"):
        optimizer.eval()
    results = {}
    with tqdm(desc="Val", total=data.num_steps(ds), disable=args.verbose < 2) as pbar:
        for mouse_id, mouse_ds in ds.items():
            mouse_result, mouse_responses = {}, {}
            for batch in mouse_ds:
                batch_result, responses = validation_step(
                    mouse_id=mouse_id,
                    batch=batch,
                    model=model,
                    criterion=criterion,
                    device=args.device,
                )
                utils.update_dict(mouse_result, batch_result)
                utils.update_dict(mouse_responses, responses)
                del responses
                pbar.update(1)
            mouse_result.update(
                metrics.compute_metrics(
                    y_true=torch.vstack(mouse_responses["y_true"]),
                    y_pred=torch.vstack(mouse_responses["y_pred"]),
                    video_ids=torch.cat(mouse_responses["video_ids"]),
                )
            )
            results[mouse_id] = mouse_result
            del mouse_result, mouse_responses
    results = utils.log_metrics(results)
    if torch.isnan(results["loss"]) or torch.abs(results["correlation"]) > 1:
        print(f"NaN loss and/or correlation larger than 1 detected.")
        results["correlation"] = torch.nan
    return results


def main(args, wandb_sweep: bool = False, ray_sweep: bool = False) -> int:
    """
    Main training and validation loop
    Args:
        args: argparse.Namespace, command line arguments
        wandb_sweep: bool, whether the script is called from wandb sweep
        ray_sweep: bool, whether the script is called from ray sweep
    Returns:
        exit_code: 0 if training is successful, 1 if training is terminated with error
    """
    if args.wandb is not None:
        utils.wandb_init(args)

    if args.clear_output_dir and args.output_dir.is_dir():
        rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.device = utils.get_device(args.device)
    utils.set_random_seed(args.seed, deterministic=args.deterministic)

    data.get_mouse_ids(args)
    data.set_neuron_idx(args)

    estimate_batch_size(args)
    if args.micro_batch_size < 1:
        return 1  # end run with error

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.data_dir,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        val_batch_size=1,
        device=args.device,
    )

    model = get_model(args, ds=train_ds, model_info=True)
    optimizer, scheduler = get_optimizer(args, model=model)
    checkpoint = Checkpoint(args, model=model, optimizer=optimizer, scheduler=scheduler)
    if scheduler is not None:
        scheduler.set_checkpoint(checkpoint)

    criterion = get_criterion(args, ds=train_ds)

    utils.save_args(args)
    epoch = checkpoint.restore(load_optimizer=True, load_scheduler=True)

    if wandb.run is not None:
        utils.wandb_update_config(args, wandb_sweep=wandb_sweep)

    train_ds = CycleDataloaders(train_ds)

    model = model.to(args.device)
    if args.compile:
        model.compile()

    best_corr, best_epoch = 0.0, epoch
    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_result = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        val_result = validate(
            args,
            ds=val_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )
        elapse = time() - start

        if args.verbose:
            print(
                f'Train\t\tloss: {train_result["loss"]:.02f}\t\t'
                f'correlation: {train_result["correlation"]:.04f}\n'
                f'Validation\tloss: {val_result["loss"]:.02f}\t\t'
                f'correlation: {val_result["correlation"]:.04f}\n'
                f"Elapse: {elapse:.02f}s"
            )

        if torch.isnan(train_result["loss"]) or torch.isnan(val_result["loss"]):
            print(f"NaN loss detected, stop training.\n")
            if best_epoch == 0 and wandb.run is not None:
                # terminate run if no improvement was made since the beginning
                wandb.finish(exit_code=1)  # end run with error
                return 1
            break

        if val_result["correlation"] > best_corr:
            best_corr, best_epoch = val_result["correlation"], epoch
            checkpoint.save(value=best_corr, epoch=epoch)

        if args.schedule_free:
            terminate = (epoch - best_epoch) >= 10
        else:
            terminate = scheduler.step(value=val_result["correlation"], epoch=epoch)

        report = {
            "train_loss": train_result["loss"],
            "train_corr": train_result["correlation"],
            "val_loss": val_result["loss"],
            "val_corr": val_result["correlation"],
            "val_poisson": val_result["poisson_loss"],
            "single_trial_correlation": val_result["single_trial_correlation"],
            "norm_corr": val_result["normalized_correlation"],
            "best_corr": best_corr,
            "learning_rate": optimizer.param_groups[-1]["lr"],
            "elapse": elapse,
            "epoch": epoch,
        }
        if wandb.run is not None:
            wandb.log(report, step=epoch)
        if ray_sweep:
            session.report(report)
        if terminate:
            print(f"No improvement in {epoch - best_epoch} epochs, stop training.\n")
            break
        del train_result, val_result

    torch.cuda.empty_cache()

    # create a clean model and restore best weights
    print("\nInitialize a new model and restore best weights.")
    model = get_model(args, ds=val_ds, model_info=False)
    checkpoint = Checkpoint(args, model=model)
    checkpoint.restore(force=True)

    eval_result = utils.evaluate(args, ds=val_ds, model=model)
    # report to wandb
    report = {"best_corr": eval_result["correlation"]["average"], "epoch": epoch}
    # result to save to yaml file
    result = {"validation": eval_result}
    test_result = {}
    for k in test_ds.keys():
        test_result[k] = utils.evaluate(args, ds=test_ds[k], model=model)
        report |= {
            f"{k}_corr": test_result[k]["correlation"]["average"],
            f"{k}_trial_corr": test_result[k]["single_trial_correlation"]["average"],
            f"{k}_norm_corr": test_result[k]["normalized_correlation"]["average"],
        }
        result |= {k: test_result[k]}
    yaml.save(args.output_dir / "evaluation.yaml", data=result)

    if args.verbose:
        print_result = lambda d: "\t".join([f"{k}: {v:.04f}" for k, v in d.items()])
        statement = "\nValidation\t"
        statement += print_result(eval_result["correlation"])
        for k in test_result.keys():
            statement += f"\n{k.capitalize()}\t"
            statement += print_result(test_result[k]["correlation"])
        print(statement)
        print(f"\nResults saved to {args.output_dir}.")

    if ray_sweep:
        session.report(report)
    if wandb.run is not None:
        wandb.log(report, step=epoch)
        wandb.finish(exit_code=0)

    return 0  # end run peacefully


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    # dataset settings
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument(
        "--transform_input",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="input transformation\n"
        "0: no transformation\n"
        "1: standardize input\n"
        "2: normalize input\n",
    )
    parser.add_argument(
        "--transform_output",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="output transformation\n"
        "0: no transformation\n"
        "1: standardize output per neuron\n"
        "2: normalize output per neuron\n",
    )
    parser.add_argument(
        "--center_crop",
        type=float,
        default=1.0,
        help="center crop the video frame to center_crop percentage.",
    )
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=str,
        default=None,
        help="Mouse to use for training. By default we use all 10 mice from "
        "the Sensorium 2023 dataset",
    )
    parser.add_argument(
        "--limit_data",
        type=int,
        default=None,
        help="limit the number of training samples.",
    )
    parser.add_argument(
        "--limit_neurons",
        type=int,
        default=None,
        help="limit the number of neurons to model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of workers for DataLoader.",
    )

    # training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="maximum epochs to train the model.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help="micro batch size to train the model. if the model is being "
        "trained on CUDA device and micro batch size 0 is provided, then "
        "automatically increase micro batch size until OOM.",
    )
    parser.add_argument(
        "--crop_frame",
        type=int,
        default=300,
        help="number of frames to take from each trial.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic algorithms in PyTorch",
    )
    parser.add_argument(
        "--autocast",
        default="auto",
        choices=["auto", "disable", "enable"],
        help="Use torch.autocast in torch.bfloat16 when training the model.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable gradient checkpointing for supported models if set to 1.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="pretrained model to restore from before training begins.",
    )

    # optimizer settings
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    # Schedule Free optimizer settings
    parser.add_argument(
        "--schedule_free", action="store_true", help="use schedule-free optimizer"
    )
    parser.add_argument("--adam_warmup_steps", type=int, default=400)
    parser.add_argument("--adam_r", type=float, default=0.0)
    parser.add_argument("--adam_weight_lr_power", type=float, default=2.0)

    parser.add_argument(
        "--criterion",
        type=str,
        default="poisson",
        help="criterion (loss function) to use.",
    )
    parser.add_argument(
        "--ds_scale",
        type=int,
        default=1,
        choices=[0, 1],
        help="scale loss by the size of the dataset",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=-1,
        help="clip gradient norm:\n"
        "0: disable gradient clipping \n"
        "-1: AutoClip (Seetharaman et al. 2020)\n"
        ">0: clip to a specific value.",
    )

    # misc
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="wandb group name, disable wandb logging if not provided.",
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None, help="wandb run ID to resume from."
    )
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2, 3])

    # model settings
    parser.add_argument(
        "--core",
        type=str,
        required=True,
        help="The core module to use.",
    )
    parser.add_argument(
        "--pretrained_core",
        type=Path,
        default=None,
        help="Path to directory where the pre-trained model is stored",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile (part of) the model for faster training",
    )
    parser.add_argument(
        "--readout",
        type=str,
        required=True,
        help="The readout module to use.",
    )
    parser.add_argument(
        "--shifter_mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0: disable shifter\n"
        "1: learn shift from pupil center\n"
        "2: learn shift from pupil center and behavior variables",
    )
    parser.add_argument(
        "--output_mode",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4],
        help="Output activation:\n"
        "0: no activation\n"
        "1: ELU + 1 activation\n"
        "2: Exponential activation\n"
        "3: SoftPlus activation with learnable beta value\n"
        "4: sigmoid activation",
    )

    temp_args = parser.parse_known_args()[0]

    # hyper-parameters for core module
    match temp_args.core.lower():
        case "linear":
            parser.add_argument("--lr", type=float, default=0.005)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            parser.add_argument("--core_weight_decay", type=float, default=0.0)
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=1,
                choices=[0, 1, 2],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input",
            )
        case "gru_baseline":
            parser.add_argument("--lr", type=float, default=0.005)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            parser.add_argument("--core_weight_decay", type=float, default=0.0)
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=1,
                choices=[0, 1, 2],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input",
            )
            # rotation-equivariant 2D CNN settings
            parser.add_argument("--core_cnn_layers", type=int, default=4)
            parser.add_argument("--core_cnn_filters", type=int, default=8)
            parser.add_argument("--core_cnn_input_kernel", type=int, default=9)
            parser.add_argument("--core_cnn_hidden_kernel", type=int, default=7)
            parser.add_argument("--core_momentum", type=float, default=0.9)
            parser.add_argument(
                "--core_linear",
                action="store_true",
                help="remove non-linearity in Stacked2D module",
            )
            parser.add_argument("--core_rotations", type=int, default=8)
            # GRU settings
            parser.add_argument("--core_rnn_input_kernel", type=int, default=9)
            parser.add_argument("--core_rnn_kernel", type=int, default=9)
            parser.add_argument("--core_rnn_size", type=int, default=64)
        case "fcnn":
            parser.add_argument("--lr", type=float, default=0.005)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            parser.add_argument("--core_weight_decay", type=float, default=0.0)
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input\n",
            )
            # factorized 3D CNN settings
            parser.add_argument("--core_spatial_input_kernel", type=int, default=11)
            parser.add_argument("--core_temporal_input_kernel", type=int, default=11)
            parser.add_argument("--core_spatial_hidden_kernel", type=int, default=5)
            parser.add_argument("--core_temporal_hidden_kernel", type=int, default=5)
            parser.add_argument("--core_num_layers", type=int, default=4)
            parser.add_argument("--core_hidden_dim", type=int, default=16)
            parser.add_argument("--core_dropout", type=float, default=0.0)
            parser.add_argument(
                "--core_linear",
                action="store_true",
                help="remove non-linearity in the core module",
            )
        case "v1t":
            parser.add_argument("--lr", type=float, default=0.001647)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.3939)
            parser.add_argument("--core_weight_decay", type=float, default=0.1789)
            # V1T settings
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=3,
                choices=[0, 1, 2, 3, 4],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input\n"
                "3: behavior MLP over behaviors\n"
                "4: behavior MLP over behaviors and pupil center",
            )
            parser.add_argument("--core_patch_size", type=int, default=8)
            parser.add_argument("--core_patch_stride", type=int, default=1)
            parser.add_argument("--core_num_blocks", type=int, default=4)
            parser.add_argument("--core_num_heads", type=int, default=4)
            parser.add_argument("--core_emb_dim", type=int, default=155)
            parser.add_argument("--core_mlp_dim", type=int, default=488)
            parser.add_argument("--core_p_dropout", type=float, default=0.0229)
            parser.add_argument("--core_t_dropout", type=float, default=0.2544)
            parser.add_argument("--core_drop_path", type=float, default=0.0)
            parser.add_argument("--core_disable_bias", action="store_true")
            parser.add_argument(
                "--core_flash_attention", type=int, default=1, choices=[0, 1]
            )
        case "vivit":
            parser.add_argument("--lr", type=float, default=0.0073)
            parser.add_argument("--core_lr", type=float, default=0.0126)
            parser.add_argument("--weight_decay", type=float, default=0.6893)
            parser.add_argument("--core_weight_decay", type=float, default=0.0783)
            # ViViT settings
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input",
            )
            parser.add_argument(
                "--core_patch_mode",
                type=int,
                default=0,
                choices=[0, 1],
                help="3D patch extraction via:\n"
                "0: tensor.unfold followed linear projection\n"
                "1: nn.Conv3d layer",
            )
            parser.add_argument(
                "--core_pad_frame",
                type=int,
                default=1,
                help="pad video frame so that the entire frame when converting it to tokens.",
            )
            parser.add_argument(
                "--core_pos_encoding",
                type=int,
                default=5,
                choices=[0, 1, 2, 3, 4, 5, 6, 7],
                help="Positional encoding:\n"
                "0: no positional encoding\n"
                "1: learnable positional encoding\n"
                "2: separate learnable spatial and temporal positional encoding\n"
                "3: learnable spatial positional encoding and sinusoidal temporal positional encoding\n"
                "4: sinusoidal spatial and temporal positional encoding\n"
                "5: Rotary Positional Embedding (RoPE, Su et al. 2021)\n"
                "6: learnable spatial positional encoding and RoPE temporal positional encoding"
                "7: SinCos spatial positional encoding and RoPE temporal positional encoding",
            )
            parser.add_argument(
                "--core_reg_tokens",
                type=int,
                default=24,
                help="learnable register tokens to the spatial and temporal Transformer",
            )
            parser.add_argument("--core_spatial_patch_size", type=int, default=7)
            parser.add_argument(
                "--core_spatial_patch_stride",
                type=int,
                default=2,
                help="stride size to extract spatial patches",
            )
            parser.add_argument("--core_spatial_depth", type=int, default=14)
            parser.add_argument("--core_temporal_patch_size", type=int, default=13)
            parser.add_argument(
                "--core_temporal_patch_stride",
                type=int,
                default=1,
                help="stride size to extract temporal patches",
            )
            parser.add_argument("--core_temporal_depth", type=int, default=14)
            parser.add_argument("--core_num_heads", type=int, default=2)
            parser.add_argument("--core_emb_dim", type=int, default=112)
            parser.add_argument("--core_head_dim", type=int, default=80)
            parser.add_argument("--core_ff_dim", type=int, default=224)
            parser.add_argument(
                "--core_ff_activation",
                type=str,
                default="relu",
                choices=["none", "tanh", "elu", "relu", "gelu", "silu", "swiglu"],
                help="Transformer block FF activation function",
            )
            parser.add_argument(
                "--core_use_causal_attention",
                action="store_true",
                help="Use causal attention mask in temporal Transformer",
            )
            parser.add_argument(
                "--core_p_dropout",
                type=float,
                default=0.17,
                help="patch embeddings dropout",
            )
            parser.add_argument(
                "--core_mha_dropout",
                type=float,
                default=0.04,
                help="Transformer block MHA dropout",
            )
            parser.add_argument(
                "--core_ff_dropout",
                type=float,
                default=0.21,
                help="Transformer block FF dropout",
            )
            parser.add_argument(
                "--core_drop_path",
                type=float,
                default=0.43,
                help="stochastic depth dropout rate",
            )
            parser.add_argument(
                "--core_activation",
                type=str,
                default="none",
                choices=["none", "tanh", "elu", "relu", "gelu", "silu"],
                help="Core output activation",
            )
            parser.add_argument("--core_parallel_attention", action="store_true")
            parser.add_argument(
                "--core_flash_attention", type=int, default=1, choices=[0, 1]
            )
            parser.add_argument(
                "--core_norm",
                type=str,
                default="layernorm",
                choices=["layernorm", "rmsnorm", "dyt"],
            )
            parser.add_argument("--core_norm_qk", action="store_true")
        case "bvivit":
            parser.add_argument("--lr", type=float, default=0.0036)
            parser.add_argument("--core_lr", type=float, default=0.0048)
            parser.add_argument("--weight_decay", type=float, default=0.3939)
            parser.add_argument("--core_weight_decay", type=float, default=0.1789)
            # ViViT settings
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=3,
                choices=[0, 1, 2, 3, 4],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input\n"
                "3: cross attention with behavior and pupil center\n"
                "4: concat and cross attention with behavior and pupil center",
            )
            parser.add_argument(
                "--core_patch_mode",
                type=int,
                default=0,
                choices=[0, 1, 2],
                help="3D patch extraction via:\n"
                "0: tensor.unfold followed linear projection\n"
                "1: F.conv3d with identity weight followed by linear projection\n"
                "2: nn.Conv3d layer",
            )
            parser.add_argument(
                "--core_pos_encoding",
                type=int,
                default=2,
                choices=[2, 5],
                help="Positional encoding:\n"
                "0: no positional encoding\n"
                "1: learnable positional encoding\n"
                "2: separate learnable spatial and temporal positional encoding\n"
                "3: learnable spatial positional encoding and sinusoidal temporal positional encoding\n"
                "4: sinusoidal spatial and temporal positional encoding\n"
                "5: Rotary Positional Embedding (RoPE, Su et al. 2021)",
            )
            parser.add_argument(
                "--core_reg_tokens",
                type=int,
                default=0,
                help="learnable register tokens to the spatial and temporal Transformer",
            )
            parser.add_argument("--core_spatial_patch_size", type=int, default=7)
            parser.add_argument(
                "--core_spatial_patch_stride",
                type=int,
                default=2,
                help="stride size to extract spatial patches",
            )
            parser.add_argument("--core_spatial_depth", type=int, default=3)
            parser.add_argument("--core_temporal_patch_size", type=int, default=25)
            parser.add_argument(
                "--core_temporal_patch_stride",
                type=int,
                default=1,
                help="stride size to extract temporal patches",
            )
            parser.add_argument("--core_temporal_depth", type=int, default=5)
            parser.add_argument("--core_num_heads", type=int, default=11)
            parser.add_argument("--core_emb_dim", type=int, default=112)
            parser.add_argument("--core_head_dim", type=int, default=48)
            parser.add_argument("--core_ff_dim", type=int, default=136)
            parser.add_argument(
                "--core_ff_activation",
                type=str,
                default="gelu",
                choices=["relu", "gelu", "swiglu"],
                help="Transformer block FF activation function",
            )
            parser.add_argument(
                "--core_p_dropout",
                type=float,
                default=0.1338,
                help="patch embeddings dropout",
            )
            parser.add_argument(
                "--core_mha_dropout",
                type=float,
                default=0.3580,
                help="Transformer block MHA dropout",
            )
            parser.add_argument(
                "--core_ff_dropout",
                type=float,
                default=0.0592,
                help="Transformer block FF dropout",
            )
            parser.add_argument(
                "--core_drop_path",
                type=float,
                default=0.0505,
                help="stochastic depth dropout rate",
            )
            parser.add_argument("--core_parallel_attention", action="store_true")
            parser.add_argument(
                "--core_flash_attention", type=int, default=1, choices=[0, 1]
            )
            parser.add_argument("--core_norm_qk", action="store_true")
            parser.add_argument("--pretrain_core", type=str, default=None)
        case _:
            parser.add_argument("--lr", type=float, default=0.001)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            parser.add_argument("--core_weight_decay", type=float, default=0.0)

    # hyper-parameters for readout modules
    match temp_args.readout:
        case "gaussian":
            parser.add_argument(
                "--readout_grid_mode",
                type=int,
                default=1,
                choices=[0, 1, 2],
                help="Readout grid predictor mode:\n"
                "0: disable grid predictor\n"
                "1: grid predictor using (x, y) neuron coordinates\n"
                "2: grid predictor using (x, y, z) neuron coordinates",
            )
            parser.add_argument(
                "--readout_bias_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="Gaussian2d readout bias mode:\n"
                "0: disable bias term\n"
                "1: initialize bias with zeros\n"
                "2: initialize bias with the mean responses",
            )
        case "depth-gaussian":
            parser.add_argument(
                "--readout_grid_mode",
                type=int,
                default=1,
                choices=[0, 1, 2],
                help="Readout grid predictor mode:\n"
                "0: disable grid predictor\n"
                "1: grid predictor using (x, y) neuron coordinates\n"
                "2: grid predictor using (x, y, z) neuron coordinates",
            )
            parser.add_argument(
                "--readout_bias_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="Gaussian2d readout bias mode:\n"
                "0: disable bias term\n"
                "1: initialize bias with zeros\n"
                "2: initialize bias with the mean responses",
            )
        case "factorized":
            parser.add_argument(
                "--readout_bias_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="Gaussian2d readout bias mode:\n"
                "0: disable bias term\n"
                "1: initialize bias with zeros\n"
                "2: initialize bias with the mean responses",
            )
            parser.add_argument("--readout_dropout", type=float, default=0.0)
        case "conv":
            parser.add_argument("--readout_groups", type=int, default=1)
            parser.add_argument("--readout_dropout", type=float, default=0.0)
        case "attention":
            parser.add_argument(
                "--readout_bias_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="Gaussian2d readout bias mode:\n"
                "0: disable bias term\n"
                "1: initialize bias with zeros\n"
                "2: initialize bias with the mean responses",
            )
            parser.add_argument("--readout_emb_dim", type=int, default=128)

    # hyper-parameters for shifter module
    if temp_args.shifter_mode > 0:
        parser.add_argument("--shifter_layers", type=int, default=3)
        parser.add_argument("--shifter_size", type=int, default=5)

    del temp_args

    main(parser.parse_args())
