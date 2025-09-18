import argparse
import os
import pickle
from datetime import datetime
from shutil import rmtree
from typing import Dict
from typing import List

import numpy as np
import torch
import wandb
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t.data import data
from viv1t.metrics import challenge_correlation
from viv1t.model import Model
from viv1t.utils import logger
from viv1t.utils import utils
from viv1t.utils import yaml

# due to some weird reasons with the DeepLake dataset, the organizer set
# the number of skip frames to 51 even though it is supposed to be 50.
SKIP = 51


def load(data_dir: str, output_dir: str, device: torch.device):
    """Load submission DataLoaders and model from output_dir checkpoint"""
    assert os.path.isdir(output_dir)
    args = argparse.Namespace()
    args.output_dir = output_dir
    utils.load_args(args)
    args.device = device
    args.batch_size = 1
    args.micro_batch_size = 1
    args.data_dir = data_dir
    args.num_workers = 0

    val_ds, test_ds = data.get_submission_ds(
        args,
        data_dir=args.data_dir,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in val_ds.items()
        },
    )
    # load model checkpoint
    filename = os.path.join(output_dir, "ckpt", "model_state.pt")
    assert os.path.exists(filename), f"Cannot find {filename}."
    ckpt = torch.load(filename, map_location="cpu")
    state_dict = model.state_dict()
    state_dict.update(ckpt["model"])
    try:
        model.load_state_dict(state_dict)
        print(
            f"Loaded checkpoint from {output_dir} "
            f"(correlation: {ckpt['value']:.04f})."
        )
    except RuntimeError as e:
        print(f"\nError loading model state from {filename}.\n{e}")
        exit()
    del ckpt
    model = model.to(device)
    model.train(False)
    return val_ds, test_ds, model


def load_all(data_dir: str, saved_models: Dict[str, str], device: torch.device):
    all_val_ds, models = {}, {}
    tiers = ["live_main", "live_bonus", "final_main", "final_bonus"]
    all_test_ds = {tier: {} for tier in tiers}
    for model_name, output_dir in saved_models.items():
        val_ds, test_ds, models[model_name] = load(
            data_dir=data_dir, output_dir=output_dir, device=device
        )
        all_val_ds[model_name] = val_ds
        for tier in tiers:
            all_test_ds[tier][model_name] = test_ds[tier]
    return all_val_ds, all_test_ds, models


@torch.inference_mode()
def predict(
    ds: Dict[str, Dict[str, DataLoader]],
    models: Dict[str, Model],
    mouse_id: str,
    index: int,
    device: torch.device,
):
    """Predict trial index over all models and return the average output"""
    prediction, response = [], None
    to_batch = lambda x: torch.unsqueeze(x, dim=0).to(device)
    for model_name, model in models.items():
        sample = ds[model_name][mouse_id].dataset.__getitem__(index, to_tensor=True)
        t = sample["video"].shape[1] - SKIP
        y_pred, _ = model(
            to_batch(sample["video"]),
            mouse_id=mouse_id,
            behaviors=to_batch(sample["behavior"]),
            pupil_centers=to_batch(sample["pupil_center"]),
        )
        prediction.append(rearrange(y_pred[0, :, -t:], "n t -> n t 1"))
        if response is None and "response" in sample:
            response = sample["response"][:, -t:]
    prediction = torch.mean(torch.cat(prediction, dim=-1), dim=-1)
    return prediction.cpu(), response.cpu()


def evaluate(
    ds: Dict[str, Dict[str, DataLoader]],
    models: Dict[str, Model],
    mouse_ids: List[str],
    device: torch.device,
):
    results = {}
    for mouse_id in tqdm(mouse_ids, desc="Evaluate"):
        y_true, y_pred = [], []
        ds_size = len(ds[list(ds.keys())[0]][mouse_id].dataset)
        for index in range(ds_size):
            prediction, response = predict(
                ds=ds,
                models=models,
                mouse_id=mouse_id,
                index=index,
                device=device,
            )
            y_pred.append(prediction)
            y_true.append(response)
        results[mouse_id] = challenge_correlation(y_true=y_true, y_pred=y_pred).item()
    correlations = list(results.values())
    if correlations:
        results["average"] = np.mean(correlations)
    return results


def generate_submission(
    ds: Dict[str, Dict[str, DataLoader]],
    models: Dict[str, Model],
    mouse_ids: List[str],
    submission_dir: str,
    tier: str,
    track: str,
    device: torch.device,
):
    cache_dir = os.path.join(submission_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    filenames = []
    for mouse_id in tqdm(mouse_ids, desc=f"{tier} {track}"):
        mouse_ds = ds[list(ds.keys())[0]][mouse_id].dataset
        trial_ids, y_pred = mouse_ds.trial_ids.tolist(), []
        for index in tqdm(range(len(mouse_ds)), desc=f"Mouse {mouse_id}", leave=False):
            prediction, _ = predict(
                ds=ds,
                models=models,
                mouse_id=mouse_id,
                index=index,
                device=device,
            )
            y_pred.append(prediction.numpy())
        filename = os.path.join(cache_dir, f"{tier}_{track}_{mouse_id}.pkl")
        with open(filename, "wb") as file:
            pickle.dump(
                {
                    "mouse": [data.MOUSE_IDS[mouse_id]] * len(y_pred),
                    "trial_indices": trial_ids,
                    "prediction": [pred.tolist() for pred in y_pred],
                    "neuron_ids": [mouse_ds.neuron_ids.tolist()] * len(y_pred),
                },
                file,
            )
        filenames.append(filename)
    return filenames


def main(args):
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.Logger(os.path.join(args.output_dir, "output.log"))
    args.device = utils.get_device(args.device)
    utils.set_random_seed(seed=args.seed)

    if args.wandb is not None:
        utils.wandb_init(args, wandb_sweep=False)

    # pretrained model to load
    # args.saved_models = {
    #     "6": "runs/factorized_baseline_seed_v2/006",
    #     "22": "runs/factorized_baseline_seed_v2/22",
    #     "15": "runs/factorized_baseline_seed_v2/15",
    #     "10": "runs/factorized_baseline_seed_v2/10",
    #     "1": "runs/factorized_baseline_seed_v2/001",
    #     "24": "runs/factorized_baseline_seed_v2/24",
    #     "12": "runs/factorized_baseline_seed_v2/12",
    #     "25": "runs/factorized_baseline_seed_v2/25",
    #     "5": "runs/factorized_baseline_seed_v2/005",
    #     "13": "runs/factorized_baseline_seed_v2/13",
    # }
    args.saved_models = {
        "009": "/home/storage/runs/vivit_ensemble/009",
        "012": "/home/storage/runs/vivit_ensemble/012",
        "002": "/home/storage/runs/vivit_ensemble/002",
        "028": "/home/storage/runs/vivit_ensemble/028",
        "015": "/home/storage/runs/vivit_ensemble/015",
    }
    # args.saved_models = {
    #     "factorized-6": "/home/storage/runs/factorized_baseline_seed_v2/006",
    #     "factorized-22": "/home/storage/runs/factorized_baseline_seed_v2/22",
    #     "factorized-15": "/home/storage/runs/factorized_baseline_seed_v2/15",
    #     "factorized-10": "/home/storage/runs/factorized_baseline_seed_v2/10",
    #     "factorized-1": "/home/storage/runs/factorized_baseline_seed_v2/001",
    #     "vivit-009": "/home/storage/runs/vivit_ensemble/009",
    #     "vivit-012": "/home/storage/runs/vivit_ensemble/012",
    #     "vivit-002": "/home/storage/runs/vivit_ensemble/002",
    #     "vivit-028": "/home/storage/runs/vivit_ensemble/028",
    #     "vivit-015": "/home/storage/runs/vivit_ensemble/015",
    # }
    # args.saved_models = {
    #     "factorized-6": "/home/storage/runs/factorized_baseline_seed_v2/006",
    #     "vivit-009": "/home/storage/runs/vivit_ensemble/009",
    # }
    assert hasattr(args, "saved_models") and args.saved_models

    # create CSV dir to save results with timestamp Year-Month-Day-Hour-Minute
    timestamp = f"{datetime.now():%Y-%m-%d-%Hh%Mm}"
    submission_dir = os.path.join(args.output_dir, "submissions", timestamp)

    val_ds, test_ds, models = load_all(
        data_dir=args.data_dir, saved_models=args.saved_models, device=args.device
    )
    utils.save_args(args)

    print("")
    val_result = evaluate(
        ds=val_ds, models=models, mouse_ids=list(val_ds.keys()), device=args.device
    )
    if args.verbose:
        print(f"Validation correlation: {val_result['average']:.04f}")
    if args.wandb is not None:
        wandb.log({"best_corr": val_result["average"]}, step=0)
    yaml.save(os.path.join(submission_dir, "evaluation.yaml"), val_result)

    mouse_ids = list(data.SENSORIUM_NEW)
    filenames = {"new": {"main": {}, "bonus": {}}}

    # generate main track submission file
    print("\nPredict main track...")
    filenames["new"]["main"]["live"] = generate_submission(
        ds=test_ds["live_main"],
        models=models,
        mouse_ids=mouse_ids,
        submission_dir=os.path.join(submission_dir, "new"),
        tier="main",
        track="live",
        device=args.device,
    )
    filenames["new"]["main"]["final"] = generate_submission(
        ds=test_ds["final_main"],
        models=models,
        mouse_ids=mouse_ids,
        submission_dir=os.path.join(submission_dir, "new"),
        tier="main",
        track="final",
        device=args.device,
    )

    # generate bonus track submission file
    print("\nPredict bonus track...")
    filenames["new"]["bonus"]["live"] = generate_submission(
        ds=test_ds["live_bonus"],
        models=models,
        mouse_ids=mouse_ids,
        submission_dir=os.path.join(submission_dir, "new"),
        tier="bonus",
        track="live",
        device=args.device,
    )
    filenames["new"]["bonus"]["final"] = generate_submission(
        ds=test_ds["final_bonus"],
        models=models,
        mouse_ids=mouse_ids,
        submission_dir=os.path.join(submission_dir, "new"),
        tier="bonus",
        track="final",
        device=args.device,
    )

    with open(os.path.join(submission_dir, "filenames.pkl"), "wb") as file:
        pickle.dump(filenames, file)

    print(f"Saved submission files to {submission_dir}.")

    # generate submission file for old mice

    # # inference data in a separate process to avoid GPU memory leak
    # queue = Queue()
    # job = Process(target=submission_wrapper, args=(args, submission_dir, queue))
    # job.start()
    # filenames = queue.get()
    # job.join()
    #
    # for version in filenames.keys():
    #     print(f"\nGenerating submission files for {version} mice...")
    #     for track in filenames[version].keys():
    #         submission.create_submission_file(
    #             filenames=filenames[version][track],
    #             track=track,
    #             submission_dir=os.path.join(submission_dir, version),
    #         )
    #     rmtree(os.path.join(submission_dir, version, "cache"))
    #
    # print(f"Saved submission files to {submission_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sensorium",
        help="path to directory where the compressed dataset is stored.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="",
        help="Device to use for computation. "
        "Use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234)

    # wandb settings
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="wandb group name, disable wandb logging if not provided.",
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None, help="wandb run ID to resume from."
    )
    # misc
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    main(parser.parse_args())
