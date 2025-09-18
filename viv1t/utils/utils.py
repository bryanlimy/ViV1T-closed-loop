import copy
import logging
import os
import random
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
import torch
import wandb
from scipy.stats import ttest_ind
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.lib.runid import generate_id

from viv1t import metrics
from viv1t.utils import yaml

os.environ["WANDB_SILENT"] = "true"


def get_timestamp() -> str:
    """Return timestamp in the format of YYYYMMDD-HHhMMm in London timezone"""
    return f"{datetime.now(tz=pytz.timezone('Europe/London')):%Y%m%d-%Hh%Mm}"


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for Python, Numpy and PyTorch.
    Args:
        seed: int, the random seed to use.
        deterministic: bool, use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def get_device(device: str | torch.device | None = None) -> torch.device:
    """return the appropriate torch.device if device is not set"""
    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            # torch.set_num_threads(4)
            # torch.set_num_interop_threads(8)
        elif torch.backends.mps.is_available():
            device = "mps"
    return torch.device(device)


def support_bf16(device: str | torch.device):
    """Check if device supports bfloat16"""
    if isinstance(device, torch.device):
        device = device.type
    match device:
        case "cpu" | "mps":
            return False
        case "cuda":
            return torch.cuda.get_device_capability(device)[0] >= 8
        case _:
            raise KeyError(f"Unknown device type {device}.")


def update_dict(target: dict[str, Any], source: dict[str, Any], replace: bool = False):
    """Update target dictionary with values from source dictionary"""
    for k, v in source.items():
        if replace:
            target[k] = v
        else:
            if k not in target:
                target[k] = []
            target[k].append(v)


def check_output(command: list):
    """Run command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def save_args(args, output_dir: Path = None):
    """Save args object as dictionary to args.output_dir/args.json"""
    if output_dir is None:
        output_dir = args.output_dir
    try:
        # get git hash and hostname
        setattr(
            args,
            "git_hash",
            check_output(["git", "-C", Path(__file__).parent, "describe", "--always"]),
        )
        setattr(args, "hostname", check_output(["hostname"]))
    except subprocess.CalledProcessError as e:
        if args.verbose > 1:
            print(f"Unable to call subprocess: {e}")
    arguments = copy.deepcopy(args.__dict__)
    yaml.save(filename=output_dir / "args.yaml", data=arguments)


def load_args(args: Any):
    """Load args object from args.output_dir/args.yaml"""
    content = yaml.load(args.output_dir / "args.yaml")
    for key, value in content.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def wandb_init(args):
    """Initialize wandb"""
    if args.wandb is None or wandb.run is not None:
        return
    try:
        config = deepcopy(args.__dict__)
        config.pop("input_shapes", None)
        config.pop("output_shapes", None)
        config.pop("neuron_idx", None)
        config.pop("verbose", None)
        config.pop("wandb", None)
        config.pop("trainable_params", None)
        config.pop("clear_output_dir", None)
        if args.wandb_id is None:
            args.wandb_id = wandb_id = generate_id()
        else:
            wandb_id = args.wandb_id
        wandb.init(
            config=config,
            project="sensorium2023",
            entity="bryanlimy",
            group=args.wandb,
            name=args.output_dir.name,
            resume="allow",
            id=wandb_id,
        )
        del config
    except AssertionError as e:
        print(f"wandb.init error: {e}\n")


def wandb_update_config(args, wandb_sweep: bool):
    """Update wandb config of the run"""
    assert wandb.run is not None, "wandb is not initialized."
    if wandb_sweep:
        wandb.config.update(
            {
                name: getattr(args, name)
                for name in [
                    "git_hash",
                    "hostname",
                    "epochs",
                    "micro_batch_size",
                    "mouse_ids",
                    "output_dir",
                    "autocast",
                    "seed",
                ]
            }
        )
    report = {"best_corr": 0, "epoch": 0}
    if hasattr(args, "trainable_params"):
        report["trainable_params"] = args.trainable_params
    wandb.log(report, step=0)


def log_metrics(results: dict[str, dict[str, Any]]):
    """Compute the mean of the metrics in results and log to Summary

    Args:
        results: Dict[str, Dict[str, List[float]]],
            a dictionary of tensors where keys are the name of the metrics
            that represent results from of a mouse.
    """
    mouse_ids = list(results.keys())
    keys = list(results[mouse_ids[0]].keys())
    for mouse_id in mouse_ids:
        for metric in keys:
            value = results[mouse_id][metric]
            if isinstance(value, list):
                if torch.is_tensor(value[0]):
                    value = torch.stack(value).cpu()
                results[mouse_id][metric] = torch.mean(value)
            elif torch.is_tensor(value):
                results[mouse_id][metric] = value.cpu()
    overall_result = {}
    for metric in keys:
        overall_result[metric[metric.find("/") + 1 :]] = torch.mean(
            torch.stack([results[mouse_id][metric] for mouse_id in mouse_ids])
        )
    return overall_result


@torch.no_grad()
def inference(
    ds: DataLoader,
    model: nn.Module,
    device: torch.device = "cpu",
    progress_bar: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """
    Inference data in test DataLoaders

    Please note that some of the trials in the Sensorium 2023 test sets have
    varying number of frames, and therefore a batch size of 1 is needed for
    those DataLoader.

    Returns:
        responses: Dict[str, torch.Tensor]
            - y_pred: List[torch.Tensor], list predicted responses in (N, T)
            - y_true: List[torch.Tensor], list of recorded responses in (N, T)
            - video_ids: List[torch.Tensor], list of video IDs
    """
    responses = {"y_true": [], "y_pred": [], "video_ids": []}
    mouse_id = ds.dataset.mouse_id
    model = model.to(device)
    model.train(False)
    for sample in tqdm(
        ds, desc=f"mouse {mouse_id}", leave=False, disable=not progress_bar
    ):
        prediction, _ = model(
            inputs=sample["video"].to(device, model.dtype),
            mouse_id=mouse_id,
            behaviors=sample["behavior"].to(device, model.dtype),
            pupil_centers=sample["pupil_center"].to(device, model.dtype),
        )
        prediction = prediction.to("cpu", torch.float32)
        responses["y_pred"].append(prediction)
        responses["y_true"].append(sample["response"])
        responses["video_ids"].append(sample["video_id"])
    # flatten the list of 3D tensors to a list of 2D tensors of shape (N, T)
    responses = {
        k: [v[i][j] for i in range(len(v)) for j in range(len(v[i]))]
        for k, v in responses.items()
    }
    return responses


def evaluate(
    args,
    ds: dict[str, DataLoader],
    model: nn.Module,
    skip: int = 50,
    progress_bar: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Evaluate DataLoaders

    Args:
        args
        ds: Dict[str, DataLoader], dictionary of DataLoader, one for each mouse.
        model: nn.Module, the model.
        skip: int, number of frames to skip at the beginning of the trial when
            computing single trial correlation.
        progress_bar: bool, show progress bar.
    Returns:
        results: correlation, single trial correlation and correlation to
        average between recorded and predicted responses
    """
    results = {
        "correlation": {},
        "single_trial_correlation": {},
        "normalized_correlation": {},
    }
    tier = list(ds.values())[0].dataset.tier

    for mouse_id, mouse_ds in tqdm(
        ds.items(), desc=f"Evaluate {tier}", disable=args.verbose < 2
    ):
        if mouse_ds.dataset.hidden_response:
            continue  # skip dataset with no response labels
        responses = inference(
            ds=mouse_ds, model=model, device=args.device, progress_bar=progress_bar
        )

        # crop response and prediction to the same length after skipping frames
        for i in range(len(responses["y_true"])):
            t = responses["y_true"][i].shape[1] - skip
            responses["y_true"][i] = responses["y_true"][i][:, -t:]
            responses["y_pred"][i] = responses["y_pred"][i][:, -t:]

        results["correlation"][mouse_id] = metrics.challenge_correlation(
            y_true=responses["y_true"], y_pred=responses["y_pred"]
        ).item()
        results["single_trial_correlation"][mouse_id] = (
            metrics.single_trial_correlation(
                y_true=responses["y_true"], y_pred=responses["y_pred"]
            ).item()
        )
        results["normalized_correlation"][mouse_id] = metrics.normalized_correlation(
            y_true=responses["y_true"],
            y_pred=responses["y_pred"],
            video_ids=torch.stack(responses["video_ids"]),
        ).item()

        del responses
        torch.cuda.empty_cache()

    for k, metric in results.items():
        results[k]["average"] = np.mean(list(metric.values()))
    return results


def restore(
    args: Any,
    model: nn.Module,
    filename: Path,
    val_ds: dict[str, DataLoader] = None,
):
    """restore model weights from checkpoint"""
    if not filename.is_file():
        print(f"File {filename} not found.")
        return

    device = model.device
    model = model.to("cpu")
    ckpt = torch.load(filename, map_location="cpu")
    state_dict = model.state_dict()
    num_params = 0
    for k in ckpt["model"].keys():
        if k in state_dict:
            state_dict[k] = ckpt["model"][k]
            num_params += 1
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(
        f"\nLoaded {num_params} parameters from {filename} "
        f"(epoch {ckpt['epoch']}, correlation: {ckpt['value']:.04f}).\n"
    )
    del ckpt

    if val_ds is not None:
        results = evaluate(args, ds=val_ds, model=model)
        print(f"Validation correlation: {results['correlation']['average']:.04f}")


def save_tuning(result: dict[str, Any], save_dir: Path, mouse_id: str):
    save_dir = save_dir / "tuning" / f"mouse{mouse_id}"
    save_dir.mkdir(parents=True, exist_ok=True)
    for k, v in result.items():
        np.save(save_dir / f"{k}.npy", v, allow_pickle=False)


def load_tuning(save_dir: Path, mouse_id: str):
    save_dir = save_dir / "tuning" / f"mouse{mouse_id}"
    filenames = save_dir.glob("*.npy")
    result = {
        k.name.replace(".npy", ""): np.load(k, allow_pickle=False) for k in filenames
    }
    return result


def get_selective_neurons(
    save_dir: Path,
    mouse_id: str,
    tuning_type: str,
    threshold: float,
) -> np.ndarray:
    tuning = load_tuning(save_dir=save_dir, mouse_id=mouse_id)
    match tuning_type:
        case "orientation":
            selective = tuning["orientation_selective"]
            SIs = tuning["OSI"]
        case "direction":
            selective = tuning["direction_selective"]
            SIs = tuning["DSI"]
        case _:
            raise ValueError(f"Unknown tuning type: {tuning_type}")
    neurons = np.where((SIs >= threshold) & selective)[0]
    return neurons


def get_p_value(array1: np.ndarray, array2: np.ndarray) -> str:
    """
    Perform t-test and compute p-value between array1 and array2, and return
    the significance level as string.
    """
    p_value = ttest_ind(array1, array2).pvalue
    text = "n.s."
    if p_value <= 0.001:
        text = "***"
    elif p_value <= 0.01:
        text = "**"
    elif p_value <= 0.05:
        text = "*"
    return text


def set_logger_handles(
    logger: logging.Logger,
    filename: Path | None = None,
    level: int = logging.INFO,
):
    """Setup logger to write to filename and console"""
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    logger.addHandler(console_handler)


def get_reliable_neurons(output_dir: Path, mouse_id: str, size: int = 30) -> np.ndarray:
    """Return the top `size` reliable neurons"""
    try:
        df = pd.read_parquet(output_dir / "neuron_reliability.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find neuron_reliability.parquet in {output_dir}. "
            f"Please run most_exciting_stimulus/estimate_neuron_reliability.py"
        )
    df = df[df.mouse == mouse_id].sort_values(
        by="rank", ascending=True, na_position="last"
    )
    neurons = df.head(size).neuron.values
    return neurons


def get_size_tuning_preference(output_dir: Path, mouse_id: str, neuron: int) -> int:
    try:
        df = pd.read_parquet(output_dir / "size_tuning_preference.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find size_tuning_preference.parquet in {output_dir}. "
            f"Please run tuning_feedback/estimate_feedbackRF.py"
        )
    neuron_preference = df[(df.mouse == mouse_id) & (df.neuron == neuron)]
    assert len(neuron_preference) == 1
    return neuron_preference.iloc[0].classic_preference


def get_neuron_weights(
    output_dir: Path, mouse_id: str, to_tensor: bool = True
) -> np.ndarray | torch.Tensor:
    """Get the model prediction performance of each neuron"""
    df = pd.read_parquet(output_dir / "neuron_reliability.parquet")
    df = df[df.mouse == mouse_id]
    df = df.sort_values(by="neuron", ascending=True)
    correlations = df.correlation.values
    # ensure there is no negative correlation values
    correlations = np.clip(correlations, a_min=0, a_max=None)
    neuron_weights = correlations / np.sum(correlations)
    neuron_weights = neuron_weights.astype(np.float32)
    if to_tensor:
        neuron_weights = torch.from_numpy(neuron_weights)
    return neuron_weights
