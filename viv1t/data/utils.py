from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from einops import repeat
from torch.utils.data import DataLoader

from viv1t.data.constants import *
from viv1t.data.statistics import load_stats


def unzip(filename: Path, unzip_dir: Path):
    """Extract zip file with filename to unzip_dir"""
    if not filename.is_file():
        raise FileNotFoundError(f"file {filename} not found.")
    print(f"Unzipping {filename}...")
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def micro_batching(batch: dict[str, torch.Tensor], batch_size: int):
    """Divide batch into micro batches"""
    indexes = np.arange(0, len(batch["video"]), step=batch_size, dtype=int)
    for i in indexes:
        yield {k: v[i : i + batch_size] for k, v in batch.items()}


def find_nan(array: np.ndarray) -> int:
    """Given a 1D array, find the first NaN value and return its index"""
    assert len(array.shape) == 1
    nans = np.where(np.isnan(array))[0]
    return nans[0] if nans.any() else len(array)


def set_shapes(args, ds: dict[str, DataLoader]):
    """Set args.input_shapes and args.output_shapes"""
    mouse_ids = list(ds.keys())
    max_frame = ds[mouse_ids[0]].dataset.crop_frame
    args.input_shapes = {
        "video": (
            ds[mouse_ids[0]].dataset.num_channels,
            max_frame,
            *ds[mouse_ids[0]].dataset.video_shape,
        ),
        "behavior": (2, max_frame),
        "pupil_center": (2, max_frame),
    }
    args.output_shapes = {
        mouse_id: (ds[mouse_id].dataset.num_neurons, max_frame)
        for mouse_id in mouse_ids
    }


def get_dataloader_kwargs(
    args,
    device: torch.device,
    num_workers: int = None,
):
    # settings for DataLoader
    num_workers = args.num_workers if num_workers is None else num_workers
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if device.type == "cuda":
        kwargs |= {
            "prefetch_factor": 2 * num_workers if num_workers else None,
            "persistent_workers": num_workers > 0,
        }
    return kwargs


def estimate_mean_response(ds: dict[str, DataLoader], size: int = 16):
    """estimate the mean responses from each neuron from size samples"""
    mean_responses = {}
    for mouse_id, mouse_ds in ds.items():
        sample = mouse_ds.dataset.__getitem__
        _size = min(size, len(mouse_ds.dataset))
        responses = torch.stack([sample(i)["response"] for i in range(_size)], dim=0)
        mean_responses[mouse_id] = torch.mean(responses, dim=(0, 2))
    return mean_responses


def num_steps(ds: dict[str, DataLoader]) -> int:
    """Return the number of total steps to iterate all the DataLoaders"""
    return sum([len(ds[k]) for k in ds.keys()])


def load_trial(
    mouse_dir: Path,
    trial_id: str | int | torch.Tensor | np.integer,
    to_tensor: bool = False,
):
    """
    Load data from a single trial in mouse_dir

    Data from each mouse has a fixed tensor shape, in order to find the duration
    of the trial, we first find the first NaN value in the recording and crop
    each tensor.
    Response is None for live and final test sets

    Args:
        mouse_dir: directory with mouse data
        trial_id: the trial ID to load
        to_tensor: convert np.ndarray to torch.Tensor
    Returns
        data: Dict[str, TENSOR]
            video: TENSOR, video in format (C, T, H, W)
            response: TENSOR, response in format (N, T) where N is num. of neurons
            behavior: TENSOR, pupil dilation and speed in format (2, T)
            pupil center: TENSOR, pupil center x and y coordinates in format (2, T)
            duration: int, the duration (i.e. num. of frames) of the trial
    """
    basename, data_dir = f"{trial_id}.npy", mouse_dir / "data"
    load = lambda key: np.load(data_dir / key / basename).astype(np.float32)
    video = load("videos")
    # find duration of the trial by searching for the first NaN in the video
    num_frames = find_nan(video[0, 0, :])
    sample = {
        "video": rearrange(video[..., :num_frames], "h w t -> 1 t h w"),
        "response": load("responses")[:, :num_frames],
        "behavior": load("behavior")[:, :num_frames],
        "pupil_center": load("pupil_center")[:, :num_frames],
        "duration": num_frames,
    }
    if to_tensor:
        sample["video"] = torch.from_numpy(sample["video"])
        sample["response"] = torch.from_numpy(sample["response"])
        sample["behavior"] = torch.from_numpy(sample["behavior"])
        sample["pupil_center"] = torch.from_numpy(sample["pupil_center"])
    return sample


def get_neuron_coordinates(
    mouse_id: str, to_tensor: bool = False
) -> np.ndarray | torch.Tensor:
    neuron_coordinates = np.load(
        METADATA_DIR / "neuron_coordinates" / f"mouse{mouse_id}.npy",
        allow_pickle=False,
    )
    neuron_coordinates = neuron_coordinates.astype(np.float32)
    return torch.from_numpy(neuron_coordinates) if to_tensor else neuron_coordinates


def get_video_ids(mouse_id: str) -> np.ndarray:
    """Return video IDs for mouse data"""
    return np.load(METADATA_DIR / "video_ids" / f"mouse{mouse_id}.npy")


def get_stimulus_ids(mouse_id: str) -> np.ndarray:
    """Return stimulus IDs for mouse data"""
    return np.load(METADATA_DIR / "stimulus_ids" / f"mouse{mouse_id}.npy")


def get_tier_ids(data_dir: Path, mouse_id: str) -> np.ndarray:
    filename = data_dir / MOUSE_IDS[mouse_id] / "meta" / "trials" / "tiers.npy"
    tiers = np.load(filename, allow_pickle=True)
    for tier in TIERS.keys():  # rename tier names
        if tier in tiers:
            tiers[tiers == tier] = TIERS[tier]
    return tiers


def get_gabor_parameters(
    mouse_id: str, trial_id: int | str | torch.Tensor
) -> np.ndarray:
    """Return the drifting gabor parameters for a given trial_id or video_id

    Returns:
        gabor_parameters: drifting gabor parameters in shape (num. frames, 3)
            and format (orientation, wavelength, frequency)
    """
    if torch.is_tensor(trial_id):
        trial_id = trial_id.item()
    filename = (
        METADATA_DIR
        / "ood_features"
        / "drifting_gabor"
        / f"mouse{mouse_id}"
        / f"{trial_id}.npy"
    )
    return np.load(filename).astype(np.float32)


def get_flashing_image_parameters(
    mouse_id: str, trial_id: int | str | torch.Tensor
) -> np.ndarray:
    """Return the flashing image parameters for a given trial_id or video_id

    Returns:
        image_ids: image IDs of the flashing images (num. frames)
    """
    if torch.is_tensor(trial_id):
        trial_id = trial_id.item()
    filename = (
        METADATA_DIR
        / "ood_features"
        / "flashing_images"
        / f"mouse{mouse_id}"
        / f"{trial_id}.npy"
    )
    return np.load(filename).astype(int)


def get_gaussian_dot_parameters(
    mouse_id: str, trial_id: int | str | torch.Tensor
) -> np.ndarray:
    """Return the Gaussian dot parameters for a given trial_id or video_id

    Returns:
        dot_parameters: drifting gabor parameters in shape (num. frames, 4)
            and format (x, y, radius, dot_is_black)
    """
    if torch.is_tensor(trial_id):
        trial_id = trial_id.item()
    filename = (
        METADATA_DIR
        / "ood_features"
        / "gaussian_dots"
        / f"mouse{mouse_id}"
        / f"{trial_id}.npy"
    )
    return np.load(filename).astype(np.float32)


def get_mean_behaviors(
    mouse_id: str, num_frames: int = MAX_FRAME
) -> (torch.Tensor, torch.Tensor):
    """
    Create a behavior and pupil center as model input parameter using the
    mean values obtained from the training set of the Sensorium 2023 dataset
    """
    mouse_stats = load_stats(mouse_id)
    behavior = repeat(mouse_stats["behavior"]["mean"], "d -> d t", t=num_frames)
    behavior = torch.from_numpy(behavior)

    pupil_center = repeat(mouse_stats["pupil_center"]["mean"], "d -> d t", t=MAX_FRAME)
    pupil_center = torch.from_numpy(pupil_center)

    return behavior, pupil_center


def get_num_neurons(mouse_id: str) -> int:
    """Get the number of neurons the mouse has"""
    stat = load_stats(mouse_id)
    return len(stat["response"]["mean"])
