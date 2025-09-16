from pathlib import Path

import h5py
import numpy as np
import torch


def write(filename: Path, data: dict[int | str, np.ndarray | torch.Tensor]):
    """Write or append content to H5 file"""
    assert type(data) == dict
    filename.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filename, mode="a") as file:
        for key, value in data.items():
            key = str(key)
            if torch.is_tensor(value):
                value = value.cpu().numpy()
            if key in file.keys():
                del file[key]
            file.create_dataset(
                key,
                shape=value.shape,
                dtype=value.dtype,
                data=value,
                compression="gzip",
                compression_opts=9,
            )


def get(filename: Path, trial_ids: list[int] | np.ndarray) -> list[np.ndarray]:
    """Return responses in the same order as trial_ids"""
    data = []
    with h5py.File(filename, mode="r") as file:
        for trial_id in trial_ids:
            trial_id = str(trial_id)
            if trial_id not in file.keys():
                raise KeyError(f"Cannot find trial {trial_id} in {filename}.")
            data.append(file[trial_id][:])
    return data


def get_trial_ids(filename: Path) -> list[int]:
    """Get all trial_ids in the H5 file"""
    with h5py.File(filename, mode="r") as file:
        trial_ids = [int(k) for k in file.keys()]
    return trial_ids


def contain_trial_id(filename: Path, trial_id: int) -> bool:
    """Check if the H5 file contains a specific trial_id"""
    trial_ids = set(get_trial_ids(filename))
    return trial_id in trial_ids


def get_shape(filename: Path, trial_id: int) -> tuple:
    """Get the shape of the response for a specific trial_id in the H5 file"""
    with h5py.File(filename, mode="r") as file:
        shape = file[str(trial_id)].shape
    return shape
