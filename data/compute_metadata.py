import argparse
import os
import warnings
from pathlib import Path

import numpy as np
from einops import rearrange
from sensorium.data import METADATA_DIR
from sensorium.data import MOUSE_IDS
from sensorium.data import STATISTICS_DIR
from sensorium.data import get_tier_ids
from sensorium.data import load_trial
from tqdm import tqdm


def measure(array: np.ndarray, axis: int | tuple):
    """
    Measure min, max, median, mean, std of array along dim
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = {
            "min": np.nanmin(array, axis=axis),
            "max": np.nanmax(array, axis=axis),
            "median": np.nanmedian(array, axis=axis),
            "mean": np.nanmean(array, axis=axis),
            "std": np.nanstd(array, axis=axis),
        }
    return stats


def compute_statistics(data_dir: Path, mouse_id: str):
    """
    Compute statistics (min, max, median, mean, std) of the training set with
    the following format:
    - video: (1)
    - responses:  (N)
    - behavior: (2)
    - pupil_center: (2)
    """
    print(f"Compute statistics for Mouse {mouse_id}")

    # read from training set
    tiers = get_tier_ids(data_dir=data_dir, mouse_id=mouse_id)
    trial_ids = np.where(tiers == "train")[0]

    # load a sample to get shapes
    load = lambda k: np.load(
        data_dir / MOUSE_IDS[mouse_id] / "data" / k / f"{trial_ids[0]}.npy"
    )
    h, w, t = load("videos").shape
    n = load("responses").shape[0]
    d1, d2 = load("behavior").shape[0], load("pupil_center").shape[0]

    empty = lambda shape: np.full(shape, fill_value=np.nan, dtype=np.float32)
    videos = empty((h, w, t, len(trial_ids)))
    responses = empty((n, t, len(trial_ids)))
    behavior = empty((d1, t, len(trial_ids)))
    pupil_center = empty((d2, t, len(trial_ids)))

    for i, trial_id in enumerate(tqdm(trial_ids)):
        sample = load_trial(data_dir / MOUSE_IDS[mouse_id], trial_id=trial_id)
        frame = sample["duration"]
        videos[:, :, :frame, i] = rearrange(sample["video"], "1 t h w -> h w t")
        responses[:, :frame, i] = sample["response"]
        behavior[:, :frame, i] = sample["behavior"]
        pupil_center[:, :frame, i] = sample["pupil_center"]

    # compute statistics over trials and time dimensions
    stats = {
        "video": measure(videos, axis=(0, 1, 2, 3)),
        "response": measure(responses, axis=(-2, -1)),
        "behavior": measure(behavior, axis=(-2, -1)),
        "pupil_center": measure(pupil_center, axis=(-2, -1)),
    }
    return stats


def compute_stats(args, mouse_id: str):
    stats = compute_statistics(data_dir=args.data_dir, mouse_id=mouse_id)
    stat_dir = STATISTICS_DIR / f"mouse{mouse_id}"
    stat_dir.mkdir(parents=True, exist_ok=True)
    for data, v in stats.items():
        save_dir = stat_dir / data
        os.makedirs(save_dir, exist_ok=True)
        for name, stat in v.items():
            np.save(save_dir / f"{name}.npy", stat)


def save_neuron_coordinates(data_dir: Path, mouse_id: str):
    neuron_coordinates = np.load(
        data_dir / "meta" / "neurons" / "cell_motor_coordinates.npy"
    )
    filename = METADATA_DIR / "neuron_coordinates" / f"mouse{mouse_id}.npy"
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, neuron_coordinates, allow_pickle=False)
    print(f"Saved mouse {mouse_id} neuron coordinates to {filename}.")


def save_video_ids(data_dir: Path, mouse_id: str) -> None:
    video_ids = np.load(data_dir / "meta" / "trials" / "video_ids.npy")
    filename = METADATA_DIR / "video_ids" / f"mouse{mouse_id}.npy"
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, video_ids)
    print(f"Saved mouse {mouse_id} video IDs to {filename}.")


def save_stimulus_ids(data_dir: Path, mouse_id: str):
    stimulus_ids = np.load(data_dir / "meta" / "trials" / "stimulus_ids.npy")
    filename = METADATA_DIR / "stimulus_ids" / f"mouse{mouse_id}.npy"
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, stimulus_ids)
    print(f"Saved mouse {mouse_id} stimulus IDs to {filename}.")


def main(args):
    for mouse_id in MOUSE_IDS.keys():
        data_dir = args.data_dir / MOUSE_IDS[mouse_id]
        if not data_dir.is_dir():
            continue
        compute_stats(args, mouse_id=mouse_id)
        save_neuron_coordinates(data_dir=data_dir, mouse_id=mouse_id)
        save_video_ids(data_dir=data_dir, mouse_id=mouse_id)
        save_stimulus_ids(data_dir=data_dir, mouse_id=mouse_id)
    print(f"Saved statistics to {STATISTICS_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).parent)
    main(parser.parse_args())
