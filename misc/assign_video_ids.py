"""
Script to extract unique video IDs from all Sensorium 2023 visual stimuli and
save them to data/metadata/video_ids
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from viv1t import data

DATA_DIR = Path("../data/sensorium")


def equal(video1: np.ndarray, video2: np.ndarray, atol: float = 1.0) -> bool:
    """Return True if more than 85% of the frames in two videos are equal"""
    assert video1.shape == video2.shape
    video1, video2 = np.nan_to_num(video1, nan=-1), np.nan_to_num(video2, nan=-1)
    frame_equal = np.all(np.abs(video1 - video2) <= atol, axis=(0, 1))
    return np.mean(frame_equal) > 0.85


def load_video(mouse_dir: Path, trial_id: int | str) -> np.ndarray:
    return np.load(mouse_dir / "data" / "videos" / f"{trial_id}.npy")


def check_repeats(video_ids: np.ndarray, tiers: np.ndarray):
    """
    Check the validation, live main, live bonus, final main and final bonus sets
    to have at least 5 repeats
    """
    for tier in data.TIERS.keys():
        if tier == "train":
            continue
        trial_ids = np.where(tiers == tier)[0]
        for video_id in np.unique(video_ids[trial_ids]):
            repeats = np.where(video_ids == video_id)[0]
            if len(repeats) < 5:
                print(
                    f"WARNING: video_id {video_id} ({data.TIERS[tier]} trials: "
                    f"{repeats.tolist()}) has less than 5 repeats."
                )
            if len(repeats) > 10:
                print(
                    f"WARNING: video_id {video_id} ({data.TIERS[tier]} trials: "
                    f"{repeats.tolist()}) has more than 10 repeats."
                )


def assign_video_ids(mouse_dir: Path) -> np.ndarray:
    """
    Iterate all visual stimuli and assign unique IDs to each unique stimulus.
    """
    mouse_id = data.MOUSE_DIRS[mouse_dir.name]
    tiers = np.load(mouse_dir / "meta" / "trials" / "tiers.npy")
    video_id, unique_videos = 0, []  # list of unique video arrays
    video_ids = np.full(len(tiers), fill_value=-1, dtype=np.int32)
    for trial_id, tier in enumerate(tqdm(tiers, desc=f"Mouse {mouse_id}")):
        if tier == "none":
            continue
        video = load_video(mouse_dir, trial_id)
        if tier != "train":  # all training trials are unique videos
            unique = True
            for unique_id in range(len(unique_videos)):
                if equal(unique_videos[unique_id], video):
                    video_ids[trial_id] = unique_id
                    unique = False
                    break
            if not unique:
                continue
        unique_videos.append(video)
        video_ids[trial_id] = video_id
        video_id += 1
    check_repeats(video_ids=video_ids, tiers=tiers)
    print(f"Found {len(unique_videos)} unique videos for Mouse {mouse_id}.\n")
    return video_ids


def main():
    for mouse_id in data.SENSORIUM:
        video_ids = assign_video_ids(mouse_dir=DATA_DIR / data.MOUSE_IDS[mouse_id])
        filename = data.METADATA_DIR / "video_ids" / f"mouse{mouse_id}.npy"
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, video_ids, allow_pickle=False)
        print(f"Saved mouse {mouse_id} video IDs to {filename}.")


if __name__ == "__main__":
    main()
