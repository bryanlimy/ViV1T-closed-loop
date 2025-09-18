"""
Estimate the centre coordinates, radius and color of all Gaussian dot
stimuli and save the results to OUTPUT_DIR
"""

import argparse
import os
from argparse import RawTextHelpFormatter
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot

OUTPUT_DIR = data.METADATA_DIR / "ood_features" / "gaussian_dots"

plot.set_font()


def plot_frame_with_circle(frame, x, y, radius, dot_is_black):
    # Create a new figure and axes
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2), dpi=120)
    # Display the frame
    ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
    # Create a circle patch
    circle = patches.Circle(
        (x, y), radius, linewidth=1, edgecolor="r", facecolor="none"
    )
    # Add the patch to the axes
    ax.add_patch(circle)
    # Create a title with the center, radius, and color information
    ax.set_title(
        f'Center: ({x:.1f}, {y:.1f}), Radius: {radius}\nDot color: {"black" if dot_is_black else "white"}',
        fontsize=10,
        linespacing=1,
        pad=0,
    )

    # Show the plot
    plt.show()
    plt.close(figure)


def project_onto_axes(image):
    projection_x = np.sum(image, axis=0)
    x = np.where(projection_x > 0)
    projection_y = np.sum(image, axis=1)
    y = np.where(projection_y > 0)
    return x, y


def extract_gaussian_dot_params(video: np.ndarray, plot: bool = False) -> np.ndarray:
    video = rearrange(video, "1 t h w -> t h w")
    t, h, w = video.shape
    params = np.full(shape=(t, 4), fill_value=np.nan, dtype=np.float32)
    for i, frame in enumerate(video):
        frame_norm = (frame - np.min(frame)) / np.ptp(frame)
        dot_is_black = np.mean(frame_norm) > 0.5
        if dot_is_black:
            frame_norm = 1 - frame_norm
        frame_norm = np.where(frame_norm > 0.5, 1, 0)
        x, y = project_onto_axes(frame_norm)
        mu_x, mu_y = np.mean(x), np.mean(y)
        radius = np.ptp(x) / 2
        params[i] = [mu_x, mu_y, radius, dot_is_black]
        if plot:
            plot_frame_with_circle(frame, mu_x, mu_y, radius, dot_is_black)
    return params


def save_parameters(output_dir: Path, trial_ids: np.ndarray, params: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in trial_ids:
        np.save(os.path.join(output_dir, f"{trial_id}.npy"), params)


def estimate_mouse(data_dir: Path, mouse_id: str):
    mouse_dir = data_dir / data.MOUSE_IDS[mouse_id]
    # get trial IDs where Gaussian dot (stimulus type: 2) was shown
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 2)[0]
    # directory to save parameters
    output_dir = OUTPUT_DIR / f"mouse{mouse_id}"
    if trial_ids.size:
        # get video IDs and find unique Gaussian dot trials
        video_ids = data.get_video_ids(mouse_id)
        for video_id in tqdm(np.unique(video_ids[trial_ids]), desc=f"Mouse {mouse_id}"):
            # get the first trial with video_id
            trial_id = np.where(video_ids == video_id)[0][0]
            sample = data.load_trial(mouse_dir=mouse_dir, trial_id=trial_id)
            assert sample["duration"] == 315
            params = extract_gaussian_dot_params(sample["video"], plot=False)
            save_parameters(
                output_dir=output_dir,
                trial_ids=np.where(video_ids == video_id)[0],
                params=params,
            )


def main(args):
    for mouse_id in data.SENSORIUM_OLD:
        estimate_mouse(args.data_dir, mouse_id)
    print(f"Saved gabor parameters to {OUTPUT_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="../data/sensorium",
        help="path to directory where the dataset is stored.",
    )
    main(parser.parse_args())
