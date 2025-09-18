"""
Estimate the orientation, wavelength, and frequency of all unique
drifting gabor stimulus and save the results to OUTPUT_DIR
"""

from itertools import product
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from tqdm import tqdm

from viv1t import data
from viv1t.data import MAX_FRAME

matplotlib.use("TkAgg")

DATA_DIR = Path("../data/sensorium")
META_DIR = data.METADATA_DIR / "ood_features" / "drifting_gabor"

BLOCK_SIZE = 25  # number of frames for each drifting Gabor filter
FPS = 30


# min and max pixel values
MIN, MAX = 0, 255


def normalize(x: np.ndarray, x_min: float = -1, x_max: float = 1):
    """Normalize the pixel range to [MIN, MAX]"""
    return (x - x_min) * (MAX - MIN) / (x_max - x_min) + MIN


def create_gabor_filter(
    w: int,
    h: int,
    sigma: float,
    wavelength: float,
    direction: float,
    frequency: float,
):
    # create a time array from 0 to 1 with the number of points equal to the number of frames
    frames = np.linspace(0, 1, BLOCK_SIZE)
    # calculate the aspect ratio
    aspect_ratio = w / h
    # create a 2D grid
    x, y = np.meshgrid(
        np.linspace(-aspect_ratio, aspect_ratio, w),
        np.linspace(-1, 1, h),
    )
    # create a Gaussian mask
    gaussian = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    # convert the direction to radians
    rad = np.radians(direction)
    # create rotated coordinates
    x_rot = x * np.cos(rad) - y * np.sin(rad)
    # create the time-evolved Gabor filter by multiplying the spatial filter by the temporal sinusoid
    pattern = np.array(
        [
            np.cos(2 * np.pi * (x_rot / wavelength + w * frequency * frame))
            for frame in frames
        ],
        dtype=np.float32,
    )
    pattern = gaussian * pattern
    pattern = normalize(pattern)
    return pattern


def create_gabor_bank(
    w: int,
    h: int,
    sigma: float,
    directions: np.ndarray,
    wavelengths: np.ndarray,
    frequencies: np.ndarray,
):
    gabor_bank = {}
    for direction, wavelength, frequency in product(
        directions, wavelengths, frequencies
    ):
        gabor_bank[(direction, wavelength, frequency)] = create_gabor_filter(
            w=w,
            h=h,
            sigma=sigma,
            wavelength=wavelength,
            direction=direction,
            frequency=frequency,
        )
    return gabor_bank


def plot_animation(
    block: np.ndarray,
    filter: np.ndarray,
    params: Tuple[float, float, float],
    corr: float,
    interval: int = int(1000 / FPS),
):
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 2), dpi=120)

    def update(frame: int):
        axes[0].cla()
        axes[1].cla()
        axes[0].imshow(block[frame], cmap="gray", animated=True, vmin=MIN, vmax=MAX)
        axes[0].set_xlabel("recorded", fontsize=8, labelpad=2)
        axes[1].imshow(filter[frame], cmap="gray", animated=True, vmin=MIN, vmax=MAX)
        axes[1].set_xlabel("generated", fontsize=8, labelpad=2)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].text(
            x=0.5,
            y=1.05,
            s=f"direction: {params[0]:.0f} {data.DIRECTIONS[int(params[0])]}   "
            f"wavelength: {params[1]:.03f}\n"
            f"frequency: {params[2]:.03f}   "
            f"corr: {corr:.2f}",
            transform=axes[0].transAxes,
        )

    ani = animation.FuncAnimation(figure, update, frames=len(block), interval=interval)
    plt.show()
    plt.close(figure)


def find_best_match(block, gabor_bank):
    correlations = {
        params: np.corrcoef(block.flatten(), gabor.real.flatten())[0, 1]
        for params, gabor in gabor_bank.items()
    }
    best_match_params = max(correlations, key=correlations.get)
    return best_match_params, correlations[best_match_params]


def extract_gabor_params(video: np.ndarray, plot: bool = False):
    video = rearrange(video, "c t h w -> t c h w")
    t, c, h, w = video.shape

    # Establish filter bank
    sigma = 0.3
    directions = np.array(list(data.DIRECTIONS.keys()), dtype=np.int32)
    wavelengths = np.array([0.18, 0.38, 0.65], dtype=np.float32)
    frequencies = np.array([0.008, 0.022, 0.055], dtype=np.float32)
    # gabor filters bank where key is (orientation, wavelength, frequency)
    gabor_bank = create_gabor_bank(
        w=w,
        h=h,
        sigma=sigma,
        directions=directions,
        wavelengths=wavelengths,
        frequencies=frequencies,
    )
    params = np.full(shape=(t, 3), fill_value=np.nan, dtype=np.float32)
    # # loop over the video in blocks of 25 frames
    for i in range(0, t, BLOCK_SIZE):
        block = video[i : i + BLOCK_SIZE, 0]
        best_fit_params, best_corr = find_best_match(block, gabor_bank)
        best_fit = gabor_bank[best_fit_params]
        if plot:
            plot_animation(block, best_fit, best_fit_params, best_corr)
        params[i : i + BLOCK_SIZE] = best_fit_params
    return params


def save_parameters(output_dir: Path, trial_ids: np.ndarray, params: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in trial_ids:
        np.save(output_dir / f"{trial_id}.npy", params)


def estimate_mouse(mouse_id: str):
    mouse_dir = DATA_DIR / data.MOUSE_IDS[mouse_id]
    # get trial IDs where drifting gabor (stimulus type: 4) was shown
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 4)[0]
    if trial_ids.size:
        # get video IDs and find unique gabor trials
        video_ids = data.get_video_ids(mouse_id)
        for video_id in tqdm(np.unique(video_ids[trial_ids]), desc=f"Mouse {mouse_id}"):
            # get the first trial with the video_id
            trial_id = np.where(video_ids == video_id)[0][0]
            sample = data.load_trial(mouse_dir=mouse_dir, trial_id=trial_id)
            assert sample["duration"] == MAX_FRAME
            params = extract_gabor_params(sample["video"], plot=False)
            save_parameters(
                output_dir=META_DIR / f"mouse{mouse_id}",
                trial_ids=np.where(video_ids == video_id)[0],
                params=params,
            )


if __name__ == "__main__":
    for mouse_id in data.SENSORIUM_OLD:
        estimate_mouse(mouse_id)
    print(f"Saved gabor parameters to {META_DIR}.")
