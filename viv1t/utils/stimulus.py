"""
Helper functions to create artificial stimuli, such as directional gratings and
center-surround gratings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from einops import repeat
from torchvision.transforms.v2 import functional as F

from viv1t import data

VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2


def resize(video: np.ndarray, height: int = 36, width: int = 64) -> np.ndarray:
    video = torch.from_numpy(video)
    video = F.resize(video, size=[height, width])
    video = video.numpy()
    return video


def normalize(
    video: torch.Tensor | np.ndarray,
    o_min: float = -1,  # original min
    o_max: float = 1,  # original max
    t_min: float = MIN,  # target min
    t_max: float = MAX,  # target max
) -> torch.Tensor | np.ndarray:
    """Normalize the pixel range from original (min, max) to target (min, max)"""
    return (video - o_min) * (t_max - t_min) / (o_max - o_min) + t_min


def adjust_contrast(
    video: np.ndarray | torch.Tensor,
    contrast_factor: float,
) -> torch.Tensor | np.ndarray:
    is_tensor = torch.is_tensor(video)
    if not is_tensor:
        video = torch.from_numpy(video)
    assert video.dtype == torch.uint8
    video = rearrange(video, "c t h w -> t c h w")
    video = F.adjust_contrast(video, contrast_factor=contrast_factor)
    video = rearrange(video, "t c h w -> c t h w")
    if not is_tensor:
        video = video.numpy()
    return video


def radius2degree(
    radius: float,
    pixel_width: int = 64,  # in pixel
    pixel_height: int = 36,  # in pixel
    monitor_width: float = 56.5,  # in cm
    monitor_height: float = 31.8,  # in cm
    monitor_distance: int = 15.0,  # in cm
):
    width_factor = monitor_width / pixel_width
    height_factor = monitor_height / pixel_height
    assert np.isclose(
        width_factor, height_factor, rtol=0.01
    ), "pixel vs screen wasn't configured correctly"
    coverage = 2 * np.rad2deg(np.arctan(width_factor * radius / monitor_distance))
    return coverage


def coverage2radius(
    stimulus_size: int,  # visual coverage in degree
    pixel_width: int = 64,  # in pixel
    pixel_height: int = 36,  # in pixel
    monitor_width: float = 56.5,  # in cm
    monitor_height: float = 31.8,  # in cm
    monitor_distance: int = 15.0,  # in cm
) -> float:
    width_factor = monitor_width / pixel_width
    height_factor = monitor_height / pixel_height
    assert np.isclose(
        width_factor, height_factor, atol=0.01
    ), "pixel vs screen wasn't configured correctly"
    radius = monitor_distance * np.tan(np.deg2rad(stimulus_size / 2))
    radius = radius / width_factor
    return radius


def create_circular_mask(
    stimulus_size: int,
    center: tuple[int | float, int | float] = (np.nan, np.nan),
    pixel_width: int = 64,  # in pixel
    pixel_height: int = 36,  # in pixel
    monitor_width: float = 56.5,  # in cm
    monitor_height: float = 31.8,  # in cm
    monitor_distance: int = 15.0,  # in cm
    num_frames: int = 1,
    to_tensor: bool = False,
) -> torch.Tensor | np.ndarray:
    """
    Create a circular boolean mask at the `center` (x coordinate, y coordinate)
    with shape (C=1, T=num_frame, height=height, width=width).

    If `center` is `(NaN, NaN)`, then default to the center of the video.
    Otherwise, set the center of the circle to `center`.
    """
    if np.any(np.isnan(center)):
        center = (pixel_width // 2, pixel_height // 2)
    radius = coverage2radius(
        stimulus_size=stimulus_size,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        monitor_width=monitor_width,
        monitor_height=monitor_height,
        monitor_distance=monitor_distance,
    )
    y, x = np.ogrid[:pixel_height, :pixel_width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    circular_mask = dist_from_center < radius
    circular_mask = repeat(circular_mask, "h w -> () t h w", t=num_frames)
    if to_tensor:
        circular_mask = torch.from_numpy(circular_mask)
    return circular_mask


def create_full_field_grating(
    direction: float,
    cpd: float,
    cpf: float,
    num_frames: int,
    height: int = VIDEO_H,
    width: int = VIDEO_W,
    phase: int = 0,
    contrast: float = 1.0,
    fps: float = data.FPS,
    to_tensor: bool = False,
) -> np.ndarray | torch.Tensor:
    """
    Create a full-field sinusoidal grating image or video

    Reference:
    - https://github.com/pulse2percept/pulse2percept/blob/master/pulse2percept/stimuli/psychophysics.py#L10

    Direction:
        0: ←    45: ↙   90: ↓   135: ↘
        180: →  225: ↗  270: ↑  315: ↖

    Args:
        direction: int, direction of the drifting sinusoidal grating
        cpd: float, spatial frequency of the grating in cycles per degree
        cpf: float, temporal frequency of the grating in cycles per frame
        num_frames: int, number of frames of the video
        height: int, the height of the video in pixel
        width: int, the width of the video in pixel
        phase: int, the initial phase of the grating in degree
        contrast: float, the contrast of the grating, between 0 and 1
        fps: float, the frames per second of the video
        to_tensor: bool, convert the video to torch.Tensor
    """
    assert 0 <= direction <= 360
    assert 0 <= phase <= 360
    assert 0 <= contrast <= 1
    # calculate spatial frequency in cycles per pixel
    dpp = radius2degree(radius=1 / 2, pixel_width=width, pixel_height=height)
    cpp = cpd * dpp
    direction = np.deg2rad(direction + 180)
    phase = np.deg2rad(phase)
    x = np.arange(width) - np.ceil(width / 2.0)
    y = np.arange(height) - np.ceil(height / 2.0)
    time = np.linspace(start=0, stop=num_frames / fps * 1000, num=num_frames)
    X, Y, T = np.meshgrid(x, y, np.arange(len(time)), indexing="xy")
    pattern = np.cos(
        -2 * np.pi * cpp * np.cos(direction) * X
        + 2 * np.pi * cpp * np.sin(direction) * Y
        + 2 * np.pi * cpf * T
        + phase
    )
    pattern = contrast * pattern
    pattern = normalize(pattern, o_min=-1, o_max=1)
    pattern = rearrange(pattern, "h w t -> () t h w")
    pattern = pattern.astype(np.uint8)
    if to_tensor:
        pattern = torch.from_numpy(pattern)
    return pattern


def create_center_surround_grating(
    stimulus_size: int | float | np.ndarray,
    center: tuple[int | float, int | float],
    center_direction: int | float | np.ndarray,
    surround_direction: int | float | np.ndarray,
    cpd: float,
    cpf: float,
    num_frames: int,
    height: int = VIDEO_H,
    width: int = VIDEO_W,
    phase: int = 0,
    contrast: float = 1.0,
    fps: float = data.FPS,
    to_tensor: bool = False,
) -> torch.Tensor:
    kws = {
        "cpd": cpd,
        "cpf": cpf,
        "num_frames": num_frames,
        "phase": phase,
        "contrast": contrast,
        "fps": fps,
        "to_tensor": to_tensor,
    }
    circle_mask = create_circular_mask(
        stimulus_size,
        center=center,
        pixel_width=width,
        pixel_height=height,
        num_frames=num_frames,
        to_tensor=to_tensor,
    )
    if surround_direction == -1:
        # generate center grating only
        center = create_full_field_grating(direction=center_direction, **kws)
        surround = GREY_COLOR
    else:
        center = create_full_field_grating(direction=center_direction, **kws)
        surround = create_full_field_grating(direction=surround_direction, **kws)
    if to_tensor:
        pattern = torch.where(circle_mask, center, surround)
    else:
        pattern = np.where(circle_mask, center, surround)
    return pattern


def load_neuron_RF_center(
    output_dir: Path, mouse_id: str, neuron: int, verbose: int = 1
) -> tuple[int | float, int | float]:
    """
    Return the center of the neuron aRF if exists, otherwise return (NaN, NaN).
    """
    try:
        RF_centers = pd.read_parquet(output_dir / "aRF.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find aRF.parquet in {output_dir}. Please estimate the "
            f"aRFs of the model with tuning_retinotopy/predict_noise.py and "
            f"tuning_retinotopy/fit_aRFs.py first."
        )
    neuron_df = RF_centers[
        (RF_centers.mouse == mouse_id) & (RF_centers.neuron == neuron)
    ]
    assert len(neuron_df) == 1
    if neuron_df.iloc[0].bad_fit:
        if verbose:
            print(f"Mouse {mouse_id} neuron {neuron:04d} does not has a good aRF fit.")
        return (np.nan, np.nan)
    center_x = np.round(neuron_df.iloc[0].center_x, decimals=0)
    center_y = np.round(neuron_df.iloc[0].center_y, decimals=0)
    return (center_x, center_y)


def load_population_RF_center(
    output_dir: Path, mouse_id: str
) -> tuple[int | float, int | float]:
    """
    Return the population aRF center
    """
    try:
        RF_centers = pd.read_parquet(output_dir / "aRF.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find aRF.parquet in {output_dir}. Please estimate the "
            f"aRFs of the model with tuning_retinotopy/predict_noise.py and "
            f"tuning_retinotopy/fit_aRFs.py first."
        )
    df = RF_centers[
        (RF_centers.mouse == mouse_id)
        & (~RF_centers.center_x.isnull())
        & ((~RF_centers.center_y.isnull()))
    ]
    center_x = np.round(np.mean(df.center_x.values), decimals=0)
    center_y = np.round(np.mean(df.center_y.values), decimals=0)
    return (int(center_x), int(center_y))


def load_group_RF_center(
    output_dir: Path, mouse_id: str, neurons: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group neurons by their aRF centers to the nearest pixels, then return
    the unique aRF centers and which group each neuron corresponds to.
    NaN RF_center means that neuron group had bad aRF fit.
    """
    try:
        RF_centers = pd.read_parquet(output_dir / "aRF.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find aRF.parquet in {output_dir}. Please estimate the "
            f"aRFs of the model with tuning_retinotopy/predict_noise.py and "
            f"tuning_retinotopy/fit_aRFs.py first."
        )
    RF_centers = RF_centers[RF_centers.mouse == mouse_id]
    if neurons is not None:
        RF_centers = RF_centers[RF_centers.neuron.isin(neurons)]
    centers = RF_centers[["neuron", "center_x", "center_y"]].to_numpy()
    # replace NaN values (i.e. neuron has bad aRF fit) to -1 because np.unique
    # has issue with multi-dimension array with NaN
    assert -1 not in centers
    centers = np.nan_to_num(centers, nan=-1)
    # round RF centers to the nearest pixel
    centers[:, 1] = np.round(centers[:, 1], decimals=0)
    centers[:, 2] = np.round(centers[:, 2], decimals=0)
    # group neurons based on their RF centers
    unique_centers, neuron_groups = np.unique(
        centers[:, 1:], return_inverse=True, axis=0
    )
    # replace -1 with NaN
    unique_centers[unique_centers == -1] = np.nan
    return unique_centers, neuron_groups
