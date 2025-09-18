from math import ceil
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation
from torch.utils.data import DataLoader

from viv1t import data
from viv1t.utils import plot

SKIP = 50  # skip the first 50 frames from each trial
MIN, MAX = 0, 255.0
MAX_FRAME = 300

FPS = 30
DPI = 200
FONTSIZE = 10

plot.set_font()


def get_behaviors(
    mouse_id: str, ds: DataLoader, device: torch.device
) -> (torch.Tensor, torch.Tensor):
    behavior, pupil_center = data.get_mean_behaviors(
        mouse_id=mouse_id, num_frames=MAX_FRAME
    )
    behavior = ds.dataset.transform_behavior(behavior)
    pupil_center = ds.dataset.transform_pupil_center(pupil_center)

    to_batch = lambda x: x.to(device)[None, :]

    return to_batch(behavior), to_batch(pupil_center)


def get_sensorium_response_stats(
    output_dir: Path, mouse_id: str, neuron: int | np.ndarray
) -> tuple[float | None, float | None]:
    """Get statistics of the predicted response on the Sensorium 2023 dataset"""
    filename = output_dir / "response_stats.parquet"
    if filename.exists():
        df = pd.read_parquet(filename)
        df = df.loc[(df.mouse == mouse_id) & (df.neuron == neuron)]
        return df["max"].max(), df["mean"].mean()
    return None, None


def get_predicted_response_stats(
    response: np.ndarray, presentation_mask: np.ndarray | None = None
) -> tuple[float, float]:
    """Get the max and sum response during stimulus presentation"""
    t = response.shape[0]
    if presentation_mask is not None:
        response = presentation_mask[-t:] * response
    return float(np.max(response)), float(np.sum(response))


def animate_stimulus(
    video: np.ndarray,
    response: np.ndarray,
    neuron: int | None,
    filename: Path,
    loss: torch.Tensor | np.ndarray | None = None,
    ds_max: float | torch.Tensor | np.ndarray = None,
    ds_mean: float | torch.Tensor | np.ndarray = None,
    presentation_mask: torch.Tensor | np.ndarray | None = None,
    skip: int = SKIP,
):
    assert len(video.shape) == 4
    t, h, w = video.shape[1], video.shape[2], video.shape[3]

    if presentation_mask is not None and torch.is_tensor(presentation_mask):
        presentation_mask = presentation_mask.to("cpu").numpy()

    f_w, f_h = 3.6, 2.3
    figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
    get_width = lambda height: height * (w / h) * (f_h / f_w)
    spatial_height = 0.68
    ax1 = figure.add_axes(rect=(0.12, 0.25, get_width(spatial_height), spatial_height))
    ax2 = figure.add_axes(rect=(0.12, 0.10, get_width(spatial_height), 0.08))

    x_ticks = np.arange(t + 10, step=10, dtype=int)
    min_value, max_value = 0, np.max(response)
    if ds_max is not None:
        max_value = max(max_value, ds_max)
    max_value = ceil(max_value)

    max_response, sum_response = get_predicted_response_stats(
        response=response, presentation_mask=presentation_mask
    )
    title = rf"N{neuron:04d} $\Delta F/F$ (max: {max_response:.1f}, sum: {sum_response:.0f})"

    x = np.arange(t)

    imshow = ax1.imshow(
        np.random.rand(h, w), cmap="gray", aspect="equal", vmin=MIN, vmax=MAX
    )
    pos = ax1.get_position()
    text = ax1.text(
        x=0,
        y=pos.y1 + 0.12,
        s="",
        ha="left",
        va="center",
        fontsize=FONTSIZE,
        transform=ax1.transAxes,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    line = ax2.plot(
        [],
        [],
        linewidth=2,
        color="forestgreen",
        zorder=1,
        clip_on=False,
    )[0]
    ax2.set_xlim(x_ticks[0], x_ticks[-1])
    plot.set_xticks(
        axis=ax2, ticks=x_ticks, tick_labels=x_ticks, tick_fontsize=FONTSIZE
    )
    ax2.set_ylim(min_value, max_value)
    plot.set_yticks(
        axis=ax2,
        ticks=[min_value, max_value],
        tick_labels=[min_value, max_value],
        tick_fontsize=FONTSIZE,
    )
    ax2.text(
        x=x_ticks[0],
        y=max_value,
        s=title,
        ha="left",
        va="bottom",
        fontsize=FONTSIZE,
        transform=ax2.transData,
    )
    if ds_max is not None:
        ax2.axhline(
            y=ds_max,
            color="orangered",
            linewidth=1.2,
            linestyle="dashed",
            zorder=0,
            clip_on=False,
        )
    if ds_mean is not None:
        ax2.axhline(
            y=ds_mean,
            color="dodgerblue",
            linewidth=1.2,
            linestyle="dotted",
            zorder=0,
            clip_on=False,
        )
    if presentation_mask is not None:
        ax2.fill_between(
            x,
            y1=0,
            y2=max_value,
            where=presentation_mask,
            facecolor="#e0e0e0",
            edgecolor="none",
            zorder=-1,
        )
    plot.set_ticks_params(axis=ax2, length=2, pad=1, linewidth=1)
    sns.despine(ax=ax2, trim=True)

    title = f" (best loss: {loss:.2f})" if loss else ""

    def animate(frame: int):
        artists = [imshow]
        imshow.set_data(video[0, frame, :, :])
        text.set_text(f"Frame: {frame :03d}" + title)
        if frame >= skip:
            line.set_data(x[skip : frame + 1], response[: frame - skip + 1])
            artists.append(line)
        return artists

    anim = FuncAnimation(
        figure, func=animate, frames=t, interval=int(1000 / FPS), blit=True
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
    plt.close(figure)


def get_animate_function(
    neuron: int | None,
    blank_size: int,
    pattern_size: int,
    ds_max: float | torch.Tensor | np.ndarray = None,
    ds_mean: float | torch.Tensor | np.ndarray = None,
    presentation_mask: torch.Tensor | np.ndarray | None = None,
    skip: int = 0,
):
    if presentation_mask is not None and torch.is_tensor(presentation_mask):
        presentation_mask = presentation_mask.cpu().numpy()
    # crop to presentation window
    start = -(blank_size + pattern_size + 10)
    end = -(blank_size - 10)

    def animate(
        video: torch.Tensor,
        response: torch.Tensor,
        loss: torch.Tensor | None,
        filename: Path,
    ):
        video = video.detach().to("cpu", torch.float32).numpy()
        response = response.detach().to("cpu", torch.float32).numpy()
        video = video[:, start:end]
        response = response[start:end]
        _presentation_mask = None
        if presentation_mask is not None:
            _presentation_mask = presentation_mask[start:end]
        animate_stimulus(
            video=video,
            response=response,
            neuron=neuron,
            loss=loss,
            filename=filename,
            ds_max=ds_max,
            ds_mean=ds_mean,
            presentation_mask=_presentation_mask,
            skip=skip,
        )

    return animate


def transform_video(ds: DataLoader, device: torch.device) -> Callable:
    """Helper function to preprocess video"""
    transform_input = ds.dataset.transform_input
    stats = {k: v.to(device) for k, v in ds.dataset.video_stats.items()}
    eps = torch.tensor(torch.finfo(torch.float32).eps, device=device)

    def preprocess_video(video: torch.Tensor) -> torch.Tensor:
        match transform_input:
            case 1:
                video = (video - stats["mean"]) / (stats["std"] + eps)
            case 2:
                video = (video - stats["min"]) / (stats["max"] - stats["min"])
        return video

    return preprocess_video


def transform_response(ds: DataLoader, device: torch.device) -> Callable:
    """Helper function to inverse transform response"""
    transform_output = ds.dataset.transform_output
    stats = {k: v.to(device) for k, v in ds.dataset.response_stats.items()}
    response_precision = ds.dataset.response_precision.to(device)

    def postprocess_response(response: torch.Tensor):
        match transform_output:
            case 1:
                response = response / response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    return postprocess_response


def get_presentation_mask(
    blank_size: int,
    pattern_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a mask to mask out the blank frames in the predicted response"""
    return torch.cat(
        [
            torch.zeros(blank_size, device=device, dtype=dtype),
            torch.ones(pattern_size, device=device, dtype=dtype),
            torch.zeros(blank_size, device=device, dtype=dtype),
        ]
    )


def save_stimulus_checkpoint(
    param: torch.Tensor,
    data: dict[str, torch.Tensor],
    step: int,
    mouse_id: str,
    neuron: int | None,
    presentation_mask: torch.Tensor | None,
    filename: Path,
):
    ckpt = {
        "param": param,
        "video": data["video"].to(torch.float32),
        "response": data["response"].to(torch.float32),
        "step": step,
        "mouse": str(mouse_id),
    }
    if neuron is not None:
        ckpt["neuron"] = int(neuron)
    if "loss" in data:
        ckpt["loss"] = data["loss"]
    if presentation_mask is not None:
        ckpt["presentation_mask"] = presentation_mask
    torch.save(ckpt, f=filename)
