"""Estimate the spatial frequency-space spectrum of the Sensorium 2023 training set"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot

plot.set_font()

DATA_DIR = Path("../data")
PLOT_DIR = Path("figures") / "kl_divergence"

TICK_FONTSIZE = 9
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
DPI = 240


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p, q = F.softmax(p, dim=0), F.softmax(q, dim=0)
    return torch.sum(p * torch.log(p / (q + 1e-8)))


def plot_histogram(
    values: torch.Tensor | np.ndarray,
    filename: Path,
    num_bins: int = 40,
    title: str = "",
    x_label: str = "",
):
    if torch.is_tensor(values):
        values = values.cpu().numpy()

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), dpi=DPI)

    h_y, h_x, _ = ax.hist(
        values,
        bins=num_bins,
        linewidth=1,
        color="dodgerblue",
        edgecolor="black",
        clip_on=False,
        weights=np.ones(len(values)) / len(values),
    )
    max_value = np.ceil(np.max(h_y) * 10) / 10

    min_x, max_x = np.floor(np.min(h_x)), np.ceil(np.max(h_x))
    x_ticks = np.linspace(min_x, max_x, 3)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=[f"{i:.1f}" for i in x_ticks],
        label=x_label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.linspace(0, max_value, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=(y_ticks * 100).astype(int),
        label="% of frame",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    if title:
        ax.set_title(title, fontsize=LABEL_FONTSIZE, pad=0)
    plot.set_ticks_params(ax, length=2, pad=1, linewidth=1)
    sns.despine(ax=ax)

    # figure.tight_layout()
    plot.save_figure(figure, filename=filename, dpi=DPI)


def process_mouse(mouse_id: str):
    # load all videos from the training set
    tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
    trial_ids = np.where(tiers == "train")[0]
    samples = [
        data.load_trial(
            mouse_dir=DATA_DIR / data.MOUSE_IDS[mouse_id],
            trial_id=trial_id,
            to_tensor=True,
        )
        for trial_id in trial_ids
    ]
    videos = torch.stack([sample["video"] for sample in samples])
    videos = rearrange(videos, "b () t h w -> b t h w")
    # compute 2D FFT over the spatial dimension of each frame
    freqs = torch.fft.fftn(videos, dim=(2, 3), norm="ortho")
    # flatten height and width dimension
    freqs = rearrange(freqs, "b t h w -> b t (h w)")
    freqs = torch.square(freqs.real) + torch.square(freqs.imag)
    freqs = torch.log(freqs + 1e-8)
    # unfold frame dimension into clips of 30 frames
    freqs = freqs.unfold(dimension=1, size=30, step=5)
    freqs = rearrange(freqs, "b c d t -> (b c) t d")
    # compute average freqs across all frame
    mean_freq = torch.mean(freqs, dim=(0, 1))
    plot_histogram(
        values=mean_freq,
        filename=PLOT_DIR / f"mouse{mouse_id}_average_freq.jpg",
        title=f"Mouse {mouse_id} freq histogram",
        x_label=r"$log((freq.real)^2 + (freq.imag)^2)$" + "\naverage across 30 frames",
    )
    divergences = []
    for b in range(freqs.shape[0]):
        clip_freq = torch.mean(freqs[b], dim=0)
        divergence = kl_divergence(mean_freq, clip_freq)
        divergences.append(divergence)
    divergences = torch.stack(divergences)
    plot_histogram(
        values=divergences,
        filename=PLOT_DIR / f"mouse{mouse_id}_KL_histogram.jpg",
        num_bins=40,
        title=f"Mouse {mouse_id} KL histogram",
        x_label="KL divergence between average\nfreq per movie and average\nfreq of all frames",
    )
    # store KL divergence and the average spatial frequency
    stat_dir = data.STATISTICS_DIR / f"mouse{mouse_id}" / "video"
    np.save(stat_dir / "freq.npy", mean_freq.cpu().numpy())
    np.save(stat_dir / "kl_divergence.npy", divergences.cpu().numpy())
    print(f"Save average frequency and KL divergence to {stat_dir}")


def main():
    for mouse_id in ["O"]:
        process_mouse(mouse_id)


if __name__ == "__main__":
    main()
