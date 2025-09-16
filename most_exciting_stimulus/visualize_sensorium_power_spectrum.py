import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from tqdm import tqdm

from viv1t import data

DATA_DIR = Path("../data")
PLOT_DIR = Path("figures") / "power_spectrum"

MAX_FRAME = 300


def load_video(mouse_dir: Path, trial_id: int | str) -> torch.Tensor:
    video = np.load(mouse_dir / "data" / "videos" / f"{trial_id}.npy")
    return torch.from_numpy(video[:, :, :MAX_FRAME]).to(torch.float32)


def load_videos(mouse_id: str) -> torch.Tensor:
    tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
    trial_ids = np.where(tiers == "train")[0]
    videos = torch.stack(
        [
            load_video(mouse_dir=DATA_DIR / data.MOUSE_IDS[mouse_id], trial_id=trial_id)
            for trial_id in trial_ids
        ]
    )
    videos = rearrange(videos, "b h w t -> b t h w")
    return videos


def rfft2d_freqs(h: int, w: int):
    """Computes 2D spectrum frequencies.

    References:
    - https://github.com/greentfrapp/lucent/blob/dev/lucent/optvis/param/spatial.py#L34
    - https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
    """
    fy = torch.fft.fftfreq(h)[:, None]
    if w % 2 == 1:
        fx = torch.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = torch.fft.fftfreq(w)[: w // 2 + 1]
    return torch.sqrt(fx * fx + fy * fy)


def main():
    mouse_id = "N"

    videos = load_videos(mouse_id)

    freq = torch.fft.fft2(videos, dim=(2, 3))

    ps = torch.abs(freq) ** 2

    freqs = rfft2d_freqs(h=videos.shape[2], w=videos.shape[3])
    print(f"Saved plot to {PLOT_DIR}")


if __name__ == "__main__":
    main()
