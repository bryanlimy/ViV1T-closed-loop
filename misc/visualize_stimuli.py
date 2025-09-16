import argparse
from pathlib import Path
from typing import Dict

import av
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from einops import repeat
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot

plt.style.use("seaborn-v0_8-deep")
plot.set_font()

matplotlib.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams["animation.embed_limit"] = 2**64

LABEL_FONTSIZE = 12
FPS = 30
DPI = 100

PLOT_DIR = Path("figures")


def animate_stimulus(video: np.ndarray, filename: Path):
    c, t, h, w = video.shape
    filename.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(filename, mode="w")

    stream = container.add_stream("libx264", rate=FPS)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "1", "preset": "slow"}

    video = repeat(video, "() T H W -> T H W C", C=3)
    video = video.astype(np.uint8)
    for i in range(video.shape[0]):
        frame = av.VideoFrame.from_ndarray(video[i], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


# def animate_stimulus(
#     video: np.ndarray, video_stats: Dict[str, float | np.ndarray], filename: Path = None
# ):
#     h, w = video.shape[2], video.shape[3]
#     f_h, f_w = (h / 16) + (7 / h), w / 16
#     a_w = 0.99
#     a_h = (h / 16) / f_h
#
#     figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
#     ax = figure.add_axes(rect=((1 - a_w) / 2, (1 - a_w) / 2, a_w, a_h))
#
#     imshow = ax.imshow(
#         np.random.rand(h, w),
#         cmap="gray",
#         aspect="equal",
#         vmin=video_stats["min"],
#         vmax=video_stats["max"],
#     )
#     pos = ax.get_position()
#     text = ax.text(
#         x=0,
#         y=pos.y1 + 0.11,
#         s="",
#         ha="left",
#         va="center",
#         fontsize=LABEL_FONTSIZE,
#         transform=ax.transAxes,
#     )
#     ax.grid(linewidth=0)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     def animate(frame: int):
#         imshow.set_data(video[0, frame, :, :])
#         text.set_text(f"Movie Frame {frame:03d}")
#
#     anim = FuncAnimation(
#         figure, animate, frames=video.shape[1], interval=int(1000 / FPS)
#     )
#     if filename is not None:
#         filename.parent.mkdir(parents=True, exist_ok=True)
#         anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
#         try:
#             plt.close(figure)
#         except AttributeError as e:
#             print(f"AttributeError in plt.close(figure): {e}.")
#     plt.close(figure)


def plot_frames(video: np.ndarray, filename: Path):
    t, h, w = video.shape[1], video.shape[2], video.shape[3]
    f_h, f_w = (h / 16) + (7 / h), w / 16
    a_w = 0.99
    a_h = (h / 16) / f_h

    for frame in range(t):
        figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
        ax = figure.add_axes(rect=((1 - a_w) / 2, (1 - a_w) / 2, a_w, a_h))
        image = video[0, frame]
        ax.imshow(image, cmap="gray", aspect="equal", vmin=0, vmax=255.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(8)
        plot.save_figure(
            figure,
            filename=filename / f"frame{frame:03d}.svg",
            layout="none",
            dpi=DPI,
        )


def main(args):
    rng = np.random.default_rng(seed=1234)

    # for mouse_id in ["K", "M"]:
    #     mouse_dir = args.data_dir / data.MOUSE_IDS[mouse_id]
    #     if not mouse_dir.is_dir():
    #         continue
    #     print(f"\nPlot visual stimuli from Mouse {mouse_id}")
    #     tiers = data.get_tier_ids(data_dir=args.data_dir, mouse_id=mouse_id)
    #     for tier in np.unique(tiers):
    #         plot_dir = PLOT_DIR / 'stimulus' / f"mouse{mouse_id}" / tier
    #         trial_ids = np.where(tiers == tier)[0]
    #         if args.random_sample:
    #             trial_ids = sorted(rng.choice(trial_ids, size=10, replace=False))
    #         for trial_id in tqdm(trial_ids, desc=tier):
    #             sample = data.load_trial(mouse_dir=mouse_dir, trial_id=trial_id)
    #             filename = plot_dir / f"mouse{mouse_id}_trial{trial_id:03d}.mp4"
    #             if not filename.exists():
    #                 animate_stimulus(video=sample["video"], filename=filename)

    mouse_id, trial_id = "A", 213
    mouse_dir = args.data_dir / data.MOUSE_IDS[mouse_id]
    video_stats = data.load_stats(mouse_id=mouse_id)["video"]
    sample = data.load_trial(mouse_dir=mouse_dir, trial_id=trial_id)
    plot_frames(
        video=sample["video"],
        filename=PLOT_DIR
        / "stimulus_frame"
        / f"mouse{mouse_id}"
        / f"trial{trial_id:03d}",
    )
    # animate_stimulus(
    #     video=sample["video"],
    #     video_stats=video_stats,
    #     filename=PLOT_DIR
    #     / f"mouse{mouse_id}"
    #     / "live_bonus"
    #     / f"trial{trial_id:03d}.gif",
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../data")
    parser.add_argument("--random_sample", action="store_true")
    main(parser.parse_args())
