from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from viv1t.data import MAX_FRAME
from viv1t.utils import plot

sns.set_style("ticks")

PATTERN_SIZE = 30
PAD = 20
BLANK_SIZE = (MAX_FRAME - PATTERN_SIZE) // 2
VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
VIDEO_MIN, VIDEO_MAX = 0, 255.0  # min and max pixel values
FPS = 30

DPI = 240
FONTSIZE = 9

OUTPUT_DIR = Path("../../runs/rochefort-lab/vivit")
PLOT_DIR = Path("figures") / "animate_stimuli" / "grating_center"

plot.set_font()


def get_grating_center(
    output_dir: Path,
    mouse_id: str,
    neuron: int,
    data: dict[str, list],
):
    df = pd.read_parquet(
        output_dir
        / "gratings"
        / "center_surround"
        / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
    )
    df = df[df.surround_direction == -1]
    df = df.sort_values(by="response", ascending=False).iloc[0]
    data["video"].append(
        rearrange(df.stimulus, "(c t h w) -> c t h w", c=1, t=70, h=VIDEO_H, w=VIDEO_W)
    )
    data["response"].append(df.raw_response.astype(np.float32))
    data["stimulus_name"].append("grating_center")


def get_grating_video_surround(
    output_dir: Path,
    mouse_id: str,
    neuron: int,
    data: dict[str, list],
):
    df = pd.read_parquet(
        output_dir
        / "gratings"
        / "center_surround"
        / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
    )
    center_direction = (
        df[df.surround_direction == -1]
        .sort_values(by="response", ascending=False)
        .iloc[0]
        .center_direction
    )
    df = (
        df[(df.center_direction == center_direction) & (df.surround_direction != -1)]
        .sort_values(by="response", ascending=False)
        .iloc[0]
    )
    data["video"].append(
        rearrange(df.stimulus, "(c t h w) -> c t h w", c=1, t=70, h=VIDEO_H, w=VIDEO_W)
    )
    data["response"].append(df.raw_response.astype(np.float32))
    data["stimulus_name"].append("grating_center_grating_video_surround")


def get_natural_video_surround(
    output_dir: Path,
    mouse_id: str,
    neuron: int,
    data: dict[str, list],
):
    df = pd.read_parquet(
        output_dir
        / "gratings"
        / "grating_center_natural_surround"
        / "dynamic_center_dynamic_surround"
        / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
    )
    df = df.sort_values(by="response", ascending=False).iloc[0]
    data["video"].append(
        rearrange(df.stimulus, "(c t h w) -> c t h w", c=1, t=70, h=VIDEO_H, w=VIDEO_W)
    )
    data["response"].append(df.raw_response.astype(np.float32))
    data["stimulus_name"].append("grating_center_natural_video_surround")


def get_generated_video_surround(
    output_dir: Path,
    experiment_name: str,
    mouse_id: str,
    neuron: int,
    data: dict[str, list],
):
    ckpt = torch.load(
        output_dir
        / "generated"
        / "center_surround"
        / "grating_center"
        / "dynamic_center_dynamic_surround"
        / experiment_name
        / f"mouse{mouse_id}"
        / f"neuron{neuron:04d}"
        / "most_exciting"
        / "ckpt.pt",
        map_location="cpu",
    )
    start = -(BLANK_SIZE + PATTERN_SIZE + PAD)
    end = -(BLANK_SIZE - PAD)
    data["video"].append(ckpt["video"][:, start:end, :, :].detach().numpy())
    data["response"].append(ckpt["response"][start:end].detach().numpy())
    data["stimulus_name"].append("grating_center_generated_video_surround")


def animate_stimulus(
    video: np.ndarray | torch.Tensor,
    response: np.ndarray | torch.Tensor,
    filename: Path,
    mouse_id: str,
    neuron: int | None = None,
    max_value: float | None = None,
):
    assert len(video.shape) == 4
    t, h, w = video.shape[1], video.shape[2], video.shape[3]

    presentation_mask = np.concat([np.zeros(PAD), np.ones(PATTERN_SIZE), np.zeros(PAD)])

    f_w, f_h = 3.6, 2.3
    figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
    get_width = lambda height: height * (w / h) * (f_h / f_w)
    spatial_height = 0.68
    ax1 = figure.add_axes(rect=(0.12, 0.25, get_width(spatial_height), spatial_height))
    ax2 = figure.add_axes(rect=(0.12, 0.10, get_width(spatial_height), 0.08))

    x_ticks = np.array([0, t], dtype=int)
    min_value = 0
    if max_value is None:
        max_value = np.max(response)

    title = rf"N{neuron:03d} $\Delta F/F$"

    x = np.arange(t)

    imshow = ax1.imshow(
        np.random.rand(h, w),
        cmap="gray",
        aspect="equal",
        vmin=VIDEO_MIN,
        vmax=VIDEO_MAX,
    )
    pos = ax1.get_position()
    text = ax1.text(
        x=0,
        y=pos.y1 + 0.11,
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
        axis=ax2,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="Time (frame)",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
        label_pad=-7,
    )
    ax2.xaxis.set_minor_locator(MultipleLocator(10))
    y_ticks = np.array([min_value, max_value], dtype=np.float32)
    ax2.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax2,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, decimals=2),
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
    ax2.fill_between(
        x,
        y1=0,
        y2=max_value,
        where=presentation_mask,
        facecolor="#e0e0e0",
        edgecolor="none",
        zorder=-1,
    )
    plot.set_ticks_params(axis=ax2, minor_length=None)
    sns.despine(ax=ax2, trim=True)

    def animate(frame: int):
        artists = [imshow]
        imshow.set_data(video[0, frame, :, :])
        text.set_text(f"Frame: {frame :03d}")
        line.set_data(x[: frame + 1], response[: frame + 1])
        artists.append(line)
        return artists

    anim = FuncAnimation(
        figure, func=animate, frames=t, interval=int(1000 / FPS), blit=True
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
    plt.close(figure)


def main():
    df = pd.read_parquet("in_silico_single_neuron_grating_center.parquet")

    for mouse_id, output_dir_name, experiment_name in [
        (
            "L",
            "015_causal_viv1t_FOV2_finetune",
            "001_cutoff_natural_init_100_steps",
        ),
        (
            "M",
            "018_causal_viv1t_VIPcre233_FOV1_finetune",
            "001_cutoff_natural_init_200_steps",
        ),
        (
            "N",
            "025_causal_viv1t_VIPcre233_FOV2_finetune",
            "001_cutoff_natural_init_200_steps",
        ),
    ]:
        neurons = df[df.mouse == mouse_id].neuron.unique()
        for neuron in neurons:
            data = {"video": [], "response": [], "stimulus_name": []}
            output_dir = (
                OUTPUT_DIR
                / output_dir_name
                / "most_exciting_stimulus"
                / "single_neuron"
            )
            get_grating_center(
                output_dir=output_dir,
                mouse_id=mouse_id,
                neuron=neuron,
                data=data,
            )
            get_grating_video_surround(
                output_dir=output_dir,
                mouse_id=mouse_id,
                neuron=neuron,
                data=data,
            )
            get_natural_video_surround(
                output_dir=output_dir,
                mouse_id=mouse_id,
                neuron=neuron,
                data=data,
            )
            get_generated_video_surround(
                output_dir=output_dir,
                experiment_name=experiment_name,
                mouse_id=mouse_id,
                neuron=neuron,
                data=data,
            )
            max_value = np.max(data["response"])
            for i in tqdm(range(4), desc=f"mouse {mouse_id} neuron {neuron:03d}"):
                animate_stimulus(
                    video=data["video"][i],
                    response=data["response"][i],
                    filename=PLOT_DIR
                    / f'mouse{mouse_id}_neuron{neuron:03d}_{data["stimulus_name"][i]}.mp4',
                    mouse_id=mouse_id,
                    neuron=neuron,
                    max_value=max_value,
                )


if __name__ == "__main__":
    main()
