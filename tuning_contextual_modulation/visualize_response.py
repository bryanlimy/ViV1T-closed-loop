from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from scipy.stats import sem
from tqdm import tqdm
from visualize_stimuli import get_stim_name

from viv1t.utils import plot
from viv1t.utils import stimulus

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

TICK_LENGTH, TICK_PAD, TICK_LINEWIDTH = 3, 2, 1.2

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLANK_SIZE, PATTERN_SIZE = 15, 30
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2
FPS = 30

PLOT_DIR = Path("figures") / "response"

CONTRAST_TYPES = Literal["high_contrast", "low_contrast"]


def load_traces(
    output_dir: Path, mouse_id: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    save_dir = output_dir / "contextual_modulation"
    responses = np.load(save_dir / f"mouse{mouse_id}.npz", allow_pickle=False)["data"]
    parameters = np.load(save_dir / "parameters.npy", allow_pickle=False)

    responses = rearrange(
        responses, "neuron block pattern frame -> neuron (block pattern) frame"
    )
    parameters = rearrange(parameters, "block pattern param -> (block pattern) param")

    # select L2/3 neurons and their size-tune preferences
    df = pd.read_parquet(output_dir / "size_tuning_preference.parquet")
    neurons = df[
        (df.mouse == mouse_id)
        & (df.classic_tuned == True)
        & (df.depth >= 200)
        & (df.depth <= 300)
    ].neuron.values
    responses = responses[neurons]

    contrasts = np.unique(parameters[:, 0])
    # select stimulus size 20 and group responses into the 4 stimulus types
    stimulus_size = 20
    low_responses, high_responses = {}, {}
    for stimulus_type in np.unique(parameters[:, -1]):
        idx = np.where(
            (parameters[:, 0] == contrasts[0])
            & (parameters[:, 1] == stimulus_size)
            & (parameters[:, 3] == stimulus_type)
        )[0]
        low_responses[int(stimulus_type)] = responses[:, idx, :]
        idx = np.where(
            (parameters[:, 0] == contrasts[1])
            & (parameters[:, 1] == stimulus_size)
            & (parameters[:, 3] == stimulus_type)
        )[0]
        high_responses[int(stimulus_type)] = responses[:, idx, :]
    return low_responses, high_responses, neurons


def plot_stimulus(contrast_type: str, plot_dir: Path):
    for stimulus_type in [0, 1, 2, 3]:
        figure, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(0.5, 0.5),
            gridspec_kw={
                "wspace": 0.2,
                "hspace": 0.1,
                "top": 1,
                "bottom": 0,
                "left": 0,
                "right": 1,
            },
            dpi=DPI,
        )

        fps = 30
        stimulus_kw = {
            "cpd": 0.04,
            "cpf": 2 / fps,
            "height": 36,
            "width": 64,
            "num_frames": 1,
            "contrast": 1 if contrast_type == "high_contrast" else 0.05,
            "fps": fps,
            "to_tensor": False,
        }
        circular_mask = stimulus.create_circular_mask(stimulus_size=60, num_frames=1)
        center = stimulus.create_full_field_grating(direction=90, **stimulus_kw)
        match stimulus_type:
            case 0:
                surround = GREY_COLOR
                filename = plot_dir / f"{contrast_type}_center.{FORMAT}"
            case 1:
                surround = stimulus.create_full_field_grating(
                    direction=90, **stimulus_kw
                )
                filename = plot_dir / f"{contrast_type}_iso.{FORMAT}"
            case 2:
                surround = stimulus.create_full_field_grating(
                    direction=0, **stimulus_kw
                )
                filename = plot_dir / f"{contrast_type}_cross.{FORMAT}"
            case 3:
                surround = stimulus.create_full_field_grating(
                    direction=90, phase=180, **stimulus_kw
                )
                filename = plot_dir / f"{contrast_type}_shift.{FORMAT}"
            case _:
                raise RuntimeError(f"Unknown stimulus type: {stimulus_kw}")
        frame = np.where(circular_mask, center, surround)
        ax.imshow(frame[0, 0, :, 14:50], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        plot.set_ticks_params(axis=ax)
        plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def plot_response(responses: dict[int, np.ndarray], filename: Path):
    nrows, ncols = 1, len(responses)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=((1 / 3) * PAPER_WIDTH, 0.6),
        gridspec_kw={"wspace": 0.4, "hspace": 0},
        dpi=DPI,
    )

    x_ticks = np.arange(responses[0].shape[-1])

    min_value, max_value = 0, 0
    for i, response in responses.items():
        mean = np.mean(response, axis=0)
        se = sem(response, axis=0)
        max_value = max(max_value, np.max(mean + se))
        min_value = min(min_value, np.min(mean - se))
        axes[i].plot(
            x_ticks,
            mean,
            color="black",
            linewidth=1.8,
            alpha=0.8,
            clip_on=False,
            zorder=1,
        )
        axes[i].fill_between(
            x_ticks,
            y1=mean - se,
            y2=mean + se,
            facecolor="black",
            edgecolor="none",
            linewidth=2,
            alpha=0.3,
            zorder=1,
            clip_on=False,
        )
        axes[i].axvspan(
            xmin=BLANK_SIZE,
            xmax=BLANK_SIZE + PATTERN_SIZE,
            facecolor="#e0e0e0",
            edgecolor="none",
            zorder=-1,
        )
        name = get_stim_name(i)
        axes[i].set_xlabel(name, fontsize=LABEL_FONTSIZE, labelpad=2)

    min_value = 0
    for i in range(ncols):
        axes[i].set_xlim(x_ticks[0] - 1, x_ticks[-1] + 1)
        axes[i].set_ylim(min_value, max_value)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        sns.despine(ax=axes[i], left=True, bottom=True)

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.98, top=0.95)

    # x_offset = -55
    x_offset = -40
    # response scale bar
    y = 0.5 * max_value
    axes[0].plot(
        [x_offset, x_offset],
        [min_value, y],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    axes[0].text(
        x=x_offset,
        y=min_value,
        s=f"a.u. ΔF/F",
        fontsize=TICK_FONTSIZE,
        rotation=90,
        va="bottom",
        ha="right",
        transform=axes[0].transData,
    )

    # timescale bar
    axes[0].plot(
        [x_offset, x_offset + FPS],
        [min_value, min_value],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    axes[0].text(
        x=x_offset + FPS / 2,
        y=min_value - (0.05 * max_value),
        s="1s",
        fontsize=TICK_FONTSIZE,
        va="top",
        ha="center",
    )

    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def main():
    models = {
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    mouse_id = "B"
    neuron = 930
    for model_name, output_dir in models.items():
        print(f"Plot contextual modulation response in {output_dir}.")
        low_responses, high_responses, neurons = load_traces(
            mouse_id=mouse_id, output_dir=output_dir
        )
        for neuron in tqdm(neurons):
            n = np.where(neurons == neuron)[0][0]
            low_response = {k: v[n] for k, v in low_responses.items()}
            high_response = {k: v[n] for k, v in high_responses.items()}
            # remove shift
            del low_response[3], high_response[3]
            plot_response(
                responses=low_response,
                filename=PLOT_DIR
                / model_name
                / f"low_contrast_mouse{mouse_id}_neuron{neuron:04d}.{FORMAT}",
            )
            plot_response(
                responses=high_response,
                filename=PLOT_DIR
                / model_name
                / f"high_contrast_mouse{mouse_id}_neuron{neuron:04d}.{FORMAT}",
            )

    plot_stimulus(contrast_type="high_contrast", plot_dir=PLOT_DIR / "stimulus")
    plot_stimulus(contrast_type="low_contrast", plot_dir=PLOT_DIR / "stimulus")


if __name__ == "__main__":
    main()
