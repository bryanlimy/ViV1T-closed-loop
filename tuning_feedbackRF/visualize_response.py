from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.stats import sem
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30


PLOT_DIR = Path("figures") / "response"


def load_data(output_dir: Path, mouse_id: str):
    save_dir = output_dir / "feedbackRF"
    responses = np.load(save_dir / f"mouse{mouse_id}.npz", allow_pickle=False)["data"]
    responses = rearrange(responses, "N block pattern T -> N (block pattern) T")

    parameters = np.load(save_dir / "parameters.npy", allow_pickle=False)
    parameters = rearrange(parameters, "block pattern param -> (block pattern) param")

    # select L2/3 neurons and their size-tune preferences
    try:
        df = pd.read_parquet(output_dir / "size_tuning_preference.parquet")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find size_tuning_preference.parquet in {output_dir}. "
            f"Please run tuning_feedback/estimate_feedbackRF.py first."
        )
    neurons = df[
        (df["mouse"] == mouse_id)
        & (df["classic_tuned"] == True)
        & (df["inverse_tuned"] == True)
        & (df["depth"] >= 200)
        & (df["depth"] <= 300)
    ].neuron.values
    responses = responses[neurons]

    stimulus_sizes = np.unique(parameters[:, 0])
    directions = np.unique(parameters[:, 1])
    stimulus_types = np.unique(parameters[:, 2])

    num_repeats = np.unique(parameters, axis=0, return_counts=True)[1][0]

    # group responses based on unique pattern
    responses_ = np.zeros(
        (
            responses.shape[0],
            len(stimulus_sizes),
            len(stimulus_types),
            len(directions) * num_repeats,
            BLOCK_SIZE,
        ),
        dtype=np.float32,
    )
    for i, stimulus_size in enumerate(stimulus_sizes):
        for j, stimulus_type in enumerate(stimulus_types):
            # find patterns with the same stimulus size and type
            idx = np.where(
                (parameters[:, 0] == stimulus_size)
                & (parameters[:, 2] == stimulus_type)
            )[0]
            responses_[:, i, j, :] = responses[:, idx, :]
    responses_ = rearrange(responses_, "N d1 d2 R T -> N d1 d2 T R")
    return responses_, neurons, stimulus_sizes, stimulus_types


def plot_size_tuning_response(
    responses: np.ndarray,
    stimulus_sizes: np.ndarray,
    stimulus_types: np.ndarray,
    filename: Path,
):
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(stimulus_sizes),
        figsize=(0.45 * PAPER_WIDTH, 1.0),
        gridspec_kw={"wspace": 0.3, "hspace": 0},
        dpi=DPI,
    )

    x_ticks = np.arange(responses.shape[2])

    min_value, max_value = 0, -np.inf
    for i, stimulus_size in enumerate(stimulus_sizes):
        for j, stimulus_type in enumerate(stimulus_types):
            color = "black" if stimulus_type == 0 else "red"
            # compute average response over repeats
            response = np.mean(responses[i, j, :, :], axis=1)
            se = sem(responses[i, j, :, :], axis=1)
            max_value = max(max_value, np.max(response + se))
            axes[i].plot(
                x_ticks,
                response,
                color=color,
                linewidth=1.2,
                alpha=0.8,
                clip_on=False,
                zorder=1,
            )
            axes[i].fill_between(
                x_ticks,
                y1=response - se,
                y2=response + se,
                facecolor=color,
                edgecolor="none",
                alpha=0.3,
                zorder=1,
                clip_on=False,
            )
            axes[i].axvspan(
                xmin=BLANK_SIZE,
                xmax=BLANK_SIZE + PATTERN_SIZE,
                facecolor="#e0e0e0",
                # facecolor="black",
                edgecolor="none",
                zorder=-1,
                # alpha=0.15,
            )

    legend = axes[-1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                label="Classical",
                linestyle="-",
                linewidth=1.5,
            ),
            Line2D(
                [0],
                [0],
                color="red",
                label="Inverse",
                linestyle="-",
                linewidth=1.5,
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        facecolor="white",
        framealpha=1.0,
        handlelength=0.5,
        labelspacing=0.0,
        handletextpad=0.35,
        borderpad=0.2,
        borderaxespad=0,
    )

    min_value = 0
    max_value = 1.2 * max_value
    y_ticks = np.linspace(min_value, max_value, 2)
    for i, ax in enumerate(axes):
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_ylim(y_ticks[0], y_ticks[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xlabel(
            int(stimulus_sizes[i]),
            fontsize=LABEL_FONTSIZE,
            labelpad=1,
        )

    # xlabel
    figure.text(
        x=0.50,
        y=0.04,
        s="Stimulus size (°)",
        fontsize=LABEL_FONTSIZE,
        va="center",
        ha="center",
    )
    figure.subplots_adjust(left=0.12, bottom=0.22, right=0.99, top=0.95)

    x_offset = -55
    # response scale bar
    axes[0].plot(
        [x_offset, x_offset],
        [0, 0.5 * max_value],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    figure.text(
        x=x_offset - 6,
        y=0,
        s=f"a.u. ΔF/F",
        fontsize=LABEL_FONTSIZE,
        rotation=90,
        va="bottom",
        ha="right",
        transform=axes[0].transData,
    )

    # timescale bar
    axes[0].plot(
        [x_offset, x_offset + FPS],
        [0, 0],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    figure.text(
        x=x_offset + FPS / 2,
        y=-0.04 * max_value,
        s="1s",
        fontsize=LABEL_FONTSIZE,
        va="top",
        ha="center",
        transform=axes[0].transData,
    )

    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def process_model(model_name: str, output_dir: Path):
    print(f"Plot feedforward and feedback response from {output_dir}.")
    mouse_ids = data.SENSORIUM_OLD
    for mouse_id in mouse_ids:
        responses, neurons, stimulus_sizes, stimulus_types = load_data(
            output_dir=output_dir, mouse_id=mouse_id
        )
        # neuron = 234
        # n = neurons.tolist().index(neuron)
        # plot_size_tuning_response(
        #     responses[n],
        #     stimulus_sizes=stimulus_sizes,
        #     stimulus_types=stimulus_types,
        #     filename=PLOT_DIR
        #     / model_name
        #     / f"mouse{mouse_id}"
        #     / f"mouse{mouse_id}_neuron{neuron:04d}_size_tuning_response.{FORMAT}",
        # )
        plot_size_tuning_response(
            responses=np.mean(responses, axis=0),
            stimulus_sizes=stimulus_sizes,
            stimulus_types=stimulus_types,
            filename=PLOT_DIR
            / model_name
            / f"mouse{mouse_id}_size_tuning_response.{FORMAT}",
        )
        if model_name == "ViV1T":
            for n, neuron in enumerate(tqdm(neurons, desc=f"mouse {mouse_id}")):
                plot_size_tuning_response(
                    responses[n],
                    stimulus_sizes=stimulus_sizes,
                    stimulus_types=stimulus_types,
                    filename=PLOT_DIR
                    / model_name
                    / f"mouse{mouse_id}"
                    / f"mouse{mouse_id}_neuron{neuron:04d}_size_tuning_response.{FORMAT}",
                )


def main():
    models = {
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    for model_name, output_dir in models.items():
        process_model(model_name=model_name, output_dir=output_dir)
    print(f"Saved plots to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
