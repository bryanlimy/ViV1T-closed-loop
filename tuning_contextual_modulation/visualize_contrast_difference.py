from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from estimate_contextual_modulation import load_data
from matplotlib.ticker import MultipleLocator
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from tqdm import tqdm
from visualize_stimuli import get_stim_name

from viv1t import data
from viv1t.utils import plot

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

TICK_LENGTH, TICK_PAD, TICK_LINEWIDTH = 3, 2, 1.2

DPI = 400
FORMAT = "jpg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

PLOT_DIR = Path("figures") / "contrast_difference"

CONTRAST_TYPES = Literal["high_contrast", "low_contrast"]


def load_trace(
    output_dir: Path, mouse_id: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    save_dir = output_dir / "contextual_modulation"
    responses = np.load(save_dir / f"mouse{mouse_id}.npz", allow_pickle=False)["data"]
    parameters = np.load(save_dir / "parameters.npy", allow_pickle=False)

    responses = rearrange(responses, "N block pattern T -> N (block pattern) T")
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
    for stim_type in np.unique(parameters[:, -1]):
        idx = np.where(
            (parameters[:, 0] == contrasts[0])
            & (parameters[:, 1] == stimulus_size)
            & (parameters[:, 3] == stim_type)
        )[0]
        low_responses[int(stim_type)] = responses[:, idx, :]
        idx = np.where(
            (parameters[:, 0] == contrasts[1])
            & (parameters[:, 1] == stimulus_size)
            & (parameters[:, 3] == stim_type)
        )[0]
        high_responses[int(stim_type)] = responses[:, idx, :]
    return low_responses, high_responses


def plot_combined_box_plot(df: pd.DataFrame, filename: Path):
    high_center = df[(df.contrast_type == "high_contrast")]["center"].values

    high_iso = df[(df.contrast_type == "high_contrast")]["iso"].values
    high_cross = df[(df.contrast_type == "high_contrast")]["cross"].values
    high_shift = df[(df.contrast_type == "high_contrast")]["shift"].values

    low_center = df[(df.contrast_type == "low_contrast")]["center"].values

    low_iso = df[(df.contrast_type == "low_contrast")]["iso"].values
    low_cross = df[(df.contrast_type == "low_contrast")]["cross"].values
    low_shift = df[(df.contrast_type == "low_contrast")]["shift"].values

    high_iso /= high_center
    high_cross /= high_center
    high_shift /= high_center

    low_iso /= low_center
    low_cross /= low_center
    low_shift /= low_center

    num_neurons = len(high_center)

    responses = [high_iso, high_cross, high_shift, low_iso, low_cross, low_shift]

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 2.2),
        dpi=DPI,
    )

    box_width, box_pad = 0.14, 0.05
    linewidth = 1.2

    x_ticks = np.array([1, 2], dtype=np.float32)
    positions = [
        x_ticks[0] - box_width - box_pad,
        x_ticks[0],
        x_ticks[0] + box_width + box_pad,
        x_ticks[1] - box_width - box_pad,
        x_ticks[1],
        x_ticks[1] + box_width + box_pad,
    ]
    box_kw = {
        "notch": False,
        "vert": True,
        "widths": box_width,
        "showfliers": False,
        "showmeans": True,
        "boxprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
        },
        "flierprops": {
            "marker": "o",
            "markersize": 2,
            "alpha": 0.5,
            "clip_on": False,
            "zorder": 10,
        },
        "capprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
        },
        "whiskerprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
        },
        "meanprops": {
            "markersize": 4,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 0.75,
            "clip_on": False,
            "zorder": 20,
        },
        "medianprops": {
            "linewidth": 1.2 * linewidth,
            "color": "royalblue",
            "solid_capstyle": "projecting",
            "clip_on": False,
            "zorder": 20,
        },
    }
    bp = ax.boxplot(responses, positions=positions, **box_kw)
    max_value = np.ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    # max_value = np.ceil(max_value / 2) * 2

    linewidth = 1
    alpha = 0.5
    # # plot neurons that flipped more than 20% between iso vs cross
    # high_neurons = np.where((100 * (high_cross - high_iso) / high_iso) >= 20)[0]
    # low_neurons = np.where((100 * (low_iso - low_cross) / low_cross) >= 20)[0]
    # flip_neurons = np.intersect1d(high_neurons, low_neurons)
    # scatter_kw = {
    #     "s": 15,
    #     "marker": ".",
    #     "alpha": alpha,
    #     "zorder": 5,
    #     "facecolors": "none",
    #     "clip_on": False,
    # }
    # for position, response in [
    #     (positions[0], high_iso),
    #     (positions[1], high_cross),
    #     (positions[3], low_iso),
    #     (positions[4], low_cross),
    # ]:
    #     ax.scatter(
    #         [position] * len(flip_neurons),
    #         response[flip_neurons],
    #         edgecolors="limegreen",
    #         **scatter_kw,
    #     )
    #     del position, response
    # for neuron in flip_neurons:
    #     ax.plot(
    #         [positions[0], positions[1]],
    #         [high_iso[neuron], high_cross[neuron]],
    #         linewidth=linewidth,
    #         color="limegreen",
    #         zorder=5,
    #         alpha=alpha,
    #     )
    #     ax.plot(
    #         [positions[3], positions[4]],
    #         [low_iso[neuron], low_cross[neuron]],
    #         linewidth=linewidth,
    #         color="limegreen",
    #         zorder=5,
    #         alpha=alpha,
    #     )
    # ax.text(
    #     x=x_ticks[-1] + 0.3,
    #     y=0.94 * max_value,
    #     s=r"$N_{neurons}$=" + str(len(flip_neurons)),
    #     ha="right",
    #     va="top",
    #     color="limegreen",
    #     fontsize=TICK_FONTSIZE,
    # )

    # plot neurons that flipped more than 20% between iso vs shift
    # high_neurons = np.where((100 * (high_shift - high_iso) / high_iso) >= 20)[0]
    # low_neurons = np.where((100 * (low_iso - low_shift) / low_shift) >= 20)[0]
    # flip_neurons_shift = np.intersect1d(high_neurons, low_neurons)
    # flip_neurons = np.intersect1d(flip_neurons_shift, flip_neurons)
    # for position, response in [
    #     (positions[0], high_iso),
    #     (positions[2], high_shift),
    #     (positions[3], low_iso),
    #     (positions[5], low_shift),
    # ]:
    #     ax.scatter(
    #         [position] * len(flip_neurons),
    #         response[flip_neurons],
    #         color="orangered",
    #         **scatter_kw,
    #     )
    #     del position, response
    # for neuron in flip_neurons:
    #     ax.plot(
    #         [positions[0], positions[2]],
    #         [high_iso[neuron], high_shift[neuron]],
    #         linewidth=linewidth,
    #         color="orangered",
    #         zorder=5,
    #         alpha=alpha,
    #     )
    #     ax.plot(
    #         [positions[3], positions[5]],
    #         [low_iso[neuron], low_shift[neuron]],
    #         linewidth=linewidth,
    #         color="orangered",
    #         zorder=5,
    #         alpha=alpha,
    #     )
    # ax.text(
    #     x=x_ticks[-1] + 0.3,
    #     y=0.88 * max_value,
    #     s=r"$N_{neurons}$=" + str(len(flip_neurons)),
    #     ha="right",
    #     va="top",
    #     color="orangered",
    #     fontsize=TICK_FONTSIZE,
    # )

    scatter_kw = {
        "s": 10,
        "marker": ".",
        "alpha": 0.4,
        "zorder": 0,
        "facecolors": "none",
        "clip_on": False,
        "edgecolors": "black",
        "linewidth": 0.75,
    }
    for i, (position, response) in enumerate(zip(positions, responses)):
        neurons = np.arange(len(response))
        # neurons = np.setdiff1d(neurons, flip_neurons)
        response = response[neurons]
        neurons = np.arange(len(response))
        outliers = np.where(response >= max_value)[0]
        inliers = np.setdiff1d(neurons, outliers)
        x = rng.normal(position, 0.02, size=len(response))
        ax.scatter(
            x[inliers],
            response[inliers],
            **scatter_kw,
        )
        # plot outlier neurons
        if outliers.size > 0:
            ax.scatter(
                x[outliers],
                np.full(outliers.shape, fill_value=max_value),
                **scatter_kw,
            )

    xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.5]

    # Plot y=1 dashed line
    ax.axhline(
        y=1,
        color="black",
        alpha=0.5,
        linestyle="dotted",
        dashes=(1, 1),
        linewidth=1,
        zorder=-1,
    )

    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=["High contrast", "Low contrast"],
        tick_fontsize=TICK_FONTSIZE,
        linespacing=0.85,
    )
    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[y_ticks[0], rf"$\geq${y_ticks[-1]}"],
        label=r"Sum norm. $\Delta$F/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        rotation=90,
    )
    if y_ticks[-1] < 100:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, pad=0, minor_length=3)
    ax.tick_params(axis="x", which="major", length=0, pad=6)
    sns.despine(ax=ax, bottom=True, trim=True)

    ax.text(
        x=x_ticks[-1] + 0.3,
        y=max_value,
        s=r"$N_{neurons}$=" + str(num_neurons),
        ha="right",
        va="top",
        fontsize=TICK_FONTSIZE,
    )

    ax.text(
        x=positions[0],
        y=max_value,
        s="iso",
        va="top",
        ha="center",
        rotation=90,
        fontsize=TICK_FONTSIZE,
    )
    ax.text(
        x=positions[1],
        y=max_value,
        s="cross",
        va="top",
        ha="center",
        rotation=90,
        fontsize=TICK_FONTSIZE,
    )
    ax.text(
        x=positions[2],
        y=max_value,
        s="shift",
        va="top",
        ha="center",
        rotation=90,
        fontsize=TICK_FONTSIZE,
    )

    ax.set_title("Prediction (Sensorium 2023)", fontsize=TICK_FONTSIZE, pad=0)

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def plot_combined_scatter(df: pd.DataFrame, filename: Path):
    high_center = df[df.contrast_type == "high_contrast"].center.values
    high_iso = df[df.contrast_type == "high_contrast"].iso.values
    high_cross = df[df.contrast_type == "high_contrast"].cross.values
    low_iso = df[df.contrast_type == "low_contrast"].iso.values
    low_cross = df[df.contrast_type == "low_contrast"].cross.values

    # normalize all response by response to high contrast center
    high_iso = high_iso / high_center
    high_cross = high_cross / high_center
    low_iso = low_iso / high_center
    low_cross = low_cross / high_center

    # compute p-value
    high_p_value = ttest_ind(high_cross, high_iso).pvalue
    if high_p_value <= 0.001:
        high_p_value = r"$p<10^3$"
    elif high_p_value <= 0.01:
        high_p_value = r"$p<10^2$"
    elif high_p_value <= 0.05:
        high_p_value = r"p<0.05"
    else:
        high_p_value = "p>0.05"

    low_p_value = ttest_ind(low_cross, low_iso).pvalue
    if low_p_value <= 0.001:
        low_p_value = r"$p<10^3$"
    elif low_p_value <= 0.01:
        low_p_value = r"$p<10^2$"
    elif low_p_value <= 0.05:
        low_p_value = r"p<0.05"
    else:
        low_p_value = "p>0.05"

    figure_width = 0.31 * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, figure_width),
        dpi=DPI,
    )

    scatter_kw = {
        "alpha": 0.2,
        "edgecolors": "none",
        "zorder": 20,
        "clip_on": False,
    }

    # clip iso and cross responses to max_value
    min_value, max_value = 0, 5
    ax.scatter(
        x=np.clip(high_cross, min_value, max_value),
        y=np.clip(high_iso, min_value, max_value),
        color="orangered",
        marker="s",
        s=3,
        label=f"High contrast ({high_p_value})",
        **scatter_kw,
    )
    ax.scatter(
        x=np.clip(low_cross, min_value, max_value),
        y=np.clip(low_iso, min_value, max_value),
        color="dodgerblue",
        marker="v",
        s=4,
        label=f"Low contrast ({low_p_value})",
        **scatter_kw,
    )
    # plot identity line
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
        clip_on=False,
    )

    ticks = np.array([min_value, max_value], dtype=int)
    tick_labels = [min_value, rf"$\geq${max_value}"]
    ax.set_xlim(ticks[0], ticks[-1])
    plot.set_xticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label="Cross [ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_ylim(ticks[0], ticks[-1])
    plot.set_yticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label="Iso [ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-8,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.05 * max_value, 0.99 * max_value),
        bbox_transform=ax.transData,
        ncols=1,
        fontsize=TICK_FONTSIZE,
        title_fontsize=TICK_FONTSIZE,
        frameon=False,
        alignment="left",
        handletextpad=0.2,
        handlelength=0.7,
        labelspacing=0.05,
        markerscale=1.4,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    for text in legend.texts:
        text.set_y(-1.2)

    plot.set_ticks_params(
        ax, length=TICK_LENGTH, pad=TICK_PAD, linewidth=TICK_LINEWIDTH
    )
    sns.despine(ax=ax)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def find_flip_neurons(
    df: pd.DataFrame, threshold: float = 0.2
) -> dict[str, np.ndarray]:
    """
    Find neurons that has cross response greater than iso response
    in high contrast, and cross response weaker than iso response in low contrast.
    """
    num_neurons = np.zeros(len(df.mouse.unique()), dtype=int)
    num_flips = np.zeros_like(num_neurons)
    flip_neurons = {}
    for i, mouse_id in enumerate(sorted(df.mouse.unique())):
        mouse = df[df.mouse == mouse_id]
        high_center = mouse[mouse.contrast_type == "high_contrast"]["center"].values
        high_iso = mouse[mouse.contrast_type == "high_contrast"]["iso"].values
        high_cross = mouse[mouse.contrast_type == "high_contrast"]["cross"].values
        low_iso = mouse[mouse.contrast_type == "low_contrast"]["iso"].values
        low_cross = mouse[mouse.contrast_type == "low_contrast"]["cross"].values
        neurons = mouse[mouse.contrast_type == "high_contrast"].neuron.values
        # normalize all response by response to high contrast center
        high_iso = high_iso / high_center
        high_cross = high_cross / high_center
        low_iso = low_iso / high_center
        low_cross = low_cross / high_center
        # check if cross response is 20% greater than iso response in high contrast
        condition1 = ((high_cross - high_iso) / high_iso) >= threshold
        # check if iso response is 20% greater than cross response in low contrast
        condition2 = ((low_iso - low_cross) / low_cross) >= threshold
        indexes = np.where(condition1 & condition2)[0]
        flip_neuron = neurons[indexes]
        print(
            f"{len(flip_neuron)} neurons out of {len(neurons)} neurons "
            f"({100 * len(flip_neuron) / len(neurons):.2f}%) in "
            f"mouse {mouse_id} are flip neurons."
        )
        num_neurons[i] = len(neurons)
        num_flips[i] = len(flip_neuron)
        flip_neurons[mouse_id] = flip_neuron
    percentages = 100 * num_flips / num_neurons
    print(
        f"Percentage of flipped neurons:\n"
        f"\tmean +/- sem: {np.mean(percentages):.03f} +/- {sem(percentages):.3f}%\n"
        f"\tmedian: {np.median(percentages):.03f}%\n"
    )
    return flip_neurons


def plot_flip_neurons_scatter(
    df: pd.DataFrame, flip_neurons: dict[str, np.ndarray], filename: Path
):
    figure_width = 0.31 * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, figure_width),
        dpi=DPI,
    )

    scatter_kw = {
        "s": 6,
        # "alpha": 0.4,
        "edgecolors": "none",
        # "zorder": 20,
        "clip_on": False,
    }

    # clip iso and cross responses to max_value
    min_value, max_value = 0, 5

    for i, mouse_id in enumerate(sorted(df.mouse.unique())):
        mouse = df[df.mouse == mouse_id]
        high_center = mouse[mouse.contrast_type == "high_contrast"]["center"].values
        high_iso = mouse[mouse.contrast_type == "high_contrast"]["iso"].values
        high_cross = mouse[mouse.contrast_type == "high_contrast"]["cross"].values
        low_iso = mouse[mouse.contrast_type == "low_contrast"]["iso"].values
        low_cross = mouse[mouse.contrast_type == "low_contrast"]["cross"].values
        # normalize all response by response to high contrast center
        high_iso = high_iso / high_center
        high_cross = high_cross / high_center
        low_iso = low_iso / high_center
        low_cross = low_cross / high_center

        neurons = mouse[mouse.contrast_type == "high_contrast"].neuron.values
        flip_neuron = flip_neurons[mouse_id]

        # indexes of the flip neurons with respect to the subpopulation
        flip_idx = np.where(np.isin(neurons, flip_neuron))[0]
        # indexes of the non-flip normal neurons
        normal_idx = np.setdiff1d(np.arange(len(neurons)), flip_idx)

        # plot non-flip neurons
        ax.scatter(
            x=np.clip(high_cross[normal_idx], min_value, max_value),
            y=np.clip(high_iso[normal_idx], min_value, max_value),
            color="gray",
            marker="s",
            zorder=5,
            alpha=0.1,
            **scatter_kw,
        )
        ax.scatter(
            x=np.clip(low_cross[normal_idx], min_value, max_value),
            y=np.clip(low_iso[normal_idx], min_value, max_value),
            color="gray",
            marker="v",
            zorder=5,
            alpha=0.1,
            **scatter_kw,
        )

        # plot flip neurons
        high_iso = np.clip(high_iso[flip_idx], min_value, max_value)
        high_cross = np.clip(high_cross[flip_idx], min_value, max_value)
        ax.scatter(
            x=high_cross,
            y=high_iso,
            color="orangered",
            label="High contrast" if i == 0 else "",
            marker="s",
            zorder=20,
            alpha=0.4,
            **scatter_kw,
        )
        low_iso = np.clip(low_iso[flip_idx], min_value, max_value)
        low_cross = np.clip(low_cross[flip_idx], min_value, max_value)
        ax.scatter(
            x=low_cross,
            y=low_iso,
            color="dodgerblue",
            label="Low contrast" if i == 0 else "",
            marker="v",
            zorder=20,
            alpha=0.4,
            **scatter_kw,
        )
        # connect dots
        for neuron in range(high_iso.shape[0]):
            ax.plot(
                [high_cross[neuron], low_cross[neuron]],
                [high_iso[neuron], low_iso[neuron]],
                linewidth=0.8,
                color="black",
                alpha=0.2,
                zorder=10,
            )

    # plot identity line
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="black",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
        clip_on=False,
    )
    # ax.plot(
    #     [0, 1],
    #     [1, 1],
    #     color="black",
    #     linestyle="--",
    #     linewidth=0.8,
    #     zorder=10,
    #     clip_on=False,
    # )
    # ax.plot(
    #     [1, 1],
    #     [0, 1],
    #     color="black",
    #     linestyle="--",
    #     linewidth=0.8,
    #     zorder=10,
    #     clip_on=False,
    # )

    ticks = np.array([min_value, max_value], dtype=int)
    tick_labels = [min_value, rf"$\geq${max_value}"]
    ax.set_xlim(ticks[0], ticks[-1])
    plot.set_xticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label="Cross [ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-4,
    )
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_ylim(ticks[0], ticks[-1])
    plot.set_yticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label="Iso [ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    num_neurons = len(df[df.contrast_type == "high_contrast"])
    num_flips = sum([len(v) for v in flip_neurons.values()])

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.05 * max_value, 0.99 * max_value),
        bbox_transform=ax.transData,
        ncols=1,
        title=f"N={num_neurons} (flip: {100* num_flips / num_neurons:.2f}%)",
        fontsize=TICK_FONTSIZE,
        title_fontsize=TICK_FONTSIZE,
        frameon=False,
        alignment="left",
        handletextpad=0.2,
        handlelength=0.7,
        labelspacing=0.05,
        markerscale=1.4,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    for text in legend.texts:
        text.set_y(-0.5)

    plot.set_ticks_params(
        ax, length=TICK_LENGTH, pad=TICK_PAD, linewidth=TICK_LINEWIDTH
    )
    sns.despine(ax=ax)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_flip_neuron_trace(
    high_response: dict[int, np.ndarray],
    low_response: dict[int, np.ndarray],
    mouse_id: str,
    neuron: int,
    filename: Path,
):
    # normalize all response by maximum response to high contrast center
    max_center = np.max(np.mean(high_response[0], axis=0))
    high_response = {k: v / max_center for k, v in high_response.items()}
    low_response = {k: v / max_center for k, v in low_response.items()}

    nrows, ncols = 2, 4
    figure_width = 0.5 * PAPER_WIDTH
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figure_width, (3 / 5) * figure_width),
        gridspec_kw={"wspace": 0.0, "hspace": 0.15},
        dpi=DPI,
    )

    x_ticks = np.arange(high_response[0].shape[-1])

    max_value = 0
    for i, response in enumerate((high_response, low_response)):
        for j, stim_type in enumerate(high_response.keys()):
            mean = np.mean(response[stim_type], axis=0)
            se = sem(response[stim_type], axis=0)
            max_value = max(max_value, np.max(mean + se))
            axes[i, j].plot(
                x_ticks,
                mean,
                color="black",
                linewidth=2,
                alpha=1,
                clip_on=False,
                zorder=1,
            )
            axes[i, j].fill_between(
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
            axes[i, j].axvspan(
                xmin=BLANK_SIZE,
                xmax=BLANK_SIZE + PATTERN_SIZE,
                facecolor="#e0e0e0",
                edgecolor="none",
                zorder=-1,
            )
            name = get_stim_name(stim_type)
            if i == 1:
                axes[i, j].set_xlabel(name, fontsize=LABEL_FONTSIZE, labelpad=2)

    min_value = 0
    for i in range(nrows):
        for j in range(ncols):
            axes[i, j].set_xlim(x_ticks[0] - 15, x_ticks[-1] + 15)
            axes[i, j].set_ylim(min_value, 1.2 * max_value)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            sns.despine(ax=axes[i, j], left=True, bottom=True)

    x_offset = -50

    # response scale bar
    response_scale = 0.5 * max_value
    axes[1, 0].plot(
        [x_offset, x_offset],
        [0, response_scale],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    axes[1, 0].text(
        x=x_offset - 4,
        y=-0.02 * max_value,
        s=f"{response_scale:.1f}ΔF/F",
        fontsize=LABEL_FONTSIZE,
        rotation=90,
        va="bottom",
        ha="right",
        transform=axes[1, 0].transData,
    )
    # timescale bar
    axes[1, 0].plot(
        [x_offset, x_offset + FPS],
        [0, 0],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    axes[1, 0].text(
        x=x_offset + FPS / 2,
        y=-0.05 * max_value,
        s="1s",
        fontsize=LABEL_FONTSIZE,
        va="top",
        ha="center",
    )

    figure.text(
        x=0,
        y=1.2 * max_value,
        s="High contrast",
        fontsize=LABEL_FONTSIZE,
        va="top",
        transform=axes[0, 0].transData,
    )
    figure.text(
        x=0,
        y=1.2 * max_value,
        s="Low contrast",
        fontsize=LABEL_FONTSIZE,
        va="top",
        transform=axes[1, 0].transData,
    )
    figure.text(
        x=BLOCK_SIZE,
        y=1.2 * max_value,
        s=f"Mouse {mouse_id} neuron {neuron}",
        fontsize=LABEL_FONTSIZE,
        va="top",
        ha="right",
        transform=axes[0, -1].transData,
    )

    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_neuron_trace(
    output_dir: Path,
    df: pd.DataFrame,
    flip_neurons: dict[str, np.ndarray],
    plot_dir: Path,
):
    for mouse_id in flip_neurons.keys():
        low_responses, high_responses = load_trace(
            mouse_id=mouse_id, output_dir=output_dir
        )
        neurons = df[
            (df.mouse == mouse_id) & (df.contrast_type == "high_contrast")
        ].neuron.values
        for neuron in tqdm(flip_neurons[mouse_id], desc=f"mouse {mouse_id}"):
            n = np.where(neurons == neuron)[0][0]
            high_response = {k: v[n] for k, v in high_responses.items()}
            low_response = {k: v[n] for k, v in low_responses.items()}
            plot_flip_neuron_trace(
                high_response=high_response,
                low_response=low_response,
                mouse_id=mouse_id,
                neuron=neuron,
                filename=plot_dir
                / "traces"
                / f"mouse{mouse_id}_neuron{neuron:04d}_trace.{FORMAT}",
            )


def process_model(model_name: str, output_dir: Path):
    print(f"Processing {model_name}...")
    df = []
    for mouse_id in tqdm(data.MOUSE_IDS.keys()):
        for contrast_type in ["high_contrast", "low_contrast"]:
            filename = output_dir / "contextual_modulation" / f"mouse{mouse_id}.npz"
            if not filename.exists():
                continue
            responses, neurons = load_data(
                mouse_id=mouse_id,
                output_dir=output_dir,
                contrast_type=contrast_type,
            )
            mouse_df = pd.DataFrame(
                {
                    "neuron": neurons,
                    "center": responses[:, 0],
                    "iso": responses[:, 1],
                    "cross": responses[:, 2],
                    "shift": responses[:, 3],
                }
            )
            mouse_df.insert(loc=0, column="mouse", value=mouse_id)
            mouse_df.insert(loc=2, column="contrast_type", value=contrast_type)
            df.append(mouse_df)
    df = pd.concat(df, ignore_index=True)
    plot_dir = PLOT_DIR / model_name
    plot_combined_box_plot(
        df=df,
        filename=plot_dir / f"contrast_difference_box_combined.{FORMAT}",
    )
    plot_combined_scatter(
        df=df,
        filename=plot_dir / f"iso_vs_cross_scatter_combined.{FORMAT}",
    )
    flip_neurons = find_flip_neurons(df=df)
    plot_flip_neurons_scatter(
        df=df,
        flip_neurons=flip_neurons,
        filename=plot_dir / f"flip_neurons.{FORMAT}",
    )
    plot_neuron_trace(
        output_dir=output_dir,
        df=df,
        flip_neurons=flip_neurons,
        plot_dir=plot_dir,
    )
    print(f"Saved plots to {plot_dir}.\n")


def main():
    models = {
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
        # "ViV1T_VIPcre232_FOV1": Path(
        #     "../runs/rochefort-lab/vivit/003_causal_viv1t_finetune"
        # ),
    }

    for name, output_dir in models.items():
        process_model(model_name=name, output_dir=output_dir)


if __name__ == "__main__":
    main()
