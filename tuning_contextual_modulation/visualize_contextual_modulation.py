from math import ceil
from pathlib import Path
from typing import List
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from sympy.physics.quantum import TimeDepKet

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import stimulus

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

TICK_LENGTH, TICK_PAD, TICK_LINEWIDTH = 3, 2, 1.2

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE

PLOT_DIR = Path("figures")

CONTRAST_TYPES = Literal["high_contrast", "low_contrast"]

# Extracted from Figure 1E of https://www.sciencedirect.com/science/article/pii/S0896627320308916
# using https://plotdigitizer.com/app
KELLER_RESULTS = np.array(
    [
        [-1.8, 0.037141009],
        [-1.375, 0.000384333],
        [-1.125, 0.00799768],
        [-0.875, 0.010749978],
        [-0.625, 0.013609375],
        [-0.375, 0.02755601],
        [-0.125, 0.041168378],
        [0.125, 0.07672107],
        [0.375, 0.162337556],
        [0.625, 0.206429369],
        [0.875, 0.180044337],
        [1.125, 0.101866058],
        [1.375, 0.03295552],
        [1.8, 0.099123771],
    ],
    dtype=np.float32,
)
KELLER_TRIANGLE = 0.6407546898022471


def compute_p_value(responses1: np.ndarray, responses2: np.ndarray):
    p_value = wilcoxon(responses1, responses2).pvalue
    if p_value <= 0.001:
        text = "***"
    elif p_value <= 0.01:
        text = "**"
    elif p_value <= 0.05:
        text = "*"
    else:
        text = "n.s."
    return text


def plot_response_amplitude(df: pd.DataFrame, filename: Path, title: str = None):
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    x_ticks = np.array([0, 1, 2, 3])

    responses = df[["center", "iso", "cross", "shift"]].to_numpy(dtype=np.float32)
    linewidth = 1.2
    box_kw = {
        "notch": False,
        "vert": True,
        "widths": 0.4,
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
    bp = ax.boxplot(responses, positions=x_ticks, **box_kw)
    min_value = 0
    max_value = ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    # get the next multiple of 4
    max_value = ceil(max_value / 4) * 4

    # plot individual response value
    neurons = np.arange(responses.shape[0])
    # ax.text(
    #     x=x_ticks[0] - 0.5,
    #     y=max_value + 0.15 * max_value,
    #     s=f"N={len(neurons)}",
    #     fontsize=TICK_FONTSIZE,
    #     va="top",
    #     ha="left",
    # )

    # # compute p-value against center response
    # for i, stim_type in enumerate((1, 2, 3)):
    #     p_value = compute_p_value(responses[:, 0], responses[:, stim_type])
    #     ax.text(
    #         x=x_ticks[i + 1],
    #         y=(1.06 if p_value != "n.s." else 1.1) * max_value,
    #         s=p_value,
    #         fontsize=TICK_FONTSIZE if p_value != "n.s." else TICK_FONTSIZE - 1,
    #         ha="center",
    #         va="top",
    #         clip_on=False,
    #         transform=ax.transData,
    #     )

    # cap number of neurons to plot 1000 otherwise the plot is really slow to load
    if len(neurons) > 1000:
        rng = np.random.default_rng(1234)
        neurons = rng.choice(neurons, size=1000, replace=False)

    scatter_kw = {
        "s": 10,
        "marker": ".",
        "alpha": 0.1,
        "zorder": 0,
        "facecolors": "none",
        "edgecolors": "black",
        "linewidth": 0.75,
        "clip_on": False,
    }
    responses = responses[neurons]
    for i, stimulus_type in enumerate((0, 1, 2, 3)):
        inliers = np.where(responses[:, stimulus_type] < max_value)[0]
        outliers = np.where(responses[:, stimulus_type] >= max_value)[0]
        # plot responses that are within max_value
        ax.scatter(
            np.random.normal(x_ticks[i], 0.06, size=len(inliers)),
            responses[inliers, stimulus_type],
            **scatter_kw,
        )
        # plot responses that exceed max_value
        ax.scatter(
            np.random.normal(x_ticks[i], 0.06, size=len(outliers)),
            np.full(outliers.shape, fill_value=max_value),
            **scatter_kw,
        )

    ax.set_xlim(x_ticks[0] - 0.5, x_ticks[-1] + 1)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=["Centre", "Iso", "Cross", "Shift"],
        tick_fontsize=TICK_FONTSIZE,
    )
    y_ticks = np.linspace(min_value, max_value, 5, dtype=int)
    # y_ticks = np.array([0, 0.25, 0.5])
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks.tolist()[:-1] + [rf"$\geq${max_value}"],
        label="Predicted ΔF/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=2,
        rotation=90,
    )
    plot.set_ticks_params(ax, length=3, pad=0, minor_length=3)
    ax.tick_params(axis="x", which="major", length=0, pad=4)
    sns.despine(ax=ax, bottom=True, trim=True)
    if title is not None:
        ax.set_title(title, fontsize=LABEL_FONTSIZE - 1, pad=8, linespacing=0.9)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_surround_suppression(df: pd.DataFrame, filename: Path, title: str = None):
    # iso = responses[:, 0] - responses[:, 1]
    # cross = responses[:, 0] - responses[:, 2]

    iso = df.center.values - df.iso.values
    cross = df.center.values - df.cross.values

    print(
        f"\t\tmedian iso: {np.median(iso):.4f}\n"
        f"\t\tmedian cross: {np.median(cross):.4f}"
    )

    iso = iso.flatten()
    cross = cross.flatten()
    df = pd.DataFrame(
        {
            "stim_type": ["iso"] * len(iso) + ["cross"] * len(cross),
            "response": np.concatenate([iso, cross]),
        }
    )
    max_value = 5

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2), dpi=DPI)
    # sns.stripplot(
    #     df,
    #     x="stim_type",
    #     y="response",
    #     size=1,
    #     alpha=0.6,
    #     jitter=0.3,
    #     color="black",
    #     ax=ax,
    # )

    outlier_pad = 1.2
    scatter_kw = {
        "s": 5,
        "marker": ".",
        "alpha": 0.3,
        "zorder": 0,
        "facecolors": "none",
        "edgecolors": "dodgerblue",
        "clip_on": False,
    }
    scatter_width = 0.07
    for i, index in [(1, iso), (2, cross)]:
        neurons = np.arange(len(index))
        outliers1 = np.where(index > max_value)[0]
        outliers2 = np.where(index < -max_value)[0]
        outliers = np.concatenate((outliers1, outliers2))
        # plot response that are not outliers
        y = index[np.setdiff1d(neurons, outliers)]
        x = np.random.normal(i, scatter_width, size=len(y))
        ax.scatter(x, y, **scatter_kw)
        # plot outliers responses that are larger
        y = np.full(outliers1.shape, fill_value=max_value + outlier_pad)
        x = np.random.normal(i, scatter_width, size=len(y))
        ax.scatter(x, y, **scatter_kw)
        # plot outliers responses that are smaller
        y = np.full(outliers2.shape, fill_value=-max_value - outlier_pad)
        x = np.random.normal(i, scatter_width, size=len(y))
        ax.scatter(x, y, **scatter_kw)
        # plot median values
        median = np.median(index)
        ax.plot(
            [i - 0.22, i + 0.22],
            [median, median],
            color="red",
            linewidth=1.5,
            zorder=10,
        )

    ax.axhline(y=0, color="black", linestyle="--", zorder=-10, alpha=0.2, linewidth=1)

    ax.set_xlim(0.5, 2.5)
    plot.set_xticks(
        ax,
        ticks=[1, 2],
        tick_labels=["iso", "cross"],
        tick_fontsize=TICK_FONTSIZE,
    )
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("")

    y_ticks = np.linspace(-max_value, max_value, 5)
    ax.set_ylim(y_ticks[0] - 2 * outlier_pad, y_ticks[-1] + 2 * outlier_pad)
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="Surround\nsuppression",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        linespacing=0.8,
        label_pad=0,
    )
    ax.tick_params(axis="y", length=5, pad=3)
    ax.text(
        x=0.52,
        y=max_value + outlier_pad,
        s=f">{max_value}",
        fontsize=TICK_FONTSIZE,
        ha="right",
        va="center",
    )
    ax.text(
        x=0.52,
        y=-max_value - outlier_pad,
        s=f">{max_value}",
        fontsize=TICK_FONTSIZE,
        ha="right",
        va="center",
    )

    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=1)

    sns.despine(ax=ax, bottom=True, trim=True)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_iso_v_cross_scatter(
    df: pd.DataFrame,
    filename: Path,
    highlight_neuron_idx: int = None,
):
    figure_width = (1 / 3) * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, figure_width),
        dpi=DPI,
    )

    responses = df[["center", "iso", "cross", "shift"]].to_numpy(dtype=np.float32)

    cross, iso = responses[:, 2], responses[:, 1]

    scatter_kw = {
        "s": 5,
        "alpha": 0.2,
        "color": "black",
        "edgecolors": "none",
        "zorder": 2,
        "clip_on": False,
    }

    # clip iso and cross responses to max_value
    min_value, max_value = 0, 10
    ax.scatter(
        x=np.clip(cross, min_value, max_value),
        y=np.clip(iso, min_value, max_value),
        **scatter_kw,
    )
    if highlight_neuron_idx is not None:
        ax.scatter(
            x=cross[highlight_neuron_idx],
            y=iso[highlight_neuron_idx],
            s=18,
            color="orangered",
            edgecolors="none",
            zorder=3,
            clip_on=False,
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

    ticks = np.array([min_value, max_value])
    tick_labels = [min_value, rf"$\geq${max_value}"]
    ax.set_xlim(ticks[0], ticks[-1])
    plot.set_xticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label="Cross [ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-5,
    )
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_ylim(ticks[0], ticks[-1])
    plot.set_yticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label="Iso [ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-4,
        rotation=90,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    p_value = wilcoxon(cross, iso).pvalue
    print(f"\t\tWilcoxon iso vs cross p-value: {p_value:.04e}")
    if p_value <= 10e-10:
        p_value = r"$p<10^{-10}$"
    elif p_value <= 0.001:
        p_value = r"$p<10^{-3}$"
    elif p_value <= 0.01:
        p_value = r"$p<10^{-2}$"
    elif p_value <= 0.05:
        p_value = r"p<0.05"
    else:
        p_value = "p>0.05"
    figure.text(
        x=0.05 * max_value,
        y=1.05 * max_value,
        s=f"{p_value}\n{len(responses)} neurons",
        fontsize=TICK_FONTSIZE,
        ha="left",
        va="top",
        linespacing=1.1,
        transform=ax.transData,
    )
    plot.set_ticks_params(ax, length=TICK_LENGTH, pad=0, minor_length=TICK_LENGTH)
    ax.tick_params(axis="x", which="major", length=TICK_LENGTH, pad=TICK_PAD)
    sns.despine(ax=ax)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_contextual_modulation_index(df: pd.DataFrame, model_name: str, filename: Path):
    mouse_ids = df.mouse.unique()

    figure_width = (1 / 3) * PAPER_WIDTH

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(figure_width, figure_width),
        width_ratios=[0.1, 0.8, 0.1],
        dpi=DPI,
    )

    x_ticks = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    x_tick_labels = ["", "-1", "", "0", "", "1", ""]
    linewidth = 1.5
    hist_kws = {
        "bins": 12,
        "range": (-1.5, 1.5),
        "color": "black",
        "histtype": "step",
        "clip_on": False,
    }

    max_value = -np.inf  # find max value of the bars
    means, medians = [], []
    for i, mouse_id in enumerate(mouse_ids):
        cmi = df[df.mouse == mouse_id].cmi.values
        h_y, h_x, _ = axes[1].hist(
            cmi,
            alpha=0.3,
            linestyle="-",
            linewidth=0.8,
            # show percentage instead of count
            weights=np.ones(len(cmi)) / len(cmi),
            **hist_kws,
        )
        means.append(np.mean(cmi))
        medians.append(np.median(cmi))
        max_value = max(max_value, np.max(h_y))

    if len(means) > 1:
        print(
            f"\t\tCMI mean +/- sem: {np.mean(means):.3f} +/- {sem(means):.3f}\n"
            f"\t\tCMI median: {np.median(medians):.3f}"
        )

    # plot average over animals
    cmi = df.cmi.values
    h_y, h_x, _ = axes[1].hist(
        cmi,
        alpha=1,
        linewidth=linewidth,
        weights=np.ones(len(cmi)) / len(cmi),
        **hist_kws,
    )
    max_value = max(max_value, np.max(h_y))
    medians.append(np.median(cmi))

    max_value = ceil(max_value * 10) / 10
    axes[1].scatter(
        medians[-1],
        max_value,
        s=25,
        marker="v",
        facecolors="black",
        edgecolors="none",
        alpha=1,
        clip_on=False,
    )

    p_value = wilcoxon(cmi).pvalue
    print(f"\t\tWilcoxon CMI p-value: {p_value:.04e}")
    axes[1].text(
        medians[-1],
        max_value * 1.01,
        plot.get_p_value_asterisk(p_value),
        alpha=1,
        clip_on=False,
        fontsize=LABEL_FONTSIZE,
        ha="center",
    )

    # plot labels
    text = f"{cmi.shape[0]} neurons"
    # if len(mouse_ids) > 1:
    #     text += f"\nfrom {len(mouse_ids)} mice"
    # if len(mouse_ids) == 1:
    #     text = f"Mouse {mouse_ids[0]}\n" + text
    figure.text(
        x=-1.9,
        y=1.03 * max_value,
        s=text,
        ha="left",
        va="top",
        fontsize=TICK_FONTSIZE,
        linespacing=1,
        transform=axes[0].transData,
    )
    axes[1].axvline(
        x=0,
        color="black",
        alpha=0.5,
        linestyle="dashed",
        linewidth=1,
        zorder=-1,
    )

    axes[1].set_xlim(x_ticks[0], x_ticks[-1])
    plot.set_xticks(
        axes[1],
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        label="Contextual modulation index",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )
    y_ticks = np.linspace(0, max_value, 2)
    axes[1].set_ylim(y_ticks[0], y_ticks[-1])
    axes[1].set_yticks([])
    sns.despine(ax=axes[1], left=True)

    plot.set_xticks(
        axes[0],
        ticks=[-1.8],
        tick_labels=["<-1.5"],
        label="",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )
    axes[0].set_xlim([-2, -1.6])
    axes[0].set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axes[0],
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="Fraction of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-7,
        linespacing=0.9,
    )
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    sns.despine(ax=axes[0])

    plot.set_xticks(
        axes[2],
        ticks=[1.8],
        tick_labels=[">1.5"],
        label="",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    axes[2].set_xlim([1.6, 2])
    axes[2].set_yticks([])
    sns.despine(ax=axes[2], left=True)

    for ax in axes:
        plot.set_ticks_params(
            ax,
            length=TICK_LENGTH,
            pad=TICK_PAD,
            minor_length=TICK_LENGTH,
            linewidth=TICK_LINEWIDTH,
        )

    plot.save_figure(figure, filename=filename, layout="constrained", dpi=DPI)


def plot_model(model_name: str, output_dir: Path) -> pd.DataFrame:
    print(f"\nPlot {model_name} contextual modulation in {output_dir}.")
    df = pd.read_parquet(output_dir / "contextual_modulation.parquet")
    df["model_name"] = model_name

    for contrast_type in ["high_contrast", "low_contrast"]:
        print(f"Plot {contrast_type}...")
        plot_dir = PLOT_DIR / contrast_type / model_name
        highlight_neuron_idx = None
        for mouse_id in df.mouse.unique():
            print(f"\tProcessing mouse {mouse_id}")
            indexes = (df.contrast_type == contrast_type) & (df.mouse == mouse_id)
            neurons = df[indexes].neuron.values
            if model_name == "ViV1T":
                # highlight neuron 930 from mouse B
                if mouse_id == "A":
                    highlight_neuron_idx = len(neurons)
                elif mouse_id == "B":
                    highlight_neuron_idx += np.where(neurons == 866)[0][0]
            plot_response_amplitude(
                df=df[indexes],
                filename=plot_dir
                / f"mouse{mouse_id}"
                / f"{contrast_type}_response_magnitude.{FORMAT}",
                title=f"mouse {mouse_id}",
            )
            # plot_surround_suppression(
            #     df=df[indexes],
            #     filename=plot_dir
            #     / f"mouse{mouse_id}"
            #     / f"surround_suppression.{FORMAT}",
            #     title=f"mouse {mouse_id}",
            # )
            plot_iso_v_cross_scatter(
                df=df[indexes],
                filename=plot_dir
                / f"mouse{mouse_id}"
                / f"{contrast_type}_iso_vs_cross_scatter.png",
            )
            plot_contextual_modulation_index(
                df=df[indexes],
                model_name=model_name,
                filename=plot_dir
                / f"mouse{mouse_id}"
                / f"{contrast_type}_cmi.{FORMAT}",
            )
        print(f"\tCombine neurons from all mice")
        plot_response_amplitude(
            df=df[df.contrast_type == contrast_type],
            filename=plot_dir / f"{contrast_type}_response_magnitude.{FORMAT}",
        )
        plot_iso_v_cross_scatter(
            df=df[df.contrast_type == contrast_type],
            filename=plot_dir / f"{contrast_type}_iso_vs_cross_scatter.jpg",
            highlight_neuron_idx=highlight_neuron_idx,
        )
        plot_contextual_modulation_index(
            df=df[df.contrast_type == contrast_type],
            model_name=model_name,
            filename=plot_dir / f"{contrast_type}_cmi.{FORMAT}",
        )
        print(f"Saved results to {plot_dir}.\n")
    return df


def plot_contextual_modulation_index_comparison(df: pd.DataFrame, filename: Path):
    model_names = ["LN", "fCNN", "DwiseNeuro", "ViV1T"]
    assert df.contrast_type.nunique() == 1

    figure_width, figure_height = (1 / 3) * PAPER_WIDTH, 3.25
    figure, axes = plt.subplots(
        nrows=5,
        ncols=3,
        figsize=(figure_width, figure_height),
        width_ratios=[0.1, 0.8, 0.1],
        height_ratios=[1, 1, 1, 1, 1],
        gridspec_kw={
            "wspace": 0.15,
            "hspace": 0.2,
            "left": 0.17,
            "right": 0.98,
            "top": 0.98,
            "bottom": 0.08,
        },
        dpi=DPI,
    )

    x_ticks = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    x_tick_labels = ["", "-1", "", "0", "", "1", ""]

    linewidth = 1.4
    linestyle = "-"

    # plot histogram extracted from Keller et al.
    bar_kw = {
        "width": 0.25,
        "facecolor": "black",
        "edgecolor": "none",
        "linewidth": linewidth,
        "linestyle": linestyle,
        "alpha": 0.8,
        "clip_on": False,
    }
    axes[-1, 0].bar(
        x=[KELLER_RESULTS[:, 0][0]],
        height=[KELLER_RESULTS[:, 1][0]],
        **bar_kw,
    )
    axes[-1, 1].bar(
        x=KELLER_RESULTS[:, 0][1:-1],
        height=KELLER_RESULTS[:, 1][1:-1],
        **bar_kw,
    )
    axes[-1, 2].bar(
        x=[KELLER_RESULTS[:, 0][-1]],
        height=[KELLER_RESULTS[:, 1][-1]],
        **bar_kw,
    )

    peak_index = KELLER_RESULTS[np.argmax(KELLER_RESULTS[:, 1]), 0]

    hist_kws = {
        "bins": 12,
        "range": (-1.5, 1.5),
        "linewidth": linewidth,
        "linestyle": linestyle,
        "histtype": "step",
        "clip_on": False,
        "zorder": 1,
        "alpha": 1,
    }
    max_values = {}
    for i, model_name in enumerate(model_names):
        cmi = df[df.model_name == model_name].cmi.values
        h_y, h_x, _ = axes[i, 1].hist(
            cmi,
            weights=np.ones(len(cmi)) / len(cmi),
            label=model_name,
            color=plot.get_color(model_name),
            **hist_kws,
        )
        max_values[model_name] = ceil(np.max(h_y) * 10) / 10

    recording_name = r"$\it{In}$ $\it{vivo}$" + "\nrecordings"
    max_values[recording_name] = ceil(np.max(KELLER_RESULTS[:, 1]) * 10) / 10
    model_names.append(recording_name)

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            max_value = max_values[model_names[i]]
            y_ticks = np.array([0, max_value])
            axes[i, j].set_ylim(y_ticks[0], y_ticks[-1])
            show_x_ticks = i == axes.shape[0] - 1
            match j:
                case 0:
                    plot.set_xticks(
                        axes[i, j],
                        ticks=[-1.8],
                        tick_labels=["<-1.5"] if show_x_ticks else [],
                        label="",
                        tick_fontsize=TICK_FONTSIZE,
                        label_fontsize=LABEL_FONTSIZE,
                        label_pad=0,
                    )
                    axes[i, j].set_xlim([-2, -1.6])
                    plot.set_yticks(
                        axes[i, j],
                        ticks=y_ticks,
                        tick_labels=(100 * y_ticks).astype(int),
                        label="% of neuron" if i == 2 else "",
                        tick_fontsize=TICK_FONTSIZE,
                        label_fontsize=LABEL_FONTSIZE,
                        label_pad=0,
                        linespacing=0.8,
                    )
                    axes[i, j].yaxis.set_minor_locator(MultipleLocator(0.1))
                    sns.despine(ax=axes[i, j])
                    figure.text(
                        x=-1.85,
                        y=max_value,
                        s=model_names[i],
                        fontsize=LABEL_FONTSIZE,
                        ha="left",
                        va="top",
                        linespacing=0.8,
                        transform=axes[i, 0].transData,
                    )
                case 1:
                    axes[i, j].axvline(
                        x=peak_index,
                        color="black",
                        alpha=0.4,
                        linestyle="dashed",
                        linewidth=linewidth,
                        zorder=-1,
                    )
                    axes[i, j].set_xlim(x_ticks[0], x_ticks[-1])
                    plot.set_xticks(
                        axes[i, j],
                        ticks=x_ticks,
                        tick_labels=x_tick_labels if show_x_ticks else [],
                        label=(
                            "Contextual modulation index"
                            if i == axes.shape[0] - 1
                            else ""
                        ),
                        tick_fontsize=TICK_FONTSIZE,
                        label_fontsize=LABEL_FONTSIZE,
                        label_pad=0,
                    )
                    axes[i, j].xaxis.set_label_coords(0.44, -0.3)
                    axes[i, j].set_yticks([])
                    sns.despine(ax=axes[i, j], left=True)
                case 2:
                    plot.set_xticks(
                        axes[i, j],
                        ticks=[1.8],
                        tick_labels=[">1.5"] if show_x_ticks else [],
                        label="",
                        tick_fontsize=TICK_FONTSIZE,
                        label_fontsize=LABEL_FONTSIZE,
                        label_pad=0,
                    )
                    axes[i, j].set_xlim([1.6, 2])
                    axes[i, j].set_yticks([])
                    sns.despine(ax=axes[i, j], left=True)
            plot.set_ticks_params(axes[i, j])
            axes[i, j].spines["bottom"].set_zorder(2)
            axes[i, j].tick_params(zorder=3)

    # Plot stimulus illustration
    w = 0.13
    h = w * figure_width / figure_height
    ax1 = figure.add_axes((0.85, 0.92, w, h))
    ax2 = figure.add_axes((0.85, 0.84, w, h))

    fps = 30
    stimulus_kw = {
        "stimulus_size": 70,
        "center": (np.nan, np.nan),
        "cpd": 0.04,
        "cpf": 2 / fps,
        "height": 36,
        "width": 64,
        "num_frames": 1,
        "phase": 0,
        "contrast": 1,
        "fps": fps,
        "to_tensor": False,
    }
    iso = stimulus.create_center_surround_grating(
        center_direction=90, surround_direction=90, **stimulus_kw
    )
    ax1.imshow(
        iso[0, 0, :, 14:50],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    ax1.set_ylabel("Iso", fontsize=TICK_FONTSIZE, labelpad=0)
    cross = stimulus.create_center_surround_grating(
        center_direction=90, surround_direction=0, **stimulus_kw
    )
    ax2.imshow(
        cross[0, 0, :, 14:50],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    ax2.set_ylabel("Cross", fontsize=TICK_FONTSIZE, labelpad=0)
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        plot.set_ticks_params(axis=ax)

    plot.save_figure(figure, filename=filename, layout="none", dpi=DPI)


def main():
    models = {
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    filename = PLOT_DIR / "cm.parquet"
    df = []
    for model_name, output_dir in models.items():
        df.append(plot_model(model_name=model_name, output_dir=output_dir))
    df = pd.concat(df, ignore_index=True)
    # df.to_parquet(filename)
    # df = pd.read_parquet(filename)
    # plot_contextual_modulation_index_comparison(
    #     df=df[df.contrast_type == "high_contrast"],
    #     filename=PLOT_DIR / "high_contrast" / f"high_contrast_cmi_comparison.{FORMAT}",
    # )


if __name__ == "__main__":
    main()
