from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.stats import sem

from viv1t.utils import plot

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8
TICK_LENGTH, TICK_PAD, TICK_LINEWIDTH = 3, 2, 1.2

DPI = 600
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches
FPS = 30

THRESHOLD = 0.2  # response has to be 20% stronger

plot.set_font()

PLOT_DIR = Path("figures")
DATA_DIR = Path("../../data")
OUTPUT_DIR = Path("../../runs")

PATTERN_SIZE = 30
BLANK_SIZE = 15


def compute_cmi(iso: pd.Series, cross: pd.Series) -> np.ndarray:
    iso = iso.apply(np.nanmean).values
    cross = cross.apply(np.nanmean).values
    cmi = (cross - iso) / (cross + iso)
    return cmi


def find_flip_neurons(df: pd.DataFrame) -> np.ndarray:
    high_iso = df.high_iso.apply(np.nanmean).values
    high_cross = df.high_cross.apply(np.nanmean).values

    low_iso = df.low_iso.apply(np.nanmean).values
    low_cross = df.low_cross.apply(np.nanmean).values

    high_contrast = ((high_cross - high_iso) / high_iso) >= THRESHOLD
    low_contrast = ((low_iso - low_cross) / low_cross) >= THRESHOLD

    flip_neurons = (high_contrast == True) & (low_contrast == True)
    return flip_neurons


def plot_contextual_modulation_index(df: pd.DataFrame, filename: Path):
    high_cmi = compute_cmi(iso=df.high_iso, cross=df.high_cross)
    low_cmi = compute_cmi(iso=df.low_iso, cross=df.low_cross)
    print(f"\tNumber of neurons: {len(high_cmi)}")
    print(
        f"\tHigh contrast CMI: mean {np.mean(high_cmi):.03f}, median: {np.median(high_cmi):.03f}\n"
        f"\tLow contrast CMI: mean {np.mean(low_cmi):.03f}, median: {np.median(low_cmi):.03f}"
    )

    figure_width = (1 / 4) * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(figure_width, figure_width), dpi=DPI
    )

    x_ticks = np.array([-1, -0.5, 0, 0.5, 1])
    x_tick_labels = ["-1", "", "", "", "1"]
    hist_kws = {
        "bins": 10,
        "alpha": 0.8,
        "range": (-1, 1),
        "histtype": "step",
        "clip_on": False,
    }

    values = [
        ("black", "-", "high", high_cmi),
        ("grey", "-", "low", low_cmi),
    ]

    max_value = -np.inf  # find max value of the bars
    legend_handles = []
    for color, linestyle, label, cmi in values:
        linewidth = 1.5 if "cross" in label else 2
        h_y, h_x, _ = ax.hist(
            cmi,
            linewidth=linewidth,
            linestyle=linestyle,
            weights=np.ones(len(cmi)) / len(cmi),
            color=color,
            label=label,
            **hist_kws,
        )
        max_value = max(max_value, np.max(h_y))
        handle = Line2D(
            [0],
            [0],
            color=color,
            label=label.capitalize(),
            linestyle=linestyle,
            linewidth=1.25,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        legend_handles.append(handle)

    max_value = ceil(max_value * 10) / 10
    max_value += 0.1

    for color, linestyle, label, cmi in values:
        if color == "grey":
            y = 0.95 * max_value
        else:
            y = 1.1 * max_value
        median = np.median(cmi)
        ax.scatter(
            np.median(cmi),
            y,
            s=20,
            marker="v",
            facecolors="none",
            edgecolors=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=1,
            clip_on=False,
        )
        p_value = stats.wilcoxon(cmi).pvalue
        print(f"\tCMI {label} contrast p-value: {p_value:.03e}")
        p_value = plot.get_p_value_asterisk(p_value)
        ax.text(
            x=median,
            y=y * (1.005 if p_value != "n.s." else 1.06),
            s=p_value,
            alpha=1,
            clip_on=False,
            fontsize=TICK_FONTSIZE if p_value == "n.s." else LABEL_FONTSIZE,
            ha="center",
        )

    legend_kw = {
        "loc": "upper left",
        "ncols": 1,
        "fontsize": TICK_FONTSIZE - 1,
        "frameon": False,
        "title_fontsize": TICK_FONTSIZE - 1,
        "handletextpad": 0.2,
        "handlelength": 0.8,
        "labelspacing": 0.15,
        "columnspacing": 0,
        "borderpad": 0,
        "borderaxespad": 0,
        "alignment": "left",
    }
    legend = ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(-0.92, max_value),
        bbox_transform=ax.transData,
        title="Contrast",
        **legend_kw,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    ax.add_artist(legend)

    ax.axvline(
        x=0,
        color="black",
        alpha=0.2,
        linestyle="dashed",
        linewidth=1.2,
        zorder=-1,
    )

    ax.set_xlim(x_ticks[0], x_ticks[-1])
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        label="CMI",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-5,
    )
    y_ticks = np.linspace(0, max_value, 2)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=(100 * y_ticks).astype(int),
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-6,
        linespacing=0.9,
        # rotation=90,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    sns.despine(ax=ax)
    plot.set_ticks_params(ax)
    # ax.tick_params(axis="x", which="both", length=TICK_LENGTH, pad=TICK_PAD)
    ax.set_axisbelow(b=False)
    plot.save_figure(figure, filename=filename, layout="constrained", dpi=DPI)


def plot_iso_v_cross_scatter(
    df: pd.DataFrame,
    model_name: str,
    contrast_type: str,
    filename: Path,
    highlight_neuron_idx: int = None,
    plot_flip_neurons_in_red: bool = False,
    max_value: float = 8,
):
    figure_width = (1 / 4) * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(figure_width, figure_width), dpi=DPI
    )

    match contrast_type:
        case "low_contrast":
            iso = df.low_iso.apply(np.nanmean).values
            cross = df.low_cross.apply(np.nanmean).values
        case "high_contrast":
            iso = df.high_iso.apply(np.nanmean).values
            cross = df.high_cross.apply(np.nanmean).values
        case _:
            raise ValueError("contrast_type must be 'low_contrast' or 'high_contrast'")

    flip_neurons = find_flip_neurons(df=df)

    min_value = 0
    # max_value = ceil(ceil(np.max([iso, cross]) * 10) / 10)
    # max_value = 8

    scatter_kw = {
        "s": 10,
        "alpha": 0.35,
        "edgecolors": "none",
        "zorder": 2,
        "clip_on": False,
    }
    if plot_flip_neurons_in_red:
        # flip_neurons = df.flip_neuron.values
        ax.scatter(
            x=np.clip(cross[~flip_neurons], min_value, max_value),
            y=np.clip(iso[~flip_neurons], min_value, max_value),
            color=plot.get_color(model_name),
            **scatter_kw,
        )
        ax.scatter(
            x=np.clip(cross[flip_neurons], min_value, max_value),
            y=np.clip(iso[flip_neurons], min_value, max_value),
            color="red",
            **scatter_kw,
        )
    else:
        ax.scatter(
            x=np.clip(cross, min_value, max_value),
            y=np.clip(iso, min_value, max_value),
            color=plot.get_color(model_name),
            **scatter_kw,
        )
    # plot average dot
    ax.scatter(
        x=np.mean(cross),
        y=np.mean(iso),
        alpha=1,
        marker="^",
        s=30,
        color="gold",
        edgecolor="black",
        zorder=10,
        linewidth=1,
        clip_on=False,
    )
    if highlight_neuron_idx is not None:
        ax.scatter(
            x=cross[highlight_neuron_idx],
            y=iso[highlight_neuron_idx],
            s=14,
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
    ax.text(
        x=min_value,
        y=1.02 * max_value,
        s=contrast_type.capitalize().replace("_", " "),
        fontsize=TICK_FONTSIZE,
        va="bottom",
        ha="left",
    )
    if max_value >= 100:
        minor_locator = max_value / 2
        max_value = 1
        label = r"(a.u. $\Delta$F/F)"
    elif max_value >= 1:
        minor_locator = 1
        label = r"($\Delta$F/F)"
    else:
        minor_locator = 0.1
        label = r"($\Delta$F/F)"

    # ticks = np.array([min_value, max_value], dtype=int)

    tick_labels = [min_value, rf"$\geq${max_value}"]
    ax.set_xlim(ticks[0], ticks[-1])
    plot.set_xticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label=f"Cross {label}",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-5,
    )

    ax.set_ylim(ticks[0], ticks[-1])
    plot.set_yticks(
        ax,
        ticks=ticks,
        tick_labels=tick_labels,
        label=f"Iso {label}",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-8,
        # rotation=90,
    )

    ax.xaxis.set_minor_locator(MultipleLocator(minor_locator))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_locator))

    p_value = stats.wilcoxon(cross, iso).pvalue
    print(f"\t{contrast_type} iso vs cross scatter p-value: {p_value:.03e}")

    plot.set_ticks_params(ax)
    # ax.tick_params(axis="x", which="major", length=TICK_LENGTH, pad=TICK_PAD)
    sns.despine(ax=ax)
    plot.save_figure(figure, filename=filename, layout="constrained", dpi=DPI)


def add_p_value(
    ax: axes.Axes,
    response1: np.ndarray,
    response2: np.ndarray,
    position1: float,
    position2: float,
    max_value: float,
    unit: float | None = None,
    num_compare: int = 1,
    alternative: str = "two-sided",
) -> float:
    p_value = stats.wilcoxon(response1, response2, alternative=alternative).pvalue
    p_value = num_compare * p_value  # adjust for multiple tests
    if unit is None:
        unit = 0.1 * max_value
    plot.add_p_value(
        ax=ax,
        x0=position1,
        x1=position2,
        y=1.04 * max_value,
        p_value=p_value,
        fontsize=LABEL_FONTSIZE,
        tick_length=0.35 * unit,
        tick_linewidth=1,
        text_pad=0.7 * unit,
    )
    return p_value


def plot_box_plot(
    df: pd.DataFrame,
    filename: Path,
    color: str = "black",
    connect_dots: bool = False,
):
    high_center = df.high_center.apply(np.nanmean).values
    high_iso = df.high_iso.apply(np.nanmean).values
    high_cross = df.high_cross.apply(np.nanmean).values

    low_center = df.low_center.apply(np.nanmean).values
    low_iso = df.low_iso.apply(np.nanmean).values
    low_cross = df.low_cross.apply(np.nanmean).values

    flip_neurons = find_flip_neurons(df=df)
    flip_neurons = np.where(flip_neurons)[0]

    high_center = high_center[flip_neurons]
    high_iso = high_iso[flip_neurons]
    high_cross = high_cross[flip_neurons]

    low_center = low_center[flip_neurons]
    low_iso = low_iso[flip_neurons]
    low_cross = low_cross[flip_neurons]

    num_neurons = len(high_center)
    print(f"\tnumber of flip neurons: {num_neurons}")

    high_responses = [high_iso / high_center, high_cross / high_center]
    low_responses = [low_iso / low_center, low_cross / low_center]

    rng = np.random.RandomState(1234)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=((1 / 3) * PAPER_WIDTH, 1.3),
        sharey=True,
        dpi=DPI,
    )

    linewidth = 1.2
    box_kw = {
        "notch": False,
        "vert": True,
        "widths": 0.4,
        "showfliers": False,
        "showmeans": False,
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
        # "meanprops": {
        #     "markersize": 4.5,
        #     "markerfacecolor": "gold",
        #     "markeredgecolor": "black",
        #     "markeredgewidth": 1,
        #     "clip_on": False,
        #     "zorder": 20,
        # },
        "medianprops": {
            "linewidth": 1.2 * linewidth,
            "color": "black",
            "solid_capstyle": "butt",
            "clip_on": False,
            "zorder": 20,
        },
    }
    scatter_kw = {
        "s": 20,
        "marker": ".",
        "alpha": 0.1 if num_neurons > 100 else 0.6,
        "zorder": 0,
        "facecolors": "none",
        "clip_on": False,
        "edgecolors": "red",
        "linewidth": 1,
    }
    line_kw = {
        "color": color,
        "linestyle": "-",
        "linewidth": 1,
        "alpha": 0.25,
        "zorder": 0,
        "clip_on": False,
    }
    x_ticks = np.arange(2)
    max_value = 0
    for contrast_type in ["high_contrast", "low_contrast"]:
        match contrast_type:
            case "high_contrast":
                ax = axes[0]
                responses = high_responses
            case "low_contrast":
                ax = axes[1]
                responses = low_responses
            case _:
                raise RuntimeError(f"Unknown contrast type: {contrast_type}")
        bp = ax.boxplot(
            responses,
            positions=x_ticks,
            **box_kw,
        )
        max_value = max(
            max_value,
            np.ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]])),
        )
        del responses, ax
    max_value = max(max_value, 4)

    unit = 0.1 * max_value
    num_compare = 1
    p_value = add_p_value(
        ax=axes[0],
        response1=high_iso,
        response2=high_cross,
        position1=x_ticks[0],
        position2=x_ticks[1],
        max_value=0.965 * max_value,
        unit=unit,
        num_compare=num_compare,
    )
    print(f"\thigh contrast iso vs cross box p-value: {p_value:.03e}")
    p_value = add_p_value(
        ax=axes[1],
        response1=low_iso,
        response2=low_cross,
        position1=x_ticks[0],
        position2=x_ticks[1],
        max_value=0.965 * max_value,
        unit=unit,
        num_compare=num_compare,
    )
    print(f"\tlow contrast iso vs cross box p-value: {p_value:.03e}")

    for contrast_type in ["high_contrast", "low_contrast"]:
        match contrast_type:
            case "high_contrast":
                ax = axes[0]
                responses = high_responses
                y_label = "Sum norm. " + r"$\Delta$F/F"
                title = "High contrast"
            case "low_contrast":
                ax = axes[1]
                responses = low_responses
                y_label = ""
                title = "Low contrast"
            case _:
                raise RuntimeError(f"Unknown contrast type: {contrast_type}")
        for i, (position, response) in enumerate(zip(x_ticks, responses)):
            neurons = np.arange(num_neurons)
            response = response[neurons]
            outliers = np.where(response >= max_value)[0]
            inliers = np.setdiff1d(neurons, outliers)
            x = rng.normal(position, 0.0, size=len(response))
            label = ""
            if i == 0 and contrast_type == "high_contrast":
                label = (
                    r"$\it{in\;silico}$" if color == "limegreen" else r"$\it{in\;vivo}$"
                )
            ax.scatter(
                x[inliers],
                response[inliers],
                label=label,
                **scatter_kw,
            )
            # plot outlier neurons
            if outliers.size > 0:
                ax.scatter(
                    x[outliers],
                    np.full(outliers.shape, fill_value=max_value),
                    **scatter_kw,
                )
            # connect dots
            if i == 0 and connect_dots:
                for neuron in np.arange(num_neurons):
                    ax.plot(
                        [x_ticks[0], x_ticks[1]],
                        np.clip(
                            [responses[0][neuron], responses[1][neuron]],
                            a_min=0,
                            a_max=max_value,
                        ),
                        **line_kw,
                    )
            del position, response

        ax.axhline(
            y=1,
            color="black",
            alpha=0.5,
            linestyle="dotted",
            dashes=(1, 1),
            linewidth=1,
            zorder=-1,
        )

        xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.5]
        ax.set_xlim(*xlim)
        plot.set_xticks(
            axis=ax,
            ticks=x_ticks,
            tick_labels=["Iso", "Cross"],
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=TICK_FONTSIZE,
            label_pad=0,
        )
        y_ticks = np.array([0, max_value], dtype=int)
        ax.set_ylim(y_ticks[0], y_ticks[-1])
        plot.set_yticks(
            ax,
            ticks=y_ticks,
            tick_labels=y_ticks,
            label=y_label,
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=TICK_FONTSIZE,
            label_pad=-1,
            linespacing=0.9,
        )
        if y_ticks[-1] < 100:
            ax.yaxis.set_minor_locator(MultipleLocator(1))
        plot.set_ticks_params(ax, minor_length=plot.PARAMS_LENGTH)
        sns.despine(ax=ax)
        ax.set_title(title, fontsize=TICK_FONTSIZE, pad=7)

    # legend = axes[0].legend(
    #     bbox_to_anchor=(xlim[0] + 0.1 * xlim[1], 0.9 * max_value),
    #     bbox_transform=axes[0].transData,
    #     loc="upper left",
    #     fontsize=TICK_FONTSIZE,
    #     frameon=False,
    #     handletextpad=0.2,
    #     handlelength=0.75,
    #     labelspacing=0.15,
    #     columnspacing=0,
    #     borderpad=0,
    #     borderaxespad=0,
    #     alignment="left",
    # )
    # for lh in legend.legend_handles:
    #     lh.set_alpha(1)
    # for text in legend.texts:
    #     text.set_y(-1)

    # if num_neurons >= 1000:
    #     num_neurons = num_neurons // 1000
    #     num_neurons = f"{num_neurons:.0f}k"
    # else:
    #     num_neurons = f"{num_neurons}"
    # axes[1].text(
    #     x=xlim[-1],
    #     y=0.02 * max_value,
    #     s=r"$N_{neurons}$=" + num_neurons,
    #     ha="right",
    #     va="bottom",
    #     fontsize=TICK_FONTSIZE - 1,
    #     color="black",
    #     clip_on=False,
    # )
    plot.save_figure(figure=figure, filename=filename, layout="constrained", dpi=DPI)


def get_flip_neuron_percentage(df: pd.DataFrame) -> np.ndarray:
    mouse_ids = df.mouse.unique()
    percentages = np.zeros(len(mouse_ids), dtype=np.float32)
    for i, mouse_id in enumerate(mouse_ids):
        flip_neurons = find_flip_neurons(df=df[df.mouse == mouse_id])
        percentages[i] = np.count_nonzero(flip_neurons) / len(flip_neurons)
    percentages = 100 * percentages
    return percentages


def find_shuffle_flip_neurons(
    iso: np.ndarray, cross: np.ndarray, rng=np.random.Generator
) -> np.ndarray:
    assert iso.shape == cross.shape
    # shuffle low and high contrast pairing
    iso = rng.permuted(iso, axis=1)
    cross = rng.permuted(cross, axis=1)
    size = iso.shape[1] // 2
    low_iso = np.mean(iso[:, :size], axis=1)
    high_iso = np.mean(iso[:, size:], axis=1)
    low_cross = np.mean(cross[:, :size], axis=1)
    high_cross = np.mean(cross[:, size:], axis=1)
    high_contrast = ((high_cross - high_iso) / high_iso) >= THRESHOLD
    low_contrast = ((low_iso - low_cross) / low_cross) >= THRESHOLD
    flip_neurons = (high_contrast == True) & (low_contrast == True)
    return np.count_nonzero(flip_neurons) / len(flip_neurons)


def flip_neuron_shuffle_test(df: pd.DataFrame, num_tests: int = 100) -> np.ndarray:
    mouse_ids = df.mouse.unique()
    percentages = np.zeros((len(mouse_ids), num_tests), dtype=np.float32)
    rng = np.random.default_rng(1234)
    for i, mouse_id in enumerate(mouse_ids):
        low_iso = np.stack(df[df.mouse == mouse_id]["low_iso"].values)
        high_iso = np.stack(df[df.mouse == mouse_id]["high_iso"].values)
        iso = np.concat([low_iso, high_iso], axis=1)
        low_cross = np.stack(df[df.mouse == mouse_id]["low_cross"].values)
        high_cross = np.stack(df[df.mouse == mouse_id]["high_cross"].values)
        cross = np.concatenate([low_cross, high_cross], axis=1)
        del low_iso, high_iso, low_cross, high_cross
        for j in range(num_tests):
            percentages[i, j] = find_shuffle_flip_neurons(
                iso=iso.copy(), cross=cross.copy(), rng=rng
            )
    # average over number of shuffles
    percentages = np.mean(percentages, axis=1)
    percentages = 100 * percentages
    return percentages


def plot_sensorium_flip_neuron_percentage(
    sensorium_df: pd.DataFrame,
    filename: Path,
):
    sensorium_percentage = get_flip_neuron_percentage(df=sensorium_df)
    print(
        f"In silico (Sensorium) real percentages: {sensorium_percentage}\n"
        f"\tAverage (%): {np.mean(sensorium_percentage):.03f} +- {sem(sensorium_percentage):.03f}"
    )
    sensorium_chance = flip_neuron_shuffle_test(df=sensorium_df)
    print(f"In silico (Sensorium) chance percentages: {sensorium_chance}")

    x_ticks = np.array([0], dtype=np.float32)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 4) * PAPER_WIDTH, 1.3),
        gridspec_kw={"wspace": 0.0, "hspace": 0},
        dpi=DPI,
    )

    linewidth = 1.2
    max_value = 0
    box_width = 0.18
    percentage_box_kw = {
        "notch": False,
        "vert": True,
        "widths": box_width,
        "showfliers": False,
        "showmeans": False,
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
            "markersize": 4.5,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 1,
            "clip_on": False,
            "zorder": 20,
        },
        "medianprops": {
            "linewidth": 1.2 * linewidth,
            "color": "black",
            "solid_capstyle": "butt",
            "clip_on": False,
            "zorder": 20,
        },
    }
    null_box_kw = {
        "notch": False,
        "vert": True,
        "widths": box_width,
        "showfliers": False,
        "showmeans": False,
        "boxprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
            "color": "grey",
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
            "color": "grey",
        },
        "whiskerprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
            "color": "grey",
        },
        "meanprops": {
            "markersize": 4.5,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 1,
            "clip_on": False,
            "zorder": 20,
        },
        "medianprops": {
            "linewidth": 1.2 * linewidth,
            "color": "grey",
            "solid_capstyle": "butt",
            "clip_on": False,
            "zorder": 20,
        },
    }
    scatter_kw = {
        "s": 40,
        "marker": ".",
        "alpha": 0.8,
        "zorder": 1,
        "facecolors": "none",
        "clip_on": False,
        "linewidth": 1.2,
    }

    pad = 0.17
    null_distributions = [sensorium_chance]
    bp = ax.boxplot(
        null_distributions,
        positions=x_ticks - pad,
        **null_box_kw,
    )
    max_value = max(max_value, max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    max_value = max(
        max_value, max([max(distribution) for distribution in null_distributions])
    )

    percentages = [sensorium_percentage]
    bp = ax.boxplot(
        percentages,
        positions=x_ticks + pad,
        **percentage_box_kw,
    )
    max_value = max(max_value, max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    max_value = max(max_value, max([np.max(percentage) for percentage in percentages]))

    for i, (percentage, color) in enumerate(zip(percentages, ["limegreen", "black"])):
        ax.scatter(
            [x_ticks[i] + pad] * len(percentage),
            percentage,
            edgecolors=color,
            **scatter_kw,
        )

    for i, (percentage, color) in enumerate(zip(null_distributions, ["grey", "grey"])):
        ax.scatter(
            [x_ticks[i] - pad] * len(percentage),
            percentage,
            edgecolors=color,
            **scatter_kw,
        )

    max_value = np.ceil(max_value / 10) * 10
    max_value = max(max_value, 20)

    xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.4]
    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=[r"$\it{in\;silico}$"],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )
    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[y_ticks[0], y_ticks[-1]],
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-6,
        linespacing=0.9,
    )
    if max_value > 10:
        ax.yaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, minor_length=plot.PARAMS_LENGTH)
    sns.despine(ax=ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            label=label,
            linestyle="-",
            linewidth=linewidth,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        for label, color in [("Shuffle", "grey"), ("Data", "black")]
    ]
    legend_kw = {
        "loc": "upper left",
        "ncols": 1,
        "fontsize": TICK_FONTSIZE - 1,
        "frameon": False,
        "title_fontsize": TICK_FONTSIZE - 1,
        "handletextpad": 0.2,
        "handlelength": 0.75,
        "labelspacing": 0.15,
        "columnspacing": 0.7,
        "borderpad": 0,
        "borderaxespad": 0,
        "alignment": "left",
    }
    legend = ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(xlim[0] + 0.05, 1.03 * max_value),
        bbox_transform=ax.transData,
        **legend_kw,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    ax.add_artist(legend)

    ax.set_axisbelow(b=True)

    plot.save_figure(figure=figure, filename=filename, layout="constrained", dpi=DPI)


def plot_flip_neuron_percentage(
    in_silico_df: pd.DataFrame,
    in_vivo_df: pd.DataFrame,
    filename: Path,
):
    in_vivo_percentage = get_flip_neuron_percentage(df=in_vivo_df)
    print(
        f"\nIn vivo (Rochefort Lab) real percentages: {in_vivo_percentage}\n"
        f"\tAverage (%): {np.mean(in_vivo_percentage):.03f} +- {sem(in_vivo_percentage):.03f}"
    )
    in_vivo_chance = flip_neuron_shuffle_test(df=in_vivo_df)
    print(f"In vivo (Rochefort Lab) chance percentages: {in_vivo_chance}\n")

    in_silico_percentage = get_flip_neuron_percentage(df=in_silico_df)
    print(
        f"In silico (Rochefort Lab) real percentages: {in_silico_percentage}\n"
        f"\tAverage (%): {np.mean(in_silico_percentage):.03f} +- {sem(in_silico_percentage):.03f}"
    )
    in_silico_chance = flip_neuron_shuffle_test(df=in_silico_df)
    print(f"In silico (Rochefort Lab) chance percentages: {in_silico_chance}\n")

    x_ticks = np.array([0, 1], dtype=np.float32)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.3),
        gridspec_kw={"wspace": 0.0, "hspace": 0},
        dpi=DPI,
    )

    linewidth = 1.2
    max_value = 0
    box_width = 0.2
    percentage_box_kw = {
        "notch": False,
        "vert": True,
        "widths": box_width,
        "showfliers": False,
        "showmeans": False,
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
            "markersize": 4.5,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 1,
            "clip_on": False,
            "zorder": 20,
        },
        "medianprops": {
            "linewidth": 1.2 * linewidth,
            "color": "black",
            "solid_capstyle": "butt",
            "clip_on": False,
            "zorder": 20,
        },
    }
    null_box_kw = {
        "notch": False,
        "vert": True,
        "widths": box_width,
        "showfliers": False,
        "showmeans": False,
        "boxprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
            "color": "grey",
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
            "color": "grey",
        },
        "whiskerprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
            "color": "grey",
        },
        "meanprops": {
            "markersize": 4.5,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 1,
            "clip_on": False,
            "zorder": 20,
        },
        "medianprops": {
            "linewidth": 1.2 * linewidth,
            "color": "grey",
            "solid_capstyle": "butt",
            "clip_on": False,
            "zorder": 20,
        },
    }
    scatter_kw = {
        "s": 40,
        "marker": ".",
        "alpha": 0.8,
        "zorder": 1,
        "facecolors": "none",
        "clip_on": False,
        "linewidth": 1.2,
    }

    pad = 0.17
    null_distributions = [in_silico_chance, in_vivo_chance]
    bp = ax.boxplot(
        null_distributions,
        positions=x_ticks - pad,
        **null_box_kw,
    )
    max_value = max(max_value, max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    max_value = max(
        max_value, max([max(distribution) for distribution in null_distributions])
    )

    percentages = [in_silico_percentage, in_vivo_percentage]
    bp = ax.boxplot(
        percentages,
        positions=x_ticks + pad,
        **percentage_box_kw,
    )
    max_value = max(max_value, max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    max_value = max(max_value, max([np.max(percentage) for percentage in percentages]))

    for i, (percentage, color) in enumerate(zip(percentages, ["limegreen", "black"])):
        ax.scatter(
            [x_ticks[i] + pad] * len(percentage),
            percentage,
            edgecolors=color,
            **scatter_kw,
        )

    for i, (percentage, color) in enumerate(zip(null_distributions, ["grey", "grey"])):
        ax.scatter(
            [x_ticks[i] - pad] * len(percentage),
            percentage,
            edgecolors=color,
            **scatter_kw,
        )

    # connect in silico and in vivo Rochefort Lab FOVs
    for i in range(len(in_silico_percentage)):
        ax.plot(
            [x_ticks[0] + pad, x_ticks[1] + pad],
            [in_silico_percentage[i], in_vivo_percentage[i]],
            linewidth=1,
            alpha=0.4,
            zorder=10,
            color="green",
        )

    max_value = np.ceil(max_value / 10) * 10
    max_value = max(max_value, 40)

    xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.4]
    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=[r"$\it{in\;silico}$", r"$\it{in\;vivo}$"],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )
    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[y_ticks[0], y_ticks[-1]],
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-6,
        linespacing=0.9,
    )
    if max_value > 10:
        ax.yaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, minor_length=plot.PARAMS_LENGTH)
    sns.despine(ax=ax)

    unit = 0.08 * max_value
    p_value = stats.wilcoxon(
        in_silico_percentage, in_vivo_percentage, alternative="two-sided"
    ).pvalue
    print(f"\nRochefort Lab in silico vs in vivo p-value: {p_value:.03e}")
    plot.add_p_value(
        ax=ax,
        x0=x_ticks[0] + pad,
        x1=x_ticks[1] + pad,
        y=1.05 * max_value,
        p_value=p_value,
        fontsize=LABEL_FONTSIZE,
        tick_length=0.4 * unit,
        tick_linewidth=1,
        text_pad=0.7 * unit,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            label=label,
            linestyle="-",
            linewidth=linewidth,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        for label, color in [("Shuffle", "grey"), ("Data", "black")]
    ]
    legend_kw = {
        "loc": "upper left",
        "ncols": 1,
        "fontsize": TICK_FONTSIZE - 1,
        "frameon": False,
        "title_fontsize": TICK_FONTSIZE - 1,
        "handletextpad": 0.2,
        "handlelength": 0.75,
        "labelspacing": 0.15,
        "columnspacing": 0.7,
        "borderpad": 0,
        "borderaxespad": 0,
        "alignment": "left",
    }
    legend = ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(xlim[0] + 0.12, 1.03 * max_value),
        bbox_transform=ax.transData,
        **legend_kw,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    ax.add_artist(legend)

    # ax.text(
    #     x=x_ticks[0],
    #     y=1.05 * max_value,
    #     s="Sensorium",
    #     fontsize=TICK_FONTSIZE,
    #     va="bottom",
    #     ha="center",
    # )
    #
    # ax.text(
    #     x=(x_ticks[1] + x_ticks[2]) / 2,
    #     y=1.05 * max_value,
    #     s="Rochefort Lab",
    #     fontsize=TICK_FONTSIZE,
    #     va="bottom",
    #     ha="center",
    # )
    ax.set_axisbelow(b=True)

    plot.save_figure(figure=figure, filename=filename, layout="constrained", dpi=DPI)


def main():
    in_vivo_df = pd.read_parquet("in_vivo_rochefort_lab.parquet")
    in_silico_df = pd.read_parquet("in_silico_rochefort_lab.parquet")
    sensorium_df = pd.read_parquet("in_silico_sensorium.parquet")

    print("\nPlot in vivo result")
    plot_dir = PLOT_DIR / "in_vivo_rochefort_lab"
    plot_contextual_modulation_index(
        df=in_vivo_df,
        filename=plot_dir / f"recorded_cmi.{FORMAT}",
    )
    for contrast_type in ["high_contrast", "low_contrast"]:
        plot_iso_v_cross_scatter(
            df=in_vivo_df,
            model_name="recorded",
            contrast_type=contrast_type,
            filename=plot_dir / f"{contrast_type}_iso_cross_recorded_scatter.jpg",
            plot_flip_neurons_in_red=True,
        )
    plot_box_plot(
        df=in_vivo_df,
        filename=plot_dir / f"in_vivo_flip_neurons.{FORMAT}",
        color="black",
        connect_dots=True,
    )

    print("\nPlot in silico (Rochefort Lab) result")
    plot_dir = PLOT_DIR / "in_silico_rochefort_lab"
    plot_contextual_modulation_index(
        df=in_silico_df,
        filename=plot_dir / f"predicted_cmi.{FORMAT}",
    )
    for contrast_type in ["high_contrast", "low_contrast"]:
        plot_iso_v_cross_scatter(
            df=in_silico_df,
            model_name="predicted",
            contrast_type=contrast_type,
            filename=plot_dir / f"{contrast_type}_iso_cross_predicted_scatter.jpg",
            plot_flip_neurons_in_red=True,
        )
    plot_box_plot(
        df=in_silico_df,
        filename=plot_dir / f"in_silico_rochefort_lab_flip_neurons.{FORMAT}",
        color="limegreen",
        connect_dots=True,
    )

    print("\nPlot in silico (Sensorium) result")
    plot_dir = PLOT_DIR / "in_silico_sensorium"
    plot_contextual_modulation_index(
        df=sensorium_df,
        filename=plot_dir / f"predicted_cmi.{FORMAT}",
    )
    for contrast_type in ["high_contrast", "low_contrast"]:
        plot_iso_v_cross_scatter(
            df=sensorium_df,
            model_name="predicted",
            contrast_type=contrast_type,
            filename=plot_dir / f"{contrast_type}_iso_cross_predicted_scatter.jpg",
            plot_flip_neurons_in_red=True,
            max_value=300,
        )
    plot_box_plot(
        df=sensorium_df,
        filename=plot_dir / f"in_silico_sensorium_flip_neurons.{FORMAT}",
        color="limegreen",
        connect_dots=True,
    )
    plot_sensorium_flip_neuron_percentage(
        sensorium_df=sensorium_df,
        filename=plot_dir / f"in_silico_flip_neuron_percentage.{FORMAT}",
    )

    plot_flip_neuron_percentage(
        in_silico_df=in_silico_df,
        in_vivo_df=in_vivo_df,
        filename=PLOT_DIR / f"rochefort_lab_flip_neuron_percentage.{FORMAT}",
    )

    print(f"\nSaved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
