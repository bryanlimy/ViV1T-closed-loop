from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from scipy.stats import wilcoxon

from viv1t import data
from viv1t.utils import plot
from viv1t.utils.utils import get_reliable_neurons

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches


plot.set_font()

PLOT_DIR = Path("figures") / "static_vs_dynamic_surround_response"

PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2

NSNS_COLOR = "midnightblue"
NSND_COLOR = "dodgerblue"
NSGS_COLOR = "orangered"
NSGD_COLOR = "crimson"


def load_natural_results(
    center_dir: Path,
    surround_dir: Path,
    mouse_id: str,
    neurons: list[int] | np.ndarray,
):
    responses = np.zeros((len(neurons), 3), dtype=np.float32)
    for n, neuron in enumerate(neurons):
        center_response = pd.read_parquet(
            center_dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
        )
        responses[n, 0] = center_response.iloc[0].response
        surround_response = pd.read_parquet(
            surround_dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
        )
        responses[n, 1] = (
            surround_response[surround_response.response_type == "most_exciting"]
            .iloc[0]
            .response
        )
        responses[n, 2] = (
            surround_response[surround_response.response_type == "most_inhibiting"]
            .iloc[0]
            .response
        )
    return responses


def load_generated_results(
    save_dir: Path,
    mouse_id: str,
    neurons: list[int] | np.ndarray,
) -> np.ndarray | None:
    save_dir = save_dir / f"mouse{mouse_id}"
    if not save_dir.is_dir():
        print(f"Cannot find {save_dir}.")
        return
    stim_types = ["center", "most_exciting", "most_inhibiting"]
    responses = np.zeros((len(neurons), len(stim_types)), dtype=np.float32)
    for n, neuron in enumerate(neurons):
        for i, stim_type in enumerate(stim_types):
            ckpt = torch.load(
                save_dir / f"neuron{neuron:04d}" / stim_type / "ckpt.pt",
                map_location="cpu",
            )
            response = ckpt["response"]
            response = response[-(BLANK_SIZE + PATTERN_SIZE) : -BLANK_SIZE]
            responses[n, i] = torch.sum(response).numpy()
    return responses


def add_p_value(
    ax: axes.Axes,
    response1: np.ndarray,
    response2: np.ndarray,
    position1: float,
    position2: float,
    max_value: float,
) -> float:
    p_value = wilcoxon(response1, response2, alternative="less").pvalue
    p_value = 8 * p_value  # adjust for multiple tests
    text_pad = 0.034 if p_value < 0.05 else 0.032
    plot.add_p_value(
        ax=ax,
        x0=position1,
        x1=position2,
        y=1.04 * max_value,
        p_value=p_value,
        fontsize=TICK_FONTSIZE if p_value < 0.05 else LABEL_FONTSIZE,
        tick_length=0.015 * max_value,
        tick_linewidth=1,
        text_pad=text_pad * max_value,
    )
    return p_value


def plot_surround_inhibitory_inset(
    ax: axes.Axes,
    top: float | np.ndarray,
    left: float | np.ndarray,
    width: float | np.ndarray,
    height: float | np.ndarray,
    nsns_inhibiting: np.ndarray,
    nsnd_inhibiting: np.ndarray,
    nsgs_inhibiting: np.ndarray,
    nsgd_inhibiting: np.ndarray,
    dot_positions: np.ndarray,
    positions: np.ndarray,
    scatter_kw: dict,
    box_kw: dict,
    colors: list[str],
    p_value_y_position: float,
):
    # add zoom in boxplot for surround inhibitory with log-scale y-axis
    responses = [nsns_inhibiting, nsnd_inhibiting, nsgs_inhibiting, nsgd_inhibiting]
    x1, x2 = positions[0] - 0.2, positions[-1] + 0.2
    y_min = np.min(responses)
    exp_min = int(np.floor(np.log10(y_min)))
    y1 = 10**exp_min
    y_max = np.max(responses)
    exp_max = int(max(np.ceil(np.log10(y_max)), 0))
    y2 = 10**exp_max

    bottom = top - height
    axin = ax.inset_axes(
        bounds=(left, bottom, width, height),
        transform=ax.transData,
    )
    bp = axin.boxplot(responses, positions=positions, **box_kw)
    max_value = np.ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]]))

    # test statistical difference between response pairs
    p_value = add_p_value(
        ax=ax,
        response1=nsnd_inhibiting,
        response2=nsns_inhibiting,
        position1=positions[0],
        position2=positions[1],
        max_value=p_value_y_position,
    )
    print(f"\tnsnd inhibiting vs nsns inhibiting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsgd_inhibiting,
        response2=nsgs_inhibiting,
        position1=positions[2],
        position2=positions[3],
        max_value=p_value_y_position,
    )
    print(f"\tnsgd inhibiting vs nsgs inhibiting p-value: {p_value:.04e}")

    axin.set_xticks([])
    axin.set_xticklabels([])
    axin.set_xlabel("")

    exp = np.linspace(exp_min, exp_max, exp_max - exp_min + 1)[::-1]
    y_ticks = 10**exp
    axin.set_yscale("log")
    plot.set_yticks(
        axis=axin,
        ticks=y_ticks,
        tick_labels=[
            r"$10^{" + str(int(exp[i])) + r"}$" if i % 2 == 0 else ""
            for i in range(len(exp))
        ],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        rotation=90,
    )
    plot.set_ticks_params(axin, length=3, pad=-1, minor_length=2)

    for i, response in enumerate(responses):
        axin.scatter(
            dot_positions[i],
            response,
            edgecolors=colors[i],
            **scatter_kw,
        )

    axin.set_xlim(x1, x2)
    axin.set_ylim(y1, y2)
    axin.yaxis.set_minor_locator(ticker.LogLocator(numticks=999, subs="auto"))

    axin.text(
        x=x1 + 0.03,
        y=y1 + 1e-6 * (y2 - y1),
        s=f"N={nsns_inhibiting.shape[0]}",
        fontsize=TICK_FONTSIZE,
        va="bottom",
        ha="left",
    )

    # Draw zoom in box
    max_value, min_value = max(0.7, 10**exp_max), min(-0.7, 10**exp_min)
    line_kw = {
        "linewidth": 1,
        "color": "black",
        "zorder": 1,
        "alpha": 0.3,
        "clip_on": False,
    }
    box_left = positions[0] - 1.4 * (box_kw["widths"] / 2)
    box_right = positions[-1] + 1.4 * (box_kw["widths"] / 2)
    # # draw left zoom-in line
    ax.plot([box_left, left], [max_value, bottom], **line_kw)
    # draw right zoom-in line
    ax.plot([box_right, left + width], [max_value, bottom], **line_kw)
    # Draw box around original box plots
    xs = [box_left, box_right, box_right, box_left, box_left]
    ys = [min_value, min_value, max_value, max_value, min_value]
    ax.plot(xs, ys, **line_kw)


def stronger(
    responses1: np.ndarray | float, responses2: np.ndarray | float
) -> np.ndarray:
    """return how much stronger/larger is responses1 with respect to responses2"""
    return 100 * (np.mean(responses1) - np.mean(responses2)) / np.mean(responses2)


def print_stats(responses: list[np.ndarray]):
    # report how much stronger responses are from one stimulus to another
    print(
        f"\tmost-exciting generated dynamic > generated static "
        f"{stronger(responses[3], responses[2]):.02f}%"
    )
    print(
        f"\tmost-inhibiting generated dynamic > generated static "
        f"{stronger(responses[7], responses[6]):.02f}%"
    )
    print("\n")


def plot_center_surround_response(
    nsns_response: np.ndarray,
    nsnd_response: np.ndarray,
    nsgs_response: np.ndarray,
    nsgd_response: np.ndarray,
    filename: Path,
):
    static_center_responses = nsns_response[:, 0]

    # normalize all responses by response to static center
    nsns_exciting = nsns_response[:, 1] / static_center_responses
    nsns_inhibiting = nsns_response[:, 2] / static_center_responses

    nsnd_exciting = nsnd_response[:, 1] / static_center_responses
    nsnd_inhibiting = nsnd_response[:, 2] / static_center_responses

    nsgs_exciting = nsgs_response[:, 1] / static_center_responses
    nsgs_inhibiting = nsgs_response[:, 2] / static_center_responses

    nsgd_exciting = nsgd_response[:, 1] / static_center_responses
    nsgd_inhibiting = nsgd_response[:, 2] / static_center_responses

    responses = [
        nsns_exciting,
        nsnd_exciting,
        nsgs_exciting,
        nsgd_exciting,
        nsns_inhibiting,
        nsnd_inhibiting,
        nsgs_inhibiting,
        nsgd_inhibiting,
    ]

    print_stats(responses=responses)

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 2.2),
        dpi=DPI,
    )

    box_width, box_pad = 0.14, 0.05
    linewidth = 1.2

    x_ticks = np.array([1, 2.1], dtype=np.float32)
    positions = [
        x_ticks[0] - 1.5 * (box_width + box_pad),  # nsns exciting
        x_ticks[0] - 0.5 * (box_width + box_pad),  # nsnd exciting
        x_ticks[0] + 0.5 * (box_width + box_pad),  # nsgs exciting
        x_ticks[0] + 1.5 * (box_width + box_pad),  # nsgd exciting
        x_ticks[1] - 1.5 * (box_width + box_pad),  # nsns inhibiting
        x_ticks[1] - 0.5 * (box_width + box_pad),  # nsnd inhibting
        x_ticks[1] + 0.5 * (box_width + box_pad),  # nsgs inhibiting
        x_ticks[1] + 1.5 * (box_width + box_pad),  # nsgd inhibiting
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
    # max_value = np.ceil(max_value / 10) * 10

    # test statistical difference between response pairs\
    p_value = add_p_value(
        ax=ax,
        response1=nsns_exciting,
        response2=nsnd_exciting,
        position1=positions[0],
        position2=positions[1],
        max_value=max_value,
    )
    print(f"\tnsns exciting vs nsnd exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsnd_exciting,
        response2=nsgd_exciting,
        position1=positions[1],
        position2=positions[3],
        max_value=1.06 * max_value,
    )
    print(f"\tnsns exciting vs nsgs exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsgs_exciting,
        response2=nsgd_exciting,
        position1=positions[2],
        position2=positions[3],
        max_value=max_value,
    )
    print(f"\tnsgs exciting vs nsgd exciting p-value: {p_value:.04e}")

    p_value = add_p_value(
        ax=ax,
        response1=nsnd_inhibiting,
        response2=nsns_inhibiting,
        position1=positions[4],
        position2=positions[5],
        max_value=max_value,
    )
    print(f"\tnsnd inhibiting vs nsns inhibiting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsgd_inhibiting,
        response2=nsgs_inhibiting,
        position1=positions[6],
        position2=positions[7],
        max_value=max_value,
    )
    print(f"\tnsgd inhibiting vs nsgs inhibiting p-value: {p_value:.04e}")

    inhibitory_positions = []
    scatter_kw = {
        "s": 10,
        "marker": ".",
        "alpha": 0.3,
        "zorder": 0,
        "facecolors": "none",
        "clip_on": False,
    }
    colors = [NSNS_COLOR, NSND_COLOR, NSGS_COLOR, NSGD_COLOR] * 2
    for i, (position, response) in enumerate(zip(positions, responses)):
        neurons = np.arange(len(response))
        outliers = np.where(response >= max_value)[0]
        inliers = np.setdiff1d(neurons, outliers)
        x = rng.normal(position, 0.02, size=len(response))
        # plot neurons that are within max_value
        if i in (4, 5, 6, 7):
            inhibitory_positions.append(x)
        ax.scatter(
            x[inliers],
            response[inliers],
            edgecolors=colors[i],
            **scatter_kw,
        )
        # plot outlier neurons
        if outliers.size > 0:
            ax.scatter(
                x[outliers],
                np.full(outliers.shape, fill_value=max_value),
                edgecolors=colors[i],
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
        tick_labels=["Most-exciting\nsurround", "Most-inhibiting\nsurround"],
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

    # inset_width = 0.45 * (xlim[1] - xlim[0])
    # inset_height = 0.55 * max_value
    # plot_surround_inhibitory_inset(
    #     ax=ax,
    #     top=max_value,
    #     left=((positions[-4] - positions[-1]) / 2) + positions[-1] - (inset_width / 2),
    #     width=inset_width,
    #     height=inset_height,
    #     nsns_inhibiting=nsns_inhibiting,
    #     nsnd_inhibiting=nsnd_inhibiting,
    #     nsgs_inhibiting=nsgs_inhibiting,
    #     nsgd_inhibiting=nsgd_inhibiting,
    #     dot_positions=inhibitory_positions,
    #     positions=positions[4:],
    #     scatter_kw=scatter_kw,
    #     box_kw=box_kw,
    #     colors=colors[4:],
    #     p_value_y_position=max_value,
    # )

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def process_model(model_name: str, output_dir: Path):
    save_dir = output_dir / "most_exciting_stimulus" / "single_neuron"
    experiment_name = "002_cutoff"
    nsns_responses = []  # natural static center and natural static surround
    nsnd_responses = []  # natural static center and natural dyamic surround
    nsgs_responses = []  # natural static center and generated static surround
    nsgd_responses = []  # natural static center and generated dynamic surround
    for mouse_id in data.SENSORIUM_OLD:
        print(f"Processing mouse {mouse_id}...")
        neurons = sorted(get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id))
        nsns_response = load_natural_results(
            center_dir=save_dir / "natural" / "center" / "static",
            surround_dir=save_dir
            / "natural"
            / "center_surround"
            / "static_center_static_surround",
            mouse_id=mouse_id,
            neurons=neurons,
        )
        nsns_responses.append(nsns_response)
        nsnd_response = load_natural_results(
            center_dir=save_dir / "natural" / "center" / "static",
            surround_dir=save_dir
            / "natural"
            / "center_surround"
            / "static_center_dynamic_surround",
            mouse_id=mouse_id,
            neurons=neurons,
        )
        nsnd_responses.append(nsnd_response)
        nsgs_response = load_generated_results(
            save_dir=save_dir
            / "generated"
            / "center_surround"
            / "natural_center"
            / "static_center_static_surround"
            / experiment_name,
            mouse_id=mouse_id,
            neurons=neurons,
        )
        nsgs_responses.append(nsgs_response)
        nsgd_response = load_generated_results(
            save_dir=save_dir
            / "generated"
            / "center_surround"
            / "natural_center"
            / "static_center_dynamic_surround"
            / experiment_name,
            mouse_id=mouse_id,
            neurons=neurons,
        )
        nsgd_responses.append(nsgd_response)
        plot_center_surround_response(
            nsns_response=nsns_response.copy(),
            nsnd_response=nsnd_response.copy(),
            nsgs_response=nsgs_response.copy(),
            nsgd_response=nsgd_response.copy(),
            filename=PLOT_DIR
            / f"static_vs_dynamic_surround_response_mouse{mouse_id}.{FORMAT}",
        )
    print(f"Combine neurons from all mice")
    nsns_responses = np.concat(nsns_responses)
    nsnd_responses = np.concat(nsnd_responses)
    nsgs_responses = np.concat(nsgs_responses)
    nsgd_responses = np.concat(nsgd_responses)
    plot_center_surround_response(
        nsns_response=nsns_responses.copy(),
        nsnd_response=nsnd_responses.copy(),
        nsgs_response=nsgs_responses.copy(),
        nsgd_response=nsgd_responses.copy(),
        filename=PLOT_DIR / f"static_vs_dynamic_surround_response.{FORMAT}",
    )
    print(f"Saved plots to {PLOT_DIR}.")


def main():
    models = {
        "ViV1T": Path("../../runs/vivit/204_causal_viv1t"),
    }

    for model_name, output_dir in models.items():
        process_model(model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
