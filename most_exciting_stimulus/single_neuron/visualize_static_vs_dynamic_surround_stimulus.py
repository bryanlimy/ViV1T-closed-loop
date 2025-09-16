import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import axes
from matplotlib.figure import SubFigure
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot
from viv1t.utils.utils import get_reliable_neurons

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

plot.set_font()

PLOT_DIR = Path("figures") / "static_vs_dynamic_surround_stimulus"


PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2
MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) // 2

PADDING = 10  # number of frames to include before and after stimulus presentation

STATIC_NATURAL_COLOR = "midnightblue"
STATIC_GENERATED_COLOR = "orangered"
DYNAMIC_GENERATED_COLOR = "crimson"


def load_natural_results(
    filename: Path,
    mouse_id: str,
    neurons: list[int] | np.ndarray,
    static_center: bool,
    static_surround: bool,
) -> dict[int, dict[str, dict[str, np.ndarray | None]]]:
    with open(filename, "rb") as file:
        results_ = pickle.load(file)
    if static_center and static_surround:
        results_ = results_["ss"]
    elif static_center and not static_surround:
        results_ = results_["sd"]
    else:
        results_ = results_["dd"]
    stim_types = ["center", "most_exciting", "most_inhibiting"]
    results = {
        n: {k: {"stimulus": None, "response": None} for k in stim_types}
        for n in neurons
    }
    start = -(BLANK_SIZE + PATTERN_SIZE + PADDING)
    end = -(BLANK_SIZE - PADDING)
    for n, neuron in enumerate(neurons):
        for i, stim_type in enumerate(stim_types):
            result = results_[stim_type][mouse_id][neuron]
            results[neuron][stim_type]["response"] = result["response"][start:end]
            results[neuron][stim_type]["stimulus"] = result["video"]
    return results


def load_generated_results(
    save_dir: Path, mouse_id: str, neurons: list[int] | np.ndarray
) -> dict[int, dict[str, dict[str, np.ndarray | None]]]:
    save_dir = save_dir / f"mouse{mouse_id}"
    stim_types = ["center", "most_exciting", "most_inhibiting"]
    results = {
        n: {k: {"stimulus": None, "response": None} for k in stim_types}
        for n in neurons
    }
    start = -(BLANK_SIZE + PATTERN_SIZE + PADDING)
    end = -(BLANK_SIZE - PADDING)
    for n, neuron in enumerate(neurons):
        for i, stim_type in enumerate(stim_types):
            ckpt = torch.load(
                save_dir / f"neuron{neuron:04d}" / stim_type / "ckpt.pt",
                map_location="cpu",
            )
            results[neuron][stim_type]["response"] = (
                ckpt["response"][start:end].to(torch.float32).detach().numpy()
            )
            results[neuron][stim_type]["stimulus"] = (
                ckpt["video"][:, -(BLANK_SIZE + PATTERN_SIZE) : -BLANK_SIZE]
                .to(torch.float32)
                .detach()
                .numpy()
            )
    return results


def plot_stimulus(
    ax: axes.Axes,
    stimulus: np.ndarray,
    figure: SubFigure,
    imshow_kws: dict,
    title: str = None,
    y_label: str = None,
):
    dynamic = len(stimulus.shape) == 3
    ax.imshow(stimulus[0] if dynamic else stimulus, **imshow_kws)
    ax.set_xticks([])
    ax.set_yticks([])
    plot.set_ticks_params(ax, linewidth=1.2)
    if dynamic:
        # plot stack of frames for dynamic stimuli
        pos = ax.get_position()
        x_offset = 0.06 * pos.width
        y_offset = 0.07 * pos.height
        for i in range(2):
            axis = figure.add_axes(
                (
                    pos.x0 + (i + 1) * x_offset,  # left
                    pos.y0 + (i + 1) * y_offset,  # bottom
                    pos.width,  # width
                    pos.height,  # height
                )
            )
            axis.imshow(stimulus[(i + 1) * 5], **imshow_kws)
            axis.set_xticks([])
            axis.set_yticks([])
            plot.set_ticks_params(axis, linewidth=1.2)
            axis.set_zorder(-i - 1)
        # add an arrow connecting the top left corners of the axes with padding
        ax.annotate(
            "",
            xy=(
                pos.x0 + 2 * x_offset + 0.02 * pos.width,
                pos.y1 + 3 * y_offset + 0.1 * pos.height,
            ),  # arrow end
            xytext=(
                pos.x0 - 0.02 * pos.width,
                pos.y1 + 0.04 * pos.height,
            ),  # arrow start
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.25,head_width=0.15",
                color="black",
                lw=1.2,
            ),
            xycoords="subfigure fraction",
            textcoords="subfigure fraction",
        )
    if title is not None:
        h, w = stimulus.shape[1:] if dynamic else stimulus.shape
        ax.text(
            x=0.57 * w if dynamic else 0.5 * w,
            y=-0.16 * h,
            s=title,
            fontsize=LABEL_FONTSIZE,
            linespacing=0.85,
            transform=ax.transData,
            ha="center",
            va="bottom",
        )
    if y_label is not None:
        ax.set_ylabel(
            y_label,
            fontsize=LABEL_FONTSIZE,
            labelpad=1.5,
            linespacing=0.9,
            y=0.05,
            va="bottom",
            ha="left",
        )


def plot_response(
    ax: axes.Axes,
    x: np.ndarray,
    response: np.ndarray,
    presentation_mask: np.ndarray,
    x_ticks: np.ndarray | list,
    x_tick_labels: np.ndarray | list,
    x_label: str = None,
    y_label: str = None,
    color: str = "black",
    max_value: float = None,
):
    if max_value is None:
        max_value = np.max(response)
    ax.plot(
        x,
        response,
        linewidth=2,
        color=color,
        zorder=1,
        clip_on=False,
    )

    ax.set_xlim(0, len(x))
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label=x_label,
        label_pad=0,
    )
    # change different styling of y ticks based on its range
    exp = None
    if max_value >= 0.8:
        max_value = int(np.ceil(max_value))
        max_tick = str(max_value)
    elif 0.1 <= max_value < 0.8:
        max_value = 0.1 * np.ceil(max_value * 10)
        max_tick = f"{max_value:.01f}"
    else:
        exp = int(np.ceil(np.log10(max_value)))
        max_value = 10**exp
        max_tick = r"$10^{" + str(exp) + r"}$"
    ax.set_ylim(0, max_value)
    y_ticks = np.array([0, max_value])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[0, max_tick],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        ha="right",
    )
    if y_label is not None:
        ax.text(
            x=-1,
            y=0.95 * max_value,
            s=y_label,
            va="bottom",
            ha="left",
            transform=ax.transData,
            fontsize=LABEL_FONTSIZE,
        )
    if max_value >= 0.8:
        # set y=1 dotted line
        ax.axhline(
            y=1,
            color="black",
            alpha=0.4,
            linestyle="dotted",
            dashes=(1, 1),
            linewidth=1.5,
            zorder=0,
            clip_on=False,
        )
    # show presentation window
    ax.fill_between(
        x,
        y1=0,
        y2=max_value,
        where=presentation_mask,
        facecolor="#e0e0e0",
        edgecolor="none",
        zorder=-1,
    )
    sns.despine(ax=ax)
    ax.set_axisbelow(False)
    plot.set_ticks_params(ax, pad=1.5)
    if exp is not None:
        tick = ax.get_yaxis().get_major_ticks()[-1]
        tick.set_pad(-0.5)


def plot_neuron(
    nsns_result: dict[str, dict[str, np.ndarray]],
    nsgs_result: dict[str, dict[str, np.ndarray]],
    nsgd_result: dict[str, dict[str, np.ndarray]],
    mouse_id: str,
    neuron: int,
    filename: Path,
) -> None:
    nsns_responses = np.array([v["response"] for k, v in nsns_result.items()])
    nsgs_responses = np.array([v["response"] for k, v in nsgs_result.items()])
    nsgd_responses = np.array([v["response"] for k, v in nsgd_result.items()])

    # normalize responses to static and dynamic stimuli by the max response to
    # static center stimulus
    max_static_center_response = np.max(nsns_responses[0])
    nsns_responses /= max_static_center_response
    nsgs_responses /= max_static_center_response
    nsgd_responses /= max_static_center_response

    x = np.arange(len(nsns_result["center"]["response"]))
    x_ticks = np.linspace(0, len(x), (len(x) // 10) + 1, dtype=int)
    start = PADDING
    end = PATTERN_SIZE + start
    presentation_mask = np.zeros_like(x)
    presentation_mask[start:end] = 1

    imshow_kws = {"aspect": "equal", "cmap": "gray", "vmin": 0, "vmax": 255}

    figure = plt.figure(figsize=((2 / 3) * PAPER_WIDTH, 2.7), dpi=DPI)

    gs = GridSpec(nrows=3, ncols=3, figure=figure)

    left, right = 0.12, 0.89
    height = 0.8  # height of a subplot w.r.t. a cell in GridSpec
    top = 0.76  # top of the cell in the first row in GridSpec
    bottom = 0.07  # bottom of the cell in the second row in the GridSpec
    wspace, hspace = 0, -0.1
    height_ratios = [0.8, 0.15]

    ################################ First row ################################
    adjust_kw = {
        "left": left,
        "bottom": top - height,
        "right": right,
        "top": top,
        "wspace": wspace,
        "hspace": hspace,
    }

    subfigure = figure.add_subfigure(gs[0, 0])
    axes1 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    axes1[0].axis("off")
    axes1[1].axis("off")

    subfigure = figure.add_subfigure(gs[0, 1])
    axes2 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes2[0],
        stimulus=nsns_result["most_exciting"]["stimulus"][0, 0],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title="Most-exciting\nsurround",
        y_label=None,
    )
    plot_response(
        ax=axes2[1],
        x=x,
        response=nsns_responses[1],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=[],
        x_label=None,
        y_label=None,
        color=STATIC_NATURAL_COLOR,
    )

    subfigure = figure.add_subfigure(gs[0, 2])
    axes3 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes3[0],
        stimulus=nsns_result["most_inhibiting"]["stimulus"][0, 0],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title="Most-inhibiting\nsurround",
        y_label=None,
    )
    plot_response(
        ax=axes3[1],
        x=x,
        response=nsns_responses[2],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=[],
        x_label=None,
        y_label=None,
        color=STATIC_NATURAL_COLOR,
    )

    ################################ Second row ################################

    mid_bottom = -0.1
    adjust_kw = {
        "left": left,
        "bottom": mid_bottom,
        "right": right,
        "top": mid_bottom + 0.95,
        "wspace": wspace,
        "hspace": 0.19,
    }
    subfigure = figure.add_subfigure(gs[1, 0])
    axes4 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes4[0],
        stimulus=nsgs_result["center"]["stimulus"][0, 0],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title="Most-exciting\nnatural centre",
        y_label="",
    )
    plot_response(
        ax=axes4[1],
        x=x,
        response=nsgs_responses[0],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=x_ticks,
        x_label="Time (frame)",
        y_label=r"Predicted $\Delta$F/F",
        color="black",
    )

    mid_bottom = 0.0
    adjust_kw = {
        "left": left,
        "bottom": mid_bottom,
        "right": right,
        "top": mid_bottom + height,
        "wspace": wspace,
        "hspace": hspace,
    }
    subfigure = figure.add_subfigure(gs[1, 1])
    axes5 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes5[0],
        stimulus=nsgs_result["most_exciting"]["stimulus"][0, 0],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title=None,
        y_label=None,
    )
    plot_response(
        ax=axes5[1],
        x=x,
        response=nsgs_responses[1],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=[],
        x_label="",
        y_label=None,
        color=STATIC_GENERATED_COLOR,
    )

    subfigure = figure.add_subfigure(gs[1, 2])
    axes6 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes6[0],
        stimulus=nsgs_result["most_inhibiting"]["stimulus"][0, 0],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title=None,
        y_label=None,
    )
    plot_response(
        ax=axes6[1],
        x=x,
        response=nsgs_responses[2],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=[],
        x_label=None,
        y_label=None,
        color=STATIC_GENERATED_COLOR,
    )

    ################################ Third row ################################
    adjust_kw = {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": bottom + height,
        "wspace": wspace,
        "hspace": hspace,
    }

    subfigure = figure.add_subfigure(gs[2, 0])
    axes7 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    axes7[0].axis("off")
    axes7[1].axis("off")

    subfigure = figure.add_subfigure(gs[2, 1])
    axes8 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes8[0],
        stimulus=nsgd_result["most_exciting"]["stimulus"][0, :],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title=None,
        y_label=None,
    )
    plot_response(
        ax=axes8[1],
        x=x,
        response=nsgd_responses[1],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=[],
        x_label=None,
        y_label=None,
        color=DYNAMIC_GENERATED_COLOR,
    )

    subfigure = figure.add_subfigure(gs[2, 2])
    axes9 = subfigure.subplots(nrows=2, ncols=1, height_ratios=height_ratios)
    subfigure.subplots_adjust(**adjust_kw)
    plot_stimulus(
        ax=axes9[0],
        stimulus=nsgd_result["most_inhibiting"]["stimulus"][0, :],
        figure=subfigure,
        imshow_kws=imshow_kws,
        title=None,
        y_label=None,
    )
    plot_response(
        ax=axes9[1],
        x=x,
        response=nsgd_responses[2],
        presentation_mask=presentation_mask,
        x_ticks=x_ticks,
        x_tick_labels=[],
        x_label=None,
        y_label=None,
        color=DYNAMIC_GENERATED_COLOR,
    )
    # # write mouse and neuron information in the bottom left of the figure
    # figure.text(
    #     x=bottom,
    #     y=left,
    #     s=f"Mouse {mouse_id} Neuron {neuron}",
    #     fontsize=LABEL_FONTSIZE,
    #     va="bottom",
    #     ha="left",
    #     transform=figure.transFigure.inverted(),
    # )
    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def process_model(model_name: str, output_dir: Path):
    save_dir = output_dir / "most_exciting_stimulus" / "single_neuron"
    experiment_name = "002_cutoff"
    for mouse_id in data.SENSORIUM_OLD:
        neurons = sorted(get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id))
        nsns_results = load_natural_results(
            filename=save_dir / "natural_center_natural_surround.pkl",
            mouse_id=mouse_id,
            neurons=neurons,
            static_center=True,
            static_surround=True,
        )
        # nsnd_results = load_natural_results()
        nsgs_results = load_generated_results(
            save_dir=save_dir
            / "generated"
            / "center_surround"
            / "natural_center"
            / "static_center_static_surround"
            / experiment_name,
            mouse_id=mouse_id,
            neurons=neurons,
        )
        nsgd_results = load_generated_results(
            save_dir=save_dir
            / "generated"
            / "center_surround"
            / "natural_center"
            / "static_center_dynamic_surround"
            / experiment_name,
            mouse_id=mouse_id,
            neurons=neurons,
        )
        # for neuron in tqdm(sorted(neurons), desc=f"mouse {mouse_id}"):
        #     plot_neuron(
        #         nsns_result=nsns_results[neuron],
        #         nsgs_result=nsgs_results[neuron],
        #         nsgd_result=nsgd_results[neuron],
        #         mouse_id=mouse_id,
        #         neuron=neuron,
        #         filename=PLOT_DIR
        #         / f"mouse{mouse_id}"
        #         / f"static_vs_dynamic_surround_stimulus_mouse{mouse_id}_neuron{neuron:04d}.{FORMAT}",
        #     )
        neuron = 4264
        plot_neuron(
            nsns_result=nsns_results[neuron],
            nsgs_result=nsgs_results[neuron],
            nsgd_result=nsgd_results[neuron],
            mouse_id=mouse_id,
            neuron=neuron,
            filename=PLOT_DIR
            / f"mouse{mouse_id}"
            / f"static_vs_dynamic_surround_stimulus_mouse{mouse_id}_neuron{neuron:04d}.{FORMAT}",
        )
        break

    print(f"Saved result to {PLOT_DIR}.")


def main():
    models = {
        "ViV1T": Path("../../runs/vivit/204_causal_viv1t"),
    }

    for model_name, output_dir in models.items():
        process_model(model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
