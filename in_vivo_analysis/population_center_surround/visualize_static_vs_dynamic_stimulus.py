from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib import axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import sem

from viv1t import data
from viv1t.utils import plot
from viv1t.utils.stimulus import load_population_RF_center

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

plot.set_font()

PLOT_DIR = Path("figures")
DATA_DIR = Path("../../data")
OUTPUT_DIR = Path("../../runs/rochefort-lab/vivit")

BLANK_SIZE = 15
PATTERN_SIZE = 30
PADDING = 10  # number of frames to include before and after stimulus presentation

MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) // 2

HEIGHT, WIDTH = 36, 64


VIDEO_IDS = [
    ("CS000", "center", "center"),
    ("CS001", "natural_static", "most_exciting"),
    # ("CS002", "natural_static", "most_inhibiting"),
    ("CS003", "natural_dynamic", "most_exciting"),
    # ("CS004", "natural_dynamic", "most_inhibiting"),
    ("CS005", "generated_static", "most_exciting"),
    # ("CS006", "generated_static", "most_inhibiting"),
    ("CS007", "generated_dynamic", "most_exciting"),
    # ("CS008", "generated_dynamic", "most_inhibiting"),
]


def load_recorded_responses(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    video_ids = np.load(
        data_dir / "meta" / "trials" / "video_ids.npy", allow_pickle=True
    )
    responses = [
        np.load(data_dir / "data" / "responses" / f"{trial_id}.npy")
        for trial_id in range(len(video_ids))
    ]
    responses = np.stack(responses)
    # reshape responses into blank - presentation - blank blocks
    responses = rearrange(
        responses,
        "trial neuron (block frame) -> neuron trial block frame",
        frame=BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE,
    )
    responses = rearrange(
        responses, "neuron trial block frame -> neuron (trial block) frame"
    )

    neurons = np.where(~np.all(np.isnan(responses), axis=(1, 2)))[0]
    responses = responses[neurons, :, :]

    video_ids = rearrange(video_ids, "trial block -> (trial block)")
    assert responses.shape[1] == len(video_ids)
    num_repeats = 10
    responses_ = np.full(
        shape=(len(VIDEO_IDS), num_repeats, responses.shape[2]),
        fill_value=np.nan,
        dtype=np.float32,
    )
    for i in range(len(VIDEO_IDS)):
        indexes = np.where(video_ids == VIDEO_IDS[i][0])[0]
        assert len(indexes) >= 5
        response = responses[:, indexes, :]
        # average response over population
        response = np.mean(response, axis=0)
        responses_[i, : len(indexes)] = response
    return responses_, neurons


def load_natural_parquet(
    filename: Path,
    mouse_id: str,
    neurons: np.ndarray,
    response_type: str,
    stimulus_type: str,
) -> (np.ndarray, np.ndarray):
    df = pd.read_parquet(filename)
    df = df[df.neuron.isin(neurons) & (df.response_type == response_type)]
    response = np.stack(df.raw_response.values, dtype=np.float32)
    response = response[:, 5:-5]
    stimulus = np.load(
        filename.parent
        / f"mouse{mouse_id}_{response_type}_{stimulus_type}_stimulus.npy"
    )
    stimulus = stimulus[0, 20:-20]
    return response, stimulus


def load_ckpt_response(filename: Path, neurons: np.ndarray) -> (np.ndarray, np.ndarray):
    ckpt = torch.load(filename, map_location="cpu")
    presentation_mask = ckpt["presentation_mask"].numpy()
    # get stimulus
    video = ckpt["video"].detach().numpy()
    stimulus = video[0, presentation_mask.astype(bool)]
    # get response with 15 frames for blank screens before and after presentation
    response = ckpt["response"].numpy()
    start = np.where(presentation_mask[-response.shape[0] :] == 1)[0][0]
    response = response[neurons, start - BLANK_SIZE : start + PATTERN_SIZE + BLANK_SIZE]
    return response, stimulus


def load_predicted_responses(
    output_dir: Path,
    experiment_name: str,
    mouse_id: str,
    neurons: np.ndarray,
) -> (np.ndarray, np.ndarray):
    save_dir = output_dir / "most_exciting_stimulus" / "population"
    responses = np.zeros(
        (len(VIDEO_IDS), len(neurons), BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE),
        dtype=np.float32,
    )
    visual_stimuli = np.zeros(
        (len(VIDEO_IDS), PATTERN_SIZE, HEIGHT, WIDTH),
        dtype=np.float32,
    )
    # load response to natural center
    responses[0], visual_stimuli[0] = load_natural_parquet(
        filename=save_dir
        / "natural"
        / "center"
        / "static"
        / f"mouse{mouse_id}.parquet",
        mouse_id=mouse_id,
        neurons=neurons,
        response_type="most_exciting",
        stimulus_type="static",
    )
    # load response to most-exciting surround static natural
    responses[1], visual_stimuli[1] = load_natural_parquet(
        filename=save_dir
        / "natural"
        / "center_surround"
        / "static_center_static_surround"
        / f"mouse{mouse_id}.parquet",
        mouse_id=mouse_id,
        neurons=neurons,
        response_type="most_exciting",
        stimulus_type="static",
    )
    # load response to most-exciting surround dynamic natural
    responses[2], visual_stimuli[2] = load_natural_parquet(
        filename=save_dir
        / "natural"
        / "center_surround"
        / "static_center_dynamic_surround"
        / f"mouse{mouse_id}.parquet",
        mouse_id=mouse_id,
        neurons=neurons,
        response_type="most_exciting",
        stimulus_type="dynamic",
    )
    # load response to most-exciting surround static generated
    responses[3], visual_stimuli[3] = load_ckpt_response(
        filename=save_dir
        / "generated"
        / "center_surround"
        / "natural_center"
        / "static_center_static_surround"
        / experiment_name
        / f"mouse{mouse_id}"
        / "most_exciting"
        / "ckpt.pt",
        neurons=neurons,
    )
    # load response to most-exciting surround dynamic generated
    responses[4], visual_stimuli[4] = load_ckpt_response(
        filename=save_dir
        / "generated"
        / "center_surround"
        / "natural_center"
        / "static_center_dynamic_surround"
        / experiment_name
        / f"mouse{mouse_id}"
        / "most_exciting"
        / "ckpt.pt",
        neurons=neurons,
    )
    # average response over population
    responses = np.mean(responses, axis=1)
    # add dimension for repeats
    responses = rearrange(responses, "video frame -> video () frame")
    return responses, visual_stimuli


def plot_stimulus(
    ax: axes.Axes,
    stimulus: np.ndarray,
    figure: plt.Figure,
    imshow_kws: dict,
    dynamic: bool,
    RF_center: tuple[int, int],
    title: str = None,
    y_label: str = None,
):
    # center crop stimulus
    size = 15
    stimulus = stimulus[
        :,
        RF_center[1] - size : RF_center[1] + size,
        RF_center[0] - size : RF_center[0] + size,
    ]
    if not dynamic:
        stimulus = stimulus[stimulus.shape[0] // 2]
    ax.imshow(stimulus[1] if dynamic else stimulus, **imshow_kws)
    ax.set_xticks([])
    ax.set_yticks([])
    plot.set_ticks_params(ax, linewidth=1.2)
    if dynamic:
        # plot stack of frames for dynamic stimuli
        pos = ax.get_position()
        x_offset = 0.1 * pos.width
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
            if i == 0:
                title_ax = axis
        # add an arrow connecting the top left corners of the axes with padding
        ax.annotate(
            "",
            xy=(
                pos.x0 + 2 * x_offset + 0.02 * pos.width,
                pos.y1 + 3 * y_offset + 0.12 * pos.height,
            ),  # arrow end
            xytext=(
                pos.x0 - 0.05 * pos.width,
                pos.y1 + 0.05 * pos.height,
            ),  # arrow start
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.25,head_width=0.17",
                color="black",
                lw=1.4,
            ),
            xycoords="subfigure fraction",
            textcoords="subfigure fraction",
        )
    h, w = stimulus.shape[-2:]
    if title is not None:
        ax.text(
            x=0.58 * w if dynamic else 0.5 * w,
            y=-0.3 * h,
            s=title,
            fontsize=TICK_FONTSIZE,
            linespacing=0.9,
            transform=ax.transData,
            ha="center",
            va="bottom",
        )
    if y_label is not None:
        ax.text(
            x=-0.28 * w,
            y=0.1 * h,
            s=y_label,
            fontsize=LABEL_FONTSIZE,
            linespacing=1,
            transform=ax.transData,
            ha="center",
            va="top",
            rotation=90,
        )


def plot_response(
    ax: axes.Axes,
    recorded_response: np.ndarray,
    predicted_response: np.ndarray,
    x_label: str = None,
    y_label: str = None,
    max_value: float = None,
    show_x_tick_label: bool = False,
    show_y_tick_label: bool = False,
    show_legend: bool = False,
):
    # plot 5 frames before and after presentation
    recorded_response = recorded_response[:, 10:-10]
    predicted_response = predicted_response[:, 10:-10]
    num_frames = recorded_response.shape[1]
    x = np.arange(num_frames)
    x_ticks = np.array([5, 15, 25, 35], dtype=int)
    presentation_mask = np.zeros(num_frames)
    presentation_mask[5 : PATTERN_SIZE + 5] = 1
    if show_x_tick_label:
        x_tick_labels = np.array([0, 10, 20, 30], dtype=int)
    else:
        x_tick_labels = [""] * len(x_ticks)

    linewidth = 1.5
    custom_handles = []
    for color, label_name, response in [
        ("limegreen", "Predicted", predicted_response),
        ("black", "Recorded", recorded_response),
    ]:
        value = np.nanmean(response, axis=0)
        se = sem(response, axis=0, nan_policy="omit") if response.shape[0] > 1 else 0
        if max_value is None:
            max_value = np.max(value + se)
        ax.plot(
            x,
            value,
            linewidth=linewidth,
            color=color,
            zorder=1,
            alpha=0.8,
            clip_on=False,
        )
        ax.fill_between(
            x,
            y1=value - se,
            y2=value + se,
            facecolor=color,
            edgecolor="none",
            linewidth=2,
            alpha=0.2,
            zorder=1,
            clip_on=False,
        )
        custom_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                label=label_name,
                linestyle="-",
                linewidth=linewidth,
                solid_capstyle="butt",
                solid_joinstyle="miter",
            )
        )

    if show_legend:
        legend = ax.legend(
            handles=custom_handles,
            loc="lower right",
            bbox_to_anchor=(len(x), -1.75 * max_value),
            bbox_transform=ax.transData,
            ncols=1,
            fontsize=TICK_FONTSIZE,
            frameon=False,
            handletextpad=0.3,
            handlelength=0.8,
            labelspacing=0.05,
            columnspacing=0,
            borderpad=0,
            borderaxespad=0,
        )
        for lh in legend.legend_handles:
            lh.set_alpha(1)

    ax.set_xlim(0, len(x))
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label=x_label,
        label_pad=-0.5,
    )
    ax.set_ylim(0, max_value)
    y_ticks = np.array([0, max_value])
    if show_y_tick_label:
        y_tick_labels = [0, f"{max_value:.01f}"]
    else:
        y_tick_labels = [""] * len(y_ticks)
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_tick_labels,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        ha="right",
    )
    if y_label is not None:
        ax.text(
            x=1.3,
            y=1.3 * max_value,
            s=y_label,
            va="top",
            ha="left",
            transform=ax.transData,
            fontsize=TICK_FONTSIZE,
        )
    # show presentation period
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
    plot.set_ticks_params(ax, length=2.5, pad=1.5)
    ax.tick_params(axis="y", which="both", pad=1)
    # if exp is not None:
    #     tick = ax.get_yaxis().get_major_ticks()[-1]
    #     tick.set_pad(-0.3)


def plot_mouse(
    visual_stimuli: np.ndarray,
    predicted_responses: np.ndarray,
    recorded_responses: np.ndarray,
    mouse_id: str,
    output_dir: Path,
    filename: Path,
):
    RF_center = load_population_RF_center(output_dir=output_dir, mouse_id=mouse_id)
    RF_center = (int(RF_center[0]), int(RF_center[1]))

    max_value = max(
        np.max(
            np.nanmean(recorded_responses, axis=1)
            + sem(recorded_responses, axis=1, nan_policy="omit")
        ),
        np.max(predicted_responses),
    )
    max_value = np.ceil(max_value * 10) / 10

    imshow_kws = {"aspect": "equal", "cmap": "gray", "vmin": 0, "vmax": 255}

    figure_width, figure_height = PAPER_WIDTH, 1.2
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=DPI)

    column_width = 0.16
    stimulus_height = 1.05 * (
        figure_width * column_width * (HEIGHT / WIDTH) / figure_height
    )
    trace_gap, traces_height = 0.06, 0.12

    top = 0.79

    lefts = np.linspace(0.04, 0.815, num=5)

    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[0],
        right=lefts[0] + column_width,
        top=top,
        bottom=top - stimulus_height,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs[0]),
        stimulus=visual_stimuli[0],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=False,
        title="Natural centre",
        y_label=None,
        RF_center=RF_center,
    )
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[0],
        right=lefts[0] + column_width,
        top=top - stimulus_height - trace_gap,
        bottom=top - stimulus_height - trace_gap - traces_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs[0]),
        recorded_response=recorded_responses[0],
        predicted_response=predicted_responses[0],
        x_label="Time (frame)",
        y_label=r"Population $\Delta$F/F",
        show_x_tick_label=True,
        show_y_tick_label=True,
        max_value=max_value,
    )

    # most-exciting static natural
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[1],
        right=lefts[1] + column_width,
        top=top,
        bottom=top - stimulus_height,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs[0]),
        stimulus=visual_stimuli[1],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=False,
        title="Natural image",
        RF_center=RF_center,
    )
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[1],
        right=lefts[1] + column_width,
        top=top - stimulus_height - trace_gap,
        bottom=top - stimulus_height - trace_gap - traces_height,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs[0]),
        recorded_response=recorded_responses[1],
        predicted_response=predicted_responses[1],
        x_label="",
        y_label="",
        max_value=max_value,
    )

    # most-exciting natural video
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[2],
        right=lefts[2] + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs[0]),
        stimulus=visual_stimuli[2],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=True,
        title="Natural video",
        RF_center=RF_center,
    )
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[2],
        right=lefts[2] + column_width,
        top=top - stimulus_height - trace_gap,
        bottom=top - stimulus_height - trace_gap - traces_height,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs[0]),
        recorded_response=recorded_responses[2],
        predicted_response=predicted_responses[2],
        x_label="",
        y_label="",
        max_value=max_value,
    )

    # most-exciting generated static
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[3],
        right=lefts[3] + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs[0]),
        stimulus=visual_stimuli[3],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=False,
        title="Gen. image",
        RF_center=RF_center,
    )
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[3],
        right=lefts[3] + column_width,
        top=top - stimulus_height - trace_gap,
        bottom=top - stimulus_height - trace_gap - traces_height,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs[0]),
        recorded_response=recorded_responses[3],
        predicted_response=predicted_responses[3],
        x_label="",
        y_label="",
        max_value=max_value,
    )

    # most-exciting generated dynamic
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[4],
        right=lefts[4] + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs[0]),
        stimulus=visual_stimuli[4],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=True,
        title="Gen. video",
        RF_center=RF_center,
    )
    gs = GridSpec(
        nrows=1,
        ncols=1,
        left=lefts[4],
        right=lefts[4] + column_width,
        top=top - stimulus_height - trace_gap,
        bottom=top - stimulus_height - trace_gap - traces_height,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs[0]),
        recorded_response=recorded_responses[4],
        predicted_response=predicted_responses[4],
        x_label="",
        y_label="",
        max_value=max_value,
        show_legend=True,
    )

    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def process_mouse(
    mouse_id: str, data_dir: Path, output_dir: Path, experiment_name: str
):
    print(f"Plot mouse {mouse_id}...")
    plot_dir = PLOT_DIR / "stimulus" / data_dir.name

    recorded_responses, neurons = load_recorded_responses(
        data_dir=data_dir / "artificial_movies",
    )
    predicted_responses, visual_stimuli = load_predicted_responses(
        output_dir=output_dir,
        experiment_name=experiment_name,
        mouse_id=mouse_id,
        neurons=neurons,
    )
    plot_mouse(
        visual_stimuli=visual_stimuli,
        predicted_responses=predicted_responses,
        recorded_responses=recorded_responses,
        mouse_id=mouse_id,
        output_dir=output_dir,
        filename=plot_dir / f"mouse{mouse_id}_static_vs_dynamic_stimulus.{FORMAT}",
    )
    print(f"Saved mouse {mouse_id} plot to {plot_dir}.")


def main():
    for mouse_id, day_name, output_dir_name, experiment_name in [
        # (
        #     "K",
        #     "day2",
        #     "003_causal_viv1t_finetune",
        #     "001_cutoff",
        # ),
        # (
        #     "L",
        #     "day2",
        #     "015_causal_viv1t_FOV2_finetune",
        #     "001_cutoff_natural_init",
        # ),
        (
            "L",
            "day3",
            "015_causal_viv1t_FOV2_finetune",
            "003_cutoff_natural_init_200_steps",
        ),
        (
            "M",
            "day2",
            "018_causal_viv1t_VIPcre233_FOV1_finetune",
            "001_cutoff_natural_init_200_steps",
        ),
        (
            "N",
            "day2",
            "025_causal_viv1t_VIPcre233_FOV2_finetune",
            "001_cutoff_natural_init_200_steps",
        ),
    ]:
        process_mouse(
            mouse_id=mouse_id,
            data_dir=DATA_DIR / data.MOUSE_IDS[f"{mouse_id}_{day_name}"],
            output_dir=OUTPUT_DIR / output_dir_name,
            experiment_name=experiment_name,
        )


if __name__ == "__main__":
    main()
