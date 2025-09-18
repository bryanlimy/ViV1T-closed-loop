from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from einops import repeat
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from scipy import stats

from viv1t import data
from viv1t.most_exciting_stimulus.cross_session_reliability import filter_neurons
from viv1t.utils import plot
from viv1t.utils import utils

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

PATTERN_SIZE = 30
BLANK_SIZE = 15


VIPcre232_VIDEO_IDS = [
    ("GCS000", "center", "center"),
    ("GCS001", "grating_dynamic", "most_exciting"),
    ("GCS002", "grating_dynamic", "most_inhibiting"),
    ("GCS003", "natural_dynamic", "most_exciting"),
    ("GCS004", "natural_dynamic", "most_inhibiting"),
    ("GCS005", "generated_dynamic", "most_exciting"),  # 100 steps
    ("GCS006", "generated_dynamic", "most_inhibiting"),  # 100 steps
    # ("GCS007", "generated_dynamic", "most_exciting"),  # 1000 steps
    # ("GCS008", "generated_dynamic", "most_inhibiting"),  # 1000 steps
]

VIPcre233_VIDEO_IDS = [
    ("GCS000", "center", "center"),
    ("GCS001", "grating_dynamic", "most_exciting"),
    ("GCS002", "grating_dynamic", "most_inhibiting"),
    ("GCS003", "natural_dynamic", "most_exciting"),
    ("GCS004", "natural_dynamic", "most_inhibiting"),
    ("GCS005", "generated_dynamic", "most_exciting"),
    ("GCS006", "generated_dynamic", "most_inhibiting"),
]


def load_recorded_data(data_dir: Path, output_dir: Path, mouse_id: str) -> pd.DataFrame:
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
    video_ids = rearrange(video_ids, "trial block -> (trial block)")
    assert responses.shape[1] == len(video_ids)

    neurons = utils.get_reliable_neurons(
        output_dir=output_dir, mouse_id=mouse_id, size=25
    )

    match mouse_id:
        case "L":
            video_ids_ = VIPcre232_VIDEO_IDS
        case "M" | "N":
            video_ids_ = VIPcre233_VIDEO_IDS
        case _:
            raise NotImplementedError

    df = {"neuron": [], "stimulus_type": [], "response_type": [], "response": []}
    for neuron in neurons:
        for i in range(len(video_ids_)):
            video_id = f"{video_ids_[i][0]}_N{neuron:03d}"
            indexes = np.where(video_ids == video_id)[0]
            assert len(indexes) >= 5
            # select response during presentation window
            response = responses[
                neuron, indexes, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE
            ]
            # sum response over presentation window
            response = np.sum(response, axis=-1)
            # average response over repeats
            response = np.mean(response, axis=0)
            df["neuron"].append(neuron)
            df["stimulus_type"].append(video_ids_[i][1])
            df["response_type"].append(video_ids_[i][2])
            df["response"].append(response)
    df = pd.DataFrame(df)
    # drop neurons that were not matched from day 1
    df = df.dropna()
    return df


def load_predicted_result(
    output_dir: Path,
    mouse_id: str,
    neurons: list[int] | np.ndarray,
    experiment_name: str,
) -> pd.DataFrame:
    blank_size = (data.MAX_FRAME - PATTERN_SIZE) // 2
    start, end = -(blank_size + PATTERN_SIZE), -blank_size
    save_dir = output_dir / "most_exciting_stimulus" / "single_neuron"
    df = {"neuron": [], "stimulus_type": [], "response_type": [], "response": []}
    for neuron in neurons:
        grating_responses = pd.read_parquet(
            save_dir
            / "gratings"
            / "center_surround"
            / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
        )
        grating_responses = grating_responses[
            (grating_responses.mouse == mouse_id) & (grating_responses.neuron == neuron)
        ]
        # load center response
        center_direction = (
            grating_responses[grating_responses.surround_direction == -1]
            .sort_values(by="response", ascending=False)
            .iloc[0]
            .center_direction
        )
        df["neuron"].append(neuron)
        df["stimulus_type"].append("center")
        df["response_type"].append("center")
        df["response"].append(
            grating_responses[
                (grating_responses.center_direction == center_direction)
                & (grating_responses.surround_direction == -1)
            ]
            .iloc[0]
            .response
        )
        # load grating surround
        surround_gratings = grating_responses[
            (grating_responses.center_direction == center_direction)
            & (grating_responses.surround_direction != -1)
        ].sort_values(by="response", ascending=False)
        for response_type in ["most_exciting", "most_inhibiting"]:
            df["neuron"].append(neuron)
            df["stimulus_type"].append("grating_dynamic")
            df["response_type"].append(response_type)
            df["response"].append(
                surround_gratings.iloc[
                    0 if response_type == "most_exciting" else -1
                ].response
            )
            surround_direction = surround_gratings.iloc[
                0 if response_type == "most_exciting" else -1
            ].surround_direction
        del grating_responses, surround_gratings

        # load natural surround
        for response_type in ["most_exciting", "most_inhibiting"]:
            response = pd.read_parquet(
                save_dir
                / "gratings"
                / "grating_center_natural_surround"
                / "dynamic_center_dynamic_surround"
                / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
            )
            df["neuron"].append(neuron)
            df["stimulus_type"].append("natural_dynamic")
            df["response_type"].append(response_type)
            df["response"].append(
                response[response.response_type == response_type].iloc[0].response
            )
        # load generated surround
        for response_type in ["most_exciting", "most_inhibiting"]:
            ckpt = torch.load(
                save_dir
                / "generated"
                / "center_surround"
                / "grating_center"
                / f"dynamic_center_dynamic_surround"
                / experiment_name
                / f"mouse{mouse_id}"
                / f"neuron{neuron:04d}"
                / response_type
                / "ckpt.pt",
                map_location="cpu",
            )
            response = ckpt["response"][start:end].numpy()
            df["neuron"].append(neuron)
            df["stimulus_type"].append("generated_dynamic")
            df["response_type"].append(response_type)
            df["response"].append(np.sum(response))
    df = pd.DataFrame(df)
    return df


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
        y=max_value,
        p_value=p_value,
        fontsize=LABEL_FONTSIZE,
        tick_length=0.3 * unit,
        tick_linewidth=1,
        text_pad=(0.6 * unit) if p_value < 0.05 else 0.45 * unit,
    )
    return p_value


def plot_center_surround_response(
    df: pd.DataFrame, filename: Path, color: str = "black"
):
    grating_center_response = df[df.response_type == "center"].response.values
    print(
        f"\tNumber of neurons: {len(grating_center_response)}\n"
        f"\tNumber of FOVs: {df.mouse.nunique()}"
    )

    # normalize all responses by response to grating center
    grating_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "grating_dynamic")
        ].response.values
        / grating_center_response
    )

    natural_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "natural_dynamic")
        ].response.values
        / grating_center_response
    )

    generated_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "generated_dynamic")
        ].response.values
        / grating_center_response
    )

    responses = [
        grating_exciting,
        natural_exciting,
        generated_exciting,
    ]

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    box_width = 0.25
    linewidth = 1.2

    x_ticks = np.array([0, 1, 2], dtype=np.float32)

    box_kw = {
        "notch": False,
        "vert": True,
        "widths": box_width,
        "showfliers": False,
        "showmeans": False,
        "boxprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
            "color": "black",
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
            "color": "black",
        },
        "whiskerprops": {
            "linewidth": linewidth,
            "clip_on": False,
            "zorder": 10,
            "color": "black",
        },
        "meanprops": {
            "markersize": 4.5,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 1.0,
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
    bp = ax.boxplot(responses, positions=x_ticks, **box_kw)
    max_value = np.ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    # max_value = np.ceil(max_value / 2) * 2

    # test statistical difference between response pairs
    unit = 0.1 * max_value
    num_compare = 3
    alternative = "less"
    p_value = add_p_value(
        ax=ax,
        response1=grating_exciting,
        response2=natural_exciting,
        position1=x_ticks[0],
        position2=x_ticks[1],
        max_value=1.07 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
    )
    print(f"\tgrating exciting vs natural exciting p-value: {p_value:.03e}")
    p_value = add_p_value(
        ax=ax,
        response1=natural_exciting,
        response2=generated_exciting,
        position1=x_ticks[1],
        position2=x_ticks[2],
        max_value=1.07 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
    )
    print(f"\tnatural exciting vs generated exciting p-value: {p_value:.03e}")
    p_value = add_p_value(
        ax=ax,
        response1=grating_exciting,
        response2=generated_exciting,
        position1=x_ticks[0],
        position2=x_ticks[2],
        max_value=1.15 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
    )
    print(f"\tgrating exciting vs generated exciting p-value: {p_value:.03e}")

    scatter_kw = {
        "s": 20,
        "marker": ".",
        "alpha": 0.5,
        "zorder": 1,
        "edgecolors": color,
        "facecolors": "none",
        "linewidth": 1,
        "clip_on": False,
    }
    for i, (position, response) in enumerate(zip(x_ticks, responses)):
        outliers = np.where(response > max_value)[0]
        inliers = np.where(response <= max_value)[0]
        x = rng.normal(position, 0.0, size=len(response))
        # plot neurons that are within max_value
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
        del x, outliers, inliers

    line_kw = {
        "color": color,
        "linestyle": "-",
        "linewidth": 1,
        "alpha": 0.15,
        "zorder": 0,
        "clip_on": False,
    }
    for i in range(len(grating_center_response)):
        ax.plot(
            x_ticks,
            np.clip(
                [responses[0][i], responses[1][i], responses[2][i]],
                a_min=0,
                a_max=max_value,
            ),
            **line_kw,
        )

    xlim = [x_ticks[0] - 0.4, x_ticks[-1] + 0.4]

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
    ax.text(
        x=xlim[-1],
        y=0.85,
        s="baseline",
        va="top",
        ha="right",
        fontsize=TICK_FONTSIZE - 1,
        color="black",
        alpha=0.5,
    )

    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=["Grating", "Natural", "Gen."],
        tick_fontsize=TICK_FONTSIZE,
        # label="Most-exciting dynamic surround",
        label_fontsize=TICK_FONTSIZE,
        linespacing=0.9,
        label_pad=1,
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
        label_pad=-8,
    )
    if y_ticks[-1] < 100:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, minor_length=3)
    sns.despine(ax=ax)

    # ax.text(
    #     x=x_ticks[0] - 0.15,
    #     y=0.97 * max_value,
    #     s=r"$N_{neurons}$=" + str(len(grating_center_response)),
    #     ha="left",
    #     va="top",
    #     fontsize=TICK_FONTSIZE,
    # )

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


# def compute_gain(df: pd.DataFrame):
#     grating_center = df[df.stimulus_type == "center"].response.values
#
#     grating_surround = df[
#         (df.stimulus_type == "grating_dynamic") & (df.response_type == "most_exciting")
#     ].response.values
#
#     natural_surround = df[
#         (df.stimulus_type == "natural_dynamic") & (df.response_type == "most_exciting")
#     ].response.values
#
#     generated_surround = df[
#         (df.stimulus_type == "generated_dynamic")
#         & (df.response_type == "most_exciting")
#     ].response.values
#
#     # normalize response by response to grating center
#     grating_surround /= grating_center
#     natural_surround /= grating_center
#     generated_surround /= grating_center
#
#     improvement = lambda a, b: np.mean(100 * (a - b) / b)
#
#     print(
#         f"\tNatural vs grating surround: "
#         f"{improvement(natural_surround, grating_surround):.03f}%\n"
#         f"\tGenerated vs natural surround: "
#         f"{improvement(generated_surround, natural_surround):.03f}%\n"
#         f"\tGenerated vs grating surround: "
#         f"{improvement(generated_surround, grating_surround):.03f}%\n"
#     )


def plot_response_gain(
    recorded_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    filename: Path,
):
    # Load recorded data
    recorded_center = recorded_df[recorded_df.stimulus_type == "center"].response.values
    recorded_grating = recorded_df[
        (recorded_df.stimulus_type == "grating_dynamic")
        & (recorded_df.response_type == "most_exciting")
    ].response.values
    recorded_natural = recorded_df[
        (recorded_df.stimulus_type == "natural_dynamic")
        & (recorded_df.response_type == "most_exciting")
    ].response.values
    recorded_generated = recorded_df[
        (recorded_df.stimulus_type == "generated_dynamic")
        & (recorded_df.response_type == "most_exciting")
    ].response.values

    # Load predicted data
    predicted_center = predicted_df[
        predicted_df.stimulus_type == "center"
    ].response.values
    predicted_grating = predicted_df[
        (predicted_df.stimulus_type == "grating_dynamic")
        & (predicted_df.response_type == "most_exciting")
    ].response.values
    predicted_natural = predicted_df[
        (predicted_df.stimulus_type == "natural_dynamic")
        & (predicted_df.response_type == "most_exciting")
    ].response.values
    predicted_generated = predicted_df[
        (predicted_df.stimulus_type == "generated_dynamic")
        & (predicted_df.response_type == "most_exciting")
    ].response.values

    improvement = lambda a, b: 100 * (a - b) / b

    recorded_gain = [
        improvement(recorded_grating, recorded_center),
        improvement(recorded_natural, recorded_center),
        improvement(recorded_generated, recorded_center),
    ]
    predicted_gain = [
        improvement(predicted_grating, predicted_center),
        improvement(predicted_natural, predicted_center),
        improvement(predicted_generated, predicted_center),
    ]

    print(
        f"\nIn silico response gain over center\n"
        f"\tgrating mean: {np.mean(predicted_gain[0]):.0f} median: {np.median(predicted_gain[0]):.0f}\n"
        f"\tnatural mean: {np.mean(predicted_gain[1]):.0f} median: {np.median(predicted_gain[1]):.0f}\n"
        f"\tgenerated mean: {np.mean(predicted_gain[2]):.0f} median: {np.median(predicted_gain[2]):.0f}\n"
    )

    print(
        f"In vivo response gain over center\n"
        f"\tgrating mean: {np.mean(recorded_gain[0]):.0f} median: {np.median(recorded_gain[0]):.0f}\n"
        f"\tnatural mean: {np.mean(recorded_gain[1]):.0f} median: {np.median(recorded_gain[1]):.0f}\n"
        f"\tgenerated mean: {np.mean(recorded_gain[2]):.0f} median: {np.median(recorded_gain[2]):.0f}\n"
    )

    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=((1 / 3) * PAPER_WIDTH, 1.5), dpi=DPI
    )

    bar_width = box_width = 0.24
    bar_pad = 0.15 * bar_width
    linewidth = 1.2

    x_ticks = np.array([0, 1, 2], dtype=np.float32)
    predicted_ticks = x_ticks - (bar_width / 2) - bar_pad
    recorded_ticks = x_ticks + (bar_width / 2) + bar_pad

    min_value = np.inf
    max_value = 0
    for values, positions, color in [
        (predicted_gain, predicted_ticks, "limegreen"),
        (recorded_gain, recorded_ticks, "black"),
    ]:
        box_kw = {
            "notch": False,
            "vert": True,
            "widths": box_width,
            "showfliers": False,
            "showmeans": False,
            "boxprops": {
                "linewidth": linewidth,
                "clip_on": False,
                "zorder": 10,
                "color": color,
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
                "color": color,
            },
            "whiskerprops": {
                "linewidth": linewidth,
                "clip_on": False,
                "zorder": 10,
                "color": color,
            },
            "meanprops": {
                "markersize": 4.5,
                "markerfacecolor": "gold",
                "markeredgecolor": "black",
                "markeredgewidth": 1.0,
                "clip_on": False,
                "zorder": 20,
            },
            "medianprops": {
                "linewidth": 1.2 * linewidth,
                "color": color,
                "solid_capstyle": "butt",
                "clip_on": False,
                "zorder": 20,
            },
        }
        bp = ax.boxplot(values, positions=positions, **box_kw)
        min_value = min(
            min_value, np.floor(min([whi.get_ydata()[1] for whi in bp["whiskers"]]))
        )
        max_value = max(
            max_value, np.ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
        )

    step = 80
    min_value = int(step * np.floor(min_value / step))
    max_value = int(step * np.ceil(max_value / step))

    unit = 0.1 * (max_value - min_value)
    num_compare = 1
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[0],
        response2=recorded_gain[0],
        position1=predicted_ticks[0],
        position2=recorded_ticks[0],
        max_value=1.05 * max_value,
        unit=unit,
        num_compare=num_compare,
    )
    print(f"Predicted vs recorded grating gain p-value: {p_value:.03f}")
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[1],
        response2=recorded_gain[1],
        position1=predicted_ticks[1],
        position2=recorded_ticks[1],
        max_value=1.05 * max_value,
        unit=unit,
        num_compare=num_compare,
    )
    print(f"Predicted vs recorded natural gain p-value: {p_value:.03f}")
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[2],
        response2=recorded_gain[2],
        position1=predicted_ticks[2],
        position2=recorded_ticks[2],
        max_value=1.05 * max_value,
        unit=unit,
        num_compare=num_compare,
    )
    print(f"Predicted vs recorded generated gain p-value: {p_value:.03f}")

    xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.5]

    if min_value < 0:
        ax.axhline(
            y=0,
            xmin=xlim[0],
            xmax=xlim[-1],
            linewidth=1,
            color="black",
            zorder=0,
        )

    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=["Grating", "Natural", "Gen."],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        linespacing=0.85,
        label_pad=1,
    )
    y_ticks = np.array([min_value, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[y_ticks[0], y_ticks[-1]],
        label=r"$\Delta$F/F" + " % increase\nover grat. centre",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-12,
        linespacing=0.8,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(step))
    plot.set_ticks_params(ax, length=3, minor_length=3, linewidth=1)
    bottom = False
    if min_value < 0:
        ax.tick_params(axis="x", which="both", length=0)
        bottom = True
    sns.despine(ax=ax, bottom=bottom)
    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def main():
    recorded_df, predicted_df = [], []
    # mouse L bad neurons
    bad_neurons = [33, 63, 158, 206]
    for mouse_id, day_name, output_dir_name, experiment_name in [
        (
            "L",
            "day3",
            "015_causal_viv1t_FOV2_finetune",
            "001_cutoff_natural_init_100_steps",
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
        recorded = load_recorded_data(
            data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "artificial_movies",
            output_dir=OUTPUT_DIR / output_dir_name,
            mouse_id=mouse_id,
        )

        neurons = recorded.neuron.unique()
        reliable_neurons = filter_neurons(
            day1_data_dir=DATA_DIR / data.MOUSE_IDS[mouse_id],
            day2_data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "natural_movies",
            percentage=0.8,
        )
        neurons = np.intersect1d(neurons, reliable_neurons)

        recorded = recorded[recorded.neuron.isin(neurons)]
        if mouse_id == "L":
            recorded = recorded[~recorded.neuron.isin(bad_neurons)]
            neurons = recorded.neuron.unique()
        recorded.insert(loc=0, column="mouse", value=mouse_id)
        recorded_df.append(recorded)

        predicted = load_predicted_result(
            output_dir=OUTPUT_DIR / output_dir_name,
            mouse_id=mouse_id,
            neurons=neurons,
            experiment_name=experiment_name,
        )
        predicted.insert(loc=0, column="mouse", value=mouse_id)
        predicted_df.append(predicted)

    print("Plot in vivo result")
    recorded_df = pd.concat(recorded_df, ignore_index=True)
    plot_center_surround_response(
        df=recorded_df,
        filename=PLOT_DIR / f"in_vivo_grating_vs_generated_surround.{FORMAT}",
        color="black",
    )

    print("\nPlot in silico result")
    predicted_df = pd.concat(predicted_df, ignore_index=True)
    plot_center_surround_response(
        df=predicted_df,
        filename=PLOT_DIR / f"in_silico_grating_vs_generated_surround.{FORMAT}",
        color="limegreen",
    )

    # print(f"\nCompare predicted responses")
    # compute_gain(df=predicted_df)
    #
    # print(f"Compare recorded responses")
    # compute_gain(df=recorded_df)

    plot_response_gain(
        recorded_df=recorded_df,
        predicted_df=predicted_df,
        filename=PLOT_DIR / f"grating_vs_generated_response_gain.{FORMAT}",
    )

    print(f"\nSaved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
