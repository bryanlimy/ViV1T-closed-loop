from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from scipy import stats

from viv1t import data
from viv1t.most_exciting_stimulus.cross_session_reliability import filter_neurons
from viv1t.utils import plot

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


VIDEO_IDS = [
    ("CS000", "center", "center"),
    ("CS001", "natural_static", "most_exciting"),
    ("CS002", "natural_static", "most_inhibiting"),
    ("CS003", "natural_dynamic", "most_exciting"),
    ("CS004", "natural_dynamic", "most_inhibiting"),
    ("CS005", "generated_static", "most_exciting"),
    ("CS006", "generated_static", "most_inhibiting"),
    ("CS007", "generated_dynamic", "most_exciting"),
    ("CS008", "generated_dynamic", "most_inhibiting"),
]


def load_recorded_data(
    data_dir: Path, output_dir: Path, mouse_id: str
) -> (pd.DataFrame, np.ndarray):
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

    neurons = np.where(~np.any(np.isnan(responses), axis=(1, 2)))[0]
    responses_ = np.full(
        shape=(responses.shape[0], len(VIDEO_IDS)),
        fill_value=np.nan,
        dtype=np.float32,
    )

    for i in range(len(VIDEO_IDS)):
        indexes = np.where(video_ids == VIDEO_IDS[i][0])[0]
        assert len(indexes) >= 5
        # select response during presentation window
        response = responses[:, indexes, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
        # sum response over presentation window
        response = np.sum(response, axis=2)
        # average response over repeats
        response = np.mean(response, axis=1)
        responses_[:, i] = response

    # normalize all response by response to center
    responses_ = responses_ / responses_[:, 0][:, None]
    # average response over population
    responses_ = np.nanmean(responses_, axis=0)

    df = pd.DataFrame(
        {
            "stimulus_type": [VIDEO_IDS[i][1] for i in range(len(VIDEO_IDS))],
            "response_type": [VIDEO_IDS[i][2] for i in range(len(VIDEO_IDS))],
            "response": responses_.tolist(),
        }
    )
    return df, neurons


def load_predicted_result(
    output_dir: Path,
    mouse_id: str,
    neurons: np.ndarray,
    experiment_name: str,
) -> pd.DataFrame:
    blank_size = (data.MAX_FRAME - PATTERN_SIZE) // 2
    start, end = -(blank_size + PATTERN_SIZE), -blank_size
    save_dir = output_dir / "most_exciting_stimulus" / "population"
    responses = np.full(
        shape=(len(neurons), len(VIDEO_IDS)), fill_value=np.nan, dtype=np.float32
    )
    i = 0

    # load center response
    ckpt = torch.load(
        save_dir
        / "generated"
        / "center_surround"
        / "natural_center"
        / f"static_center_static_surround"
        / experiment_name
        / f"mouse{mouse_id}"
        / "center"
        / "ckpt.pt",
        map_location="cpu",
    )
    response = ckpt["response"][neurons, start:end].numpy()
    # sum response over presentation
    response = np.sum(response, axis=1)
    responses[:, i] = response
    i += 1
    # load natural surround
    for stimulus_type in ["natural_static", "natural_dynamic"]:
        for response_type in ["most_exciting", "most_inhibiting"]:
            df = pd.read_parquet(
                save_dir
                / "natural"
                / "center_surround"
                / f"static_center_{'static' if 'static' in stimulus_type else 'dynamic'}_surround"
                / f"mouse{mouse_id}.parquet"
            )
            df = df[(df.neuron.isin(neurons)) & (df.response_type == response_type)]
            df.sort_values(by="neuron", ascending=True)
            response = df.response.values
            responses[:, i] = response
            i += 1
    # load generated surround
    for stimulus_type in ["generated_static", "generated_dynamic"]:
        for response_type in ["most_exciting", "most_inhibiting"]:
            ckpt = torch.load(
                save_dir
                / "generated"
                / "center_surround"
                / "natural_center"
                / f"static_center_{'static' if 'static' in stimulus_type else 'dynamic'}_surround"
                / experiment_name
                / f"mouse{mouse_id}"
                / response_type
                / "ckpt.pt",
                map_location="cpu",
            )
            response = ckpt["response"][neurons, start:end].numpy()
            response = np.sum(response, axis=1)
            responses[:, i] = response
            i += 1

    # normalize all response by response to center
    responses = responses / responses[:, 0][:, None]
    # average response over population
    responses = np.nanmean(responses, axis=0)

    df = pd.DataFrame(
        {
            "stimulus_type": [VIDEO_IDS[i][1] for i in range(len(VIDEO_IDS))],
            "response_type": [VIDEO_IDS[i][2] for i in range(len(VIDEO_IDS))],
            "response": responses.tolist(),
        }
    )
    return df


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


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
    test_method: str = "wilcoxon",
) -> float:
    match test_method:
        case "wilcoxon":
            p_value = stats.wilcoxon(
                response1, response2, alternative=alternative
            ).pvalue
        case "ttest_rel":
            p_value = stats.ttest_rel(
                response1, response2, alternative=alternative
            ).pvalue
        case _:
            raise NotImplementedError
    p_value = num_compare * p_value  # adjust for multiple tests
    if unit is None:
        unit = 0.01 * max_value
    plot.add_p_value(
        ax=ax,
        x0=position1,
        x1=position2,
        y=max_value,
        p_value=p_value,
        fontsize=LABEL_FONTSIZE,
        tick_length=0.3 * unit,
        tick_linewidth=1,
        text_pad=(0.6 * unit) if p_value < 0.05 else 0.55 * unit,
    )
    return p_value


def plot_center_surround_response(
    df: pd.DataFrame, filename: Path, color: str = "black"
):
    natural_image = df[
        (df.response_type == "most_exciting") & (df.stimulus_type == "natural_static")
    ].response.values

    natural_video = df[
        (df.response_type == "most_exciting") & (df.stimulus_type == "natural_dynamic")
    ].response.values

    generated_image = df[
        (df.response_type == "most_exciting") & (df.stimulus_type == "generated_static")
    ].response.values

    generated_video = df[
        (df.response_type == "most_exciting")
        & (df.stimulus_type == "generated_dynamic")
    ].response.values

    responses = [natural_image, natural_video, generated_image, generated_video]

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.4),
        dpi=DPI,
    )

    bar_width = 0.4
    linewidth = 1.2
    linestyle = "-"
    box_width = 0.35
    x_ticks = np.array([1, 2, 3, 4], dtype=np.float32)

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
    max_value = np.ceil(np.max(responses))
    max_value = max(max_value, 3)

    # test statistical difference between response pairs
    unit = 0.1 * max_value
    num_compare = 1
    alternative = "less"
    test_method = "ttest_rel"
    p_value = add_p_value(
        ax=ax,
        response1=natural_image,
        response2=natural_video,
        position1=x_ticks[0],
        position2=x_ticks[1],
        max_value=1.02 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
        test_method=test_method,
    )
    print(f"\tNatural image vs natural video p-value: {p_value:.03f}")
    p_value = add_p_value(
        ax=ax,
        response1=generated_image,
        response2=generated_video,
        position1=x_ticks[2],
        position2=x_ticks[3],
        max_value=1.02 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
        test_method=test_method,
    )
    print(f"\tGenerated image vs generated video p-value: {p_value:.03f}")

    p_value = add_p_value(
        ax=ax,
        response1=natural_image,
        response2=generated_image,
        position1=x_ticks[0],
        position2=x_ticks[2],
        max_value=0.85 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
        test_method=test_method,
    )
    print(f"\tNatural image vs generated image p-value: {p_value:.03f}")

    p_value = add_p_value(
        ax=ax,
        response1=natural_video,
        response2=generated_video,
        position1=x_ticks[1],
        position2=x_ticks[3],
        max_value=0.95 * max_value,
        unit=unit,
        num_compare=num_compare,
        alternative=alternative,
        test_method=test_method,
    )
    print(f"\tNatural video vs generated video p-value: {p_value:.03f}")

    scatter_kw = {
        "s": 30,
        "marker": ".",
        "alpha": 0.8,
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
        ax.scatter(x[inliers], response[inliers], **scatter_kw)
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
        "alpha": 0.25,
        "zorder": 0,
        "clip_on": False,
    }
    for n in range(df.mouse.nunique()):
        for index in [(0, 1), (2, 3)]:
            ax.plot(
                [x_ticks[index[0]], x_ticks[index[1]]],
                [responses[i][n] for i in index],
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
        y=0.95,
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
        tick_labels=[
            "Nat.\nimage",
            "Nat.\nvideo",
            "Gen.\nimage",
            "Gen.\nvideo",
        ],
        tick_fontsize=TICK_FONTSIZE,
        # label="Most-exciting surround",
        label_fontsize=TICK_FONTSIZE,
        linespacing=0.9,
        label_pad=1,
    )
    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[y_ticks[0], y_ticks[-1]],
        label="Sum avg.\nnorm. " + r"$\Delta$F/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        linespacing=1.2,
    )
    if y_ticks[-1] < 100:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, minor_length=3)
    sns.despine(ax=ax)

    # ax.text(
    #     x=x_ticks[0] - 0.25,
    #     y=0.05 * max_value,
    #     s=r"$N_{mice}$=2" + "\n" + r"$N_{scans}$=" + str(df.mouse.nunique()),
    #     ha="left",
    #     va="bottom",
    #     fontsize=TICK_FONTSIZE,
    #     linespacing=1,
    # )

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def compute_gain(df: pd.DataFrame):
    natural_static = df[
        (df.stimulus_type == "natural_static") & (df.response_type == "most_exciting")
    ].response.values
    natural_dynamic = df[
        (df.stimulus_type == "natural_dynamic") & (df.response_type == "most_exciting")
    ].response.values

    generated_static = df[
        (df.stimulus_type == "generated_static") & (df.response_type == "most_exciting")
    ].response.values
    generated_dynamic = df[
        (df.stimulus_type == "generated_dynamic")
        & (df.response_type == "most_exciting")
    ].response.values

    improvement = lambda a, b: np.median(100 * (a - b) / b)

    print(
        f"\tNatural static vs natural dynamic surround: "
        f"{improvement(natural_dynamic, natural_static):.03f}%\n"
        f"\tGenerated static vs generated dynamic surround: "
        f"{improvement(generated_dynamic, generated_static):.03f}%\n"
        f"\tNatural static vs generated static surround: "
        f"{improvement(generated_static, natural_static):.03f}%\n"
        # f"\tNatural static vs generated dynamic surround: "
        # f"{improvement(generated_dynamic, natural_static):.03f}%\n"
        f"\tNatural dynamic vs generated dynamic surround: "
        f"{improvement(generated_dynamic, natural_dynamic):.03f}%\n"
    )


def plot_response_gain(
    recorded_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    filename: Path,
):
    # Load recorded data
    recorded_center = recorded_df[recorded_df.stimulus_type == "center"].response.values
    recorded_natural_image = recorded_df[
        (recorded_df.stimulus_type == "natural_static")
        & (recorded_df.response_type == "most_exciting")
    ].response.values
    recorded_natural_video = recorded_df[
        (recorded_df.stimulus_type == "natural_dynamic")
        & (recorded_df.response_type == "most_exciting")
    ].response.values
    recorded_generated_image = recorded_df[
        (recorded_df.stimulus_type == "generated_static")
        & (recorded_df.response_type == "most_exciting")
    ].response.values
    recorded_generated_video = recorded_df[
        (recorded_df.stimulus_type == "generated_dynamic")
        & (recorded_df.response_type == "most_exciting")
    ].response.values

    # Load predicted data
    predicted_center = predicted_df[
        predicted_df.stimulus_type == "center"
    ].response.values
    predicted_natural_image = predicted_df[
        (predicted_df.stimulus_type == "natural_static")
        & (predicted_df.response_type == "most_exciting")
    ].response.values
    predicted_natural_video = predicted_df[
        (predicted_df.stimulus_type == "natural_dynamic")
        & (predicted_df.response_type == "most_exciting")
    ].response.values
    predicted_generated_image = predicted_df[
        (predicted_df.stimulus_type == "generated_static")
        & (predicted_df.response_type == "most_exciting")
    ].response.values
    predicted_generated_video = predicted_df[
        (predicted_df.stimulus_type == "generated_dynamic")
        & (predicted_df.response_type == "most_exciting")
    ].response.values

    improvement = lambda a, b: 100 * (a - b) / b

    recorded_gain = [
        improvement(recorded_natural_image, recorded_center),
        improvement(recorded_natural_video, recorded_center),
        improvement(recorded_generated_image, recorded_center),
        improvement(recorded_generated_video, recorded_center),
    ]
    predicted_gain = [
        improvement(predicted_natural_image, predicted_center),
        improvement(predicted_natural_video, predicted_center),
        improvement(predicted_generated_image, predicted_center),
        improvement(predicted_generated_video, predicted_center),
    ]

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.4),
        dpi=DPI,
    )

    bar_width = box_width = 0.3
    bar_pad = 0.1 * bar_width
    linewidth = 1.2

    x_ticks = np.array([0, 1, 2, 3], dtype=np.float32)
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
    min_value = int(20 * np.floor(min_value / 20))
    max_value = int(20 * np.ceil(max_value / 20))

    unit = 0.1 * (max_value - min_value)
    num_compare = 1
    test_method = "ttest_rel"
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[0],
        response2=recorded_gain[0],
        position1=predicted_ticks[0],
        position2=recorded_ticks[0],
        max_value=1.02 * max_value,
        unit=unit,
        num_compare=num_compare,
        test_method=test_method,
    )
    print(f"\tPredicted vs recorded natural image gain p-value: {p_value:.03f}")
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[1],
        response2=recorded_gain[1],
        position1=predicted_ticks[1],
        position2=recorded_ticks[1],
        max_value=1.02 * max_value,
        unit=unit,
        num_compare=num_compare,
        test_method=test_method,
    )
    print(f"\tPredicted vs recorded natural video gain p-value: {p_value:.03f}")
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[2],
        response2=recorded_gain[2],
        position1=predicted_ticks[2],
        position2=recorded_ticks[2],
        max_value=1.02 * max_value,
        unit=unit,
        num_compare=num_compare,
        test_method=test_method,
    )
    print(f"\tPredicted vs recorded generated image gain p-value: {p_value:.03f}")
    p_value = add_p_value(
        ax=ax,
        response1=predicted_gain[3],
        response2=recorded_gain[3],
        position1=predicted_ticks[3],
        position2=recorded_ticks[3],
        max_value=1.02 * max_value,
        unit=unit,
        num_compare=num_compare,
        test_method=test_method,
    )
    print(f"\tPredicted vs recorded generated video gain p-value: {p_value:.03f}")

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
        tick_labels=["Nat.\nimage", "Nat.\nvideo", "Gen.\nimage", "Gen.\nvideo"],
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
        label=r"$\Delta$F/F" + " % increase\nover nat. centre",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-12,
        linespacing=0.8,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(20))
    plot.set_ticks_params(ax, length=3, minor_length=2.5, linewidth=1)
    bottom = False
    if min_value < 0:
        ax.tick_params(axis="x", which="both", length=0)
        bottom = True
    sns.despine(ax=ax, bottom=bottom)

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def main():
    recorded_df, predicted_df = [], []
    for mouse_id, day_name, output_dir_name, experiment_name in [
        (
            "K",
            "day2",
            "003_causal_viv1t_finetune",
            "001_cutoff",
        ),
        (
            "L",
            "day2",
            "015_causal_viv1t_FOV2_finetune",
            "001_cutoff_natural_init",
        ),
        # (
        #     "L",
        #     "day3",
        #     "015_causal_viv1t_FOV2_finetune",
        #     "003_cutoff_natural_init_200_steps",
        # ),
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
        recorded, recorded_neurons = load_recorded_data(
            data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "artificial_movies",
            output_dir=OUTPUT_DIR / output_dir_name,
            mouse_id=mouse_id,
        )
        recorded.insert(loc=0, column="mouse", value=f"{mouse_id}_{day_name}")
        recorded_df.append(recorded)

        predicted = load_predicted_result(
            output_dir=OUTPUT_DIR / output_dir_name,
            mouse_id=mouse_id,
            neurons=recorded_neurons,
            experiment_name=experiment_name,
        )
        predicted.insert(loc=0, column="mouse", value=f"{mouse_id}_{day_name}")
        predicted_df.append(predicted)

    print(f"Plot in vivo result")
    recorded_df = pd.concat(recorded_df, ignore_index=True)
    plot_center_surround_response(
        df=recorded_df,
        filename=PLOT_DIR / f"in_vivo_static_vs_dynamic_surround.{FORMAT}",
        color="black",
    )

    print(f"\nPlot in silico result")
    predicted_df = pd.concat(predicted_df, ignore_index=True)
    plot_center_surround_response(
        df=predicted_df,
        filename=PLOT_DIR / f"in_silico_static_vs_dynamic_surround.{FORMAT}",
        color="limegreen",
    )

    print(f"\nCompare predicted responses")
    compute_gain(df=predicted_df)

    print(f"Compare recorded responses")
    compute_gain(df=recorded_df)

    print(f"\nPlot predicted vs recorded gains")
    plot_response_gain(
        recorded_df=recorded_df,
        predicted_df=predicted_df,
        filename=PLOT_DIR / f"static_vs_dynamic_response_gain.{FORMAT}",
    )

    print(f"Saved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
