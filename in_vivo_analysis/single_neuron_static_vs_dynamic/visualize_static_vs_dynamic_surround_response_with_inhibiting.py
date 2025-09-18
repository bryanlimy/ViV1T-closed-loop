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
from scipy.stats import wilcoxon

from viv1t import data
from viv1t.most_exciting_stimulus.cross_session_reliability import filter_neurons
from viv1t.utils import plot
from viv1t.utils.utils import get_reliable_neurons

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "jpg"
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

    neurons = get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id, size=25)

    df = {"neuron": [], "stimulus_type": [], "response_type": [], "response": []}
    for neuron in neurons:
        for i in range(len(VIDEO_IDS)):
            video_id = f"{VIDEO_IDS[i][0]}_N{neuron:03d}"
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
            df["stimulus_type"].append(VIDEO_IDS[i][1])
            df["response_type"].append(VIDEO_IDS[i][2])
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
        # load center response
        ckpt = torch.load(
            save_dir
            / "generated"
            / "center_surround"
            / "natural_center"
            / f"static_center_static_surround"
            / experiment_name
            / f"mouse{mouse_id}"
            / f"neuron{neuron:04d}"
            / "center"
            / "ckpt.pt",
            map_location="cpu",
        )
        response = ckpt["response"][start:end].numpy()
        df["neuron"].append(neuron)
        df["stimulus_type"].append("center")
        df["response_type"].append("center")
        df["response"].append(np.sum(response))
        # load natural surround
        for stimulus_type in ["natural_static", "natural_dynamic"]:
            for response_type in ["most_exciting", "most_inhibiting"]:
                response = pd.read_parquet(
                    save_dir
                    / "natural"
                    / "center_surround"
                    / f"static_center_{'static' if 'static' in stimulus_type else 'dynamic'}_surround"
                    / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
                )
                df["neuron"].append(neuron)
                df["stimulus_type"].append(stimulus_type)
                df["response_type"].append(response_type)
                df["response"].append(
                    response[response.response_type == response_type].iloc[0].response
                )
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
                    / f"neuron{neuron:04d}"
                    / response_type
                    / "ckpt.pt",
                    map_location="cpu",
                )
                response = ckpt["response"][start:end].numpy()
                df["neuron"].append(neuron)
                df["stimulus_type"].append(stimulus_type)
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
) -> float:
    p_value = wilcoxon(response1, response2, alternative="less").pvalue
    p_value = 6 * p_value  # adjust for multiple tests
    text_pad = 0.04 if p_value < 0.05 else 0.04
    plot.add_p_value(
        ax=ax,
        x0=position1,
        x1=position2,
        y=1.04 * max_value,
        p_value=p_value,
        fontsize=TICK_FONTSIZE if p_value < 0.05 else LABEL_FONTSIZE,
        tick_length=0.02 * max_value,
        tick_linewidth=1,
        text_pad=text_pad * max_value,
    )
    return p_value


def plot_center_surround_response(
    df: pd.DataFrame, filename: Path, color: str = "black"
):
    static_center_responses = df[df.response_type == "center"].response.values

    # normalize all responses by response to static center
    nsns_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "natural_static")
        ].response.values
        / static_center_responses
    )
    nsns_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
            & (df.stimulus_type == "natural_static")
        ].response.values
        / static_center_responses
    )

    nsnd_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "natural_dynamic")
        ].response.values
        / static_center_responses
    )
    nsnd_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
            & (df.stimulus_type == "natural_dynamic")
        ].response.values
        / static_center_responses
    )

    nsgs_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "generated_static")
        ].response.values
        / static_center_responses
    )
    nsgs_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
            & (df.stimulus_type == "generated_static")
        ].response.values
        / static_center_responses
    )

    nsgd_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "generated_dynamic")
        ].response.values
        / static_center_responses
    )
    nsgd_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
            & (df.stimulus_type == "generated_dynamic")
        ].response.values
        / static_center_responses
    )

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

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.8),
        dpi=DPI,
    )

    box_width, box_pad = 0.1, 0.1
    linewidth = 1.2

    x_ticks = np.array([1, 1.9], dtype=np.float32)
    positions = [
        x_ticks[0] - 1.5 * (box_width + box_pad),  # nsns exciting
        x_ticks[0] - 0.5 * (box_width + box_pad),  # nsnd exciting
        x_ticks[0] + 0.5 * (box_width + box_pad),  # nsgs exciting
        x_ticks[0] + 1.5 * (box_width + box_pad),  # nsgd exciting
        x_ticks[1] - 1.5 * (box_width + box_pad),  # nsns inhibiting
        x_ticks[1] - 0.5 * (box_width + box_pad),  # nsnd inhibiting
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
            "color": "royalblue",
            "solid_capstyle": "projecting",
            "clip_on": False,
            "zorder": 20,
        },
    }
    bp = ax.boxplot(responses, positions=positions, **box_kw)
    max_value = np.ceil(max([whi.get_ydata()[1] for whi in bp["whiskers"]]))
    # max_value = np.ceil(np.max(responses))

    # test statistical difference between response pairs
    p_value = add_p_value(
        ax=ax,
        response1=nsns_exciting,
        response2=nsnd_exciting,
        position1=positions[0],
        position2=positions[1],
        max_value=1.01 * max_value,
    )
    print(f"\tnsns exciting vs nsnd exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsnd_exciting,
        response2=nsgd_exciting,
        position1=positions[1],
        position2=positions[3],
        max_value=1.08 * max_value,
    )
    print(f"\tnsns exciting vs nsgs exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsgs_exciting,
        response2=nsgd_exciting,
        position1=positions[2],
        position2=positions[3],
        max_value=1.01 * max_value,
    )
    print(f"\tnsgs exciting vs nsgd exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsnd_inhibiting,
        response2=nsns_inhibiting,
        position1=positions[4],
        position2=positions[5],
        max_value=1.01 * max_value,
    )
    print(f"\tnsnd inhibiting vs nsns inhibiting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsgd_inhibiting,
        response2=nsnd_inhibiting,
        position1=positions[5],
        position2=positions[7],
        max_value=1.08 * max_value,
    )
    print(f"\tnsgs inhibiting vs nsns inhibiting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=nsgd_inhibiting,
        response2=nsgs_inhibiting,
        position1=positions[6],
        position2=positions[7],
        max_value=1.01 * max_value,
    )
    print(f"\tnsgd inhibiting vs nsgs inhibiting p-value: {p_value:.04e}")

    scatter_kw = {
        "s": 15,
        "marker": ".",
        "alpha": 0.5,
        "zorder": 1,
        "edgecolors": color,
        "facecolors": "none",
        "linewidth": 0.8,
        "clip_on": False,
    }
    for i, (position, response) in enumerate(zip(positions, responses)):
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
    for n in range(len(static_center_responses)):
        for index in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            ax.plot(
                [positions[i] for i in index],
                np.clip(
                    [responses[i][n] for i in index],
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

    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=positions,
        tick_labels=["Nat.\nimg", "Nat.\nvid.", "Gen.\nimg", "Gen.\nvid."] * 2,
        tick_fontsize=TICK_FONTSIZE,
        linespacing=0.9,
        ha="center",
        # rotation=45,
        # va="top",
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
        label_pad=-10,
    )
    if y_ticks[-1] < 100:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, minor_length=3)
    ax.tick_params(axis="x", which="major", length=0, pad=2)
    sns.despine(ax=ax, bottom=True, trim=True)

    ax.text(
        x=x_ticks[-1] + 0.3,
        y=0.97 * max_value,
        s=r"$N_{neurons}$=" + str(len(static_center_responses)),
        ha="right",
        va="top",
        fontsize=TICK_FONTSIZE,
    )

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def plot_distribution(df: pd.DataFrame, response_type: str, filename: Path) -> None:
    center_responses = df[df.stimulus_type == "center"].response.values
    center_responses = repeat(
        center_responses, "n -> (n d)", d=df.stimulus_type.nunique()
    )
    df["response"] = df["response"] / center_responses
    df = df[df.stimulus_type != "center"]
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(0.5 * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    min_value = np.floor(df.response.min())
    max_value = np.ceil(df.response.max())
    y_ticks = np.array([min_value, max_value])

    sns.boxplot(
        data=df,
        x="stimulus_type",
        y="response",
        hue="model",
        width=0.5,
        gap=0.2,
        fill=False,
        flierprops={"marker": ".", "markersize": 5},
        palette={"recorded": "black", "predicted": "limegreen"},
        linewidth=1,
        linecolor="black",
        order=[
            "natural_static",
            "natural_dynamic",
            "generated_static",
            "generated_dynamic",
        ],
        ax=ax,
    )
    x_ticks = ax.get_xticks()
    ax.set_xlim(x_ticks[0] - 0.4, x_ticks[-1] + 0.4)
    sns.move_legend(
        ax,
        loc="upper left",
        bbox_to_anchor=(x_ticks[0] - 0.3, max_value),
        bbox_transform=ax.transData,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.3,
        handlelength=0.7,
        labelspacing=0.08,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        labels=["Recorded", "Predicted"],
        fontsize=TICK_FONTSIZE,
    )
    ax.axhline(y=1, linewidth=1, linestyle="--", color="black", alpha=0.6, zorder=0)

    x_tick_labels = (
        ["MENI", "MENV", "MEGI", "MEGV"]
        if "most_exciting" == response_type
        else ["MINI", "MINV", "MIGI", "MIGV"]
    )
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        label=f"{response_type.replace('_', '-').capitalize()} stimulus",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=y_ticks.astype(int),
        label=r"Sum norm $\Delta$F/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-4,
    )
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, pad=2, minor_length=3)
    sns.despine(ax=ax)
    ax.text(
        x=x_ticks[0] - 0.3,
        y=0.8 * max_value,
        s=r"$N_{neurons}$=" + str(df.neuron.nunique()),
        ha="left",
        va="top",
        fontsize=TICK_FONTSIZE,
    )
    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def main():
    df = []
    selected_neurons = {}
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
        mouse_df = load_recorded_data(
            data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "artificial_movies",
            output_dir=OUTPUT_DIR / output_dir_name,
            mouse_id=mouse_id,
        )
        reliable_neurons = filter_neurons(
            day1_data_dir=DATA_DIR / data.MOUSE_IDS[mouse_id],
            day2_data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "natural_movies",
            percentage=0.8,
        )
        mouse_df = mouse_df[mouse_df.neuron.isin(reliable_neurons)]
        selected_neurons[mouse_id] = mouse_df.neuron.unique()
        mouse_df.insert(loc=0, column="mouse", value=mouse_id)
        df.append(mouse_df)
    df = pd.concat(df, ignore_index=True)
    plot_center_surround_response(
        df=df,
        filename=PLOT_DIR / f"in_vivo_static_vs_dynamic_surround.{FORMAT}",
        color="black",
    )

    df = []
    for mouse_id, output_dir_name, experiment_name in [
        (
            "K",
            "003_causal_viv1t_finetune",
            "001_cutoff",
        ),
        (
            "L",
            "015_causal_viv1t_FOV2_finetune",
            "001_cutoff_natural_init",
        ),
        (
            "M",
            "018_causal_viv1t_VIPcre233_FOV1_finetune",
            "001_cutoff_natural_init_200_steps",
        ),
        (
            "N",
            "025_causal_viv1t_VIPcre233_FOV2_finetune",
            "001_cutoff_natural_init_200_steps",
        ),
    ]:
        mouse_df = load_predicted_result(
            output_dir=OUTPUT_DIR / output_dir_name,
            mouse_id=mouse_id,
            neurons=selected_neurons[mouse_id],
            experiment_name=experiment_name,
        )
        mouse_df.insert(loc=0, column="mouse", value=mouse_id)
        df.append(mouse_df)
    df = pd.concat(df, ignore_index=True)
    plot_center_surround_response(
        df=df,
        filename=PLOT_DIR / f"in_silico_static_vs_dynamic_surround.{FORMAT}",
        color="limegreen",
    )

    print(f"Saved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
