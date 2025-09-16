from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
from viv1t.utils import plot
from viv1t.utils.utils import get_reliable_neurons

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "jpg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

plot.set_font()

PLOT_DIR = Path("figures") / "VIPcre233_FOV1_day2" / "200_steps"


PATTERN_SIZE = 30
BLANK_SIZE = 15


GRATING_COLOR = "black"
GRATING_SURROUND_COLOR = "forestgreen"
DYNAMIC_NATURAL_COLOR = "dodgerblue"
DYNAMIC_GENERATED_COLOR = "crimson"


VIDEO_IDS = [
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


def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
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
    return responses, video_ids


def select_responses(responses: np.ndarray, video_ids: np.ndarray, neurons: np.ndarray):
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


def plot_center_surround_response(df: pd.DataFrame, filename: Path):
    grating_center_response = df[df.response_type == "center"].response.values

    # normalize all responses by response to grating center
    grating_exciting = (
        df[
            (df.response_type == "most_exciting")
            & (df.stimulus_type == "grating_dynamic")
        ].response.values
        / grating_center_response
    )
    grating_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
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
    natural_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
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
    generated_inhibiting = (
        df[
            (df.response_type == "most_inhibiting")
            & (df.stimulus_type == "generated_dynamic")
        ].response.values
        / grating_center_response
    )

    responses = [
        grating_exciting,
        natural_exciting,
        generated_exciting,
        grating_inhibiting,
        natural_inhibiting,
        generated_inhibiting,
    ]

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.8),
        dpi=DPI,
    )

    box_width, box_pad = 0.12, 0.06
    linewidth = 1.2

    x_ticks = np.array([1, 2.1], dtype=np.float32)
    positions = [
        x_ticks[0] - 1 * (box_width + box_pad),
        x_ticks[0] - 0 * (box_width + box_pad),
        x_ticks[0] + 1 * (box_width + box_pad),
        x_ticks[1] - 1 * (box_width + box_pad),
        x_ticks[1] - 0 * (box_width + box_pad),
        x_ticks[1] + 1 * (box_width + box_pad),
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

    # test statistical difference between response pairs
    p_value = add_p_value(
        ax=ax,
        response1=grating_exciting,
        response2=natural_exciting,
        position1=positions[0],
        position2=positions[1],
        max_value=max_value,
    )
    print(f"\tgrating exciting vs natural exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=grating_exciting,
        response2=generated_exciting,
        position1=positions[0],
        position2=positions[2],
        max_value=1.07 * max_value,
    )
    print(f"\tgrating exciting vs generated exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=natural_exciting,
        response2=generated_exciting,
        position1=positions[1],
        position2=positions[2],
        max_value=max_value,
    )
    print(f"\tnatural exciting vs generated exciting p-value: {p_value:.04e}")

    p_value = add_p_value(
        ax=ax,
        response1=natural_inhibiting,
        response2=grating_inhibiting,
        position1=positions[4],
        position2=positions[3],
        max_value=max_value,
    )
    print(f"\tnatural inhibiting vs grating exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=generated_inhibiting,
        response2=grating_inhibiting,
        position1=positions[5],
        position2=positions[3],
        max_value=1.07 * max_value,
    )
    print(f"\tgenerated inhibiting vs grating exciting p-value: {p_value:.04e}")
    p_value = add_p_value(
        ax=ax,
        response1=generated_inhibiting,
        response2=natural_inhibiting,
        position1=positions[5],
        position2=positions[4],
        max_value=max_value,
    )
    print(f"\tgenerated inhibiting vs grating inhibiting p-value: {p_value:.04e}")

    scatter_kw = {
        "s": 10,
        "marker": ".",
        "alpha": 0.5,
        "zorder": 1,
        "facecolors": "none",
        "clip_on": False,
    }
    colors = [
        GRATING_SURROUND_COLOR,
        DYNAMIC_NATURAL_COLOR,
        DYNAMIC_GENERATED_COLOR,
    ] * 2
    for i, (position, response) in enumerate(zip(positions, responses)):
        outliers = np.where(response > max_value)[0]
        inliers = np.where(response <= max_value)[0]
        x = rng.normal(position, 0.0, size=len(response))
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

    line_kw = {
        "color": "black",
        "linestyle": "-",
        "linewidth": 1,
        "alpha": 0.3,
        "zorder": 0,
        "clip_on": False,
    }
    for i in range(len(grating_center_response)):
        ax.plot(
            positions[:3],
            np.clip(
                [responses[0][i], responses[1][i], responses[2][i]],
                a_min=0,
                a_max=max_value,
            ),
            **line_kw,
        )
        ax.plot(
            positions[3:],
            np.clip(
                [responses[3][i], responses[4][i], responses[5][i]],
                a_min=0,
                a_max=max_value,
            ),
            **line_kw,
        )

    xlim = [x_ticks[0] - 0.45, x_ticks[-1] + 0.45]

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

    ax.text(
        x=x_ticks[-1] + 0.3,
        y=max_value,
        s=r"$N_{neurons}$=" + str(len(grating_center_response)),
        ha="right",
        va="top",
        fontsize=TICK_FONTSIZE,
    )

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def load_predicted_result(
    output_dir: Path, mouse_id: str, neurons: list[int] | np.ndarray
) -> pd.DataFrame:
    blank_size = (data.MAX_FRAME - PATTERN_SIZE) // 2
    start, end = -(blank_size + PATTERN_SIZE), -blank_size
    save_dir = output_dir / "most_exciting_stimulus" / "single_neuron"
    df = {"neuron": [], "stimulus_type": [], "response_type": [], "response": []}
    experiment_name = "001_cutoff_natural_init_200_steps"
    for neuron in neurons:
        # load center response
        ckpt = torch.load(
            save_dir
            / "generated"
            / "center_surround"
            / "grating_center"
            / "dynamic_center_dynamic_surround"
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
        # load grating surround
        grating_responses = pd.read_parquet(
            save_dir
            / "gratings"
            / "center_surround"
            / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
        )
        grating_responses = grating_responses[
            (grating_responses.mouse == mouse_id) & (grating_responses.neuron == neuron)
        ]
        center_direction = (
            grating_responses[grating_responses.surround_direction == -1]
            .sort_values(by="response", ascending=False)
            .iloc[0]
            .center_direction
        )
        surround_gratings = grating_responses[
            grating_responses.center_direction == center_direction
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
        del grating_responses, surround_gratings
        # load natural surround
        for stimulus_type in ["natural_dynamic"]:
            for response_type in ["most_exciting", "most_inhibiting"]:
                response = pd.read_parquet(
                    save_dir
                    / "gratings"
                    / "grating_center_natural_surround"
                    / f"dynamic_center_{'static' if 'static' in stimulus_type else 'dynamic'}_surround"
                    / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
                )
                df["neuron"].append(neuron)
                df["stimulus_type"].append(stimulus_type)
                df["response_type"].append(response_type)
                df["response"].append(
                    response[response.response_type == response_type].iloc[0].response
                )
        # load generated surround
        for stimulus_type in ["generated_dynamic"]:
            for response_type in ["most_exciting", "most_inhibiting"]:
                ckpt = torch.load(
                    save_dir
                    / "generated"
                    / "center_surround"
                    / "grating_center"
                    / f"dynamic_center_{'static' if 'static' in stimulus_type else 'dynamic'}_surround"
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
    data_dir = Path("../../data/rochefort-lab/VIPcre233_FOV1_day2/artificial_movies")
    output_dir = Path(
        "../../runs/rochefort-lab/vivit/018_causal_viv1t_VIPcre233_FOV1_finetune"
    )
    mouse_id = "M"
    responses, video_ids = load_data(data_dir=data_dir)
    neurons = get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id, size=25)
    # bad_neurons = np.array([33, 63, 158, 206], dtype=int)
    # neurons = np.setdiff1d(neurons, bad_neurons)
    recorded_df = select_responses(
        responses=responses, video_ids=video_ids, neurons=neurons
    )
    recorded_df.insert(loc=0, column="model", value="recorded")
    predicted_df = load_predicted_result(
        output_dir=output_dir, mouse_id=mouse_id, neurons=recorded_df.neuron.unique()
    )
    predicted_df.insert(loc=0, column="model", value="predicted")
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    plot_center_surround_response(
        df=df[df.model == "recorded"],
        filename=PLOT_DIR / f"in_vivo_grating_vs_generated_surround.{FORMAT}",
    )
    for response_type in ["most_exciting", "most_inhibiting"]:
        plot_distribution(
            df=df[df.response_type.isin([response_type, "center"])].copy(deep=True),
            response_type=response_type,
            filename=PLOT_DIR / f"single_neuron_response_{response_type}.{FORMAT}",
        )
    print(f"Saved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
