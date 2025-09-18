from math import ceil
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
from scipy.stats import sem
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

NSNS_COLOR = "midnightblue"
NSND_COLOR = "dodgerblue"
NSGS_COLOR = "orangered"
NSGD_COLOR = "crimson"

VIDEO_IDS = [
    ("CS000", "center", "center"),
    ("CS001", "natural_static", "most_exciting"),
    ("CS002", "natural_static", "most_inhibiting"),
    ("CS003", "natural_dynamic", "most_exciting"),
    ("CS004", "natural_dynamic", "most_inhibiting"),
    ("CS005", "generated_static", "most_exciting"),  # 100 steps
    ("CS006", "generated_static", "most_inhibiting"),  # 100 steps
    ("CS007", "generated_dynamic", "most_exciting"),  # 100 steps
    ("CS008", "generated_dynamic", "most_inhibiting"),  # 100 steps
    # ("CS009", "generated_static", "most_exciting"),  # 200 steps
    # ("CS010", "generated_static", "most_inhibiting"),  # 200 steps
    # ("CS011", "generated_dynamic", "most_exciting"),  # 200 steps
    # ("CS012", "generated_dynamic", "most_inhibiting"),  # 200 steps
    # ("CS013", "generated_static", "most_exciting"),  # 500 steps
    # ("CS014", "generated_static", "most_inhibiting"),  # 500 steps
    # ("CS015", "generated_dynamic", "most_exciting"),  # 500 steps
    # ("CS016", "generated_dynamic", "most_inhibiting"),  # 500 steps
    # ("CS017", "generated_static", "most_exciting"),  # 1000 steps
    # ("CS018", "generated_static", "most_inhibiting"),  # 1000 steps
    # ("CS019", "generated_dynamic", "most_exciting"),  # 1000 steps
    # ("CS020", "generated_dynamic", "most_inhibiting"),  # 1000 steps
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


def select_responses(responses: np.ndarray, video_ids: np.ndarray):
    df = {"neuron": [], "stimulus_type": [], "response_type": [], "response": []}
    neurons = np.arange(responses.shape[0], dtype=int)
    for i in range(len(VIDEO_IDS)):
        indexes = np.where(video_ids == VIDEO_IDS[i][0])[0]
        assert len(indexes) >= 5
        # select response during presentation window
        response = responses[:, indexes, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
        # sum response over presentation window
        response = np.sum(response, axis=2)
        # average response over repeats
        response = np.mean(response, axis=1)
        df["neuron"].extend(neurons)
        df["stimulus_type"].extend([VIDEO_IDS[i][1]] * len(neurons))
        df["response_type"].extend([VIDEO_IDS[i][2]] * len(neurons))
        df["response"].extend(response)
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
    response_type: str,
) -> float:
    match response_type:
        case "most_exciting":
            alternative = "less"
        case "most_inhibiting":
            alternative = "greater"
        case _:
            raise RuntimeError(f"Unknown response type: {response_type}")
    p_value = wilcoxon(response1, response2, alternative=alternative).pvalue
    p_value = 8 * p_value  # adjust for multiple tests
    text_pad = 0.035 if p_value < 0.05 else 0.032
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


def plot_center_surround_response(df: pd.DataFrame, filename: Path):
    stimulus_types = [
        "natural_static",
        "natural_dynamic",
        "generated_static",
        "generated_dynamic",
    ]
    most_exciting_responses = [
        df[
            (df["response_type"] == "most_exciting")
            & (df["stimulus_type"] == stimulus_type)
        ].response.values
        for stimulus_type in stimulus_types
    ]
    most_exciting_responses = np.array(most_exciting_responses, dtype=np.float32)

    most_inhibiting_responses = [
        df[
            (df["response_type"] == "most_inhibiting")
            & (df["stimulus_type"] == stimulus_type)
        ].response.values
        for stimulus_type in stimulus_types
    ]
    most_inhibiting_responses = np.array(most_inhibiting_responses, dtype=np.float32)

    # normalize response by response to natural static center
    center_response = df[df.response_type == "center"].response.values
    most_exciting_responses /= center_response
    most_inhibiting_responses /= center_response

    linewidth = 1.2
    box_width, box_pad = 0.14, 0.06

    x_ticks = np.array([1, 2], dtype=np.float32)
    most_exciting_ticks = [
        x_ticks[0] - 1.5 * (box_width + box_pad),
        x_ticks[0] - 0.5 * (box_width + box_pad),
        x_ticks[0] + 0.5 * (box_width + box_pad),
        x_ticks[0] + 1.5 * (box_width + box_pad),
    ]
    most_inhibiting_ticks = [
        x_ticks[1] - 1.5 * (box_width + box_pad),
        x_ticks[1] - 0.5 * (box_width + box_pad),
        x_ticks[1] + 0.5 * (box_width + box_pad),
        x_ticks[1] + 1.5 * (box_width + box_pad),
    ]

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 2.2),
        dpi=DPI,
    )

    linewidth = 1.2
    linestyle = "-"

    max_value = 0

    for positions, responses in [
        (most_exciting_ticks, most_exciting_responses),
        (most_inhibiting_ticks, most_inhibiting_responses),
    ]:
        height = np.mean(responses, axis=1)
        yerr = sem(responses, axis=1)
        ax.bar(
            x=positions,
            height=height,
            yerr=yerr,
            width=box_width,
            facecolor="none",
            edgecolor="black",
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=1,
            zorder=1,
            error_kw={"linewidth": linewidth, "zorder": 1},
        )
        max_value = max(max_value, np.max(height + yerr))
    max_value = ceil(max_value)

    # for response_type, positions, responses in [
    #     ("most_exciting", most_exciting_ticks, most_exciting_responses),
    #     ("most_inhibiting", most_inhibiting_ticks, most_inhibiting_responses),
    # ]:
    #     p_value = add_p_value(
    #         ax=ax,
    #         response1=responses[0],
    #         response2=responses[1],
    #         position1=positions[0],
    #         position2=positions[1],
    #         max_value=max_value,
    #         response_type=response_type,
    #     )
    #     p_value = add_p_value(
    #         ax=ax,
    #         response1=responses[2],
    #         response2=responses[3],
    #         position1=positions[2],
    #         position2=positions[3],
    #         max_value=max_value,
    #         response_type=response_type,
    #     )
    #     p_value = add_p_value(
    #         ax=ax,
    #         response1=responses[1],
    #         response2=responses[3],
    #         position1=positions[1],
    #         position2=positions[3],
    #         max_value=1.06 * max_value,
    #         response_type=response_type,
    #     )
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

    xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.5]
    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=most_exciting_ticks + most_inhibiting_ticks,
        tick_labels=["Nat. img.", "Nat. vid.", "Gen. img.", "Gen. vid"] * 2,
        tick_fontsize=TICK_FONTSIZE,
        linespacing=0.85,
        rotation=270,
        va="top",
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
        label_pad=-5,
        rotation=90,
    )
    if y_ticks[-1] < 100:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, pad=0, minor_length=3)
    ax.tick_params(axis="x", which="major", pad=2)
    sns.despine(ax=ax)
    ax.text(
        x=x_ticks[-1] + 0.3,
        y=max_value,
        s=r"$N_{neurons}$=" + str(len(center_response)),
        ha="right",
        va="top",
        fontsize=TICK_FONTSIZE,
    )
    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def load_predicted_result(output_dir: Path, mouse_id: str) -> pd.DataFrame:
    blank_size = (data.MAX_FRAME - PATTERN_SIZE) // 2
    start, end = -(blank_size + PATTERN_SIZE), -blank_size
    save_dir = output_dir / "most_exciting_stimulus" / "population"
    df = {"neuron": [], "stimulus_type": [], "response_type": [], "response": []}
    experiment_name = "001_cutoff_natural_init_200_steps"
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
    response = ckpt["response"][:, start:end].numpy()
    neurons = np.arange(len(response), dtype=int)
    df["neuron"].extend(neurons)
    df["stimulus_type"].extend(["center"] * len(neurons))
    df["response_type"].extend(["center"] * len(neurons))
    df["response"].extend(np.sum(response, axis=-1))
    # load natural surround
    for stimulus_type in ["natural_static", "natural_dynamic"]:
        for response_type in ["most_exciting", "most_inhibiting"]:
            response = pd.read_parquet(
                save_dir
                / "natural"
                / "center_surround"
                / f"static_center_{'static' if 'static' in stimulus_type else 'dynamic'}_surround"
                / f"mouse{mouse_id}.parquet"
            )
            df["neuron"].extend(neurons)
            df["stimulus_type"].extend([stimulus_type] * len(neurons))
            df["response_type"].extend([response_type] * len(neurons))
            df["response"].extend(
                response[response.response_type == response_type].response.values
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
                / response_type
                / "ckpt.pt",
                map_location="cpu",
            )
            response = ckpt["response"][:, start:end].numpy()
            df["neuron"].extend(neurons)
            df["stimulus_type"].extend([stimulus_type] * len(neurons))
            df["response_type"].extend([response_type] * len(neurons))
            df["response"].extend(np.sum(response, axis=-1))
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

    sns.violinplot(
        data=df,
        x="stimulus_type",
        y="response",
        hue="model",
        width=0.5,
        # gap=0.2,
        fill=False,
        split=True,
        inner="quart",
        # flierprops={"marker": ".", "markersize": 5},
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
        log_scale=True,
    )
    # "flierprops": {
    #     "marker": "o",
    #     "markersize": 2,
    #     "alpha": 0.5,
    #     "clip_on": False,
    #     "zorder": 10,
    # },
    # sns.boxplot(
    #     data=df,
    #     x="stimulus_type",
    #     y="response",
    #     hue="model",
    #     width=0.5,
    #     gap=0.2,
    #     fill=False,
    #     showmeans=True,
    #     meanprops={
    #         "markersize": 4,
    #         "markerfacecolor": "gold",
    #         "markeredgecolor": "black",
    #         "markeredgewidth": 0.75,
    #         "clip_on": False,
    #         "zorder": 20,
    #     },
    #     showfliers=False,
    #     flierprops={
    #         "marker": ".",
    #         "markersize": 5,
    #     },
    #     palette={"recorded": "black", "predicted": "limegreen"},
    #     linewidth=1,
    #     linecolor="black",
    #     order=[
    #         "natural_static",
    #         "natural_dynamic",
    #         "generated_static",
    #         "generated_dynamic",
    #     ],
    #     ax=ax,
    # )
    x_ticks = ax.get_xticks()
    ax.set_xlim(x_ticks[0] - 0.4, x_ticks[-1] + 0.4)
    sns.move_legend(
        ax,
        loc="best",
        # bbox_to_anchor=(x_ticks[0] - 0.3, max_value),
        # bbox_transform=ax.transData,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.3,
        handlelength=0.7,
        labelspacing=0.08,
        columnspacing=0,
        borderpad=0.2,
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
    # plot.set_yticks(
    #     axis=ax,
    #     ticks=y_ticks,
    #     tick_labels=y_ticks.astype(int),
    #     label=r"Sum norm $\Delta$F/F",
    #     tick_fontsize=TICK_FONTSIZE,
    #     label_fontsize=TICK_FONTSIZE,
    #     label_pad=-4,
    # )
    # ax.set_ylim(y_ticks[0], y_ticks[-1])
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="y", which="both", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(r"Sum $\Delta$F/F", fontsize=LABEL_FONTSIZE, labelpad=0)
    plot.set_ticks_params(ax, length=3, pad=2, minor_length=2)
    sns.despine(ax=ax)
    # ax.text(
    #     x=x_ticks[0] - 0.3,
    #     y=0.8 * max_value,
    #     s=r"$N_{neurons}$=" + str(df.neuron.nunique()),
    #     ha="left",
    #     va="top",
    #     fontsize=TICK_FONTSIZE,
    # )
    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def main():
    data_dir = Path("../../data/rochefort-lab/VIPcre233_FOV1_day2/artificial_movies")
    output_dir = Path(
        "/Users/bryanlimy/Git/sensorium2023/runs/rochefort-lab/vivit/018_causal_viv1t_VIPcre233_FOV1_finetune"
    )
    mouse_id = "M"
    responses, video_ids = load_data(data_dir=data_dir)
    recorded_df = select_responses(responses=responses, video_ids=video_ids)
    recorded_df.insert(loc=0, column="model", value="recorded")
    predicted_df = load_predicted_result(output_dir=output_dir, mouse_id=mouse_id)
    # only keep neurons that were matched between the two sessions
    predicted_df = predicted_df[
        predicted_df["neuron"].isin(recorded_df["neuron"].values)
    ]
    predicted_df.insert(loc=0, column="model", value="predicted")
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    plot_center_surround_response(
        df=recorded_df,
        filename=PLOT_DIR / f"population_center_surround_response.{FORMAT}",
    )
    for response_type in ["most_exciting", "most_inhibiting"]:
        plot_distribution(
            df=df[df.response_type.isin([response_type, "center"])].copy(deep=True),
            response_type=response_type,
            filename=PLOT_DIR / f"population_center_surround_{response_type}.{FORMAT}",
        )
    print(f"Saved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
