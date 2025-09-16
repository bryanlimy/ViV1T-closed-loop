from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from scipy.stats import sem
from scipy.stats import wilcoxon

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

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
    ("FF000", "natural_static", "most_exciting"),
    ("FF001", "natural_static", "most_inhibiting"),
    ("FF002", "natural_dynamic", "most_exciting"),
    ("FF003", "natural_dynamic", "most_inhibiting"),
    ("FF004", "generated_static", "most_exciting"),
    ("FF005", "generated_static", "most_inhibiting"),
    ("FF006", "generated_dynamic", "most_exciting"),
    ("FF007", "generated_dynamic", "most_inhibiting"),
]


def get_recorded_grating_response(
    responses: np.ndarray,
    mouse_id: str,
    video_ids: np.ndarray,
    neuron_weights: np.ndarray | None = None,
) -> list:
    # load response to grating center
    match mouse_id:
        case "K":
            grating_video_ids = ["CM009", "CM013"]
        case "L":
            grating_video_ids = ["CM017", "CM021", "CM025", "CM029"]
        case "M":
            grating_video_ids = [
                "GCS001_N254",
                "GCS002_N002",
                "GCS002_N012",
                "GCS002_N044",
                "GCS002_N186",
                "GCS002_N301",
            ]
        case "N":
            grating_video_ids = [
                "GCS001_N010",
                "GCS001_N045",
                "GCS001_N214",
                "GCS002_N098",
            ]
        case _:
            raise NotImplementedError(f"Unknown mouse_id: {mouse_id}")
    grating_responses = np.zeros(
        (len(grating_video_ids), responses.shape[0]), dtype=np.float32
    )
    for i, video_id in enumerate(grating_video_ids):
        indexes = np.where(video_ids == video_id)[0]
        assert len(indexes) >= 5
        grating_response = responses[:, indexes, BLANK_SIZE : PATTERN_SIZE + BLANK_SIZE]
        # sum response over presentation window
        grating_response = np.sum(grating_response, axis=2)
        # average response over repeats
        grating_response = np.mean(grating_response, axis=1)
        grating_responses[i] = grating_response
    if neuron_weights is not None:
        grating_responses = grating_responses * neuron_weights[None, :]
    # compute population average
    grating_response = np.nanmean(grating_responses, axis=1)
    print(f"mouse {mouse_id} max grating: {np.argmax(grating_response)}")
    grating_response = np.max(grating_response)
    return grating_response


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

    # average response over population
    responses_ = np.nanmean(responses_, axis=0)

    df = pd.DataFrame(
        {
            "stimulus_type": [VIDEO_IDS[i][1] for i in range(len(VIDEO_IDS))],
            "response_type": [VIDEO_IDS[i][2] for i in range(len(VIDEO_IDS))],
            "response": responses_.tolist(),
        }
    )

    # load response to full-field grating
    grating_response = get_recorded_grating_response(
        responses=responses,
        mouse_id=mouse_id,
        video_ids=video_ids,
    )
    grating_df = pd.DataFrame(
        {
            "stimulus_type": ["grating"],
            "response_type": ["most_exciting"],
            "response": [grating_response],
        }
    )
    df = pd.concat([df, grating_df], ignore_index=True)

    return df, neurons


def load_grating_prediction(output_dir: Path, mouse_id: str) -> np.ndarray:
    save_dir = output_dir / "contextual_modulation_population"
    # shape (num. neurons, num. pattern, num. frames)
    responses = np.load(save_dir / f"mouse{mouse_id}.npz", allow_pickle=False)["data"]
    # shape (num. trial, num. pattern, parameter)
    parameters = np.load(save_dir / "parameters.npy", allow_pickle=False)
    # remove blank frames
    responses = responses[:, :, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    # sum response over frames for each pattern
    responses = np.sum(responses, axis=-1)
    responses = rearrange(responses, "neuron block pattern -> neuron (block pattern)")
    parameters = rearrange(parameters, "block pattern param -> (block pattern) param")
    # get response to high contrast center grating
    indexes = np.where(
        (parameters[:, 0] == 1) & (parameters[:, 1] == 20) & (parameters[:, 3] == 0)
    )[0]
    # average response over repeats
    responses = np.mean(responses[:, indexes], axis=-1)
    return responses


def load_predicted_result(
    output_dir: Path, mouse_id: str, neurons: np.ndarray, experiment_name: str
) -> pd.DataFrame:
    blank_size = (data.MAX_FRAME - PATTERN_SIZE) // 2
    start, end = -(blank_size + PATTERN_SIZE), -blank_size
    save_dir = output_dir / "most_exciting_stimulus" / "population"
    responses = np.full(
        shape=(len(neurons), len(VIDEO_IDS)), fill_value=np.nan, dtype=np.float32
    )
    i = 0

    for stimulus_type in ["natural_static", "natural_dynamic"]:
        for response_type in ["most_exciting", "most_inhibiting"]:
            df = pd.read_parquet(
                save_dir
                / "natural"
                / "full_field"
                / ("static" if "static" in stimulus_type else "dynamic")
                / f"mouse{mouse_id}.parquet"
            )
            df = df[(df.neuron.isin(neurons)) & (df.response_type == response_type)]
            df.sort_values(by="neuron", ascending=True)
            response = df.response.values
            responses[:, i] = response
            i += 1
    for stimulus_type in ["generated_static", "generated_dynamic"]:
        for response_type in ["most_exciting", "most_inhibiting"]:
            ckpt = torch.load(
                save_dir
                / "generated"
                / "full_field"
                / ("static" if "static" in stimulus_type else "dynamic")
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
    # average response over population
    responses = np.nanmean(responses, axis=0)
    df = pd.DataFrame(
        {
            "stimulus_type": [VIDEO_IDS[i][1] for i in range(len(VIDEO_IDS))],
            "response_type": [VIDEO_IDS[i][2] for i in range(len(VIDEO_IDS))],
            "response": responses.tolist(),
        }
    )

    # load center grating response
    grating_response = load_grating_prediction(output_dir=output_dir, mouse_id=mouse_id)
    grating_response = np.mean(grating_response[neurons])
    grating_df = pd.DataFrame(
        {
            "stimulus_type": ["grating"],
            "response_type": ["most_exciting"],
            "response": [grating_response],
        }
    )

    df = pd.concat([df, grating_df], ignore_index=True)
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
    p_value = 4 * p_value  # adjust for multiple tests
    text_pad = 0.045 if p_value < 0.05 else 0.036
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


def plot_full_field_response(df: pd.DataFrame, filename: Path, color: str = "black"):
    grating = df[df.stimulus_type == "grating"].response.values
    ns_exciting = df[
        (df.response_type == "most_exciting") & (df.stimulus_type == "natural_static")
    ].response.values
    nd_exciting = df[
        (df.response_type == "most_exciting") & (df.stimulus_type == "natural_dynamic")
    ].response.values
    gs_exciting = df[
        (df.response_type == "most_exciting") & (df.stimulus_type == "generated_static")
    ].response.values
    gd_exciting = df[
        (df.response_type == "most_exciting")
        & (df.stimulus_type == "generated_dynamic")
    ].response.values

    responses = [
        grating,
        ns_exciting,
        nd_exciting,
        gs_exciting,
        gd_exciting,
    ]

    linewidth = 1.2
    linestyle = "-"
    bar_width = 0.4
    box_width = 0.35
    x_ticks = np.array([1, 2, 3, 4, 5], dtype=np.float32)

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.4),
        dpi=DPI,
    )

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
    bp = ax.boxplot(responses, positions=x_ticks, **box_kw)
    max_value = np.ceil(np.max(responses))

    # p_value = add_p_value(
    #     ax=ax,
    #     response1=responses[0],
    #     response2=responses[1],
    #     position1=x_ticks[0],
    #     position2=x_ticks[1],
    #     max_value=max_value,
    #     response_type="most_exciting",
    # )
    # p_value = add_p_value(
    #     ax=ax,
    #     response1=responses[1],
    #     response2=responses[2],
    #     position1=x_ticks[1],
    #     position2=x_ticks[2],
    #     max_value=max_value,
    #     response_type="most_exciting",
    # )
    # p_value = add_p_value(
    #     ax=ax,
    #     response1=responses[2],
    #     response2=responses[3],
    #     position1=x_ticks[2],
    #     position2=x_ticks[3],
    #     max_value=max_value,
    #     response_type="most_exciting",
    # )
    # p_value = add_p_value(
    #     ax=ax,
    #     response1=responses[3],
    #     response2=responses[4],
    #     position1=x_ticks[3],
    #     position2=x_ticks[4],
    #     max_value=max_value,
    #     response_type="most_exciting",
    # )

    scatter_kw = {
        "s": 30,
        "marker": ".",
        "alpha": 0.8,
        "zorder": 0,
        "facecolors": "none",
        "edgecolors": color,
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
        "alpha": 0.25,
        "zorder": 0,
        "clip_on": False,
    }
    for n in range(df.mouse.nunique()):
        ax.plot(
            x_ticks,
            np.clip(
                [responses[i][n] for i in range(len(x_ticks))],
                a_min=0,
                a_max=max_value,
            ),
            **line_kw,
        )

    xlim = [x_ticks[0] - 0.5, x_ticks[-1] + 0.4]
    ax.set_xlim(*xlim)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=[
            "Grating",
            "Natural\nimage",
            "Natural\nvideo",
            "Gen.\nimage",
            "Gen.\nvideo",
        ],
        tick_fontsize=TICK_FONTSIZE,
        linespacing=0.85,
        # rotation=45,
    )
    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label=r"Sum $\Delta$F/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-3,
    )
    if y_ticks[-1] < 100:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    plot.set_ticks_params(ax, length=3, minor_length=3)
    sns.despine(ax=ax)

    ax.text(
        x=x_ticks[0] - 0.3,
        y=0.05 * max_value,
        s=r"$N_{mice}$=2" + "\n" + r"$N_{scans}$=" + str(df.mouse.nunique()),
        ha="left",
        va="bottom",
        fontsize=TICK_FONTSIZE,
        linespacing=1,
    )

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def main():
    recorded_df, predicted_df = [], []
    for mouse_id, day_name, output_dir_name, experiment_name in [
        (
            "K",
            "day2",
            "003_causal_viv1t_finetune",
            "002_30frames_cutoff_population",
        ),
        (
            "L",
            "day2",
            "015_causal_viv1t_FOV2_finetune",
            "001_cutoff_natural_init",
        ),
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

    print("Plot in vivo result")
    recorded_df = pd.concat(recorded_df, ignore_index=True)
    plot_full_field_response(
        df=recorded_df,
        filename=PLOT_DIR / f"in_vivo_full_field_response.{FORMAT}",
        color="black",
    )

    print("Plot in silico result")
    predicted_df = pd.concat(predicted_df, ignore_index=True)
    plot_full_field_response(
        df=predicted_df,
        filename=PLOT_DIR / f"in_silico_full_field_response.{FORMAT}",
        color="limegreen",
    )


if __name__ == "__main__":
    main()
