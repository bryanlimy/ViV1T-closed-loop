from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from scipy.stats import ranksums
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches


plot.set_font()

PLOT_DIR = Path("figures") / "population_full_field_response"

PATTERN_SIZE = 15
PADDING = 10  # number of frames to include before and after stimulus presentation

GRATING_COLOR = "black"
STATIC_NATURAL_COLOR = "midnightblue"
DYNAMIC_NATURAL_COLOR = "dodgerblue"
STATIC_GENERATED_COLOR = "orangered"
DYNAMIC_GENERATED_COLOR = "crimson"


def load_grating_result(
    output_dir: Path, mouse_id: str, response_type: str
) -> pd.DataFrame | None:
    filename = output_dir / "most_exciting_stimulus" / "population" / "gratings.pkl"
    df = pd.read_pickle(filename)
    if mouse_id not in df.mouse.unique():
        return None
    # get most-exciting grating from the recorded response
    recorded_df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
    match response_type:
        case "most_exciting":
            target_response = recorded_df.response.max()
        case "most_inhibiting":
            target_response = recorded_df.response.min()
        case _:
            raise ValueError(f"Unknown response_type: {response_type}")
    recorded_df = recorded_df[recorded_df.response == target_response]
    # get predicted response to the same directional grating
    predicted_df = df[
        (df.mouse == mouse_id)
        & (df.model == "predicted")
        & (df.direction == recorded_df.iloc[0].direction)
        & (df.wavelength == recorded_df.iloc[0].wavelength)
        & (df.frequency == recorded_df.iloc[0].frequency)
    ]
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    df.drop(
        columns=["direction", "wavelength", "frequency", "raw_response", "stimulus"],
        inplace=True,
    )
    df["stimulus_type"] = "grating"
    return df


def load_natural_image_result(
    output_dir: Path, mouse_id: str, response_type: str
) -> pd.DataFrame | None:
    filename = output_dir / "most_exciting_stimulus" / "population" / "images.pkl"
    df = pd.read_pickle(filename)
    if mouse_id not in df.mouse.unique():
        return None
    df.drop(columns=["raw_response", "stimulus"], inplace=True)
    recorded_df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
    match response_type:
        case "most_exciting":
            recorded_df = recorded_df.sort_values(by=["response"], ascending=False)
        case "most_inhibiting":
            recorded_df = recorded_df.sort_values(by=["response"], ascending=True)
        case _:
            raise ValueError(f"Unknown response_type: {response_type}")
    recorded_df = recorded_df[recorded_df.response == recorded_df.iloc[0].response]
    # get predicted response to the same image
    predicted_df = df[
        (df.mouse == mouse_id)
        & (df.model == "predicted")
        & (df.image == recorded_df.iloc[0].image)
    ]
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    df.drop(columns=["image"], inplace=True)
    df["stimulus_type"] = "natural_image"
    return df


def load_natural_video_result(
    output_dir: Path, mouse_id: str, response_type: str
) -> pd.DataFrame | None:
    filename = output_dir / "most_exciting_stimulus" / "population" / "videos.pkl"
    df = pd.read_pickle(filename)
    if mouse_id not in df.mouse.unique():
        return None
    recorded_df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
    match response_type:
        case "most_exciting":
            recorded_df = recorded_df.sort_values(by=["response"], ascending=False)
        case "most_inhibiting":
            recorded_df = recorded_df.sort_values(by=["response"], ascending=True)
        case _:
            raise ValueError(f"Unknown response_type: {response_type}")
    recorded_df = recorded_df[recorded_df.response == recorded_df.iloc[0].response]
    # get predicted response to the same video clip
    predicted_df = df[
        (df.mouse == mouse_id)
        & (df.model == "predicted")
        & (df.video_id == recorded_df.iloc[0].video_id)
        & (df.frame_id == recorded_df.iloc[0].frame_id)
    ]
    del df
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    df.drop(columns=["video_id", "frame_id", "raw_response"], inplace=True)
    df["stimulus_type"] = "natural_video"
    return df


def load_generated_result(
    output_dir: Path,
    experiment_name: str,
    stimulus_type: str,
    response_type: str,
    mouse_id: str,
    neurons: np.ndarray | None = None,
) -> pd.DataFrame:
    assert stimulus_type in ("generated_image", "generated_video")
    assert response_type in ("most_exciting", "most_inhibiting")
    # max_response = load_max_response(filename=Path("ViV1T.npz"), mouse_id=mouse_id)
    filename = (
        output_dir
        / "most_exciting_stimulus"
        / "population"
        / "generated"
        / "full_field"
        / ("static" if stimulus_type == "generated_image" else "dynamic")
        / experiment_name
        / f"mouse{mouse_id}"
        / response_type
        / "ckpt.pt"
    )
    ckpt = torch.load(filename, map_location="cpu")
    # compute population average
    responses = ckpt["response"].numpy()
    blank_size2 = int(np.floor((data.MAX_FRAME - PATTERN_SIZE) / 2))
    responses = responses[:, -(blank_size2 + PATTERN_SIZE) : -(blank_size2)]
    if neurons is not None:
        responses = responses[neurons, :]
    # average over population
    responses = np.mean(responses, axis=0)
    # sum over presentation window
    response = np.sum(responses)
    df = pd.DataFrame(
        {
            "mouse": [mouse_id],
            "model": ["predicted"],
            "response": [response],
            "stimulus_type": [stimulus_type],
        }
    )
    return df


def stat_tests(responses1: np.ndarray, responses2: np.ndarray, name: str):
    print(
        f"{name}\n"
        f"\tttest_ind: {ttest_ind(responses1, responses2).pvalue:.04f}\n"
        f"\tttest_rel: {ttest_rel(responses1, responses2).pvalue:.04f}\n"
        f"\twilcoxon: {wilcoxon(responses1, responses2).pvalue:.04f}\n"
    )


def stronger(
    responses1: np.ndarray | float, responses2: np.ndarray | float
) -> np.ndarray:
    """return how much stronger/larger is responses1 with respect to responses2"""
    return 100 * (responses1 - responses2) / responses2


def statistical_tests(responses: list[np.ndarray]):
    # statistical test between recorded and predicted responses
    stat_tests(responses[0], responses[1], name="grating")
    stat_tests(responses[2], responses[3], name="natural image")
    stat_tests(responses[4], responses[5], name="natural video")
    # report how much stronger responses are from one stimulus to another
    print(
        f"predicted natural images vs. grating: "
        f"{stronger(np.mean(responses[3]), np.mean(responses[1])):.02f}%"
    )
    print(
        f"predicted natural video vs. grating: "
        f"{stronger(np.mean(responses[5]), np.mean(responses[1])):.02f}%\n"
    )
    print(
        f"predicted generated images vs. grating: "
        f"{stronger(np.mean(responses[6]), np.mean(responses[1])):.02f}%"
    )
    print(
        f"predicted generated video vs. grating: "
        f"{stronger(np.mean(responses[7]), np.mean(responses[1])):.02f}%\n"
    )
    print(
        f"predicted generated images vs. natural video: "
        f"{stronger(np.mean(responses[6]), np.mean(responses[5])):.02f}%"
    )
    print(
        f"predicted generated video vs. natural video: "
        f"{stronger(np.mean(responses[7]), np.mean(responses[5])):.02f}%\n"
    )
    # test if natural stimulus has stronger response than grating
    print(
        f"predicted natural image vs grating ranksum: "
        f"{ranksums(responses[3], responses[1], alternative='greater').pvalue:.04f}"
    )
    print(
        f"predicted natural video vs grating ranksum: "
        f"{ranksums(responses[5], responses[1], alternative='greater').pvalue:.04f}\n"
    )
    # test if generated stimulus has stronger response than natural stimulus
    print(
        f"predicted generated image vs natural video ranksum: "
        f"{ranksums(responses[6], responses[5], alternative='greater').pvalue:.04f}\n"
    )
    print(
        f"predicted generated video vs natural video ranksum: "
        f"{ranksums(responses[7], responses[5], alternative='greater').pvalue:.04f}\n"
    )


def add_p_value(
    ax: axes.Axes,
    response1: np.ndarray,
    response2: np.ndarray,
    position1: float,
    position2: float,
    max_value: float,
) -> float:
    p_value = wilcoxon(response1, response2, alternative="two-sided").pvalue
    y = 1.05 * np.max([response1, response2])
    plot.add_p_value(
        ax=ax,
        x0=position1,
        x1=position2,
        y=y + 0.05 * max_value,
        p_value=p_value,
        fontsize=LABEL_FONTSIZE,
        tick_length=0.013 * max_value,
        tick_linewidth=1,
        text_pad=0.028 * max_value,
    )
    return p_value


def plot_response(
    df: pd.DataFrame,
    model_name: str,
    filename: Path,
):
    recorded_responses = [
        df[
            (df.model == "recorded") & (df.stimulus_type == stimulus_type)
        ].response.values
        for stimulus_type in ["grating", "natural_image", "natural_video"]
    ]

    predicted_responses = [
        df[
            (df.model == "predicted") & (df.stimulus_type == stimulus_type)
        ].response.values
        for stimulus_type in [
            "grating",
            "natural_image",
            "natural_video",
            "generated_image",
            "generated_video",
        ]
    ]

    # statistical_tests(responses=responses)

    rng = np.random.RandomState(1234)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 2.3),
        dpi=DPI,
    )

    linewidth = 1.2
    linestyle = "-"
    bar_width = 0.3
    bar_pad = 0.05
    x_ticks = np.array([1, 2, 3, 4, 5])
    recorded_positions = x_ticks[:3] - (bar_width / 2) - bar_pad
    predicted_positions = x_ticks + (bar_width / 2) + bar_pad

    scatter_kw = {
        "s": 30,
        "marker": ".",
        "alpha": 0.8,
        "zorder": 0,
        "facecolors": "none",
        "clip_on": False,
    }
    for model_name, positions, responses in [
        ("recorded", recorded_positions, recorded_responses),
        ("predicted", predicted_positions, predicted_responses),
    ]:
        color = plot.get_color(model_name)
        ax.bar(
            x=positions,
            height=[np.mean(r) for r in responses],
            yerr=[sem(r) for r in responses],
            width=bar_width,
            facecolor="none",
            edgecolor=plot.get_color(model_name),
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=1,
            zorder=1,
            error_kw={"linewidth": linewidth, "zorder": 1},
            # clip_on=False,
        )
        max_value = max([np.max(r) for r in responses])
        max_value = np.ceil(max_value / 10) * 10

        for i in range(len(responses)):
            ax.scatter(
                [positions[i]] * len(responses[i]),
                responses[i],
                edgecolors=color,
                **scatter_kw,
                label=model_name.capitalize() if i == 0 else "",
            )
    # max_value = max(max_value, 380)

    for i in range(3):
        add_p_value(
            ax=ax,
            response1=recorded_responses[i],
            response2=predicted_responses[i],
            position1=recorded_positions[i],
            position2=predicted_positions[i],
            max_value=max_value,
        )

    xlim = [recorded_positions[0] - 0.3, predicted_positions[-1] + 0.2]
    ax.set_xlim(*xlim)
    x_tick_labels = (
        ["MEG", "MENI", "MENV", "MEGI", "MEGV"]
        if "most_exciting" in filename.stem
        else ["MIG", "MINI", "MINV", "MIGI", "MIGV"]
    )
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        tick_fontsize=TICK_FONTSIZE,
        linespacing=0.85,
        # rotation=0,
        # va="top",
    )

    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="Sum ΔF/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        rotation=90,
    )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(recorded_positions[0], max_value),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.2,
        handlelength=0.7,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        bbox_transform=ax.transData,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plot.set_ticks_params(ax, pad=1)
    ax.tick_params(axis="y", length=3, pad=0, labelsize=TICK_FONTSIZE)
    # ax.tick_params(axis="y", which="minor", length=2.2)
    # ax.tick_params(axis="x", length=3, pad=1, labelsize=TICK_FONTSIZE)
    sns.despine(ax=ax)

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def process_model(model_name: str, output_dir: Path):
    experiment_name = "003_cutoff_population"
    for response_type in ["most_exciting", "most_inhibiting"]:
        print(f"Find {response_type} responses")
        df = []
        for mouse_id in data.SENSORIUM_OLD:
            # neurons = utils.get_reliable_neurons(
            #     output_dir=output_dir, mouse_id=mouse_id, size=6000
            # )
            print(f"Process mouse {mouse_id}...")
            grating_df = load_grating_result(
                output_dir=output_dir,
                mouse_id=mouse_id,
                response_type=response_type,
            )
            if grating_df is not None:
                df.append(grating_df)
            natural_image_df = load_natural_image_result(
                output_dir=output_dir,
                mouse_id=mouse_id,
                response_type=response_type,
            )
            if natural_image_df is not None:
                df.append(natural_image_df)
            natural_video_df = load_natural_video_result(
                output_dir=output_dir,
                mouse_id=mouse_id,
                response_type=response_type,
            )
            df.append(natural_video_df)
            mei_df = load_generated_result(
                output_dir=output_dir,
                experiment_name=experiment_name,
                stimulus_type="generated_image",
                response_type=response_type,
                mouse_id=mouse_id,
                # neurons=neurons,
            )
            df.append(mei_df)
            mev_df = load_generated_result(
                output_dir=output_dir,
                experiment_name=experiment_name,
                stimulus_type="generated_video",
                response_type=response_type,
                mouse_id=mouse_id,
                # neurons=neurons,
            )
            df.append(mev_df)
        df = pd.concat(df, ignore_index=True)
        plot_response(
            df=df,
            model_name=model_name,
            filename=PLOT_DIR / f"full_field_{response_type}_response.{FORMAT}",
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
