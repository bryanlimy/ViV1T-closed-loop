from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from scipy.stats import wilcoxon

from viv1t import metrics
from viv1t.utils import plot

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "jpg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

plot.set_font()

PLOT_DIR = Path("figures") / "cross_session_reliability"


def load_data(data_dir: Path) -> dict[int, np.ndarray]:
    tiers = np.load(data_dir / "meta" / "trials" / "tiers.npy", allow_pickle=True)
    # load trials that were presented on the second day as well
    trial_ids = np.where(tiers == "live_main")[0]
    responses = [
        np.load(data_dir / "data" / "responses" / f"{trial_id}.npy")
        for trial_id in trial_ids
    ]
    responses = np.stack(responses)
    video_ids = np.load(data_dir / "meta" / "trials" / "video_ids.npy")
    video_ids = video_ids[trial_ids]
    # group responses by video ID
    responses_ = {}
    for video_id in np.unique(video_ids):
        trial_ids = np.where(video_ids == video_id)[0]
        responses_[video_id] = rearrange(
            responses[trial_ids],
            "repeat neuron frame -> neuron repeat frame",
        )
    return responses_


def compute_across_day_correlation(
    responses_day1: dict[int, np.ndarray],
    responses_day2: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = []
    video_ids = list(responses_day1.keys())
    for video_id in video_ids:
        # average response over repeats
        responses1 = np.mean(responses_day1[video_id], axis=1)
        responses2 = np.mean(responses_day2[video_id], axis=1)
        # compute correlation for each neuron
        corr = metrics.correlation(
            y1=responses1[None, ...],
            y2=responses2[None, ...],
        )[0]
        df.append(
            pd.DataFrame(
                {"video_id": video_id, "neuron": np.arange(len(corr)), "corr": corr}
            )
        )
    df = pd.concat(df, ignore_index=True)
    df.insert(loc=0, column="type", value="across")
    return df


def compute_within_day_correlation(
    responses: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Compute the correlation between the responses of the first 5 trials and
    last 5 trials of the same video on the same recording session.
    """
    df = []
    video_ids = list(responses.keys())
    for video_id in video_ids:
        response = responses[video_id]
        # index = np.arange(response.shape[1])
        # index = rng.permutation(index)
        # response = response[:, index]
        # average response over 5 repeats
        responses1 = np.mean(response[:, :5], axis=1)
        responses2 = np.mean(response[:, 5:], axis=1)
        # compute correlation for each neuron
        corr = metrics.correlation(
            y1=responses1[None, ...],
            y2=responses2[None, ...],
        )[0]
        df.append(
            pd.DataFrame(
                {"video_id": video_id, "neuron": np.arange(len(corr)), "corr": corr}
            )
        )
    df = pd.concat(df, ignore_index=True)
    # drop neurons that were not matched from day 1
    df = df.dropna()
    df.insert(loc=0, column="type", value="within")
    return df


def plot_reliability(df: pd.DataFrame, filename: Path):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3), dpi=DPI)

    video_ids = df.video_id.unique()
    x_ticks = np.arange(len(video_ids))
    width = 0.8
    sns.violinplot(
        data=df,
        x="video_id",
        y="corr",
        inner="quart",
        palette=["orangered", "dodgerblue"],
        hue="type",
        split=True,
        fill=False,
        order=video_ids,
        width=width,
        linewidth=1,
        linecolor="black",
        ax=ax,
    )
    sns.move_legend(
        ax,
        loc="lower center",
        bbox_to_anchor=((x_ticks[-1] - x_ticks[0]) / 2, -1.05),
        bbox_transform=ax.transData,
        title="",
        ncol=2,
        frameon=False,
        fontsize=TICK_FONTSIZE,
    )

    ax.scatter(
        x=x_ticks - (width / 4),
        y=df[df["type"] == "across"].groupby(by="video_id")["corr"].mean(),
        s=20,
        marker="v",
        facecolor="gold",
        edgecolor="black",
        clip_on=False,
        zorder=20,
    )
    ax.scatter(
        x=x_ticks + (width / 4),
        y=df[df["type"] == "within"].groupby(by="video_id")["corr"].mean(),
        s=20,
        marker="v",
        facecolor="gold",
        edgecolor="black",
        clip_on=False,
        zorder=20,
    )
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=video_ids,
        label="Video ID",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    ax.set_ylim(-1, 1)
    y_ticks = np.linspace(-1, 1, 9)
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, decimals=2),
        label="Correlation",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    ax.set_title(
        "Correlation coefficient between Day 1 and 4 responses\nto 10 unique test videos",
        fontsize=LABEL_FONTSIZE,
        pad=0,
    )
    ax.text(
        x=-0.4,
        y=-0.97,
        s=f"N={df.neuron.nunique()}",
        va="bottom",
        ha="left",
        fontsize=TICK_FONTSIZE,
    )
    sns.despine(ax=ax)
    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def find_matched_neurons(responses_day2: dict[int, np.ndarray]) -> np.ndarray:
    responses = list(responses_day2.values())[0]
    neurons = np.where(np.all(~np.isnan(responses), axis=(1, 2)))[0]
    return neurons


def main():
    data_dir = Path("../data/rochefort-lab")
    output_dir = Path("../runs/rochefort-lab/vivit/015_causal_viv1t_FOV2_finetune")

    data_dir_day1 = data_dir / "VIPcre232_FOV2_day1"
    responses_day1 = load_data(data_dir=data_dir_day1)

    data_dir_day2 = data_dir / "VIPcre232_FOV2_day4" / "natural_movies"
    responses_day2 = load_data(data_dir=data_dir_day2)
    assert responses_day1.keys() == responses_day2.keys()

    neurons = find_matched_neurons(responses_day2)

    rng = np.random.default_rng(1234)
    across_day_df = compute_across_day_correlation(
        responses_day1=responses_day1,
        responses_day2=responses_day2,
        rng=rng,
    )
    within_day_correlation = compute_within_day_correlation(
        responses=responses_day1,
        rng=rng,
    )
    df = pd.concat([across_day_df, within_day_correlation], ignore_index=True)

    # only keep neurons that are matched between days
    df = df[df["neuron"].isin(neurons)]

    plot_reliability(
        df=df,
        filename=PLOT_DIR
        / "VIPcre232_FOV2_day1_day4"
        / f"average_reliability.{FORMAT}",
    )

    cross_session_reliability_df = (
        across_day_df.groupby(by="neuron")["corr"].mean().reset_index()
    )
    cross_session_reliability_df.to_parquet(
        output_dir / "cross_session_reliability.parquet"
    )
    print(f"Saved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
