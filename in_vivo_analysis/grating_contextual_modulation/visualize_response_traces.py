import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from scipy.stats import sem
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import stimulus

THRESHOLD = 0.2  # response has to be 20% stronger

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches
FPS = 30

plot.set_font()

PLOT_DIR = Path("figures") / "calcium_traces"
DATA_DIR = Path("../../data")
OUTPUT_DIR = Path("../../runs")


BLANK_SIZE, PATTERN_SIZE = 15, 30
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2


VIPcre232_FOV1_VIDEO_IDS = [
    ["CM000", "CM004"],  # low contrast center
    ["CM001", "CM005"],  # low contrast iso
    ["CM002", "CM006"],  # low contrast cross
    ["CM008", "CM012"],  # high contrast center
    ["CM009", "CM013"],  # high contrast iso
    ["CM010", "CM014"],  # high contrast cross
]

VIPcre232_FOV2_1_VIDEO_IDS = [
    ["CM004", "CM012"],  # low contrast center (↑ or ↓)
    ["CM005", "CM013"],  # low contrast iso
    ["CM006", "CM014"],  # low contrast cross
    ["CM020", "CM028"],  # high contrast center (↑ or ↓)
    ["CM021", "CM029"],  # high contrast iso
    ["CM022", "CM030"],  # high contrast cross
]

VIPcre232_FOV2_2_VIDEO_IDS = [
    ["CM000", "CM008"],  # low contrast center (← or →)
    ["CM001", "CM009"],  # low contrast iso
    ["CM002", "CM010"],  # low contrast cross
    ["CM016", "CM024"],  # high contrast center (← or →)
    ["CM017", "CM025"],  # high contrast iso
    ["CM018", "CM026"],  # high contrast cross
]

VIPcre233_FOV1_VIDEO_IDS = [
    ["CM000", "CM003"],  # low contrast center
    ["CM001", "CM004"],  # low contrast iso
    ["CM002", "CM005"],  # low contrast cross
    ["CM006", "CM009"],  # high contrast center
    ["CM007", "CM010"],  # high contrast iso
    ["CM008", "CM011"],  # high contrast cross
]

VIPcre233_FOV2_VIDEO_IDS = [
    ["CM000", "CM003"],  # low contrast center
    ["CM001", "CM004"],  # low contrast iso
    ["CM002", "CM005"],  # low contrast cross
    ["CM006", "CM009"],  # high contrast center
    ["CM007", "CM010"],  # high contrast iso
    ["CM008", "CM011"],  # high contrast cross
]


def select_responsive_neurons(responses: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # select high contrast response only
        responses = responses[:, 3, :, :]
        # average response over repeats
        responses = np.nanmean(responses, axis=1)
        # select response before presentation and average over the 4 conditions
        before = responses[:, :BLANK_SIZE]
        # set threshold to 3 times the standard deviation of grey response plus mean.
        threshold = np.nanmean(before, axis=1) + 2 * np.nanstd(before, axis=1)
        # select response during presentation
        during = responses[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
        # get maximum response for each neuron during presentation
        max_responses = np.nanmax(during, axis=1)
        neurons = np.where(max_responses >= threshold)[0]
    return neurons


def select_RF_neurons(output_dir: Path, mouse_id: str) -> np.ndarray:
    """
    Return neurons with estimated RF center is within 10° of the stimulus center
    """
    population_RF = stimulus.load_population_RF_center(
        output_dir=output_dir, mouse_id=mouse_id
    )
    monitor_info = data.MONITOR_INFO[mouse_id]
    circular_mask = stimulus.create_circular_mask(
        stimulus_size=10,
        center=population_RF,
        monitor_width=monitor_info["width"],
        monitor_height=monitor_info["height"],
        monitor_distance=monitor_info["distance"],
        num_frames=1,
    )  # (C, T, H, W)
    circular_mask = circular_mask[0, 0]  # remove time and channel dimensions
    neuron_RFs = pd.read_parquet(output_dir / "aRF.parquet")
    RF_x = neuron_RFs.center_x.values
    RF_y = neuron_RFs.center_y.values
    bad_RFs = neuron_RFs.bad_fit.values
    # replace bad RF fits with 0
    bad_RF_x = np.where((RF_x < 0) | (RF_x > monitor_info["width"]))[0]
    bad_RF_y = np.where((RF_y < 0) | (RF_y > monitor_info["height"]))[0]
    bad_RFs[bad_RF_x], bad_RFs[bad_RF_y] = True, True
    RF_x[bad_RFs], RF_y[bad_RFs] = 0, 0
    # round RF to nearest pixel
    RF_x = np.round(RF_x, decimals=0).astype(int)
    RF_y = np.round(RF_y, decimals=0).astype(int)
    # check if neuron RF is in 20 degree of population RF
    within = circular_mask[RF_y, RF_x]
    within[bad_RFs] = False
    RF_neurons = np.where(within == True)[0]
    return RF_neurons


def filter_neurons(
    output_dir: Path, mouse_id: str, responses: np.ndarray
) -> np.ndarray:
    neurons = np.arange(responses.shape[0], dtype=int)
    responsive_neurons = select_responsive_neurons(responses=responses.copy())
    neurons = np.intersect1d(neurons, responsive_neurons)
    RF_neurons = select_RF_neurons(output_dir=output_dir, mouse_id=mouse_id)
    neurons = np.intersect1d(neurons, RF_neurons)
    return neurons


def load_recorded_data(
    data_dir: Path,
    video_id_groups: list[list[str]],
    output_dir: Path,
    mouse_id: str,
) -> dict[str, np.ndarray]:
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

    matched_neurons = np.where(~np.any(np.isnan(responses), axis=(1, 2)))[0]

    num_neurons = responses.shape[0]
    num_repeats = 20
    num_frames = responses.shape[2]
    responses_ = np.full(
        shape=(num_neurons, len(video_id_groups), num_repeats, num_frames),
        fill_value=np.nan,
        dtype=np.float32,
    )
    for i in range(len(video_id_groups)):
        indexes = np.where(np.isin(video_ids, video_id_groups[i]))[0]
        assert len(indexes) >= 5
        responses_[:, i, : len(indexes)] = responses[:, indexes]
    responses = responses_.copy()
    del responses_, num_repeats, num_frames, video_ids

    selective_neurons = filter_neurons(
        output_dir=output_dir,
        mouse_id=mouse_id,
        responses=responses.copy(),
    )
    # selective_neurons = np.arange(num_neurons, dtype=int)
    neurons = np.intersect1d(matched_neurons, selective_neurons)

    # select response during presentation
    # responses = responses[:, :, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    # # sum response during presentation
    # responses = np.sum(responses, axis=3)

    responses_ = {
        "low_center": responses[neurons, 0],
        "low_iso": responses[neurons, 1],
        "low_cross": responses[neurons, 2],
        "high_center": responses[neurons, 3],
        "high_iso": responses[neurons, 4],
        "high_cross": responses[neurons, 5],
    }

    return responses_


def plot_traces(responses: dict[str, np.ndarray], neuron: int, filename: Path):
    figure_width = (1 / 4) * PAPER_WIDTH
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(figure_width, figure_width),
        gridspec_kw={
            "wspace": 0.2,
            "hspace": 0.5,
            "top": 0.88,
            "bottom": 0.12,
            "left": 0.25,
            "right": 0.96,
        },
        height_ratios=[1, 1],
        dpi=DPI,
    )

    contrast_types = ["high", "low"]
    stimulus_types = ["center", "iso", "cross"]

    x_ticks = np.arange(responses["high_center"].shape[1])

    min_value, max_value = np.inf, -np.inf
    for i, contrast_type in enumerate(contrast_types):
        for j, stimulus_type in enumerate(stimulus_types):
            response = responses[f"{contrast_type}_{stimulus_type}"]
            # compute trial-averaged response
            value = np.nanmean(response, axis=0)
            error = sem(response, axis=0, nan_policy="omit")
            min_value = min(min_value, np.min(value - error))
            max_value = max(max_value, np.max(value + error))

            axes[i, j].plot(
                x_ticks,
                value,
                color="black",
                linewidth=1.2,
                alpha=0.8,
                clip_on=False,
                zorder=1,
            )
            axes[i, j].fill_between(
                x_ticks,
                y1=value - error,
                y2=value + error,
                facecolor="black",
                edgecolor="none",
                linewidth=2,
                alpha=0.3,
                zorder=1,
                clip_on=False,
            )
            axes[i, j].axvspan(
                xmin=BLANK_SIZE,
                xmax=BLANK_SIZE + PATTERN_SIZE,
                facecolor="#e0e0e0",
                edgecolor="none",
                zorder=-1,
            )

            if i == 1:
                if stimulus_type == "center":
                    stimulus_type = "centre"
                axes[i, j].set_xlabel(
                    stimulus_type.capitalize(),
                    fontsize=TICK_FONTSIZE,
                    labelpad=1.5,
                )

    for i, contrast_type in enumerate(contrast_types):
        for j, stimulus_type in enumerate(stimulus_types):
            axes[i, j].set_xlim(x_ticks[0] - 1, x_ticks[-1] + 1)
            axes[i, j].set_ylim(min_value, max_value)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            sns.despine(ax=axes[i, j], left=True, bottom=True)

    x_offset = -45
    # response scale bar
    y = max_value
    axes[1, 0].plot(
        [x_offset, x_offset],
        [min_value, y],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    axes[1, 0].text(
        x=x_offset - 3,
        y=min_value,
        s=f"{y:.01f}ΔF/F",
        fontsize=TICK_FONTSIZE,
        rotation=90,
        va="bottom",
        ha="right",
    )

    # timescale bar
    axes[1, 0].plot(
        [x_offset, x_offset + FPS],
        [min_value, min_value],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    axes[1, 0].text(
        x=x_offset + FPS / 2,
        y=min_value - (0.05 * max_value),
        s="1s",
        fontsize=TICK_FONTSIZE,
        va="top",
        ha="center",
    )

    axes[0, 0].text(
        x=0,
        y=1.02 * max_value,
        s=f"High contrast",
        fontsize=LABEL_FONTSIZE,
        va="bottom",
        ha="left",
    )
    axes[1, 0].text(
        x=0,
        y=1.02 * max_value,
        s=f"Low contrast",
        fontsize=LABEL_FONTSIZE,
        va="bottom",
        ha="left",
    )

    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def find_flip_neurons(responses: dict[str, np.ndarray]) -> np.ndarray:
    # select response during stimulus presentation
    start, end = BLANK_SIZE, BLANK_SIZE + PATTERN_SIZE
    low_iso = responses["low_iso"][:, :, start:end]
    low_cross = responses["low_cross"][:, :, start:end]

    high_iso = responses["high_iso"][:, :, start:end]
    high_cross = responses["high_cross"][:, :, start:end]

    # sum response over presentation and average over repeats
    low_iso = np.nanmean(np.sum(low_iso, axis=-1), axis=-1)
    low_cross = np.nanmean(np.sum(low_cross, axis=-1), axis=-1)

    high_iso = np.nanmean(np.sum(high_iso, axis=-1), axis=-1)
    high_cross = np.nanmean(np.sum(high_cross, axis=-1), axis=-1)

    high_contrast = ((high_cross - high_iso) / high_iso) >= THRESHOLD
    low_contrast = ((low_iso - low_cross) / low_cross) >= THRESHOLD

    flip_neurons = (high_contrast == True) & (low_contrast == True)
    return flip_neurons


def main():
    output_dir = OUTPUT_DIR / "rochefort-lab" / "vivit"

    for mouse_id, day_name, output_dir_name, video_id_groups in tqdm(
        [
            (
                "K",
                "day2",
                "003_causal_viv1t_finetune",
                VIPcre232_FOV1_VIDEO_IDS,
            ),
            (
                "L",
                "day3",
                "015_causal_viv1t_FOV2_finetune",
                VIPcre232_FOV2_1_VIDEO_IDS,
            ),
            (
                "M",
                "day2",
                "018_causal_viv1t_VIPcre233_FOV1_finetune",
                VIPcre233_FOV1_VIDEO_IDS,
            ),
            (
                "N",
                "day2",
                "025_causal_viv1t_VIPcre233_FOV2_finetune",
                VIPcre233_FOV2_VIDEO_IDS,
            ),
        ],
        desc="Load Rochefort Lab",
    ):
        responses = load_recorded_data(
            data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "artificial_movies",
            video_id_groups=video_id_groups,
            output_dir=output_dir / output_dir_name,
            mouse_id=mouse_id,
        )
        flip_neurons = find_flip_neurons(responses=responses)
        for neuron in tqdm(np.where(flip_neurons)[0], desc=f"mouse {mouse_id}"):
            plot_traces(
                responses={k: v[neuron] for k, v in responses.items()},
                neuron=neuron,
                filename=PLOT_DIR
                / f"mouse{mouse_id}"
                / f"mouse{mouse_id}_neuron{neuron:04d}.{FORMAT}",
            )

    print(f"Saved result to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
