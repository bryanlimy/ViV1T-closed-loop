from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import zscore

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import plot

plot.set_font()


TICK_FONTSIZE = 9
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10

DPI = 180

BLANK_SIZE, PATTERN_SIZE = 10, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

SENSORIUM_DIR = Path("../data/sensorium")

DATA_DIR = Path("../data/contextual_modulation")

PLOT_DIR = Path("figures") / "response_onset"


def load_data(
    output_dir: Path, ds_name: str, mouse_id: str
) -> (np.ndarray, np.ndarray, np.ndarray):
    # load responses in format (trial, neuron, frame)
    filename = (
        output_dir
        / "contextual_modulation"
        / ds_name
        / "responses"
        / f"mouse{mouse_id}.h5"
    )
    trial_ids = h5.get_trial_ids(filename)
    responses = np.stack(h5.get(filename, trial_ids=trial_ids), dtype=np.float32)
    params = np.load(DATA_DIR / ds_name / "meta" / "trials" / "params.npy")
    params = params[trial_ids]  # rearrange params order to match trial_ids

    num_frames = responses.shape[2]
    params = params[:, -num_frames:, :]

    responses = rearrange(
        responses,
        "trial neuron (block frame) -> neuron (trial block) frame",
        frame=BLOCK_SIZE,
    )
    params = rearrange(
        params,
        "trial (block frame) param -> (trial block) param frame",
        frame=BLOCK_SIZE,
    )
    params = params[:, :, BLANK_SIZE]

    # select L2/3 neurons and their size-tune preferences
    df = pd.read_csv(output_dir / "response_amplitudes.csv")
    responsive_df = df.loc[
        (df["mouse"] == mouse_id)
        & (df["classic_tuned"] == True)
        & (df["depth"] >= 200)
        & (df["depth"] <= 300)
    ]
    neurons = responsive_df.neuron.values

    # filter selective neurons and average their responses over repeats
    center_responses, cross_responses, shift_responses = [], [], []
    for neuron in neurons:
        center_ids = np.where((params[:, 0] == 20) & (params[:, 2] == 0))[0]
        center_responses.append(np.mean(responses[neuron, center_ids, :], axis=0))

        cross_ids = np.where((params[:, 0] == 20) & (params[:, 2] == 2))[0]
        cross_responses.append(np.mean(responses[neuron, cross_ids, :], axis=0))

        shift_ids = np.where((params[:, 0] == 20) & (params[:, 2] == 3))[0]
        shift_responses.append(np.mean(responses[neuron, shift_ids, :], axis=0))

    center_responses = np.array(center_responses, dtype=np.float32)
    cross_responses = np.array(cross_responses, dtype=np.float32)
    shift_responses = np.array(shift_responses, dtype=np.float32)
    return center_responses, cross_responses, shift_responses


def get_color(i: int):
    match i:
        case 0:
            color = "dodgerblue"
            label = "Center"
        case 1:
            color = "orangered"
            label = "Cross"
        case 2:
            color = "forestgreen"
            label = "Shift"
        case _:
            raise ValueError("Invalid case")
    return color, label


def filter_neurons(
    center_responses: np.ndarray,
    cross_responses: np.ndarray,
    shift_responses: np.ndarray,
    threshold: float = 3.29,
):
    # concat classical and inverse responses in the frame dimension
    responses = np.concat([center_responses, cross_responses, shift_responses], axis=1)
    # responses = np.concat(
    #     [
    #         center_responses[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE],
    #         cross_responses[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE],
    #         shift_responses[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE],
    #     ],
    #     axis=1,
    # )
    # compute z-score over frame dimension
    z_scores = zscore(responses, axis=1)

    # get z-score in the center stimulus presentation window
    start = 0
    center_z_scores = z_scores[
        :, start + BLANK_SIZE : start + BLANK_SIZE + PATTERN_SIZE
    ]

    # get z-score in the cross stimulus presentation window
    start += BLOCK_SIZE
    cross_z_scores = z_scores[:, start + BLANK_SIZE : start + BLANK_SIZE + PATTERN_SIZE]

    # get z-score in the shift stimulus presentation window
    start += BLOCK_SIZE
    shift_z_scores = z_scores[:, start + BLANK_SIZE : start + BLANK_SIZE + PATTERN_SIZE]

    # center_z_scores = z_scores[:, :PATTERN_SIZE]
    # cross_z_scores = z_scores[:, PATTERN_SIZE : 2 * PATTERN_SIZE]
    # shift_z_scores = z_scores[:, 2 * PATTERN_SIZE : 3 * PATTERN_SIZE]

    # find neurons that has onset response above threshold to center stimulus
    center_neurons = np.where(np.max(center_z_scores, axis=1) >= threshold)[0]
    # find neurons that has onset response above threshold to cross stimulus
    cross_neurons = np.where(np.max(cross_z_scores, axis=1) >= threshold)[0]
    # find neurons that has onset response above threshold to shift stimulus
    shift_neurons = np.where(np.max(shift_z_scores, axis=1) >= threshold)[0]

    # find neurons that has onset response above threshold to all stimuli
    # neurons = reduce(np.intersect1d, [center_neurons, cross_neurons, shift_neurons])
    # neurons = reduce(np.union1d, [center_neurons, cross_neurons, shift_neurons])

    # center_timings = np.zeros(len(neurons), dtype=np.float32)
    # cross_timings = np.zeros_like(center_timings)
    # shift_timings = np.zeros_like(center_timings)
    # for i, neuron in enumerate(neurons):
    #     center_onset = np.where(center_z_scores[neuron] >= threshold)[0]
    #     center_timings[i] = center_onset[0] + BLANK_SIZE
    #     cross_onset = np.where(cross_z_scores[neuron] >= threshold)[0]
    #     cross_timings[i] = cross_onset[0] + BLANK_SIZE
    #     shift_onset = np.where(shift_z_scores[neuron] >= threshold)[0]
    #     shift_timings[i] = shift_onset[0] + BLANK_SIZE
    center_timings = np.zeros(len(center_neurons), dtype=np.float32)
    for i, neuron in enumerate(center_neurons):
        center_onset = np.where(center_z_scores[neuron] >= threshold)[0]
        center_timings[i] = center_onset[0] + BLANK_SIZE

    cross_timings = np.zeros(len(cross_neurons), dtype=np.float32)
    for i, neuron in enumerate(cross_neurons):
        cross_onset = np.where(cross_z_scores[neuron] >= threshold)[0]
        cross_timings[i] = cross_onset[0] + BLANK_SIZE

    shift_timings = np.zeros(len(shift_neurons), dtype=np.float32)
    for i, neuron in enumerate(shift_neurons):
        shift_onset = np.where(shift_z_scores[neuron] >= threshold)[0]
        shift_timings[i] = shift_onset[0] + BLANK_SIZE

    return (
        center_responses[center_neurons],
        cross_responses[cross_neurons],
        shift_responses[shift_neurons],
        center_timings,
        cross_timings,
        shift_timings,
    )


def plot_response_time(
    center_responses: np.ndarray,
    cross_responses: np.ndarray,
    shift_responses: np.ndarray,
    filename: Path,
    mouse_id: str = None,
    title: str = None,
):
    (
        center_responses,
        cross_responses,
        shift_responses,
        center_onsets,
        cross_onsets,
        shift_onsets,
    ) = filter_neurons(
        center_responses=center_responses,
        cross_responses=cross_responses,
        shift_responses=shift_responses,
    )
    # print(f"\t\tNumber of neurons: {len(neurons)}")
    # if len(center_responses) == 0:
    #     return

    # center_responses = center_responses[neurons]
    # cross_responses = cross_responses[neurons]
    # shift_responses = shift_responses[neurons]
    num_neurons = [len(center_responses), len(cross_responses), len(shift_responses)]

    # normalize individual responses
    center_responses = center_responses / np.max(center_responses, axis=1)[:, None]
    cross_responses = cross_responses / np.max(cross_responses, axis=1)[:, None]
    shift_responses = shift_responses / np.max(shift_responses, axis=1)[:, None]

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5), dpi=DPI)

    x = np.arange(center_responses.shape[-1])
    min_value, max_value = np.inf, -np.inf
    for i, response in [
        (0, center_responses),
        (1, cross_responses),
        (2, shift_responses),
    ]:
        color, label = get_color(i)
        ax.plot(
            x,
            np.mean(response, axis=0),
            color=color,
            linewidth=2,
            alpha=0.8,
            zorder=i + 2,
            clip_on=False,
            label=label,
        )
        ax.fill_between(
            x,
            y1=np.mean(response, axis=0) - sem(response, axis=0),
            y2=np.mean(response, axis=0) + sem(response, axis=0),
            facecolor=color,
            edgecolor="none",
            alpha=0.4,
            zorder=i + 1,
            clip_on=False,
        )
        # for n in range(response.shape[0]):
        #     ax.plot(
        #         x,
        #         response[n],
        #         color=color,
        #         linewidth=0.8,
        #         alpha=0.05,
        #         clip_on=False,
        #         zorder=1,
        #     )
        min_value = min(min_value, np.min(response))
        max_value = max(max_value, np.max(response))

    # get the next multiple of 2
    # max_value = 2 * np.ceil(max_value / 2)
    max_value = 1

    print(
        f"\t\tCenter onset mean: {np.mean(center_onsets):.02f} "
        f"median: {np.median(center_onsets):.02f}\n"
        f"\t\tCross onset mean: {np.mean(cross_onsets):.02f} "
        f"median: {np.median(cross_onsets):.02f}\n"
        f"\t\tShift onset mean: {np.mean(shift_onsets):.02f} "
        f"median: {np.median(shift_onsets):.02f}"
    )

    for i, onset in enumerate((center_onsets, cross_onsets, shift_onsets)):
        color, _ = get_color(i)
        # plot triangle to indicate preferred size
        ax.scatter(
            x=np.median(onset),
            y=max_value,
            s=25,
            marker="v",
            facecolors=color,
            edgecolors="none",
            alpha=0.9,
            clip_on=False,
        )

    ax.axvspan(
        xmin=BLANK_SIZE,
        xmax=BLANK_SIZE + PATTERN_SIZE,
        facecolor="#e0e0e0",
        edgecolor="none",
        zorder=-1,
    )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(2, 0.98 * max_value),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="",
        handletextpad=0.35,
        handlelength=0.6,
        labelspacing=0.2,
        columnspacing=1,
        borderpad=0,
        borderaxespad=0,
        bbox_transform=ax.transData,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    x_ticks = np.linspace(0, BLOCK_SIZE, 6, dtype=int)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    max_time = ceil((1000 * BLOCK_SIZE / FPS) / 100) * 100
    time_ticks = np.linspace(0, max_time, 6, dtype=int)
    plot.set_xticks(
        ax,
        # ticks=np.linspace(0, (max_time / 1000) * FPS, len(time_ticks)),
        # tick_labels=time_ticks,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="Frame",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.arange(0, max_value + 0.2, 0.2)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="Response (norm. ΔF/F)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(2))

    plot.set_ticks_params(ax, length=2, pad=1, linewidth=1)
    sns.despine(ax=ax)

    if mouse_id is not None:
        ax.text(
            x=np.max(x_ticks) - 1,
            y=0.01 * max_value,
            s=f"Mouse {mouse_id}",
            fontsize=TICK_FONTSIZE,
            va="bottom",
            ha="right",
            zorder=10,
        )
    ax.text(
        x=1,
        y=0.01 * max_value,
        s=f"N={', '.join(map(str, num_neurons))}",
        fontsize=TICK_FONTSIZE,
        va="bottom",
        ha="left",
        zorder=10,
    )

    # ax.text(
    #     x=2,
    #     y=0.06 * max_value,
    #     s=f"mouse {mouse_id}\nstimulus size {stim_size}°",
    #     fontsize=TICK_FONTSIZE,
    #     va="bottom",
    #     ha="left",
    # )

    if title is not None:
        ax.set_title(title, fontsize=LABEL_FONTSIZE, pad=2)

    # Add inset boxplot for onset timing
    axin = inset_axes(
        parent_axes=ax,
        width=0.5,
        height=0.6,
        loc="upper right",
        bbox_to_anchor=(0.95 * 50, 0.9 * max_value),
        bbox_transform=ax.transData,
    )
    box_x_ticks = np.array([0, 0.4, 0.8])
    box_linewidth = 1
    bp = axin.boxplot(
        [center_onsets, cross_onsets, shift_onsets],
        notch=False,
        vert=True,
        positions=box_x_ticks,
        widths=0.2,
        showfliers=True,
        showmeans=True,
        boxprops={"linewidth": box_linewidth, "clip_on": False},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
        capprops={"linewidth": box_linewidth, "clip_on": False},
        whiskerprops={"linewidth": box_linewidth, "clip_on": False},
        meanprops={
            "markersize": 4,
            "markerfacecolor": "gold",
            "markeredgecolor": "black",
            "markeredgewidth": 0.75,
            "clip_on": False,
        },
        medianprops={
            "color": "red",
            "solid_capstyle": "projecting",
            "linewidth": 1,
            "clip_on": False,
        },
    )
    axin.set_xlim(box_x_ticks[0] - 0.2, box_x_ticks[-1] + 0.2)
    plot.set_xticks(
        axis=axin,
        ticks=box_x_ticks,
        tick_labels=["Center", "Cross", "Shift"],
        tick_fontsize=TICK_FONTSIZE,
        label_pad=0,
        rotation=270,
        va="top",
    )
    box_min_value = min([whi.get_ydata()[1] for whi in bp["whiskers"]]) - 1
    box_min_value = max(0, 2 * np.floor(box_min_value / 2))
    box_max_value = max([whi.get_ydata()[1] for whi in bp["whiskers"]]) + 1
    box_max_value = 2 * np.ceil(box_max_value / 2)
    box_min_value = 10
    box_max_value = 40
    box_y_ticks = np.linspace(box_min_value, box_max_value, 2)
    axin.set_ylim(box_y_ticks[0], box_y_ticks[-1])
    plot.set_yticks(
        axis=axin,
        ticks=box_y_ticks,
        tick_labels=box_y_ticks.astype(int),
        label="Onset\n(frame)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-2,
        linespacing=0.9,
    )
    sns.despine(ax=axin)
    axin.yaxis.set_minor_locator(MultipleLocator(2))
    plot.set_ticks_params(axis=axin, length=2, pad=1, linewidth=1)
    height = box_max_value - box_min_value
    try:
        p_value = ttest_ind(center_onsets, cross_onsets).pvalue
        plot.add_p_value(
            ax=axin,
            x0=box_x_ticks[0],
            x1=box_x_ticks[1],
            y=1.1 * height + box_min_value,
            p_value=p_value,
            fontsize=TICK_FONTSIZE,
            tick_length=0.04 * height,
            tick_linewidth=0.8,
            text_pad=0.1 * height,
        )
    except Exception as e:
        print(f"Exception: {e}")

    try:
        p_value = ttest_ind(cross_onsets, shift_onsets).pvalue
        plot.add_p_value(
            ax=axin,
            x0=box_x_ticks[1],
            x1=box_x_ticks[2],
            y=1.1 * height + box_min_value,
            p_value=p_value,
            fontsize=TICK_FONTSIZE,
            tick_length=0.04 * height,
            tick_linewidth=0.8,
            text_pad=0.1 * height,
        )
    except Exception as e:
        print(f"Exception: {e}")
    try:
        p_value = ttest_ind(center_onsets, shift_onsets).pvalue
        plot.add_p_value(
            ax=axin,
            x0=box_x_ticks[0],
            x1=box_x_ticks[2],
            y=1.3 * height + box_min_value,
            p_value=p_value,
            fontsize=TICK_FONTSIZE,
            tick_length=0.04 * height,
            tick_linewidth=0.8,
            text_pad=0.1 * height,
        )
    except Exception as e:
        print(f"Exception: {e}")

    plot.save_figure(figure, filename=filename, dpi=4 * DPI)


def main():
    models = {
        # "LN": Path("../runs/fCNN/015_linear_fCNN"),
        # "fCNN no behavior": Path("../runs/fCNN/030_fCNN_noBehavior"),
        # "fCNN": Path("../runs/fCNN/029_fCNN_noClipGrad"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        # "ViV1T": Path("../runs/vivit/159_viv1t_elu"),
        "ViV1T_causal": Path("../runs/vivit/172_viv1t_causal"),
    }
    ds_name = "dynamic_high_contrast"

    for name, output_dir in models.items():
        print(f"Processing {name}...")
        center_responses = []
        cross_responses = []
        shift_responses = []

        for mouse_id in data.SENSORIUM_OLD:
            print(f"\tProcess mouse {mouse_id}...")
            center_response, cross_response, shift_response = load_data(
                output_dir=output_dir, ds_name=ds_name, mouse_id=mouse_id
            )
            plot_response_time(
                center_responses=center_response,
                cross_responses=cross_response,
                shift_responses=shift_response,
                mouse_id=mouse_id,
                filename=PLOT_DIR / name / f"mouse{mouse_id}.svg",
            )
            center_responses.append(center_response)
            cross_responses.append(cross_response)
            shift_responses.append(shift_response)

        center_responses = np.concat(center_responses, axis=0)
        cross_responses = np.concat(cross_responses, axis=0)
        shift_responses = np.concat(shift_responses, axis=0)

        print("\tPull neurons from all mice.")
        plot_response_time(
            center_responses=center_responses,
            cross_responses=cross_responses,
            shift_responses=shift_responses,
            filename=PLOT_DIR / name / "response_onset.svg",
        )


if __name__ == "__main__":
    main()
