import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib import axes
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import kruskal
from scipy.stats import median_test
from scipy.stats import ranksums
from scipy.stats import sem
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon
from scipy.stats import zscore
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot

plot.set_font()

THRESHOLD = 1.0  # z-score threshold for response onset

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLANK_SIZE, PATTERN_SIZE = 15, 15
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

PLOT_DIR = Path("figures") / "response_onset"

STIMULUS_SIZE = 20  # stimulus size to estimate response onset


def load_data(output_dir: Path, mouse_id: str) -> (np.ndarray, np.ndarray):
    save_dir = output_dir / "feedbackRF_onset"
    responses = np.load(save_dir / f"mouse{mouse_id}.npz", allow_pickle=False)["data"]
    responses = rearrange(responses, "N trial pattern T -> N (trial pattern) T")

    parameters = np.load(save_dir / "parameters.npy", allow_pickle=False)
    parameters = rearrange(parameters, "trial pattern param -> (trial pattern) param")

    # filter selective neurons and stimulus size 20 as per Keller et al. 2020:
    #   Response dynamics. To estimate the temporal response profile to inverse
    #   stimuli (Fig. 3), we presented patches of gratings and inverse gratings
    #   at a single size (1,000 trials each). These gratings were presented
    #   either at 15° or 20°, for 0.5 s interleaved by 1 s of grey screen. The
    #   initial phase of the drifting gratings was randomized to avoid
    #   overestimating the onset delay of the response for simple-cell-like
    #   receptive fields.
    # get response to classical stimulus
    classic_indices = np.where(
        (parameters[:, 0] == STIMULUS_SIZE) & (parameters[:, 2] == 0)
    )[0]
    classical_responses = rearrange(
        responses[:, classic_indices, :], "N repeat T -> N T repeat"
    )
    # get response to inverse stimulus
    inverse_indices = np.where(
        (parameters[:, 0] == STIMULUS_SIZE) & (parameters[:, 2] == 1)
    )[0]
    inverse_responses = rearrange(
        responses[:, inverse_indices, :], "N repeat T -> N T repeat"
    )
    return classical_responses, inverse_responses


def z_scores_to_onsets(z_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Find the first frame that exceeds the z-score threshold for each neuron"""
    onsets = np.full((z_scores.shape[0],), fill_value=np.nan, dtype=np.float32)
    neuron_indexes, frame_indexes = np.where(z_scores >= threshold)
    neurons, first_frame = np.unique(neuron_indexes, return_index=True)
    onsets[neurons] = frame_indexes[first_frame]
    # convert onset from frame to seconds
    onsets = onsets / FPS
    return onsets


def estimate_response_onsets(
    classical_responses: np.ndarray,
    inverse_responses: np.ndarray,
    threshold: float = THRESHOLD,
) -> (np.ndarray, np.ndarray):
    assert classical_responses.shape == inverse_responses.shape
    assert classical_responses.ndim == 3
    # compute response average over repeats
    classical_responses = np.mean(classical_responses, axis=-1)
    inverse_responses = np.mean(inverse_responses, axis=-1)
    # indexes of the presentation window
    start, end = BLANK_SIZE, BLANK_SIZE + PATTERN_SIZE
    # concatenate response to classical and inverse response over the frame dimension
    responses = np.concat([classical_responses, inverse_responses], axis=1)
    # compute z-score over frame dimension for responses to classical and inverse stimuli
    z_scores = zscore(responses, axis=1)
    # extract z-scores during stimulus presentation
    classical_z_scores = z_scores[:, :BLOCK_SIZE][:, start:end]
    classical_onsets = z_scores_to_onsets(
        z_scores=classical_z_scores,
        threshold=threshold,
    )
    # perform the same test over the response to inverse stimuli
    inverse_z_scores = z_scores[:, BLOCK_SIZE:][:, start:end]
    inverse_onsets = z_scores_to_onsets(
        z_scores=inverse_z_scores,
        threshold=threshold,
    )
    return classical_onsets, inverse_onsets


def plot_response_inset(
    ax: axes.Axes,
    classical_responses: np.ndarray,
    inverse_responses: np.ndarray,
    classical_onsets: np.ndarray,
    inverse_onsets: np.ndarray,
    show_median: bool = True,
):
    """
    Plot the response 5 frames before and the first half (i.e. 0.5s) of the
    stimulus presentation
    """
    assert classical_responses.shape == inverse_responses.shape
    if classical_responses.ndim == 3:
        # compute average response over repeat
        classical_responses = np.mean(classical_responses, axis=-1)
        inverse_responses = np.mean(inverse_responses, axis=-1)
        # normalize response by max response to classical stimuli
        max_response = np.max(classical_responses, axis=1, keepdims=True)
        classical_responses = classical_responses / max_response
        inverse_responses = inverse_responses / max_response
    else:
        max_response = np.max(np.mean(classical_responses, axis=1), axis=0)
        classical_responses = classical_responses / max_response
        inverse_responses = inverse_responses / max_response
        # change response format to (repeat, frame)
        classical_responses = rearrange(classical_responses, "T R -> R T")
        inverse_responses = rearrange(inverse_responses, "T R -> R T")

    blank_size = 3  # number of blank frames to show
    start = BLANK_SIZE - blank_size  # frame to start
    end_time = 0.3  # last frame (in seconds) to plot
    end = BLANK_SIZE + int(end_time * FPS)  # frame to end

    x = np.arange(classical_responses.shape[1])
    x = x[start : end + 1]

    min_value, max_value = 0, 1
    for i, responses in [(0, classical_responses), (1, inverse_responses)]:
        color = "black" if i == 0 else "red"
        response = np.mean(responses, axis=0)[x]
        se = sem(responses, axis=0)[x]
        ax.plot(
            x,
            response,
            color=color,
            linewidth=1.8,
            alpha=0.9,
            zorder=i + 2,
            clip_on=False,
        )
        ax.fill_between(
            x,
            y1=response - se,
            y2=response + se,
            facecolor=color,
            edgecolor="none",
            alpha=0.3,
            zorder=i + 1,
            clip_on=False,
        )
        max_value = max(max_value, np.max(response + se))

    if show_median:
        for i, onsets in enumerate((classical_onsets, inverse_onsets)):
            ax.scatter(
                x=np.median(onsets) * FPS + BLANK_SIZE,
                y=1.065 * max_value,
                s=23,
                marker="v",
                facecolors="black" if i == 0 else "red",
                edgecolors="none",
                alpha=0.9,
                clip_on=False,
                zorder=2 + i,
            )
    else:
        # show z-score threshold
        response = np.concat(
            [np.mean(classical_responses, axis=0), np.mean(inverse_responses, axis=0)]
        )
        threshold = (THRESHOLD * np.std(response)) + np.mean(response)
        ax.axhline(
            y=threshold,
            color="black",
            linestyle="dotted",
            dashes=(1, 1),
            alpha=0.8,
            linewidth=1,
            zorder=1,
        )

    # set gray background to stimulus presentation window
    ax.axvspan(
        xmin=start + blank_size,
        xmax=end,
        facecolor="#e0e0e0",
        edgecolor="none",
        zorder=-1,
        clip_on=False,
    )
    # set white background to the entire inset plot
    ax.axvspan(
        xmin=start,
        xmax=end,
        ymax=max_value,
        facecolor="white",
        edgecolor="none",
        zorder=-10,
        clip_on=False,
    )

    x_ticks = np.array([start + blank_size, end], dtype=int)
    x_tick_labels = ["0", end_time]
    ax.set_xlim(start, end)

    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        label="Time (s)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.linspace(0, 1, 2)
    ax.set_ylim(min_value, max_value)
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks.astype(int),
        label="Norm.\nΔF/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-7,
        linespacing=0.95,
    )
    # ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    plot.set_ticks_params(ax, pad=1)
    sns.despine(ax=ax)


def plot_response_onset(
    classical_responses: np.ndarray,
    classical_onsets: np.ndarray,
    inverse_responses: np.ndarray,
    inverse_onsets: np.ndarray,
    filename: Path,
    mouse_id: str = None,
    neuron: int | None = None,
):
    # find neurons that are both classical and inverse neurons
    classical_neurons = np.where(~np.isnan(classical_onsets))[0]
    inverse_neurons = np.where(~np.isnan(inverse_onsets))[0]
    joint_neurons = np.intersect1d(classical_neurons, inverse_neurons)

    num_neurons = len(joint_neurons)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=((1 / 2) * PAPER_WIDTH, 1.7),
        gridspec_kw={"width_ratios": [0.83, 0.22], "wspace": 0.05, "hspace": 0},
        dpi=DPI,
    )

    # make scatter plot of inverse response delay vs classical response delay
    min_value, max_value = 0, 0.5  # in seconds

    cmap = "binary"
    hb = axes[0].hexbin(
        x=np.clip(classical_onsets[joint_neurons], min_value, max_value),
        y=np.clip(inverse_onsets[joint_neurons], min_value, max_value),
        gridsize=13,
        extent=(min_value, max_value, min_value, max_value),
        cmap=cmap,
        linewidths=0.2,
        # edgecolors="grey",
        mincnt=1,
        clip_on=True,
        zorder=1,
    )
    vmax = hb.get_array().max()
    if vmax % 2 != 0:
        vmax += 1

    if neuron is not None:
        axes[0].scatter(
            x=classical_onsets[neuron],
            y=inverse_onsets[neuron],
            s=80,
            marker=".",
            alpha=0.9,
            zorder=100,
            color="limegreen",
            edgecolor="none",
            clip_on=False,
        )
    else:
        onset_difference = 1000 * (
            inverse_onsets[joint_neurons] - classical_onsets[joint_neurons]
        )
        print(
            f"\t\tOnset difference: "
            f"mean: {np.mean(onset_difference):.02f} +/- {sem(onset_difference):.02f}ms, "
            f"median: {np.median(onset_difference):.02f}ms\n"
            f"\t\tnum. neurons: {len(joint_neurons)}"
        )

    # Plot identity line
    axes[0].plot(
        [min_value, max_value],
        [min_value, max_value],
        ls="--",
        linewidth=1.0,
        color="black",
        alpha=0.5,
        zorder=2,
    )

    ticks = np.array([min_value, max_value], dtype=np.float32)
    tick_labels = [f"{min_value:.01f}", rf"$\geq${max_value:.01f}"]
    # tick_labels = [min_value, max_value]
    plot.set_xticks(
        axes[0],
        ticks=ticks,
        tick_labels=tick_labels,
        label="Classical\nresponse delay (s)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        linespacing=0.9,
    )
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axes[0].set_xlim(ticks[0], ticks[-1])
    plot.set_yticks(
        axes[0],
        ticks=ticks,
        tick_labels=tick_labels,
        label="Inverse\nresponse delay (s)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-14,
        linespacing=0.9,
    )
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axes[0].set_ylim(ticks[0], ticks[-1])

    # if mouse_id is not None:
    #     text = f"Mouse {mouse_id}"
    # else:
    #     text = r"$N_{mice}$=5"
    text = "L2/3"
    axes[0].text(
        x=0.04 * max_value,
        y=1.02 * max_value,
        s=text,
        fontsize=TICK_FONTSIZE,
        va="top",
        ha="left",
        linespacing=1,
        transform=axes[0].transData,
    )

    axes[0].text(
        x=max_value,
        y=0.04 * max_value,
        s=r"$N_{neurons}$=" + f"{num_neurons}",
        fontsize=TICK_FONTSIZE,
        va="bottom",
        ha="right",
        linespacing=1,
        transform=axes[0].transData,
    )

    plot.set_ticks_params(axes[0], minor_length=None)
    # adjust y-axis tick label pad for >= (\geq)
    for tick in axes[0].get_yaxis().get_major_ticks():
        if "geq" in tick.label1._text:
            tick.set_pad(1.5)
    sns.despine(ax=axes[0])

    cbar_ax = inset_axes(
        parent_axes=axes[0],
        width="100%",
        height="100%",
        bbox_to_anchor=(
            max_value - (0.32 * max_value),
            0.23 * max_value,
            0.2 * max_value,
            0.04 * max_value,
        ),
        bbox_transform=axes[0].transData,
        borderpad=0,
    )
    cbar = figure.colorbar(hb, cax=cbar_ax, orientation="horizontal")
    vmin = 0
    cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
    ticks = np.linspace(vmin, vmax, 2)
    cbar_linewidth = 1.0
    plot.set_xticks(
        axis=cbar_ax,
        ticks=ticks,
        tick_labels=ticks.astype(int),
        tick_fontsize=TICK_FONTSIZE,
        # label="num.\nneurons",
        label_fontsize=TICK_FONTSIZE,
        linespacing=0.8,
    )
    # cbar_ax.set_title(f"num.\nneurons", fontsize=TICK_FONTSIZE, pad=0, linespacing=0.8)
    # cbar_ax.yaxis.set_label_coords(x=2.8, y=0.5 * vmax, transform=cbar_ax.transData)
    plot.set_ticks_params(axis=cbar_ax, length=2, pad=1, linewidth=cbar_linewidth)
    cbar.outline.set_linewidth(cbar_linewidth)
    cbar.outline.set_color("black")

    # Plot responses in inset plot
    axin = inset_axes(
        parent_axes=axes[0],
        width="40%",
        height="40%",
        loc="upper right",
        borderpad=0,
    )
    if neuron is None:
        plot_response_inset(
            ax=axin,
            classical_responses=classical_responses[joint_neurons],
            inverse_responses=inverse_responses[joint_neurons],
            classical_onsets=classical_onsets[joint_neurons],
            inverse_onsets=inverse_onsets[joint_neurons],
        )
    else:
        plot_response_inset(
            ax=axin,
            classical_responses=classical_responses[neuron],
            inverse_responses=inverse_responses[neuron],
            classical_onsets=classical_onsets[neuron],
            inverse_onsets=inverse_onsets[neuron],
            show_median=False,
        )

    # Plot response onsets boxplot
    box_x_ticks = np.array([0, 0.35])
    box_linewidth = 1.2
    # separately select classical and inverse neurons instead of selecting
    # neurons that are both classically and inversely tuned to match
    # Figure 3g in Keller et al.
    classical_onsets = classical_onsets[classical_neurons]
    inverse_onsets = inverse_onsets[inverse_neurons]
    if neuron is None:
        print(
            f"\t\tNeurons considered for statistical test: classical "
            f"neurons {len(classical_neurons)}, inverse neurons {len(inverse_neurons)}"
        )
    bp = axes[1].boxplot(
        [classical_onsets, inverse_onsets],
        notch=False,
        vert=True,
        positions=box_x_ticks,
        widths=0.23,
        showfliers=False,
        showmeans=False,
        capwidths=[0.15, 0.15],
        boxprops={"linewidth": box_linewidth, "clip_on": False, "zorder": 10},
        flierprops={
            "marker": "o",
            "markersize": 2,
            "alpha": 0.5,
            "clip_on": False,
            "zorder": 10,
        },
        capprops={"linewidth": box_linewidth, "clip_on": False, "zorder": 10},
        whiskerprops={"linewidth": box_linewidth, "clip_on": False, "zorder": 10},
        # meanprops={
        #     "markersize": 5,
        #     "markerfacecolor": "gold",
        #     "markeredgecolor": "black",
        #     "markeredgewidth": box_linewidth,
        #     "clip_on": False,
        #     "zorder": 20,
        # },
        medianprops={
            "color": "royalblue",
            "solid_capstyle": "butt",
            "linewidth": 1.5 * box_linewidth,
            "clip_on": False,
            "zorder": 20,
        },
    )

    colors = ["black", "red"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_color(color)
    for whisker, color in zip(bp["whiskers"], ["black", "black", "red", "red"]):
        whisker.set_color(color)
    for cap, color in zip(bp["caps"], ["black", "black", "red", "red"]):
        cap.set_color(color)
    for median, color in zip(bp["medians"], ["black", "red"]):
        median.set_color(color)

    box_min_value = 0.0
    # box_max_value = 0.3
    box_max_value = max([whi.get_ydata()[1] for whi in bp["whiskers"]])
    box_max_value = np.ceil(box_max_value * 10) / 10
    box_y_ticks = np.array([box_min_value, box_max_value], dtype=np.float32)

    scatter_kw = {
        "s": 40,
        "marker": ".",
        "alpha": 0.4,
        "zorder": 0,
        # "facecolors": "none",
        "clip_on": False,
    }
    for i, (position, onsets) in enumerate(
        zip(box_x_ticks, [classical_onsets, inverse_onsets])
    ):
        color = "black" if i == 0 else "red"
        inliers = np.where(onsets < box_max_value)[0]
        outliers = np.where(onsets >= box_max_value)[0]
        # plot responses that are within max_value
        axes[1].scatter(
            np.random.normal(position, 0.0, size=len(inliers)),
            onsets[inliers],
            facecolors=color,
            edgecolors="none",
            **scatter_kw,
        )
        if outliers.size:
            # plot responses that exceed max_value
            axes[1].scatter(
                np.random.normal(position, 0.0, size=len(outliers)),
                np.full(outliers.shape, fill_value=box_max_value),
                facecolors=color,
                edgecolors="none",
                **scatter_kw,
            )

    height = box_max_value - box_min_value
    p_value = ranksums(classical_onsets, inverse_onsets).pvalue
    if neuron is None:
        print(f"\t\tWilcoxon rank sum test p_value = {p_value:.04e}")
    plot.add_p_value(
        ax=axes[1],
        x0=box_x_ticks[0],
        x1=box_x_ticks[1],
        y=1.05 * height + box_min_value,
        p_value=p_value,
        fontsize=LABEL_FONTSIZE,
        tick_length=0.018 * height,
        tick_linewidth=1,
        text_pad=(0.05 if p_value <= 0.05 else 0.045) * height,
    )
    axes[1].set_xlim(box_x_ticks[0] - 0.25, box_x_ticks[-1] + 0.2)
    axes[1].set_xticks([])
    axes[1].set_ylim(box_y_ticks[0], box_y_ticks[-1])
    plot.set_yticks(
        axis=axes[1],
        ticks=box_y_ticks,
        tick_labels=np.round(box_y_ticks, 1),
        label="Delay (s)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-8,
    )
    axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    sns.despine(ax=axes[1])
    plot.set_ticks_params(axis=axes[1], minor_length=None)

    plot.save_figure(figure, filename=filename, dpi=DPI)


def process_mouse(
    model_name: str, output_dir: Path, plot_per_neuron: bool = False
) -> pd.DataFrame:
    print(f"\nProcessing {model_name}...")
    size_tuning = pd.read_parquet(output_dir / "size_tuning_preference.parquet")
    classical_responses, inverse_responses = [], []
    classical_onsets, inverse_onsets = [], []
    mouse_neurons = []
    df = []
    for mouse_id in size_tuning.mouse.unique():
        print(f"\tProcess mouse {mouse_id}...")
        classical_response, inverse_response = load_data(
            output_dir=output_dir, mouse_id=mouse_id
        )
        classical_onset, inverse_onset = estimate_response_onsets(
            classical_responses=classical_response.copy(),
            inverse_responses=inverse_response.copy(),
        )
        # select classic-tuned and inverse-tuned L2/3 neurons and neurons that
        # have preferred stimulus size within +/- 10° of the presented stimulus
        # size (i.e. 20°) as per Keller et al.:
        #   We excluded units for which the responses did not exceed the
        #   response threshold defined above. Furthermore, for the population
        #   responding to the classical stimulus (Fig. 3g), units were excluded
        #   if their preferred classical size was larger than the presented
        #   stimulus size (±10°). For the inverse-tuned population (Fig. 3g),
        #   units were excluded if their preferred inverse size was smaller
        #   than the presented size (±10°). For the inverse-tuned subpopulation
        #   of units responding to both (Fig. 3c–f), both classical and
        #   inverse sizes were required to be within 10° of the presented
        #   stimulus size.
        stimulus_sizes = [STIMULUS_SIZE - 10, STIMULUS_SIZE, STIMULUS_SIZE + 10]
        neurons = size_tuning[
            (size_tuning.mouse == mouse_id)
            & (size_tuning.classic_tuned == True)
            & (size_tuning.inverse_tuned == True)
            & (size_tuning.classic_preference.isin(stimulus_sizes))
            & (size_tuning.inverse_preference.isin(stimulus_sizes))
            & (size_tuning.depth >= 200)
            & (size_tuning.depth <= 300)
        ].neuron.values
        classical_response = classical_response[neurons]
        classical_onset = classical_onset[neurons]
        inverse_response = inverse_response[neurons]
        inverse_onset = inverse_onset[neurons]
        if np.all(np.isnan(classical_onset)) or np.all(np.isnan(inverse_onset)):
            print(f"All NaN onsets for {model_name} {mouse_id}.")
            continue
        plot_response_onset(
            classical_responses=classical_response,
            classical_onsets=classical_onset,
            inverse_responses=inverse_response,
            inverse_onsets=inverse_onset,
            mouse_id=mouse_id,
            filename=PLOT_DIR / model_name / f"response_onset_mouse{mouse_id}.{FORMAT}",
        )
        classical_responses.append(classical_response)
        inverse_responses.append(inverse_response)
        classical_onsets.append(classical_onset)
        inverse_onsets.append(inverse_onset)
        mouse_neurons.extend([(mouse_id, neuron) for neuron in neurons])
        df.append(
            pd.DataFrame(
                {
                    "mouse": [mouse_id] * len(neurons),
                    "neuron": neurons.tolist(),
                    "classical_onset": classical_onset.tolist(),
                    "inverse_onset": inverse_onset.tolist(),
                }
            )
        )
        del classical_response, classical_onset
        del inverse_response, inverse_onset
    if not classical_responses or not inverse_responses:
        return
    print("\tCombine neurons from all mice")
    classical_responses = np.concat(classical_responses, axis=0)
    inverse_responses = np.concat(inverse_responses, axis=0)
    classical_onsets = np.concat(classical_onsets, axis=0)
    inverse_onsets = np.concat(inverse_onsets, axis=0)
    df = pd.concat(df, ignore_index=True)
    df["model_name"] = model_name
    plot_response_onset(
        classical_responses=classical_responses,
        classical_onsets=classical_onsets,
        inverse_responses=inverse_responses,
        inverse_onsets=inverse_onsets,
        filename=PLOT_DIR / model_name / f"response_onset.{FORMAT}",
    )
    with open(PLOT_DIR / model_name / "responses.pkl", "wb") as file:
        pickle.dump(
            {
                "mouse_neurons": mouse_neurons,
                "classical_responses": classical_responses,
                "inverse_responses": inverse_responses,
                "classical_onsets": classical_onsets,
                "inverse_onsets": inverse_onsets,
            },
            file,
        )
    if plot_per_neuron:
        print("Plot traces of individual joint neurons")
        classical_indexes = np.where(~np.isnan(classical_onsets))[0]
        inverse_indexes = np.where(~np.isnan(inverse_onsets))[0]
        joint_indexes = np.intersect1d(classical_indexes, inverse_indexes)
        for n in tqdm(joint_indexes):
            classical_onset = classical_onsets[n]
            inverse_onset = inverse_onsets[n]
            classical_peak = np.argmax(np.mean(classical_responses[n], axis=1))
            classical_peak = classical_peak - BLANK_SIZE
            if (
                (2 <= classical_peak <= 8)
                and (classical_onset < inverse_onset)
                and (inverse_onset <= 0.3)
            ):
                mouse_id, neuron = mouse_neurons[n]
                plot_response_onset(
                    classical_responses=classical_responses,
                    classical_onsets=classical_onsets,
                    inverse_responses=inverse_responses,
                    inverse_onsets=inverse_onsets,
                    filename=PLOT_DIR
                    / model_name
                    / f"per_neuron"
                    / f"response_onset_mouse{mouse_id}_neuron{neuron:04d}.{FORMAT}",
                    neuron=n,
                )
    return df


def plot_onset_delay_inset(
    ax: axes.Axes, classical_response: np.ndarray, inverse_response: np.ndarray
):
    assert classical_response.shape == classical_response.shape
    max_response = np.max(np.mean(classical_response, axis=1), axis=0)
    classical_response = classical_response / max_response
    inverse_response = inverse_response / max_response
    # change response format to (repeat, frame)
    classical_response = rearrange(classical_response, "T R -> R T")
    inverse_response = rearrange(inverse_response, "T R -> R T")

    blank_size = 3  # number of blank frames to show
    start = BLANK_SIZE - blank_size  # frame to start
    end_time = 0.3  # last frame (in seconds) to plot
    end = BLANK_SIZE + int(end_time * FPS)  # frame to end

    x = np.arange(classical_response.shape[1])
    x = x[start : end + 1]

    min_value, max_value = 0, 1
    for i, response in [(0, classical_response), (1, inverse_response)]:
        color = "black" if i == 0 else "red"
        value = np.mean(response, axis=0)[x]
        se = sem(response, axis=0)[x]
        ax.plot(
            x,
            value,
            color=color,
            linewidth=1.8,
            alpha=0.9,
            zorder=i + 2,
            clip_on=False,
        )
        ax.fill_between(
            x,
            y1=value - se,
            y2=value + se,
            facecolor=color,
            edgecolor="none",
            alpha=0.3,
            zorder=i + 1,
            clip_on=False,
        )
        max_value = max(max_value, np.max(value + se))

    # show z-score threshold
    classical_response = np.mean(classical_response, axis=0)
    inverse_response = np.mean(inverse_response, axis=0)
    response = np.concat([classical_response, inverse_response])
    threshold = (THRESHOLD * np.std(response)) + np.mean(response)
    ax.axhline(
        y=threshold,
        color="black",
        linestyle="dotted",
        dashes=(1, 1),
        alpha=0.8,
        linewidth=1,
        zorder=1,
    )

    # set gray background to stimulus presentation window
    ax.axvspan(
        xmin=start + blank_size,
        xmax=end,
        facecolor="#e0e0e0",
        edgecolor="none",
        zorder=-1,
        clip_on=False,
    )
    # set white background to the entire inset plot
    ax.axvspan(
        xmin=start,
        xmax=end,
        ymax=max_value,
        facecolor="white",
        edgecolor="none",
        zorder=-10,
        clip_on=False,
    )

    # find the points where the responses intersect the threshold
    classical_onset = np.where(classical_response >= threshold)[0][0]
    inverse_onset = np.where(inverse_response >= threshold)[0][0]

    line_kw = {
        "linestyle": "--",
        "linewidth": 1,
        "color": "black",
        "zorder": 10,
        "solid_capstyle": "round",
        "solid_joinstyle": "miter",
    }
    ax.plot(
        [classical_onset, classical_onset],
        [0, classical_response[classical_onset]],
        **line_kw,
    )
    ax.plot(
        [inverse_onset, inverse_onset],
        [0, inverse_response[inverse_onset]],
        **line_kw,
    )
    ax.text(
        x=(classical_onset + inverse_onset) / 2,
        y=-0.05,
        s="delay",
        fontsize=TICK_FONTSIZE,
        va="top",
        ha="center",
        clip_on=False,
        transform=ax.transData,
    )

    ax.set_xlim(start, end)
    ax.set_xticks([])

    ax.set_ylim(min_value, max_value)
    plot.set_yticks(
        ax,
        ticks=[threshold],
        tick_labels=["onset"],
        label="",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-6,
        linespacing=1.0,
        rotation=90,
    )
    ax.text(
        x=x[0] - 0.1 * np.max(x),
        y=-0.24 * max_value,
        s="ΔF\n(a.u.)",
        fontsize=TICK_FONTSIZE,
        va="bottom",
        ha="center",
        linespacing=0.7,
    )

    plot.set_ticks_params(ax, length=2.5, pad=0)
    sns.despine(ax=ax)


def plot_onset_difference_comparison(
    df: pd.DataFrame,
    filename: Path,
    classical_response: np.ndarray | None = None,
    inverse_response: np.ndarray | None = None,
):
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    # only keep neurons that have classical and inverse onsets
    df = df[(df.classical_onset.notna()) & (df.inverse_onset.notna())]
    # average classical and inverse onsets by model and animal
    df = df.groupby(by=["model_name", "mouse"], as_index=False).median()
    # remove neuron column
    df.drop(columns=["neuron"], inplace=True)

    model_names = ["LN", "fCNN", "DwiseNeuro", "ViV1T"]
    x = np.arange(len(model_names)) + 1
    positions = x.copy()

    delays, colors = [], []
    for model_name in model_names:
        classical_onsets = df[df.model_name == model_name].classical_onset.values
        inverse_onsets = df[df.model_name == model_name].inverse_onset.values
        delays.append(inverse_onsets - classical_onsets)
        colors.append(plot.get_color(model_name))

    height = [np.mean(delay) for delay in delays]
    error = [sem(delay) for delay in delays]
    linestyle, linewidth = "-", 1.2
    bar_width = 0.35
    ax.bar(
        x=positions,
        height=height,
        yerr=error,
        width=bar_width,
        facecolor="none",
        edgecolor=colors,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=1,
        zorder=1,
        error_kw={"linewidth": 1.0, "zorder": 1},
        clip_on=False,
    )

    min_value = -0.15
    max_value = 0.15
    y_ticks = np.array([min_value, 0, max_value], dtype=np.float32)

    scatter_kw = {
        "s": 30,
        "marker": ".",
        "alpha": 0.8,
        "zorder": 1,
        "facecolors": "none",
        "clip_on": False,
    }
    for i in range(len(positions)):
        ax.scatter(
            [positions[i]] * len(delays[i]),
            delays[i],
            edgecolors=colors[i],
            **scatter_kw,
        )

    # plot Keller et al. response onset delay average
    #   Whereas the response to classical stimuli showed a fast initial
    #   transient followed by a plateau, the response to inverse stimuli
    #   slowly progressed towards steady state (Fig. 3c–e) and was delayed
    #   relative to the classical response (50 ± 20 ms; mean ± s.e.m.; 15 units)
    recorded_average_onset_delay = 0.05
    ax.axhline(
        y=recorded_average_onset_delay,
        linewidth=linewidth,
        linestyle="dotted",
        color="black",
        zorder=0,
    )
    ax.fill_between(
        x=np.arange(x[0] - 0.5, x[-1] + 1.5),
        y1=0.05 - 0.02,
        y2=0.05 + 0.02,
        alpha=0.15,
        zorder=0,
        facecolor="black",
        edgecolor="none",
    )

    for i, model_name in enumerate(model_names):
        print(
            f"model {model_name} onset delay difference:"
            f"{recorded_average_onset_delay - np.mean(delays[i]):.04f}s"
        )
    # ax.text(
    #     x=0.6,
    #     y=-0.094,
    #     s="Keller et al.",
    #     fontsize=TICK_FONTSIZE,
    #     va="bottom",
    #     ha="left",
    # )

    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xticks([])

    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=(y_ticks * 1000).astype(int),
        label="Onset delay (ms)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-9,
        linespacing=1.2,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.spines[["bottom"]].set_position("center")

    sns.despine(ax=ax)
    plot.set_ticks_params(axis=ax, minor_length=3)

    if classical_response is not None and inverse_response is not None:
        # Plot responses in inset plot
        axin = inset_axes(
            parent_axes=ax,
            width=0.5,
            height=0.45,
            loc="lower right",
            bbox_to_anchor=(positions[-1] + 0.45, -0.9 * max_value),
            bbox_transform=ax.transData,
            borderpad=0,
        )
        plot_onset_delay_inset(
            ax=axin,
            classical_response=classical_response,
            inverse_response=inverse_response,
        )

    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    # models = {
    #     # "LN": Path("../runs/fCNN/036_linear_fCNN"),
    #     # "fCNN": Path("../runs/fCNN/038_fCNN"),
    #     # "DwiseNeuro": Path("../runs/lRomul"),
    #     "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    # }
    # df = []
    # for model_name, output_dir in models.items():
    #     df.append(
    #         process_mouse(
    #             model_name=model_name,
    #             output_dir=output_dir,
    #             plot_per_neuron=model_name == "ViV1T",
    #         )
    #     )
    # df = pd.concat(df, ignore_index=True)

    filename = PLOT_DIR / "response_onsets.parquet"
    # df.to_parquet(filename)
    df = pd.read_parquet(filename)

    with open(PLOT_DIR / "ViV1T" / "responses.pkl", "rb") as file:
        data = pickle.load(file)
    index = None
    for i, mouse_neuron in enumerate(data["mouse_neurons"]):
        if mouse_neuron == ("E", 1587):
            index = i
            break
    classical_response = data["classical_responses"][index]
    inverse_response = data["inverse_responses"][index]
    plot_onset_difference_comparison(
        df=df,
        filename=PLOT_DIR / f"response_onset_difference_comparison.{FORMAT}",
        classical_response=classical_response,
        inverse_response=inverse_response,
    )

    print(f"Saved plots to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
