from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import sem
from scipy.stats import wilcoxon

from viv1t.utils import plot
from viv1t.utils import stimulus

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures") / "size_tuning_curve"

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

# Extracted from Extended Figure 1B (bottom right) of https://www.nature.com/articles/s41586-020-2319-4
# using https://plotdigitizer.com/app
KELLER_CLASSIC = np.array(
    [
        [0, 0.000898646747888011],
        [5, 0.6614583126658927],
        [15, 0.765786430677428],
        [25, 0.7754588970533302],
        [35, 0.6036368776393599],
        [45, 0.47950476255388297],
        [55, 0.32127331405949083],
        [65, 0.25367595371638524],
        [75, 0.2686694989925324],
        [85, 0.21515422288220162],
    ],
    dtype=np.float32,
)
KELLER_CLASSIC_MEDIAN = 11.709846680301045
KELLER_INVERSE = np.array(
    [
        [0, 0.000898646747888011],
        [5, 0.3022645186836017],
        [15, 0.7335104430257322],
        [25, 0.7709265745035988],
        [35, 0.7736097837967898],
        [45, 0.608401968914765],
        [55, 0.42692687473085067],
        [65, 0.3179758311969037],
        [75, 0.2584669375086282],
        [85, 0.22379219014182697],
    ],
    dtype=np.float32,
)
KELLER_INVERSE_MEDIAN = 24.182512692980023


def get_stimulus_sizes(output_dir: Path) -> np.ndarray:
    parameters = np.load(
        output_dir / "feedbackRF" / "parameters.npy", allow_pickle=False
    )
    return np.unique(parameters[..., 0])


def normalize(responses: np.ndarray) -> np.ndarray:
    """
    Normalize response by the maximum response to classical stimuli per neuron

    Args:
        responses: np.ndarray, response in shape (neurons, stimulus size, classical or inverse)
    """
    assert responses.ndim == 3 and responses.shape[2] == 2
    max_responses = np.max(responses[:, :, 0], axis=1)
    responses = responses / max_responses[:, None, None]
    return responses


def compute_inverse_tuning_index(responses: np.ndarray):
    # response to full-field stimulus
    R_ff = responses[:, 0, 1]
    # maximum response to classical and inverse stimulus
    R_cla = np.max(responses[:, :, 0], axis=1)
    R_inv = np.max(responses[:, :, 1], axis=1)
    iti = ((R_inv - R_cla) / (2 * ((R_inv - R_ff) + (R_cla - R_ff)))) + 0.5
    print(f"\tITI mean: {np.mean(iti):.03f}, median: {np.median(iti):.03f}")
    return iti


def iti_inset(axin: Axes, iti: np.ndarray):
    x_ticks = np.array([0, 1], dtype=np.float32)
    hist_kws = {
        "bins": 15,
        "color": "black",
        "range": (0, 1),
        "histtype": "step",
        "linewidth": 1.2,
        "clip_on": False,
    }
    h_y, h_x, _ = axin.hist(
        iti,
        alpha=0.9,
        # show percentage instead of count
        weights=np.ones(len(iti)) / len(iti),
        **hist_kws,
    )
    max_value = np.max(h_y)
    max_value = np.ceil(max_value * 10) / 10
    axin.scatter(
        np.median(iti),
        max_value,
        s=25,
        marker="v",
        facecolors="black",
        edgecolors="none",
        alpha=1,
        clip_on=False,
    )
    axin.axvline(
        x=0.5,
        color="black",
        alpha=1,
        linestyle="dashed",
        linewidth=0.8,
        zorder=-1,
    )

    axin.set_xlim(x_ticks[0], x_ticks[-1])
    plot.set_xticks(
        axin,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="ITI",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )
    # axin.xaxis.set_minor_locator(MultipleLocator(0.5))

    y_ticks = np.linspace(0, max_value, 2)
    axin.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axin,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="Fraction",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-10,
        linespacing=0.9,
    )
    # axin.yaxis.set_minor_locator(MultipleLocator(0.1))

    sns.despine(ax=axin, trim=True)
    plot.set_ticks_params(axin)


def plot_size_tuning_curve(
    responses: np.ndarray,
    stimulus_sizes: list[int] | np.ndarray,
    filename: Path,
    mouse_id: str | None = None,
    iti: np.ndarray | None = None,
):
    responses = normalize(responses)

    population_response = np.mean(responses, axis=0)
    population_sem = sem(responses, axis=0)

    max_value = max(np.max(population_response + population_sem), 1.0)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.7),
        dpi=DPI,
    )

    linewidth = 1.5

    for stim_type in [0, 1]:
        color = "red" if stim_type else "black"
        ax.errorbar(
            x=stimulus_sizes,
            y=population_response[:, stim_type],
            yerr=population_sem[:, stim_type],
            fmt=".",
            elinewidth=1.25,
            capsize=0.7,
            capthick=1.25,
            alpha=1,
            clip_on=False,
            color=color,
            linestyle="",
            markersize=10,
            zorder=stim_type,
        )
        ax.plot(
            stimulus_sizes,
            population_response[:, stim_type],
            marker="none",
            color=color,
            alpha=1,
            linewidth=linewidth,
            clip_on=False,
            zorder=stim_type,
            label="Inverse" if stim_type else "Classical",
        )
        # plot triangle to indicate preferred size
        ax.scatter(
            x=np.median(stimulus_sizes[np.argmax(responses[:, :, stim_type], axis=1)]),
            y=max_value,
            s=25,
            marker="v",
            facecolors="black" if stim_type == 0 else "red",
            edgecolors="none",
            alpha=1,
            clip_on=False,
        )

    legend = ax.legend(
        loc="lower right",
        bbox_to_anchor=(90, 0.03 * max_value),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.4,
        handlelength=0.7,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        bbox_transform=ax.transData,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    x_ticks = np.linspace(stimulus_sizes[0], stimulus_sizes[-1], 4, dtype=int)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="Stimulus size [°]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim(x_ticks[0], x_ticks[-1])

    y_ticks = np.linspace(0, max_value, 6, dtype=np.float32)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="Response [norm. ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    text = f"Same {len(responses)} neurons"
    # if mouse_id is not None:
    #     text = f"Mouse {mouse_id} " + text
    ax.text(
        x=0.04 * x_ticks[-1],
        y=0.03 * max_value,
        s=text,
        fontsize=TICK_FONTSIZE,
    )

    sns.despine(ax=ax, trim=True)
    plot.set_ticks_params(ax)

    # Add response box plot
    if iti is not None:
        axin = inset_axes(
            parent_axes=ax,
            width="25%",
            height="45%",
            loc="upper right",
            borderpad=0.15,
        )
        iti_inset(axin=axin, iti=iti)

    plot.save_figure(figure, filename=filename, dpi=DPI)


def response_boxplot_inset(
    axin: Axes,
    classical_responses: np.ndarray,
    inverse_responses: np.ndarray,
):
    assert classical_responses.shape == inverse_responses.shape
    x_ticks, linewidth = np.array([0, 0.27]), 1.2
    # take the maximum response over stimulus size
    classical_responses = np.max(classical_responses, axis=1)
    inverse_responses = np.max(inverse_responses, axis=1)
    bp = axin.boxplot(
        [classical_responses, inverse_responses],
        notch=False,
        vert=True,
        positions=x_ticks,
        widths=0.2,
        showfliers=False,
        showmeans=False,
        boxprops={"linewidth": linewidth, "clip_on": False, "zorder": 10},
        flierprops={
            "marker": "o",
            "markersize": 2,
            "alpha": 0.5,
            "clip_on": False,
            "zorder": 10,
        },
        capprops={"linewidth": linewidth, "clip_on": False, "zorder": 10},
        whiskerprops={"linewidth": linewidth, "clip_on": False, "zorder": 10},
        # meanprops={
        #     "markersize": 5,
        #     "markerfacecolor": "gold",
        #     "markeredgecolor": "black",
        #     "markeredgewidth": linewidth,
        #     "clip_on": False,
        #     "zorder": 20,
        # },
        medianprops={
            "color": "black",
            "solid_capstyle": "butt",
            "linewidth": 1.5 * linewidth,
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
    axin.set_xlim(x_ticks[0] - 0.2, x_ticks[-1] + 0.2)
    axin.set_xticks([])
    min_value = 0
    max_value = max([whi.get_ydata()[1] for whi in bp["whiskers"]]) + 1
    # round up to nearest even number
    max_value = int(2 * np.ceil(max_value / 2))
    y_ticks = np.array([min_value, max_value], dtype=int)
    axin.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=axin,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label=r"$\Delta$F/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-8,
    )
    axin.yaxis.set_minor_locator(MultipleLocator(10))

    p_value = wilcoxon(classical_responses, inverse_responses).pvalue
    print(f"\tWilcoxon signed-rank test p-value = {p_value:.04e}")
    height = max_value - min_value
    plot.add_p_value(
        ax=axin,
        x0=x_ticks[0],
        x1=x_ticks[1],
        y=1.06 * max_value,
        p_value=p_value,
        tick_length=0.05 * height,
        tick_linewidth=1,
        fontsize=LABEL_FONTSIZE,
        text_pad=0.11 * height,
    )

    # scatter_kw = {
    #     "s": 10,
    #     "marker": ".",
    #     "alpha": 0.05,
    #     "zorder": 0,
    #     # "facecolors": "none",
    #     "clip_on": False,
    # }
    # for i, (position, onsets) in enumerate(
    #     zip(x_ticks, [classical_responses, inverse_responses])
    # ):
    #     color = "black" if i == 0 else "red"
    #     neurons = np.arange(len(onsets))
    #     outliers = np.where(onsets >= max_value)[0]
    #     # plot responses that are within max_value
    #     y = onsets[np.setdiff1d(neurons, outliers)]
    #     x = np.random.normal(position, 0.03, size=len(y))
    #     axin.scatter(x, y, facecolors=color, edgecolors="none", **scatter_kw)
    #     # plot responses that exceed max_value
    #     y = np.full(outliers.shape, fill_value=max_value)
    #     x = np.random.normal(position, 0.03, size=len(y))
    #     axin.scatter(x, y, facecolors=color, edgecolors="none", **scatter_kw)

    sns.despine(ax=axin)
    plot.set_ticks_params(axin, length=3, pad=1, minor_length=3)


def plot_size_tuning_curve_across_animals(
    mouse_responses: dict[str, np.ndarray],
    stimulus_sizes: np.ndarray,
    filename: Path,
    title: str = None,
):
    responses, num_neurons = [], []
    for mouse_id in mouse_responses.keys():
        num_neurons.append(len(mouse_responses[mouse_id]))
        responses.append(normalize(mouse_responses[mouse_id]))

    responses = np.concatenate(responses, axis=0)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.7),
        dpi=DPI,
    )

    population_response = np.mean(responses, axis=0)
    population_sem = sem(responses, axis=0)

    max_value = 1.0
    linewidth = 1.5

    for stim_type in [0, 1]:
        color = "red" if stim_type else "black"
        ax.errorbar(
            x=stimulus_sizes,
            y=population_response[:, stim_type],
            yerr=population_sem[:, stim_type],
            fmt=".",
            elinewidth=1.25,
            capsize=0.7,
            capthick=1.25,
            alpha=1,
            clip_on=False,
            color=color,
            linestyle="",
            markersize=10,
            zorder=stim_type,
        )
        ax.plot(
            stimulus_sizes,
            population_response[:, stim_type],
            marker="none",
            color=color,
            alpha=1,
            linewidth=linewidth,
            clip_on=False,
            zorder=stim_type,
            label="Inverse" if stim_type else "Classical",
        )
        # plot average response for each mouse
        for mouse_id, mouse_response in mouse_responses.items():
            mouse_response = np.mean(normalize(mouse_response), axis=0)
            ax.plot(
                stimulus_sizes,
                mouse_response[:, stim_type],
                marker="",
                alpha=0.2,
                color=color,
                linestyle="dotted",
                linewidth=linewidth,
                zorder=stim_type,
            )
            max_value = max(max_value, np.max(mouse_response[:, stim_type]))

    # round up to nearest even number
    max_value = 0.2 * np.ceil(10 * max_value / 2)
    max_value = round(max_value.item(), 1)
    for stim_type in [0, 1]:
        # plot triangle to indicate preferred size
        ax.scatter(
            x=np.median(stimulus_sizes[np.argmax(responses[:, :, stim_type], axis=1)]),
            y=max_value,
            s=30,
            marker="v",
            facecolors="black" if stim_type == 0 else "red",
            edgecolors="none",
            alpha=1,
            clip_on=False,
        )

    x_ticks = np.linspace(stimulus_sizes[0], stimulus_sizes[-1], 4, dtype=int)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="Stimulus size [°]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim(x_ticks[0], x_ticks[-1])

    y_ticks = np.arange(0, max_value + 0.2, 0.2, dtype=np.float32)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=[
            y if i % 2 == 0 else "" for i, y in enumerate(np.round(y_ticks, 1))
        ],
        label="Response [norm. ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=1,
    )

    # legend = ax.legend(
    #    loc="lower left",
    #    bbox_to_anchor=(0.03 * x_ticks[-1], 0.08 * max_value),
    #    ncols=2,
    #    fontsize=TICK_FONTSIZE,
    #    frameon=False,
    #    title="",
    #    handletextpad=0.35,
    #    handlelength=0.5,
    #    markerscale=0.25,
    #    labelspacing=0.0,
    #    columnspacing=1,
    #    borderpad=0,
    #    borderaxespad=0,
    #    bbox_transform=ax.transData,
    # )
    # for lh in legend.legend_handles:
    #    lh.set_alpha(1)

    # text = r"$N_{neurons}$=" + f"{sum(num_neurons)}   "
    # text += r"$N_{mice}$=" + f"{len(num_neurons)}"
    text = f"Same {sum(num_neurons)} neurons"
    ax.text(
        x=0.04 * x_ticks[-1],
        y=0.02 * max_value,
        s=text,
        fontsize=TICK_FONTSIZE,
        ha="left",
        va="bottom",
    )

    if title is not None:
        ax.set_title(title, fontsize=LABEL_FONTSIZE, pad=2)
    sns.despine(ax=ax, trim=True)
    plot.set_ticks_params(ax, length=3)

    # Add response box plot
    axin = inset_axes(
        parent_axes=ax,
        width="15%",
        height="40%",
        loc="upper right",
        borderpad=0.35,
    )
    raw_responses = np.concat(list(mouse_responses.values()))
    response_boxplot_inset(
        axin=axin,
        classical_responses=raw_responses[:, :, 0],
        inverse_responses=raw_responses[:, :, 1],
    )

    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_model(
    model_name: str, stimulus_sizes: np.ndarray, output_dir: Path
) -> pd.DataFrame:
    df = pd.read_parquet(output_dir / "size_tuning_preference.parquet")
    df["model_name"] = model_name

    # plot L2/3 neurons feedforward and feedback size-tuning curves
    mouse_responses = {}
    for mouse_id in df.mouse.unique():
        print(f"Plot size-tuning curve for mouse {mouse_id}...")
        indexes = (
            (df.mouse == mouse_id)
            & (df.classic_tuned == True)
            & (df.depth >= 200)
            & (df.depth <= 300)
        )
        classic_tuning_curves = np.array(
            df[indexes].classic_tuning_curve.values.tolist(),
            dtype=np.float32,
        )
        inverse_tuning_curves = np.array(
            df[indexes].inverse_tuning_curve.values.tolist(),
            dtype=np.float32,
        )
        responses = np.stack([classic_tuning_curves, inverse_tuning_curves], axis=-1)
        iti = compute_inverse_tuning_index(responses=responses)
        plot_size_tuning_curve(
            responses=responses,
            stimulus_sizes=stimulus_sizes,
            filename=PLOT_DIR
            / model_name
            / f"size_tuning_curve_mouse{mouse_id}.{FORMAT}",
            mouse_id=mouse_id,
            iti=iti,
        )
        mouse_responses[mouse_id] = responses
    print("Combine neurons from all mice")
    responses = np.concat(list(mouse_responses.values()), axis=0)
    preferred_sizes = np.argmax(responses, axis=1) * 10
    print(
        f"Preferred classical stimulus size:\n"
        f"\tmean: {np.mean(preferred_sizes[:, 0]):.02f}"
        f" +/- {sem(preferred_sizes[:, 0]):.02f}◦\n"
        f"\tmedian: {np.median(preferred_sizes[:, 0]):.02f}◦\n"
        f"Preferred inverse stimulus size:\n"
        f"\tmean: {np.mean(preferred_sizes[:, 1]):.02f}"
        f" +/- {sem(preferred_sizes[:, 1]):.02f}◦\n"
        f"\tmedian: {np.median(preferred_sizes[:, 1]):.02f}◦\n"
        f"num. neurons: {len(responses)}"
    )
    plot_size_tuning_curve(
        responses=responses,
        stimulus_sizes=stimulus_sizes,
        filename=PLOT_DIR / model_name / f"size_tuning_curve_combine.{FORMAT}",
    )
    plot_size_tuning_curve_across_animals(
        mouse_responses=mouse_responses,
        stimulus_sizes=stimulus_sizes,
        filename=PLOT_DIR / model_name / f"size_tuning_curve.{FORMAT}",
    )
    return df


def plot_size_tuning_curve_comparison(
    df: pd.DataFrame,
    stimulus_sizes: np.ndarray,
    stimulus_type: str,
    filename: Path,
):
    model_names = df.model_name.unique()
    assert stimulus_type in ("classical", "inverse")
    figure_width, figure_height = (1 / 3) * PAPER_WIDTH, 1.5
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, figure_height),
        dpi=DPI,
    )

    max_value, max_value = 0, 0
    linewidth = 1.8
    linestyle = "-" if stimulus_type == "classical" else (0, (0.75, 0.75))

    preferred_stimulus_sizes: dict[str, np.ndarray] = {}

    for model_name in model_names:
        preferred_stimulus_sizes[model_name] = np.zeros(2, dtype=np.float32)
        color = plot.get_color(model_name)
        indexes = (
            (df.model_name == model_name)
            & (df.classic_tuned == True)
            & (df.depth >= 200)
            & (df.depth <= 300)
        )
        classic_tuning_curves = np.array(
            df[indexes].classic_tuning_curve.values.tolist(),
            dtype=np.float32,
        )
        inverse_tuning_curves = np.array(
            df[indexes].inverse_tuning_curve.values.tolist(),
            dtype=np.float32,
        )
        responses = np.stack([classic_tuning_curves, inverse_tuning_curves], axis=-1)
        responses = normalize(responses)
        tuning_curves = responses[..., (0 if stimulus_type == "classical" else 1)]
        values = np.mean(tuning_curves, axis=0)
        error_bars = sem(tuning_curves, axis=0)
        ax.errorbar(
            x=stimulus_sizes,
            y=values,
            yerr=error_bars,
            fmt=".",
            elinewidth=1.25,
            capsize=0.7,
            capthick=1.25,
            alpha=0.8,
            clip_on=False,
            color=color,
            linestyle="",
            markersize=5,
            zorder=plot.get_zorder(model_name),
        )
        ax.plot(
            stimulus_sizes,
            values,
            marker="none",
            color=color,
            alpha=0.7,
            linewidth=linewidth,
            linestyle=linestyle,
            clip_on=False,
            zorder=plot.get_zorder(model_name),
        )
        max_value = max(max_value, np.max(values + error_bars))
        preferred_stimulus_sizes[model_name] = np.median(
            stimulus_sizes[np.argmax(tuning_curves, axis=1)]
        )

    # round up to nearest even number
    max_value = 0.2 * np.ceil(10 * max_value / 2)
    max_value = round(max_value.item(), 1)

    # ax.text(
    #     x=stimulus_sizes[-1],
    #     y=max_value,
    #     s=stimulus_type.capitalize(),
    #     fontsize=LABEL_FONTSIZE,
    #     ha="right",
    #     va="top",
    #     transform=ax.transData,
    #     zorder=1,
    # )

    # Plot Keller et al. size tuning curve
    # recorded_x, recorded_y = ax.get_xlim()
    match stimulus_type:
        case "classical":
            recorded = KELLER_CLASSIC
        case "inverse":
            recorded = KELLER_INVERSE
        case _:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")
    ax.plot(
        recorded[:, 0],
        recorded[:, 1],
        marker=".",
        markersize=5,
        color="black",
        alpha=0.7,
        linewidth=linewidth,
        linestyle=linestyle,
        clip_on=False,
        zorder=plot.get_zorder("recorded"),
        label=r"$\it{In}$ $\it{vivo}$ recording",
    )
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(6, 0.01 * max_value),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.3,
        handlelength=1,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        bbox_transform=ax.transData,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    x_ticks = np.linspace(stimulus_sizes[0], stimulus_sizes[-1], 4, dtype=int)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="Stimulus size [°]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    y_ticks = np.array([0, max_value], dtype=np.float32)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="Response\n[norm. ΔF/F]",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-10,
        linespacing=0.95,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    sns.despine(ax=ax, trim=True)
    plot.set_ticks_params(ax, minor_length=3)

    # Plot stimulus illustration
    w = 0.15
    h = w * figure_width / figure_height
    ax = figure.add_axes((0.75, 0.6, w, h))

    fps = 30
    grating = stimulus.create_full_field_grating(
        direction=90,
        cpd=0.04,
        cpf=2 / fps,
        num_frames=1,
        height=36,
        width=64,
        phase=0,
        contrast=1,
    )
    circular_mask = stimulus.create_circular_mask(
        stimulus_size=50, pixel_height=36, pixel_width=64
    )

    if stimulus_type == "classical":
        stim = np.where(circular_mask, grating, 255 / 2)
    else:
        stim = np.where(circular_mask, 255 / 2, grating)
    ax.imshow(stim[0, 0, :, 15:51], cmap="gray", vmin=0, vmax=255, zorder=-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        "Classical" if stimulus_type == "classical" else "Inverse",
        fontsize=TICK_FONTSIZE,
        labelpad=2,
    )
    plot.set_ticks_params(
        axis=ax,
        color="black" if stimulus_type == "classical" else "red",
    )

    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    models = {
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }

    stimulus_sizes = get_stimulus_sizes(output_dir=list(models.values())[0])

    df = []
    for model_name, output_dir in models.items():
        print(f"\nPlot size-tuning curves for {model_name} in {output_dir}...")
        df.append(
            plot_model(
                model_name=model_name,
                stimulus_sizes=stimulus_sizes,
                output_dir=output_dir,
            )
        )
    df = pd.concat(df, ignore_index=True)

    # plot_size_tuning_curve_comparison(
    #     df=df,
    #     stimulus_sizes=stimulus_sizes,
    #     stimulus_type="classical",
    #     filename=PLOT_DIR / f"classical_size_tuning_curve_comparison.{FORMAT}",
    # )
    # plot_size_tuning_curve_comparison(
    #     df=df,
    #     stimulus_sizes=stimulus_sizes,
    #     stimulus_type="inverse",
    #     filename=PLOT_DIR / f"inverse_size_tuning_curve_comparison.{FORMAT}",
    # )


if __name__ == "__main__":
    main()
