from pathlib import Path
from typing import Dict
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils
from viv1t.utils.utils import load_tuning

plot.set_font()


TICK_FONTSIZE = 6
LABEL_FONTSIZE = 6
TITLE_FONTSIZE = 7
figure_width_matplotlib_inches = 0.32 * 5.1666
DPI = 500

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures/selectivity_distribution")

SI_THRESHOLD = 0.0
MOUSE_IDS = ["B", "C", "E"]


def get_percentage(
    output_dir: Path,
    tuning_type: str,
):
    result = {}
    for mouse_id in MOUSE_IDS:
        tuning = load_tuning(save_dir=output_dir, mouse_id=mouse_id)
        SIs = tuning["OSI" if tuning_type == "orientation" else "DSI"]
        num_neurons = len(SIs)
        neurons = np.where(SIs >= SI_THRESHOLD)[0]
        tuning_curves = tuning["tuning_curve"][neurons]
        if tuning_type == "orientation":
            # convert direction tuning curves to orientation tuning curves
            tuning_curves = (tuning_curves[:, :4] + tuning_curves[:, 4:]) / 2
        preference = np.argmax(tuning_curves, axis=1)
        angles = tuning_curves.shape[1]
        percentage = np.bincount(preference, minlength=angles) / num_neurons
        result[mouse_id] = 100 * percentage
    return result


def plot_percentage(
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    tuning_type: str,
    mouse_id: str,
    title: str = None,
    filename: Path = None,
):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=DPI)

    y_max, bar_pad, edgewidth = 0.5, 0.005, 0.5
    alpha = 1.0
    bar_width = 0.5
    cmap = matplotlib.colormaps["Reds"]

    x_ticks = np.arange(len(results))
    bottom = np.zeros(len(results))
    directions = np.array(list(data.DIRECTIONS.keys()))
    if tuning_type == "orientation":
        directions = directions[:4]

    colors = cmap(np.linspace(0, 1, len(directions)))

    zorder = len(directions)
    for i, direction in enumerate(directions):
        for j, model in enumerate(results.keys()):
            bottom[j] += results[model][tuning_type][mouse_id][i]
        ax.bar(
            x_ticks,
            bottom,
            label=f"{direction}°",
            color=colors[i],
            width=bar_width,
            alpha=alpha,
            linewidth=edgewidth,
            edgecolor="black",
            zorder=zorder,
        )
        zorder -= 1

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=4,
        frameon=False,
        handletextpad=0.3,
        handlelength=0.6,
        markerscale=0.6,
        columnspacing=1,
        fontsize=TICK_FONTSIZE,
    )
    y_max = np.ceil(np.max(bottom) * 0.1) * 10
    y_ticks = np.linspace(0, y_max, 3)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=y_ticks.astype(int),
        label="% of selective neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    for y in y_ticks[1:-1]:
        ax.axhline(y=y, color="gray", linewidth=1, alpha=0.3, zorder=-1, clip_on=False)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=list(results.keys()),
        label=tuning_type.capitalize(),
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    if title is not None:
        ax.set_title(title, fontsize=LABEL_FONTSIZE)
    sns.despine(ax=ax, top=True, right=True, trim=False)
    plot.set_ticks_params(axis=ax, length=2)
    if filename is not None:
        plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_selective_neuron_percentage(
    models: Dict[str, Path],
    tuning_types: List[str],
    plot_dir: Path,
):
    results = {}
    for model, output_dir in models.items():
        results[model] = {}
        for tuning_type in tuning_types:
            results[model][tuning_type] = get_percentage(
                output_dir=output_dir, tuning_type=tuning_type
            )

    for tuning_type in tuning_types:
        for mouse_id in MOUSE_IDS:
            plot_percentage(
                results,
                tuning_type=tuning_type,
                mouse_id=mouse_id,
                title=f"Mouse {mouse_id} (SI threshold {SI_THRESHOLD:.1f})",
                filename=plot_dir
                / f"mouse{mouse_id}"
                / f"{tuning_type}_percentage.jpg",
            )


def bin_si(si: np.ndarray) -> np.ndarray:
    """Group selectivity indexes in bins of 0 to < 0.3, >=0.3 to < 0.6 and >= 0.6"""
    counts = np.zeros((3,), dtype=int)
    # count SI values between >=0 and < 0.3
    counts[0] = len(np.where((si >= 0) & (si < 0.3))[0])
    # count SI values between >= 0.3 and < 0.6
    counts[1] = len(np.where((si >= 0.3) & (si < 0.6))[0])
    # count SI values >= 0.6
    counts[2] = len(np.where(si >= 0.6)[0])
    assert len(si) == np.sum(counts)
    return counts


def plot_si_distribution(models: Dict[str, Path], tuning_type: str, filename: Path):
    results = {model: [] for model in models.keys()}
    for mouse_id in MOUSE_IDS:
        # selective neurons based on recorded data
        neurons = utils.get_selective_neurons(
            # save_dir=Path("../runs/vivit/172_viv1t_causal"),
            save_dir=data.METADATA_DIR,
            mouse_id=mouse_id,
            threshold=SI_THRESHOLD,
            tuning_type=tuning_type,
        )
        for model, output_dir in models.items():
            tuning = load_tuning(save_dir=output_dir, mouse_id=mouse_id)
            indexes = tuning["OSI" if tuning_type == "orientation" else "DSI"]
            indexes = indexes[neurons]
            results[model].append(bin_si(indexes) / len(neurons))
    results = {k: np.vstack(v) for k, v in results.items()}

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width_matplotlib_inches, figure_width_matplotlib_inches * 0.8),
        dpi=DPI,
    )

    bar_width = 1
    bar_pad, edgewidth = 0.005, 0.4
    error_kw = {"elinewidth": 0.8, "capsize": 1.5, "markeredgewidth": 0.6}

    x_ticks = np.arange(3) * 6
    x_labels = ["[0, 3)", "[0.3, 0.6)", r"$\geq0.6$"]

    for i, model in enumerate(models.keys()):
        match model:
            case "recorded":
                x = x_ticks - 2 * bar_width
            case "LN":
                x = x_ticks - bar_width
            case "fCNN":
                x = x_ticks
            case "DwiseNeuro":
                x = x_ticks + bar_width
            case "ViV1T":
                x = x_ticks + 2 * bar_width
        ax.bar(
            x,
            height=np.mean(results[model], axis=0),
            width=bar_width,
            yerr=sem(results[model], axis=0),
            align="center",
            alpha=0.7,
            color=plot.get_color(model),
            edgecolor="black",
            linewidth=edgewidth,
            error_kw=error_kw,
            label=model,
        )

    ax.set_xlim(-3.5 * bar_width, x_ticks[-1] + 3.5 * bar_width)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_labels,
        label=f"{tuning_type.capitalize()} selectivity index",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=1,
    )

    y_ticks = np.linspace(0, 1, 6)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(x_ticks[-1] + 3.5 * bar_width, 1.0),
        bbox_transform=ax.transData,
        ncols=1,
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        title="",
        handletextpad=0.5,
        handlelength=0.6,
        labelspacing=0.3,
        borderpad=0,
        borderaxespad=0,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    for text in legend.texts:
        text.set_y(0)

    plot.set_ticks_params(ax, length=4, pad=2)
    sns.despine(ax=ax)

    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    models = {
        "recorded": data.METADATA_DIR,
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t_kj8kztj6"),
    }
    tuning_types = ["orientation", "direction"]

    # plot_selective_neuron_percentage(
    #     models=models,
    #     tuning_types=tuning_types,
    #     plot_dir=PLOT_DIR / "selective_percentage",
    # )
    for tuning_type in tuning_types:
        plot_si_distribution(
            models=models,
            tuning_type=tuning_type,
            filename=PLOT_DIR / f"{tuning_type}_si_distribution.jpg",
        )

    print(f"plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
