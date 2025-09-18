from math import ceil
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import sem
from tqdm import tqdm

from viv1t.utils import plot
from viv1t.utils import utils
from viv1t.utils import yaml

plot.set_font()

TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 11
DPI = 240

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures/limit_data")
OUTPUT_DIR = Path("../runs/limit_data/")


def load_result(filename: Path, key: str, metric: str = "correlation"):
    result = None
    if filename.is_file():
        result = yaml.load(filename)[key][metric]
        result.pop("average", None)
    return result


def load_results(model_dir: Path, metric: str = "correlation"):
    results = {}
    output_dirs = model_dir.glob("*")
    for output_dir in output_dirs:
        if not output_dir.is_dir():
            continue
        dir_name = str(output_dir.name)
        train_size = int(dir_name[0 : dir_name.find("_")])
        correlations = yaml.load(output_dir / "evaluation_type.yaml")
        for stimulus_name in correlations.keys():
            if stimulus_name not in results:
                results[stimulus_name] = {}
            correlation = correlations[stimulus_name][metric]
            correlation.pop("average", None)
            results[stimulus_name][train_size] = correlation
    # sort result by train size
    for stimulus_name in results.keys():
        results[stimulus_name] = dict(sorted(results[stimulus_name].items()))
    return results


def plot_histogram(
    viv1t: Dict[int, Dict[str, float]],
    lRomul: Dict[int, Dict[str, float]],
    title: str,
    metric: str,
    filename: Path,
):
    assert viv1t.keys() == lRomul.keys()

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2), dpi=DPI)

    x_ticks = np.arange(len(viv1t.keys()))
    x_labels = [f"{k}" for k in viv1t.keys()]

    width = 0.4
    for i, result in enumerate([lRomul, viv1t]):
        y_values = np.array([np.mean(list(v.values())) for v in result.values()])
        y_errors = np.array([sem(list(v.values())) for v in result.values()])
        match i:
            case 0:
                model_name = "DwiseNeuro"
                _x_ticks = x_ticks - (width / 2)
            case 1:
                model_name = "ViV1T"
                _x_ticks = x_ticks + (width / 2)
            case _:
                raise NotImplementedError
        ax.bar(
            _x_ticks,
            y_values,
            width=width,
            color=plot.get_color(model_name),
            alpha=0.8,
            edgecolor="black",
            linewidth=0.6,
            yerr=y_errors,
            capsize=2,
            error_kw={"linewidth": 0.8, "capthick": 0.8},
            label=model_name,
            clip_on=False,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncols=2,
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        title="",
        handletextpad=0.35,
        handlelength=0.6,
        markerscale=0.6,
        columnspacing=1,
    )

    ax.set_xlim(x_ticks[0] - 0.6, x_ticks[-1] + 0.6)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_labels,
        label="Training size (% of 350 samples)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    # y_max = 0.1 * ceil(np.max(y_values + y_errors) * 10)
    match metric:
        case "correlation":
            y_max = 0.3
            y_ticks = np.linspace(0, y_max, 4)
            y_label = "Single trial correlation"
        case "normalized_correlation":
            y_max = 1
            y_ticks = np.linspace(0, y_max, 6)
            y_label = "Normalized correlation"
        case _:
            raise ValueError(f"Invalid metric: {metric}")

    ax.set_ylim(0, y_max + 0.01)
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label=y_label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    plot.set_ticks_params(ax, length=2, pad=1)

    ax.set_title(title, fontsize=LABEL_FONTSIZE, linespacing=0.9, pad=0)
    ax.spines["left"].set_bounds(0, y_max)  # trim y-axis spine
    sns.despine(ax=ax)

    plot.save_figure(figure, filename=filename, dpi=2 * DPI)


def main():
    for metric in ["correlation", "normalized_correlation"]:
        lRomul = load_results(model_dir=OUTPUT_DIR / "lRomul", metric=metric)
        viv1t = load_results(model_dir=OUTPUT_DIR / "viv1t", metric=metric)
        plot_histogram(
            viv1t=viv1t["movie"],
            lRomul=lRomul["movie"],
            title="Prediction performance\nto unseen movies",
            metric=metric,
            filename=PLOT_DIR / metric / "movie_performance.png",
        )
        plot_histogram(
            viv1t=viv1t["drifting gabor"],
            lRomul=lRomul["drifting gabor"],
            title="Prediction performance\nto drifting Gabor",
            metric=metric,
            filename=PLOT_DIR / metric / "gabor_performance.png",
        )
        plot_histogram(
            viv1t=viv1t["directional pink noise"],
            lRomul=lRomul["directional pink noise"],
            title="Prediction performance\nto directional pink noise",
            metric=metric,
            filename=PLOT_DIR / metric / "pink_noise_performance.png",
        )
        plot_histogram(
            viv1t=viv1t["gaussian dots"],
            lRomul=lRomul["gaussian dots"],
            title="Prediction performance\nto Gaussian dots",
            metric=metric,
            filename=PLOT_DIR / metric / "gaussian_dots_performance.png",
        )
    print(f"saved plots to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
