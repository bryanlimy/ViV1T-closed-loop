import warnings
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib import axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy import stats
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import stimulus

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8
TICK_LENGTH, TICK_PAD, TICK_LINEWIDTH = 3, 2, 1.2

DPI = 600
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches
FPS = 30

THRESHOLD = 0.2  # response has to be 20% stronger

plot.set_font()

PLOT_DIR = Path("figures")
DATA_DIR = Path("../../data")
OUTPUT_DIR = Path("../../runs")


def get_percentage_function(df: pd.DataFrame, thresholds: np.ndarray) -> np.ndarray:
    num_mice = df.mouse.nunique()
    percentages = np.full(
        shape=(len(thresholds), num_mice), fill_value=np.nan, dtype=np.float32
    )
    for i, threshold in enumerate(thresholds):
        for j, mouse_id in enumerate(df.mouse.unique()):
            high_iso = df[df.mouse == mouse_id].high_iso.values
            high_cross = df[df.mouse == mouse_id].high_cross.values
            low_iso = df[df.mouse == mouse_id].low_iso.values
            low_cross = df[df.mouse == mouse_id].low_cross.values

            num_neurons = len(high_iso)

            high_contrast = ((high_cross - high_iso) / high_iso) >= threshold
            low_contrast = ((low_iso - low_cross) / low_cross) >= threshold
            flip_neurons = (high_contrast == True) & (low_contrast == True)
            percentages[i, j] = np.count_nonzero(flip_neurons) / num_neurons
    percentages = 100 * percentages
    return percentages


def main():
    in_vivo_df = pd.read_parquet(PLOT_DIR / "in_vivo.parquet")
    in_silico_df = pd.read_parquet(PLOT_DIR / "in_silico_rochefort_lab.parquet")
    sensorium_df = pd.read_parquet(PLOT_DIR / "in_silico_sensorium.parquet")

    thresholds = np.arange(31) / 100
    sensorium_percentages = get_percentage_function(sensorium_df, thresholds=thresholds)
    in_vivo_percentages = get_percentage_function(in_vivo_df, thresholds=thresholds)
    in_silico_percentages = get_percentage_function(in_silico_df, thresholds=thresholds)

    filename = PLOT_DIR / "flip_neuron_percentage_function.jpg"

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    min_value, max_value = 0, 0
    for percentages, color, linestyle, label in [
        (in_vivo_percentages, "black", "-", "in vivo (RL)"),
        (in_silico_percentages, "limegreen", "-", "in silico (RL)"),
        (sensorium_percentages, "limegreen", (0, (1, 1)), "in silico (Sen.)"),
    ]:
        mean = np.mean(percentages, axis=1)
        se = stats.sem(percentages, axis=1)
        max_value = max(max_value, np.max(mean + se))
        min_value = min(min_value, np.min(mean - se))
        ax.plot(
            thresholds,
            mean,
            color=color,
            linewidth=1.8,
            linestyle=linestyle,
            alpha=0.8,
            clip_on=False,
            zorder=1,
            label=label,
        )
        ax.fill_between(
            thresholds,
            y1=mean - se,
            y2=mean + se,
            facecolor=color,
            edgecolor="none",
            linewidth=2,
            alpha=0.3,
            zorder=1,
            clip_on=False,
        )

    max_value = np.ceil(max_value / 10) * 10

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.3, max_value),
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

    ax.set_xlim(thresholds[0], thresholds[-1])
    plot.set_xticks(
        axis=ax,
        ticks=[0, 0.1, 0.2, 0.3],
        tick_labels=[0, 10, 20, 30],
        label="Threshold (%)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=0,
    )
    y_ticks = np.array([min_value, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-6,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plot.set_ticks_params(ax)
    sns.despine(ax=ax)

    plot.save_figure(figure, filename=filename, dpi=DPI)


if __name__ == "__main__":
    main()
