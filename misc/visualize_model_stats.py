from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from viv1t.utils import plot

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

PLOT_DIR = Path("figures")


def main():
    model_names = ["LN", "fCNN", "DwiseNeuro", "ViV1T"]
    colors = [plot.get_color(model_name) for model_name in model_names]

    inference_time = [37.2, 34.7, 0.45, 10.41]
    prediction_performance = [0.247, 0.551, 0.660, 0.696]

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((4 / 9) * PAPER_WIDTH, 1.5),
        # gridspec_kw={
        #     "wspace": 0,
        #     "hspace": 0,
        #     "left": 0.16,
        #     "top": 0.87,
        #     "right": 0.83,
        #     "bottom": 0.19,
        # },
        dpi=DPI,
    )

    linewidth = 1.2
    bar_width = 0.25
    pad = 0.165

    # plt inference time
    x = np.arange(len(model_names)) - pad
    color = "black"
    ax.bar(
        x,
        inference_time,
        width=bar_width,
        edgecolor=color,
        facecolor="none",
        linewidth=linewidth,
    )
    for i in range(len(inference_time)):
        ax.text(
            x=x[i] + 0.01 * np.max(x),
            y=inference_time[i] + 0.025 * np.max(inference_time),
            s=np.round(inference_time[i], 1),
            ha="center",
            va="bottom",
            fontsize=TICK_FONTSIZE,
            color=color,
            rotation=90,
        )

    y_ticks = np.linspace(0, 40, 5, dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=["0", "", "", "", "40"],
        label="Trial(s)\nper second",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-8,
        linespacing=0.9,
    )
    plot.set_ticks_params(ax, minor_length=3)

    x_ticks = np.arange(len(model_names))
    ax.set_xlim(x_ticks[0] - 0.4, x_ticks[-1] + 0.4)
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=[],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    ax.tick_params(axis="x", length=3, pad=0)
    for i in range(len(model_names)):
        ax.text(
            x=x_ticks[i],
            # y=-2.5 if i % 2 != 0 else -6.5,
            y=-2.6,
            s=model_names[i],
            fontsize=TICK_FONTSIZE,
            va="top",
            ha="center",
            color=colors[i],
            alpha=1,
        )

    sns.despine(ax=ax, right=True, top=True)

    ax_right = ax.twinx()
    color = "dimgray"
    x = np.arange(len(model_names)) + pad
    ax_right.bar(
        x,
        prediction_performance,
        width=bar_width,
        edgecolor=color,
        facecolor="none",
        linewidth=linewidth,
    )
    for i in range(len(prediction_performance)):
        ax_right.text(
            x=x[i] + 0.01 * np.max(x),
            y=prediction_performance[i] + 0.025 * np.max(prediction_performance),
            s=f"{prediction_performance[i]:.03f}",
            ha="center",
            va="bottom",
            fontsize=TICK_FONTSIZE,
            color=color,
            rotation=90,
        )

    y_ticks = np.array([0, 0.7])
    ax_right.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax_right,
        label="Prediction\nperformance",
        ticks=y_ticks,
        tick_labels=y_ticks,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label_pad=-10,
        linespacing=0.9,
    )
    ax_right.yaxis.label.set_color(color)
    ax_right.yaxis.set_minor_locator(MultipleLocator(0.1))
    plot.set_ticks_params(ax_right, minor_length=3, color=color)
    sns.despine(ax=ax_right, top=True, right=False)

    ax.set_zorder(ax_right.get_zorder() + 1)
    ax.patch.set_visible(False)

    filename = PLOT_DIR / f"model_stats.{FORMAT}"
    plot.save_figure(
        figure, filename=filename, pad_inches=0, layout="constrained", dpi=DPI
    )
    print(f"Saved plot to {filename}.")


if __name__ == "__main__":
    main()
