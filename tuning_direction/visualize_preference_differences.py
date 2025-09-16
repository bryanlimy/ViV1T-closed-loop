from collections import defaultdict
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from scipy.stats import sem
from visualize_tuning_width import get_tuning_curves

from viv1t import data
from viv1t.data.constants import DIRECTIONS
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()


DATA_DIR = Path("../data")
# DATA_DIR = Path("/mnt/storage/data/sensorium")
PLOT_DIR = Path("figures") / "preference_difference"

SI_THRESHOLD = 0.2

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches


def get_differences(array1: np.ndarray, array2: np.ndarray, size: int):
    """Get the element-wise closest distance between two arrays in circular space"""
    diff = np.abs(array1 - array2)
    return np.minimum(size - diff, diff)


def get_tuning_preferences(
    output_dir: Path,
    tuning_type: str,
    selective_dir: Path,
) -> pd.DataFrame:
    mouse_ids = ["B", "C", "E"]

    results = defaultdict(list)
    for mouse_id in mouse_ids:
        neurons = utils.get_selective_neurons(
            save_dir=selective_dir,
            mouse_id=mouse_id,
            threshold=SI_THRESHOLD,
            tuning_type=tuning_type,
        )

        real_tuning_curves = get_tuning_curves(
            data.METADATA_DIR, mouse_id=mouse_id, tuning_type=tuning_type
        )[0][neurons]
        real_preferences = np.argmax(real_tuning_curves, axis=1)

        model_tuning_curves = get_tuning_curves(
            output_dir, mouse_id=mouse_id, tuning_type=tuning_type
        )[0][neurons]
        model_preferences = np.argmax(model_tuning_curves, axis=1)

        results["pref_true"].extend(
            [list(DIRECTIONS.keys())[pref] for pref in real_preferences]
        )
        results["pref_pred"].extend(
            [list(DIRECTIONS.keys())[pref] for pref in model_preferences]
        )
        results["mouse"].extend([mouse_id] * len(model_preferences))
        results["value"].extend(
            get_differences(
                real_preferences, model_preferences, size=real_tuning_curves.shape[1]
            )
            * 45
        )
    return pd.DataFrame(results)


def plot_tuning_preference_heatmap(
    df: pd.DataFrame,
    tuning_type: str,
    filename: Path,
):
    heatmap_data = pd.crosstab(df["pref_pred"], df["pref_true"], normalize="all")
    heatmap_data = 100 * heatmap_data

    figure_width = (1 / 3) * PAPER_WIDTH
    figure, ax = plt.subplots(
        figsize=(figure_width, 0.8 * figure_width),
        layout="constrained",
        dpi=DPI,
    )

    spine_linewidth = 1

    min_value = 0
    max_value = int(np.max(heatmap_data))

    heatmap = sns.heatmap(
        heatmap_data,
        vmin=min_value,
        vmax=max_value,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.65, "pad": 0.05},
        cmap="binary",  # Reds
    )
    heatmap.figure.axes[-1].tick_params(labelsize=TICK_FONTSIZE, length=0)

    # colorbar
    cbar_ax = ax.collections[0].colorbar.ax
    cbar_ax.set_ylim(min_value, max_value)
    y_ticks = np.array([min_value, max_value], dtype=int)
    plot.set_yticks(
        axis=cbar_ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="% neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-1,
    )
    for spine in cbar_ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(spine_linewidth)
    plot.set_ticks_params(axis=cbar_ax, length=2, pad=1, linewidth=spine_linewidth)

    unique_values = sorted(set(df["pref_true"]).union(set(df["pref_pred"])))

    ticks = np.arange(len(unique_values))
    plot.set_xticks(
        ax,
        ticks=ticks + 0.55,
        tick_labels=[str(v) for v in unique_values],
        label=f"recorded pref. {tuning_type[:3]}.",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=1,
        rotation=90,
        va="top",
        ha="center",
    )
    plot.set_yticks(
        ax,
        ticks=ticks + 0.44,
        tick_labels=[str(v) for v in unique_values],
        label=f"predicted pref. {tuning_type[:3]}.",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=1,
        rotation=0,
    )
    # plot
    # ax.tick_params(axis="x", length=0, pad=1)
    # ax.tick_params(axis="y", length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(True)
    plot.set_ticks_params(axis=ax, length=0, pad=2, linewidth=spine_linewidth)
    plt.gca().invert_yaxis()
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_tuning_preference_differences(
    df: pd.DataFrame,
    tuning_type: str,
    filename: Path,
    title: str = None,
    x_label: str = "Δ pref. (°)",
):
    num_mice = df.mouse.nunique()
    num_neurons = len(df)
    match tuning_type:
        case "orientation":
            x = np.arange(3)
            angles = x * 45
        case "direction":
            x = np.arange(5)
            angles = x * 45
        case _:
            raise ValueError(f"Unknown tuning type: {tuning_type}")
    # sum the number of neurons with the same preference difference for each mouse
    df = df.groupby(["mouse", "value"]).size().reset_index(name="count")
    # each if each mouse has the same value as angles
    # if not, add the entry and set value and percentage to 0
    concat = [df]
    for mouse in df["mouse"].unique():
        for angle in angles:
            if df[df["value"] == angle].empty:
                concat.append(
                    pd.DataFrame({"mouse": [mouse], "value": [angle], "count": [0]})
                )
    df = pd.concat(concat, ignore_index=True)
    # normalize the count to percentage
    df["percentage"] = 100 * df["count"] / df.groupby("mouse")["count"].transform("sum")
    # sort the values by mouse and then by value
    df = df.sort_values(by=["mouse", "value"])
    # compute average percentage over mouse
    height = df.groupby("value").percentage.apply(np.mean).values
    # compute sem over mouse
    yerr = df.groupby("value").percentage.apply(sem).values
    # print("Bar heights: ", height)
    # print("Bar errors: ", yerr)

    figure_width = (1 / 3) * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, 0.8 * figure_width),
        dpi=DPI,
    )

    bar_width = 0.6
    edgewidth = 1.0
    # error_kw = {"elinewidth": 1.5, "capsize": 2, "capthick": 1.5}
    error_kw = {"linewidth": 1.2}

    ax.bar(
        x=x,
        height=height,
        yerr=yerr,
        width=bar_width,
        align="center",
        alpha=0.8,
        color="none",
        edgecolor="black",
        linewidth=edgewidth,
        error_kw=error_kw,
        clip_on=False,
        # label=model,
    )

    max_value = 10 * ceil(0.1 * max(height))

    x_tick_labels = [int(v) for v in angles]
    # if tuning_type == "orientation":
    #
    # else:
    #     x_tick_labels = [int(v) if i % 2 == 0 else "" for i, v in enumerate(angles)]
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plot.set_xticks(
        axis=ax,
        ticks=x,
        tick_labels=x_tick_labels,
        label=x_label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.linspace(0, max_value, int(max_value / 10) + 1)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    y_tick_labels = ["0"] + [""] * (len(y_ticks) - 2) + [int(max_value)]
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_tick_labels,
        label="% neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )

    if title is not None:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=3)
    plot.set_ticks_params(ax)
    sns.despine(ax=ax)

    # text = r"$N_{mice}$=" + f"{num_mice}"
    # text += "\n" + r"$N_{neurons}$=" + f"{num_neurons}"
    # ax.text(
    #     x=0.5 * x[-1],
    #     y=0.98 * max_value,
    #     s=text,
    #     fontsize=TICK_FONTSIZE,
    #     ha="left",
    #     va="top",
    #     linespacing=0.95,
    # )

    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_tuning_difference_comparison(
    df: pd.DataFrame,
    tuning_type: str,
    filename: Path,
    x_label: str = "Δ pref. (°)",
):
    match tuning_type:
        case "orientation":
            x = np.arange(3, dtype=np.float32)
            angles = x * 45
        case "direction":
            x = np.arange(5, dtype=np.float32)
            angles = x * 45
        case _:
            raise ValueError(f"Unknown tuning type: {tuning_type}")

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 2) * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    bar_width = 0.23
    edgewidth = 1.0
    error_kw = {"elinewidth": 0.8, "capsize": 0, "capthick": 0, "clip_on": False}
    max_value = 0
    for i, model_name in enumerate(df.model_name.unique()):
        model_x = x.copy()
        width = bar_width / 2
        if i == 0:
            model_x -= width + bar_width
        elif i == 1:
            model_x -= width
        elif i == 2:
            model_x += width
        elif i == 3:
            model_x += width + bar_width
        model_df = df[df.model_name == model_name]
        # sum the number of neurons with the same preference difference for each mouse
        model_df = model_df.groupby(["mouse", "value"]).size().reset_index(name="count")
        # each if each mouse has the same value as angles
        # if not, add the entry and set value and percentage to 0
        concat = [model_df]
        for mouse in model_df["mouse"].unique():
            for angle in angles:
                if model_df[model_df["value"] == angle].empty:
                    concat.append(
                        pd.DataFrame({"mouse": [mouse], "value": [angle], "count": [0]})
                    )
        model_df = pd.concat(concat, ignore_index=True)
        # normalize the count to percentage
        model_df["percentage"] = (
            100
            * model_df["count"]
            / model_df.groupby("mouse")["count"].transform("sum")
        )
        # sort the values by mouse and then by value
        model_df = model_df.sort_values(by=["mouse", "value"])
        # compute average percentage over mouse
        height = model_df.groupby("value").percentage.apply(np.mean).values
        # compute sem over mouse
        yerr = model_df.groupby("value").percentage.apply(sem).values

        ax.bar(
            x=model_x,
            height=height,
            yerr=yerr,
            width=bar_width,
            align="center",
            alpha=0.8,
            color=plot.get_color(model_name),
            edgecolor="none",
            linewidth=edgewidth,
            error_kw=error_kw,
            clip_on=False,
            label=model_name,
        )

        max_value = max(max_value, np.max(height))

    max_value = 10 * ceil(0.1 * max_value)

    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(x[-1] + 0.6, max_value),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.3,
        handlelength=0.7,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        bbox_transform=ax.transData,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    plot.set_xticks(
        axis=ax,
        ticks=x,
        tick_labels=angles.astype(int),
        label=x_label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    y_ticks = np.array([0, max_value])
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    sns.despine(ax=ax)
    plot.set_ticks_params(ax, minor_length=3)

    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    models = {
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    tuning_types = ["direction", "orientation"]

    for tuning_type in tuning_types:
        for model_name, output_dir in models.items():
            df = get_tuning_preferences(
                output_dir=output_dir,
                tuning_type=tuning_type,
                selective_dir=output_dir,
            )
            plot_tuning_preference_differences(
                df=df,
                tuning_type=tuning_type,
                filename=PLOT_DIR
                / model_name
                / f"{tuning_type}_preference_differences.{FORMAT}",
                title="| Recorded - Predicted |",
                x_label=f"Δ pref. {tuning_type[:3]} (°)",
            )
            plot_tuning_preference_heatmap(
                df=df,
                tuning_type=tuning_type,
                filename=PLOT_DIR
                / model_name
                / f"{tuning_type}_preference_heatmap.{FORMAT}",
            )
        df = []
        for model_name, output_dir in models.items():
            df_ = get_tuning_preferences(
                output_dir=output_dir,
                tuning_type=tuning_type,
                selective_dir=data.METADATA_DIR,
            )
            df_["model_name"] = model_name
            df.append(df_)
            del df_
        df = pd.concat(df, ignore_index=True)
        plot_tuning_difference_comparison(
            df=df,
            tuning_type=tuning_type,
            filename=PLOT_DIR
            / f"{tuning_type}_preference_difference_comparison.{FORMAT}",
            x_label=f"Δ preferred {tuning_type} (°)",
        )
    print(f"saved plots to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
