from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from estimate_selectivity_indexes import load_data
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.offsetbox import AuxTransformBox
from matplotlib.patches import FancyArrow
from scipy.stats import sem

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

SI_THRESHOLD = 0.2
plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

DATA_DIR = Path("../data")
PLOT_DIR = Path("figures") / "tuning_curves"
OUTPUT_DIR = Path("../runs/vivit/204_causal_viv1t")


def replace_ticks_with_arrows(ax: Axes, angles: list[float] | np.ndarray):
    """Replaces the ticks on the given axis with rotated arrows."""
    ax.set_xticks(angles)
    for angle in angles:
        # Create arrow
        arrow = FancyArrow(
            x=0,
            y=0,
            dx=0,
            dy=0.28,
            width=0.03,
            head_width=0.06,
            head_length=0.16,
            length_includes_head=True,
            facecolor="gray",
            edgecolor="none",
            alpha=0.6,
        )
        # Create perpendicular line
        line = Line2D(
            xdata=[-0.03, 0.03],
            ydata=[0, 0],
            color="black",
            linewidth=2,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )
        # Create a transformation that includes rotation and scaling
        transform_box = AuxTransformBox(
            ax.transAxes + mtransforms.Affine2D().rotate_deg(angle + 90)
        )
        transform_box.add_artist(arrow)
        transform_box.add_artist(line)
        # Position the arrow just below the x-axis
        ab = AnnotationBbox(
            offsetbox=transform_box,
            xy=(angle + 0.5, -0.45),
            frameon=False,
            box_alignment=(0.5, 0.3),
            xycoords=("data", "axes fraction"),
            pad=0,
            bboxprops=dict(edgecolor="none"),
        )
        ax.add_artist(ab)
    # Remove the default xtick labels
    ax.set_xticklabels([])


def plot_tuning_curves_joint(
    tuning_curves: dict[str, np.ndarray],
    neurons: list[int] | np.ndarray,
    filename: Path,
    legend: bool = False,
    label: bool = True,
):
    figure, axs = plt.subplots(
        nrows=len(neurons),
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 1.3),
        gridspec_kw={
            "top": 0.99,
            "right": 1,
            "bottom": 0.15,
            "left": 0.05,
            "wspace": 0.0,
            "hspace": 0.25,
        },
        dpi=DPI,
        sharex=True,
    )

    directions = np.array(list(data.DIRECTIONS.keys()), dtype=int)

    min_value, max_value = 0, 1.3
    custom_handles = []
    for i, (ax, neuron) in enumerate(zip(axs.flatten(), neurons), 1):
        for model, responses in tuning_curves.items():
            response = responses[neuron]  # response.shape = (direction, repeats)
            value = np.mean(response, axis=1)
            yerr = sem(response, axis=1)
            neuron_max = np.max(value + yerr)
            value = value / neuron_max
            yerr = yerr / neuron_max
            color = plot.get_color(model)
            ax.plot(
                directions,
                value,
                alpha=0.4,
                label=model.capitalize(),
                color=color,
                linewidth=1.5,
                clip_on=False,
                zorder=-1,
            )
            ax.errorbar(
                directions,
                value,
                yerr=yerr,
                fmt=".",
                elinewidth=1.5,
                capsize=2,
                capthick=1.5,
                alpha=0.8,
                clip_on=False,
                color=color,
                linestyle="",
                markersize=3,
                zorder=0 if model == "recorded" else 1,
            )
            ax.xaxis.set_ticks(directions)
            ax.yaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            ax.spines["left"].set_visible(False)
            ax.set_ylim(min_value, max_value)
            sns.despine(ax=ax, top=True, right=True, left=True, trim=True)
            plot.set_ticks_params(ax)
            if i == 1:
                custom_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        label=model.capitalize(),
                        linestyle="-",
                        linewidth=2,
                        solid_capstyle="butt",
                        solid_joinstyle="miter",
                    )
                )

    for i, ax in enumerate(axs.flatten(), 1):
        ax.text(
            x=directions[0] - 5,
            y=max_value,
            s=f"Neuron {i}",
            ha="left",
            va="top",
            fontsize=TICK_FONTSIZE,
        )

    if legend:
        legend = axs[0].legend(
            handles=custom_handles,
            loc="upper right",
            bbox_to_anchor=(directions[-1], max_value),
            bbox_transform=axs[0].transData,
            ncols=1,
            fontsize=TICK_FONTSIZE,
            frameon=False,
            handletextpad=0.3,
            handlelength=0.8,
            labelspacing=0.05,
            columnspacing=0,
            borderpad=0,
            borderaxespad=0,
        )
        axs[-1].text(
            x=-10,
            y=0.1,
            s="ΔF/F",
            fontsize=LABEL_FONTSIZE,
            ha="right",
            rotation=90,
        )

        # Adjust the title position
        for lh in legend.legend_handles:
            lh.set_alpha(1)
    replace_ticks_with_arrows(axs[-1], directions)
    plot.save_figure(figure, filename=filename, layout="none", dpi=DPI)


def plot_tuning_curves(
    tuning_curves: dict[str, np.ndarray],
    neuron: int,
    filename: Path,
    legend: bool = False,
    label: bool = True,
    neuron_id: int | None = None,
):
    if neuron_id is None:
        neuron_id = neuron

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2), dpi=DPI)

    directions = np.array(list(data.DIRECTIONS.keys()), dtype=int)

    min_value, max_value = 0, 1.0
    linewidth = 2
    for model, responses in tuning_curves.items():
        response = responses[neuron]  # response.shape = (direction, repeats)
        response = response / np.max(np.mean(response, axis=1))
        values = np.mean(response, axis=1)
        error_bars = sem(response, axis=1)
        color = plot.get_color(model)
        ax.plot(
            directions,
            values,
            alpha=0.4,
            label=model,
            color=color,
            linewidth=linewidth,
            clip_on=False,
            zorder=-1,
        )
        ax.errorbar(
            directions,
            values,
            yerr=error_bars,
            fmt=".",
            elinewidth=2,
            capsize=2.5,
            capthick=2,
            alpha=0.8,
            clip_on=False,
            color=color,
            linestyle="",
            markersize=3,
            zorder=0 if model == "recorded" else 1,
        )
        max_value = max(max_value, np.max(values + error_bars))

    if legend:
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(directions[0], max_value),
            ncols=1,
            fontsize=TICK_FONTSIZE,
            frameon=False,
            title="",
            handletextpad=0.35,
            handlelength=0.5,
            markerscale=0.5,
            labelspacing=0.0,
            columnspacing=1,
            borderpad=0,
            borderaxespad=0,
            bbox_transform=ax.transData,
        )
        ax.text(
            x=directions[-1],
            y=1.5,
            s="ΔF/F",
            fontsize=TICK_FONTSIZE,
            ha="right",
        )

        # Adjust the title position
        for lh in legend.legend_handles:
            lh.set_alpha(1)
        for text in legend.texts:
            text.set_y(1)

    ax.set_yticks([])
    ax.set_ylim(min_value, max_value)

    # ax.set_ylabel("ΔF/F", fontsize=LABEL_FONTSIZE)
    if label:
        replace_ticks_with_arrows(ax, directions)
    else:
        plot.set_xticks(
            ax,
            ticks=directions,
            tick_labels=[],
            label="",
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=LABEL_FONTSIZE,
            label_pad=0,
        )
    ax.set_title(
        f"N{neuron_id}", fontsize=TICK_FONTSIZE, ha="left", va="top", y=0.95, x=0.82
    )
    # figure.text(
    #    x=directions[-1],
    #    y=max_value,
    #    s=f"Neuron {neuron_id}",
    #    fontsize=LABEL_FONTSIZE,
    #    ha="right",
    #    va="top",
    #    transform=ax.transData,
    # )
    sns.despine(ax=ax, top=True, right=True, left=True, trim=True)
    plot.set_ticks_params(ax, length=5, pad=3)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    mouse_id = "B"
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 4)[0]
    tuning_curves = {
        "recorded": load_data(
            mouse_id=mouse_id, output_dir=DATA_DIR, trial_ids=trial_ids
        ),
        "predicted": load_data(
            mouse_id=mouse_id, output_dir=OUTPUT_DIR, trial_ids=trial_ids
        ),
    }
    selective_neurons = utils.get_selective_neurons(
        save_dir=OUTPUT_DIR,
        mouse_id=mouse_id,
        threshold=SI_THRESHOLD,
        tuning_type="direction",
    )

    # randomly select 3 neurons
    rng = np.random.default_rng(1234)
    neurons = rng.choice(selective_neurons, size=3, replace=False)
    neurons = sorted(neurons)
    print(neurons)
    plot_tuning_curves_joint(
        tuning_curves,
        neurons=neurons,
        filename=PLOT_DIR / f"mouse{mouse_id}_example_tuning_curves.{FORMAT}",
        legend=True,
        label=False,
    )
    for i, neuron in enumerate(neurons):
        plot_tuning_curves(
            tuning_curves,
            neuron=neuron,
            # legend=i == 0,
            legend=False,
            filename=PLOT_DIR / f"mouse{mouse_id}_neuron{neuron}.{FORMAT}",
            label=i == len(neurons) - 1,
            neuron_id=i + 2,
        )
    neuron = 7815
    plot_tuning_curves(
        tuning_curves,
        neuron=neuron,
        legend=True,
        filename=PLOT_DIR / f"tuning_curve.{FORMAT}",
        label=False,
        neuron_id=1,
    )
    print(f"saved plots to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
