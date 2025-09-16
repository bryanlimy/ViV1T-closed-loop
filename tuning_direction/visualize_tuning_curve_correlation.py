from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.offsetbox import AuxTransformBox
from matplotlib.patches import FancyArrow
from matplotlib.ticker import MultipleLocator
from scipy.stats import sem
from tqdm import tqdm

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()


DATA_DIR = Path("../data/sensorium")
# DATA_DIR = Path("/mnt/storage/data/sensorium")
PLOT_DIR = Path("figures") / "tuning_curve_correlation"

SI_THRESHOLD = 0.2

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLOCK_SIZE = 25  # number of frames for each drifting Gabor filter


def get_shuffled_tuning_curves(
    mouse_id: str,
    output_dir: Path,
    rng: np.random.RandomState,
    tuning_type: str,
) -> np.ndarray:
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 4)[0]

    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot find {filename}.")

    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]

    # reshape responses to (neurons, trials, blocks, frames)
    responses = rearrange(
        responses,
        "trial neuron (block frame) -> neuron trial block frame",
        frame=BLOCK_SIZE,
    )
    # compute average response per block
    responses = np.mean(responses, axis=-1)

    # get direction parameters for each frame
    gabor_parameters = np.array(
        [
            data.get_gabor_parameters(mouse_id, trial_id=trial_id)
            for trial_id in trial_ids
        ],
        dtype=np.float32,
    )
    directions = gabor_parameters[:, -num_frames:, 0]
    # get direction for each block
    directions = rearrange(
        directions, "trial (block frame) -> trial block frame", frame=BLOCK_SIZE
    )
    directions = directions[:, :, 0].astype(int)

    # combine block and trial dimensions
    responses = rearrange(responses, "neuron trial block -> neuron (trial block)")
    directions = rearrange(directions, "trial block -> (trial block)")

    # shuffle responses
    responses = np.stack([rng.permutation(response) for response in responses])

    # find the minimum number of presentations in each direction
    unique_directions, counts = np.unique(directions, return_counts=True)
    min_samples = min(counts)

    # randomly select min_samples for each direction
    tuning_curves = np.zeros(
        (num_neurons, len(unique_directions), min_samples), dtype=np.float32
    )
    for i, direction in enumerate(unique_directions):
        index = np.where(directions == direction)[0]
        index = rng.choice(index, size=min_samples, replace=False)
        tuning_curves[:, i, :] = responses[:, index]

    tuning_curves = np.mean(tuning_curves, axis=2)  # average response over repeats
    if tuning_type == "orientation":
        tuning_curves = (tuning_curves[:, :4] + tuning_curves[:, 4:]) / 2
    return tuning_curves


def replace_ticks_with_arrows(ax: Axes, angles: list[float] | np.ndarray):
    """Replaces the ticks on the given axis with rotated arrows."""
    ax.set_xticks(angles)
    for angle in angles:
        # Create arrow
        arrow = FancyArrow(
            x=0,
            y=0,
            dx=0,
            dy=0.3,
            width=0.035,
            head_width=0.08,
            head_length=0.15,
            length_includes_head=True,
            facecolor="gray",
            edgecolor="none",
            alpha=0.6,
        )
        # Create perpendicular line
        line = Line2D(
            xdata=[-0.035, 0.035],
            ydata=[0, 0],
            color="black",
            linewidth=1.6,
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
            xy=(angle + 0.5, -0.525),
            frameon=False,
            box_alignment=(0.5, 0.3),
            xycoords=("data", "axes fraction"),
            pad=0,
            bboxprops=dict(edgecolor="none"),
        )
        ax.add_artist(ab)
    # Remove the default xtick labels
    ax.set_xticklabels([])


def get_tuning_curves(save_dir: Path, mouse_id: str, tuning_type: str) -> np.ndarray:
    tuning = utils.load_tuning(save_dir, mouse_id=mouse_id)
    tuning_curves = tuning["tuning_curve"]
    if tuning_type == "orientation":
        tuning_curves = (tuning_curves[:, :4] + tuning_curves[:, 4:]) / 2
    return tuning_curves


def compute_tuning_curve_correlation(
    tuning_curves1: np.ndarray, tuning_curves2: np.ndarray
) -> np.ndarray:
    assert tuning_curves1.shape == tuning_curves2.shape
    correlations = np.stack(
        [
            np.corrcoef(tuning_curves1[n], tuning_curves2[n])[0, 1]
            for n in range(tuning_curves1.shape[0])
        ],
        dtype=np.float32,
    )
    return correlations


def plot_correlations(
    correlations: pd.DataFrame,
    filename: Path,
    annotate_neurons: bool = False,
):
    figure_width = (1 / 3) * PAPER_WIDTH
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, 1.3),
        gridspec_kw={
            "top": 0.95,
            "right": 0.95,
            "bottom": 0.22,
            "left": 0.14,
            "wspace": 0,
            "hspace": 0,
        },
        dpi=DPI,
    )

    model_names = correlations.model_name.unique()
    tuning_types = correlations.tuning_type.unique()

    max_value = 0
    legend_handles, colors = [], {}
    for model_name in model_names:
        color = plot.get_color(model_name)
        colors[model_name] = color
        for tuning_type in tuning_types:
            correlation = correlations[
                (correlations.tuning_type == tuning_type)
                & (correlations.model_name == model_name)
            ].correlation.values
            linewidth = 1.5 if tuning_type == "direction" else 1.0
            linestyle = (0, (0.75, 0.75)) if tuning_type == "orientation" else "-"
            zorder = 20 if tuning_type == "direction" else 10
            legend_kws = {
                "solid_capstyle": "butt",
                "solid_joinstyle": "miter",
                "linewidth": 2,
                "linestyle": linestyle,
                "color": color,
                "label": "Predicted" if "viv1t" in model_name.lower() else model_name,
            }
            h_y, h_x, _ = ax.hist(
                correlation,
                bins=20,
                alpha=0.8,
                range=(-1.0, 1.0),
                histtype="step",
                linewidth=linewidth,
                linestyle=linestyle,
                clip_on=False,
                weights=np.ones(len(correlation)) / len(correlation),
                color=color,
                zorder=zorder,
            )
            legend_handles.append(Line2D([0], [0], **legend_kws))
            max_value = max(max_value, np.max(h_y))
    max_value = np.ceil(max_value * 10) / 10
    # max_value = 0.5
    # for i, (model_name, handles) in enumerate(list(legend_handles.items())[::-1]):
    #     legend = ax.legend(
    #         handles=handles,
    #         loc="upper left",
    #         bbox_to_anchor=(-0.9, max_value - (0.2 * max_value * i)),
    #         bbox_transform=ax.transData,
    #         frameon=False,
    #         fontsize=TICK_FONTSIZE,
    #         title_fontsize=TICK_FONTSIZE,
    #         alignment="left",
    #         handletextpad=0.3,
    #         handlelength=0.8,
    #         labelspacing=0.05,
    #         columnspacing=0,
    #         borderpad=0,
    #         borderaxespad=0,
    #     )
    #     legend.get_title().set_color(colors[model_name])
    #     for text in legend.get_texts():
    #         text.set_color(colors[model_name])
    #     for lh in legend.legend_handles:
    #         lh.set_alpha(1)
    #     ax.add_artist(legend)
    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(-0.9, max_value),
        bbox_transform=ax.transData,
        frameon=False,
        fontsize=TICK_FONTSIZE,
        title_fontsize=TICK_FONTSIZE,
        alignment="left",
        handletextpad=0.3,
        handlelength=0.8,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    ax.add_artist(legend)

    ax.set_xlim(-1, 1)
    # ax.set_title(
    #     f"Recorded vs. predicted\ntuning curve correlation",
    #     fontsize=TITLE_FONTSIZE,
    #     pad=0,
    #     linespacing=0.9,
    #     y=1.03,
    # )

    plot.set_xticks(
        ax,
        ticks=[-1, 0, 1],
        tick_labels=["-1", "0", "1"],
        label="Tuning curve correlation",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    yticks = np.array([0, max_value])
    plot.set_yticks(
        ax,
        ticks=yticks,
        tick_labels=(100 * yticks).astype(int),
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )
    ax.set_ylim(yticks[0], yticks[-1])
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    sns.despine(ax=ax)
    plot.set_ticks_params(ax, length=3, minor_length=3)

    if annotate_neurons:
        df = correlations[
            (correlations.model_name == "ViV1T") & (correlations.mouse == "B")
        ]
        if df.size > 0:
            neurons = [("Neuron 1", 7666), ("Neuron 2", 7676), ("Neuron 3", 7747)]
            for neuron_name, neuron in neurons:
                x = df[(df.neuron == neuron)].iloc[0].correlation
                y = h_y[np.where(x < h_x)[0][0] - 1]
                ax.scatter(
                    x=x,
                    y=y,
                    s=25,
                    marker="*",
                    # alpha=0.8,
                    # linewidth=1.2,
                    zorder=30,
                    edgecolors="limegreen",
                    facecolors="none",
                    clip_on=False,
                )
                ax.text(
                    x=x - 0.04,
                    y=y + (0.04 if neuron_name == "Neuron 1" else 0.01),
                    s=neuron_name,
                    fontsize=TICK_FONTSIZE - 1,
                    va="bottom",
                    ha="right",
                    zorder=30,
                )

    plot.save_figure(figure, filename=filename, layout="none", dpi=DPI)


def plot_correlation_comparison(
    correlations: pd.DataFrame, tuning_type: str, filename: Path
):
    figure_width, figure_height = (1 / 2) * PAPER_WIDTH, 1.5
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(figure_width, figure_height),
        dpi=DPI,
    )

    assert (
        correlations.tuning_type.nunique() == 1
        and correlations.tuning_type.unique()[0] == tuning_type
    )
    model_names = correlations.model_name.unique()

    linewidth = 1.5
    linestyle = "-"
    legend_kws = {
        "solid_capstyle": "butt",
        "solid_joinstyle": "miter",
        "linewidth": 2,
        "linestyle": linestyle,
    }

    max_value = 0

    colors = {}
    legend_handles = {}
    for model_name in model_names:
        color = plot.get_color(model_name)
        colors[model_name] = color
        correlation = correlations[
            correlations.model_name == model_name
        ].correlation.values
        h_y, h_x, _ = ax.hist(
            correlation,
            bins=25,
            alpha=0.8,
            range=(-1.0, 1.0),
            histtype="step",
            linewidth=linewidth,
            linestyle=linestyle,
            clip_on=False,
            weights=np.ones(len(correlation)) / len(correlation),
            color=color,
            # label=model_name,
        )
        legend_handles[model_name] = Line2D(
            [0], [0], color=color, label=model_name, **legend_kws
        )
        max_value = max(max_value, np.max(h_y))

    max_value = 0.1 * np.ceil(max_value * 10)

    # ax.text(
    #     x=-0.93,
    #     y=max_value,
    #     s=tuning_type.capitalize(),
    #     fontsize=TICK_FONTSIZE,
    #     transform=ax.transData,
    #     va="top",
    #     ha="left",
    # )
    # y_legend = 0.1 * max_value
    for i, (model_name, handles) in enumerate(legend_handles.items()):
        legend = ax.legend(
            handles=[handles],
            loc="upper left",
            bbox_to_anchor=(0.12, max_value - (0.1 * max_value * i)),
            bbox_transform=ax.transData,
            frameon=False,
            fontsize=TICK_FONTSIZE,
            alignment="left",
            handletextpad=0.3,
            handlelength=0.8,
            labelspacing=0.05,
            columnspacing=0,
            borderpad=0,
            borderaxespad=0,
        )
        for lh in legend.legend_handles:
            lh.set_alpha(1)
        ax.add_artist(legend)

    ax.set_xlim(-1, 1)
    plot.set_xticks(
        ax,
        ticks=[-1, 0, 1],
        tick_labels=["-1", "0", "1"],
        label=f"{tuning_type.capitalize()} tuning curve correlation",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )
    yticks = np.array([0, max_value])
    plot.set_yticks(
        ax,
        ticks=yticks,
        tick_labels=(100 * yticks).astype(int),
        label="% of neurons",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
    )
    ax.set_ylim(yticks[0], yticks[-1])
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    sns.despine(ax=ax)
    plot.set_ticks_params(ax, minor_length=3)

    # Plot example tuning curves of a neuron
    if tuning_type == "direction":
        ax = figure.add_axes((0.16, 0.8, 0.38, 0.13))
        recorded = get_tuning_curves(
            save_dir=data.METADATA_DIR,
            mouse_id="B",
            tuning_type=tuning_type,
        )
        predicted = get_tuning_curves(
            save_dir=Path("../runs/vivit/204_causal_viv1t"),
            mouse_id="B",
            tuning_type=tuning_type,
        )
        neuron = 28
        x = np.array(list(data.DIRECTIONS.keys()), dtype=int)
        line_kw = {
            "linewidth": linewidth,
            "linestyle": linestyle,
            "clip_on": False,
            "marker": ".",
            "markersize": 6,
            "alpha": 0.7,
        }
        ax.plot(
            x,
            recorded[neuron] / np.max(recorded[neuron]),
            color="black",
            zorder=1,
            **line_kw,
        )
        ax.plot(
            x,
            predicted[neuron] / np.max(predicted[neuron]),
            color=plot.get_color("ViV1T"),
            zorder=2,
            **line_kw,
        )
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(
            x=-7,
            y=0.1,
            s="ΔF/F",
            fontsize=TICK_FONTSIZE,
            ha="right",
            rotation=90,
        )
        ax.text(
            x=x[0] - 5,
            y=1.2,
            s="Example neuron",
            fontsize=TICK_FONTSIZE,
            va="top",
            ha="left",
        )
        sns.despine(ax=ax, left=True)
        plot.set_ticks_params(ax, length=2.5)
        ax.spines["bottom"].set_zorder(0)
        replace_ticks_with_arrows(ax=ax, angles=x)

    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    models = {
        "Shuffle": data.METADATA_DIR,
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    mouse_ids = ["B", "C", "E"]
    tuning_types = ["direction"]

    rng = np.random.RandomState(1234)

    correlations = []
    for tuning_type in tuning_types:
        for mouse_id in mouse_ids:
            recorded = get_tuning_curves(
                save_dir=data.METADATA_DIR, mouse_id=mouse_id, tuning_type=tuning_type
            )
            for model_name, output_dir in tqdm(
                models.items(), desc=f"{tuning_type} mouse {mouse_id}"
            ):
                assert output_dir.exists(), f"{output_dir} does not exist."
                # selective neurons based on recorded data
                selective_neurons = utils.get_selective_neurons(
                    # save_dir=output_dir,
                    save_dir=data.METADATA_DIR,
                    mouse_id=mouse_id,
                    threshold=SI_THRESHOLD,
                    tuning_type=tuning_type,
                )
                if model_name == "Shuffle":
                    # predicted = get_shuffled_tuning_curves(
                    #     mouse_id=mouse_id,
                    #     output_dir=DATA_DIR,
                    #     rng=rng,
                    #     tuning_type=tuning_type,
                    # )
                    predicted = recorded.copy()
                    for n in range(predicted.shape[0]):
                        predicted[n] = rng.permutation(predicted[n])
                else:
                    predicted = get_tuning_curves(
                        save_dir=output_dir,
                        mouse_id=mouse_id,
                        tuning_type=tuning_type,
                    )
                correlation = compute_tuning_curve_correlation(
                    tuning_curves1=recorded[selective_neurons],
                    tuning_curves2=predicted[selective_neurons],
                )
                correlation = pd.DataFrame(
                    {"neuron": selective_neurons, "correlation": correlation}
                )
                correlation["model_name"] = model_name
                correlation["tuning_type"] = tuning_type
                correlation["mouse"] = mouse_id
                correlations.append(correlation)
    correlations = pd.concat(correlations, ignore_index=True)
    for model_name in correlations.model_name.unique():
        if model_name == "Shuffle":
            continue
        for mouse_id in mouse_ids:
            plot_correlations(
                correlations=correlations[
                    (correlations.mouse == mouse_id)
                    & correlations.model_name.isin(["Shuffle", model_name])
                ],
                filename=PLOT_DIR
                / model_name
                / f"tuning_curve_correlation_mouse{mouse_id}.{FORMAT}",
            )
        plot_correlations(
            correlations=correlations[
                correlations.model_name.isin(["Shuffle", model_name])
            ],
            filename=PLOT_DIR / model_name / f"tuning_curve_correlation.{FORMAT}",
            # annotate_neurons=model_name == "ViV1T",
            annotate_neurons=False,
        )

    for tuning_type in tuning_types:
        for model_name in correlations.model_name.unique():
            correlation = correlations[
                (correlations.model_name == model_name)
                & (correlations.tuning_type == tuning_type)
            ].correlation.values
            print(
                f"Model {model_name} {tuning_type} tuning curve (N={len(correlation)}) average "
                f"correlation: {np.mean(correlation):.03f} +/- {sem(correlation):.03f}\n"
            )

    for tuning_type in tuning_types:
        plot_correlation_comparison(
            correlations=correlations[correlations.tuning_type == tuning_type],
            tuning_type=tuning_type,
            filename=PLOT_DIR / f"{tuning_type}_tuning_curve_correlation.{FORMAT}",
        )

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
