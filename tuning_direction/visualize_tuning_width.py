from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from scipy.stats import sem
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()


DATA_DIR = Path("../data")
PLOT_DIR = Path("figures") / "tuning_width"

SI_THRESHOLD = 0.2

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches


def width_barplot(
    ax: Axes,
    recorded_widths: np.ndarray,
    predicted_widths: np.ndarray,
    model: str,
):
    means = np.array([np.mean(recorded_widths), np.mean(predicted_widths)])
    sems = np.array([sem(recorded_widths), sem(predicted_widths)])
    ax.bar(
        x=[0, 1],
        height=means,
        yerr=sems,
        color=[plot.get_color("recorded"), plot.get_color(model)],
    )
    ax.set_xticks([])
    min_value = 0
    max_value = max(means + sems) + 1
    # round up to nearest even number
    max_value = int(2 * np.ceil(max_value / 2))
    y_ticks = np.linspace(min_value, max_value, 2, dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=y_ticks.astype(int),
        label="Width (°)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-10,
    )
    # ax.yaxis.set_minor_locator(MultipleLocator(10))

    p_value = wilcoxon(recorded_widths, predicted_widths).pvalue
    print(f"Wilcoxon signed-rank test p-value = {p_value:.04e}")
    height = max_value - min_value
    plot.add_p_value(
        ax=ax,
        x0=0,
        x1=1,
        y=0.02 * height + max_value,
        p_value=p_value,
        tick_length=0.04 * height,
        tick_linewidth=1,
        fontsize=LABEL_FONTSIZE,
        text_pad=0.07 * height,
    )

    sns.despine(ax=ax)
    plot.set_ticks_params(ax)


def plot_tuning_width(
    tuning_curves: dict[str, list[np.ndarray]],
    SIs: dict[str, list[np.ndarray]],
    tuning_type: str,
    filename: Path,
    title: str = None,
    num_neurons: dict[str, int] = None,
    plot_legend: bool = False,
):
    figure_width = (1 / 3) * PAPER_WIDTH
    figure, ax = plt.subplots(
        figsize=(figure_width, 0.8 * figure_width),
        dpi=DPI,
    )

    min_value, max_value = 0.0, 1.0

    bound, n = (180, 9) if tuning_type == "direction" else (90, 5)
    x = np.linspace(-bound, bound, n, dtype=np.float32)
    custom_handles = []
    for i, model in enumerate(tuning_curves):
        if isinstance(tuning_curves[model], list):
            tuning_curves[model] = np.vstack(
                [np.mean(tc, axis=0) for tc in tuning_curves[model]]
            )
        tuning_curve = np.vstack(tuning_curves[model])
        color = plot.get_color(model)
        values = np.mean(tuning_curve, axis=0)
        error_bars = sem(tuning_curve, axis=0)
        ax.plot(
            x,
            values,
            label=model.capitalize(),
            linestyle="-",
            linewidth=2.5,
            color=color,
            alpha=0.4,
            clip_on=False,
            zorder=i + 1,
        )
        ax.errorbar(
            x=x,
            y=values,
            yerr=error_bars,
            fmt=".",
            elinewidth=1.5,
            capsize=2,
            capthick=1.5,
            alpha=0.8,
            clip_on=False,
            color=color,
            linestyle="",
            markersize=3,
            zorder=i + 1,
        )
        ax.text(
            x=x[-1],
            y=max_value - (0.12 * (i)),
            s=f"{'OSI' if tuning_type == 'orientation' else 'DSI'}={np.mean(SIs[model]):.2f}",
            color=color,
            fontsize=TICK_FONTSIZE,
            ha="right",
            va="top",
        )
        label = model.capitalize() if model != "ViV1T" else "Predicted"
        custom_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                label=label,
                linestyle="-",
                linewidth=1.8,
                solid_capstyle="butt",
                solid_joinstyle="miter",
            )
        )
    if plot_legend:
        l_x, l_y = -0.85 * bound, 0.21 * max_value
        legend = ax.legend(
            handles=custom_handles,
            loc="upper left",
            bbox_to_anchor=(l_x, l_y),
            bbox_transform=ax.transData,
            ncols=1,
            fontsize=TICK_FONTSIZE,
            frameon=False,
            title="",
            handletextpad=0.2,
            handlelength=0.65,
            labelspacing=0.05,
            columnspacing=0,
            borderpad=0,
            borderaxespad=0,
        )
        for lh in legend.legend_handles:
            lh.set_alpha(1)

        model_name = list(tuning_curves.keys())[1]
        color = plot.get_color(model)
        recorded = np.delete(tuning_curves["recorded"], 4, axis=1)
        predicted = np.delete(tuning_curves[model_name], 4, axis=1)
        pvalue = wilcoxon(recorded.flatten(), predicted.flatten())[1]

        print(f"\tWilcoxon signed rank test: {pvalue:.03e}")
        left = l_x - 0.07 * bound
        right = left + 0.04 * bound
        offset = 0.1 * max_value
        top = l_y + 0.06 * max_value - offset
        bottom = l_y - 0.04 * max_value - offset
        ax.plot(
            [right, left, left, right],
            [top, top, bottom, bottom],
            color="black",
            linewidth=1.0,
            clip_on=False,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            transform=ax.transData,
        )
        p_text = plot.get_p_value_asterisk(pvalue)
        ax.text(
            left - 0.07 * bound,
            (top - bottom) / 2 + bottom,
            s=p_text,
            ha="center",
            va="center",
            rotation=90,
            fontsize=TICK_FONTSIZE,
            transform=ax.transData,
        )

    ymin, ymax = 0, 1
    ax.set_ylim(ymin, ymax)
    y_ticks = np.linspace(ymin, ymax, 6)
    y_tick_labels = [ymin] + [""] * (len(y_ticks) - 2) + [ymax]
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_tick_labels,
        label="Norm. ΔF/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-3,
    )

    if tuning_type == "orientation":
        x_tick_labels = [int(v) for v in x]
    else:
        x_tick_labels = [int(v) if abs(v) in (0, 90, 180) else "" for v in x]
    ax.set_xlim(x[0] - 30, x[-1])
    plot.set_xticks(
        ax,
        ticks=x,
        tick_labels=x_tick_labels,
        label="Degrees",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
        linespacing=0.9,
    )
    # plot degree symbol next to each tick
    # for i, pos in enumerate(ax.get_xticklabels()):
    #    if tuning_type == "direction" and i % 2 != 0:
    #        continue
    #    pos = pos.get_window_extent().transformed(ax.transAxes.inverted())
    #    ax.text(
    #        pos.x1 - 0.08 * (pos.x1 - pos.x0),
    #        pos.y1 - 0.6 * (pos.y1 - pos.y0),
    #        "°",
    #        ha="left",
    #        va="bottom",
    #        fontsize=TICK_FONTSIZE,
    #        transform=ax.transAxes,
    #    )

    plot.set_ticks_params(ax)
    if num_neurons is not None:
        ax.text(
            x=x[0],
            y=0.01 * max_value,
            s=f"N={', '.join(map(str, num_neurons.values()))}",
            fontsize=TICK_FONTSIZE,
            ha="left",
            va="bottom",
        )

    # if title is not None:
    #     ax.set_title(title, fontsize=TITLE_FONTSIZE, ha="left", loc="left", pad=3)
    sns.despine(ax=ax)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def get_tuning_curves(dir: Path, mouse_id: str, tuning_type: str):
    tuning_curves = utils.load_tuning(dir, mouse_id=mouse_id)
    SIs = tuning_curves["OSI" if tuning_type == "orientation" else "DSI"]
    tuning_curves = tuning_curves["tuning_curve"]
    if tuning_type == "orientation":
        tuning_curves = (tuning_curves[:, :4] + tuning_curves[:, 4:]) / 2
    tuning_curves /= np.max(tuning_curves, axis=1, keepdims=True)
    return tuning_curves, SIs


def get_aligned_tuning_curves(
    tuning_curves: np.ndarray, preferences: np.ndarray, tuning_type: str
):
    assert len(tuning_curves) == len(preferences)
    shifts = -preferences + (3 if tuning_type == "direction" else 1)
    for i in range(tuning_curves.shape[0]):
        tuning_curves[i] = np.roll(tuning_curves[i], shifts[i])
    # append last to beginning to get circular tuning curve
    tuning_curves = np.concatenate(
        (tuning_curves[:, -1][:, None], tuning_curves), axis=1
    )
    return tuning_curves


def estimate_population_tuning(
    models: Dict[str, Path],
    mouse_id: str,
    tuning_type: str,
    reference: Path,
) -> dict[str, np.ndarray]:
    results = {}
    SIs = {}
    # selective neurons based on recorded data
    neurons = utils.get_selective_neurons(
        save_dir=reference,
        mouse_id=mouse_id,
        threshold=SI_THRESHOLD,
        tuning_type=tuning_type,
    )

    for model, output_dir in models.items():
        # tuning = utils.load_tuning(output_dir, mouse_id=mouse_id)
        # SIs = tuning["OSI" if tuning_type == "orientation" else "DSI"]
        # neurons = np.where(SIs >= SI_THRESHOLD)[0]
        results[model] = {}
        tuning_curves, model_SIs = get_tuning_curves(
            output_dir, mouse_id=mouse_id, tuning_type=tuning_type
        )
        SIs[model] = model_SIs[neurons]
        tuning_curves = tuning_curves[neurons]
        results[model] = get_aligned_tuning_curves(
            tuning_curves=tuning_curves,
            preferences=np.argmax(tuning_curves, axis=1),
            tuning_type=tuning_type,
        )
    return results, SIs


def main():
    models = {
        "recorded": data.METADATA_DIR,
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    tuning_types = ["orientation", "direction"]
    mouse_ids = ["B", "C", "E"]

    for name, output_dir in models.items():
        if name == "recorded":
            continue
        print(f"\nProcessing model {name}...")
        plot_dir = PLOT_DIR / name
        for tuning_type in tuning_types:
            num_neurons = {}
            models_ = {"recorded": models["recorded"], name: output_dir}
            tuning_curves = {model: [] for model in models_}
            SIs = {model: [] for model in models_}
            for mouse_id in mouse_ids:
                print(f"Processing mouse {mouse_id}...")
                results, mouse_SIs = estimate_population_tuning(
                    models_,
                    mouse_id=mouse_id,
                    tuning_type=tuning_type,
                    reference=models[name],
                    # reference=data.METADATA_DIR,
                )
                plot_tuning_width(
                    tuning_curves=results,
                    SIs=mouse_SIs,
                    title=f"Mouse {mouse_id}",
                    tuning_type=tuning_type,
                    filename=plot_dir
                    / f"{tuning_type}_tuning_width_mouse{mouse_id}.{FORMAT}",
                    plot_legend=tuning_type == "orientation",
                )
                # store the average aligned tuning curve for each mouse and model
                for k, v in results.items():
                    # tuning_curves[k].append(np.mean(v, axis=0))
                    tuning_curves[k].append(v)
                for k, v in mouse_SIs.items():
                    SIs[k].append(v)
                if mouse_id not in num_neurons:
                    num_neurons[mouse_id] = len(results["recorded"])
            print("Average over mice")
            plot_tuning_width(
                tuning_curves=tuning_curves,
                SIs={k: np.concatenate(v) for k, v in SIs.items()},
                title=f"{tuning_type.capitalize()}",
                tuning_type=tuning_type,
                filename=plot_dir / f"{tuning_type}_tuning_width.{FORMAT}",
                # num_neurons=num_neurons,
                plot_legend=True,
            )

    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
