from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from scipy.stats import sem
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()

TICK_FONTSIZE = 9
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
DPI = 240

BLOCK_SIZE = 25

H_INTERVAL = 25  # horizontal distance interval
SI_THRESHOLD = 0.2

DATA_DIR = Path("../dataa")
PLOT_DIR = Path("figures/tuning_similarity_by_depth")


def compute_distance(a: np.ndarray, b: np.ndarray) -> (float, float):
    """
    compute horizontal (Euclidean) and vertical (absolute) distance between
    two 3D points
    """
    assert a.shape == b.shape == (3,)
    h = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    v = abs(a[2] - b[2])
    return h, v


def random_tuning_similarity(mouse_ids: List[str], num_pairs: int = 1000):
    """
    Randomly select pairs of selective tuning curves from the recorded data
    and compute their average tuning similarity.
    """
    results = []
    rng = np.random.default_rng(1234)
    for mouse_id in mouse_ids:
        tuning = utils.load_tuning(data.METADATA_DIR, mouse_id=mouse_id)
        selective_neurons = np.where(tuning["DSI"] >= 0)[0]
        tuning_curves = tuning["tuning_curve"][selective_neurons]
        neurons = np.arange(len(tuning_curves))
        group1 = sorted(rng.choice(neurons, num_pairs, replace=False))
        neurons = np.setdiff1d(neurons, group1, assume_unique=True)
        group2 = sorted(rng.choice(neurons, num_pairs, replace=False))
        corr = np.corrcoef(tuning_curves[group1], tuning_curves[group2])
        results.append(np.mean(corr[0, 1:]))
    return np.array(results, dtype=np.float32)


def plot_tuning_similarity(
    df: pd.DataFrame,
    title: str,
    filename: Path,
    label: str = "Tuning similarity",
    random_similarity: float = None,
):
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2.5), dpi=DPI)
    horizontal_distances = np.array(df.columns, dtype=int)
    vertical_distances = np.array(df.index, dtype=int)
    colors = sns.color_palette("bright", n_colors=len(vertical_distances))
    alpha, linewidth = 0.7, 1.5
    for i, depth in enumerate(vertical_distances):
        color = colors[i]
        values = df.loc[depth].map(np.mean)
        error_bars = df.loc[depth].map(sem)
        zorder = len(vertical_distances) - i
        ax.plot(
            horizontal_distances,
            values,
            linestyle="-",
            linewidth=linewidth,
            color=color,
            alpha=0.4,
            markeredgecolor=None,
            clip_on=False,
            zorder=zorder,
            label=f"{depth}µm",
        )
        ax.errorbar(
            x=horizontal_distances,
            y=values,
            yerr=error_bars,
            fmt=".",
            linestyle="",
            markersize=4,
            capsize=2,
            color=color,
            alpha=alpha,
            clip_on=False,
            zorder=zorder,
        )

    # plot average over delta_v
    ax.plot(
        horizontal_distances,
        df.map(np.mean).mean(axis=0),
        color="black",
        linewidth=linewidth,
        alpha=alpha,
        zorder=-1,
        label="average",
    )

    if random_similarity is not None:
        ax.axhline(
            y=random_similarity,
            color="grey",
            linestyle="dotted",
            alpha=0.8,
            zorder=-1,
            linewidth=linewidth,
            label="random",
        )

    y_ticks = np.linspace(0, 0.6, 7)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 1),
        label=label,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    for y in y_ticks[1:-1]:
        ax.axhline(y=y, color="gray", linewidth=1, alpha=0.3, zorder=-2)
    x_ticks = np.array(horizontal_distances)
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_ticks.astype(int),
        label="Cortical distance (d) (µm)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    ax.set_xlim(x_ticks[0] - 10, x_ticks[-1])
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.3, 1.05),
        ncols=2,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="Depth (µm)",
        alignment="left",
        handletextpad=0.35,
        handlelength=0.6,
        markerscale=1,
        labelspacing=0.2,
        columnspacing=0.5,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    sns.despine(ax=ax)
    plot.set_ticks_params(ax, length=2)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def estimate_tuning_similarity(
    save_dir: Path, mouse_id: str
) -> tuple[pd.DataFrame, float]:
    tuning = utils.load_tuning(save_dir, mouse_id=mouse_id)
    tuning_curves = tuning["tuning_curve"]
    neuron_coordinates = data.get_neuron_coordinates(mouse_id=mouse_id)

    # filter out non-selective neurons based on real recordings
    SI = utils.load_tuning(data.METADATA_DIR, mouse_id=mouse_id)["DSI"]
    selective_neurons = np.where(SI >= SI_THRESHOLD)[0]

    depths = np.unique(neuron_coordinates[:, 2])
    print(f"consider depths: {depths}")
    neurons = np.where(np.isin(neuron_coordinates[:, 2], depths))[0]

    selective_neurons = np.intersect1d(selective_neurons, neurons)

    # compute correlation coefficient of tuning curves for all neuron pairs
    tuning_similarities = np.corrcoef(tuning_curves)

    # get all combinations of selective neuron pairs
    neuron_pairs = np.array(list(combinations(np.sort(selective_neurons), r=2)))

    results = defaultdict(lambda: defaultdict(list))
    for n1, n2 in tqdm(neuron_pairs, desc=f"mouse {mouse_id}"):
        delta_h, delta_v = compute_distance(
            neuron_coordinates[n1], neuron_coordinates[n2]
        )
        if delta_h >= 7 * H_INTERVAL:
            continue
        delta_h = H_INTERVAL * (round(delta_h / H_INTERVAL) + 1)
        depth = min(neuron_coordinates[n1, 2], neuron_coordinates[n2, 2])
        if delta_v > 25 or depth % 10 != 0:
            continue
        results[depth][delta_h].append(tuning_similarities[n1, n2])

    df = pd.DataFrame(results).sort_index(axis=0).sort_index(axis=1).T

    print("number of pairs in each vertical and horizontal distance group:")
    print(df.map(len))

    # randomly sample 500 pairs of neurons and compute their average tuning similarity
    rng = np.random.default_rng(1234)
    random_pairs = rng.choice(len(neuron_pairs), size=5000, replace=False)
    random_pairs = neuron_pairs[random_pairs]
    random_similarity = tuning_similarities[random_pairs[:, 0], random_pairs[:, 1]]
    random_similarity = np.mean(random_similarity)
    print(f"random similarity: {random_similarity:.3f}")

    return df, random_similarity


def main():
    models = {
        "recorded": data.METADATA_DIR,
        # "LN": Path("../runs/fCNN/015_linear_fCNN"),
        # "fCNN": Path("../runs/fCNN/029_fCNN_noClipGrad"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        # "ViViT": Path("../runs/vivit/162_vivit"),
        # "ViV1T": Path("../runs/vivit/159_viv1t_elu"),
    }
    mouse_ids = ["B", "C", "E"]

    for model, output_dir in models.items():
        assert output_dir.exists()
        print(f"\n\nPlot {model} responses...")
        for mouse_id in mouse_ids:
            print(f"\nProcessing mouse {mouse_id}...")
            df, random = estimate_tuning_similarity(output_dir, mouse_id=mouse_id)
            plot_tuning_similarity(
                df,
                title=f"Mouse {mouse_id} {model}",
                filename=PLOT_DIR / model / f"mouse{mouse_id}.jpg",
                label="Tuning similarity",
                random_similarity=random,
            )


if __name__ == "__main__":
    main()
