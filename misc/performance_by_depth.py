from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from einops import rearrange

from viv1t import data
from viv1t import metrics
from viv1t.data import get_neuron_coordinates
from viv1t.utils import plot
from viv1t.utils import utils

utils.set_random_seed(1234)

plot.set_font()

TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10
DPI = 512

DATA_DIR = Path("../data/sensorium")
VIVIT_DIR = Path("../runs/vivit/047_vivit_RoPE_regTokens4_cropFrame300/predict")
FCNN_DIR = Path("../runs/fCNN/009_fCNN/predict")


def get_name(tier: str):
    match tier:
        case "train":
            return "Train"
        case "validation":
            return "Validation"
        case "live_main":
            return "Live main"
        case "live_bonus":
            return "Live bonus"
        case "final_main":
            return "Final main"
        case "final_bonus":
            return "Final bonus"
        case _:
            raise ValueError(f"Invalid tier: {tier}")


def load_responses(mouse_id: str, predict_dir: Path, tier: str):
    # get trial IDs
    mouse_dir = DATA_DIR / data.MOUSE_IDS[mouse_id]
    tiers = np.load(mouse_dir / "meta" / "trials" / "tiers.npy")
    trial_ids = np.where(tiers == tier)[0]
    # load responses and predictions
    true_dir = predict_dir / f"mouse{mouse_id}" / "y_true"
    pred_dir = predict_dir / f"mouse{mouse_id}" / "y_pred"
    result = {"y_true": [], "y_pred": []}
    for trial_id in trial_ids:
        result["y_true"].append(np.load(true_dir / f"{trial_id}.npy"))
        result["y_pred"].append(np.load(pred_dir / f"{trial_id}.npy"))
    result = {k: np.stack(v) for k, v in result.items()}
    result["neuron_coordinates"] = get_neuron_coordinates(DATA_DIR, mouse_id=mouse_id)
    return result


def compute_correlation(mouse_id: str, predict_dir: Path, tier: str):
    responses = load_responses(mouse_id, predict_dir=predict_dir, tier=tier)
    # compute single trial correlation used in challenge
    y_true = rearrange(torch.from_numpy(responses["y_true"]), "b n t -> (b t) n")
    y_pred = rearrange(torch.from_numpy(responses["y_pred"]), "b n t -> (b t) n")
    correlation = metrics.correlation(y1=y_true, y2=y_pred, dim=0).numpy()
    z_corr = {}
    for depth in np.unique(responses["neuron_coordinates"][:, 2]):
        neurons = np.where(responses["neuron_coordinates"][:, 2] == depth)[0]
        z_corr[depth] = correlation[neurons]
    return z_corr


def plot_correlation_by_depth(
    fcnn_result: Dict[int, np.ndarray],
    vivit_result: Dict[int, np.ndarray],
    tier: str,
    mouse_id: str,
    filename: Path,
):
    assert fcnn_result.keys() == vivit_result.keys()

    depths = np.array(list(fcnn_result.keys()), dtype=np.int32)
    fcnn_means = np.array([np.mean(fcnn_result[depth]) for depth in depths])
    fcnn_stds = np.array([np.std(fcnn_result[depth]) for depth in depths])
    fcnn_mean = np.mean([np.concatenate(list(fcnn_result.values()))])

    vivit_means = np.array([np.mean(vivit_result[depth]) for depth in depths])
    vivit_stds = np.array([np.std(vivit_result[depth]) for depth in depths])
    vivit_mean = np.mean([np.concatenate(list(vivit_result.values()))])

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.5), dpi=DPI)

    y_max, bar_pad, edgewidth = 0.5, 0.005, 0.0
    alpha = 1.0
    bar_width = 4
    error_kw = {"elinewidth": 1, "capsize": 1.2}
    hline_kw = {"alpha": 0.4, "linestyle": "--", "linewidth": 1, "zorder": -1}

    x_ticks = np.linspace(0, 120, 10)
    ax.set_xlim(x_ticks[0] - 10, x_ticks[-1] + 10)
    ax.bar(
        x=x_ticks - bar_width / 2,
        height=fcnn_means,
        width=bar_width,
        label="fCNN",
        color="dodgerblue",
        edgecolor="dodgerblue",
        alpha=alpha,
        linewidth=edgewidth,
        error_kw=error_kw,
    )
    ax.axhline(y=fcnn_mean, color="dodgerblue", **hline_kw)
    ax.bar(
        x=x_ticks + bar_width / 2,
        height=vivit_means,
        width=bar_width,
        label="ViViT",
        color="orangered",
        edgecolor="orangered",
        alpha=alpha,
        linewidth=edgewidth,
        error_kw=error_kw,
    )
    ax.axhline(y=vivit_mean, color="orangered", **hline_kw)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=2,
        frameon=False,
        handletextpad=0.35,
        handlelength=0.6,
        markerscale=0.8,
        columnspacing=2,
        fontsize=LABEL_FONTSIZE,
    )
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=depths,
        label="Depth (z-axis in μm)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )
    ax.set_ylim(0, y_max)
    y_ticks = np.linspace(0, y_max, 4)
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks.round(1),
        label="Single trial correlation",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
    )

    # text_heights = []
    for i in range(len(vivit_means)):
        x_start = x_ticks[i]
        lx = x_start - bar_width / 2
        rx = x_start + bar_width / 2
        y = max(fcnn_means[i], vivit_means[i]) + 0.02
        barx = [lx, lx, rx, rx]
        y_offset = 0.005
        bary = [y, y + y_offset, y + y_offset, y]
        ax.plot(barx, bary, color="black", linewidth=1)
        ax.text(
            x=(lx + rx) / 2,
            y=y + 0.005,
            s=f"+{100*(vivit_means[i] - fcnn_means[i])/fcnn_means[i]:.0f}%",
            ha="center",
            va="bottom",
            fontsize=TICK_FONTSIZE - 1,
        )
        # text_heights.append(text_height)

    ax.set_title(f"Mouse {mouse_id} {get_name(tier)}", fontsize=TITLE_FONTSIZE)

    sns.despine(ax=ax, top=True, right=True, trim=False)
    plot.set_ticks_params(axis=ax, length=2)

    plot.save_figure(figure, filename=filename, dpi=DPI)

    print(f"Saved {filename}")


def main():
    plot_dir = Path("figures/correlation_by_depth")
    tiers = ["live_main", "live_bonus"]
    for mouse_id in data.SENSORIUM_OLD:
        for tier in tiers:
            vivit_result = compute_correlation(
                mouse_id=mouse_id, predict_dir=VIVIT_DIR, tier=tier
            )
            fcnn_result = compute_correlation(
                mouse_id=mouse_id, predict_dir=FCNN_DIR, tier=tier
            )
            plot_correlation_by_depth(
                fcnn_result=fcnn_result,
                vivit_result=vivit_result,
                mouse_id=mouse_id,
                tier=tier,
                filename=plot_dir / f"mouse{mouse_id}_{data.TIERS[tier]}.jpg",
            )


if __name__ == "__main__":
    main()
