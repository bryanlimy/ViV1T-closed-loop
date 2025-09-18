from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from viv1t import data
from viv1t import metrics
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import yaml

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8


DATA_DIR = Path("../data")
PLOT_DIR = Path("figures/limit_data_and_neurons")
OUTPUT_DIR = Path("../runs/limit_data_and_neurons/")

MAX_SAMPLE = 350
DATA_SIZES = (MAX_SAMPLE * (0.01 * np.linspace(10, 100, 10))).astype(int)
NEURON_SIZES = np.array([10, 50, 100, 500, 1000, 2000, 4000, 8000], dtype=int)

FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches
DPI = 400


def load_responses(
    response_dir: Path, mouse_id: str, trial_ids: np.ndarray | None = None
) -> list[torch.Tensor]:
    # load responses and predictions
    filename = response_dir / f"mouse{mouse_id}.h5"
    responses = h5.get(filename, trial_ids=trial_ids)
    responses = [torch.from_numpy(response) for response in responses]
    return responses


def compute_best_model_performance(output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the best model prediction performance based on different subset of neurons
    """
    best_main = np.zeros((len(data.SENSORIUM_OLD), len(NEURON_SIZES)), dtype=np.float32)
    best_bonus = np.zeros_like(best_main)
    for tier_type in ["main", "bonus"]:
        for i, mouse_id in enumerate(
            tqdm(data.SENSORIUM_OLD, desc=f"compute best {tier_type}")
        ):
            tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
            trial_ids = np.where(tiers == f"live_{tier_type}")[0]
            recorded_responses = load_responses(
                response_dir=DATA_DIR / "responses",
                mouse_id=mouse_id,
                trial_ids=trial_ids,
            )
            predicted_responses = load_responses(
                response_dir=output_dir / "responses",
                mouse_id=mouse_id,
                trial_ids=trial_ids,
            )
            num_neurons = recorded_responses[0].shape[0]
            for j, neuron_size in enumerate(NEURON_SIZES):
                # select the same set of neurons that were used in limit_neurons
                rng = np.random.RandomState(1234)
                if neuron_size != NEURON_SIZES[-1]:
                    neurons = rng.choice(num_neurons, size=neuron_size, replace=False)
                    neurons = np.sort(neurons)
                else:
                    neurons = np.arange(num_neurons)  # use all neurons in the last case
                correlation = metrics.challenge_correlation(
                    y_true=[
                        recorded_responses[i][neurons]
                        for i in range(len(recorded_responses))
                    ],
                    y_pred=[
                        predicted_responses[i][neurons]
                        for i in range(len(predicted_responses))
                    ],
                )
                match tier_type:
                    case "main":
                        best_main[i, j] = correlation.numpy()
                    case "bonus":
                        best_bonus[i, j] = correlation.numpy()
                    case _:
                        raise NotImplementedError
    # average prediction performance over animals
    best_main = np.mean(best_main, axis=0)
    best_bonus = np.mean(best_bonus, axis=0)
    return best_main, best_bonus


def load_results(model_dir: Path) -> (np.ndarray, np.ndarray):
    run_id, seed = 1, 1234
    live_main = np.zeros((len(DATA_SIZES), len(NEURON_SIZES)), dtype=np.float32)
    live_bonus = np.zeros((len(DATA_SIZES), len(NEURON_SIZES)), dtype=np.float32)
    for i, limit_data in enumerate(DATA_SIZES):
        for j, limit_neurons in enumerate(NEURON_SIZES):
            output_dir = (
                model_dir
                / f"{run_id:03d}_{limit_data:03d}data_{limit_neurons:04d}neuron_{seed:04d}seed"
            )
            if output_dir.is_dir():
                evaluation = yaml.load(output_dir / "evaluation.yaml")
                live_main[i, j] = evaluation["live_main"]["correlation"]["average"]
                live_bonus[i, j] = evaluation["live_bonus"]["correlation"]["average"]
            run_id += 1
    return live_main, live_bonus


def plot_results(live_main: np.ndarray, live_bonus: np.ndarray, plot_dir: Path):
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(PAPER_WIDTH, 2.5),
        gridspec_kw={
            "wspace": 0.1,
            "left": 0.06,
            "top": 0.93,
            "right": 0.91,
            "bottom": 0.11,
        },
        dpi=DPI,
    )

    pos = axes[1].get_position()
    cbar_width, cbar_height = 0.03 * (pos.x1 - pos.x0), 0.5 * (pos.y1 - pos.y0)
    cbar_ax = figure.add_axes(
        rect=(
            1.015 * pos.x1,
            (((pos.y1 - pos.y0) / 2) + pos.y0) - (cbar_height / 2),
            cbar_width,
            cbar_height,
        )
    )

    cmap = "rocket_r"
    vmin, vmax = 0, 100

    heatmap_kw = {
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap,
        "annot": True,
        "linewidths": 1,
        "linecolor": "black",
        "fmt": ".0f",
        "annot_kws": {"fontsize": TICK_FONTSIZE},
        "rasterized": False,
        # "square": True,
    }

    sns.heatmap(live_main, ax=axes[0], cbar=False, **heatmap_kw)
    sns.heatmap(live_bonus, ax=axes[1], cbar=True, cbar_ax=cbar_ax, **heatmap_kw)

    x_ticks = np.arange(len(NEURON_SIZES)) + 0.5
    x_tick_labels = ["10", "50", "100", "500", "1k", "2k", "4k", "8k"]
    y_ticks = np.arange(len(DATA_SIZES)) + 0.5
    y_tick_labels = DATA_SIZES
    for i, ax in enumerate(axes):
        plot.set_xticks(
            axis=ax,
            ticks=x_ticks,
            tick_labels=x_tick_labels,
            label="Num. neurons per animal",
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=TICK_FONTSIZE,
            label_pad=1,
        )
        if i == 0:
            plot.set_yticks(
                axis=ax,
                ticks=y_ticks,
                tick_labels=(100 * y_tick_labels / MAX_SAMPLE).astype(int),
                label="Num. samples (% of 350)",
                tick_fontsize=TICK_FONTSIZE,
                label_fontsize=TICK_FONTSIZE,
                label_pad=1,
            )
            ax.set_title("Unseen natural movies", fontsize=LABEL_FONTSIZE, pad=3)
        else:
            ax.set_yticks([])
            ax.set_title("Unseen artificial stimuli", fontsize=LABEL_FONTSIZE, pad=3)
        # plot.set_ticks_params(axis=ax)

    cbar_ticks = np.linspace(0, 100, 5, dtype=int)
    plot.set_yticks(
        axis=cbar_ax,
        ticks=cbar_ticks,
        tick_labels=cbar_ticks,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        rotation=90,
        label="% of the full model performance",
    )

    for ax in [axes[0], axes[1], cbar_ax]:
        for spine in ax.spines:
            ax.spines[spine].set_visible(True)
        plot.set_ticks_params(axis=ax, pad=1)

    plot.save_figure(
        figure,
        filename=plot_dir / f"data_efficiency.{FORMAT}",
        layout="none",
        dpi=DPI,
    )


def main():
    best_main_results, best_bonus_results = compute_best_model_performance(
        output_dir=Path("../runs/vivit/204_causal_viv1t")
    )
    live_main_results, live_bonus_results = load_results(model_dir=OUTPUT_DIR)

    # normalize live_main and live_bonus results by results from the best model
    live_main = 100 * live_main_results / best_main_results
    live_bonus = 100 * live_bonus_results / best_bonus_results

    plot_results(
        live_main=live_main,
        live_bonus=live_bonus,
        plot_dir=PLOT_DIR,
    )
    print(f"saved plots to {PLOT_DIR}.")


if __name__ == "__main__":
    main()
