import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from scipy.stats import norm

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import plot

DATA_DIR = Path("../data/sensorium")

MAX_FRAME = 300

plot.set_font()

TICK_FONTSIZE = 10
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
DPI = 240

THRESHOLD = 0.01  # p-value threshold


def load_response(filename: Path, trial_ids: np.ndarray) -> np.ndarray:
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    return responses


def load_gaussian_dot_parameters(mouse_id: str, trial_ids: np.ndarray) -> np.ndarray:
    dot_parameters = np.stack(
        [
            data.get_gaussian_dot_parameters(mouse_id, trial_id=trial_id)
            for trial_id in trial_ids
        ]
    )
    dot_parameters = dot_parameters[:, :MAX_FRAME, :]
    return dot_parameters


def plot_receptive_fields(tuning_curves, selective, preferred, mean_responses, fig_dir):
    for neuron in range(tuning_curves.shape[1]):
        if selective[neuron]:
            heatmap = pd.DataFrame(
                {
                    "x": [x for x, _ in mean_responses.keys()],
                    "y": [y for _, y in mean_responses.keys()],
                    "value": tuning_curves[:, neuron],
                }
            ).pivot(index="y", columns="x", values="value")
            plt.figure(figsize=(10, 8))
            sns.heatmap(heatmap, cmap="viridis", cbar=True)
            plt.title(
                f"neuron {neuron}, selective: {selective[neuron]}, preferred: {preferred[neuron]}"
            )
            plt.savefig(fig_dir / f"neuron{neuron}.png", dpi=300)
            plt.close()


def estimate_mouse(label: str, mouse_id: str, output_dir: Path, fig_dir: Path):

    # find trials with Gaussian dot stimuli
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 2)[0]

    # load responses
    responses = load_response(
        filename=output_dir / f"mouse{mouse_id}.h5",
        trial_ids=trial_ids,
    )

    # load Gaussian dot parameters [x, y, radius, dot_is_black]
    dot_parameters = load_gaussian_dot_parameters(
        mouse_id=mouse_id, trial_ids=trial_ids
    )
    dot_parameters = dot_parameters[:, -responses.shape[2] :, :]
    dot_parameters = rearrange(dot_parameters, "t f p -> (t f) p")
    responses = rearrange(responses, "t n f -> (t f) n")

    # compute mean responses per position, rounding to nearest 10
    bin = 10
    dot_parameters[:, 0] = np.floor(dot_parameters[:, 0] / bin) * bin
    dot_parameters[:, 1] = np.floor(dot_parameters[:, 1] / bin) * bin
    positions = np.unique(dot_parameters[:, :2], axis=0)
    tuning_curves = np.full((positions.shape[0], responses.shape[1]), 0.0)
    for i, pos in enumerate(positions):
        selection = np.where(
            np.all(dot_parameters[:, :2] == pos, axis=1) & (dot_parameters[:, 3] == 0)
        )
        tuning_curves[i, :] = np.mean(responses[selection], axis=0)

    # z-score
    tuning_curves = (tuning_curves - np.nanmean(tuning_curves, axis=0)) / np.nanstd(
        tuning_curves, axis=0
    )
    tuning_curves = np.nan_to_num(tuning_curves)

    # threshold
    selective = np.any(tuning_curves > norm.ppf(1 - THRESHOLD), axis=0)

    # extract preferred
    preferred = np.take(positions, np.argmax(tuning_curves, axis=0), axis=0)
    preferred[~selective] = [np.nan, np.nan]

    # plot_receptive_fields(
    #    tuning_curves=tuning_curves,
    #    selective=selective,
    #    preferred=preferred,
    #    mean_responses=mean_responses,
    #    fig_dir=fig_dir,
    # )
    # population_tuning_curve = np.mean(tuning_curves, axis=1)
    # population_heatmap = pd.DataFrame(
    #    {
    #        "x": [x for x, _ in mean_responses.keys()],
    #        "y": [y for _, y in mean_responses.keys()],
    #        "value": population_tuning_curve,
    #    }
    # ).pivot(index="y", columns="x", values="value")
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(population_heatmap, cmap="viridis", cbar=True)
    # plt.savefig(fig_dir / "population.png", dpi=300)
    # plt.close()

    # normalised heatmap for presentation
    percentage_presented = {
        tuple(pos): np.sum(np.all(dot_parameters[:, :2] == pos, axis=1))
        / dot_parameters.shape[0]
        * 100
        for pos in positions
    }
    heatmap_data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "value": percentage_presented.values(),
        }
    ).pivot(index="y", columns="x", values="value")
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    ax.invert_yaxis()
    plt.title(f"percentage presented ({label})")
    plt.savefig(fig_dir / "presentation_map.png", dpi=300)

    # normalised heatmap for selective neurons
    percentage_selective = {
        tuple(pos): np.sum(np.all(preferred[selective] == pos, axis=1))
        / np.sum(selective)
        * 100
        for pos in positions
    }
    heatmap_data = pd.DataFrame(
        {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "value": percentage_selective.values(),
        }
    ).pivot(index="y", columns="x", values="value")

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    ax.invert_yaxis()
    plt.title(f"percentage selective ({label})")
    plt.savefig(fig_dir / "selectivitiy_map.png", dpi=300)


def main():
    models = {
        "DATA": DATA_DIR / "responses",
        "ViV1T": Path("../runs/vivit/140_vivit_AdamWScheduleFree/predict"),
    }

    for model, output_dir in models.items():
        for mouse_id in ["A", "D", "E"]:
            fig_dir = Path(f"figures/receptive_field/{model}/{mouse_id}")
            os.makedirs(fig_dir, exist_ok=True)
            estimate_mouse(label=model, mouse_id=mouse_id, output_dir=output_dir, fig_dir=fig_dir)


if __name__ == "__main__":
    main()
