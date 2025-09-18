from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches
FPS = 30

plot.set_font()


PLOT_DIR = Path("figures") / "calcium_trace"

DATA_DIR = Path("../data")
RECORDED_DIR = DATA_DIR / "responses"
OUTPUT_DIR = Path("../runs/")


def load_data(data_dir: Path, mouse_id: str, trial_ids: np.ndarray) -> np.ndarray:
    filename = data_dir / f"mouse{mouse_id}.h5"
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    return responses


def plot_traces(
    recorded_response: np.ndarray,
    predicted_response: np.ndarray,
    neuron: int,
    filename: Path,
):
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 0.3),
        gridspec_kw={
            "wspace": 0.2,
            "hspace": 0.1,
            "top": 0.9,
            "bottom": 0.1,
            "left": 0.2,
            "right": 0.98,
        },
        dpi=DPI,
    )

    recorded_response /= np.max(recorded_response)
    predicted_response /= np.max(predicted_response)

    x_ticks = np.arange(len(recorded_response))

    ax.plot(
        x_ticks,
        recorded_response,
        color="black",
        linewidth=1.2,
        alpha=0.8,
        clip_on=False,
        zorder=1,
    )
    ax.plot(
        x_ticks,
        predicted_response,
        color="limegreen",
        linewidth=1.2,
        alpha=0.8,
        clip_on=False,
        zorder=1,
    )
    min_value = np.min([recorded_response, predicted_response])
    max_value = np.max([recorded_response, predicted_response])
    # max_value = max(max_value, 600)

    ax.set_xlim(x_ticks[0] - 1, x_ticks[-1] + 1)
    ax.set_ylim(min_value, max_value)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    x_offset = -45
    # response scale bar
    y = max_value
    ax.plot(
        [x_offset, x_offset],
        [min_value, y],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    ax.text(
        x=x_offset - 1,
        y=min_value,
        s=r"$\Delta$F/F",
        fontsize=TICK_FONTSIZE - 1,
        rotation=90,
        va="bottom",
        ha="right",
    )

    # timescale bar
    ax.plot(
        [x_offset, x_offset + FPS],
        [min_value, min_value],
        color="black",
        linewidth=1,
        clip_on=False,
        solid_capstyle="projecting",
    )
    ax.text(
        x=x_offset + FPS / 2,
        y=min_value,
        s="1s",
        fontsize=TICK_FONTSIZE - 1,
        va="bottom",
        ha="center",
    )

    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def main():
    mouse_id = "A"
    output_dir = OUTPUT_DIR / "vivit" / "204_causal_viv1t"
    predicted_dir = output_dir / "responses"

    tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
    trial_ids = np.where((tiers == "final_main") | (tiers == "live_main"))[0]

    recorded_responses = load_data(
        data_dir=RECORDED_DIR,
        mouse_id=mouse_id,
        trial_ids=trial_ids,
    )
    predicted_responses = load_data(
        data_dir=predicted_dir,
        mouse_id=mouse_id,
        trial_ids=trial_ids,
    )

    neurons = sorted(
        utils.get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id, size=50)
    )

    for i, trial_id in enumerate(tqdm(trial_ids)):
        for neuron in neurons:
            recorded_response = recorded_responses[i, neuron]
            predicted_response = predicted_responses[i, neuron]
            corr = np.corrcoef(recorded_response, predicted_response)[0, 1]
            if corr < 0.8:
                continue
            plot_traces(
                recorded_response=recorded_response,
                predicted_response=predicted_response,
                neuron=neuron,
                filename=PLOT_DIR
                / f"mouse{mouse_id}"
                / f"trial{trial_id:03d}"
                / f"trial{trial_id:03d}_neuron{neuron:04d}.{FORMAT}",
            )


if __name__ == "__main__":
    main()
