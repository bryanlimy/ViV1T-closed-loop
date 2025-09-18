from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

from viv1t.utils import h5
from viv1t.utils import plot

plot.set_font()

TICK_FONTSIZE = 10
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 12

DPI = 90

BLANK_SIZE, PATTERN_SIZE = 10, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

SENSORIUM_DIR = Path("../data/sensorium")

DATA_DIR = Path("../data/feedbackRF")

OUTPUT_DIR = Path("../runs/vivit/159_viv1t_elu")
# OUTPUT_DIR = Path("../runs/fCNN/029_fCNN_noClipGrad")

RESPONSE_DIR = OUTPUT_DIR / "feedbackRF" / "responses"

PLOT_DIR = Path("figures") / "stimulus"


def load_data(mouse_id: str):
    # load responses in format (trial, neuron, frame)
    filename = RESPONSE_DIR / f"mouse{mouse_id}.h5"
    trial_ids = h5.get_trial_ids(filename)
    params = np.load(DATA_DIR / "meta" / "trials" / "params.npy")
    params = params[trial_ids]  # rearrange params order to match trial_ids

    num_frames = h5.get_shape(filename, trial_id=trial_ids[0])[-1]
    params = params[:, -num_frames:, :]

    params = rearrange(
        params,
        "trial (block frame) param -> (trial block) param frame",
        frame=BLOCK_SIZE,
    )

    videos = np.stack(
        [
            np.load(DATA_DIR / "data" / "videos" / f"{trial_id}.npy")
            for trial_id in trial_ids
        ],
        axis=0,
    )
    videos = videos[:, :, :, -num_frames:]
    videos = rearrange(
        videos,
        "trial height width (block frame) -> (trial block) frame height width",
        frame=BLOCK_SIZE,
    )

    # remove blank frames
    params = params[:, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    videos = videos[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE, ...]

    params = params[:, :, -1]

    print(f"videos shape: {videos.shape}\n" f"params shape: {params.shape}")

    coverages = np.unique(params[:, 0])
    print(f"visual coverages: {coverages}")

    return videos, params


def get_stim_name(stim_type: int):
    match stim_type:
        case 0:
            return "classical"
        case 1:
            return "inverse"
        case _:
            raise ValueError(f"Invalid stimulus type: {stim_type}")


def plot_stimulus(
    videos: np.ndarray,
    params: np.ndarray,
    stim_type: int,
    plot_dir: Path,
):
    for coverage in [0, 40, 80]:
        block_ids = np.where((params[:, 0] == coverage) & (params[:, -1] == stim_type))
        block_id = block_ids[0][0]
        figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 2), dpi=DPI)
        ax.imshow(
            videos[block_id, 0],
            cmap="gray",
            aspect="equal",
            vmin=0,
            vmax=255,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black" if stim_type == 0 else "red")
            spine.set_linewidth(8)
        plot.save_figure(
            figure,
            filename=plot_dir / f"{get_stim_name(stim_type)}_{coverage}°.svg",
            dpi=DPI,
        )


def main():
    mouse_id = "B"
    videos, params = load_data(mouse_id=mouse_id)
    stim_types = sorted(np.unique(params[:, -1]))
    for stim_type in stim_types:
        plot_stimulus(
            videos=videos,
            params=params,
            stim_type=stim_type,
            plot_dir=PLOT_DIR,
        )
    print(f'Plots saved to "{PLOT_DIR}"')


if __name__ == "__main__":
    main()
