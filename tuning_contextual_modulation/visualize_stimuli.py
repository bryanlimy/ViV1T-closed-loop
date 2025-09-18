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

DPI = 240

BLANK_SIZE, PATTERN_SIZE = 10, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

SENSORIUM_DIR = Path("../data/sensorium")

DS_NAME = "dynamic_low_contrast"

DATA_DIR = Path("../data/contextual_modulation") / DS_NAME

OUTPUT_DIR = Path("../runs/vivit/172_viv1t_causal")
# OUTPUT_DIR = Path("../runs/fCNN/029_fCNN_noClipGrad")

RESPONSE_DIR = OUTPUT_DIR / "contextual_modulation" / DS_NAME / "responses"

PLOT_DIR = Path("figures") / DS_NAME


def get_stim_name(stim_type: int):
    match stim_type:
        case 0:
            return "Centre"
        case 1:
            return "Iso"
        case 2:
            return "Cross"
        case 3:
            return "Shift"
        case _:
            raise ValueError(f"Unknown stim_type: {stim_type}")


def plot_stimulus(
    videos: np.ndarray,
    params: np.ndarray,
    stim_type: int,
    plot_dir: Path,
):
    for coverage in [30]:
        block_ids = np.where((params[:, 0] == coverage) & (params[:, -1] == stim_type))
        block_id = block_ids[0][0]
        figure, ax = plt.subplots(nrows=1, ncols=1, dpi=DPI)

        center_x = videos.shape[3] // 2
        center_y = videos.shape[2] // 2
        zoom_size = 22  # Size of the zoom-in region

        # Calculate the bounds of the zoom-in region
        x_min = max(center_x - zoom_size // 2, 0)
        x_max = min(center_x + zoom_size // 2, videos.shape[3])
        y_min = max(center_y - zoom_size // 2, 0)
        y_max = min(center_y + zoom_size // 2, videos.shape[2])
        ax.imshow(
            videos[block_id, 10][y_min : y_max + 1, x_min : x_max + 1],
            cmap="gray",
            aspect="equal",
            vmin=0,
            vmax=255,
        )
        ax.axis("off")
        # for spine in ax.spines.values():
        #    spine.set_edgecolor("black")
        #    spine.set_linewidth(6)
        plt.savefig(
            plot_dir / f"{get_stim_name(stim_type)}_{coverage}°.svg",
            dpi=DPI,
            bbox_inches="tight",
        )


def main():
    mouse_id = "B"
    videos, params = load_data(mouse_id=mouse_id)
    stim_types = sorted(np.unique(params[:, -1]))
    (PLOT_DIR / "stimulus").mkdir(parents=True, exist_ok=True)
    for stim_type in stim_types:
        plot_stimulus(
            videos=videos,
            params=params,
            stim_type=stim_type,
            plot_dir=PLOT_DIR / "stimulus",
        )


if __name__ == "__main__":
    main()
