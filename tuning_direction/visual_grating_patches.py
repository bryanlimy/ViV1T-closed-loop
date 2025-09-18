from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange

from viv1t import data
from viv1t.utils import plot

plot.set_font()
DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures") / "experimental_setup"
TICK_FONTSIZE = 10
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
DPI = 240

BLOCK_SIZE = 25  # number of frames for each drifting Gabor filter


def load_data(mouse_id: str = "B") -> (np.ndarray, np.ndarray):
    # get drifting Gabor trial IDs
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 4)[0]

    # get direction parameters for each frame
    gabor_parameters = np.array(
        [
            data.get_gabor_parameters(mouse_id, trial_id=trial_id)
            for trial_id in trial_ids
        ],
        dtype=np.float32,
    )
    directions = gabor_parameters[:, :, 0]
    # get direction for each block
    directions = rearrange(
        directions, "trial (block frame) -> trial block frame", frame=BLOCK_SIZE
    )
    directions = directions[:, :, 0].astype(int)

    video_dir = DATA_DIR / data.MOUSE_IDS[mouse_id] / "data" / "videos"
    videos = np.stack(
        [np.load(video_dir / f"{trial_id}.npy") for trial_id in trial_ids],
        axis=0,
    )
    videos = videos[:, :, :, : data.MAX_FRAME]
    videos = rearrange(
        videos,
        "trial H W (block frame) -> trial block frame H W",
        frame=BLOCK_SIZE,
    )

    # combine block and trial dimensions
    directions = rearrange(directions, "trial block -> (trial block)")
    videos = rearrange(videos, "trial block frame H W -> (trial block) frame H W")

    videos = videos / 255.0

    return directions, videos


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty : starty + cropy, startx : startx + cropx]


def plot_image(image: np.ndarray, filename: Path, square: bool):
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(9, 9) if square else (16, 9),
        dpi=DPI,
    )
    if square:
        image = crop_center(img=image, cropx=32, cropy=32)
    ax.imshow(image, cmap="gray", aspect="equal", vmin=0.3, vmax=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(40 if square else 15)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def main():
    directions, videos = load_data()

    index = np.where(directions == 0)[0][0]
    plot_image(
        videos[index][0],
        filename=PLOT_DIR / f"drifting_gabor_main°.png",
        square=False,
    )

    unique_direction = np.unique(directions)
    for i, direction in enumerate(unique_direction):
        index = np.where(directions == direction)[0][2]
        plot_image(
            videos[index][0],
            filename=PLOT_DIR / f"drifting_gabor_{direction}°.png",
            square=True,
        )


if __name__ == "__main__":
    main()
