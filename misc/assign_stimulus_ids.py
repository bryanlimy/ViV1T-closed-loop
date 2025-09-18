"""
Assign a stimulus type to each stimulus
```
STIMULUS_TYPES = {
    0: "movie",
    1: "directional pink noise",
    2: "gaussian dots",
    3: "random dot kinematogram",
    4: "drifting gabor",
    5: "image",
}
```
Main function work flow
- given mouse_id
- load `video_ids` and `tiers` for the mouse
- create a numpy array `stimulus_ids` that has the same shape as `video_ids`
- automatically assign all train, validation (oracle), live main and bonus main trials to `movie`
- for each unique stimulus (given by `video_ids`)
  - plot and present the first 10 frames of the stimulus
  - ask user to input the stimulus type (according to `STIMULUS_TYPES` above)
  - assign the `stimulus_id` to all trials with the same video ID
- return `stimulus_ids`
"""

from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from viv1t import data
from viv1t.data import get_tier_ids

DATA_DIR = Path("../data/sensorium")


def plot_trial(mouse_dir: Path, trial_id: str | int):
    video = data.load_trial(mouse_dir=mouse_dir, trial_id=trial_id)["video"]
    figure, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(5, 1.5),
        gridspec_kw={"wspace": 0.1, "hspace": 0},
        dpi=120,
    )
    for i, frame in enumerate(range(0, 40, 10)):
        axes[i].imshow(video[0, frame, :, :], cmap="gray")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlabel(f"frame {frame}", labelpad=2, fontsize=8)
    axes[2].set_title(f"Trial ID {trial_id}", pad=1, fontsize=8)
    plt.show()
    return figure


def annotate_mouse(mouse_id: str) -> np.ndarray:
    mouse_dir = DATA_DIR / data.MOUSE_IDS[mouse_id]
    # load video IDs
    video_ids = data.get_video_ids(mouse_id)
    # load tiers
    tiers = get_tier_ids(DATA_DIR, mouse_id=mouse_id)
    # create stimulus_ids array
    stimulus_ids = np.full_like(video_ids, fill_value=-1)
    # annotate train and validation trials as movies
    for tier in ("train", "validation", "live_main", "final_main"):
        trial_ids = np.where(tiers == tier)[0]
        stimulus_ids[trial_ids] = 0
    for tier in ("live_bonus", "final_bonus"):
        trial_ids = np.where(tiers == tier)[0]
        unique_videos = np.unique(video_ids[trial_ids])
        for unique_video in unique_videos:
            # get the first trial with this video ID
            repeated_trials = np.where(video_ids == unique_video)[0]
            figure = plot_trial(mouse_dir=mouse_dir, trial_id=repeated_trials[0])
            sleep(1)
            while True:
                video_id = int(
                    input(
                        f"stimulus type for video ID {unique_video} (trial {repeated_trials[0]}): "
                    )
                )
                if video_id in data.STIMULUS_TYPES.keys():
                    break
                print(f"stimulus type must be one of {data.STIMULUS_TYPES.keys()}")

            stimulus_ids[repeated_trials] = int(video_id)
            plt.close(figure)
    return stimulus_ids


def annotate():
    for mouse_id in data.SENSORIUM:
        stimulus_ids = annotate_mouse(mouse_id)
        filename = data.METADATA_DIR / "stimulus_ids" / f"mouse{mouse_id}.npy"
        filename.parent.mkdir(parents=True, exist_ok=True)
        np.save(filename, stimulus_ids, allow_pickle=False)
        print(f"Saved mouse {mouse_id} stimulus IDs to {filename}.")


if __name__ == "__main__":
    annotate()
