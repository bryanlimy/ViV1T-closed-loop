"""
Find the most-exciting/preferred directional grating in the recorded population response
"""

from pathlib import Path

import numpy as np
import pandas as pd
from einops import rearrange
from einops import repeat
from numpy.lib.stride_tricks import sliding_window_view

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import utils

DATA_DIR = Path("../../data")

PATTERN_SIZE = 15  # slide window size to match flashing image presentation window
BLANK_SIZE = 10  # number of frames before and after each image presentation to extract


def load_max_response(filename: Path, mouse_id: str) -> np.ndarray:
    max_responses = np.load(filename, allow_pickle=True)
    return max_responses[mouse_id]


def find_preferred_video(
    output_dir: Path,
    mouse_id: str,
    trial_ids: np.ndarray,
    video_ids: np.ndarray,
    model_name: str,
    neurons: np.ndarray | None = None,
) -> pd.DataFrame:
    # responses shape (trial, neuron, time)
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]

    # group video by video IDs
    video_ids_ = np.unique(video_ids)
    responses_, min_repeat = [], np.inf
    for video_id in video_ids_:
        responses_.append(responses[video_ids == video_id])
        min_repeat = min(min_repeat, np.count_nonzero(video_ids == video_id))

    # randomly select repeat so that all unique video IDs have the same number
    # of repeats
    rng = np.random.default_rng(1234)
    responses = np.stack(
        [r[rng.choice(len(r), size=min_repeat, replace=False)] for r in responses_]
    )
    responses = rearrange(
        responses, "video repeat neuron frame -> video repeat frame neuron"
    )
    video_ids = video_ids_.copy()
    del responses_, video_ids_

    if neurons is not None:
        responses = responses[:, :, :, neurons]
    # average response over repeats
    responses = np.mean(responses, axis=1)
    # mean response over population
    responses = np.mean(responses, axis=-1)

    frame_ids = np.arange(num_frames) + (data.MAX_FRAME - num_frames)
    frame_ids = repeat(frame_ids, "frame -> video frame", video=len(video_ids))

    # use a sliding window to segment videos of size (BLANK_SIZE + WINDOW_SIZE + BLANK_SIZE)
    window_size = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
    responses = sliding_window_view(responses, window_shape=window_size, axis=1)
    frame_ids = sliding_window_view(frame_ids, window_shape=window_size, axis=1)

    video_ids = repeat(video_ids, "video -> (video block)", block=responses.shape[1])
    responses = rearrange(responses, "video block frame -> (video block) frame")
    frame_ids = rearrange(frame_ids[:, :, BLANK_SIZE], "video block -> (video block)")

    # select the PATTERN_SIZE presentation window
    response = responses[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    # compute response sum over presentation window
    response = np.sum(response, axis=1)
    df = pd.DataFrame(
        {
            "mouse": mouse_id,
            "model": model_name,
            "video_id": video_ids,
            "frame_id": frame_ids,
            "response": response,
            "raw_response": responses.tolist(),
        }
    )
    print(
        f"Top 10 most-exciting video clips:\n"
        f"{df.sort_values(by='response', ascending=False).loc[:, ['model', 'video_id', 'frame_id', 'response']].head(10)}"
    )
    return df


def main():
    recorded_dir = Path("../../data/sensorium")
    predicted_dir = Path("../../runs/vivit/204_causal_viv1t")
    # predicted_dir = Path("../../runs/lRomul")
    # predicted_dir = Path("../../runs/fCNN/038_fCNN")

    df = []
    for mouse_id in data.SENSORIUM_OLD:
        # neurons = utils.get_reliable_neurons(
        #     output_dir=predicted_dir, mouse_id=mouse_id, size=1000
        # )
        print(f"\nProcessing mouse {mouse_id}...")
        # load live_main and final_main test sets which are all natural movies
        tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
        trial_ids = np.where((tiers == "live_main") | (tiers == "final_main"))[0]
        video_ids = data.get_video_ids(mouse_id=mouse_id)[trial_ids]
        recorded_df = find_preferred_video(
            output_dir=recorded_dir,
            mouse_id=mouse_id,
            trial_ids=trial_ids.copy(),
            video_ids=video_ids.copy(),
            model_name="recorded",
            # neurons=neurons,
        )
        df.append(recorded_df)
        predicted_df = find_preferred_video(
            output_dir=predicted_dir,
            mouse_id=mouse_id,
            trial_ids=trial_ids.copy(),
            video_ids=video_ids.copy(),
            model_name="predicted",
            # neurons=neurons,
        )
        df.append(predicted_df)
    df = pd.concat(df, ignore_index=True)
    filename = predicted_dir / "most_exciting_stimulus" / "population" / "videos.pkl"
    df.to_pickle(filename)
    print(f"\nResult saved to {filename}.")


if __name__ == "__main__":
    main()
