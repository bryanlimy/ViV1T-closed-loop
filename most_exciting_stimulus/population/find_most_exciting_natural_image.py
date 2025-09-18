"""
Find the most-exciting/preferred directional grating in the recorded population response
"""

from pathlib import Path

import numpy as np
import pandas as pd
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kendalltau
from tqdm import tqdm

from viv1t import data
from viv1t.data import get_flashing_image_parameters
from viv1t.utils import h5
from viv1t.utils import utils

DATA_DIR = Path("../../data")

PATTERN_SIZE = 15  # slide window size to match flashing image presentation window
BLANK_SIZE = 10  # number of frames before and after each image presentation to extract


def load_max_response(filename: Path, mouse_id: str) -> np.ndarray:
    max_responses = np.load(filename, allow_pickle=True)
    return max_responses[mouse_id]


def load_video(mouse_id: str, trial_id: int) -> np.ndarray:
    video = np.load(
        DATA_DIR / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    video = video[..., : data.MAX_FRAME]
    video = np.round(video, decimals=0)
    video = rearrange(video, "h w t -> () t h w")
    return video


def load_data(
    mouse_id: str, output_dir: Path, trial_ids: np.ndarray
) -> (np.ndarray, dict[int, np.ndarray], np.ndarray):
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot find {filename}.")

    # responses shape (trial, neuron, time)
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]

    image_ids = np.array(
        [
            get_flashing_image_parameters(mouse_id, trial_id=trial_id)
            for trial_id in trial_ids
        ],
        dtype=int,
    )
    image_ids = image_ids[:, -num_frames:]

    images = {}
    responses_ = {}
    for image_id in np.unique(image_ids):
        if image_id == -1:
            continue  # ignore blank screen
        for i in range(image_ids.shape[0]):
            if image_id not in image_ids[i]:
                continue
            indexes = np.where(image_ids[i] == image_id)[0]
            # requires presentation to be at least PATTERN_SIZE presentation window
            if len(indexes) < PATTERN_SIZE:
                continue
            start, end = indexes[0], indexes[-1] + 1
            # for some reason, the first and last frame of the presentation window
            # are at a much lower contrast
            if len(indexes) - PATTERN_SIZE == 1:
                start += 1
            elif len(indexes) - PATTERN_SIZE == 2:
                start += 1
                end -= 1
            # require presentation to have at least BLANK_SIZE blank screen
            # before and after the presentation
            assert end - start == PATTERN_SIZE
            start -= BLANK_SIZE
            end += BLANK_SIZE
            if start <= 0 or end >= num_frames:
                continue
            if image_id not in responses_:
                responses_[image_id] = []
            responses_[image_id].append(responses[i, :, start:end])
            # get image from video
            if image_id not in images:
                video = load_video(mouse_id=mouse_id, trial_id=trial_ids[i])
                video = video[:, -num_frames:, :, :]
                frame = video[0, start + BLANK_SIZE : end - BLANK_SIZE]
                images[image_id] = frame

    # randomly select repeat so that all unique images have equal number of repeats
    rng = np.random.default_rng(1234)
    min_repeat = min([len(r) for r in responses_.values()])
    image_ids = np.array(list(responses_.keys()), dtype=int)
    responses_ = np.stack(
        [
            np.stack(r)[rng.choice(len(r), size=min_repeat, replace=False)]
            for r in responses_.values()
        ]
    )
    responses_ = rearrange(
        responses_, "image repeat neuron frame -> image repeat frame neuron"
    )
    assert len(image_ids) == len(responses_) == len(images)
    return responses_, images, image_ids


def find_preferred_image(
    responses: np.ndarray,
    images: dict[int, np.ndarray],
    image_ids: np.ndarray,
    mouse_id: str,
    model_name: str,
    neurons: np.ndarray | None = None,
) -> (pd.DataFrame, np.ndarray):
    if neurons is not None:
        responses = responses[:, :, :, neurons]
    # average response over repeats
    responses = np.mean(responses, axis=1)
    # mean response over population
    responses = np.mean(responses, axis=-1)
    # get response in presentation window
    response = responses[:, BLANK_SIZE:-BLANK_SIZE]
    # sum response over presentation window
    response = np.sum(response, axis=1)
    # select the direction with the strongest response
    index = np.argmax(response)
    most_exciting_image = image_ids[index]
    print(f"{model_name} mouse {mouse_id} most exciting image {most_exciting_image}")
    most_exciting_ranking = image_ids[np.argsort(response)[::-1]]
    print(f"Top 10 images: {most_exciting_ranking[:10]}")
    df = pd.DataFrame(
        {
            "mouse": mouse_id,
            "model": model_name,
            "image": image_ids,
            "response": response,
            "raw_response": responses.tolist(),
            "stimulus": list(images.values()),
        }
    )
    return (df, most_exciting_ranking)


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
        stimulus_ids = data.get_stimulus_ids(mouse_id)
        trial_ids = np.where(stimulus_ids == 5)[0]
        if not trial_ids.size:
            continue  # mouse does not have flashing image stimulus
        recorded_responses, images, image_ids = load_data(
            mouse_id=mouse_id,
            output_dir=recorded_dir,
            trial_ids=trial_ids,
        )
        recorded_df, recorded_ranking = find_preferred_image(
            responses=recorded_responses,
            images=images,
            image_ids=image_ids,
            mouse_id=mouse_id,
            model_name="recorded",
            # neurons=neurons,
        )
        df.append(recorded_df)
        predicted_responses, images, image_ids = load_data(
            mouse_id=mouse_id,
            output_dir=predicted_dir,
            trial_ids=trial_ids,
        )
        predicted_df, predicted_ranking = find_preferred_image(
            responses=predicted_responses,
            images=images,
            image_ids=image_ids,
            mouse_id=mouse_id,
            model_name="predicted",
            # neurons=neurons,
        )
        df.append(predicted_df)
        sim = kendalltau(recorded_ranking, predicted_ranking)
        size = 10
        accuracy = (
            len(np.intersect1d(recorded_ranking[:size], predicted_ranking[:size]))
            / size
        )
        print(f"Top 10% accuracy: {100*accuracy:.02f}%")
        print(f"kendall tau stat: {sim.statistic:.04f} p-value: {sim.pvalue:.04f}")
    df = pd.concat(df, ignore_index=True)
    filename = predicted_dir / "most_exciting_stimulus" / "population" / "images.pkl"
    df.to_pickle(filename)
    print(f"\nSaved result to {filename}.")


if __name__ == "__main__":
    main()
