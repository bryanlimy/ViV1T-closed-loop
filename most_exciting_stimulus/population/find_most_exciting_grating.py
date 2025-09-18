"""
Find the most-exciting/preferred directional grating in the recorded population response
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from einops import rearrange
from numpy.lib.stride_tricks import sliding_window_view

from viv1t import data
from viv1t.data import get_gabor_parameters
from viv1t.utils import h5
from viv1t.utils import utils

DATA_DIR = Path("../../data")

BLOCK_SIZE = 25  # drifting Gabor presentation window
WINDOW_SIZE = 15  # slide window size to match flashing image presentation window


def load_max_response(filename: Path, mouse_id: str) -> np.ndarray:
    max_responses = np.load(filename, allow_pickle=True)
    return max_responses[mouse_id]


def load_video(mouse_id: str, trial_id: int) -> np.ndarray:
    video = np.load(
        DATA_DIR / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    video = video[..., : data.MAX_FRAME]
    video = np.round(video, decimals=0)
    video = rearrange(video, "h w t -> t h w")
    return video


def load_data(
    mouse_id: str, output_dir: Path, trial_ids: np.ndarray
) -> (np.ndarray, np.ndarray):
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot find {filename}.")

    # responses shape (trial, neuron, time)
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]
    # separate response to each direction of BLOCK_SIZE
    responses = rearrange(
        responses,
        "trial neuron (block frame) -> (trial block) frame neuron",
        frame=BLOCK_SIZE,
    )

    # get direction parameters for each frame
    parameters = np.array(
        [get_gabor_parameters(mouse_id, trial_id=trial_id) for trial_id in trial_ids],
        dtype=np.float32,
    )
    parameters = parameters[:, -num_frames:]
    parameters = rearrange(
        parameters,
        "trial (block frame) param -> (trial block) frame param",
        frame=BLOCK_SIZE,
    )
    assert np.all(parameters == parameters[:, 0][:, None])
    parameters = parameters[:, 0, :]

    # load stimuli
    videos = np.stack(
        [load_video(mouse_id=mouse_id, trial_id=trial_id) for trial_id in trial_ids]
    )
    videos = videos[:, -num_frames:, :, :]
    videos = rearrange(
        videos,
        "trial (block frame) H W  -> (trial block) frame H W",
        frame=BLOCK_SIZE,
    )

    assert parameters.shape[0] == responses.shape[0] == videos.shape[0]

    # find the minimum number of presentations in each Gabor parameter configuration
    unique_parameters, counts = np.unique(parameters, return_counts=True, axis=0)
    min_repeats = np.min(counts)

    # randomly select min_samples for each direction
    rng = np.random.default_rng(seed=1234)
    responses_ = np.zeros(
        (len(unique_parameters), min_repeats, BLOCK_SIZE, num_neurons), dtype=np.float32
    )
    videos_ = np.zeros(
        (len(unique_parameters), BLOCK_SIZE, videos.shape[2], videos.shape[3]),
        dtype=np.float32,
    )
    for i, parameter in enumerate(unique_parameters):
        index = np.where(np.all(parameters == parameter, axis=1))[0]
        index = rng.choice(index, size=min_repeats, replace=False)
        responses_[i] = responses[index]
        videos_[i] = videos[index[0]]
    return responses_, videos_, unique_parameters


def find_preferred_direction(
    responses: np.ndarray,
    videos: np.ndarray,
    parameters: np.ndarray,
    mouse_id: str,
    model_name: str,
    neurons: np.ndarray | None = None,
) -> pd.DataFrame:
    if neurons is not None:
        responses = responses[:, :, :, neurons]
    # average response over repeats
    responses = np.mean(responses, axis=1)
    # mean response over population
    responses = np.mean(responses, axis=-1)
    # use a sliding window of WINDOW_SIZE to find the direction with the most
    # response of the window
    response = sliding_window_view(responses, window_shape=WINDOW_SIZE, axis=1)
    # sum response over sliding window
    response = np.sum(response, axis=2)
    # select the window with the most response
    response = np.max(response, axis=1)
    df = pd.DataFrame(
        {
            "mouse": mouse_id,
            "model": model_name,
            "direction": parameters[:, 0],
            "wavelength": parameters[:, 1],
            "frequency": parameters[:, 2],
            "response": response,
            "raw_response": responses.tolist(),
            "stimulus": videos.tolist(),
        }
    )
    print(
        f"Most-exciting Gabor parameters:\n"
        f"{df.sort_values(by='response', ascending=False).loc[:, ['model', 'direction', 'wavelength' ,'frequency', 'response']].head(5)}"
    )
    return df


def plot_traces(
    responses: np.ndarray, videos: np.ndarray, unique_directions: np.ndarray
):
    figure, axes = plt.subplots(nrows=4, ncols=4, height_ratios=[0.8, 0.2] * 2)

    d = 0
    max_value = 0
    for row in [0, 2]:
        for col in [0, 1, 2, 3]:
            axes[row, col].imshow(
                videos[d, videos.shape[1] // 2], cmap="gray", vmin=0, vmax=255
            )
            axes[row, col].set_title(
                f"Direction {unique_directions[d]}", fontsize=8, pad=0
            )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            response = np.mean(responses[d], axis=(0, 2))
            axes[row + 1, col].plot(response, linewidth=1.5)
            max_value = max(max_value, np.max(response))
            d += 1
    max_value = np.ceil(max_value)
    for row in [1, 3]:
        for col in [0, 1, 2, 3]:
            axes[row, col].set_ylim(0, max_value)
            sns.despine(ax=axes[row, col])
    plt.show()


def main():
    recorded_dir = Path("../../data/sensorium")
    predicted_dir = Path("../../runs/vivit/204_causal_viv1t")

    df = []
    for mouse_id in data.SENSORIUM_OLD:
        # neurons = utils.get_reliable_neurons(
        #     output_dir=predicted_dir, mouse_id=mouse_id, size=1000
        # )
        print(f"\nProcessing mouse: {mouse_id}...")
        stimulus_ids = data.get_stimulus_ids(mouse_id)
        trial_ids = np.where(stimulus_ids == 4)[0]
        if not trial_ids.size:
            continue  # mouse does not have drifting gabor stimulus
        recorded_responses, videos, parameters = load_data(
            mouse_id=mouse_id,
            output_dir=recorded_dir,
            trial_ids=trial_ids,
        )
        # plot_traces(recorded_responses, videos, parameters)
        recorded_df = find_preferred_direction(
            responses=recorded_responses,
            videos=videos,
            parameters=parameters,
            mouse_id=mouse_id,
            model_name="recorded",
            # neurons=neurons,
        )
        df.append(recorded_df)
        predicted_responses, videos, parameters = load_data(
            mouse_id=mouse_id,
            output_dir=predicted_dir,
            trial_ids=trial_ids,
        )
        predicted_df = find_preferred_direction(
            responses=predicted_responses,
            videos=videos,
            parameters=parameters,
            mouse_id=mouse_id,
            model_name="predicted",
            # neurons=neurons,
        )
        df.append(predicted_df)
    df = pd.concat(df, ignore_index=True)
    filename = predicted_dir / "most_exciting_stimulus" / "population" / "gratings.pkl"
    df.to_pickle(filename)
    print(f"\nResult saved to {filename}.")


if __name__ == "__main__":
    main()
