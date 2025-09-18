from pathlib import Path

import numpy as np
from einops import rearrange

from viv1t import metrics


def load_response_to_natural_movies(data_dir: Path) -> dict[int, np.ndarray]:
    """Load response to test set natural movies"""
    tiers = np.load(data_dir / "meta" / "trials" / "tiers.npy", allow_pickle=True)
    trial_ids = np.where(tiers == "live_main")[0]
    responses = [
        np.load(data_dir / "data" / "responses" / f"{trial_id}.npy")
        for trial_id in trial_ids
    ]
    responses = np.stack(responses)
    video_ids = np.load(data_dir / "meta" / "trials" / "video_ids.npy")
    video_ids = video_ids[trial_ids]
    responses_ = {}
    for video_id in np.unique(video_ids):
        trial_ids = np.where(video_ids == video_id)[0]
        responses_[video_id] = rearrange(
            responses[trial_ids],
            "repeat neuron frame -> neuron repeat frame",
        )
    return responses_


def check_response_overlap(
    day1_response: np.ndarray,
    day2_response: np.ndarray,
    percentage: float = 0.8,
) -> np.ndarray:
    """
    Return True if the average day2_response is within mean + standard
    deviation of day1_response for over 90% of the timestamps
    """
    mean = np.mean(day1_response, axis=1)
    sd = np.std(day1_response, axis=1)
    response = np.mean(day2_response, axis=1)
    duration = response.shape[0]
    # check average day 2 response to be within mean +/- sd for day 1 response
    overlap = (response <= (mean + sd)) & (response >= (mean - sd))
    # check overlap percentage number of time steps
    overlap = (np.sum(overlap, axis=1) / duration) >= percentage
    return overlap


def filter_neurons(
    day1_data_dir: Path, day2_data_dir: Path, percentage: float = 0.8
) -> np.ndarray:
    """
    Returns neurons with reliable response to natural stimuli between day 1
    and day N recording.
    """
    day1_responses = load_response_to_natural_movies(data_dir=day1_data_dir)
    day2_responses = load_response_to_natural_movies(data_dir=day2_data_dir)
    video_ids = list(day1_responses.keys())
    num_neurons = day1_responses[video_ids[0]].shape[0]
    reliability = np.zeros((len(video_ids), num_neurons), dtype=bool)
    for i, video_id in enumerate(video_ids):
        reliability[i] = check_response_overlap(
            day1_response=day1_responses[video_id],
            day2_response=day2_responses[video_id],
            percentage=percentage,
        )
    # check neural response overlap for at least half of the unique videos
    neurons = np.where(np.sum(reliability, axis=0) >= (len(video_ids) // 2))[0]
    return neurons
