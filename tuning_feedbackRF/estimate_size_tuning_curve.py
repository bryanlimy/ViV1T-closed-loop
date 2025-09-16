from pathlib import Path

import numpy as np
import pandas as pd
from einops import rearrange
from scipy.stats import sem
from scipy.stats import wilcoxon
from scipy.stats import zscore

from viv1t import data

BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
FPS = 30

THRESHOLD = 1.98  # z-score threshold


def get_stimulus_sizes(output_dir: Path) -> np.ndarray:
    parameters = np.load(
        output_dir / "feedbackRF" / "parameters.npy", allow_pickle=False
    )
    return np.unique(parameters[..., 0])


def normalize(responses: np.ndarray) -> np.ndarray:
    """
    Normalize response by the maximum response to classical stimuli per neuron

    Args:
        responses: np.ndarray, response in shape (neurons, stimulus size, classical or inverse)
    """
    assert responses.ndim == 3 and responses.shape[2] == 2
    max_responses = np.max(responses[:, :, 0], axis=1)
    responses = responses / max_responses[:, None, None]
    return responses


def compute_pattern_average(responses: np.ndarray):
    """
    Compute the average response to each block pattern so that, for each pattern,
    the response of a neuron is represented by a single value.
    """
    # select response during stimulus presentation
    responses = responses[:, :, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    # remove the first 0.5s of each trial as per Keller et al.:
    #   The response amplitude to a stimulus was computed as the average
    #   response over the duration of the stimulus presentation (excluding the
    #   first 0.5 s of each trial for two-photon experiments owing to the delay
    #   and slow rise of calcium indicators).
    responses = responses[:, :, :, FPS // 2 :]
    # average response over frames for each pattern
    responses = np.mean(responses, axis=-1)
    return responses


def reshape_responses(
    responses: np.ndarray, parameters: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Reshape responses to (neuron, stimulus size, stimulus type) and
    compute the average over repeated presentations of the same pattern.
    """
    responses = rearrange(responses, "neuron block pattern -> neuron (block pattern)")
    parameters = rearrange(parameters, "block pattern param -> (block pattern) param")
    stimulus_sizes = np.unique(parameters[:, 0])
    stimulus_types = np.unique(parameters[:, 2])
    responses_ = np.zeros(
        (responses.shape[0], len(stimulus_sizes), len(stimulus_types)),
        dtype=np.float32,
    )
    for i, stimulus_size in enumerate(stimulus_sizes):
        for j, stimulus_type in enumerate(stimulus_types):
            # find patterns with the same stimulus size and type
            idx = np.where(
                (parameters[:, 0] == stimulus_size)
                & (parameters[:, 2] == stimulus_type)
            )[0]
            # compute average response over repeats
            responses_[:, i, j] = np.mean(responses[:, idx], axis=-1)
    return responses_, parameters


def load_data(mouse_id: str, output_dir: Path):
    # shape (num. neurons, num. samples, num. frames)
    responses = np.load(
        output_dir / "feedbackRF" / f"mouse{mouse_id}.npz", allow_pickle=False
    )["data"]
    # shape (num. samples, num. frames, parameter)
    parameters = np.load(
        output_dir / "feedbackRF" / "parameters.npy", allow_pickle=False
    )
    responses = compute_pattern_average(responses)
    responses, parameters = reshape_responses(responses, parameters)
    return responses, parameters


def get_classic_tuned_neurons(responses: np.ndarray, threshold: float = THRESHOLD):
    """
    Extract neurons that are responsive to classical size tuning stimuli

    From Keller et al. 2020:
    Page 9, Inclusion criteria for Figure 1c:
        We estimated the centre of the receptive field by fitting
        the responses to patches of gratings presented along a grid with a
        two-dimensional Gaussian. Neurons were included if they significantly
        responded to patches of gratings at any location within 10° of their
        estimated centres, their estimated centres were within 10° of the centre
        of the size-tuning stimuli, and they significantly responded to at least
        one classical size-tuning stimulus.
    Page 8, Response amplitude:
        Response amplitude. The response amplitude to a stimulus was
        computed as the average response over the duration of the stimulus
        presentation (excluding the first 0.5 s of each trial for two-photon
        experiments owing to the delay and slow rise of calcium indicators).
        Responses were normalized by the maximum response over the relevant
        stimulus parameter space and then averaged over neurons or units. We
        defined significant responses as responses that exceeded a z-score of
        3.29 (corresponding to P < 10−3) or 5.33 (corresponding to P < 10−7;
        for two-photon experiments in L4).

    Here we use a z-score of 1.96 (p-value 0.05) as a threshold
    """
    assert len(responses.shape) == 3
    responses = normalize(responses)
    # only consider classical patterns
    responses = responses[:, :, 0]
    # compute z-scores over different patterns
    z_scores = zscore(responses, axis=-1)
    # include neurons that response significantly to at least one non-zero
    # stimulus size
    responsive_to_any = np.any(z_scores >= threshold, axis=-1)
    responsive_to_zero = z_scores[:, 0] >= threshold
    classic_neurons = np.where(responsive_to_any & ~responsive_to_zero)[0]
    return classic_neurons


def get_inverse_tuned_neurons(responses: np.ndarray, threshold: float = THRESHOLD):
    """
    Extract neurons that are responsive to inverse size tuning stimuli

    From Keller et al. 2020, Page 8 Defining inverse-tuned neurons:
        Neurons were defined as inverse-tuned if they significantly responded
        to at least one classical and one inverse stimulus and if their
        response to at least one inverse stimulus of any size centred on
        their ffRF was significantly larger than that to a full-field stimulus
        (or approximated by the response to the largest classical or smallest
        inverse stimulus presented).
    """
    assert len(responses.shape) == 3
    # get classical neurons
    classic_neurons = get_classic_tuned_neurons(
        responses=responses.copy(), threshold=threshold
    )
    responses = normalize(responses)
    # get neurons that response significantly to at least one inverse stimulus
    responses = responses[:, :, 1]
    # compute z-scores over different patterns
    z_scores = zscore(responses, axis=-1)
    # neurons that are responsive to any condition
    responsive_to_any = np.any(z_scores >= threshold, axis=-1)
    responsive_to_zero = z_scores[:, 0] >= threshold
    # check if neurons response to at least one inverse stimulus of any size
    inverse_neurons = np.where(responsive_to_any & ~responsive_to_zero)[0]
    # check if neurons are also classic-tuned
    inverse_neurons = np.intersect1d(classic_neurons, inverse_neurons)
    # check if response to the preferred inverse stimulus size was significantly
    # larger than that to a full-field stimulus
    stronger = np.where(np.max(responses[:, 1:], axis=1) > responses[:, 0])[0]
    inverse_neurons = np.intersect1d(inverse_neurons, stronger)
    return inverse_neurons


def get_preferred_size(responses: np.ndarray):
    """Get the preferred stimulus size for each neuron to each stimulus type"""
    assert len(responses.shape) == 3
    isnan = np.any(np.isnan(rearrange(responses, "n d1 d2 -> n (d1 d2)")), axis=1)
    classic_preferences = np.argmax(responses[:, :, 0], axis=-1) * 10.0
    classic_preferences[isnan] = np.nan
    inverse_preferences = np.argmax(responses[:, :, 1], axis=-1) * 10.0
    inverse_preferences[isnan] = np.nan
    return classic_preferences, inverse_preferences


def process_model(model_name: str, output_dir: Path):
    print(f"Estimate feedback for responses in {output_dir}.")
    # filter neurons and save size-tuning profile
    df = []
    mouse_responses = {}
    for mouse_id in data.MOUSE_IDS.keys():
        filename = output_dir / "feedbackRF" / f"mouse{mouse_id}.npz"
        if not filename.exists():
            continue
        print(f"Processing mouse {mouse_id}")
        responses, parameters = load_data(mouse_id=mouse_id, output_dir=output_dir)
        neurons = np.arange(responses.shape[0])
        classic_neurons = get_classic_tuned_neurons(responses=responses.copy())
        inverse_neurons = get_inverse_tuned_neurons(responses=responses.copy())
        print(
            f"\t{100 * len(classic_neurons) / len(neurons):.2f}% neurons are classic-tuned neurons.\n"
            f"\t{100 * len(inverse_neurons) / len(neurons):.2f}% neurons are inverse-tuned neurons."
        )
        classic_preferences, inverse_preferences = get_preferred_size(responses)
        mouse_df = pd.DataFrame(
            {
                "neuron": neurons,
                "depth": data.get_neuron_coordinates(mouse_id=mouse_id)[:, 2],
                "classic_tuned": np.isin(neurons, classic_neurons),
                "inverse_tuned": np.isin(neurons, inverse_neurons),
                "classic_preference": classic_preferences,
                "inverse_preference": inverse_preferences,
                "classic_tuning_curve": responses[:, :, 0].tolist(),
                "inverse_tuning_curve": responses[:, :, 1].tolist(),
            }
        )
        mouse_df.insert(loc=0, column="mouse", value=mouse_id)
        df.append(mouse_df)
        mouse_responses[mouse_id] = responses
    # store response amplitudes to disk which is used for contextual modulation
    # analysis as well
    df = pd.concat(df, ignore_index=True)
    filename = output_dir / "size_tuning_preference.parquet"
    df.to_parquet(filename, index=False)

    responses = np.concat(list(mouse_responses.values()), axis=0)
    preferred_sizes = np.argmax(responses, axis=1) * 10
    print(
        f"Preferred classical stimulus size:\n"
        f"\tmean: {np.mean(preferred_sizes[:, 0]):.02f}"
        f" +/- {sem(preferred_sizes[:, 0]):.02f}◦\n"
        f"\tmedian: {np.median(preferred_sizes[:, 0]):.02f}◦\n"
        f"Preferred inverse stimulus size:\n"
        f"\tmean: {np.mean(preferred_sizes[:, 1]):.02f}"
        f" +/- {sem(preferred_sizes[:, 1]):.02f}◦\n"
        f"\tmedian: {np.median(preferred_sizes[:, 1]) :.02f}◦\n"
        f"num. neurons: {len(responses)}"
    )
    print(f"Saved size-tuning results to {filename}.\n")


def main():
    models = {
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    for model_name, output_dir in models.items():
        process_model(model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
