import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import stimulus

FPS = 30

plot.set_font()

DATA_DIR = Path("../../data")
OUTPUT_DIR = Path("../../runs")

PATTERN_SIZE = 30
BLANK_SIZE = 15

VIPcre232_FOV1_day2_VIDEO_IDS = [
    ["CM000", "CM004"],  # low contrast center
    ["CM001", "CM005"],  # low contrast iso
    ["CM002", "CM006"],  # low contrast cross
    # ["CM003", "CM007"],  # low contrast shift
    ["CM008", "CM012"],  # high contrast center
    ["CM009", "CM013"],  # high contrast iso
    ["CM010", "CM014"],  # high contrast cross
    # ["CM011", "CM015"],  # high contrast shift
]


VIPcre232_FOV2_day2_VIDEO_IDS = [
    ["CM004", "CM012"],  # low contrast center (↑ or ↓)
    ["CM005", "CM013"],  # low contrast iso
    ["CM006", "CM014"],  # low contrast cross
    # ["CM007", "CM015"],  # low contrast shift
    ["CM020", "CM028"],  # high contrast center (↑ or ↓)
    ["CM021", "CM029"],  # high contrast iso
    ["CM022", "CM030"],  # high contrast cross
    # ["CM023", "CM031"],  # high contrast shift
]

VIPcre232_FOV2_day4_VIDEO_IDS = [
    ["CM000", "CM003"],  # low contrast center (↑ or ↓)
    ["CM001", "CM004"],  # low contrast iso
    ["CM003", "CM005"],  # low contrast cross
    ["CM006", "CM009"],  # high contrast center (↑ or ↓)
    ["CM007", "CM010"],  # high contrast iso
    ["CM008", "CM011"],  # high contrast cross
]

VIPcre233_FOV1_day2_VIDEO_IDS = [
    ["CM000", "CM003"],  # low contrast center
    ["CM001", "CM004"],  # low contrast iso
    ["CM002", "CM005"],  # low contrast cross
    ["CM006", "CM009"],  # high contrast center
    ["CM007", "CM010"],  # high contrast iso
    ["CM008", "CM011"],  # high contrast cross
]

VIPcre233_FOV2_day2_VIDEO_IDS = [
    ["CM000", "CM003"],  # low contrast center
    ["CM001", "CM004"],  # low contrast iso
    ["CM002", "CM005"],  # low contrast cross
    ["CM006", "CM009"],  # high contrast center
    ["CM007", "CM010"],  # high contrast iso
    ["CM008", "CM011"],  # high contrast cross
]


def select_responsive_neurons(responses: np.ndarray) -> np.ndarray:
    """Select neurons that responses significantly to high contrast center"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # select high contrast response only
        responses = responses[:, 3, :, :]
        # average response over repeats
        responses = np.nanmean(responses, axis=1)
        # select response before presentation and average over the 4 conditions
        before = responses[:, :BLANK_SIZE]
        # set threshold to 2 times the standard deviation of grey response plus mean.
        threshold = np.nanmean(before, axis=1) + 2 * np.nanstd(before, axis=1)
        # select response during presentation
        during = responses[:, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
        # get maximum response for each neuron during presentation
        max_responses = np.nanmax(during, axis=1)
        neurons = np.where(max_responses >= threshold)[0]
    return neurons


def select_reliable_neurons(
    responses: np.ndarray, neurons: np.ndarray | None = None
) -> np.ndarray:
    """Select neurons that have reliable response to high contrast center"""
    # select response duration stimulus presentation
    responses = responses[:, :, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    triu = np.triu_indices(responses.shape[2], k=1)
    num_neurons = responses.shape[0]
    num_repeats = responses.shape[2]

    num_random = 100
    real_correlation = np.full(num_neurons, fill_value=np.nan, dtype=np.float32)
    permuted_correlations = np.full(
        (num_neurons, num_random), fill_value=np.nan, dtype=np.float32
    )
    rng = np.random.RandomState(1234)

    if neurons is None:
        neurons = np.arange(num_neurons)
    for neuron in tqdm(neurons, desc="Neuron reliability"):
        response = responses[neuron]
        if np.all(np.isnan(response)):
            continue

        # compute pairwise correlation to high contrast center
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation = np.corrcoef(response[3])
        real_correlation[neuron] = np.nanmean(correlation[triu])

        # permutation test on randomly select responses
        response = rearrange(response, "stimulus repeat time -> (stimulus repeat) time")
        for i in range(num_random):
            indexes = rng.choice(response.shape[0], size=num_repeats, replace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                correlation = np.corrcoef(response[indexes, :])
            permuted_correlations[neuron, i] = np.nanmean(correlation[triu])

    # include a neuron if the real correlation is larger than 95% of the permuted correlations
    counts = np.sum(real_correlation[:, None] >= permuted_correlations, axis=1)
    percentages = counts / num_random
    neurons = np.where(percentages >= 0.95)[0]
    return neurons


def select_RF_neurons(output_dir: Path, mouse_id: str) -> np.ndarray:
    """
    Return neurons with estimated RF center is within 10° of the stimulus center
    """
    population_RF = stimulus.load_population_RF_center(
        output_dir=output_dir, mouse_id=mouse_id
    )
    monitor_info = data.MONITOR_INFO[mouse_id]
    circular_mask = stimulus.create_circular_mask(
        stimulus_size=10,
        center=population_RF,
        monitor_width=monitor_info["width"],
        monitor_height=monitor_info["height"],
        monitor_distance=monitor_info["distance"],
        num_frames=1,
    )  # (C, T, H, W)
    circular_mask = circular_mask[0, 0]  # remove time and channel dimensions
    neuron_RFs = pd.read_parquet(output_dir / "aRF.parquet")
    RF_x = neuron_RFs.center_x.values
    RF_y = neuron_RFs.center_y.values
    bad_RFs = neuron_RFs.bad_fit.values
    # replace bad RF fits with 0
    bad_RF_x = np.where((RF_x < 0) | (RF_x > monitor_info["width"]))[0]
    bad_RF_y = np.where((RF_y < 0) | (RF_y > monitor_info["height"]))[0]
    bad_RFs[bad_RF_x], bad_RFs[bad_RF_y] = True, True
    RF_x[bad_RFs], RF_y[bad_RFs] = 0, 0
    # round RF to nearest pixel
    RF_x = np.round(RF_x, decimals=0).astype(int)
    RF_y = np.round(RF_y, decimals=0).astype(int)
    # check if neuron RF is in 20 degree of population RF
    within = circular_mask[RF_y, RF_x]
    within[bad_RFs] = False
    RF_neurons = np.where(within == True)[0]
    return RF_neurons


def filter_neurons(
    output_dir: Path, mouse_id: str, responses: np.ndarray
) -> np.ndarray:
    # remove NaN neurons that were not matched between day 1 and 2
    neurons = np.where(~np.isnan(responses[:, 0, 0, 0]))[0]
    # select neurons that responded significantly to high contrast center
    responsive_neurons = select_responsive_neurons(responses=responses.copy())
    neurons = np.intersect1d(neurons, responsive_neurons)
    # select neurons with RF within 10 degree of the presentation center
    RF_neurons = select_RF_neurons(output_dir=output_dir, mouse_id=mouse_id)
    neurons = np.intersect1d(neurons, RF_neurons)
    # select neurons that responded reliably to high contrast center
    reliable_neurons = select_reliable_neurons(responses=responses.copy())
    neurons = np.intersect1d(neurons, reliable_neurons)
    return neurons


def to_parquet(responses: np.ndarray, output_dir: Path, mouse_id: str) -> pd.DataFrame:
    neurons = filter_neurons(
        output_dir=output_dir,
        mouse_id=mouse_id,
        responses=responses.copy(),
    )
    # select response during presentation
    responses = responses[neurons, :, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    # sum response during presentation
    responses = np.sum(responses, axis=3)
    df = pd.DataFrame(
        {
            "neuron": neurons,
            "low_center": responses[:, 0, :].tolist(),
            "low_iso": responses[:, 1, :].tolist(),
            "low_cross": responses[:, 2, :].tolist(),
            "high_center": responses[:, 3, :].tolist(),
            "high_iso": responses[:, 4, :].tolist(),
            "high_cross": responses[:, 5, :].tolist(),
        }
    )
    df.insert(loc=0, column="mouse", value=mouse_id)
    return df


def load_recorded_data(
    data_dir: Path,
    video_id_groups: list[list[str]],
    output_dir: Path,
    mouse_id: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    video_ids = np.load(
        data_dir / "meta" / "trials" / "video_ids.npy", allow_pickle=True
    )
    responses = [
        np.load(data_dir / "data" / "responses" / f"{trial_id}.npy")
        for trial_id in range(len(video_ids))
    ]
    responses = np.stack(responses)
    # reshape responses into blank - presentation - blank blocks
    responses = rearrange(
        responses,
        "trial neuron (block frame) -> neuron trial block frame",
        frame=BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE,
    )
    responses = rearrange(
        responses, "neuron trial block frame -> neuron (trial block) frame"
    )
    video_ids = rearrange(video_ids, "trial block -> (trial block)")
    assert responses.shape[1] == len(video_ids)

    num_neurons = responses.shape[0]
    num_repeats = 10
    num_frames = responses.shape[2]
    responses_ = np.full(
        shape=(num_neurons, len(video_id_groups), num_repeats, num_frames),
        fill_value=np.nan,
        dtype=np.float32,
    )
    for i in range(len(video_id_groups)):
        indexes = np.where(np.isin(video_ids, video_id_groups[i]))[0]
        assert len(indexes) >= 5
        if len(indexes) > num_repeats:
            indexes = rng.choice(indexes, size=num_repeats, replace=False)
        responses_[:, i, : len(indexes)] = responses[:, indexes]
    responses = responses_.copy()
    df = to_parquet(responses=responses, output_dir=output_dir, mouse_id=mouse_id)
    print(f"Select (in vivo) {len(df)} neurons out of {num_neurons} neurons.")
    return df


def load_predicted_data(
    output_dir: Path,
    mouse_id: str,
    rng: np.random.RandomState,
    recorded_neurons: np.ndarray | None = None,
) -> pd.DataFrame:
    data_dir = output_dir / "contextual_modulation_population"

    responses = np.load(data_dir / f"mouse{mouse_id}.npz")["data"]
    parameters = np.load(data_dir / "parameters.npy")

    responses = rearrange(
        responses, "neuron trial block pattern -> neuron (trial block) pattern"
    )
    parameters = rearrange(parameters, "trial block param -> (trial block) param")

    stimulus_size = 20
    num_repeats = 10
    neurons = np.arange(responses.shape[0], dtype=int)

    responses_ = np.full(
        shape=(len(neurons), 6, num_repeats, BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE),
        fill_value=np.nan,
        dtype=np.float32,
    )
    i = 0
    for contrast_id, contrast_type in [(0.05, "low_contrast"), (1, "high_contrast")]:
        for stimulus_id, stimulus_type in [(0, "center"), (1, "iso"), (2, "cross")]:
            indexes = np.where(
                (parameters[:, 0] == contrast_id)
                & (parameters[:, 1] == stimulus_size)
                & (parameters[:, 3] == stimulus_id)
            )[0]
            assert len(indexes) >= 5
            if len(indexes) > num_repeats:
                indexes = rng.choice(indexes, size=num_repeats, replace=False)
            responses_[:, i, :, :] = responses[:, indexes]
            i += 1
    responses = responses_
    del responses_, parameters, i, stimulus_size

    df = to_parquet(responses=responses, output_dir=output_dir, mouse_id=mouse_id)

    # select only L2/3 neurons for Sensorium data
    if mouse_id in data.SENSORIUM:
        depths = data.get_neuron_coordinates(mouse_id=mouse_id)[:, 2]
        neurons = np.intersect1d(
            neurons,
            np.where((depths >= 100) & (depths <= 300))[0],
        )
        df = df[df.neuron.isin(neurons)]
    # filter recorded neurons
    if recorded_neurons is not None:
        df = df[df.neuron.isin(recorded_neurons)]
    print(f"Select (in silico) {len(df)} neurons out of {len(neurons)} neurons.")
    return df


def main():
    rng = np.random.RandomState(1234)
    output_dir = OUTPUT_DIR / "rochefort-lab" / "vivit"
    in_vivo_df, in_silico_df = [], []
    print(f"Load Rochefort Lab data")
    for mouse_id, day_name, output_dir_name, video_id_groups in [
        (
            "K",
            "day2",
            "003_causal_viv1t_finetune",
            VIPcre232_FOV1_day2_VIDEO_IDS,
        ),
        (
            "L",
            "day2",
            "015_causal_viv1t_FOV2_finetune",
            VIPcre232_FOV2_day2_VIDEO_IDS,
        ),
        # (
        #     "L",
        #     "day4",
        #     "015_causal_viv1t_FOV2_finetune",
        #     VIPcre232_FOV2_day4_VIDEO_IDS,
        # ),
        (
            "M",
            "day2",
            "018_causal_viv1t_VIPcre233_FOV1_finetune",
            VIPcre233_FOV1_day2_VIDEO_IDS,
        ),
        (
            "N",
            "day2",
            "025_causal_viv1t_VIPcre233_FOV2_finetune",
            VIPcre233_FOV2_day2_VIDEO_IDS,
        ),
    ]:
        print(f"\nmouse {mouse_id}")
        in_vivo = load_recorded_data(
            data_dir=DATA_DIR
            / data.MOUSE_IDS[f"{mouse_id}_{day_name}"]
            / "artificial_movies",
            video_id_groups=video_id_groups,
            output_dir=output_dir / output_dir_name,
            mouse_id=mouse_id,
            rng=rng,
        )
        in_vivo_df.append(in_vivo)
        in_silico = load_predicted_data(
            output_dir=output_dir / output_dir_name,
            mouse_id=mouse_id,
            rng=rng,
        )
        in_silico_df.append(in_silico)
        del in_vivo, in_silico
    in_vivo_df = pd.concat(in_vivo_df, ignore_index=True)
    in_silico_df = pd.concat(in_silico_df, ignore_index=True)

    in_vivo_df.to_parquet("in_vivo_rochefort_lab.parquet")
    in_silico_df.to_parquet("in_silico_rochefort_lab.parquet")

    sensorium_df = []
    print(f"\nLoad Sensorium data")
    for mouse_id in data.SENSORIUM_OLD:
        print(f"\nmouse {mouse_id}")
        sensorium_df.append(
            load_predicted_data(
                output_dir=OUTPUT_DIR / "vivit" / "204_causal_viv1t",
                mouse_id=mouse_id,
                rng=rng,
            )
        )
    sensorium_df = pd.concat(sensorium_df, ignore_index=True)
    sensorium_df.to_parquet("in_silico_sensorium.parquet")

    print(f"Saved in vivo and in silico responses to parquet files.")


if __name__ == "__main__":
    main()
