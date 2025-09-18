from multiprocessing import cpu_count
from pathlib import Path
from typing import Tuple

import numpy as np
from einops import rearrange
from numpy.random import Generator
from numpy.random import SeedSequence
from numpy.random import default_rng
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from viv1t import data
from viv1t.data import DIRECTIONS
from viv1t.data import get_gabor_parameters
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils

DATA_DIR = Path("../data/sensorium")

plot.set_font()

TICK_FONTSIZE = 10
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
DPI = 240

BLOCK_SIZE = 25  # number of frames for each drifting Gabor filter


THRESHOLD = 0.01  # p-value threshold
NUM_PERMUTATIONS = 100  # number of permutations
PARALLEL = True  # use parallel processing


def load_data(mouse_id: str, output_dir: Path, trial_ids: np.ndarray) -> np.ndarray:
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot find {filename}.")

    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]

    # reshape responses to (neurons, trials, blocks, frames)
    responses = rearrange(
        responses,
        "trial neuron (block frame) -> neuron trial block frame",
        frame=BLOCK_SIZE,
    )
    # compute average response per block
    responses = np.mean(responses, axis=-1)

    # get direction parameters for each frame
    gabor_parameters = np.array(
        [get_gabor_parameters(mouse_id, trial_id=trial_id) for trial_id in trial_ids],
        dtype=np.float32,
    )
    directions = gabor_parameters[:, -num_frames:, 0]
    # get direction for each block
    directions = rearrange(
        directions, "trial (block frame) -> trial block frame", frame=BLOCK_SIZE
    )
    directions = directions[:, :, 0].astype(int)

    # combine block and trial dimensions
    responses = rearrange(responses, "neuron trial block -> neuron (trial block)")
    directions = rearrange(directions, "trial block -> (trial block)")

    # find the minimum number of presentations in each direction
    unique_directions, counts = np.unique(directions, return_counts=True)
    min_samples = min(counts)

    # randomly select min_samples for each direction
    rng = np.random.default_rng(seed=1234)
    data = np.zeros(
        (num_neurons, len(unique_directions), min_samples), dtype=np.float32
    )
    for i, direction in enumerate(unique_directions):
        index = np.where(directions == direction)[0]
        index = rng.choice(index, size=min_samples, replace=False)
        data[:, i, :] = responses[:, index]

    return data


def is_responsive(
    si: np.ndarray, permuted: np.ndarray, threshold: float = THRESHOLD
) -> bool:
    p_value = np.sum(permuted >= si) / len(permuted)
    # return np.nan if p_value > threshold else si
    return p_value <= threshold


def compute_OSI(tuning_curve: np.ndarray) -> np.ndarray:
    index = tuning_curve * np.exp(2j * np.radians(list(DIRECTIONS.keys())))
    osi = np.abs(np.sum(index)) / np.sum(tuning_curve)
    return osi


def compute_DSI(tuning_curve: np.ndarray) -> np.ndarray:
    index = tuning_curve * np.exp(1j * np.radians(list(DIRECTIONS.keys())))
    dsi = np.abs(np.sum(index)) / np.sum(tuning_curve)
    return dsi


def compute_tuning_curve(arguments: Tuple[np.ndarray, Generator]):
    responses, rng = arguments

    # number of samples for each direction
    num_samples = responses.shape[-1]

    # compute tuning curve of the real recordings
    tuning_curve = np.mean(responses, axis=-1)
    osi = compute_OSI(tuning_curve)
    dsi = compute_DSI(tuning_curve)

    # randomly shuffle the responses and compute permuted tuning curves
    all_response = rearrange(responses, "direction sample -> (direction sample)")
    permuted_osi = np.zeros(NUM_PERMUTATIONS, dtype=np.float32)
    permuted_dsi = np.zeros(NUM_PERMUTATIONS, dtype=np.float32)
    for i in range(NUM_PERMUTATIONS):
        permuted_tuning_curve = np.mean(
            rearrange(
                rng.permutation(all_response),
                "(direction sample) -> direction sample",
                sample=num_samples,
            ),
            axis=-1,
        )
        permuted_osi[i] = compute_OSI(permuted_tuning_curve)
        permuted_dsi[i] = compute_DSI(permuted_tuning_curve)
    orientation_selective = is_responsive(si=osi, permuted=permuted_osi)
    direction_selective = is_responsive(si=dsi, permuted=permuted_dsi)

    return tuning_curve, osi, dsi, orientation_selective, direction_selective


def compute_direction_tuning(responses: np.ndarray):
    num_neurons = responses.shape[0]
    seeds = SeedSequence(1234).spawn(num_neurons)

    tuning_curve = np.zeros((num_neurons, len(DIRECTIONS)), dtype=np.float32)
    OSI = np.zeros(num_neurons, dtype=np.float32)
    DSI = np.zeros(num_neurons, dtype=np.float32)
    orientation_selective = np.zeros(num_neurons, dtype=bool)
    direction_selective = np.zeros(num_neurons, dtype=bool)

    if PARALLEL:
        result = process_map(
            compute_tuning_curve,
            [(responses[n], default_rng(seed=seeds[n])) for n in range(num_neurons)],
            max_workers=cpu_count() - 2,
            chunksize=1,
            desc="Neuron",
        )
        assert len(result) == num_neurons
    else:
        result = [
            compute_tuning_curve((responses[n], default_rng(seed=seeds[n])))
            for n in tqdm(range(num_neurons), desc="Neuron")
        ]

    for n in range(num_neurons):
        (
            tuning_curve[n],
            OSI[n],
            DSI[n],
            orientation_selective[n],
            direction_selective[n],
        ) = result[n]

    print(
        f"{np.count_nonzero(orientation_selective)}/{num_neurons} "
        f"neurons are orientation selective.\n"
        f"{np.count_nonzero(direction_selective)}/{num_neurons} "
        f"neurons are direction selective.\n"
    )
    return tuning_curve, OSI, DSI, orientation_selective, direction_selective


def estimate_mouse(mouse_id: str, output_dir: Path, meta_dir: Path):
    # find trials with drifting gabor stimuli
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 4)[0]
    if not trial_ids.size:
        return  # mouse does not have drifting gabor stimulus
    print(f"Estimating SIs for mouse {mouse_id} ")
    responses = load_data(
        mouse_id=mouse_id,
        output_dir=output_dir,
        trial_ids=trial_ids,
    )
    tuning_curve, OSI, DSI, orientation_selective, direction_selective = (
        compute_direction_tuning(responses=responses)
    )
    utils.save_tuning(
        result={
            "tuning_curve": tuning_curve,
            "OSI": OSI,
            "DSI": DSI,
            "orientation_selective": orientation_selective,
            "direction_selective": direction_selective,
        },
        save_dir=meta_dir,
        mouse_id=mouse_id,
    )


def main():
    models = {
        "recorded": DATA_DIR,
        # "LN": Path("../runs/fCNN/036_linear_fCNN"),
        # "fCNN no behavior": Path("../runs/fCNN/030_fCNN_noBehavior"),
        # "fCNN": Path("../runs/fCNN/038_fCNN"),
        # "DwiseNeuro": Path("../runs/lRomul"),
        # "ViViT": Path("../runs/vivit/162_vivit"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }

    for model, output_dir in models.items():
        print(f"Estimate OSI and DSI for responses in {output_dir}.\n")
        meta_dir = data.METADATA_DIR if output_dir == DATA_DIR else output_dir
        for mouse_id in data.SENSORIUM_OLD:
            estimate_mouse(mouse_id=mouse_id, output_dir=output_dir, meta_dir=meta_dir)
        print(f"Saved results to {meta_dir}.\n")


if __name__ == "__main__":
    main()
