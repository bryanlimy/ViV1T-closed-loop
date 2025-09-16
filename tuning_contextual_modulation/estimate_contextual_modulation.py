from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from einops import rearrange
from scipy.stats import wilcoxon

from viv1t import data

BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE


CONTRAST_TYPES = Literal["high_contrast", "low_contrast"]


def compute_p_value(responses1: np.ndarray, responses2: np.ndarray):
    p_value = wilcoxon(responses1, responses2).pvalue
    if p_value <= 0.001:
        text = "***"
    elif p_value <= 0.01:
        text = "**"
    elif p_value <= 0.05:
        text = "*"
    else:
        text = "n.s."
    return text


def get_stim_type(stim_type: int):
    match stim_type:
        case 0:
            return "center"
        case 1:
            return "iso"
        case 2:
            return "cross"
        case 3:
            return "shift"


def compute_pattern_average(responses: np.ndarray):
    """
    Compute the average response to each block pattern so that, for each pattern,
    the response of a neuron is represented by a single value.
    """
    # remove blank frames
    responses = responses[:, :, :, BLANK_SIZE : BLANK_SIZE + PATTERN_SIZE]
    # average response over frames for each pattern
    responses = np.mean(responses, axis=-1)
    return responses


def reshape_responses(responses: np.ndarray, parameters: np.ndarray):
    """
    Reshape responses to (neuron, contrast type, stimulus size, stimulus type) and
    compute the average over repeated presentations of the same pattern.
    """
    responses = rearrange(responses, "neuron block pattern -> neuron (block pattern)")
    parameters = rearrange(parameters, "block pattern param -> (block pattern) param")

    contrasts = np.unique(parameters[:, 0])
    stim_sizes = np.unique(parameters[:, 1])
    stim_types = np.unique(parameters[:, 3])

    outputs = np.zeros(
        (responses.shape[0], len(contrasts), len(stim_sizes), len(stim_types)),
        dtype=np.float32,
    )
    for i, contrast in enumerate(contrasts):
        for j, stim_size in enumerate(stim_sizes):
            for k, stim_type in enumerate(stim_types):
                idx = np.where(
                    (parameters[:, 0] == contrast)
                    & (parameters[:, 1] == stim_size)
                    & (parameters[:, 3] == stim_type)
                )[0]
                # compute average response over repeats
                outputs[:, i, j, k] = np.mean(responses[:, idx], axis=-1)
    return outputs


def load_data(
    mouse_id: str, output_dir: Path, contrast_type: CONTRAST_TYPES
) -> (np.ndarray, np.ndarray):
    save_dir = output_dir / "contextual_modulation"
    responses = np.load(save_dir / f"mouse{mouse_id}.npz", allow_pickle=False)["data"]
    parameters = np.load(save_dir / "parameters.npy", allow_pickle=False)

    responses = compute_pattern_average(responses)
    responses = reshape_responses(responses, parameters)

    # select response to stimulus size 20°
    assert len(np.unique(parameters[..., 0])) == 2
    stimulus_sizes = np.unique(parameters[..., 1]).tolist()
    stimulus_size = 20
    stimulus_index = stimulus_sizes.index(stimulus_size)

    # select L2/3 neurons that have preferred stimulus size within 10° of the
    # presented stimulus
    df = pd.read_parquet(output_dir / "size_tuning_preference.parquet")
    neurons = df[
        (df.mouse == mouse_id)
        # & (df.classic_tuned == True)
        & (
            df.classic_preference.isin(
                [stimulus_size - 10, stimulus_size, stimulus_size + 10]
            )
        )
        & (df.depth >= 100)
        & (df.depth <= 300)
    ].neuron.values
    # neurons = np.where(~np.isnan(responses[:, 0, 0, 0]))[0]

    match contrast_type:
        case "low_contrast":
            responses = responses[neurons, 0, stimulus_index, :]
        case "high_contrast":
            responses = responses[neurons, 1, stimulus_index, :]
        case _:
            raise NotImplementedError(f"Unknown contrast type {contrast_type}")
    return responses, neurons


def compute_contextual_modulation_index(
    iso: np.ndarray, cross: np.ndarray
) -> np.ndarray:
    return (cross - iso) / (cross + iso)


def process_model(model_name: str, output_dir: Path):
    print(f"\nEstimate {model_name} contextual modulation in {output_dir}.")
    df = []
    for contrast_type in ["high_contrast", "low_contrast"]:
        print(f"Process {contrast_type}...")
        for mouse_id in data.MOUSE_IDS.keys():
            filename = output_dir / "contextual_modulation" / f"mouse{mouse_id}.npz"
            if not filename.exists():
                continue
            print(f"\tProcessing mouse {mouse_id}")
            response, neurons = load_data(
                mouse_id=mouse_id,
                output_dir=output_dir,
                contrast_type=contrast_type,
            )
            df.append(
                pd.DataFrame(
                    {
                        "mouse": [mouse_id] * len(neurons),
                        "contrast_type": [contrast_type] * len(neurons),
                        "neuron": neurons,
                        "center": response[:, 0],
                        "iso": response[:, 1],
                        "cross": response[:, 2],
                        "shift": response[:, 3],
                        "cmi": compute_contextual_modulation_index(
                            iso=response[:, 1], cross=response[:, 2]
                        ),
                    }
                )
            )
    df = pd.concat(df, ignore_index=True)
    filename = output_dir / "contextual_modulation.parquet"
    df.to_parquet(filename)
    print(f"Saved contextual modulation results to {filename}.")


def main():
    models = {
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
        # "ViV1T_rochefort": Path(
        #     "../runs/rochefort-lab/vivit/003_causal_viv1t_finetune"
        # ),
    }
    for model_name, output_dir in models.items():
        process_model(model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
