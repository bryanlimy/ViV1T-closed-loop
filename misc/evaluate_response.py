"""
Helper script to evaluate all predicted responses against recorded responses and
store results in YAML files. We compute metrics over dataset tiers (e.g. train,
validation, live_main, live_bonus, etc.) and over stimulus type (e.g. movie,
direction pink noise, drifting gabor, etc.).

Please run data/save_response.py to store all recorded responses into H5 files first.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from viv1t import data
from viv1t import metrics
from viv1t.data import get_tier_ids
from viv1t.data import get_video_ids
from viv1t.utils import h5
from viv1t.utils import yaml


def load_responses(
    response_dir: Path, mouse_id: str, trial_ids: np.ndarray | None = None
) -> list[torch.Tensor]:
    # load responses and predictions
    filename = response_dir / f"mouse{mouse_id}.h5"
    responses = h5.get(filename, trial_ids=trial_ids)
    responses = [torch.from_numpy(response) for response in responses]
    return responses


def compute_correlation(
    data_dir: Path,
    response_dir: Path,
    mouse_id: str,
    trial_ids: np.ndarray,
    tier: str = None,
) -> (dict[str, float], pd.DataFrame | None):
    y_true = load_responses(
        data_dir / "responses", mouse_id=mouse_id, trial_ids=trial_ids
    )
    y_pred = load_responses(response_dir, mouse_id=mouse_id, trial_ids=trial_ids)
    # compute single trial correlation used in Sensorium 2023
    challenge_correlation = metrics.challenge_correlation(
        y_true=y_true, y_pred=y_pred, per_neuron=True
    )
    # compute single trial correlation
    single_trial_correlation = metrics.single_trial_correlation(
        y_true=y_true, y_pred=y_pred, per_neuron=True
    )
    # compute normalized correlation
    video_ids = get_video_ids(mouse_id=mouse_id)[trial_ids]
    video_ids = torch.from_numpy(video_ids)
    normalized_correlation = metrics.normalized_correlation(
        y_true=y_true,
        y_pred=y_pred,
        video_ids=video_ids,
        per_neuron=True,
    )
    # compute correlation to average
    correlation_to_average = metrics.correlation_to_average(
        y_true=y_true,
        y_pred=y_pred,
        video_ids=video_ids,
        per_neuron=True,
    )
    population_result = {
        "correlation": torch.mean(challenge_correlation).item(),
        "single_trial_correlation": torch.mean(single_trial_correlation).item(),
        "normalized_correlation": torch.nanmean(normalized_correlation).item(),
        "correlation_to_average": torch.nanmean(correlation_to_average).item(),
    }
    neuron_result = None
    if tier is not None:
        num_neurons = len(challenge_correlation)
        neuron_result = pd.DataFrame(
            {
                "mouse": [mouse_id] * num_neurons,
                "neuron": list(range(num_neurons)),
                "tier": [tier] * num_neurons,
                "correlation": challenge_correlation.tolist(),
                "single_trial_correlation": single_trial_correlation.tolist(),
                "normalized_correlation": normalized_correlation.tolist(),
                "correlation_to_average": correlation_to_average.tolist(),
            }
        )
    return population_result, neuron_result


def compute_correlation_by_tier(
    args, response_dir: Path
) -> (dict[str, dict[str, dict[str, float]]], pd.DataFrame):
    population_results = {}
    neuron_results = []
    for tier in [
        "validation",
        "live_main",
        "live_bonus",
        "final_main",
        "final_bonus",
    ]:
        tier_result = {
            "correlation": {},
            "single_trial_correlation": {},
            "normalized_correlation": {},
            "correlation_to_average": {},
        }
        for mouse_id in tqdm(data.MOUSE_IDS.keys(), desc=tier):
            if not (response_dir / f"mouse{mouse_id}.h5").exists():
                continue
            tiers = get_tier_ids(data_dir=args.data_dir, mouse_id=mouse_id)
            trial_ids = np.where(tiers == tier)[0]
            if trial_ids.size == 0:
                continue
            population_result, neuron_result = compute_correlation(
                data_dir=args.data_dir,
                response_dir=response_dir,
                mouse_id=mouse_id,
                trial_ids=trial_ids,
                tier=tier,
            )
            for k, v in population_result.items():
                tier_result[k][mouse_id] = v
            neuron_results.append(neuron_result)
        # compute average result over animals
        for k, v in tier_result.items():
            tier_result[k]["average"] = np.mean(list(v.values())).item()
        if tier == "validation":
            print(
                f"Validation correlation: "
                f"{tier_result['correlation']['average']:.04f}\n"
            )
        population_results[tier] = tier_result
    neuron_results = pd.concat(neuron_results, ignore_index=True)
    return population_results, neuron_results


def compute_correlation_by_type(args, response_dir: Path):
    results = {}
    for stimulus_id, stimulus_name in data.STIMULUS_TYPES.items():
        type_result = {
            "correlation": {},
            "single_trial_correlation": {},
            "normalized_correlation": {},
            "correlation_to_average": {},
        }
        for mouse_id in tqdm(data.MOUSE_IDS.keys(), desc=f"{stimulus_name}"):
            if not (response_dir / f"mouse{mouse_id}.h5").exists():
                continue
            if stimulus_name == "movie":
                # jointly evaluate all test set natural movies
                tiers = get_tier_ids(data_dir=args.data_dir, mouse_id=mouse_id)
                trial_ids = np.where((tiers == "live_main") | (tiers == "final_main"))[
                    0
                ]
            else:
                stimulus_ids = data.get_stimulus_ids(mouse_id)
                trial_ids = np.where(stimulus_ids == stimulus_id)[0]
            if len(trial_ids) == 0:
                continue
            result, _ = compute_correlation(
                data_dir=args.data_dir,
                response_dir=response_dir,
                mouse_id=mouse_id,
                trial_ids=trial_ids,
            )
            for k, v in result.items():
                type_result[k][mouse_id] = v
            del result
        for k, v in type_result.items():
            type_result[k]["average"] = np.mean(list(v.values())).item()
        results[stimulus_name] = type_result
    return results


def main(args):
    response_dir = args.output_dir / "responses"
    print(f"Evaluate responses in {response_dir}.")

    # evaluate response based on dataset tier (e.g. train, validation, live_main, etc.)
    print("Evaluate response by dataset tier")
    results_by_tier, neuron_results_by_tier = compute_correlation_by_tier(
        args, response_dir=response_dir
    )
    yaml.save(args.output_dir / "evaluation_tier.yaml", results_by_tier)
    neuron_results_by_tier.to_parquet(
        args.output_dir / "neuron_performance.parquet", index=False
    )

    # evaluate response based on stimulus type (e.g. movie, directional pink noise, etc.)
    print("\nEvaluate response by stimulus type")
    results_by_type = compute_correlation_by_type(args, response_dir=response_dir)
    yaml.save(args.output_dir / "evaluation_type.yaml", results_by_type)

    print(f"Results saved to {args.output_dir}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../data")
    parser.add_argument("--output_dir", type=Path, required=True)
    main(parser.parse_args())
