import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from einops import rearrange
from scipy.stats import rankdata
from tqdm import tqdm

from viv1t import data
from viv1t import metrics
from viv1t.utils import h5


def load_responses(filename: Path, trial_ids: np.ndarray) -> np.ndarray:
    responses = h5.get(filename, trial_ids=trial_ids)
    responses = np.stack(responses)
    return responses


def compute_trial_to_trial_reliability(data_dir: Path, mouse_id: str) -> np.ndarray:
    # load recorded responses from validation set
    tiers = data.get_tier_ids(data_dir=data_dir, mouse_id=mouse_id)
    trial_ids = np.where(tiers == "validation")[0]
    responses = load_responses(
        filename=data_dir / "responses" / f"mouse{mouse_id}.h5",
        trial_ids=trial_ids,
    )
    num_neurons = responses.shape[1]
    correlations = []
    # group responses by unique videos and their repeats
    video_ids = data.get_video_ids(mouse_id=mouse_id)[trial_ids]
    for video_id in tqdm(np.unique(video_ids), desc="Trial reliability"):
        idx = np.where(video_ids == video_id)[0]
        num_repeats = len(idx)
        if num_repeats < 5:
            raise ValueError(f"Only {num_repeats} repeats for video ID {video_id}.")
        triu = np.triu_indices(num_repeats, k=1)
        response = responses[idx]
        # compute pairwise correlation of each repeat and average across repeat
        corr = np.stack(
            [
                np.nanmean(np.corrcoef(response[:, n, :], dtype=np.float32)[triu])
                for n in range(num_neurons)
            ]
        )
        correlations.append(corr)
    correlations = np.stack(correlations)
    # average correlation over unique video
    correlations = np.mean(correlations, axis=0)
    return correlations


def prediction_performance(
    data_dir: Path, output_dir: Path, mouse_id: str
) -> np.ndarray:
    """
    Compute the correlation between the average (over repeat) recorded
    and predicted responses
    """
    # load recorded and predicted responses from validation set
    tiers = data.get_tier_ids(data_dir=data_dir, mouse_id=mouse_id)
    trial_ids = np.where(tiers == "validation")[0]
    recorded_responses = load_responses(
        filename=data_dir / "responses" / f"mouse{mouse_id}.h5",
        trial_ids=trial_ids,
    )
    predicted_responses = load_responses(
        filename=output_dir / "responses" / f"mouse{mouse_id}.h5",
        trial_ids=trial_ids,
    )
    correlation = []
    # group responses by unique videos and their repeats
    video_ids = data.get_video_ids(mouse_id=mouse_id)[trial_ids]
    for video_id in tqdm(np.unique(video_ids), desc="Prediction performance"):
        idx = np.where(video_ids == video_id)[0]
        num_repeats = len(idx)
        if num_repeats < 5:
            raise ValueError(f"Only {num_repeats} repeats for video ID {video_id}.")
        # average response over repeated presentations
        y_true = np.mean(recorded_responses[idx], axis=0)
        y_pred = np.mean(predicted_responses[idx], axis=0)
        correlation.append(metrics.correlation(y1=y_true, y2=y_pred, dim=1))
    correlation = np.stack(correlation)
    # average correlation over unique videos
    correlation = np.mean(correlation, axis=0)
    return correlation


def rank_neuron_reliability(
    output_dir: Path,
    mouse_id: str,
    trial_reliability: np.ndarray,
    correlation: np.ndarray,
) -> np.ndarray:
    """
    Rank neuron reliability with the 3 criteria:
    - trial to trial reliability in the recorded responses. i.e. the
        pairwise correlation of recorded response to the same stimuli with
        multiple repeated presentation.
    - prediction performance of the neuron, measured in correlation between
        predicted and recorded responses averaged over repeats.
    - the predicted neuron has a good aRF fit. See tuning_retinotopy/README.md for more.
    We select the top percent% neurons in each criterion and return the intersection
    of the 3 subset of neurons.
    """
    # get neurons with bad aRF fits
    aRFs = pd.read_parquet(output_dir / "aRF.parquet")
    bad_neurons = aRFs[(aRFs.mouse == mouse_id) & (aRFs.bad_fit == True)].neuron.values
    # scipy.stats.rankdata sort in ascending order hence the negative sign
    trial_reliability_rank = rankdata(-trial_reliability, method="ordinal")
    correlation_rank = rankdata(-correlation, method="ordinal")
    joint_rank = trial_reliability_rank + correlation_rank
    # remove neurons that do not have good aRF fit or not size tuned
    joint_rank = joint_rank.astype(np.float32)
    joint_rank[bad_neurons] = np.nan
    joint_rank = rankdata(joint_rank, method="ordinal", nan_policy="omit")
    return joint_rank


def main(args):
    df = []
    for mouse_id in data.MOUSE_IDS.keys():
        if not (args.output_dir / "responses" / f"mouse{mouse_id}.h5").exists():
            continue
        print(f"\nProcessing mouse {mouse_id}...")
        trial_reliability = compute_trial_to_trial_reliability(
            data_dir=args.data_dir, mouse_id=mouse_id
        )
        correlation = prediction_performance(
            data_dir=args.data_dir, output_dir=args.output_dir, mouse_id=mouse_id
        )
        num_neurons = len(trial_reliability)
        rank = rank_neuron_reliability(
            output_dir=args.output_dir,
            mouse_id=mouse_id,
            trial_reliability=trial_reliability,
            correlation=correlation,
        )
        df.append(
            pd.DataFrame(
                {
                    "mouse": [mouse_id] * num_neurons,
                    "neuron": np.arange(num_neurons),
                    "rank": rank,
                    "trial_reliability": trial_reliability,
                    "correlation": correlation,
                }
            )
        )
    df = pd.concat(df, ignore_index=True)
    filename = args.output_dir / "neuron_reliability.parquet"
    df.to_parquet(filename)
    print(f"Saved result to {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../data")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    main(parser.parse_args())
