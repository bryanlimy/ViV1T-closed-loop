"""
Helper script to save all responses to h5 files for quicker loading and
metric calculations.

All responses are cropped to have a maximum of 300 frames and skip the first
50 frames, to match the Sensorium 2023 challenge metric calculation.
"""

import argparse
from pathlib import Path

import numpy as np
from sensorium.data import MOUSE_IDS
from sensorium.data import ROCHEFORT_LAB
from sensorium.data import SENSORIUM_OLD
from sensorium.data import get_tier_ids
from sensorium.data import load_trial
from sensorium.utils import h5
from tqdm import tqdm

SKIP = 50
MAX_FRAME = 300

MICE_WITH_TEST_LABELS = SENSORIUM_OLD + ROCHEFORT_LAB


def save_responses(mouse_id: str, tier: str, data_dir: Path):
    mouse_dir = data_dir / MOUSE_IDS[mouse_id]
    if not mouse_dir.is_dir():
        return
    tiers = get_tier_ids(data_dir=data_dir, mouse_id=mouse_id)
    if tier not in tiers:
        return
    trial_ids = np.where(tiers == tier)[0]
    filename = data_dir / "responses" / f"mouse{mouse_id}.h5"
    responses = {}
    for trial_id in tqdm(trial_ids, desc=f"mouse {mouse_id} {tier}"):
        sample = load_trial(mouse_dir, trial_id=trial_id, to_tensor=False)
        response = sample["response"][:, :MAX_FRAME]
        t = response.shape[1] - SKIP
        response = response[:, -t:]
        responses[trial_id] = response
    h5.write(filename, data=responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    for mouse_id in MOUSE_IDS.keys():
        for tier in (
            "train",
            "validation",
            "live_main",
            "live_bonus",
            "final_main",
            "final_bonus",
        ):
            if mouse_id not in MICE_WITH_TEST_LABELS and tier != "validation":
                continue
            save_responses(mouse_id, tier, data_dir=args.data_dir)
