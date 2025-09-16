import argparse
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from viv1t import data
from viv1t.model import Model
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils
from viv1t.utils.attention_rollout import fold_spatial_attention

plot.set_font()
utils.set_random_seed(1234)


SKIP = 50  # number of frames to skip for metric calculation
DPI = 240
MAX_FRAMES = 300


DATA_DIR = Path("../data/sensorium")


def write_h5(dir: Path, data: Dict[str, np.ndarray]):
    trial_ids = data.pop("trial_ids", None)
    for trial_id in trial_ids:
        for k, v in data.items():
            h5.write(dir / f"{k}.h5", data={int(trial_id): v})


def load_h5(
    dir: Path, trial_ids: List[int] | np.ndarray
) -> Dict[str, List[np.ndarray]]:
    results = {
        "videos": [],
        "pupil_centers": [],
        "behaviors": [],
        "spatial_attentions": [],
        "temporal_attentions": [],
    }
    for k in results.keys():
        results[k] = h5.get(dir / f"{k}.h5", trial_ids=trial_ids)
    results["trial_ids"] = trial_ids
    return results


def main(args):
    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM)

    args.device = torch.device("cpu")
    utils.load_args(args)
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: data.get_neuron_coordinates(DATA_DIR, mouse_id, to_tensor=True)
            for mouse_id in args.mouse_ids
        },
    )
    patch_size = model.core.tokenizer.kernel_size[1]
    stride = model.core.tokenizer.stride[1]
    padding = model.core.tokenizer.padding

    # file to store the core attention matrix
    attention_dir = args.output_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = args.output_dir / "plots" / "average_attention_rollout"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for mouse_id in args.mouse_ids:
        print(f"Plot attention matrix from mouse {mouse_id}")
        cache_dir = attention_dir / f"mouse{mouse_id}"
        tiers = data.get_tier_ids(DATA_DIR, mouse_id=mouse_id)
        for tier in [
            "validation",
            "live_main",
            "live_bonus",
            "final_main",
            "final_bonus",
        ]:
            trial_ids = np.where(tiers == tier)[0]
            video_ids = data.get_video_ids(mouse_id)
            for video_id in tqdm(np.unique(video_ids[trial_ids]), desc=tier):
                # trial_ids with the same video_id
                trial_ids = np.where(video_ids == video_id)[0]
                result = load_h5(cache_dir, trial_ids)

                result["spatial_attentions"] = [
                    fold_spatial_attention(
                        result["spatial_attentions"][i],
                        frame_size=result["videos"][i].shape[2:],
                        patch_size=patch_size,
                        stride=stride,
                        padding=padding,
                    )
                    for i in range(len(result["spatial_attentions"]))
                ]

                # take the first sample of video and behavior variables, and compute
                # average spatial and temporal attention over repeated trials
                sample = {
                    "video": result["videos"][0],
                    "behavior": result["behaviors"][0],
                    "pupil_center": result["pupil_centers"][0],
                    "spatial_attention": np.mean(result["spatial_attentions"], axis=0),
                    "temporal_attention": np.mean(
                        result["temporal_attentions"], axis=0
                    ),
                }
                # compute average spatial and temporal attention over repeated trials
                plot.animate_attention_map(
                    sample=sample,
                    filename=plot_dir
                    / f"mouse{mouse_id}"
                    / tier
                    / f"trial{trial_ids[0]:03d}_average.{args.format}",
                    spatial_title='average spatial "attention"',
                    temporal_title='average temporal "attention"',
                )

    print(f"Saved figures to {plot_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    main(parser.parse_args())
