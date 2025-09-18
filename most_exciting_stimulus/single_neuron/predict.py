"""
Given the most exciting natural center, find the most exciting natural surround.

To reduce the number of computation cost, we only focus on the top 30 reliable
neurons from each mouse.
"""

import argparse
import logging
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from viv1t import data
from viv1t.data.data import MovieDataset
from viv1t.model import Model
from viv1t.utils import plot
from viv1t.utils import stimulus
from viv1t.utils import utils
from viv1t.utils.load_model import load_model
from viv1t.utils.plot import animate_stimulus

plot.set_font()

PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2
PAD = 20

VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) // 2

RESPONSE_TYPES = Literal["most_exciting", "most_inhibiting"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

BLANK = torch.full((1, BLANK_SIZE, VIDEO_H, VIDEO_W), fill_value=GREY_COLOR)


@torch.inference_mode()
def inference(
    model: Model,
    video: torch.Tensor,
    mouse_id: str,
    neuron: int,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
    dataset: MovieDataset,
):
    start = -(BLANK_SIZE + PATTERN_SIZE + PAD)
    end = -(BLANK_SIZE - PAD)
    to_batch = lambda a: a.to(model.device, model.dtype)[None, ...]
    video = torch.cat([BLANK, video, BLANK], dim=1)
    response, _ = model(
        inputs=to_batch(dataset.transform_video(video)),
        mouse_id=mouse_id,
        behaviors=to_batch(behavior),
        pupil_centers=to_batch(pupil_center),
    )
    response = response.to("cpu", torch.float32)
    response = dataset.i_transform_response(response)
    response = response[0, neuron, start:end]
    video = video[:, start:end, :, :]
    return response, video


def load_video(
    data_dir: Path,
    mouse_id: str,
    trial_id: int,
    frame: int,
    static: bool,
) -> torch.Tensor:
    video = np.load(
        data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    if static:
        video = repeat(video[:, :, frame], "h w -> h w t", t=PATTERN_SIZE)
    else:
        video = video[:, :, frame : frame + PATTERN_SIZE]
    video = rearrange(video, "h w t -> () t h w")
    video = torch.from_numpy(video)
    video = video.to(torch.float32)
    return video


def process_neuron(
    args,
    mouse_id: str,
    neuron: int,
    model: Model,
    dataset: MovieDataset,
    save_dir: Path,
):
    behavior, pupil_center = data.get_mean_behaviors(
        mouse_id, num_frames=data.MAX_FRAME
    )
    behavior = dataset.transform_behavior(behavior)
    pupil_center = dataset.transform_pupil_center(pupil_center)

    monitor_info = data.MONITOR_INFO[mouse_id]
    circular_mask_kw = {
        "center": stimulus.load_neuron_RF_center(
            output_dir=args.output_dir,
            mouse_id=mouse_id,
            neuron=neuron,
        ),
        "pixel_width": VIDEO_W,
        "pixel_height": VIDEO_H,
        "monitor_width": monitor_info["width"],
        "monitor_height": monitor_info["height"],
        "monitor_distance": monitor_info["distance"],
        "num_frames": PATTERN_SIZE,
        "to_tensor": True,
    }
    circular_mask = stimulus.create_circular_mask(
        stimulus_size=args.stimulus_size,
        **circular_mask_kw,
    )
    outer_circular_mask = None
    if args.outer_stimulus_size is not None:
        outer_circular_mask = stimulus.create_circular_mask(
            stimulus_size=args.outer_stimulus_size,
            **circular_mask_kw,
        )

    # natural center
    dir = save_dir / "center" / "static"
    df = pd.read_parquet(dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet")
    natural_center = load_video(
        data_dir=args.data_dir,
        mouse_id=mouse_id,
        trial_id=df.iloc[0].trial,
        frame=df.iloc[0].frame,
        static=True,
    )
    video = torch.where(circular_mask, natural_center, GREY_COLOR)
    response, video = inference(
        model=model,
        video=video,
        mouse_id=mouse_id,
        neuron=neuron,
        behavior=behavior,
        pupil_center=pupil_center,
        dataset=dataset,
    )
    new_df = pd.DataFrame(
        {
            "mouse": [mouse_id],
            "neuron": [neuron],
            "trial": [df.iloc[0].trial.item()],
            "frame": [df.iloc[0].frame.item()],
            "response": [torch.sum(response[PAD:-PAD]).item()],
            "raw_response": [response.tolist()],
            "stimulus": [rearrange(video, "c t h w -> (c t h w)").tolist()],
            "stimulus_type": [df.iloc[0].stim_type],
        }
    )
    dir = dir / "new"
    dir.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet")

    for stimulus_type in ["static", "dynamic"]:
        new_df = []
        for response_type in ["most_exciting", "most_inhibiting"]:
            dir = (
                save_dir / "center_surround" / f"static_center_{stimulus_type}_surround"
            )
            df = pd.read_parquet(dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet")
            df = df[df.response_type == response_type]
            assert len(df) == 1
            natural_surround = load_video(
                data_dir=args.data_dir,
                mouse_id=mouse_id,
                trial_id=df.iloc[0].trial,
                frame=df.iloc[0].frame,
                static=stimulus_type == "static",
            )
            video = torch.where(circular_mask, natural_center, natural_surround)
            if outer_circular_mask is not None:
                video = torch.where(outer_circular_mask, video, GREY_COLOR)
            response, video = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            new_df.append(
                pd.DataFrame(
                    {
                        "mouse": [mouse_id],
                        "neuron": [neuron],
                        "trial": [df.iloc[0].trial.item()],
                        "frame": [df.iloc[0].frame.item()],
                        "response": [torch.sum(response[PAD:-PAD]).item()],
                        "raw_response": [response.tolist()],
                        "stimulus": [rearrange(video, "c t h w -> (c t h w)").tolist()],
                        "stimulus_type": [stimulus_type],
                        "response_type": [response_type],
                    }
                )
            )
        new_df = pd.concat(new_df, ignore_index=True)
        dir = dir / "new"
        dir.mkdir(parents=True, exist_ok=True)
        new_df.to_parquet(dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet")


def process_mouse(
    args,
    mouse_id: str,
    model: Model,
    dataset: MovieDataset,
    save_dir: Path,
):
    reliable_neurons = sorted(
        utils.get_reliable_neurons(
            output_dir=args.output_dir,
            mouse_id=mouse_id,
        )
    )
    for neuron in tqdm(reliable_neurons):
        process_neuron(
            args,
            mouse_id=mouse_id,
            neuron=neuron,
            model=model,
            dataset=dataset,
            save_dir=save_dir,
        )


def main(args):
    utils.set_random_seed(args.seed)
    args.device = utils.get_device(args.device)
    model, ds = load_model(args, evaluate=args.evaluate, compile=args.compile)
    model.train(False)

    save_dir = args.output_dir / "most_exciting_stimulus" / "single_neuron" / "natural"

    mouse_id = args.mouse_ids[0]
    process_mouse(
        args,
        mouse_id=mouse_id,
        model=model,
        dataset=ds[mouse_id].dataset,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default="../../data")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the model on the validation set after loading the checkpoint.",
    )
    parser.add_argument(
        "--stimulus_size",
        type=int,
        default=20,
        help="stimulus size to use if use_preferred_size is not set.",
    )
    parser.add_argument(
        "--outer_stimulus_size",
        type=int,
        default=None,
        help="outer stimulus size to mask out far-away pixels",
    )
    parser.add_argument(
        "--use_RF_center",
        action="store_true",
        help="Use the estimate aRF center as the position of the circular center mask.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "use the best available device if --device is not specified.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "bf16"],
        default="32",
        help="Precision to use for inference, both model weights and input "
        "data would be converted.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
