import argparse
import logging
from argparse import RawTextHelpFormatter
from itertools import product
from pathlib import Path
from typing import Any

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

SKIP = 50  # skip the first 50 frames from each trial


VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2
FPS = 30

PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2

# Grating parameters
CPD = 0.04  # spatial frequency in cycle per degree
CPF = 2 / FPS  # temporal frequency of 2Hz in cycle per frame
CONTRAST = 1
PHASE = 0

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class GratingDataset(Dataset):
    def __init__(
        self,
        args,
        mouse_id: str,
        dataset: MovieDataset,
        device: torch.device,
    ):
        super(GratingDataset, self).__init__()
        self.mouse_id = mouse_id
        self.num_neurons = dataset.num_neurons
        self.num_repeats = args.num_repeats

        rng = np.random.default_rng(args.seed)

        self.directions = np.array(list(data.DIRECTIONS.keys()), dtype=np.float32)
        # create repeats
        self.directions = repeat(self.directions, "d -> (r d)", r=self.num_repeats)
        self.gratings = torch.stack(
            [
                stimulus.create_full_field_grating(
                    direction=direction,
                    cpd=CPD,
                    cpf=CPF,
                    num_frames=PATTERN_SIZE,
                    height=VIDEO_H,
                    width=VIDEO_W,
                    phase=rng.choice(360, size=1).item(),  # randomize initial phase,
                    contrast=CONTRAST,
                    fps=FPS,
                    to_tensor=True,
                )
                for direction in self.directions
            ]
        )

        # shuffle trial order
        trial_ids = np.arange(len(self.directions), dtype=int)
        trial_ids = rng.permutation(trial_ids)
        self.directions = self.directions[trial_ids]
        self.gratings = self.gratings[trial_ids]

        self.transform_video = dataset.transform_video

        # load response statistics to device for quicker inference
        self.transform_output = dataset.transform_output
        self.response_stats = {
            k: v.to(device) for k, v in dataset.response_stats.items()
        }
        self.response_precision = dataset.response_precision.to(device)

        behavior, pupil_center = data.get_mean_behaviors(
            mouse_id, num_frames=data.MAX_FRAME
        )
        self.behavior = dataset.transform_behavior(behavior)
        self.pupil_center = dataset.transform_pupil_center(pupil_center)

        self.blank = torch.full(
            (1, BLANK_SIZE, VIDEO_H, VIDEO_W), fill_value=GREY_COLOR
        )

    def __len__(self) -> int:
        return len(self.directions)

    def i_transform_response(self, response: torch.Tensor) -> torch.Tensor:
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response / self.response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    def __getitem__(
        self, idx: int | torch.Tensor, to_tensor: bool = True
    ) -> dict[str, Any]:
        direction = self.directions[idx]
        video = torch.cat([self.blank, self.gratings[idx].clone(), self.blank], dim=1)
        return {
            "video": self.transform_video(video),
            "mouse_id": self.mouse_id,
            "behavior": self.behavior,
            "pupil_center": self.pupil_center,
            "raw_video": video,
            "direction": direction,
        }


@torch.inference_mode()
def inference(
    model: Model,
    mouse_id: str,
    ds: DataLoader,
    save_dir: Path,
    animate: bool = False,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    device, dtype = model.device, model.dtype
    num_batches, num_samples = len(ds), len(ds.dataset)
    num_neurons = ds.dataset.num_neurons

    # configuration for plot animation
    pad = 10
    start = -(BLANK_SIZE + PATTERN_SIZE + pad)
    end = -(BLANK_SIZE - pad)
    mask = np.concat([np.zeros(pad), np.ones(PATTERN_SIZE), np.zeros(pad)])
    make_plot = np.random.choice(num_samples, size=min(2, num_samples), replace=False)

    directions = torch.zeros(num_samples, dtype=torch.int32)
    responses = torch.zeros(num_samples, num_neurons, dtype=torch.int32)
    index = 0
    for i, batch in enumerate(tqdm(ds, desc=f"Mouse {mouse_id}")):
        response, _ = model(
            inputs=batch["video"].to(device, dtype),
            mouse_id=mouse_id,
            behaviors=batch["behavior"].to(device, dtype),
            pupil_centers=batch["pupil_center"].to(device, dtype),
        )
        response = response.to(torch.float32)
        response = ds.dataset.i_transform_response(response)
        if animate and i in make_plot:  # plot 10% of the trials
            animate_stimulus(
                video=batch["raw_video"][0, :, start:end, :, :].numpy(),
                response=response[0, :, start:end].cpu().numpy(),
                filename=save_dir
                / "figures"
                / f"mouse{mouse_id}"
                / f"mouse{mouse_id}_{batch['direction'][0]}.mp4",
                presentation_mask=mask,
                skip=0,
            )
        # get neuron response within the stimulus presentation window then
        # round response to nearest integer to save space
        response = response[:, :, -(BLANK_SIZE + PATTERN_SIZE) : -BLANK_SIZE]
        response = torch.round(torch.sum(response, dim=-1), decimals=0)
        batch_size = len(response)
        responses[index : index + batch_size] = response.to("cpu", torch.int32)
        directions[index : index + batch_size] = batch["direction"].to(torch.int32)
        index += batch_size
    return responses, directions


def record_response(
    mouse_id: str, responses: torch.Tensor, directions: torch.Tensor
) -> pd.DataFrame:
    unique_directions = torch.unique(directions)
    num_directions = len(unique_directions)
    num_neurons = responses.shape[1]
    neurons = np.arange(num_neurons, dtype=int)
    responses = responses.to(torch.float32)
    # group responses by direction and average response over repeats
    responses_ = torch.zeros((num_neurons, num_directions), dtype=torch.int32)
    for i, direction in enumerate(unique_directions):
        trial_ids = torch.where(directions == direction)
        response = torch.mean(responses[trial_ids], dim=0)
        responses_[:, i] = response
    # get most-exciting response, trial and frame
    max_responses, preferred_directions = torch.max(responses_, dim=1)
    preferred_directions = unique_directions[preferred_directions]
    most_exciting = pd.DataFrame(
        {
            "neuron": neurons,
            "direction": preferred_directions,
            "response": max_responses,
        }
    )
    most_exciting.insert(loc=1, column="response_type", value="most_exciting")
    # get most-inhibiting response, trial and frame
    min_responses, unpreferred_directions = torch.min(responses_, dim=1)
    unpreferred_directions = unique_directions[unpreferred_directions]
    most_inhibiting = pd.DataFrame(
        {
            "neuron": neurons,
            "direction": unpreferred_directions,
            "response": min_responses,
        }
    )
    most_inhibiting.insert(loc=1, column="response_type", value="most_inhibiting")
    df = pd.concat([most_exciting, most_inhibiting], ignore_index=True)
    df.insert(loc=0, column="mouse", value=mouse_id)
    df.insert(loc=2, column="stim_type", value="dynamic")
    return df


def process_mouse(
    args,
    mouse_id: str,
    model: Model,
    dataset: MovieDataset,
    save_dir: Path,
):
    ds = DataLoader(
        dataset=GratingDataset(
            args,
            mouse_id=mouse_id,
            dataset=dataset,
            device=args.device,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    responses, directions = inference(
        model=model,
        mouse_id=mouse_id,
        ds=ds,
        save_dir=save_dir,
        animate=args.animate,
    )
    df = record_response(
        mouse_id=mouse_id,
        responses=responses,
        directions=directions,
    )
    filename = save_dir / "responses" / f"mouse{mouse_id}.parquet"
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filename)


def main(args):
    utils.set_random_seed(args.seed)
    args.device = utils.get_device(args.device)
    model, ds = load_model(args, evaluate=args.evaluate, compile=args.compile)
    model.train(False)

    save_dir = (
        args.output_dir
        / "most_exciting_stimulus"
        / "single_neuron"
        / "gratings"
        / "full_field"
        / "dynamic"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    utils.set_logger_handles(
        logger=logger,
        filename=save_dir / f"output-{utils.get_timestamp()}.log",
        level=logging.INFO,
    )

    for mouse_id in data.SENSORIUM_OLD:
        process_mouse(
            args,
            model=model,
            mouse_id=mouse_id,
            dataset=ds[mouse_id].dataset,
            save_dir=save_dir,
        )

    print(f"Saved result to {save_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default="../../data")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the model on the validation set after loading the checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_repeats", type=int, default=10)
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
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
