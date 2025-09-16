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
PAD = 20

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
        neuron: int,
        dataset: MovieDataset,
        device: torch.device,
    ):
        super(GratingDataset, self).__init__()
        self.mouse_id = mouse_id
        self.num_neurons = dataset.num_neurons

        # create center gratings with 8 directions, then all combinations of center
        # and surround gratings with 8 directions.
        directions = list(data.DIRECTIONS.keys())
        # parameters for center gratings with no surround gratings
        parameters = list(product(directions, [-1]))
        # parameters for center and surround gratings
        parameters += list(product(directions, directions))
        self.parameters = np.array(parameters, dtype=np.float32)

        # create full field gratings for fasting center surround grating construction
        self.full_field_gratings = {
            direction: stimulus.create_full_field_grating(
                direction=direction,
                cpd=CPD,
                cpf=CPF,
                num_frames=PATTERN_SIZE,
                height=VIDEO_H,
                width=VIDEO_W,
                phase=PHASE,
                contrast=CONTRAST,
                fps=FPS,
                to_tensor=True,
            )
            for direction in data.DIRECTIONS.keys()
        }
        # add direction -1 for no grating
        self.full_field_gratings[-1] = torch.full_like(
            list(self.full_field_gratings.values())[0], GREY_COLOR
        )

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

        if args.use_preferred_size:
            args.stimulus_size = utils.get_size_tuning_preference(
                output_dir=args.output_dir, mouse_id=mouse_id, neuron=neuron
            )
        self.circular_mask = stimulus.create_circular_mask(
            stimulus_size=args.stimulus_size,
            **circular_mask_kw,
        )
        self.outer_circular_mask = None
        if args.outer_stimulus_size is not None:
            self.outer_circular_mask = stimulus.create_circular_mask(
                stimulus_size=args.outer_stimulus_size,
                **circular_mask_kw,
            )

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
        return len(self.parameters)

    def i_transform_response(self, response: torch.Tensor) -> torch.Tensor:
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response / self.response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    def create_video(
        self,
        center_direction: np.ndarray,
        surround_direction: np.ndarray,
    ) -> torch.tensor:
        center_grating = self.full_field_gratings[center_direction].clone()
        surround_grating = self.full_field_gratings[surround_direction].clone()
        video = torch.where(self.circular_mask, center_grating, surround_grating)
        if self.outer_circular_mask is not None:
            video = torch.where(self.outer_circular_mask, video, GREY_COLOR)
        video = torch.cat([self.blank, video, self.blank], dim=1)
        return video

    def __getitem__(
        self, idx: int | torch.Tensor, to_tensor: bool = True
    ) -> dict[str, Any]:
        center_direction, surround_direction = self.parameters[idx]
        video = self.create_video(
            center_direction=center_direction,
            surround_direction=surround_direction,
        )
        return {
            "video": self.transform_video(video),
            "mouse_id": self.mouse_id,
            "behavior": self.behavior,
            "pupil_center": self.pupil_center,
            "raw_video": video,
            "center_direction": center_direction,
            "surround_direction": surround_direction,
        }


@torch.inference_mode()
def inference(
    model: Model,
    mouse_id: str,
    neuron: int,
    ds: DataLoader,
    save_dir: Path,
    animate: bool = False,
):
    device, dtype = model.device, model.dtype
    num_batches, num_samples = len(ds), len(ds.dataset)

    start = -(BLANK_SIZE + PATTERN_SIZE + PAD)
    end = -(BLANK_SIZE - PAD)
    mask = np.concat([np.zeros(PAD), np.ones(PATTERN_SIZE), np.zeros(PAD)])
    make_plot = np.random.choice(num_samples, size=min(2, num_samples), replace=False)

    center_directions = torch.zeros(num_samples, dtype=torch.int32)
    surround_directions = torch.zeros(num_samples, dtype=torch.int32)
    responses = torch.zeros(
        size=(num_samples, (PAD + PATTERN_SIZE + PAD)), dtype=torch.float32
    )
    visual_stimuli = torch.zeros(
        size=(num_samples, 1, (PAD + PATTERN_SIZE + PAD), VIDEO_H, VIDEO_W),
        device="cpu",
        dtype=torch.int32,
    )
    index = 0
    for i, batch in enumerate(tqdm(ds, desc=f"Mouse {mouse_id} neuron {neuron:04d}")):
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
                response=response[0, neuron, start:end].cpu().numpy(),
                neuron=neuron,
                filename=save_dir
                / "figures"
                / f"mouse{mouse_id}"
                / f"mouse{mouse_id}_neuron{neuron:04d}_center{batch['center_direction'][0]}_surround{batch['surround_direction'][0]}.mp4",
                presentation_mask=mask,
                skip=0,
            )
        batch_size = response.shape[0]
        # get neuron response within the stimulus presentation window then
        # round response to nearest integer to save space
        response = response[:, neuron, start:end]
        responses[index : index + batch_size] = response.to("cpu")
        center_directions[index : index + batch_size] = batch["center_direction"].to(
            torch.int32
        )
        surround_directions[index : index + batch_size] = batch[
            "surround_direction"
        ].to(torch.int32)
        visual_stimuli[index : index + batch_size] = batch["raw_video"][
            :, :, start:end, :, :
        ]
        index += batch_size
    return responses, center_directions, surround_directions, visual_stimuli


def process_mouse(
    args,
    mouse_id: str,
    model: Model,
    dataset: MovieDataset,
    save_dir: Path,
):
    # find the most exciting natural surround given the most exciting natural center
    # for the reliable neurons
    reliable_neurons = sorted(
        utils.get_reliable_neurons(
            output_dir=args.output_dir,
            mouse_id=mouse_id,
        )
    )
    for neuron in reliable_neurons:
        filename = save_dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
        if filename.exists():
            continue
        ds = DataLoader(
            dataset=GratingDataset(
                args,
                mouse_id=mouse_id,
                neuron=neuron,
                dataset=dataset,
                device=args.device,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        responses, center_directions, surround_directions, visual_stimuli = inference(
            model=model,
            mouse_id=mouse_id,
            neuron=neuron,
            ds=ds,
            save_dir=save_dir,
            animate=args.animate,
        )
        df = pd.DataFrame(
            {
                "mouse": [mouse_id] * len(responses),
                "neuron": [neuron] * len(responses),
                "center_direction": center_directions,
                "surround_direction": surround_directions,
                "response": torch.sum(responses[:, PAD:-PAD], dim=1),
                "raw_response": responses.tolist(),
                "stimulus_type": ["dynamic"] * len(responses),
                "stimulus": rearrange(
                    visual_stimuli, "b c t h w -> b (c t h w)"
                ).tolist(),
            }
        )
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
        / "center_surround"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    utils.set_logger_handles(
        logger=logger,
        filename=save_dir / f"output-{utils.get_timestamp()}.log",
        level=logging.INFO,
    )

    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM_OLD)

    for mouse_id in args.mouse_ids:
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
    parser.add_argument("--data_dir", type=Path, default="../../../data")
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
        "--use_preferred_size",
        action="store_true",
        help="use neuron preferred stimulus size",
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
        default=60,
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
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
