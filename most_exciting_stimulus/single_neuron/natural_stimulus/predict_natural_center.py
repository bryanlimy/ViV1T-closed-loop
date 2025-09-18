import argparse
import logging
from argparse import RawTextHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

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

MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) // 2

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def load_videos(data_dir: Path, mouse_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all natural movie videos from the training set"""
    tiers = data.get_tier_ids(data_dir=data_dir, mouse_id=mouse_id)
    trial_ids = np.where(tiers == "train")[0]
    samples = [
        data.load_trial(
            mouse_dir=data_dir / data.MOUSE_IDS[mouse_id],
            trial_id=trial_id,
            to_tensor=True,
        )
        for trial_id in trial_ids
    ]
    videos = torch.stack([sample["video"] for sample in samples])
    return videos, torch.from_numpy(trial_ids)


class NaturalCenterDataset(Dataset):
    def __init__(
        self,
        args,
        mouse_id: str,
        neuron: int,
        videos: torch.Tensor,
        trial_ids: torch.Tensor,
        dataset: MovieDataset,
        device: torch.device,
    ):
        self.mouse_id = mouse_id
        self.static = args.static
        self.step = args.step

        _, c, t, h, w = videos.shape
        self.video_c, self.video_h, self.video_w = c, h, w
        self.videos, self.trial_ids = videos, trial_ids
        self.clip_ids = self.get_clip_ids(videos=self.videos, trial_ids=self.trial_ids)

        if args.use_preferred_size:
            args.stimulus_size = utils.get_size_tuning_preference(
                output_dir=args.output_dir, mouse_id=mouse_id, neuron=neuron
            )
        monitor_info = data.MONITOR_INFO[mouse_id]
        circular_mask = stimulus.create_circular_mask(
            stimulus_size=args.stimulus_size,
            center=stimulus.load_neuron_RF_center(
                output_dir=args.output_dir,
                mouse_id=mouse_id,
                neuron=neuron,
            ),
            pixel_width=w,
            pixel_height=h,
            monitor_width=monitor_info["width"],
            monitor_height=monitor_info["height"],
            monitor_distance=monitor_info["distance"],
            num_frames=t,
            to_tensor=True,
        )
        self.videos = torch.where(circular_mask, self.videos, GREY_COLOR)

        self.transform_video = dataset.transform_video

        # load response statistics to device for quicker inference
        self.transform_output = dataset.transform_output
        self.response_stats = {
            k: v.to(device) for k, v in dataset.response_stats.items()
        }
        self.response_precision = dataset.response_precision.to(device)

        behavior, pupil_center = data.get_mean_behaviors(mouse_id, num_frames=t)
        self.behavior = dataset.transform_behavior(behavior)
        self.pupil_center = dataset.transform_pupil_center(pupil_center)

        self.blank = torch.full((c, BLANK_SIZE, h, w), fill_value=GREY_COLOR)

    def get_clip_ids(
        self, videos: torch.Tensor, trial_ids: torch.Tensor
    ) -> torch.Tensor:
        b, _, t, h, w = videos.shape
        frames = torch.arange(t)
        if self.static:
            frames = frames[:: self.step]
            trial_ids = repeat(trial_ids, "b -> (b t)", t=len(frames))
            frames = repeat(frames, "t -> (b t)", b=b)
        else:
            # calculate the number of clips that can be extracted from each video
            # using a sliding window of step size
            frames = frames.unfold(dimension=0, size=PATTERN_SIZE, step=self.step)
            frames = frames[:, 0]
            num_clips = frames.shape[0]
            trial_ids = repeat(trial_ids, "batch -> (batch clip)", clip=num_clips)
            frames = repeat(frames, "clip -> (batch clip)", batch=b)
        clip_ids = torch.stack((trial_ids, frames), dim=1)
        return clip_ids

    def __len__(self):
        return len(self.clip_ids)

    def i_transform_response(self, response: torch.Tensor) -> torch.Tensor:
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response / self.response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    def prepare_video(
        self, trial_id: int | torch.Tensor, frame: int | torch.Tensor
    ) -> torch.tensor:
        i = torch.where(self.trial_ids == trial_id)[0].item()
        video = self.videos[i]
        if self.static:
            video = video[:, frame]
            video = repeat(video, "c h w -> c t h w", t=PATTERN_SIZE)
        else:
            video = video[:, frame : frame + PATTERN_SIZE]
        # video = torch.where(self.circular_mask, video, GREY_COLOR)
        video = torch.cat([self.blank, video, self.blank], dim=1)
        return video

    def __getitem__(
        self, idx: int | torch.Tensor, to_tensor: bool = True
    ) -> dict[str, torch.Tensor | str | int]:
        trial_id, frame = self.clip_ids[idx]
        video = self.prepare_video(trial_id=trial_id, frame=frame)
        return {
            "video": self.transform_video(video),
            "mouse_id": self.mouse_id,
            "behavior": self.behavior,
            "pupil_center": self.pupil_center,
            "raw_video": video,
            "trial_id": trial_id,
            "frame": frame,
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
    make_plot = np.random.choice(num_batches, size=min(2, num_batches), replace=False)

    trial_ids = torch.zeros(num_samples, dtype=torch.int32)
    frames = torch.zeros(num_samples, dtype=torch.int32)
    responses = torch.zeros(
        size=(num_samples, (PAD + PATTERN_SIZE + PAD)),
        device="cpu",
        dtype=torch.float32,
    )
    visual_stimuli = torch.zeros(
        size=(
            num_samples,
            ds.dataset.video_c,
            (PAD + PATTERN_SIZE + PAD),
            ds.dataset.video_h,
            ds.dataset.video_w,
        ),
        dtype=torch.int32,
    )

    index = 0
    with logging_redirect_tqdm(loggers=[logger]):
        for i, batch in enumerate(
            tqdm(ds, desc=f"Mouse {mouse_id} neuron {neuron:04d}")
        ):
            response, _ = model(
                inputs=batch["video"].to(device, dtype),
                mouse_id=mouse_id,
                behaviors=batch["behavior"].to(device, dtype),
                pupil_centers=batch["pupil_center"].to(device, dtype),
            )
            response = response.to(torch.float32)
            response = ds.dataset.i_transform_response(response)
            if animate and i in make_plot:
                animate_stimulus(
                    video=batch["raw_video"][0, :, start:end, :, :].numpy(),
                    response=response[0, neuron, start:end].cpu().numpy(),
                    filename=save_dir
                    / "figures"
                    / f"mouse{mouse_id}"
                    / f"neuron{neuron:04d}_trial{batch['trial_id'][0]:03d}_frame{batch['frame'][0]:03d}.mp4",
                    presentation_mask=mask,
                    skip=0,
                )
            batch_size = response.shape[0]
            # get neuron response within the stimulus presentation window
            response = response[:, neuron, start:end]
            responses[index : index + batch_size] = response.to("cpu")
            trial_ids[index : index + batch_size] = batch["trial_id"].to(torch.int32)
            frames[index : index + batch_size] = batch["frame"].to(torch.int32)
            visual_stimuli[index : index + batch_size] = batch["raw_video"][
                :, :, start:end, :, :
            ]
            index += batch_size
    return responses, trial_ids, frames, visual_stimuli


def process_mouse(
    args,
    mouse_id: str,
    model: Model,
    dataset: MovieDataset,
    save_dir: Path,
):
    videos_, trial_ids_ = load_videos(data_dir=args.data_dir, mouse_id=mouse_id)
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
            dataset=NaturalCenterDataset(
                args,
                mouse_id=mouse_id,
                neuron=neuron,
                videos=videos_,
                trial_ids=trial_ids_,
                dataset=dataset,
                device=model.device,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        responses, trial_ids, frames, visual_stimuli = inference(
            model=model,
            mouse_id=mouse_id,
            neuron=neuron,
            ds=ds,
            save_dir=save_dir,
            animate=args.animate,
        )
        # find the trial and frame that elicit maximum response
        sum_responses = torch.sum(responses[:, PAD:-PAD], dim=1)
        i = torch.argmax(sum_responses)
        df = pd.DataFrame(
            {
                "mouse": [mouse_id],
                "neuron": [neuron],
                "trial": [trial_ids[i].item()],
                "frame": [frames[i].item()],
                "response": [sum_responses[i].item()],
                "raw_response": [responses[i].tolist()],
                "stimulus_type": ["static" if args.static else "dynamic"],
                "stimulus": [
                    rearrange(visual_stimuli[i], "c t h w -> (c t h w)").tolist()
                ],
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
        / "natural"
        / "center"
        / ("static" if args.static else "dynamic")
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

    logger.info(f"Saved result to {save_dir}.")


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
        "--step",
        type=int,
        default=5,
        help="step size (number of frames) of the sliding window over the natural movies",
    )
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
    parser.add_argument(
        "--static",
        action="store_true",
        help="find the most exciting natural static frame.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
