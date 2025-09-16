"""
Estimate the artificial receptive fields (aRFs) of a trained model.

We show num_samples white static noise frame to the model and compute the sum
of the response within the presentation window. We then compute the average of
the white noises weighted by the response.
"""

import argparse
import warnings
from argparse import RawTextHelpFormatter
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import torch
from einops import einsum
from einops import repeat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from viv1t import data
from viv1t.data import get_mean_behaviors
from viv1t.data.data import MovieDataset
from viv1t.model import Model
from viv1t.utils import plot
from viv1t.utils import utils
from viv1t.utils.load_model import load_model

warnings.simplefilter("error", opt.OptimizeWarning)

plot.set_font()

H, W = 36, 64
SKIP = 50  # skip the first 50 frames from each trial
PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2

MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) // 2


DPI = 240
TICK_FONTSIZE, LABEL_FONTSIZE = 10, 10
PLOT_DIR = Path("figures") / "noise_presentation"


class WhiteNoiseDataset(Dataset):
    def __init__(
        self,
        mouse_id: str,
        noises: torch.Tensor,
        dataset: MovieDataset,
        device: torch.device,
    ):
        self.mouse_id = mouse_id
        self.noises = MAX * noises  # scale noise to [0, 255) from [0, 1)
        self.num_neurons = dataset.num_neurons

        self.transform_video = dataset.transform_video

        # load response statistics to device for quicker inference
        self.transform_output = dataset.transform_output
        self.response_stats = {
            k: v.to(device) for k, v in dataset.response_stats.items()
        }
        self.response_precision = dataset.response_precision.to(device)

        behavior, pupil_center = get_mean_behaviors(
            mouse_id=mouse_id, num_frames=data.MAX_FRAME
        )
        self.behavior = dataset.transform_behavior(behavior)
        self.pupil_center = dataset.transform_pupil_center(pupil_center)

        self.blank = torch.full((1, BLANK_SIZE, H, W), fill_value=GREY_COLOR)

    def __len__(self):
        return len(self.noises)

    def i_transform_response(self, response: torch.Tensor) -> torch.Tensor:
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response / self.response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    def prepare_video(self, noise: torch.Tensor) -> torch.tensor:
        noise = repeat(noise, "c h w -> c t h w", t=PATTERN_SIZE)
        video = torch.cat([self.blank, noise, self.blank], dim=1)
        return video

    def __getitem__(
        self, idx: int | torch.Tensor, to_tensor: bool = True
    ) -> dict[str, torch.Tensor | str]:
        video = self.prepare_video(self.noises[idx])
        return {
            "video": self.transform_video(video),
            "mouse_id": self.mouse_id,
            "behavior": self.behavior,
            "pupil_center": self.pupil_center,
            "raw_video": video,
        }


@torch.inference_mode()
def inference(
    model: Model,
    mouse_id: str,
    ds: DataLoader,
    save_dir: Path,
    animate: bool = False,
) -> torch.Tensor:
    device, dtype = model.device, model.dtype
    num_batches, num_samples = len(ds), len(ds.dataset)
    num_neurons = ds.dataset.num_neurons

    # randomly select 10 trials  to plot
    pad = 10
    start = -(BLANK_SIZE + PATTERN_SIZE + pad)
    end = -(BLANK_SIZE - pad)
    mask = np.concat([np.zeros(pad), np.ones(PATTERN_SIZE), np.zeros(pad)])
    make_plot = np.random.choice(num_batches, size=min(5, num_batches), replace=False)

    responses = torch.zeros((num_samples, num_neurons), dtype=torch.float32)
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
        if animate and i in make_plot:
            plot.animate_stimulus(
                video=batch["raw_video"][0, :, start:end, :, :].cpu().numpy(),
                response=torch.mean(response[0, :, start:end], dim=0).cpu().numpy(),
                filename=save_dir / "figures" / f"mouse{mouse_id}_trial{i}.mp4",
                presentation_mask=mask,
                skip=0,
            )
        # sum the response during stimulus presentation
        response = response[:, :, -(BLANK_SIZE + PATTERN_SIZE) : -BLANK_SIZE]
        response = torch.sum(response, dim=-1)
        batch_size = len(response)
        responses[index : index + batch_size] = response.cpu()
        index += batch_size
    return responses


def estimate_mouse(
    args,
    mouse_id: str,
    model: Model,
    sensorium_dataset: MovieDataset,
    save_dir: Path,
):
    # generate num_samples of static white noise from uniform distribution
    noises = torch.rand((args.num_samples, 1, H, W), dtype=torch.float32)
    ds = DataLoader(
        dataset=WhiteNoiseDataset(
            mouse_id=mouse_id,
            noises=noises,
            dataset=sensorium_dataset,
            device=model.device,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    responses = inference(
        model=model,
        mouse_id=mouse_id,
        ds=ds,
        save_dir=save_dir,
        animate=args.animate,
    )
    # compute white noise average weighted by the response
    aRFs = einsum(responses, noises, "b n, b c h w -> n c h w").numpy()
    np.savez_compressed(
        file=save_dir / f"mouse{mouse_id}.npz",
        data=aRFs,
        allow_pickle=False,
    )


def main(args):
    utils.set_random_seed(args.seed)
    args.device = utils.get_device(args.device)

    save_dir = args.output_dir / "aRFs"
    save_dir.mkdir(parents=True, exist_ok=True)

    model, ds = load_model(args, evaluate=args.evaluate, compile=args.compile)
    model.train(False)

    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM_OLD)

    for mouse_id in args.mouse_ids:
        estimate_mouse(
            args,
            mouse_id=mouse_id,
            model=model,
            sensorium_dataset=ds[mouse_id].dataset,
            save_dir=save_dir,
        )

    print(f"Saved aRFs to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="../data",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100000)
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
        default=None,
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the model on the validation set after loading the checkpoint.",
    )
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
