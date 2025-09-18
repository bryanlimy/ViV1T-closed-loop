"""Predict aRFs estimated from estimate_aRFs.py"""

import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import repeat
from torch.utils.data import DataLoader

from viv1t import data
from viv1t.model import Model
from viv1t.utils import load_model
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()

SKIP = 50  # skip the first 50 frames from each trial
W, H = 64, 36
PATTERN_SIZE = 30
MAX_FRAME = 300
BLANK_SIZE = (MAX_FRAME - PATTERN_SIZE) // 2
MIN, MAX = 0, 255.0


DPI = 240
TICK_FONTSIZE, LABEL_FONTSIZE = 10, 10
PLOT_DIR = Path("figures") / "predict_aRFs"


def get_behaviors(
    mouse_id: str, ds: DataLoader, device: torch.device
) -> (torch.Tensor, torch.Tensor):
    behavior, pupil_center = data.get_mean_behaviors(
        mouse_id=mouse_id, num_frames=MAX_FRAME
    )
    behavior = ds.dataset.transform_behavior(behavior)
    pupil_center = ds.dataset.transform_pupil_center(pupil_center)

    to_batch = lambda x: x.to(device)[None, :]

    return to_batch(behavior), to_batch(pupil_center)


def get_sensorium_response_stats(
    output_dir: Path, mouse_id: str, neuron: int
) -> tuple[float | None, float | None]:
    """Get statistics of the predicted response on the Sensorium 2023 dataset"""
    filename = output_dir / "response_stats.parquet"
    if filename.exists():
        df = pd.read_parquet(filename)
        df = df.loc[(df.mouse == mouse_id) & (df.neuron == neuron)]
        return df["max"].max(), df["mean"].mean()
    return None, None


def prepare_video(aRF: torch.Tensor) -> torch.tensor:
    aRF = repeat(aRF, "c h w -> c t h w", t=PATTERN_SIZE)
    blank = torch.full(
        (1, BLANK_SIZE, H, W),
        fill_value=(MAX - MIN) / 2,
        device=aRF.device,
    )
    video = torch.cat([blank, aRF, blank], dim=1)
    return video[None, ...]


@torch.inference_mode()
def inference(
    model: Model,
    mouse_id: str,
    ds: DataLoader,
    video: torch.Tensor,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
) -> torch.Tensor:
    device, dtype = model.device, model.dtype
    response, _ = model(
        inputs=ds.dataset.transform_video(video).to(device, dtype),
        mouse_id=mouse_id,
        behaviors=behavior,
        pupil_centers=pupil_center,
    )
    response = response.to("cpu", torch.float32)[0]
    response = ds.dataset.i_transform_response(response)
    return response


def main(args):
    aRFs_dir = args.output_dir / "aRFs"
    filename = aRFs_dir / f"mouse{args.mouse_id}_aRFs.npy"
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find aRF file {filename}")

    utils.set_random_seed(args.seed)
    args.device = utils.get_device(args.device)

    model, ds = load_model(args)
    model = model.to(args.device)
    model.train(False)
    ds = ds[args.mouse_id]

    behavior, pupil_center = get_behaviors(
        mouse_id=args.mouse_id, ds=ds, device=args.device
    )

    aRFs = np.load(filename, allow_pickle=False)
    aRF = aRFs[args.neuron]
    aRF = 255.0 * (aRF - aRF.min()) / (aRF.max() - aRF.min())
    aRF = torch.from_numpy(aRF)
    video = prepare_video(aRF)

    response = inference(
        model=model,
        mouse_id=args.mouse_id,
        ds=ds,
        video=video,
        behavior=behavior,
        pupil_center=pupil_center,
    )
    # get response to frame presentation
    pad = 10
    start = -(BLANK_SIZE + PATTERN_SIZE + pad)
    end = -(BLANK_SIZE - pad)
    response = response[args.neuron, start:end].numpy()
    presentation_mask = np.concat(
        [np.zeros(pad), np.ones(PATTERN_SIZE), np.zeros(pad)]
    ).astype(bool)

    sensorium_max, sensorium_mean = get_sensorium_response_stats(
        output_dir=args.output_dir, mouse_id=args.mouse_id, neuron=args.neuron
    )

    plot.animate_stimulus(
        video=video[0, :, start:end, :, :].numpy(),
        response=response,
        filename=PLOT_DIR / f"mouse{args.mouse_id}_neuron{args.neuron:04d}.gif",
        ds_max=sensorium_max,
        ds_mean=sensorium_mean,
        max_response=response[presentation_mask].max(),
        sum_response=response[presentation_mask].sum(),
        neuron=args.neuron,
        presentation_mask=presentation_mask,
        skip=0,
    )

    print(f"Saved aRFs to {PLOT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="../data/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument("--mouse_id", type=str, default="A", choices=data.SENSORIUM_OLD)
    parser.add_argument("--neuron", type=int, default=4286)
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "use the best available device if --device is not specified.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "bf16"],
        default=None,
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
