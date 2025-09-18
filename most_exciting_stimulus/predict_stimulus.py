import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
import torch
import torch.fft

from viv1t.model import Model
from viv1t.most_exciting_stimulus import utils as mes_utils
from viv1t.utils import utils
from viv1t.utils.load_model import load_model
from viv1t.utils.plot import animate_stimulus

SKIP = 50  # skip the first 50 frames from each trial
MIN, MAX = 0, 255.0


def load_stimulus(stimulus_dir: Path) -> (np.ndarray, dict):
    ckpt_filename = stimulus_dir / "ckpt.pt"
    assert ckpt_filename.exists(), f'Checkpoint file not found: "{ckpt_filename}"'
    ckpt = torch.load(ckpt_filename, map_location="cpu", weights_only=True)
    video = ckpt["video"]
    print(
        f"Load stimulus from {ckpt_filename} (step: {ckpt['step']}, "
        f"loss: {ckpt['loss']:.02f}, peak: {ckpt['response'].max():.0f})"
    )
    info = {}
    info["mouse"] = ckpt["mouse"]
    if "neuron" in ckpt:
        info["neuron"] = ckpt["neuron"]
    else:
        info["neuron"] = None
    if "presentation_mask" in ckpt:
        info["presentation_mask"] = ckpt["presentation_mask"]
    else:
        info["presentation_mask"] = None
    return video, info


@torch.inference_mode()
def predict(
    model: Model,
    video: torch.Tensor,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
    mouse_id: str,
    neuron: int | None,
    transform_video: Callable,
    i_transform_response: Callable,
) -> torch.Tensor:
    t = video.size(1) - SKIP
    response, _ = model(
        inputs=transform_video(video)[None, ...].to(model.device, model.dtype),
        mouse_id=mouse_id,
        behaviors=behavior.to(model.device, model.dtype),
        pupil_centers=pupil_center.to(model.device, model.dtype),
    )
    response = response.to("cpu", torch.float32)[:, :, -t:]
    response = i_transform_response(response)
    response = response[0]
    if neuron is not None:
        response = response[neuron]
    return response


def main(args: Any):
    args.device = utils.get_device(args.device)

    # load model
    model, ds = load_model(args)
    device = model.device
    model.train(False)

    # load MEI/MEV
    video, info = load_stimulus(args.input_dir)
    # load behavioral variables for mouse
    ds = ds[info["mouse"]]
    behavior, pupil_center = mes_utils.get_behaviors(
        mouse_id=info["mouse"], ds=ds, device=device
    )
    channel, _, height, width = model.input_shapes["video"]

    save_dir = (
        args.output_dir
        / "most_exciting_stimulus"
        / "predict"
        / str(args.input_dir).replace("../", "").replace("/", "_")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    response = predict(
        model=model,
        video=video,
        behavior=behavior,
        pupil_center=pupil_center,
        mouse_id=info["mouse"],
        neuron=info["neuron"],
        transform_video=ds.dataset.transform_video,
        i_transform_response=ds.dataset.i_transform_response,
    )

    _response = response
    if info["neuron"] is None:
        # compute population average
        _response = torch.mean(response, dim=0)

    animate_stimulus(
        video=video.detach().cpu().numpy(),
        response=_response.cpu().numpy(),
        neuron=info["neuron"],
        filename=save_dir / "prediction.mp4",
        presentation_mask=info["presentation_mask"],
    )

    result = {"video": video, "response": response}
    torch.save(result, f=save_dir / "ckpt.pt")

    if info["presentation_mask"] is not None:
        t = response.shape[0]
        response = info["presentation_mask"][-t:] * response

    print(
        f"\nPredicted response sum: {response.sum():.02f}, "
        f"peak: {response.max():.0f}."
    )

    print(f"Saved result to {save_dir}")
    return result


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
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to the MEI/MEV checkpoint dir",
    )
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
        default="32",
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
