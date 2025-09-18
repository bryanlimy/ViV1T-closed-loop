"""
Helper function to predict all the most-exciting/most-inhibiting
natural/grating/generated center-surround responses for quick plotting.
"""

import argparse
import pickle
from argparse import RawTextHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from einops import repeat
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t import data
from viv1t.data.data import MovieDataset
from viv1t.model import Model
from viv1t.utils import stimulus
from viv1t.utils import utils
from viv1t.utils.load_model import load_model

SKIP = 50  # skip the first 50 frames from each trial
PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2
FPS = 30

VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2

# Grating parameters
CPD = 0.04  # spatial frequency in cycle per degree
CPF = 2 / FPS  # temporal frequency of 2Hz in cycle per frame
CONTRAST = 1
PHASE = 0


BLANK = torch.full((1, BLANK_SIZE, VIDEO_H, VIDEO_W), fill_value=GREY_COLOR)


@torch.inference_mode()
def inference(
    model: Model,
    video: torch.Tensor,
    mouse_id: str,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
    dataset: MovieDataset,
) -> torch.Tensor:
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
    response = response[0, :, :]
    return response


def load_full_field_video(
    data_dir: Path,
    save_dir: Path,
    mouse_id: str,
    static: bool,
    response_type: str,
) -> torch.Tensor:
    filename = save_dir / f"mouse{mouse_id}.parquet"
    df = pd.read_parquet(filename)
    df = df[
        (df.mouse == mouse_id)
        & (df.stim_type == ("static" if static else "dynamic"))
        & (df.response_type == response_type)
    ]
    assert len(df) == 1
    trial_id, frame = df.iloc[0].trial, df.iloc[0].frame
    # extract stimulus from Sensorium 2023 dataset
    video = np.load(
        data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    if static:
        video = repeat(video[:, :, frame], "h w -> h w t", t=PATTERN_SIZE)
    else:
        video = video[:, :, frame : frame + PATTERN_SIZE]
    video = rearrange(video, "h w t -> () t h w")
    video = torch.from_numpy(video)
    return video


@torch.inference_mode()
def inference_natural_full_field(
    model: Model,
    ds: dict[str, DataLoader],
    data_dir: Path,
    output_dir: Path,
):
    print(f"\nInference natural full field")
    save_dir = output_dir / "most_exciting_stimulus" / "population"
    configs = ["static", "dynamic"]
    result = {
        config: {stim_type: {} for stim_type in ["most_exciting", "most_inhibiting"]}
        for config in configs
    }
    for config in configs:
        for mouse_id in tqdm(data.SENSORIUM_OLD, desc=f"natural {config}"):
            dataset: MovieDataset = ds[mouse_id].dataset
            behavior, pupil_center = data.get_mean_behaviors(
                mouse_id, num_frames=data.MAX_FRAME
            )
            behavior = dataset.transform_behavior(behavior)
            pupil_center = dataset.transform_pupil_center(pupil_center)

            video = load_full_field_video(
                data_dir=data_dir,
                save_dir=save_dir / "natural" / "full_field" / config,
                mouse_id=mouse_id,
                static=config == "static",
                response_type="most_exciting",
            )
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result[config]["most_exciting"][mouse_id] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            video = load_full_field_video(
                data_dir=data_dir,
                save_dir=save_dir / "natural" / "full_field" / config,
                mouse_id=mouse_id,
                static=config == "static",
                response_type="most_inhibiting",
            )
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result[config]["most_inhibiting"][mouse_id] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
    filename = save_dir / "natural_full_field.pkl"
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    print(f"Saved responses to natural full field to {filename}.")


@torch.inference_mode()
def inference_grating_full_field(
    model: Model, ds: dict[str, DataLoader], output_dir: Path
):
    print(f"Inference grating full field")
    save_dir = output_dir / "most_exciting_stimulus" / "population"
    result = {stim_type: {} for stim_type in ["most_exciting", "most_inhibiting"]}
    grating_kws = {
        "cpd": CPD,
        "cpf": CPF,
        "num_frames": PATTERN_SIZE,
        "contrast": CONTRAST,
        # "phase": PHASE,
        "height": VIDEO_H,
        "width": VIDEO_W,
        "to_tensor": True,
    }
    rng = np.random.default_rng(1234)
    for mouse_id in tqdm(data.SENSORIUM_OLD, desc="grating"):
        dataset: MovieDataset = ds[mouse_id].dataset
        behavior, pupil_center = data.get_mean_behaviors(
            mouse_id, num_frames=data.MAX_FRAME
        )
        behavior = dataset.transform_behavior(behavior)
        pupil_center = dataset.transform_pupil_center(pupil_center)

        df = pd.read_parquet(
            save_dir
            / "gratings"
            / "full_field"
            / "dynamic"
            / f"mouse{mouse_id}.parquet"
        )
        most_exciting = df[
            (df.mouse == mouse_id) & (df.response_type == "most_exciting")
        ]
        assert len(most_exciting) == 1
        most_exciting = most_exciting.iloc[0]
        video_, responses = None, []
        for _ in range(10):
            video = stimulus.create_full_field_grating(
                direction=most_exciting.direction,
                phase=rng.choice(360, size=1).item(),  # randomize initial phase,
                **grating_kws,
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            responses.append(response)
            if video_ is None:
                video_ = video
        responses = torch.stack(responses, dim=0)
        response = torch.mean(responses, dim=0)  # average response over repeats
        result["most_exciting"][mouse_id] = {
            "response": response.numpy(),
            "video": video_.numpy(),
        }
        most_inhibiting = df[
            (df.mouse == mouse_id) & (df.response_type == "most_inhibiting")
        ]
        assert len(most_inhibiting) == 1
        most_inhibiting = most_inhibiting.iloc[0]
        video_, responses = None, []
        for _ in range(10):
            video = stimulus.create_full_field_grating(
                direction=most_inhibiting.direction,
                **grating_kws,
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            responses.append(response)
            if video_ is None:
                video_ = video
        responses = torch.stack(responses, dim=0)
        response = torch.mean(responses, dim=0)  # average response over repeats
        result["most_inhibiting"][mouse_id] = {
            "response": response.numpy(),
            "video": video_.numpy(),
        }
    filename = save_dir / "grating_full_field.pkl"
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    print(f"Saved responses to grating full field to {filename}.")


def main(args):
    utils.set_random_seed(args.seed)
    args.device = utils.get_device(args.device)
    model, ds = load_model(args, evaluate=args.evaluate, compile=args.compile)
    model.train(False)

    # inference_natural_full_field(
    #     model=model,
    #     ds=ds,
    #     data_dir=args.data_dir,
    #     output_dir=args.output_dir,
    # )
    inference_grating_full_field(
        model=model,
        ds=ds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="../../data/sensorium",
        help="path to directory where the Sensorium 2023 dataset is stored.",
    )
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
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
