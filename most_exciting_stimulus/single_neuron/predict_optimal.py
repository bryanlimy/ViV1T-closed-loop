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


def get_stimulus_size(output_dir: Path, mouse_id: str, neuron: int) -> int:
    stimulus_size = utils.get_size_tuning_preference(
        output_dir=output_dir, mouse_id=mouse_id, neuron=neuron
    )
    if stimulus_size <= 0:
        print(
            f"Mouse {mouse_id} neuron {neuron:04d} has a preferred stimulus "
            f"size of {stimulus_size}. Skipping this neuron."
        )
        stimulus_size = 20
    return stimulus_size


@torch.inference_mode()
def inference(
    model: Model,
    video: torch.Tensor,
    mouse_id: str,
    neuron: int,
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
    response = response[0, neuron, :]
    return response


@torch.inference_mode()
def inference_grating_center_surround(
    model: Model,
    ds: dict[str, DataLoader],
    output_dir: Path,
):
    print(f"Inference grating center with grating surround")
    df = pd.read_parquet(output_dir / "most_exciting_gratings.parquet")
    result = {
        stim_type: {mouse: {} for mouse in data.SENSORIUM_OLD}
        for stim_type in ["center", "most_exciting", "most_inhibiting"]
    }
    grating_kws = {
        "cpd": CPD,
        "cpf": CPF,
        "num_frames": PATTERN_SIZE,
        "phase": PHASE,
        "contrast": CONTRAST,
        "height": VIDEO_H,
        "width": VIDEO_W,
        "to_tensor": True,
    }
    for mouse_id in data.SENSORIUM_OLD:
        reliable_neurons = sorted(
            utils.get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id)
        )
        dataset: MovieDataset = ds[mouse_id].dataset
        behavior, pupil_center = data.get_mean_behaviors(
            mouse_id, num_frames=data.MAX_FRAME
        )
        behavior = dataset.transform_behavior(behavior)
        pupil_center = dataset.transform_pupil_center(pupil_center)
        for neuron in tqdm(reliable_neurons, desc=f"Mouse {mouse_id}"):
            most_exciting = df[
                (df.mouse == mouse_id)
                & (df.neuron == neuron)
                & (df.response_type == "most_exciting")
            ]
            assert len(most_exciting) == 1
            most_exciting = most_exciting.iloc[0]
            stimulus_size = get_stimulus_size(
                output_dir=output_dir, mouse_id=mouse_id, neuron=neuron
            )
            RF_center = stimulus.load_neuron_RF_center(
                output_dir=output_dir,
                mouse_id=mouse_id,
                neuron=neuron,
            )
            # get response for center
            video = stimulus.create_center_surround_grating(
                stimulus_size=stimulus_size,
                center=RF_center,
                center_direction=most_exciting.center_direction,
                surround_direction=-1,
                **grating_kws,
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                neuron=neuron,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["center"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response for most-exciting center surround
            video = stimulus.create_center_surround_grating(
                stimulus_size=stimulus_size,
                center=RF_center,
                center_direction=most_exciting.center_direction,
                surround_direction=most_exciting.surround_direction,
                **grating_kws,
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                neuron=neuron,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["most_exciting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response for most-inhibiting center surround
            most_inhibiting = df[
                (df.mouse == mouse_id)
                & (df.neuron == neuron)
                & (df.response_type == "most_inhibiting")
            ]
            assert len(most_inhibiting) == 1
            most_inhibiting = most_inhibiting.iloc[0]
            video = stimulus.create_center_surround_grating(
                stimulus_size=stimulus_size,
                center=RF_center,
                center_direction=most_inhibiting.center_direction,
                surround_direction=most_inhibiting.surround_direction,
                **grating_kws,
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                neuron=neuron,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["most_inhibiting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
    filename = (
        output_dir / "most_exciting_stimulus" / "grating_center_grating_surround.pkl"
    )
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    print(f"Saved responses to grating center and grating surround to {filename}.")


def load_center_grating_with_natural_surround_stimulus(
    mouse_id: str,
    neuron: int,
    center_direction: int,
    circular_mask: torch.Tensor,
    data_dir: Path,
    surround_dir: Path | None,
    response_type: str | None,
) -> torch.Tensor:
    center_grating = stimulus.create_full_field_grating(
        direction=center_direction,
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
    if surround_dir is None:
        natural_surround = GREY_COLOR
    else:
        assert response_type in ("most_exciting", "most_inhibiting")
        df = pd.read_parquet(
            surround_dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
        )
        df = df[df.response_type == response_type]
        assert len(df) == 1
        trial_id, frame = df.iloc[0].trial, df.iloc[0].frame
        natural_surround = np.load(
            data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
        )
        natural_surround = natural_surround[:, :, frame : frame + PATTERN_SIZE]
        natural_surround = rearrange(natural_surround, "h w t -> () t h w")
        natural_surround = torch.from_numpy(natural_surround)
    video = torch.where(circular_mask, center_grating, natural_surround)
    return video


@torch.inference_mode()
def inference_grating_center_natural_surround(
    model: Model,
    ds: dict[str, DataLoader],
    data_dir: Path,
    output_dir: Path,
):
    print(f"\nInference grating center with natural surround")
    save_dir = (
        output_dir
        / "most_exciting_stimulus"
        / "gratings"
        / "grating_center_natural_surround"
        / "dynamic_center_dynamic_surround"
    )
    df = pd.read_parquet(output_dir / "most_exciting_gratings.parquet")
    result = {
        stim_type: {mouse: {} for mouse in data.SENSORIUM_OLD}
        for stim_type in ["center", "most_exciting", "most_inhibiting"]
    }
    for mouse_id in data.SENSORIUM_OLD:
        reliable_neurons = sorted(
            utils.get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id)
        )
        dataset: MovieDataset = ds[mouse_id].dataset
        behavior, pupil_center = data.get_mean_behaviors(
            mouse_id, num_frames=data.MAX_FRAME
        )
        behavior = dataset.transform_behavior(behavior)
        pupil_center = dataset.transform_pupil_center(pupil_center)
        for neuron in tqdm(reliable_neurons, desc=f"Mouse {mouse_id}"):
            most_exciting = df[
                (df.mouse == mouse_id)
                & (df.neuron == neuron)
                & (df.response_type == "most_exciting")
            ]
            assert len(most_exciting) == 1
            center_direction = most_exciting.iloc[0].center_direction
            circular_mask = stimulus.create_circular_mask(
                stimulus_size=get_stimulus_size(
                    output_dir=output_dir, mouse_id=mouse_id, neuron=neuron
                ),
                center=stimulus.load_neuron_RF_center(
                    output_dir=output_dir,
                    mouse_id=mouse_id,
                    neuron=neuron,
                ),
                width=VIDEO_W,
                height=VIDEO_H,
                num_frames=PATTERN_SIZE,
                to_tensor=True,
            )
            # get response for center
            video = load_center_grating_with_natural_surround_stimulus(
                mouse_id=mouse_id,
                neuron=neuron,
                center_direction=center_direction,
                circular_mask=circular_mask,
                data_dir=data_dir,
                surround_dir=None,
                response_type=None,
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                neuron=neuron,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["center"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response for most-exciting grating center with natural surround
            video = load_center_grating_with_natural_surround_stimulus(
                mouse_id=mouse_id,
                neuron=neuron,
                center_direction=center_direction,
                circular_mask=circular_mask,
                data_dir=data_dir,
                surround_dir=save_dir,
                response_type="most_exciting",
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                neuron=neuron,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["most_exciting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response for most-inhibiting grating center and natural surround
            video = load_center_grating_with_natural_surround_stimulus(
                mouse_id=mouse_id,
                neuron=neuron,
                center_direction=center_direction,
                circular_mask=circular_mask,
                data_dir=data_dir,
                surround_dir=save_dir,
                response_type="most_inhibiting",
            )
            response = inference(
                model=model,
                mouse_id=mouse_id,
                neuron=neuron,
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["most_inhibiting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
    filename = (
        output_dir / "most_exciting_stimulus" / "grating_center_natural_surround.pkl"
    )
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    print(f"Saved responses to grating center and natural surround to {filename}.")


def load_video(
    data_dir: Path,
    save_dir: Path,
    mouse_id: str,
    neuron: int,
    static: bool,
    response_type: str | None = None,
) -> torch.Tensor:
    filename = save_dir / f"mouse{mouse_id}_neuron{neuron:04d}.parquet"
    df = pd.read_parquet(filename)
    df = df[
        (df.mouse == mouse_id)
        & (df.neuron == neuron)
        & (df.stim_type == ("static" if static else "dynamic"))
    ]
    if response_type is not None:
        df = df[df.response_type == response_type]
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
def inference_natural_center_surround(
    model: Model,
    ds: dict[str, DataLoader],
    data_dir: Path,
    output_dir: Path,
):
    print(f"\nInference natural center with natural surround")
    save_dir = output_dir / "most_exciting_stimulus" / "single_neuron" / "natural"
    result = {
        config: {
            stim_type: {mouse: {} for mouse in data.SENSORIUM_OLD}
            for stim_type in ["center", "most_exciting", "most_inhibiting"]
        }
        for config in ["ss", "sd", "dd"]
    }
    for mouse_id in data.SENSORIUM_OLD:
        reliable_neurons = sorted(
            utils.get_reliable_neurons(output_dir=output_dir, mouse_id=mouse_id)
        )
        dataset: MovieDataset = ds[mouse_id].dataset
        behavior, pupil_center = data.get_mean_behaviors(
            mouse_id, num_frames=data.MAX_FRAME
        )
        behavior = dataset.transform_behavior(behavior)
        pupil_center = dataset.transform_pupil_center(pupil_center)

        for neuron in tqdm(reliable_neurons, desc=f"Mouse {mouse_id}"):
            stimulus_size = get_stimulus_size(
                output_dir=output_dir, mouse_id=mouse_id, neuron=neuron
            )
            RF_center = stimulus.load_neuron_RF_center(
                output_dir=output_dir,
                mouse_id=mouse_id,
                neuron=neuron,
            )
            circular_mask = stimulus.create_circular_mask(
                stimulus_size=stimulus_size,
                center=RF_center,
                pixel_width=VIDEO_W,
                pixel_height=VIDEO_H,
                num_frames=PATTERN_SIZE,
                to_tensor=True,
            )
            ###### STATIC CENTER and STATIC SURROUND ######
            # get response to static center
            center_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir / "center" / "static",
                mouse_id=mouse_id,
                neuron=neuron,
                static=True,
            )
            video = torch.where(circular_mask, center_video, GREY_COLOR)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["ss"]["center"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response to static center and most-exciting static surround
            surround_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir / "center_surround" / "static_center_static_surround",
                mouse_id=mouse_id,
                neuron=neuron,
                static=True,
                response_type="most_exciting",
            )
            video = torch.where(circular_mask, center_video, surround_video)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["ss"]["most_exciting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response to static center and most-inhibiting static surround
            surround_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir / "center_surround" / "static_center_static_surround",
                mouse_id=mouse_id,
                neuron=neuron,
                static=True,
                response_type="most_inhibiting",
            )
            video = torch.where(circular_mask, center_video, surround_video)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["ss"]["most_inhibiting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            ###### STATIC CENTER and DYNAMIC SURROUND ######
            # get response to static center
            center_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir / "center" / "static",
                mouse_id=mouse_id,
                neuron=neuron,
                static=True,
            )
            video = torch.where(circular_mask, center_video, GREY_COLOR)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["sd"]["center"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response to static center and most-exciting static surround
            surround_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir
                / "center_surround"
                / "static_center_dynamic_surround",
                mouse_id=mouse_id,
                neuron=neuron,
                static=False,
                response_type="most_exciting",
            )
            video = torch.where(circular_mask, center_video, surround_video)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["sd"]["most_exciting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response to static center and most-inhibiting static surround
            surround_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir
                / "center_surround"
                / "static_center_dynamic_surround",
                mouse_id=mouse_id,
                neuron=neuron,
                static=False,
                response_type="most_inhibiting",
            )
            video = torch.where(circular_mask, center_video, surround_video)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["sd"]["most_inhibiting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            ###### DYNAMIC CENTER and DYNAMIC SURROUND ######
            # get response to static center
            center_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir / "center" / "dynamic",
                mouse_id=mouse_id,
                neuron=neuron,
                static=False,
            )
            video = torch.where(circular_mask, center_video, GREY_COLOR)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["dd"]["center"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response to static center and most-exciting static surround
            surround_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir
                / "center_surround"
                / "dynamic_center_dynamic_surround",
                mouse_id=mouse_id,
                neuron=neuron,
                static=False,
                response_type="most_exciting",
            )
            video = torch.where(circular_mask, center_video, surround_video)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["dd"]["most_exciting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
            # get response to static center and most-inhibiting static surround
            surround_video = load_video(
                data_dir=data_dir,
                save_dir=save_dir
                / "center_surround"
                / "dynamic_center_dynamic_surround",
                mouse_id=mouse_id,
                neuron=neuron,
                static=False,
                response_type="most_inhibiting",
            )
            video = torch.where(circular_mask, center_video, surround_video)
            response = inference(
                model=model,
                video=video,
                mouse_id=mouse_id,
                neuron=neuron,
                behavior=behavior,
                pupil_center=pupil_center,
                dataset=dataset,
            )
            result["dd"]["most_inhibiting"][mouse_id][neuron] = {
                "response": response.numpy(),
                "video": video.numpy(),
            }
    filename = (
        output_dir / "most_exciting_stimulus" / "natural_center_natural_surround.pkl"
    )
    with open(filename, "wb") as file:
        pickle.dump(result, file)
    print(f"Saved responses to natural center and natural surround to {filename}.")


def main(args):
    utils.set_random_seed(args.seed)
    args.device = utils.get_device(args.device)
    model, ds = load_model(args, evaluate=args.evaluate, compile=args.compile)
    model.train(False)

    inference_grating_center_surround(
        model=model,
        ds=ds,
        output_dir=args.output_dir,
    )
    inference_grating_center_natural_surround(
        model=model,
        ds=ds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    inference_natural_center_surround(
        model=model,
        ds=ds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="../data/sensorium",
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
