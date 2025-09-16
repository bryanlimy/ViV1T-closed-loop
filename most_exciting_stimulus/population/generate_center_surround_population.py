import argparse
import logging
from argparse import RawTextHelpFormatter
from pathlib import Path
from shutil import rmtree
from typing import Any
from typing import Callable
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.fft
from einops import rearrange
from einops import repeat
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from viv1t import data
from viv1t.model import Model
from viv1t.most_exciting_stimulus import utils as mes_utils
from viv1t.most_exciting_stimulus.parameters import get_param
from viv1t.utils import stimulus
from viv1t.utils import utils
from viv1t.utils.load_model import load_model
from viv1t.utils.plot import animate_stimulus

SKIP = 50  # skip the first 50 frames from each trial
PATTERN_SIZE = 30
BLANK_SIZE = (data.MAX_FRAME - PATTERN_SIZE) // 2

VIDEO_H, VIDEO_W = 36, 64  # resolution of the video
MIN, MAX = 0, 255.0  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2
FPS = 30

MODE = Literal[0, 1, 2]

FFT_NORM = "ortho"
RESPONSE_TYPE = Literal["most_exciting", "most_inhibiting"]

# Grating parameters
CPD = 0.04  # spatial frequency in cycle per degree
CPF = 2 / FPS  # temporal frequency of 2Hz in cycle per frame
CONTRAST = 1
PHASE = 0

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def get_most_exciting_natural_center(
    args, mouse_id: str, device: torch.device
) -> torch.Tensor:
    filename = (
        args.output_dir
        / "most_exciting_stimulus"
        / "population"
        / "natural"
        / "center"
        / ("static" if args.static_center else "dynamic")
        / f"mouse{mouse_id}.parquet"
    )
    df = pd.read_parquet(filename)
    df = df[(df.mouse == mouse_id) & (df.response_type == "most_exciting")]
    assert df.trial.nunique() == df.frame.nunique() == 1
    trial_id, frame = df.iloc[0].trial, df.iloc[0].frame
    # extract stimulus from Sensorium 2023 dataset
    video = np.load(
        args.data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    if args.static_center:
        video = repeat(video[:, :, frame], "h w -> h w t", t=PATTERN_SIZE)
    else:
        video = video[:, :, frame : frame + PATTERN_SIZE]
    video = rearrange(video, "h w t -> () t h w")
    video = torch.from_numpy(video)
    return video.to(device, torch.float32)


def get_most_exciting_grating_center(
    args, mouse_id: str, device: torch.device
) -> torch.Tensor:
    filename = (
        args.output_dir
        / "most_exciting_stimulus"
        / "population"
        / "gratings"
        / "center_surround"
        / f"mouse{mouse_id}.parquet"
    )
    assert filename.exists(), f"Cannot find {filename}."
    df = pd.read_parquet(filename)
    df = df.loc[(df.mouse == mouse_id) & (df.surround_direction == -1)]
    # get most-exciting grating center direction
    center_direction = (
        df.sort_values(by="response", ascending=False).iloc[0].center_direction
    )
    video = stimulus.create_full_field_grating(
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
    return video.to(device, torch.float32)


def get_most_exciting_natural_surround(
    args, mouse_id: str, response_type: str
) -> torch.Tensor:
    filename = (
        args.output_dir
        / "most_exciting_stimulus"
        / "population"
        / "natural"
        / "center_surround"
        / f"static_center_{'static' if args.static_surround else 'dynamic'}_surround"
        / f"mouse{mouse_id}.parquet"
    )
    df = pd.read_parquet(filename)
    df = df[(df.mouse == mouse_id) & (df.response_type == response_type)]
    assert df.trial.nunique() == df.frame.nunique() == 1
    trial_id, frame = df.iloc[0].trial, df.iloc[0].frame
    video = np.load(
        args.data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    if args.static_surround:
        video = repeat(video[:, :, frame], "h w -> h w t", t=1)
    else:
        video = video[:, :, frame : frame + PATTERN_SIZE]
    video = rearrange(video, "h w t -> () t h w")
    video = torch.from_numpy(video).to(torch.float32)
    # standardize video
    video = (video - torch.mean(video)) / torch.std(video)
    return video


def get_transforms(
    ds: DataLoader,
    mouse_id: str,
    scale: torch.Tensor,
    max_value: torch.Tensor,
    circular_mask: torch.Tensor,
    video_shape: tuple[int, int, int, int],
    device: torch.device,
    mode: MODE,
    center_video: torch.Tensor | None = None,
    output_dir: Path = None,
    outer_stimulus_size: int | None = None,
):
    preprocess_video = mes_utils.transform_video(ds=ds, device=device)
    blank = torch.full(
        (video_shape[0], BLANK_SIZE, video_shape[2], video_shape[3]),
        fill_value=GREY_COLOR,
        device=device,
        requires_grad=False,
    )
    grey_color = torch.tensor(GREY_COLOR, dtype=torch.float32, device=device)

    monitor_info = data.MONITOR_INFO[mouse_id]
    outer_circular_mask = None
    if outer_stimulus_size is not None:
        outer_circular_mask = stimulus.create_circular_mask(
            stimulus_size=outer_stimulus_size,
            center=stimulus.load_population_RF_center(
                output_dir=output_dir,
                mouse_id=mouse_id,
            ),
            pixel_width=video_shape[3],
            pixel_height=video_shape[2],
            monitor_width=monitor_info["width"],
            monitor_height=monitor_info["height"],
            monitor_distance=monitor_info["distance"],
            num_frames=PATTERN_SIZE,
            to_tensor=True,
        )
        outer_circular_mask = outer_circular_mask.to(device)

    def get_video(param: torch.Tensor) -> torch.Tensor:
        """Construct video from trainable parameter"""
        param = scale * param
        if torch.is_complex(param):
            param = torch.fft.irfftn(param, s=video_shape, norm=FFT_NORM)
        if param.shape[1] == 1:
            # duplicate the single frame to create a video
            param = repeat(param, "c 1 h w -> c t h w", t=PATTERN_SIZE)
        video = max_value * torch.sigmoid(param)
        match mode:
            case 0:
                if center_video is not None:
                    video = center_video
                video = torch.where(circular_mask, video, grey_color)
            case 1 | 2:
                if outer_circular_mask is not None:
                    video = torch.where(outer_circular_mask, video, grey_color)
                video = torch.where(circular_mask, center_video, video)
        return video

    def get_batch(param: torch.Tensor) -> dict[str, torch.Tensor]:
        """Build model input"""
        raw_video = get_video(param)
        video = torch.cat([blank, raw_video, blank], dim=1)
        return {
            "inputs": rearrange(preprocess_video(video), "c t h w -> () c t h w"),
            "video": video,
            "raw_video": raw_video,
        }

    return get_batch


def kl_divergence(p: torch.Tensor, q: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p, q = F.softmax(p, dim=dim), F.softmax(q, dim=dim)
    return torch.sum(p * torch.log(p / (q + 1e-8)), dim=dim)


def load_average_freq(mouse_id: str, device: torch.device) -> torch.Tensor:
    natural_freq = np.load(
        data.STATISTICS_DIR / f"mouse{mouse_id}" / "video" / "freq.npy"
    )
    natural_freq = torch.from_numpy(natural_freq).to(device)
    return natural_freq


def loss_function(
    response: torch.Tensor,
    video: torch.Tensor,
    presentation_mask: torch.Tensor,
    loss_mode: str,
    response_type: RESPONSE_TYPE,
    neuron_weights: torch.Tensor,
    natural_freq: torch.Tensor = None,
    alpha: float | torch.Tensor = 0.0,
) -> (torch.Tensor, torch.Tensor):
    response = response[:, presentation_mask]
    match loss_mode:
        case "sum":
            response = torch.sum(response, dim=1)
        case "max":
            response = torch.max(response, dim=1)
        case _:
            raise NotImplementedError(f"Unknown loss mode: {loss_mode}")
    # scale response by neuron weights
    response = response * neuron_weights
    match response_type:
        case "most_exciting":
            loss = -torch.sum(response)
        case "most_inhibiting":
            loss = torch.sum(response)
        case _:
            raise NotImplementedError(f"Unknown response type: {response_type}")
    kl = torch.tensor(0, dtype=torch.float32, device=loss.device)
    if natural_freq is not None:
        video = video[:, presentation_mask, :, :].to(torch.float32)
        freq = torch.fft.fftn(video, dim=(2, 3), norm=FFT_NORM)
        freq = rearrange(freq, "() t h w -> t (h w)")
        freq = torch.square(freq.real) + torch.square(freq.imag)
        freq = torch.log(freq + 1e-8)
        freq = torch.mean(freq, dim=0)
        kl = kl_divergence(p=natural_freq, q=freq)
    if alpha > 0:
        loss = loss + alpha * kl
    return loss, kl


def train_step(
    args: Any,
    param: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: Model,
    get_batch: Callable,
    mouse_id: str,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
    postprocess_response: Callable,
    response_type: RESPONSE_TYPE,
    presentation_mask: torch.Tensor,
    neuron_weights: torch.Tensor,
    natural_freq: torch.Tensor,
    alpha: float | torch.Tensor = 0.0,
) -> dict[str, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    batch = get_batch(param=param)
    t = batch["inputs"].size(2) - SKIP
    response, _ = model(
        inputs=batch["inputs"].to(model.device, model.dtype),
        mouse_id=mouse_id,
        behaviors=behavior.to(model.device, model.dtype),
        pupil_centers=pupil_center.to(model.device, model.dtype),
    )
    response = response.to(torch.float32)[:, :, -t:]
    loss, kl_loss = loss_function(
        response=response[0],
        video=batch["video"][:, -t:, :, :],
        presentation_mask=presentation_mask[-t:],
        loss_mode=args.loss_mode,
        response_type=response_type,
        neuron_weights=neuron_weights,
        natural_freq=natural_freq,
        alpha=alpha,
    )
    loss.backward()
    optimizer.step()
    raw_response = postprocess_response(response.detach())[0]
    response = raw_response[-(BLANK_SIZE + PATTERN_SIZE) : -BLANK_SIZE]
    return {
        "response": raw_response,
        "video": batch["video"],
        "raw_video": batch["raw_video"],
        "loss": loss.detach(),
        "kl_loss": kl_loss.detach(),
        "max_response": response.max(),
        "sum_response": response.sum(),
    }


@torch.inference_mode()
def validate_step(
    args: Any,
    param: torch.Tensor,
    model: Model,
    get_batch: Callable,
    mouse_id: str,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
    postprocess_response: Callable,
    response_type: RESPONSE_TYPE,
    presentation_mask: torch.Tensor,
    neuron_weights: torch.Tensor,
    natural_freq: torch.Tensor,
    alpha: float | torch.Tensor = 0.0,
) -> dict[str, torch.Tensor]:
    batch = get_batch(param=param)
    t = batch["inputs"].size(2) - SKIP
    response, _ = model(
        inputs=batch["inputs"].to(model.device, model.dtype),
        mouse_id=mouse_id,
        behaviors=behavior.to(model.device, model.dtype),
        pupil_centers=pupil_center.to(model.device, model.dtype),
    )
    response = response.to(torch.float32)[:, :, -t:]
    loss, kl_loss = loss_function(
        response=response[0],
        video=batch["video"][:, -t:, :, :],
        presentation_mask=presentation_mask[-t:],
        loss_mode=args.loss_mode,
        response_type=response_type,
        neuron_weights=neuron_weights,
        natural_freq=natural_freq,
        alpha=alpha,
    )
    raw_response = postprocess_response(response)[0]
    response = raw_response[-(BLANK_SIZE + PATTERN_SIZE) : -BLANK_SIZE]
    return {
        "response": raw_response,
        "video": batch["video"],
        "raw_video": batch["raw_video"],
        "loss": loss,
        "kl_loss": kl_loss,
        "max_response": response.max(),
        "sum_response": response.sum(),
    }


def generate_stimulus(
    args: Any,
    mode: MODE,
    model: Model,
    mouse_id: str,
    stimulus_size: int,
    ds: DataLoader,
    save_dir: Path,
    response_type: RESPONSE_TYPE,
    center_video: torch.Tensor | None = None,
):
    device = args.device
    model, dtype = model.to(device), model.dtype
    model.train(False)

    num_steps = args.num_steps
    if center_video is None:
        if args.use_natural_center:
            center_video = get_most_exciting_natural_center(
                args, mouse_id=mouse_id, device=device
            )
            num_steps = 0
        elif args.use_grating_center:
            center_video = get_most_exciting_grating_center(
                args, mouse_id=mouse_id, device=device
            )
            num_steps = 0

    match mode:
        case 0:
            if center_video is None:
                logger.info("Generate center most-exciting stimulus")
            elif args.use_natural_center:
                logger.info("Use most-exciting natural center")
            elif args.use_grating_center:
                logger.info("Use most-exciting grating center")
            save_dir = save_dir / "center"
            num_frames = 1 if args.static_center else PATTERN_SIZE
        case 1:
            logger.info("\nGenerate most-exciting surround stimulus")
            save_dir = save_dir / "most_exciting"
            num_frames = 1 if args.static_surround else PATTERN_SIZE
        case 2:
            logger.info("\nGenerate most-inhibiting surround stimulus")
            save_dir = save_dir / "most_inhibiting"
            num_frames = 1 if args.static_surround else PATTERN_SIZE
        case _:
            raise NotImplementedError(f"Unknown mode: {mode}")
    save_dir.mkdir(parents=True, exist_ok=True)

    channel, _, height, width = model.input_shapes["video"]
    behavior, pupil_center = mes_utils.get_behaviors(
        mouse_id=mouse_id, ds=ds, device=device
    )

    param = None
    if args.natural_init:
        param = get_most_exciting_natural_surround(
            args, mouse_id=mouse_id, response_type=response_type
        )
    param, scale = get_param(
        shape=model.input_shapes["video"],
        num_frames=num_frames,
        method=args.method,
        spatial_cutoff=args.spatial_cutoff if args.method == "cutoff" else None,
        temporal_cutoff=args.temporal_cutoff if args.method == "cutoff" else None,
        sd=args.sd,
        norm=FFT_NORM,
        device=device,
        param=param,
    )
    max_value = torch.tensor(MAX, device=param.device)

    monitor_info = data.MONITOR_INFO[mouse_id]
    circular_mask = stimulus.create_circular_mask(
        stimulus_size=stimulus_size,
        center=stimulus.load_population_RF_center(
            output_dir=args.output_dir,
            mouse_id=mouse_id,
        ),
        pixel_width=width,
        pixel_height=height,
        monitor_width=monitor_info["width"],
        monitor_height=monitor_info["height"],
        monitor_distance=monitor_info["distance"],
        num_frames=PATTERN_SIZE,
        to_tensor=True,
    )
    circular_mask = circular_mask.to(device)

    get_batch = get_transforms(
        ds=ds,
        mouse_id=mouse_id,
        scale=scale,
        max_value=max_value,
        circular_mask=circular_mask,
        video_shape=(channel, PATTERN_SIZE, height, width),
        device=device,
        mode=mode,
        center_video=center_video,
        output_dir=args.output_dir,
        outer_stimulus_size=args.outer_stimulus_size,
    )
    presentation_mask = mes_utils.get_presentation_mask(
        blank_size=BLANK_SIZE, pattern_size=PATTERN_SIZE, device=device
    )
    optimizer = torch.optim.AdamW([param], lr=args.lr, weight_decay=args.weight_decay)

    natural_freq = load_average_freq(mouse_id=mouse_id, device=device)

    step_kwargs = {
        "model": model,
        "get_batch": get_batch,
        "mouse_id": mouse_id,
        "behavior": behavior,
        "pupil_center": pupil_center,
        "postprocess_response": mes_utils.transform_response(ds=ds, device=device),
        "response_type": response_type,
        "presentation_mask": presentation_mask.to(torch.bool),
        "neuron_weights": utils.get_neuron_weights(
            output_dir=args.output_dir, mouse_id=mouse_id
        ).to(device),
        "natural_freq": natural_freq,
        "alpha": torch.tensor(args.alpha, device=device) if args.method == "kl" else 0,
    }

    ckpt_filename = save_dir / "ckpt.pt"
    save_ckpt = lambda p, d, s: mes_utils.save_stimulus_checkpoint(
        param=p,
        data=d,
        step=s,
        mouse_id=mouse_id,
        neuron=None,
        presentation_mask=presentation_mask,
        filename=ckpt_filename,
    )

    step = 0

    start = -(BLANK_SIZE + PATTERN_SIZE + 10)
    stop = -(BLANK_SIZE - 10)

    val_result = validate_step(args, param=param, **step_kwargs)
    best_value, best_step = val_result["sum_response"], 0
    if args.animate:
        animate_stimulus(
            video=val_result["video"][:, start:stop],
            response=torch.mean(val_result["response"], dim=0)[start:stop],
            loss=val_result["loss"],
            filename=save_dir / "plots" / f"step{step:04d}.{args.format}",
            presentation_mask=presentation_mask[start:stop],
            skip=0,
        )
    if num_steps == 0:
        save_ckpt(param, val_result, step)
    if args.verbose > 1:
        logger.info(
            f"Initial Loss: {val_result['loss']:.02f}, "
            f"KL Loss: {val_result['kl_loss']:.02f}, "
            f"Sum: {val_result['sum_response']:.1f}"
        )

    has_improved = lambda current, best: (
        (current > best) if mode in (0, 1) else (current < best)
    )

    try:
        with logging_redirect_tqdm(loggers=[logger]):
            for step in tqdm(range(1, num_steps + 1)):
                train_result = train_step(
                    args, param=param, optimizer=optimizer, **step_kwargs
                )
                if step % (10 if num_steps <= 100 else 200) == 0 and args.verbose > 1:
                    logger.info(
                        f"Step {step}/{args.num_steps}, "
                        f"Loss: {train_result['loss']:.2f}, "
                        f'KL loss: {train_result["kl_loss"]:.2f}, '
                        f"Sum: {train_result['sum_response']:.1f}"
                    )
                if args.animate and step % 100 == 0:
                    animate_stimulus(
                        video=train_result["video"][:, start:stop],
                        response=torch.mean(train_result["response"], dim=0)[
                            start:stop
                        ],
                        loss=train_result["loss"],
                        filename=save_dir / "plots" / f"step{step:04d}.{args.format}",
                        presentation_mask=presentation_mask[start:stop],
                        skip=0,
                    )
                if has_improved(current=train_result["sum_response"], best=best_value):
                    best_value, best_step = train_result["sum_response"], step
                    save_ckpt(param, train_result, step)
    except KeyboardInterrupt:
        logger.info(f"Interrupted optimization at step {step}.")

    logger.info(f"Best value: {best_value:.1f} at step {best_step}.")

    if ckpt_filename.exists():
        ckpt = torch.load(ckpt_filename, map_location="cpu", weights_only=True)
        param.data = ckpt["param"].to(device)

    final_result = validate_step(args, param=param, **step_kwargs)
    animate_stimulus(
        video=final_result["video"][:, start:stop],
        response=torch.mean(final_result["response"], dim=0)[start:stop],
        loss=final_result["loss"],
        filename=save_dir / f"best.{args.format}",
        presentation_mask=presentation_mask[start:stop],
        skip=0,
    )
    if args.verbose:
        logger.info(
            f"Final Loss: {final_result['loss']:.02f}, "
            f"KL Loss: {final_result['kl_loss']:.2f}, "
            f"Sum: {final_result['sum_response']:.1f}"
        )
    # return center video
    return final_result["raw_video"]


def generate_center_surround_stimulus(
    args: Any,
    model: Model,
    mouse_id: str,
    ds: DataLoader,
    save_dir: Path,
):
    save_dir = save_dir / f"mouse{mouse_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"\n\nGenerate center-surround most-exciting and most-inhibiting "
        f"{'static' if args.static_surround else 'dynamic'} stimuli "
        f"for mouse {mouse_id}. (outer stimulus size {args.outer_stimulus_size}, "
        f"natural_init: {args.natural_init})\n"
    )
    utils.set_random_seed(seed=args.seed)

    # generate the most exciting center or use most exciting natural frame/clip
    center_video = generate_stimulus(
        args,
        mode=0,
        model=model,
        mouse_id=mouse_id,
        stimulus_size=args.stimulus_size,
        ds=ds,
        save_dir=save_dir,
        response_type="most_exciting",
    )
    # generate the most exciting surround given fixed center
    generate_stimulus(
        args,
        mode=1,
        model=model,
        mouse_id=mouse_id,
        stimulus_size=args.stimulus_size,
        ds=ds,
        save_dir=save_dir,
        center_video=center_video,
        response_type="most_exciting",
    )
    # generate the least exciting surround given fixed center
    generate_stimulus(
        args,
        mode=2,
        model=model,
        mouse_id=mouse_id,
        stimulus_size=args.stimulus_size,
        ds=ds,
        save_dir=save_dir,
        center_video=center_video,
        response_type="most_inhibiting",
    )


def main(args):
    if args.use_natural_center:
        center = "natural"
    elif args.use_grating_center:
        center = "grating"
    else:
        center = "generated"
    save_dir = (
        args.output_dir
        / "most_exciting_stimulus"
        / "population"
        / "generated"
        / "center_surround"
        / f"{center}_center"
        / f"{'static' if args.static_center else 'dynamic'}_center_{'static' if args.static_surround else 'dynamic'}_surround"
        / args.save_dir
    )
    if args.overwrite and save_dir.is_dir():
        rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    args.device = utils.get_device(args.device)
    model, ds = load_model(
        args,
        evaluate=args.evaluate,
        compile=args.compile,
        grad_checkpointing=args.grad_checkpointing,
    )
    utils.save_args(args, output_dir=save_dir)

    utils.set_logger_handles(
        logger=logger, filename=save_dir / "output.log", level=logging.INFO
    )

    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM_OLD)

    model.train(False)
    model.requires_grad_(requires_grad=False)

    for mouse_id in args.mouse_ids:
        generate_center_surround_stimulus(
            args,
            model=model,
            mouse_id=mouse_id,
            ds=ds[mouse_id],
            save_dir=save_dir,
        )

    logger.info(f"\n\nSaved results to {save_dir}.")


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
        "--save_dir",
        type=Path,
        required=True,
        help="name of the directory to store the generated stimuli",
    )
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument(
        "--use_natural_center",
        action="store_true",
        help="Use the most exciting center frame or clip extracted from the Sensorium 2023 training set.",
    )
    parser.add_argument(
        "--use_grating_center",
        action="store_true",
        help="Use the most exciting grating center for that neuron.",
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
        help="stimulus size to use if use_preferred_size is not set.",
    )
    parser.add_argument(
        "--use_RF_center",
        action="store_true",
        help="Use the estimate aRF center as the position of the circular center mask.",
    )
    parser.add_argument(
        "--natural_init",
        action="store_true",
        help="Initialize image/video with the most-exciting/most-inhibiting natural image/video instead of white noise.",
    )
    parser.add_argument("--static_center", action="store_true")
    parser.add_argument("--static_surround", action="store_true")
    parser.add_argument(
        "--sd",
        type=float,
        default=0.01,
        help="Parameter initialize noise standard deviation.",
    )
    parser.add_argument("--loss_mode", type=str, default="sum", choices=["sum", "max"])
    parser.add_argument(
        "--animate",
        action="store_true",
        help="plot intermediate results",
    )
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite save_dir if exists."
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
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable gradient checkpointing for supported models if set to 1.",
    )
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument(
        "--method", type=str, choices=["pixel", "kl", "cutoff"], required=True
    )

    temp_args = parser.parse_known_args()[0]
    match temp_args.method:
        case "pixel":
            pass
        case "kl":
            parser.add_argument(
                "--alpha",
                type=float,
                default=1.0,
                help="Alpha value for KL divergence loss penalty",
            )
        case "cutoff":
            parser.add_argument(
                "--spatial_cutoff",
                type=float,
                default=0.5,
                help="the percentage of lower frequency in the spatial dimension to retain.",
            )
            parser.add_argument(
                "--temporal_cutoff",
                type=float,
                default=1.0,
                help="the percentage of lower frequency in the temporal dimension to retain.",
            )
        case _:
            raise ValueError(f"Unknown method: {temp_args.method}")
    del temp_args

    main(parser.parse_args())
