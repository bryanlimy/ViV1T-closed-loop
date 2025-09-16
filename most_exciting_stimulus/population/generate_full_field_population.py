import argparse
import logging
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.fft
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
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

# flashing images were presented from 15 frames (0.5s) in the Sensorium 2023 dataset
# so here we also use 15 frames.
PATTERN_SIZE = 30
BLANK_SIZE1 = int(np.ceil((data.MAX_FRAME - PATTERN_SIZE) / 2))
BLANK_SIZE2 = int(np.floor((data.MAX_FRAME - PATTERN_SIZE) / 2))

MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) / 2

FFT_NORM = "ortho"
RESPONSE_TYPE = Literal["most_exciting", "most_inhibiting"]

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def get_most_exciting_natural_full_field(
    args, mouse_id: str, response_type: str
) -> torch.Tensor:
    filename = (
        args.output_dir
        / "most_exciting_stimulus"
        / "population"
        / "natural"
        / "full_field"
        / ("static" if args.static else "dynamic")
        / f"mouse{mouse_id}.parquet"
    )
    df = pd.read_parquet(filename)
    df = df[(df.mouse == mouse_id) & (df.response_type == response_type)]
    assert df.trial.nunique() == df.frame.nunique() == 1
    trial_id, frame = df.iloc[0].trial, df.iloc[0].frame
    video = np.load(
        args.data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    if args.static:
        video = repeat(video[:, :, frame], "h w -> h w t", t=1)
    else:
        video = video[:, :, frame : frame + PATTERN_SIZE]
    video = rearrange(video, "h w t -> () t h w")
    video = torch.from_numpy(video).to(torch.float32)
    # standardize video
    video = (video - torch.mean(video)) / torch.std(video)
    return video


def load_video(data_dir: Path, mouse_id: str, trial_id: int) -> np.ndarray:
    video = np.load(
        data_dir / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    video = video[..., : data.MAX_FRAME]
    video = np.round(video, decimals=0)
    video = rearrange(video, "h w t -> () t h w")
    return video


def get_transforms(
    ds: DataLoader,
    scale: torch.Tensor,
    max_value: torch.Tensor,
    video_shape: tuple[int, int, int, int],
    device: torch.device,
    mouse_id: str,
    use_natural_blank: bool = False,
    data_dir: Path = None,
    output_dir: Path = None,
    outer_stimulus_size: int | None = None,
):
    preprocess_video = mes_utils.transform_video(ds=ds, device=device)
    assert BLANK_SIZE1 + PATTERN_SIZE + BLANK_SIZE2 == data.MAX_FRAME
    blank1 = torch.full(
        (video_shape[0], BLANK_SIZE1, video_shape[2], video_shape[3]),
        fill_value=GREY_COLOR,
        device=device,
        requires_grad=False,
    )
    blank2 = torch.full(
        (video_shape[0], BLANK_SIZE2, video_shape[2], video_shape[3]),
        fill_value=GREY_COLOR,
        device=device,
        requires_grad=False,
    )
    if use_natural_blank:
        # replace before and after stimulus presentation with the before and after
        # of most exciting natural video clip in Sensorium 2023
        filename = output_dir / "most_exciting_stimulus" / "population" / "videos.pkl"
        assert filename.exists(), f"Cannot find {filename.resolve()}"
        df = pd.read_pickle(filename)
        df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
        df = df.loc[:, ["video_id", "frame_id", "response"]]
        df = df.sort_values(by="response", ascending=False).iloc[0]
        video_id, frame_id = int(df.video_id), int(df.frame_id)
        video_ids = data.get_video_ids(mouse_id=mouse_id)
        trial_id = np.where(video_ids == video_id)[0][0]
        video = load_video(data_dir=data_dir, mouse_id=mouse_id, trial_id=trial_id)
        size = 10
        before = video[:, (frame_id - size) : frame_id]
        after = video[:, (frame_id + PATTERN_SIZE) : (frame_id + PATTERN_SIZE + size)]
        blank1[:, -size:] = torch.from_numpy(before)
        blank2[:, :size] = torch.from_numpy(after)
    grey_color = torch.tensor(GREY_COLOR, dtype=torch.float32, device=device)
    outer_circular_mask = None
    if outer_stimulus_size is not None:
        monitor_info = data.MONITOR_INFO[mouse_id]
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
        if outer_circular_mask is not None:
            video = torch.where(outer_circular_mask, video, grey_color)
        video = torch.cat([blank1, video, blank2], dim=1)
        return video

    def get_batch(
        param: torch.Tensor,
        behavior: torch.Tensor,
        pupil_center: torch.Tensor,
        mouse_id: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[dict[str, torch.Tensor | str], torch.Tensor]:
        """Build model input"""
        video = get_video(param)
        return {
            "inputs": preprocess_video(video)[None, ...].to(device, dtype),
            "mouse_id": mouse_id,
            "behaviors": behavior.to(device, dtype),
            "pupil_centers": pupil_center.to(device, dtype),
        }, video

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
    batch, video = get_batch(
        param=param,
        mouse_id=mouse_id,
        behavior=behavior,
        pupil_center=pupil_center,
        device=model.device,
        dtype=model.dtype,
    )
    t = batch["inputs"].size(2) - SKIP
    response, _ = model(**batch)
    response = response.to(torch.float32)[:, :, -t:]
    loss, kl_loss = loss_function(
        response=response[0],
        video=video[:, -t:, :, :],
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
    # get response during stimulus presentation
    response = raw_response[:, -(BLANK_SIZE2 + PATTERN_SIZE) : -BLANK_SIZE2]
    return {
        "response": raw_response,
        "video": video,
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
    batch, video = get_batch(
        param=param,
        mouse_id=mouse_id,
        behavior=behavior,
        pupil_center=pupil_center,
        device=model.device,
        dtype=model.dtype,
    )
    t = batch["inputs"].size(2) - SKIP
    response, _ = model(**batch)
    response = response.to(torch.float32)[:, :, -t:]
    loss, kl_loss = loss_function(
        response=response[0],
        video=video[:, -t:, :, :],
        presentation_mask=presentation_mask[-t:],
        loss_mode=args.loss_mode,
        response_type=response_type,
        neuron_weights=neuron_weights,
        natural_freq=natural_freq,
        alpha=alpha,
    )
    raw_response = postprocess_response(response)[0]
    # get response during stimulus presentation
    response = raw_response[:, -(BLANK_SIZE2 + PATTERN_SIZE) : -BLANK_SIZE2]
    return {
        "response": raw_response,
        "video": video,
        "loss": loss,
        "kl_loss": kl_loss,
        "max_response": response.max(),
        "sum_response": response.sum(),
    }


def generate_stimulus(
    args: Any,
    model: Model,
    mouse_id: str,
    ds: DataLoader,
    save_dir: Path,
    response_type: RESPONSE_TYPE,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    assert response_type in ("most_exciting", "most_inhibiting")
    logger.info(
        f"\n\nGenerate full-field {response_type.replace('_', '-')} "
        f"{'static' if args.static else 'dynamic'} stimulus for mouse "
        f"{mouse_id} population. (outer stimulus size "
        f"{args.outer_stimulus_size}, natural_init: {args.natural_init})\n"
    )

    utils.set_random_seed(args.seed)

    device = args.device
    channel, _, height, width = model.input_shapes["video"]
    behavior, pupil_center = mes_utils.get_behaviors(
        mouse_id=mouse_id, ds=ds, device=device
    )

    param = None
    if args.natural_init:
        param = get_most_exciting_natural_full_field(
            args, mouse_id=mouse_id, response_type=response_type
        )
    param, scale = get_param(
        shape=model.input_shapes["video"],
        num_frames=1 if args.static else PATTERN_SIZE,
        method=args.method,
        spatial_cutoff=args.spatial_cutoff if args.method == "cutoff" else None,
        temporal_cutoff=args.temporal_cutoff if args.method == "cutoff" else None,
        sd=args.sd,
        norm=FFT_NORM,
        device=device,
        param=param,
    )
    max_value = torch.tensor(MAX, device=param.device)

    get_batch = get_transforms(
        ds=ds,
        scale=scale,
        max_value=max_value,
        video_shape=(channel, PATTERN_SIZE, height, width),
        device=device,
        mouse_id=mouse_id,
        use_natural_blank=args.use_natural_blank,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        outer_stimulus_size=args.outer_stimulus_size,
    )
    presentation_mask = torch.cat(
        [
            torch.zeros(BLANK_SIZE1, device=device),
            torch.ones(PATTERN_SIZE, device=device),
            torch.zeros(BLANK_SIZE2, device=device),
        ]
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

    start = -(BLANK_SIZE2 + PATTERN_SIZE + 20)
    stop = -(BLANK_SIZE2 - 20)

    val_result = validate_step(args, param=param, **step_kwargs)
    best_value, best_step = val_result["sum_response"], 0
    if args.animate:
        animate_stimulus(
            video=val_result["video"].cpu().numpy()[:, start:stop],
            response=torch.mean(val_result["response"], dim=0)
            .cpu()
            .numpy()[start:stop],
            filename=save_dir / "plot" / f"step{step:04d}.{args.format}",
            presentation_mask=presentation_mask.cpu().numpy()[start:stop],
            skip=0,
        )
    if args.verbose > 1:
        logger.info(
            f"Initial Loss: {val_result['loss']:.02f}, "
            f"KL Loss: {val_result['kl_loss']:.02f}, "
            f"Sum: {val_result['sum_response']:.1f}"
        )

    has_improved = lambda current, best: (
        (current > best) if response_type == "most_exciting" else (current < best)
    )

    num_steps = args.num_steps
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
                        video=train_result["video"].cpu().numpy()[:, start:stop],
                        response=torch.mean(train_result["response"], dim=0)
                        .cpu()
                        .numpy()[start:stop],
                        filename=save_dir / "plot" / f"step{step:04d}.{args.format}",
                        presentation_mask=presentation_mask.cpu().numpy()[start:stop],
                        skip=0,
                    )
                if has_improved(current=train_result["sum_response"], best=best_value):
                    best_value, best_step = train_result["sum_response"], step
                    save_ckpt(p=param, d=train_result, s=step)

    except KeyboardInterrupt:
        logger.info(f"Interrupted optimization at step {step}.")

    logger.info(f"Best value: {best_value:.1f} at step {best_step}.")

    if ckpt_filename.exists():
        ckpt = torch.load(ckpt_filename, map_location="cpu", weights_only=True)
        param.data = ckpt["param"].to(device)

    final_result = validate_step(args, param=param, **step_kwargs)
    animate_stimulus(
        video=final_result["video"].cpu().numpy()[:, start:stop],
        response=torch.mean(final_result["response"], dim=0).cpu().numpy()[start:stop],
        filename=save_dir / f"best.{args.format}",
        presentation_mask=presentation_mask.cpu().numpy()[start:stop],
        skip=0,
    )
    if args.verbose:
        logger.info(
            f"Final Loss: {final_result['loss']:.02f}, "
            f"KL Loss: {final_result['kl_loss']:.2f}, "
            f"Sum: {final_result['sum_response']:.1f}"
        )
    logger.info(f"Saved results to {save_dir}.")


def main(args: Any):
    args.device = utils.get_device(args.device)
    model, ds = load_model(
        args,
        evaluate=args.evaluate,
        compile=args.compile,
        grad_checkpointing=args.grad_checkpointing,
    )
    model.train(False)

    save_dir = (
        args.output_dir
        / "most_exciting_stimulus"
        / "population"
        / "generated"
        / "full_field"
        / ("static" if args.static else "dynamic")
        / args.save_dir
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    utils.save_args(args, output_dir=save_dir)
    utils.set_logger_handles(
        logger=logger, filename=save_dir / "output.log", level=logging.INFO
    )

    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM_OLD)

    for mouse_id in args.mouse_ids:
        generate_stimulus(
            args,
            model=model,
            mouse_id=mouse_id,
            ds=ds[mouse_id],
            save_dir=save_dir / f"mouse{mouse_id}" / "most_exciting",
            response_type="most_exciting",
        )
        generate_stimulus(
            args,
            model=model,
            mouse_id=mouse_id,
            ds=ds[mouse_id],
            save_dir=save_dir / f"mouse{mouse_id}" / "most_inhibiting",
            response_type="most_inhibiting",
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
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="name of the directory to store the generated stimuli",
    )
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
    parser.add_argument(
        "--static",
        action="store_true",
        help="generate static stimulus and repeat it for the duration of the presentation.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--sd", type=float, default=0.01)
    parser.add_argument("--loss_mode", type=str, default="sum", choices=["sum", "max"])
    parser.add_argument(
        "--animate", action="store_true", help="plot intermediate results"
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

    parser.add_argument("--use_natural_blank", action="store_true")
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
