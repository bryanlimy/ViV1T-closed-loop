import argparse
from argparse import RawTextHelpFormatter
from itertools import product
from pathlib import Path

import numpy as np
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

plot.set_font()


# resolution of the video
VIDEO_H, VIDEO_W = 36, 64
MIN, MAX = 0, 255  # min and max pixel values
GREY_COLOR = (MAX - MIN) // 2
FPS = 30

NUM_REPEATS = 12  # number of repeats for each unique pattern
BLANK_SIZE, PATTERN_SIZE = 15, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE + BLANK_SIZE
MAX_FRAME = data.MAX_FRAME
PATTERNS_PER_TRIAL = 4
SKIP = MAX_FRAME - PATTERNS_PER_TRIAL * BLOCK_SIZE  # number of initial frames to skip
assert SKIP >= 50, "Skip frames must be more than 50 as per Sensorium 2023"


def create_stimulus(args) -> tuple[torch.Tensor, np.ndarray]:
    """
    Create classical and inverse grating stimulus

    Returns:
        gratings: torch.Tensor, (num. trials, PATTERN_PER_TRIAL, C, PATTERN_SIZE, H, W)
        parameters: np.ndarray, (num. trials, PATTERN_PER_TRIAL, 3)
    """
    rng = np.random.default_rng(args.seed)
    stim_sizes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    directions = [90, 270]  # only generate horizontal directions
    stim_types = [0, 1]  # stimulus type: 0 - classical, 1 - inverse
    parameters = np.array(
        list(product(stim_sizes, directions, stim_types)),
        dtype=np.float32,
    )
    num_patterns = len(parameters) * NUM_REPEATS
    assert (
        num_patterns % PATTERNS_PER_TRIAL
    ) == 0, f"The number of patterns ({num_patterns}) must be divisible by {PATTERNS_PER_TRIAL}"
    # repeat each unique pattern NUM_REPEATS times
    parameters = repeat(
        parameters, "block param -> (repeat block) param", repeat=NUM_REPEATS
    )
    # shuffle pattern orders
    pattern_ids = rng.permutation(np.arange(len(parameters), dtype=int))
    parameters = parameters[pattern_ids]
    # create full-field gratings different directions
    C, H, W = 1, VIDEO_H, VIDEO_W
    gratings = torch.zeros((len(parameters), C, PATTERN_SIZE, H, W), dtype=torch.uint8)

    cpd = 0.04  # spatial frequency
    cpf = 2 / FPS  # temporal frequency of 2Hz
    for i in range(parameters.shape[0]):
        stim_size, direction, stim_type = parameters[i]
        grating = stimulus.create_full_field_grating(
            direction=direction,
            cpd=cpd,
            cpf=cpf,
            num_frames=PATTERN_SIZE,
            height=H,
            width=W,
            phase=rng.choice(360, size=1).item(),  # randomize initial phase
            contrast=1.0,
            fps=FPS,
            to_tensor=True,
        )
        gratings[i] = grating
        del grating, stim_size, direction, stim_type
    # rearrange gratings to show PATTERNS_PER_TRIAL number of patterns per trial
    gratings = rearrange(
        gratings,
        "(trial block) C frame H W -> trial block C frame H W",
        block=PATTERNS_PER_TRIAL,
    )
    parameters = rearrange(
        parameters,
        "(trial block) param -> trial block param",
        block=PATTERNS_PER_TRIAL,
    )
    # shuffle trials
    num_trials = len(parameters)
    trial_ids = rng.permutation(np.arange(num_trials, dtype=int))
    gratings, parameters = gratings[trial_ids], parameters[trial_ids]
    return gratings, parameters


class GratingDataset(Dataset):
    def __init__(
        self,
        mouse_id: str,
        neurons: np.ndarray,
        RF_center: tuple[int, int] | np.ndarray,
        gratings: torch.Tensor,
        parameters: np.ndarray,
        dataset: MovieDataset,
        device: torch.device,
    ):
        self.mouse_id = mouse_id
        self.neurons = neurons
        self.num_neurons = dataset.num_neurons

        self.gratings = gratings.to(torch.float32)
        self.parameters = parameters

        # create circular masks with different sizes
        monitor_info = data.MONITOR_INFO[mouse_id]
        C, T, H, W = self.gratings.shape[2:]
        stimulus_sizes = np.unique(self.parameters[:, :, 0])
        self.circular_masks = {
            stimulus_size: stimulus.create_circular_mask(
                stimulus_size=stimulus_size,
                center=RF_center,
                pixel_width=W,
                pixel_height=H,
                monitor_width=monitor_info["width"],
                monitor_height=monitor_info["height"],
                monitor_distance=monitor_info["distance"],
                num_frames=T,
                to_tensor=True,
            )
            for stimulus_size in stimulus_sizes
        }
        # blank screen to fill the beginning with a trial
        self.skip_blank = torch.full(
            (C, SKIP, H, W), fill_value=GREY_COLOR, dtype=torch.float32
        )
        # blank screen before and after each pattern presentation
        self.blank = torch.full(
            (C, BLANK_SIZE, H, W), fill_value=GREY_COLOR, dtype=torch.float32
        )

        self.transform_video = dataset.transform_video

        # load response statistics to device for quicker inference
        self.transform_output = dataset.transform_output
        self.response_stats = {
            k: v.to(device) for k, v in dataset.response_stats.items()
        }
        self.response_precision = dataset.response_precision.to(device)

        behavior, pupil_center = data.get_mean_behaviors(
            mouse_id=mouse_id, num_frames=data.MAX_FRAME
        )
        self.behavior = dataset.transform_behavior(behavior)
        self.pupil_center = dataset.transform_pupil_center(pupil_center)

    def __len__(self):
        return len(self.parameters)

    def i_transform_response(self, response: torch.Tensor) -> torch.Tensor:
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response / self.response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    def prepare_video(self, idx: int | torch.Tensor) -> torch.tensor:
        """Combine grating pattern blocks to a single trial of MAX_FRAME length"""
        gratings, parameters = self.gratings[idx], self.parameters[idx]
        video = [self.skip_blank]
        for i in range(gratings.shape[0]):
            grating, parameter = gratings[i], parameters[i]
            stimulus_size, stimulus_type = parameter[0], parameter[2]
            circular_mask = self.circular_masks[stimulus_size]
            if stimulus_type == 1:
                circular_mask = ~circular_mask
            grating = torch.where(circular_mask, grating, GREY_COLOR)
            grating = torch.concat([self.blank, grating, self.blank], dim=1)
            video.append(grating)
        video = torch.concat(video, dim=1)
        return video

    def __getitem__(
        self, idx: int | torch.Tensor, to_tensor: bool = True
    ) -> dict[str, torch.Tensor | str]:
        video = self.prepare_video(idx)
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
    ds: DataLoader,
    mouse_id: str,
    neurons: np.ndarray,
    animate: bool = False,
    save_dir: Path = None,
    desc: str = "",
) -> torch.Tensor:
    device, dtype = model.device, model.dtype
    dataset: GratingDataset = ds.dataset
    num_batches, num_trials = len(ds), len(dataset)
    num_frames = MAX_FRAME - SKIP
    responses = torch.zeros((len(neurons), num_trials, num_frames), dtype=torch.float32)
    index = 0
    for i, batch in enumerate(tqdm(ds, desc=desc)):
        response, _ = model(
            inputs=batch["video"].to(device, dtype),
            mouse_id=mouse_id,
            behaviors=batch["behavior"].to(device, dtype),
            pupil_centers=batch["pupil_center"].to(device, dtype),
        )
        response = response.to(torch.float32)
        response = dataset.i_transform_response(response).cpu()
        # if animate and np.random.random() <= 0.001:
        if animate:
            plot.animate_stimulus(
                video=batch["raw_video"][0, :, -num_frames:].numpy(),
                response=response[0, neurons[0], -num_frames:].numpy(),
                neuron=neurons[0],
                filename=save_dir
                / "figures"
                / f"mouse{mouse_id}"
                / f"neuron{neurons[0]:04d}_trial{i:04d}.mp4",
                skip=0,
            )
        batch_size = len(response)
        response = rearrange(response[:, neurons, -num_frames:], "B N T -> N B T")
        responses[:, index : index + batch_size, :] = response
        index += batch_size
    return responses


def process_mouse(
    args,
    model: Model,
    dataset: MovieDataset,
    mouse_id: str,
    gratings: torch.Tensor,
    parameters: np.ndarray,
    save_dir: Path,
):
    num_neurons = dataset.num_neurons
    num_trials = gratings.shape[0]
    num_frames = MAX_FRAME - SKIP
    responses = np.zeros((num_neurons, num_trials, num_frames), dtype=np.float32)
    if args.use_RF_center:
        RF_centers, neuron_groups = stimulus.load_group_RF_center(
            output_dir=args.output_dir, mouse_id=mouse_id
        )
    else:
        # use center of the video
        RF_centers = np.array([[VIDEO_W / 2, VIDEO_H / 2]])
        neuron_groups = np.zeros(num_neurons, dtype=int)
    for i in range(len(RF_centers)):
        neurons = np.where(neuron_groups == i)[0]
        if np.any(np.isnan(RF_centers[i])):
            # set response to NaN for neurons that do not have good aRF fit
            responses[neurons] = np.nan
            continue
        ds = DataLoader(
            dataset=GratingDataset(
                mouse_id=mouse_id,
                neurons=neurons,
                RF_center=RF_centers[i],
                gratings=gratings,
                parameters=parameters,
                dataset=dataset,
                device=args.device,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )
        response = inference(
            model=model,
            ds=ds,
            mouse_id=mouse_id,
            neurons=neurons,
            animate=args.animate,
            save_dir=save_dir,
            desc=f"Mouse {mouse_id} N={len(neurons)} ({i+1:03d}/{len(RF_centers):03d})",
        )
        responses[neurons] = response
        del response
    # group response based on pattern so that it has the same shape as parameters
    responses = rearrange(
        responses,
        "neuron trial (block frame) -> neuron trial block frame",
        block=PATTERNS_PER_TRIAL,
        frame=BLOCK_SIZE,
    )
    np.savez_compressed(
        file=save_dir / f"mouse{mouse_id}.npz",
        data=responses,
        allow_pickle=False,
    )


def main(args):
    utils.set_random_seed(args.seed)
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"Cannot find output_dir {args.output_dir}.")

    save_dir = args.output_dir / "feedbackRF"
    save_dir.mkdir(parents=True, exist_ok=True)

    args.device = utils.get_device(args.device)
    model, ds = load_model(args, evaluate=args.evaluate, compile=args.compile)
    model.train(False)

    gratings, parameters = create_stimulus(args)
    np.savez_compressed(
        file=save_dir / "gratings.npz", data=gratings, allow_pickle=False
    )
    np.save(file=save_dir / "parameters.npy", arr=parameters, allow_pickle=False)

    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM_OLD)

    print(f"Save result to {save_dir}.")

    for mouse_id in args.mouse_ids:
        process_mouse(
            args,
            model=model,
            dataset=ds[mouse_id].dataset,
            mouse_id=mouse_id,
            gratings=gratings.clone(),
            parameters=parameters.copy(),
            save_dir=save_dir,
        )

    print(f"Save prediction to {save_dir}.\n\n")


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
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate the model on the validation set after loading the checkpoint.",
    )
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
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
