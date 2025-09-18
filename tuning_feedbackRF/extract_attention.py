import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t import Checkpoint
from viv1t import data
from viv1t.data import SENSORIUM_OLD
from viv1t.data import get_training_ds
from viv1t.data.data import MovieDataset
from viv1t.model import Model
from viv1t.model.core.vivit import ViViTCore
from viv1t.utils import h5
from viv1t.utils import utils
from viv1t.utils.attention_rollout import Recorder
from viv1t.utils.attention_rollout import spatial_attention_rollout
from viv1t.utils.attention_rollout import temporal_attention_rollout

SKIP = 50  # skip the first 50 frames from each trial
MAX_FRAME = 300
SENSORIUM_DIR = Path("../data/sensorium")


class GratingDataset(MovieDataset):
    def __init__(
        self,
        tier: str,
        data_dir: Path,
        mouse_id: str,
        transform_input: int,
        transform_output: int,
        crop_frame: int = -1,
        center_crop: float = 1.0,
        limit_data: int = None,
        verbose: int = 0,
    ):
        super(GratingDataset, self).__init__(
            tier=tier,
            data_dir=SENSORIUM_DIR,
            mouse_id=mouse_id,
            transform_input=transform_input,
            transform_output=transform_output,
            crop_frame=crop_frame,
            center_crop=center_crop,
            limit_data=limit_data,
            verbose=verbose,
        )
        self.data_dir = data_dir

        meta_dir = data_dir / "meta" / "trials"
        self.video_ids = np.load(meta_dir / "video_ids.npy")
        self.tiers = np.load(meta_dir / "tiers.npy").astype(np.string_)
        self.stimulus_ids = np.full_like(self.video_ids, 6, dtype=np.int32)
        self.trial_ids = np.arange(len(self.video_ids), dtype=np.int32)

        # load behavior and pupil center statistics which are the mean values
        # calculated from the training set
        self.behavior = np.load(
            data_dir / "data" / "behavior" / f"{data.MOUSE_IDS[mouse_id]}.npy"
        ).astype(np.float32)
        self.pupil_center = np.load(
            data_dir / "data" / "pupil_center" / f"{data.MOUSE_IDS[mouse_id]}.npy"
        ).astype(np.float32)

    def load_sample(self, trial_id: int | str | torch.Tensor, to_tensor: bool = False):
        """
        Load sample from disk and apply transformation

        The Sensorium 2023 challenge only consider the first 300 frames, even
        though some trials have more than 300 frames
        """
        t = MAX_FRAME

        sample = dict(duration=t)
        sample["video"] = np.load(
            self.data_dir / "data" / "videos" / f"{trial_id}.npy"
        ).astype(np.float32)
        sample["video"] = rearrange(sample["video"], "h w t -> 1 t h w")

        sample["behavior"] = np.copy(self.behavior)
        sample["pupil_center"] = np.copy(self.pupil_center)

        # set responses to zeros
        sample["response"] = np.zeros((self.num_neurons, t), dtype=np.float32)

        if to_tensor:
            for k, v in sample.items():
                if isinstance(v, np.ndarray):
                    sample[k] = torch.from_numpy(v)

        # crop to max frames if trial is longer
        sample["video"] = sample["video"][:, :t]
        sample["response"] = sample["response"][:, :t]
        sample["behavior"] = sample["behavior"][:, :t]
        sample["pupil_center"] = sample["pupil_center"][:, :t]
        if self.transform_input:
            sample["video"] = self.transform_video(sample["video"])
            sample["behavior"] = self.transform_behavior(sample["behavior"])
            sample["pupil_center"] = self.transform_pupil_center(sample["pupil_center"])
        if self.transform_output:
            sample["response"] = self.transform_response(sample["response"])
        return sample


@torch.inference_mode()
def forward(
    core: ViViTCore,
    spatial_transformer: Recorder,
    temporal_transformer: Recorder,
    mouse_id: str,
    video: torch.Tensor,
    behavior: torch.Tensor,
    pupil_center: torch.Tensor,
):
    outputs = video
    outputs = core.tokenizer(outputs, behavior, pupil_center)
    b, t, p, _ = outputs.shape

    reg_tokens = core.vivit.reg_tokens > 0
    if reg_tokens:
        outputs = core.vivit.add_spatial_reg_tokens(outputs)

    outputs = rearrange(outputs, "b t p c -> (b t) p c")
    outputs, spatial_attentions = spatial_transformer(
        inputs=outputs,
        mouse_id=mouse_id,
        behaviors=behavior,
        pupil_centers=pupil_center,
    )
    outputs = rearrange(outputs, "(b t) p c -> b t p c", b=b)

    if reg_tokens:
        outputs = core.vivit.remove_spatial_reg_tokens(outputs)
        outputs = core.vivit.add_temporal_reg_tokens(outputs)

    outputs = rearrange(outputs, "b t p c -> (b p) t c")
    outputs, temporal_attentions = temporal_transformer(
        inputs=outputs,
        mouse_id=mouse_id,
        behaviors=behavior,
        pupil_centers=pupil_center,
    )
    outputs = rearrange(outputs, "(b p) t c -> b t p c", b=b)

    if reg_tokens:
        outputs = core.vivit.remove_temporal_reg_tokens(outputs)

    outputs = core.rearrange(outputs)
    outputs = core.activation(outputs)

    # remove register tokens from attention matrix
    if reg_tokens:
        spatial_attentions = spatial_attentions[:, :, :, :p, :p]
        temporal_attentions = temporal_attentions[:, :, :, :t, :t]

    return outputs, spatial_attentions, temporal_attentions


@torch.inference_mode()
def inference(
    ds: DataLoader,
    model: Model,
    save_dir: Path,
    device: torch.device = "cpu",
):
    """
    Inference model on the grating patterns with behavior and pupil center
    from the validation sets. The average response per trial from all the
    behaviors from the validation set are stored to save_dir.
    """
    model.to(device)
    model.train(False)
    mouse_id = ds.dataset.mouse_id

    i_transform_videos = ds.dataset.i_transform_videos

    core = model.core
    assert isinstance(core, ViViTCore)

    spatial_transformer = Recorder(core.vivit.spatial_transformer)
    temporal_transformer = Recorder(core.vivit.temporal_transformer)

    for batch in tqdm(ds, desc=f"mouse {mouse_id}"):
        _, _, t, h, w = batch["video"].shape
        output, spatial_attention, temporal_attention = forward(
            core=core,
            spatial_transformer=spatial_transformer,
            temporal_transformer=temporal_transformer,
            mouse_id=mouse_id,
            video=batch["video"].to(device, model.dtype),
            behavior=batch["behavior"].to(device, model.dtype),
            pupil_center=batch["pupil_center"].to(device, model.dtype),
        )
        # compute attention rollout maps within the loop to avoid OOM
        spatial_attention = spatial_attention_rollout(spatial_attention)
        temporal_attention = temporal_attention_rollout(temporal_attention)

        spatial_attention = spatial_attention.to("cpu", torch.float32)
        temporal_attention = temporal_attention.to("cpu", torch.float32)
        # crop frames
        t -= SKIP
        trial_id = batch["trial_id"][0].item()
        video = i_transform_videos(batch["video"][:, :, -t:, ...]).numpy()[0]
        spatial_attention = spatial_attention[-t:, ...].numpy()
        temporal_attention = temporal_attention[-t:].numpy()

        h5.write(save_dir / "videos.h5", {trial_id: video})
        h5.write(save_dir / "spatial_attentions.h5", {trial_id: spatial_attention})
        h5.write(save_dir / "temporal_attentions.h5", {trial_id: temporal_attention})

        spatial_transformer.clear()
        temporal_transformer.clear()

    spatial_transformer.eject()
    del spatial_transformer
    temporal_transformer.eject()
    del temporal_transformer

    torch.cuda.empty_cache()


def main(args):
    args.data_dir = args.data_dir / ("center" if args.center_only else "full")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Cannot find data_dir {args.data_dir}")
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"Cannot find output_dir {args.output_dir}.")

    args.device = utils.get_device(args.device)
    if args.mouse_ids is None:
        del args.mouse_ids
    utils.load_args(args)

    # set batch size to 1 following the starter kit
    args.batch_size = 1

    _, val_ds, _ = get_training_ds(
        args,
        data_dir=SENSORIUM_DIR,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    if args.core_flash_attention:
        args.core_flash_attention = False
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in val_ds.items()
        },
    )
    checkpoint = Checkpoint(args, model=model)
    checkpoint.restore(force=True)

    match args.precision:
        case "16":
            print(f"Perform inference in float16")
            model = model.to(torch.float16)
        case "bf16":
            print(f"Perform inference in bfloat16")
            model = model.to(torch.bfloat16)
        case "32":
            print(f"Perform inference in float32")
            model = model.to(torch.float32)

    val_result = utils.evaluate(args, ds=val_ds, model=model)
    print(f"Validation correlation: {val_result['correlation']['average']:.04f}")

    del val_ds

    grating_ds = {
        mouse_id: DataLoader(
            GratingDataset(
                tier="live_bonus",
                data_dir=args.data_dir,
                mouse_id=mouse_id,
                transform_input=args.transform_input,
                transform_output=args.transform_output,
                verbose=args.verbose,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for mouse_id in ["C", "E"]
    }

    save_dir = (
        args.output_dir
        / "feedback_grating"
        / ("center" if args.center_only else "full")
        / "attention"
    )

    if save_dir.is_dir():
        rmtree(save_dir)

    print(f"\nExtract attention maps from grating patterns.")
    for mouse_id, mouse_ds in grating_ds.items():
        inference(
            ds=mouse_ds,
            model=model,
            save_dir=save_dir / f"mouse{mouse_id}",
            device=args.device,
        )

    print(f"Save prediction to {save_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="../data/feedback_grating",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help="Use only the center of the grating pattern.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=str,
        default=None,
        help="Mouse to use for training.",
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
