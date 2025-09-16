import argparse
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t import data
from viv1t.checkpoint import Checkpoint
from viv1t.model import Model
from viv1t.model.core.vivit import ViViTCore
from viv1t.utils import h5
from viv1t.utils import utils
from viv1t.utils.attention_rollout import Recorder
from viv1t.utils.attention_rollout import spatial_attention_rollout
from viv1t.utils.attention_rollout import temporal_attention_rollout

utils.set_random_seed(1234)

SKIP = 50  # number of frames to skip for metric calculation


def load(args) -> (Dict[str, DataLoader], Model):
    """Load validation DataLoaders and model from args.output_dir checkpoint"""
    assert args.output_dir.is_dir()
    utils.load_args(args)
    val_ds, test_ds = data.get_submission_ds(
        args,
        data_dir=args.data_dir,
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
    return val_ds, test_ds, model


def write_h5(dir: Path, data: Dict[str, List[np.ndarray]]):
    trial_ids = data.pop("trial_ids", None)
    for k, v in data.items():
        for i, trial_id in enumerate(trial_ids):
            h5.write(dir / f"{k}.h5", data={int(trial_id): v[i]})


@torch.inference_mode()
def forward(
    core: ViViTCore,
    spatial_transformer: Recorder,
    temporal_transformer: Recorder,
    mouse_id: str,
    videos: torch.Tensor,
    behaviors: torch.Tensor,
    pupil_centers: torch.Tensor,
):
    outputs = videos
    outputs = core.tokenizer(outputs, behaviors, pupil_centers)
    b, t, p, _ = outputs.shape

    reg_tokens = core.vivit.reg_tokens > 0
    if reg_tokens:
        outputs = core.vivit.add_spatial_reg_tokens(outputs)

    outputs = rearrange(outputs, "b t p c -> (b t) p c")
    outputs, spatial_attentions = spatial_transformer(
        inputs=outputs,
        mouse_id=mouse_id,
        behaviors=behaviors,
        pupil_centers=pupil_centers,
    )
    outputs = rearrange(outputs, "(b t) p c -> b t p c", b=b)

    if reg_tokens:
        outputs = core.vivit.remove_spatial_reg_tokens(outputs)
        outputs = core.vivit.add_temporal_reg_tokens(outputs)

    outputs = rearrange(outputs, "b t p c -> (b p) t c")
    outputs, temporal_attentions = temporal_transformer(
        inputs=outputs,
        mouse_id=mouse_id,
        behaviors=behaviors,
        pupil_centers=pupil_centers,
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
def extract_attention_maps(
    ds: DataLoader,
    model: Model,
    attention_dir: Path,
    device: torch.device = "cpu",
    desc: str = None,
) -> Dict[str, List[np.ndarray]]:
    model.to(device)
    model.train(False)
    mouse_id = ds.dataset.mouse_id

    i_transform_videos = ds.dataset.i_transform_videos
    i_transform_behaviors = ds.dataset.i_transform_behaviors
    i_transform_pupil_centers = ds.dataset.i_transform_pupil_centers

    core = model.core
    assert isinstance(core, ViViTCore)

    spatial_transformer = Recorder(core.vivit.spatial_transformer)
    temporal_transformer = Recorder(core.vivit.temporal_transformer)
    results = {
        "videos": [],
        "pupil_centers": [],
        "behaviors": [],
        "spatial_attentions": [],
        "temporal_attentions": [],
        "trial_ids": [],
    }
    for batch in tqdm(ds, desc=f"mouse {mouse_id}" if desc is None else desc):
        _, _, t, h, w = batch["video"].shape

        outputs, spatial_attentions, temporal_attentions = forward(
            core=core,
            spatial_transformer=spatial_transformer,
            temporal_transformer=temporal_transformer,
            mouse_id=mouse_id,
            videos=batch["video"].to(device, model.dtype),
            behaviors=batch["behavior"].to(device, model.dtype),
            pupil_centers=batch["pupil_center"].to(device, model.dtype),
        )

        # compute attention rollout maps within the loop to avoid OOM
        spatial_attentions = spatial_attention_rollout(spatial_attentions)
        temporal_attentions = temporal_attention_rollout(temporal_attentions)

        spatial_attentions = spatial_attentions.to("cpu", torch.float32)
        temporal_attentions = temporal_attentions.to("cpu", torch.float32)

        # crop frames
        t -= SKIP
        results["videos"].append(
            i_transform_videos(batch["video"][:, :, -t:, ...]).numpy()[0]
        )
        results["behaviors"].append(
            i_transform_behaviors(batch["behavior"][:, :, -t:]).numpy()[0]
        )
        results["pupil_centers"].append(
            i_transform_pupil_centers(batch["pupil_center"][:, :, -t:]).numpy()[0]
        )
        results["spatial_attentions"].append(spatial_attentions[-t:, ...].numpy())
        results["temporal_attentions"].append(temporal_attentions[-t:].numpy())
        results["trial_ids"].append(batch["trial_id"].item())

        spatial_transformer.clear()
        temporal_transformer.clear()
        del outputs, spatial_attentions, temporal_attentions

    spatial_transformer.eject()
    del spatial_transformer
    temporal_transformer.eject()
    del temporal_transformer

    write_h5(attention_dir, results)
    torch.cuda.empty_cache()


def main(args):
    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)

    utils.set_random_seed(args.seed)
    args.batch_size = args.micro_batch_size = 1
    args.device = utils.get_device(args.device)

    if args.mouse_ids is None:
        args.mouse_ids = list(data.SENSORIUM)

    val_ds, test_ds, model = load(args)

    # evaluate validation set to ensure the correlation aligns with expectation
    val_result = utils.evaluate(args, ds=val_ds, model=model)
    print(f"Validation correlation: {val_result['correlation']['average']:.04f}")

    # path to directory to store the core attention matrix
    attention_dir = args.output_dir / "attention"
    attention_dir.mkdir(parents=True, exist_ok=True)

    for mouse_id in args.mouse_ids:
        print(f"\nExtract attention matrix for mouse {mouse_id}")
        for tier, ds in [
            ("validation", val_ds[mouse_id]),
            ("live_main", test_ds["live_main"][mouse_id]),
            ("live_bonus", test_ds["live_bonus"][mouse_id]),
            ("final_main", test_ds["final_main"][mouse_id]),
            ("final_bonus", test_ds["final_bonus"][mouse_id]),
        ]:
            extract_attention_maps(
                ds=ds,
                model=model,
                attention_dir=attention_dir / f"mouse{mouse_id}",
                device=args.device,
                desc=tier,
            )

    print(f"Saved attention matrix to {attention_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../data/sensorium")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mouse_ids", nargs="+", type=str, default=None)
    parser.add_argument("--device", default="", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
