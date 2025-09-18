import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from viv1t import data
from viv1t.checkpoint import Checkpoint
from viv1t.model import Model
from viv1t.model.core.vivit import ViViTCore
from viv1t.utils import plot
from viv1t.utils import utils
from viv1t.utils.attention_rollout import Recorder
from viv1t.utils.attention_rollout import fold_spatial_attention
from viv1t.utils.attention_rollout import spatial_attention_rollout
from viv1t.utils.attention_rollout import temporal_attention_rollout

plot.set_font()

DPI = 240
SKIP = 50
H, W = 36, 64
FPS = 30

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures/attention_map")
OUTPUT_DIR = Path("../runs/vivit/164_viv1t_spaitalPS5")


def load(mouse_id: str):
    args = argparse.Namespace()
    args.data_dir = DATA_DIR
    args.output_dir = OUTPUT_DIR
    args.batch_size = 1
    args.mouse_ids = [mouse_id]
    args.device = torch.device("cpu")
    utils.load_args(args)
    if args.core_flash_attention:
        args.core_flash_attention = False
    _, val_ds, _ = data.get_training_ds(
        args,
        data_dir=args.data_dir,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in val_ds.items()
        },
    )
    checkpoint = Checkpoint(args, model=model)
    checkpoint.restore(force=True)
    return val_ds, model


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
    outputs = core.tokenizer(outputs, behaviors=behaviors, pupil_centers=pupil_centers)
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
    trial_id: int,
    device: torch.device = "cpu",
    reverse: bool = False,
    shuffle: bool = False,
) -> Dict[str, np.ndarray]:
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
    results = dict()
    for batch in ds:
        if batch["trial_id"] != trial_id:  # TODO: inefficient but work for now
            continue
        print(f"extracting attention map from mouse {mouse_id} trial {trial_id}")

        videos = batch["video"].to(device)
        behaviors = batch["behavior"].to(device)
        pupil_centers = batch["pupil_center"].to(device)

        _, _, t, h, w = videos.shape

        if reverse:
            videos = torch.flip(videos, dims=[2])
            behaviors = torch.flip(behaviors, dims=[-1])
            pupil_centers = torch.flip(pupil_centers, dims=[-1])

        if shuffle:
            frames = np.arange(t)
            # shuffle the 100-150 frames and 200-250 frames
            # frames[100:150] = np.random.permutation(frames[100:150])
            # frames[200:250] = np.random.permutation(frames[200:250])
            # videos = videos[:, :, frames, ...]
            # behaviors = behaviors[:, :, frames]
            # pupil_centers = pupil_centers[:, :, frames]
            for i in (100, 150, 200, 250):
                videos[:, :, i : i + 20, ...] = 0

        outputs, spatial_attentions, temporal_attentions = forward(
            core=core,
            spatial_transformer=spatial_transformer,
            temporal_transformer=temporal_transformer,
            mouse_id=mouse_id,
            videos=videos,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )

        # compute attention rollout maps within the loop to avoid OOM
        spatial_attentions = spatial_attention_rollout(spatial_attentions)
        temporal_attentions = temporal_attention_rollout(temporal_attentions)

        # crop frames
        t -= SKIP
        results["video"] = i_transform_videos(videos[:, :, -t:, ...])[0]
        results["behavior"] = i_transform_behaviors(behaviors[:, :, -t:])[0]
        results["pupil_center"] = i_transform_pupil_centers(pupil_centers[:, :, -t:])[0]
        results["spatial_attention"] = spatial_attentions[-t:, ...]
        results["temporal_attention"] = temporal_attentions[-t:]
        results["trial_id"] = batch["trial_id"].item()

        spatial_transformer.clear()
        temporal_transformer.clear()
        del outputs, videos, behaviors, pupil_centers
        del spatial_attentions, temporal_attentions

    spatial_transformer.eject()
    del spatial_transformer
    temporal_transformer.eject()
    del temporal_transformer

    return results


def animate(
    mouse_id: str,
    trial_id: int,
    reverse: bool = False,
    shuffle: bool = False,
    fps: int = FPS,
):
    val_ds, model = load(mouse_id)
    result = extract_attention_maps(
        ds=val_ds[mouse_id],
        model=model,
        trial_id=trial_id,
        reverse=reverse,
        shuffle=shuffle,
    )
    result["spatial_attention"] = fold_spatial_attention(
        result["spatial_attention"],
        frame_size=(H, W),
        patch_size=model.core.tokenizer.kernel_size[1],
        stride=model.core.tokenizer.stride[1],
        padding=model.core.tokenizer.padding,
    )
    for k, v in result.items():
        if torch.is_tensor(v):
            result[k] = v.numpy()

    filename = f"mouse{mouse_id}_trial{trial_id:03d}"
    if reverse:
        filename += "_reverse"
    if shuffle:
        filename += "_shuffle"
    filename = PLOT_DIR / f"{filename}.mp4"
    plot.animate_attention_map(result, filename=filename, fps=fps, dpi=DPI)
    print(f"save plot to {filename}.")


if __name__ == "__main__":
    animate(mouse_id="A", trial_id=9, fps=10)
    # animate(mouse_id="G", trial_id=7, fps=10)
    # animate(mouse_id="B", trial_id=86)
