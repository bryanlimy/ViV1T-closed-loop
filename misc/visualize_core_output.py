import argparse
import pickle
from pathlib import Path
from typing import Dict
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from celluloid import Camera
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

from viv1t.data import SENSORIUM_NEW
from viv1t.data import get_training_ds
from viv1t.model import Model
from viv1t.scheduler import Scheduler
from viv1t.utils import plot
from viv1t.utils import utils

matplotlib.rcParams["animation.embed_limit"] = 2**64


COLORMAP = "turbo"
TICK_FONTSIZE = 9
LABEL_FONTSIZE = 10
TITLE_FONTSIZE = 11
FPS = 30
DPI = 240
SKIP = 50


def load(args, output_dir: Path):
    """Load validation DataLoaders and model from output_dir checkpoint"""
    assert output_dir.is_dir()
    model_args = argparse.Namespace()
    model_args.output_dir = output_dir
    utils.load_args(model_args)
    model_args.device = args.device
    model_args.batch_size = args.batch_size
    model_args.micro_batch_size = args.micro_batch_size
    model_args.data_dir = args.data_dir

    _, val_ds = get_training_ds(
        model_args,
        data_dir=model_args.data_dir,
        mouse_ids=model_args.mouse_ids,
        batch_size=model_args.batch_size,
        device=model_args.device,
    )
    model = Model(
        model_args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in val_ds.items()
        },
    )
    scheduler = Scheduler(model_args, model=model, save_optimizer=False)
    scheduler.restore(force=True)
    del model_args
    return val_ds, model


@torch.inference_mode()
def inference(
    model: Model,
    ds: Dict[str, DataLoader],
    mouse_id: str,
    device: torch.device,
    num_samples: int = None,
):
    model.to(device)
    model.train(False)
    if num_samples is None:
        num_samples = len(ds[mouse_id])
    i_transform_videos = ds[mouse_id].dataset.i_transform_videos
    result = {"videos": [], "core_outputs": []}
    for batch in tqdm(ds[mouse_id], total=num_samples):
        if len(result["videos"]) >= num_samples:
            break
        core_outputs = model.core(
            inputs=batch["video"].to(device),
            mouse_id=mouse_id,
            behaviors=batch["behavior"].to(device),
            pupil_centers=batch["pupil_center"].to(device),
        )
        t = core_outputs.size(2)
        video = i_transform_videos(batch["video"][:, :, -t:]).numpy().astype(np.float16)
        core_outputs = core_outputs.cpu().numpy().astype(np.float16)
        result["videos"].append(video)
        result["core_outputs"].append(core_outputs)
    result["videos"] = np.vstack(result["videos"])
    result["core_outputs"] = np.vstack(result["core_outputs"])
    return result


def normalize(x: np.ndarray):
    return (x - x.min()) / (x.max() - x.min())


def resize(x: np.ndarray, size: Tuple[int]):
    return F.resize(torch.from_numpy(x), size=list(size), antialias=False).numpy()


def animate_core_output(
    core_output: np.ndarray,
    title: str = None,
    temporal_channel: str = "$N_T$",
    filename: Path = None,
):
    core_output = normalize(core_output)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(6, 1.5),
        gridspec_kw={"wspace": 0.05, "hspace": 0.0},
        dpi=DPI,
    )
    h, w = core_output.shape[-2:]
    frames = core_output.shape[1]
    h_ticks = np.linspace(0, h - 1, 4, dtype=int)
    w_ticks = np.linspace(0, w - 1, 4, dtype=int)

    pos = axes[-1].get_position()

    cbar_height = 0.2
    cbar_ax = figure.add_axes(
        [
            pos.x1 + 0.01,
            0.5 * (pos.y1 - pos.y0 - cbar_height) + pos.y0,
            0.009,
            cbar_height,
        ]
    )

    camera = Camera(figure)

    if title is not None:
        figure.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.93)

    for frame in range(frames):
        axes[0].imshow(
            np.mean(core_output, axis=0)[frame],
            cmap=COLORMAP,
            aspect="equal",
            interpolation=None,
        )
        plot.set_xticks(
            axes[0],
            ticks=w_ticks,
            tick_labels=w_ticks,
            tick_fontsize=TICK_FONTSIZE,
        )
        plot.set_yticks(
            axes[0],
            ticks=h_ticks,
            tick_labels=h_ticks,
            tick_fontsize=TICK_FONTSIZE,
        )
        plot.set_ticks_params(axes[0], length=2, pad=1)
        axes[0].set_title("average", fontsize=LABEL_FONTSIZE, pad=2)

        # plot first 2 channels
        for c in range(2):
            axes[c + 1].imshow(
                core_output[c, frame],
                cmap=COLORMAP,
                aspect="equal",
                interpolation=None,
            )
            axes[c + 1].set_xticks([])
            axes[c + 1].set_yticks([])
            axes[c + 1].set_title(
                f"(B, {c}, {temporal_channel}, H', W')",
                fontsize=LABEL_FONTSIZE,
                pad=2,
            )

        cbar = plt.colorbar(cm.ScalarMappable(cmap=COLORMAP), cax=cbar_ax, shrink=0.5)
        cbar.mappable.set_clim(0, 1)
        cbar_yticks = np.linspace(0, 1, 2, dtype=int)
        plot.set_yticks(
            axis=cbar_ax,
            ticks=cbar_yticks,
            tick_labels=cbar_yticks,
            tick_fontsize=TICK_FONTSIZE,
        )
        plot.set_ticks_params(cbar_ax, length=1.5, pad=1)

        camera.snap()

    animation = camera.animate()
    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        animation.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
    plt.close(figure)


def extract_core_outputs(args, output_dir: Path, plot_dir: Path):
    cache_dir = output_dir / "core_outputs"

    if not cache_dir.is_dir() or args.overwrite:
        cache_dir.mkdir(parents=True, exist_ok=True)
        val_ds, model = load(args, output_dir)

        for mouse_id in SENSORIUM_NEW:
            print(f"Extract core outputs for mouse {mouse_id}")
            result = inference(
                model,
                ds=val_ds,
                mouse_id=mouse_id,
                device=args.device,
            )
            with open(cache_dir / f"mouse{mouse_id}.pkl", "wb") as file:
                pickle.dump(result, file)

    for mouse_id in ["G"]:
        print(f"Plot animations for mouse {mouse_id}...")
        with open(cache_dir / f"mouse{mouse_id}.pkl", "rb") as file:
            result = pickle.load(file)
        for i in tqdm(range(len(result["videos"])), desc=f"Moues {mouse_id}"):
            animate_core_output(
                core_output=result["core_outputs"][i],
                temporal_channel="$N_T$" if "vivit" in output_dir else "T'",
                filename=plot_dir / f"mouse{mouse_id}" / f"trial{i:03d}.gif",
            )
            if i > 10:
                break


def main(args):
    plot.set_font()
    utils.set_random_seed(1234)

    args.device = torch.device(args.device)
    args.batch_size, args.micro_batch_size = 1, 1

    plot_dir = Path("figures/core_output")

    extract_core_outputs(
        args,
        output_dir=Path("../runs/best_vivit"),
        plot_dir=plot_dir / "vivit",
    )

    # extract_core_outputs(
    #     args,
    #     output_dir="../runs/best_fcnn",
    #     plot_dir=os.path.join(plot_dir, "fcnn"),
    # )

    print(f"Saved results to {plot_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../data/sensorium")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite the cache file if exists."
    )
    main(parser.parse_args())
