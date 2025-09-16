import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from viv1t import data
from viv1t.model import Model
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils
from viv1t.utils.attention_rollout import fold_spatial_attention

plot.set_font()


BLANK_SIZE, PATTERN_SIZE = 20, 30
BLOCK_SIZE = BLANK_SIZE + PATTERN_SIZE

DATA_DIR = Path("../data/feedback_grating/center")
SENSORIUM_DIR = Path("../data/sensorium")


TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10
DPI, FPS = 240, 30


def animate_attention_map(
    videos: np.ndarray,
    spatial_attentions: np.ndarray,
    temporal_attentions: np.ndarray,
    params: np.ndarray,
    filename: Path,
    alpha: float = 0.8,
    spatial_title: str = 'spatial "attention"',
    temporal_title: str = 'temporal "attention"',
):
    """Animate attention map overlay over video frames"""
    turbo_color = plot.TURBO(np.arange(256))[:, :3]

    # colormap
    mappable = cm.ScalarMappable(cmap=plot.TURBO)
    mappable.set_clim(0, 1)

    figure, axes = plt.subplots(
        nrows=4,
        ncols=6,
        figsize=(8, 4),
        gridspec_kw={"wspace": 0.05, "hspace": -0.2},
        dpi=DPI,
    )

    classical_axes = axes[:2, :].flatten()
    inverse_axes = axes[2:, :].flatten()
    axes = axes.flatten()

    # pos = axes[-2].get_position()

    # # add colorbar
    # cbar_width, cbar_height = 0.008, 0.04
    # cbar_ax = figure.add_axes(
    #     rect=(
    #         pos.x1 + 0.01,
    #         pos.y0,
    #         cbar_width,
    #         cbar_height,
    #     )
    # )
    # plt.colorbar(mappable, cax=cbar_ax, shrink=0.5)
    # cbar_yticks = np.linspace(0, 1, 2, dtype=int)
    # plot.set_yticks(
    #     axis=cbar_ax,
    #     ticks=cbar_yticks,
    #     tick_labels=cbar_yticks,
    #     tick_fontsize=TICK_FONTSIZE,
    # )
    # plot.set_ticks_params(cbar_ax, length=1.5, pad=1)

    stimulus_sizes = np.unique(params[:, 0])

    def animate(frame: int):
        for ax in axes:
            ax.cla()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(linewidth=0)

        classical_axes[-1].set_visible(False)
        inverse_axes[-1].set_visible(False)

        block_size = videos.shape[1]
        for stim_type in (0, 1):
            stim_ax = classical_axes if stim_type == 0 else inverse_axes
            for i, stimulus_size in enumerate(stimulus_sizes):
                pattern_id = np.where(
                    (params[:, 0] == stimulus_size) & (params[:, -1] == stim_type)
                )[0]
                if frame < block_size:
                    pattern_id = pattern_id[0]
                else:
                    pattern_id = pattern_id[1]
                # plot spatial attention map overlay on frame
                image = videos[pattern_id, frame % block_size]
                heatmap = spatial_attentions[pattern_id, frame % block_size]
                heatmap = turbo_color[np.uint8(255.0 * heatmap)] * 255.0
                heatmap = alpha * heatmap + (1 - alpha) * image[..., None]
                stim_ax[i].imshow(
                    heatmap.astype(np.uint8),
                    cmap=mappable.cmap,
                    vmin=0,
                    vmax=255,
                    interpolation=None,
                )
                stim_ax[i].set_title(
                    f"{stimulus_size:.0f}°", pad=1, fontsize=TICK_FONTSIZE
                )

    anim = FuncAnimation(
        figure,
        animate,
        frames=videos.shape[1] * 2,
        interval=int(1000 / FPS),
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
    try:
        plt.close(figure)
    except AttributeError as e:
        print(f"AttributeError in plt.close(figure): {e}.")


def estimate_mouse_attention(
    args,
    mouse_id: str,
    mouse_dir: Path,
    patch_size,
    stride,
    padding,
    plot_dir: Path,
):
    if not mouse_dir.is_dir():
        print(f"Cannot find {mouse_dir}")
        return
    print(f"Processing mouse {mouse_id}...")

    trial_ids = h5.get_trial_ids(mouse_dir / "videos.h5")
    videos = np.stack(h5.get(mouse_dir / "videos.h5", trial_ids=trial_ids))
    spatial_patch_attentions = np.stack(
        h5.get(mouse_dir / "spatial_attentions.h5", trial_ids=trial_ids)
    )
    temporal_attentions = np.stack(
        h5.get(mouse_dir / "temporal_attentions.h5", trial_ids=trial_ids)
    )
    num_frames = temporal_attentions.shape[1]
    frame_size = videos.shape[-2:]

    spatial_attentions = np.stack(
        [
            fold_spatial_attention(
                spatial_patch_attentions[i],
                frame_size=frame_size,
                patch_size=patch_size,
                stride=stride,
                padding=padding,
            )
            for i in range(len(spatial_patch_attentions))
        ]
    )

    params = np.load(DATA_DIR / "meta" / "trials" / "params.npy")
    # rearrange params order to match trial_ids
    params = params[trial_ids, -num_frames:, :]

    # randomly plot num_samples trials
    rng = np.random.default_rng(1234)
    for i in tqdm(
        rng.choice(trial_ids, size=args.num_samples, replace=False),
        desc=f"plot trial",
    ):
        plot.animate_attention_map(
            sample={
                "video": videos[i],
                "spatial_attention": spatial_attentions[i],
                "temporal_attention": temporal_attentions[i],
                "behavior": None,
                "pupil_center": None,
            },
            filename=plot_dir / f"trial{i:03d}.{args.format}",
        )

    # rearrange to block size
    assert videos.shape[1] == 1, "videos should have one channel"
    videos = rearrange(
        videos[:, 0, ...],
        "trial (block frame) height width -> (trial block) frame height width",
        frame=BLOCK_SIZE,
    )
    spatial_attentions = rearrange(
        spatial_attentions,
        "trial (block frame) height width -> (trial block) frame height width",
        frame=BLOCK_SIZE,
    )
    temporal_attentions = rearrange(
        temporal_attentions,
        "trial (block frame) -> (trial block) frame",
        frame=BLOCK_SIZE,
    )
    params = rearrange(
        params,
        "trial (block frame) param -> (trial block) param frame",
        frame=BLOCK_SIZE,
    )
    params = params[:, :, -1]  # parameters in each block is the same

    # remove blank screen blocks
    non_blank = np.where(np.all(~np.isnan(params), axis=1))[0]
    params = params[non_blank]
    videos = videos[non_blank]
    spatial_attentions = spatial_attentions[non_blank]
    temporal_attentions = temporal_attentions[non_blank]

    # compute average attention over repeated patterns
    unique_params = np.unique(params, axis=0)
    average_spatial_attentions = np.zeros(
        (len(unique_params), *spatial_attentions.shape[1:]), dtype=np.float32
    )
    average_temporal_attentions = np.zeros(
        (len(unique_params), *temporal_attentions.shape[1:]), dtype=np.float32
    )
    pattern_videos = np.zeros((len(unique_params), *videos.shape[1:]), dtype=np.float32)
    for i, param in enumerate(unique_params):
        pattern_ids = np.where(np.all(params == param, axis=1))[0]
        average_spatial_attentions[i] = np.mean(spatial_attentions[pattern_ids], axis=0)
        average_temporal_attentions[i] = np.mean(
            temporal_attentions[pattern_ids], axis=0
        )
        pattern_videos[i] = videos[pattern_ids[0]]

    animate_attention_map(
        videos=pattern_videos,
        spatial_attentions=average_spatial_attentions,
        temporal_attentions=average_temporal_attentions,
        params=unique_params,
        filename=plot_dir / "average_attention.mp4",
    )


def main(args):
    args.device = torch.device("cpu")
    utils.load_args(args)
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: data.get_neuron_coordinates(
                SENSORIUM_DIR, mouse_id, to_tensor=True
            )
            for mouse_id in args.mouse_ids
        },
    )
    patch_size = model.core.tokenizer.kernel_size[1]
    stride = model.core.tokenizer.stride[1]
    padding = model.core.tokenizer.padding

    del model

    plot_dir = args.output_dir / "plots" / "feedback_grating" / "attention_rollout"
    plot_dir.mkdir(parents=True, exist_ok=True)

    attention_dir = args.output_dir / "feedback_grating" / "center" / "attention"

    for mouse_id in ["C", "E"]:
        estimate_mouse_attention(
            args,
            mouse_id=mouse_id,
            mouse_dir=attention_dir / f"mouse{mouse_id}",
            patch_size=patch_size,
            stride=stride,
            padding=padding,
            plot_dir=plot_dir / f"mouse{mouse_id}",
        )
    print(f"Saved figures to {plot_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    main(parser.parse_args())
