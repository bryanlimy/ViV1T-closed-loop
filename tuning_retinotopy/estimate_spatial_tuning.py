"""
Compute spatial selectivity index (SSI) for each neuron following the procedure
detailed in Wang et al. (2023) page 13.

Reference:
- https://www.biorxiv.org/content/10.1101/2023.03.21.533548v2.full.pdf
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import einsum
from einops import rearrange
from einops import repeat

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()

TICK_FONTSIZE = 10
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 14
DPI = 240

DATA_DIR = Path("../data/sensorium")
RESPONSE_DIR = Path("../data/sensorium/responses")
PLOT_DIR = Path("figures/tuning/spatial_tuning/")
H, W = 36, 64  # height and width of the video frame

plot.set_font()


def load_video(mouse_id: str, trial_id: int | str) -> (np.ndarray, np.ndarray):
    max_frame = 300
    sample = data.load_trial(
        mouse_dir=DATA_DIR / data.MOUSE_IDS[mouse_id],
        trial_id=trial_id,
    )
    video = sample["video"][:, :max_frame, :, :]
    feature_dir = (
        data.METADATA_DIR / "ood_features" / "gaussian_dots" / f"mouse{mouse_id}"
    )
    feature = np.load(feature_dir / f"{trial_id}.npy")
    feature = feature[:max_frame, :]
    return video, feature


def filter_frame(
    video: np.ndarray, black_dots: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
):
    """Remove frames where white Gaussian dot is presented."""
    filter = np.where(black_dots == 1)[0]
    video = video[filter, :, :]
    y_true = y_true[:, filter]
    y_pred = y_pred[:, filter]
    return video, y_true, y_pred


def load_result(
    response_dir: Path, mouse_id: str, trial_ids: np.ndarray
) -> Dict[str, np.ndarray]:
    responses = h5.get(RESPONSE_DIR / f"mouse{mouse_id}.h5", trial_ids=trial_ids)
    predictions = h5.get(
        response_dir / "predict" / f"mouse{mouse_id}.h5", trial_ids=trial_ids
    )

    result = {"y_true": [], "y_pred": [], "video": []}
    for i, trial_id in enumerate(trial_ids):
        t = responses[i].shape[1]
        video, feature = load_video(mouse_id=mouse_id, trial_id=trial_id)
        video, feature = video[0, -t:, :, :], feature[-t:, :]
        video, y_true, y_pred = filter_frame(
            video,
            black_dots=feature[:, -1],
            y_true=responses[i],
            y_pred=predictions[i],
        )
        result["video"].append(video)
        result["y_true"].append(rearrange(y_true, "n t -> t n"))
        result["y_pred"].append(rearrange(y_pred, "n t -> t n"))
        del y_true, y_pred, video, feature
    result = {k: np.concatenate(v, axis=0) for k, v in result.items()}
    return result


def plot_sta(
    sta_true: np.ndarray,
    sta_pred: np.ndarray,
    filename: Path,
    model_name: str = "Model",
):
    num_neurons = 5
    neurons = sorted(
        np.random.choice(sta_true.shape[0], size=num_neurons, replace=False)
    )
    figure, axes = plt.subplots(
        nrows=num_neurons,
        ncols=2,
        gridspec_kw={"wspace": 0.05, "hspace": 0.1},
        figsize=(5, 8),
    )
    for i in range(num_neurons):
        axes[i, 0].imshow(sta_true[neurons[i]], cmap="gray", aspect="equal")
        axes[i, 1].imshow(sta_pred[neurons[i]], cmap="gray", aspect="equal")
        axes[i, 0].set_ylabel(
            f"Neuron {neurons[i]}", fontsize=LABEL_FONTSIZE, labelpad=0
        )
        for j in range(2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    axes[0, 0].set_title("Recorded", fontsize=TITLE_FONTSIZE)
    axes[0, 1].set_title(model_name, fontsize=TITLE_FONTSIZE)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def plot_hexbin(
    SIs_true: np.ndarray,
    SIs_pred: np.ndarray,
    filename: Path,
    name: str,
    vmax: int = None,
):
    df = pd.DataFrame({"true_SI": SIs_true, "pred_SI": SIs_pred})

    cmap = "dodgerblue"

    g = sns.jointplot(
        data=df, x="true_SI", y="pred_SI", kind="hex", height=4, color=cmap
    )
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()

    if vmax is None:
        vmax = g.ax_joint.collections[0].get_array().max()

    ax = g.ax_joint
    figure = g.figure

    s_min = min(np.nanmin(SIs_true), np.nanmin(SIs_pred))
    s_max = max(np.nanmax(SIs_true), np.nanmax(SIs_pred))
    ax.set_xlim(s_min, s_max)
    ax.set_ylim(s_min, s_max)
    ticks = np.linspace(s_min, s_max, 5)
    plot.set_xticks(
        axis=ax,
        ticks=ticks,
        tick_labels=np.round(ticks, 1),
        label=r"$\it{in \> vivo}$ " + name,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=1,
    )
    plot.set_yticks(
        axis=ax,
        ticks=ticks,
        tick_labels=np.round(ticks, 1),
        label=r"$\it{in \> silico}$ " + name,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=1,
    )
    plot.set_ticks_params(ax, length=2)

    pos = ax.get_position()
    cbar_width, cbar_height = 0.02 * (pos.x1 - pos.x0), 0.4 * (pos.y1 - pos.y0)
    cbar_ax = figure.add_axes(
        rect=(
            pos.x1 + 0.025,
            pos.y0 + ((pos.y1 - pos.y0) / 2) - (cbar_height / 2),
            cbar_width,
            cbar_height,
        )
    )
    cbar = plt.colorbar(cmap=cmap, cax=cbar_ax)
    cbar.mappable.set_clim(0, vmax)
    cbar_yticks = np.linspace(0, vmax, 3, dtype=int)
    plot.set_yticks(
        axis=cbar_ax,
        ticks=cbar_yticks,
        tick_labels=cbar_yticks,
        tick_fontsize=TICK_FONTSIZE,
    )
    cbar_ax.set_ylabel("Count", fontsize=LABEL_FONTSIZE, labelpad=10, rotation=270)
    plot.set_ticks_params(cbar_ax, length=2)

    # compute correlation coefficient
    corr = df.corr().iloc[0, 1]
    ax.set_title(f"Correlation: {corr:.4f}", fontsize=TITLE_FONTSIZE)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def spike_triggered_average(video: np.ndarray, response: np.ndarray) -> np.ndarray:
    """
    Compute the per-neuron spike triggered average (STA) of the Gaussian dot
    stimulus.
    """
    video, response = torch.from_numpy(video), torch.from_numpy(response)

    s_0 = rearrange(video[:, 0, 0], "t -> t 1 1")  # get the background of each frame
    sta = einsum(response, video - s_0, "n t, t h w -> n h w")
    sta = sta / repeat(torch.sum(response, dim=1), "n -> n 1 1")
    return sta.numpy()


def spatial_selectivity(sta: np.ndarray):
    sta = torch.from_numpy(sta)

    N = sta.shape[0]
    sta = rearrange(sta, "n h w -> n (h w)")
    # create an array of 2D pixel coordinates
    ww, hh = torch.meshgrid(
        torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="xy"
    )
    x = repeat(torch.stack([hh, ww]), "d h w -> n (h w) d", n=N)
    z = rearrange(torch.sum(sta, dim=1), "n -> n 1 1")
    x_bar = rearrange(einsum(sta, x, "n p, n p d -> n d"), "n d -> n 1 d")
    x_bar = x_bar / z
    residual = rearrange(x - x_bar, "n p d -> n p d 1")
    sigma = einsum(sta, residual, residual, "n p, n p d1 d2, n p d3 d4 -> n d1 d3")
    sigma = sigma / z
    ssi = -torch.logdet(sigma)
    return ssi.numpy()


def spatial_tuning(mouse_id: str, output_dir: Path, model_name: str = "Model"):
    # find trials with Gaussian dot stimuli
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 2)[0]
    if not trial_ids.size:
        return  # mouse does not have drifting gabor stimulus

    utils.set_random_seed(1234)
    print(f"compute SSI for mouse {mouse_id}")

    plot_dir = PLOT_DIR / output_dir.name / f"mouse{mouse_id}"

    # load all recorded and predicted responses for trial_ids
    result = load_result(output_dir, mouse_id=mouse_id, trial_ids=trial_ids)

    # convert everything to neuron-major
    y_true = rearrange(result["y_true"], "t n -> n t")
    y_pred = rearrange(result["y_pred"], "t n -> n t")
    video = result["video"]

    sta_true = spike_triggered_average(video=video, response=y_true)
    sta_pred = spike_triggered_average(video=video, response=y_pred)

    plot_sta(
        sta_true=sta_true,
        sta_pred=sta_pred,
        filename=plot_dir / f"mouse{mouse_id}_STA.png",
        model_name=model_name,
    )

    true_SSI = spatial_selectivity(sta=sta_true)
    pred_SSI = spatial_selectivity(sta=sta_pred)

    utils.save_tuning(
        result={"SSI": true_SSI},
        save_dir=data.METADATA_DIR,
        mouse_id=mouse_id,
    )
    utils.save_tuning(
        result={"SSI": pred_SSI},
        save_dir=output_dir,
        mouse_id=mouse_id,
    )

    vmax = None
    # match mouse_id:
    #     case "A":
    #         vmax = 76
    #     case "D":
    #         vmax = 79
    #     case "E":
    #         vmax = 91
    plot_hexbin(
        SIs_true=true_SSI,
        SIs_pred=pred_SSI,
        filename=plot_dir / f"mouse{mouse_id}_SSI.png",
        name="SSI",
        vmax=vmax,
    )
    print(f"saved result to {plot_dir}")


def main():
    for model_name, output_dir in [
        ("fCNN", Path("../runs/fCNN/012_fCNN")),
        ("ViV1T", Path("../runs/vivit/140_vivit_AdamWScheduleFree")),
    ]:
        for mouse_id in ("A", "D", "E"):
            spatial_tuning(
                mouse_id=mouse_id, output_dir=output_dir, model_name=model_name
            )


if __name__ == "__main__":
    main()
