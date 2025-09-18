import argparse
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation
from mei_utils import get_sensorium_response_stats
from predict_stimulus import predict
from tqdm import tqdm

from viv1t.utils import plot

plot.set_font()

FPS = 30
FONTSIZE = 11
DPI = 200


def get_grid_result(MEVs_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    cutoff = np.linspace(0, 1, 11)
    size = len(cutoff)
    peak_responses = np.zeros((size, size), dtype=np.float32)
    videos = np.zeros((size, size, 300, 36, 64), dtype=np.float32)
    for i, spatial_cutoff in enumerate(cutoff):
        for j, temporal_cutoff in enumerate(cutoff):
            filename = (
                MEVs_dir
                / f"{count:03d}_spatial{spatial_cutoff:.01f}_temporal{temporal_cutoff:.01f}"
                / "ckpt.pt"
            )
            if not filename.exists():
                print(f"File {filename.parent.name} not found.")
                count += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                predict_result = torch.load(filename, map_location="cpu")
            videos[i, j] = predict_result["video"].numpy()[0]
            peak_responses[i, j] = predict_result["response"].max().numpy()
            count += 1
    return peak_responses, videos


def plot_peak_responses(
    args, mouse_id: str, neuron: int, peak_responses: np.ndarray, plot_dir: Path
):
    sensorium_max, _ = get_sensorium_response_stats(
        args.output_dir, mouse_id=mouse_id, neuron=neuron
    )
    response_gain = 100 * (peak_responses - sensorium_max) / sensorium_max

    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=DPI)

    sns.heatmap(
        response_gain,
        # vmin=-40,
        # vmax=150,
        annot=True,
        ax=ax,
        # cmap=sns.cm.rocket_r,
        center=0,
        linewidths=0.1,
        linecolor="black",
        fmt=".0f",
        cbar_kws={
            "shrink": 0.5,
            "pad": 0.01,
            "label": "Improvement against sensorium peak (%)",
        },
        square=True,
    )

    ticks = np.linspace(0, 1, 11)
    plot.set_xticks(
        axis=ax,
        ticks=np.arange(len(ticks)) + 0.5,
        tick_labels=np.round(ticks, 1),
        label="Temporal freq. keep rate",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
    )
    plot.set_yticks(
        axis=ax,
        ticks=np.arange(len(ticks)) + 0.5,
        tick_labels=np.round(ticks, 1),
        label="Spatial freq. keep rate",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
    )
    ax.set_title(
        "MEV peak response increase\nover max predicted response in Sensorium",
        fontsize=FONTSIZE,
    )
    plot.set_ticks_params(axis=ax, length=2, pad=1)
    plot.save_figure(figure, filename=plot_dir / "peak_responses.jpg", dpi=DPI)


MAX_FRAME = 300
PATTERN_SIZE = 30
BLANK_SIZE = (MAX_FRAME - PATTERN_SIZE) // 2


def plot_MEVs(videos: np.ndarray, plot_dir: Path):
    nrows, ncols = videos.shape[0], videos.shape[1]

    start = -(BLANK_SIZE + PATTERN_SIZE + 10)
    end = -(BLANK_SIZE - 10)
    videos = videos[:, :, start:end, :, :]
    t = videos.shape[2]

    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8, 5),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
        dpi=DPI,
    )

    figure.text(
        x=0.5,
        y=0.045,
        s="Temporal freq. keep rate",
        ha="center",
        va="center",
        fontsize=FONTSIZE,
    )
    figure.text(
        x=0.08,
        y=0.5,
        s="Spatial freq. keep rate",
        ha="center",
        va="center",
        fontsize=FONTSIZE,
        rotation=90,
    )

    cutoff = np.linspace(0, 1, 11)

    # initialize all axes
    imshows = {}
    for i, spatial_cutoff in enumerate(cutoff):
        for j, temporal_cutoff in enumerate(cutoff):
            ax = axes[i, j]
            imshows[(i, j)] = ax.imshow(
                np.random.rand(videos.shape[3], videos.shape[4]),
                cmap="gray",
                aspect="equal",
                vmin=0,
                vmax=255.0,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if i == nrows - 1:
                ax.set_xlabel(f"{temporal_cutoff:.1f}", fontsize=FONTSIZE, labelpad=2)
            if j == 0:
                ax.set_ylabel(f"{spatial_cutoff:.1f}", fontsize=FONTSIZE, labelpad=1)

    def animate(frame: int):
        for i, spatial_cutoff in enumerate(cutoff):
            for j, temporal_cutoff in enumerate(cutoff):
                imshows[(i, j)].set_data(videos[i, j, frame])
        return list(imshows.values())

    anim = FuncAnimation(
        figure,
        func=animate,
        frames=tqdm(range(t), file=sys.stdout, desc="Animate MEVs", leave=False),
        interval=int(1000 / FPS),
        blit=True,
    )

    plot_dir.mkdir(parents=True, exist_ok=True)
    anim.save(plot_dir / "MEVs.mp4", fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
    plt.close(figure)


from kneed import KneeLocator


def find_elbow(data, theta):

    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # return index of elbow
    return np.where(rotated_vector == rotated_vector[:, 1].min())[0][0]


def get_data_radiant(data):
    return np.arctan2(
        data[:, 1].max() - data[:, 1].min(), data[:, 0].max() - data[:, 0].min()
    )


def find_elbow_point(
    args, peak_response: np.ndarray, mouse_id: str, neuron: int, plot_dir: Path
):
    sensorium_max, _ = get_sensorium_response_stats(
        args.output_dir, mouse_id=mouse_id, neuron=neuron
    )
    response_gain = 100 * (peak_response - sensorium_max) / sensorium_max

    x = np.arange(0.1, 1.0, 0.1)
    spatial_peaks = response_gain.max(axis=1)
    temporal_peaks = response_gain.max(axis=0)

    # spatial_kn = KneeLocator(
    #     x,
    #     spatial_peaks,
    #     curve="convex",
    #     direction="increasing",
    # )
    # temporal_kn = KneeLocator(
    #     x,
    #     temporal_peaks,
    #     curve="convex",
    #     direction="increasing",
    # )

    spatial_data = np.column_stack((x, spatial_peaks))
    spatial_elbow = find_elbow(
        spatial_data,
        get_data_radiant(spatial_data),
    )
    spatial_elbow = x[spatial_elbow]

    temporal_data = np.column_stack((x, temporal_peaks))
    temporal_elbow = find_elbow(
        temporal_data,
        get_data_radiant(temporal_data),
    )
    temporal_elbow = x[temporal_elbow]

    s_min, s_max = np.floor(spatial_peaks.min()), np.ceil(spatial_peaks.max())
    t_min, t_max = np.floor(temporal_peaks.min()), np.ceil(temporal_peaks.max())

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        gridspec_kw={"wspace": 0.3, "hspace": 0.1},
        figsize=(7, 2),
        dpi=DPI,
    )

    axes[0].plot(
        x, spatial_peaks, clip_on=False, linestyle="-", marker="o", color="orangered"
    )
    axes[0].vlines(
        spatial_elbow, ymin=s_min, ymax=s_max, linestyles="--", color="black"
    )
    plot.set_xticks(
        axis=axes[0],
        ticks=x,
        tick_labels=np.round(x, 2),
        label="Spatial freq. keep rate",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
    )
    axes[0].set_ylim(s_min, s_max)
    y_ticks = np.linspace(s_min, s_max, 3)
    plot.set_yticks(
        axis=axes[0],
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 2),
        label="Peak response",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
    )

    axes[1].plot(
        x, temporal_peaks, clip_on=False, linestyle="-", marker="o", color="dodgerblue"
    )
    axes[1].vlines(
        temporal_elbow, ymin=t_min, ymax=t_max, linestyles="--", color="black"
    )
    plot.set_xticks(
        axis=axes[1],
        ticks=x,
        tick_labels=np.round(x, 2),
        label="Temporal freq. keep rate",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
    )
    axes[1].set_ylim(t_min, t_max)
    y_ticks = np.linspace(t_min, t_max, 3)
    plot.set_yticks(
        axis=axes[1],
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, 2),
        label="Peak response",
        tick_fontsize=FONTSIZE,
        label_fontsize=FONTSIZE,
    )

    for ax in axes:
        sns.despine(ax=ax)
        plot.set_ticks_params(axis=ax, length=2, pad=1)
    plot.save_figure(figure, filename=plot_dir / "elbow.jpg", dpi=DPI)


import matplotlib.pyplot as plt
import numpy as np


def curvature_analysis(peak_responses: np.ndarray, sensorium_max: np.ndarray):
    accuracy_matrix = 100 * (peak_responses - sensorium_max) / sensorium_max
    # Compute the first derivatives
    dx = np.gradient(accuracy_matrix, axis=1)  # Partial derivative along x (columns)
    dy = np.gradient(accuracy_matrix, axis=0)  # Partial derivative along y (rows)

    # Compute the second derivatives
    dxx = np.gradient(dx, axis=1)  # Second partial derivative along x
    dyy = np.gradient(dy, axis=0)  # Second partial derivative along y
    dxy = np.gradient(dx, axis=0)  # Mixed partial derivative

    # Compute the curvature
    numerator = np.abs(dxx * dy**2 - 2 * dx * dy * dxy + dyy * dx**2)
    denominator = (dx**2 + dy**2 + 1e-10) ** (3 / 2)  # Avoid division by zero
    curvature = numerator / denominator

    # Find the maximum curvature point
    max_curvature_idx = np.unravel_index(np.argmax(curvature), curvature.shape)
    max_curvature_point = (max_curvature_idx[0], max_curvature_idx[1])
    print(f"Elbow point (max curvature) at index: {max_curvature_point}")

    # Plotting
    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        gridspec_kw={"wspace": 0.4, "hspace": 0.0},
        figsize=(14, 5),
        dpi=DPI,
    )

    # Accuracy heatmap
    acc_map = axes[0].imshow(accuracy_matrix, cmap="inferno", vmin=0, vmax=200)
    axes[0].set_title("Response gain")
    axes[0].scatter(
        *max_curvature_point[::-1],
        color="white",
        edgecolor="black",
        s=40,
        label="Elbow Point",
    )
    axes[0].legend()

    # Gradient magnitude
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    g_min, g_max = np.floor(gradient_magnitude.min()), np.ceil(gradient_magnitude.max())
    grad_map = axes[1].imshow(gradient_magnitude, cmap="plasma", vmin=g_min, vmax=g_max)
    axes[1].set_title("Gradient Magnitude")

    # Curvature heatmap
    c_min, c_max = np.floor(curvature.min()), np.ceil(curvature.max())
    curv_map = axes[2].imshow(curvature, cmap="magma", vmin=c_min, vmax=c_max)
    axes[2].set_title("Curvature")
    axes[2].scatter(
        *max_curvature_point[::-1],
        color="white",
        edgecolor="black",
        s=40,
        label="Elbow Point",
    )
    axes[2].legend()

    ticks = np.arange(0.1, 1.0, 0.1)
    for i in range(len(axes)):
        plot.set_xticks(
            axis=axes[i],
            ticks=np.arange(len(ticks)),
            tick_labels=np.round(ticks, 1),
            label="Temporal freq. keep rate",
            tick_fontsize=FONTSIZE,
            label_fontsize=FONTSIZE,
        )
        plot.set_yticks(
            axis=axes[i],
            ticks=np.arange(len(ticks)),
            tick_labels=np.round(ticks, 1),
            label="Spatial freq. keep rate",
            tick_fontsize=FONTSIZE,
            label_fontsize=FONTSIZE,
        )
        pos = axes[i].get_position()
        cbar_width, cbar_height = 0.03 * (pos.x1 - pos.x0), 0.5 * (pos.y1 - pos.y0)
        cbar_ax = figure.add_axes(
            rect=(
                1.02 * pos.x1,
                (((pos.y1 - pos.y0) / 2) + pos.y0) - (cbar_height / 2),
                cbar_width,
                cbar_height,
            )
        )
        cbar = figure.colorbar([acc_map, grad_map, curv_map][i], cax=cbar_ax)
        # cbar.mappable.set_clim(vmin=0, vmax=200)

    plt.show()


def maximum_gradient(peak_responses: np.ndarray, sensorium_max: np.ndarray):
    accuracy_matrix = 100 * (peak_responses - sensorium_max) / sensorium_max
    # Compute gradients
    dx = np.gradient(accuracy_matrix, axis=1)
    dy = np.gradient(accuracy_matrix, axis=0)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # Find the maximum gradient point
    max_gradient_idx = np.unravel_index(
        np.argmax(gradient_magnitude), gradient_magnitude.shape
    )
    max_gradient_point = (max_gradient_idx[0], max_gradient_idx[1])
    print(f"Elbow point (max gradient) at index: {max_gradient_point}")

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Accuracy heatmap
    acc_map = ax[0].imshow(accuracy_matrix, cmap="viridis")
    ax[0].set_title("Accuracy Matrix")
    ax[0].scatter(*max_gradient_point[::-1], color="red", label="Elbow Point")
    ax[0].legend()
    cbar_acc = fig.colorbar(acc_map, ax=ax[0], orientation="vertical")
    cbar_acc.set_label("Accuracy")

    # Gradient magnitude heatmap
    grad_map = ax[1].imshow(gradient_magnitude, cmap="plasma")
    ax[1].set_title("Gradient Magnitude")
    ax[1].scatter(*max_gradient_point[::-1], color="red", label="Elbow Point")
    cbar_grad = fig.colorbar(grad_map, ax=ax[1], orientation="vertical")
    cbar_grad.set_label("Gradient Magnitude")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_laplace


def ridge_detection(peak_responses: np.ndarray, sensorium_max: np.ndarray):
    accuracy_matrix = 100 * (peak_responses - sensorium_max) / sensorium_max
    # Optional: Smooth the matrix to simulate a realistic surface
    accuracy_matrix = gaussian_filter(accuracy_matrix, sigma=1)

    # Compute second-order derivatives (Hessian matrix components)
    dxx = gaussian_filter(
        accuracy_matrix, sigma=1, order=(0, 2)
    )  # Second derivative along x
    dyy = gaussian_filter(
        accuracy_matrix, sigma=1, order=(2, 0)
    )  # Second derivative along y
    dxy = gaussian_filter(accuracy_matrix, sigma=1, order=(1, 1))  # Mixed derivative

    # Compute determinant of the Hessian matrix
    hessian_det = dxx * dyy - dxy**2

    # Find ridges by identifying local maxima of the determinant of the Hessian
    ridges = hessian_det > np.percentile(hessian_det, 95)  # Threshold top 5% as ridges

    # Get indices of ridge points
    ridge_indices = np.argwhere(ridges)
    print("Ridge indices:", ridge_indices)

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original accuracy matrix
    acc_map = ax[0].imshow(accuracy_matrix, cmap="viridis")
    ax[0].set_title("Accuracy Matrix")
    cbar_acc = fig.colorbar(acc_map, ax=ax[0], orientation="vertical")
    cbar_acc.set_label("Accuracy")

    # Determinant of the Hessian
    hessian_map = ax[1].imshow(hessian_det, cmap="magma")
    ax[1].set_title("Hessian Determinant")
    cbar_hessian = fig.colorbar(hessian_map, ax=ax[1], orientation="vertical")
    cbar_hessian.set_label("Hessian Determinant")

    # Ridge detection
    ax[2].imshow(accuracy_matrix, cmap="viridis")
    ax[2].scatter(
        ridge_indices[:, 1], ridge_indices[:, 0], color="red", label="Ridge Points"
    )
    ax[2].set_title("Ridge Detection")
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def main(args):
    mouse_id, neuron = "A", 4286
    plot_dir = args.output_dir / "MEVs" / "figures"
    peak_responses, videos = get_grid_result(MEVs_dir=args.MEVs_dir)

    plot_peak_responses(
        args,
        mouse_id=mouse_id,
        neuron=neuron,
        peak_responses=peak_responses,
        plot_dir=plot_dir,
    )
    # plot_MEVs(
    #     videos=videos,
    #     plot_dir=plot_dir,
    # )

    # find_elbow_point(
    #     args,
    #     peak_response=peak_responses,
    #     mouse_id=mouse_id,
    #     neuron=neuron,
    #     plot_dir=plot_dir,
    # )

    sensorium_max, _ = get_sensorium_response_stats(
        args.output_dir, mouse_id=mouse_id, neuron=neuron
    )
    curvature_analysis(peak_responses, sensorium_max=sensorium_max)
    maximum_gradient(peak_responses, sensorium_max=sensorium_max)
    ridge_detection(peak_responses, sensorium_max=sensorium_max)
    print(f"\nSaved plot to {plot_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="../data/sensorium")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--MEVs_dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
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
