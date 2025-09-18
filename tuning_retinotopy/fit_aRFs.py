import argparse
import warnings
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from einops import rearrange
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

warnings.simplefilter("error", opt.OptimizeWarning)

HEIGHT, WIDTH = 36, 64

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "jpg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

PLOT_DIR = Path("figures") / "aRFs"
PARALLEL = True  # use parallel processing

plot.set_font()


def Gaussian2d(
    xy: tuple[np.ndarray, np.ndarray],
    amplitude: float,
    xo: float,
    yo: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
):

    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2))
    )
    return rearrange(g, "h w -> (h w)")


def fit_aRF(
    arguments: tuple[np.ndarray, np.ndarray, np.ndarray, np.random.RandomState],
) -> np.ndarray:
    aRF, x, y, rng = arguments

    h, w = aRF.shape[1:]
    data = rearrange(aRF, "1 h w -> (h w)")
    data_noisy = data + 0.2 * rng.normal(size=data.shape)
    try:
        popt, pcov = opt.curve_fit(
            f=Gaussian2d,
            xdata=(x, y),
            ydata=data_noisy,
            p0=(3, w // 2, h // 2, 10, 10, 0, 10),
        )
    except (RuntimeError, opt.OptimizeWarning):
        popt = np.full(7, fill_value=np.nan)
    return popt.astype(np.float32)


def fit_aRFs(
    aRFs: np.ndarray, mouse_id: str, rng: np.random.RandomState
) -> pd.DataFrame:
    """Fit 2D Gaussian to each aRFs using SciPy curve_fit

    Gaussian fit reference: https://stackoverflow.com/a/21566831

    Returns:
        popts: pd.DataFrame, a (num. units, 7) array with fitted parameters in
            [amplitude, center x, center y, sigma x, sigma y, theta, offset]
    """
    assert len(aRFs.shape) == 4
    num_units = aRFs.shape[0]

    # standardize RFs and take absolute values to remove background noise
    mean = np.mean(aRFs, axis=(1, 2, 3), keepdims=True)
    std = np.std(aRFs, axis=(1, 2, 3), keepdims=True)
    aRFs = (aRFs - mean) / std
    aRFs = np.abs(aRFs)

    height, width = aRFs.shape[2:]
    x, y = np.linspace(0, width - 1, width), np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    if PARALLEL:
        popts = process_map(
            fit_aRF,
            [(aRFs[i], x, y, rng) for i in range(num_units)],
            max_workers=cpu_count() - 2,
            chunksize=1,
            desc=f"mouse {mouse_id}",
        )
    else:
        popts = [
            fit_aRF((aRFs[i], x, y, rng))
            for i in tqdm(range(num_units), desc=f"mouse {mouse_id}")
        ]

    popts = np.stack(popts)

    # filter out the last 5% of the results to eliminate poor fit
    num_drops = int(0.05 * len(popts))
    large_sigma_x = np.argsort(popts[:, 3])[-num_drops:]
    large_sigma_y = np.argsort(popts[:, 4])[-num_drops:]
    drop_units = np.unique(np.concatenate((large_sigma_x, large_sigma_y), axis=0))
    popts[drop_units] = np.nan

    print(
        f"sigma X: {np.nanmean(popts[:, 3]):.03f} "
        f"+/- {np.nanstd(popts[:, 3]):.03f}\n"
        f"sigma Y: {np.nanmean(popts[:, 4]):.03f} "
        f"+/- {np.nanstd(popts[:, 4]):.03f}"
    )

    parameters = pd.DataFrame(
        {
            "neuron": list(range(popts.shape[0])),
            "bad_fit": np.any(np.isnan(popts), axis=1),
            "amplitude": popts[:, 0],
            "center_x": popts[:, 1],
            "center_y": popts[:, 2],
            "sigma_x": popts[:, 3],
            "sigma_y": popts[:, 4],
            "theta": popts[:, 5],
            "offset": popts[:, 6],
        }
    )
    return parameters


def plot_aRF(axis, aRF: np.ndarray, parameter: pd.DataFrame, title: str = None):
    height, width = aRF.shape[1], aRF.shape[2]
    x, y = np.linspace(0, width - 1, width), np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    normalize = lambda a: (a - a.min()) / (a.max() - a.min())
    aRF = normalize(aRF)
    axis.imshow(aRF[0], aspect="equal", cmap="gray", vmin=0, vmax=1)
    assert len(parameter.index) == 1
    if not parameter.iloc[0].bad_fit:
        fitted = Gaussian2d(
            (x, y),
            amplitude=parameter.iloc[0].amplitude,
            xo=parameter.iloc[0].center_x,
            yo=parameter.iloc[0].center_y,
            sigma_x=parameter.iloc[0].sigma_x,
            sigma_y=parameter.iloc[0].sigma_y,
            theta=parameter.iloc[0].theta,
            offset=parameter.iloc[0].offset,
        )
        fitted = rearrange(fitted, "(h w) -> h w", h=HEIGHT, w=WIDTH)
        fitted = fitted.reshape(height, width)
        # plot contour of 1 standard deviation ellipse
        axis.contour(
            x,
            y,
            normalize(fitted),
            levels=[0.158],  # 1 standard deviation
            alpha=0.7,
            linewidths=1.5,
            colors="red",
        )
    if title is not None:
        axis.set_xlabel(title, labelpad=2, fontsize=LABEL_FONTSIZE)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)


def plot_aRFs(
    aRFs: np.ndarray,
    parameters: pd.DataFrame,
    filename: Path,
    title: str = None,
    neurons: list[int] = None,
):
    nrows, ncols = 2, 3
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        gridspec_kw={"wspace": 0, "hspace": 0},
        figsize=((3 / 5) * PAPER_WIDTH, 1.4),
        dpi=DPI,
    )
    axes = axes.flatten()
    for n, neuron in enumerate(neurons):
        plot_aRF(
            axis=axes[n],
            aRF=aRFs[neuron],
            parameter=parameters.loc[parameters.neuron == neuron],
            title=f"Neuron #{neuron:04d}",
        )
    # if title is not None:
    #     axes[1].set_title(title, fontsize=LABEL_FONTSIZE, pad=4)
    plot.save_figure(figure, filename=filename, dpi=DPI)


def load_centers(parameters: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x, y = parameters.center_x.to_numpy(), parameters.center_y.to_numpy()
    neurons = np.intersect1d(np.where(x < WIDTH)[0], np.where(y < HEIGHT)[0])
    return x[neurons], y[neurons]


def plot_centers_KDE(
    model_name: str,
    parameters: pd.DataFrame,
    filename: Path,
    mouse_id: str,
):
    x, y = load_centers(parameters)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=((2 / 5) * PAPER_WIDTH, 1.4),
        gridspec_kw={"wspace": 0, "hspace": 0},
        dpi=DPI,
    )
    ax = axes[0]
    cbar_ax = axes[1]
    ax.set_position([0.11, 0.17, 0.7, 0.7])
    # force aspect ratio to 16:9
    ax.imshow(
        np.random.rand(HEIGHT, WIDTH),
        aspect="equal",
        cmap="gray",
        alpha=0,
    )
    pos = ax.get_position()
    cbar_ax.set_position(
        [pos.x1 + 0.03 * pos.width, pos.y0, 0.04 * pos.width, pos.height]
    )
    thresh, levels = 0.01, 10
    sns.kdeplot(
        x=x,
        y=y,
        ax=ax,
        fill=True,
        levels=np.linspace(thresh, 1.0, levels),
        thresh=thresh,
        cmap="inferno",
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"orientation": "vertical"},
    )
    ax.text(
        x=0.96 * WIDTH,
        y=2,
        s=f"{model_name[:12]}\nmouse {mouse_id}",
        fontsize=LABEL_FONTSIZE,
        va="top",
        ha="right",
        transform=ax.transData,
        linespacing=0.9,
    )
    x_ticks = np.array([0, WIDTH], dtype=int)
    ax.set_xlim(x_ticks[0], x_ticks[-1])
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_ticks,
        label="Width (px)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        linespacing=0.9,
    )
    y_ticks = np.array([0, HEIGHT], dtype=int)
    ax.set_ylim(HEIGHT, 0)
    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="Height (px)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        linespacing=0.9,
    )
    # y-ticks for colorbar
    y_ticks = cbar_ax.get_yticks()
    plot.set_yticks(
        axis=cbar_ax,
        ticks=y_ticks,
        tick_labels=[
            (f"{y_ticks[i]:.02f}" if i in (0, len(y_ticks) - 1) else "")
            for i in range(len(y_ticks))
        ],
        tick_fontsize=TICK_FONTSIZE,
    )
    linewidth = 1.2
    plot.set_ticks_params(axis=ax, length=2, linewidth=linewidth)
    plot.set_ticks_params(axis=cbar_ax, length=2, linewidth=linewidth)
    for _ax in [ax, cbar_ax]:
        for spine in _ax.spines.values():
            spine.set_linewidth(linewidth)
    ax.set_title("aRF center KDE", fontsize=LABEL_FONTSIZE, pad=3)

    plot.save_figure(figure, filename=filename, dpi=DPI)


def process_mouse(args, mouse_id: str, filename: Path):
    utils.set_random_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    aRFs = np.load(filename, allow_pickle=False)
    aRFs = aRFs["data"]
    parameters = fit_aRFs(aRFs=aRFs, mouse_id=mouse_id, rng=rng)
    parameters.insert(loc=0, column="mouse", value=mouse_id)

    model_name = args.output_dir.name
    # plot aRF and their 2D Gaussian fit of 6 randomly selected neurons

    neurons = sorted(rng.choice(aRFs.shape[0], size=6, replace=False))
    plot_aRFs(
        aRFs=aRFs,
        parameters=parameters,
        filename=PLOT_DIR / model_name / f"mouse{mouse_id}_aRFs.{FORMAT}",
        title=f"{model_name} aRFs (mouse {mouse_id})",
        neurons=neurons,
    )
    try:
        plot_centers_KDE(
            model_name=model_name,
            parameters=parameters,
            filename=PLOT_DIR / model_name / f"mouse{mouse_id}_aRF_center_KDE.{FORMAT}",
            mouse_id=mouse_id,
        )
    except ValueError as e:
        print(f"ValueError in sns.kdeplot: {e}")
    return parameters


def main(args):
    print(f"Process {args.output_dir}...")
    parameters = []
    for mouse_id in data.MOUSE_IDS.keys():
        filename = args.output_dir / "aRFs" / f"mouse{mouse_id}.npz"
        if filename.exists():
            parameter = process_mouse(args=args, mouse_id=mouse_id, filename=filename)
            parameters.append(parameter)
    parameters = pd.concat(parameters, ignore_index=True)
    filename = args.output_dir / "aRF.parquet"
    parameters.to_parquet(filename)
    print(f"Saved artificial RF parameters to {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    main(parser.parse_args())
