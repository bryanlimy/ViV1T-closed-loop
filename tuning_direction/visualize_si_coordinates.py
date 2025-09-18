import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()
plt.style.use("seaborn-v0_8-deep")


white = "#ffffff"
colormaps.register(LinearSegmentedColormap.from_list("0", [white, "#1d4e31"]))
colormaps.register(LinearSegmentedColormap.from_list("45", [white, "#5dc685"]))
colormaps.register(LinearSegmentedColormap.from_list("90", [white, "#e21d23"]))
colormaps.register(LinearSegmentedColormap.from_list("135", [white, "#ef682e"]))
colormaps.register(LinearSegmentedColormap.from_list("180", [white, "#ffc533"]))
colormaps.register(LinearSegmentedColormap.from_list("225", [white, "#9d926c"]))
colormaps.register(LinearSegmentedColormap.from_list("270", [white, "#3b5fa5"]))
colormaps.register(LinearSegmentedColormap.from_list("315", [white, "#c2d3ea"]))

TICK_FONTSIZE = 10
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 12

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures/selectivity_coordinates")


def get_mouse_info(
    save_dir: Path, mouse_id: str
) -> (np.ndarray, Dict[str, np.ndarray]):
    neuron_coordinates = data.get_neuron_coordinates(mouse_id=mouse_id)
    tuning_properties = utils.load_tuning(save_dir, mouse_id=mouse_id)
    return neuron_coordinates, tuning_properties


def get_colormap(si: str, direction: int = None):
    match (si, direction):
        case ("OSI", None):
            cmap = "Oranges"
        case ("DSI", None):
            cmap = "Greens"
        case ("SSI", None):
            cmap = "GnBu"
        case ("OSI", 0):
            cmap = "0"
        case ("OSI", 45):
            cmap = "90"
        case ("OSI", 90):
            cmap = "180"
        case ("OSI", 135):
            cmap = "270"
        case ("DSI", 0):
            cmap = "0"
        case ("DSI", 45):
            cmap = "45"
        case ("DSI", 90):
            cmap = "90"
        case ("DSI", 135):
            cmap = "135"
        case ("DSI", 180):
            cmap = "180"
        case ("DSI", 225):
            cmap = "225"
        case ("DSI", 270):
            cmap = "270"
        case ("DSI", 315):
            cmap = "315"
        case _:
            raise ValueError(f"Unknown SI: {si}, direction: {direction}.")
    return cm.ScalarMappable(cmap=cmap)


def normalize(x: np.ndarray):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def set_title(
    figure: plt.Figure,
    ax: plt.Axes,
    title: str,
    fontsize: int = TITLE_FONTSIZE,
    pad: float = 0,
):
    pos = ax.get_position()
    figure.text(
        x=0.5 * (pos.x1 - pos.x0) + pos.x0,
        y=pos.y1 + pad,
        s=title,
        fontsize=fontsize,
        ha="center",
    )


def plot_si_coordinates(
    neuron_coordinates: np.ndarray,
    si_values: np.ndarray,
    si_threshold: float,
    si_name: str = None,
    title: str = None,
    filename: Path = None,
    dpi: int = 240,
):
    num_neurons = len(neuron_coordinates)
    if si_threshold > 0:
        neurons = np.where(si_values >= si_threshold)[0]
    else:
        neurons = np.where(~np.isnan(si_values))[0]

    f_h, f_w = 4, 10
    figure = plt.figure(figsize=(f_w, f_h), dpi=dpi, facecolor="white")

    # create 3 equally spaced subplots with 3d projection
    width = 0.35
    height = width * f_w / f_h
    bottom = 0.5 - (height / 2)
    mid_points = np.linspace(0, 1, 4)
    mid_points = mid_points[1:] - (mid_points[1:] - mid_points[:-1]) / 2
    left_points = mid_points - (width / 2)
    # axes for 3D scatter plots
    s_axes = []
    for left in left_points:
        s_axes.append(
            figure.add_axes(
                rect=(left, bottom, width, height),
                projection="3d",
            )
        )
    # axis for colorbar
    cbar_ax = figure.add_axes(
        rect=(
            mid_points[-1] + (width / 2) - 0.085,
            bottom + height - 0.125,
            0.007,
            0.09,
        )
    )
    # axes for histograms
    width = 0.5 * width
    left_points = mid_points - (width / 2)
    h_axes = []
    for left in left_points:
        h_axes.append(
            figure.add_axes(
                rect=(left, bottom - 0.1, width, 0.1),
            )
        )
    del mid_points, left_points, width, height, bottom

    # colorbar
    c_min = np.nanmin(si_values)
    c_max = np.nanmax(si_values)

    mappable = get_colormap(si=si_name)
    mappable.set_clim(c_min, c_max)
    plt.colorbar(mappable, cax=cbar_ax, shrink=0.5)
    plot.set_yticks(
        cbar_ax,
        ticks=[c_min, c_max],
        tick_labels=np.round([c_min, c_max], 1),
        tick_fontsize=TICK_FONTSIZE,
    )

    cbar_pos = cbar_ax.get_position()
    figure.text(
        x=cbar_pos.x0 - 0.03, y=cbar_pos.y0, s=si_name[:3], fontsize=TICK_FONTSIZE
    )
    plot.set_ticks_params(cbar_ax, length=2)

    get_ticks = lambda coors: np.linspace(
        np.floor(np.min(coors) * 0.1) * 10 - 20,
        np.ceil(np.max(coors) * 0.1) * 10 + 20,
        5,
        dtype=int,
    )
    x_ticks = get_ticks(neuron_coordinates[:, 0])
    y_ticks = get_ticks(neuron_coordinates[:, 1])
    z_ticks = get_ticks(neuron_coordinates[:, 2])

    for ax in s_axes:
        colors = np.zeros((num_neurons, 4), dtype=np.float32)
        colors[neurons] = mappable.to_rgba(0.8)
        colors[neurons, -1] = normalize(si_values[neurons])

        ax.scatter(
            neuron_coordinates[neurons, 0],
            neuron_coordinates[neurons, 1],
            neuron_coordinates[neurons, 2],
            s=10,
            c=colors[neurons],
            marker=".",
            depthshade=False,
        )
        ax.set_xlabel("x coordinate (μm)", fontsize=TICK_FONTSIZE)
        ax.set_ylabel("y coordinate (μm)", fontsize=TICK_FONTSIZE)
        ax.set_zlabel("z coordinate (μm)", fontsize=TICK_FONTSIZE)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_xticks(x_ticks, labels=x_ticks, fontsize=TICK_FONTSIZE)
        ax.set_ylim(y_ticks[0], y_ticks[-1])
        ax.set_yticks(y_ticks, labels=y_ticks, fontsize=TICK_FONTSIZE)
        ax.set_zlim(z_ticks[0], z_ticks[-1])
        ax.set_zticks(z_ticks, labels=z_ticks, fontsize=TICK_FONTSIZE)
        ax.invert_zaxis()
        plot.set_ticks_params(ax, length=2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.0, 0.0, 0.0, 0.3))
            axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 0.4)

    # set orientation of the 3D scatter plot for each vide
    pad = -0.1
    set_title(figure, ax=s_axes[0], title="x-y view", fontsize=LABEL_FONTSIZE, pad=pad)
    s_axes[0].view_init(elev=90, azim=-90, roll=0)
    s_axes[0].set_zticks([])
    s_axes[0].set_zlabel("")

    set_title(figure, ax=s_axes[1], title="x-z view", fontsize=LABEL_FONTSIZE, pad=pad)
    s_axes[1].view_init(elev=1, azim=90, roll=0)
    s_axes[1].set_yticks([])
    s_axes[1].set_ylabel("")

    set_title(figure, ax=s_axes[2], title="y-z view", fontsize=LABEL_FONTSIZE, pad=pad)
    s_axes[2].view_init(elev=1, azim=0, roll=0)
    s_axes[2].set_xticks([])
    s_axes[2].set_xlabel("")

    # plot histograms
    ticks = [("x", x_ticks), ("y", y_ticks), ("z", z_ticks)]
    for i, ax in enumerate(h_axes):
        counts, bins = np.histogram(neuron_coordinates[neurons, i], bins=10)
        percentages = 100 * counts / sum(counts)
        ax.hist(
            x=bins[:-1],
            bins=bins,
            weights=percentages,
            color=mappable.cmap(0.7),
            edgecolor="black",
            zorder=1,
        )
        ax.set_xlim(ticks[i][1][0], ticks[i][1][-1])
        plot.set_xticks(
            ax,
            ticks=ticks[i][1],
            tick_labels=ticks[i][1],
            label=f"{ticks[i][0]} coordinates (μm)",
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=TICK_FONTSIZE,
        )
        yticks = [0, 20]
        plot.set_yticks(
            ax,
            ticks=yticks,
            tick_labels=[f"{y}%" for y in yticks],
            tick_fontsize=TICK_FONTSIZE,
        )
        for y in (10, 20):
            ax.axhline(
                y=y, color="gray", linewidth=1, alpha=0.5, zorder=-1, clip_on=False
            )
        sns.despine(ax=ax, trim=True)

    h_axes[1].set_title(
        f"SI threshold {si_threshold:.02f}",
        fontsize=LABEL_FONTSIZE,
    )

    set_title(
        figure,
        ax=s_axes[1],
        title=title,
        fontsize=TITLE_FONTSIZE,
        pad=-0.05,
    )

    plot.save_figure(figure, filename=filename, dpi=dpi)


def plot_direction_coordinates(
    neuron_coordinates: np.ndarray,
    direction: int | np.ndarray,
    neurons: np.ndarray,
    si_values: np.ndarray,
    si_threshold: float,
    si_name: str = None,
    title: str = None,
    filename: Path = None,
    dpi: int = 240,
):
    num_neurons = len(neuron_coordinates)
    if si_threshold > 0:
        si_neurons = np.where(si_values >= si_threshold)[0]
    else:
        si_neurons = np.where(~np.isnan(si_values))[0]

    neurons = np.intersect1d(neurons, si_neurons, assume_unique=True)

    f_h, f_w = 4, 10
    figure = plt.figure(figsize=(f_w, f_h), dpi=dpi, facecolor="white")

    # create 3 equally spaced subplots with 3d projection
    width = 0.35
    height = width * f_w / f_h
    bottom = 0.5 - (height / 2)
    mid_points = np.linspace(0, 1, 4)
    mid_points = mid_points[1:] - (mid_points[1:] - mid_points[:-1]) / 2
    left_points = mid_points - (width / 2)
    # axes for 3D scatter plots
    s_axes = []
    for left in left_points:
        s_axes.append(
            figure.add_axes(
                rect=(left, bottom, width, height),
                projection="3d",
            )
        )
    # axis for colorbar
    cbar_ax = figure.add_axes(
        rect=(
            mid_points[-1] + (width / 2) - 0.085,
            bottom + height - 0.125,
            0.007,
            0.09,
        )
    )
    # axes for histograms
    width = 0.5 * width
    left_points = mid_points - (width / 2)
    h_axes = []
    for left in left_points:
        h_axes.append(
            figure.add_axes(
                rect=(left, bottom - 0.1, width, 0.1),
            )
        )
    del mid_points, left_points, width, height, bottom

    # colorbar
    c_min = np.nanmin(si_values)
    c_max = np.nanmax(si_values)

    mappable = get_colormap(si=si_name, direction=direction)
    mappable.set_clim(c_min, c_max)
    plt.colorbar(mappable, cax=cbar_ax, shrink=0.5)
    plot.set_yticks(
        cbar_ax,
        ticks=[c_min, c_max],
        tick_labels=np.round([c_min, c_max], 1),
        tick_fontsize=TICK_FONTSIZE,
    )

    cbar_pos = cbar_ax.get_position()
    figure.text(
        x=cbar_pos.x0 - 0.03, y=cbar_pos.y0, s=si_name[:3], fontsize=TICK_FONTSIZE
    )
    plot.set_ticks_params(cbar_ax, length=2)

    get_ticks = lambda coors: np.linspace(
        np.floor(np.min(coors) * 0.1) * 10 - 20,
        np.ceil(np.max(coors) * 0.1) * 10 + 20,
        5,
        dtype=int,
    )
    x_ticks = get_ticks(neuron_coordinates[:, 0])
    y_ticks = get_ticks(neuron_coordinates[:, 1])
    z_ticks = get_ticks(neuron_coordinates[:, 2])

    for ax in s_axes:
        colors = np.zeros((num_neurons, 4), dtype=np.float32)
        colors[neurons] = mappable.to_rgba(0.8)
        colors[neurons, -1] = normalize(si_values[neurons])

        ax.scatter(
            neuron_coordinates[neurons, 0],
            neuron_coordinates[neurons, 1],
            neuron_coordinates[neurons, 2],
            s=10,
            c=colors[neurons],
            marker=".",
            depthshade=False,
        )
        ax.set_xlabel("x coordinate (μm)", fontsize=TICK_FONTSIZE)
        ax.set_ylabel("y coordinate (μm)", fontsize=TICK_FONTSIZE)
        ax.set_zlabel("z coordinate (μm)", fontsize=TICK_FONTSIZE)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_xticks(x_ticks, labels=x_ticks, fontsize=TICK_FONTSIZE)
        ax.set_ylim(y_ticks[0], y_ticks[-1])
        ax.set_yticks(y_ticks, labels=y_ticks, fontsize=TICK_FONTSIZE)
        ax.set_zlim(z_ticks[0], z_ticks[-1])
        ax.set_zticks(z_ticks, labels=z_ticks, fontsize=TICK_FONTSIZE)
        ax.invert_zaxis()
        plot.set_ticks_params(ax, length=2)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.0, 0.0, 0.0, 0.3))
            axis._axinfo["grid"]["color"] = (0.0, 0.0, 0.0, 0.4)

    # set orientation of the 3D scatter plot for each vide
    pad = -0.1
    set_title(figure, ax=s_axes[0], title="x-y view", fontsize=LABEL_FONTSIZE, pad=pad)
    s_axes[0].view_init(elev=90, azim=-90, roll=0)
    s_axes[0].set_zticks([])
    s_axes[0].set_zlabel("")

    set_title(figure, ax=s_axes[1], title="x-z view", fontsize=LABEL_FONTSIZE, pad=pad)
    s_axes[1].view_init(elev=1, azim=90, roll=0)
    s_axes[1].set_yticks([])
    s_axes[1].set_ylabel("")

    set_title(figure, ax=s_axes[2], title="y-z view", fontsize=LABEL_FONTSIZE, pad=pad)
    s_axes[2].view_init(elev=1, azim=0, roll=0)
    s_axes[2].set_xticks([])
    s_axes[2].set_xlabel("")

    # plot histograms
    ticks = [("x", x_ticks), ("y", y_ticks), ("z", z_ticks)]
    for i, ax in enumerate(h_axes):
        counts, bins = np.histogram(neuron_coordinates[neurons, i], bins=10)
        percentages = 100 * counts / sum(counts)
        ax.hist(
            x=bins[:-1],
            bins=bins,
            weights=percentages,
            color=mappable.cmap(0.7),
            edgecolor="black",
            zorder=1,
        )
        ax.set_xlim(ticks[i][1][0], ticks[i][1][-1])
        plot.set_xticks(
            ax,
            ticks=ticks[i][1],
            tick_labels=ticks[i][1],
            label=f"{ticks[i][0]} coordinates (μm)",
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=TICK_FONTSIZE,
        )
        yticks = [0, 20]
        plot.set_yticks(
            ax,
            ticks=yticks,
            tick_labels=[f"{y}%" for y in yticks],
            tick_fontsize=TICK_FONTSIZE,
        )
        for y in (10, 20):
            ax.axhline(
                y=y, color="gray", linewidth=1, alpha=0.5, zorder=-1, clip_on=False
            )
        sns.despine(ax=ax, trim=True)

    h_axes[1].set_title(
        f"SI threshold {si_threshold:.02f}",
        fontsize=LABEL_FONTSIZE,
    )

    set_title(
        figure,
        ax=s_axes[1],
        title=title,
        fontsize=TITLE_FONTSIZE,
        pad=-0.05,
    )

    plot.save_figure(figure, filename=filename, dpi=dpi)


def main(args):
    for name, save_dir in [
        ("recorded", data.METADATA_DIR),
        # ("fCNN", Path("../runs/fCNN/009_fCNN/")),
        # ("ViViT", Path("../runs/vivit/047_vivit_RoPE_regTokens4_cropFrame300/")),
    ]:
        print(f"plot {name} responses...")
        for mouse_id in data.SENSORIUM_OLD:
            if mouse_id not in ("B", "C", "E"):
                continue
            neuron_coordinates, tuning_properties = get_mouse_info(
                save_dir, mouse_id=mouse_id
            )
            plot_dir = PLOT_DIR / name / f"mouse{mouse_id}"
            for si_name in ("OSI", "DSI", "SSI"):
                if si_name not in tuning_properties:
                    continue
                print(f"\tplot {si_name} for mouse {mouse_id}")
                plot_si_coordinates(
                    neuron_coordinates=neuron_coordinates,
                    si_values=tuning_properties[si_name],
                    si_threshold=args.si_threshold,
                    si_name=si_name,
                    title=f"Mouse {mouse_id} {name} {si_name}",
                    filename=plot_dir / f"mouse{mouse_id}_{si_name}.jpg",
                    dpi=args.dpi,
                )
                if si_name == "SSI":
                    continue
                tuning_curves = tuning_properties["tuning_curve"]
                directions = np.array(list(tuning_curves[0].keys()), dtype=np.int32)
                tuning_curves = np.array(
                    [
                        np.array(list(tuning_curve.values()))
                        for tuning_curve in tuning_curves.values()
                    ],
                    dtype=np.float32,
                )
                if si_name == "OSI":
                    # combine opposite directions for a single orientation value
                    directions = directions[:4]
                    tuning_curves = tuning_curves[:, :4] + tuning_curves[:, 4:]
                preferred_directions = np.argmax(tuning_curves, axis=-1)
                for i, direction in enumerate(directions):
                    plot_direction_coordinates(
                        neuron_coordinates=neuron_coordinates,
                        direction=direction,
                        neurons=np.where(preferred_directions == i)[0],
                        si_values=tuning_properties[si_name],
                        si_threshold=args.si_threshold,
                        si_name=si_name,
                        title=f"Mouse {mouse_id} {name} {si_name} ({direction}°)",
                        filename=plot_dir
                        / "preferred_directions"
                        / f"mouse{mouse_id}_{si_name}_{direction}°.jpg",
                        dpi=args.dpi,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--si_threshold", type=float, default=0.3)
    main(parser.parse_args())
