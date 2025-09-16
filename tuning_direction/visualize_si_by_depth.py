import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()
plt.style.use("seaborn-v0_8-deep")


TICK_FONTSIZE = 10
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 12

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures/si_by_depth")


def get_mouse_info(
    save_dir: Path, mouse_id: str
) -> (np.ndarray, Dict[str, np.ndarray]):
    neuron_coordinates = data.get_neuron_coordinates(mouse_id=mouse_id)
    tuning_properties = utils.load_tuning(save_dir, mouse_id=mouse_id)
    tuning_properties.pop("tuning_curves", None)
    tuning_properties.pop("tuning_curves_tf_sf", None)
    return neuron_coordinates, tuning_properties


def get_color(si_name: str):
    match si_name:
        case "OSI" | "OSI_tf_sf":
            cmap = "orangered"
        case "DSI" | "DSI_tf_sf":
            cmap = "limegreen"
        case "SSI":
            cmap = "dodgerblue"
        case _:
            raise ValueError(f"Unknown selectivity index name: {si_name}")
    return cmap


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


def plot_si_by_depth(
    neuron_coordinates: np.ndarray,
    si_name: str,
    si_values: np.ndarray,
    si_quantile: float = 0.0,
    title: str = None,
    filename: Path = None,
    dpi: int = 240,
):
    # remove non-selective neurons
    remove_neurons = np.where(np.isnan(si_values))[0]
    if si_quantile > 0:
        si_threshold = np.nanquantile(si_values, q=si_quantile)
        remove_neurons = np.union1d(
            remove_neurons, np.where(si_values < si_threshold)[0]
        )
    si_values = np.delete(si_values, remove_neurons)
    neuron_coordinates = np.delete(neuron_coordinates, remove_neurons, axis=0)

    unique_depths = np.unique(neuron_coordinates[:, 2])

    df = pd.DataFrame({"depth": neuron_coordinates[:, 2], "si": si_values})

    figure, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(6, 3), dpi=dpi, facecolor="white"
    )

    # sns.violinplot(
    #     data=df,
    #     x="depth",
    #     y="si",
    #     color=get_color(si_name),
    #     inner="box",
    #     linewidth=0.7,
    #     linecolor="black",
    #     zorder=1,
    #     ax=ax,
    # )
    sns.stripplot(
        data=df,
        x="depth",
        y="si",
        color=get_color(si_name),
        size=1,
        zorder=1,
        ax=ax,
    )
    sns.despine(ax=ax, top=True, right=True)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    plot.set_xticks(
        axis=ax,
        ticks=ax.get_xticks(),
        tick_labels=unique_depths.astype(int),
        label="Depth (μm)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
    )
    yticks = np.linspace(
        np.floor(df["si"].min()),
        np.ceil(df["si"].max()),
        3,
    )
    ax.set_ylim(yticks[0], yticks[-1])
    plot.set_yticks(
        axis=ax,
        ticks=yticks,
        tick_labels=np.round(yticks, 1),
        label="SI" if si_quantile == 0.0 else f"SI (threshold q={si_quantile})",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
    )
    for y in yticks:
        ax.axhline(y=y, color="gray", linewidth=1, alpha=0.5, zorder=-1, clip_on=False)

    plot.save_figure(figure, filename=filename, dpi=dpi)


def main(args):
    for name, save_dir in [
        ("recorded", data.METADATA_DIR),
        # ("fCNN", Path("../runs/fCNN/009_fCNN/")),
        # ("ViViT", Path("../runs/vivit/047_vivit_RoPE_regTokens4_cropFrame300/")),
    ]:
        print(f"plot {name} responses...")
        for mouse_id in data.SENSORIUM_OLD:
            neuron_coordinates, tuning_properties = get_mouse_info(
                save_dir, mouse_id=mouse_id
            )
            if not tuning_properties:
                continue
            plot_dir = PLOT_DIR / name / f"mouse{mouse_id}"
            for si_name, si_values in tuning_properties.items():
                print(f"\tplot {si_name} for mouse {mouse_id}")
                for si_quantile in (0.0, 0.5, 0.95):
                    plot_si_by_depth(
                        neuron_coordinates=neuron_coordinates,
                        si_name=si_name,
                        si_values=si_values,
                        si_quantile=si_quantile,
                        title=f"Mouse {mouse_id} {name} {si_name.replace('_tf_sf', ' (TF/SF)')}",
                        filename=plot_dir
                        / f"quantile{si_quantile}"
                        / f"mouse{mouse_id}_{si_name}.jpg",
                        dpi=args.dpi,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=240)
    main(parser.parse_args())
