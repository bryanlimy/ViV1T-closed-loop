from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import viv1t.data.utils as utils
from viv1t.utils import plot

plot.set_font()


DPI = 240
TICK_FONTSIZE = 9
LABEL_FONTSIZE = 10

DATA_DIR = Path("../data/sensorium")
PLOT_DIR = Path("figures") / "tuning_similarity_illustration"


def plot_planes(coordinates: np.ndarray, filename: Path):
    rng = np.random.default_rng(1234)

    linewidth = 1.2

    figure = plt.figure(figsize=(3, 3), layout="constrained", dpi=DPI)
    ax = figure.add_subplot(111, projection="3d")
    ax.view_init(elev=8, azim=-50, roll=-0.5)

    # Get unique z-planes
    unique_z_planes = np.unique(coordinates[:, 2])
    delta_z = unique_z_planes[1] - unique_z_planes[0]

    for i, z_plane in enumerate(unique_z_planes):
        # Select points in the current z-plane
        neurons = np.where(coordinates[:, 2] == z_plane)[0]
        # Subsample to only show fewer points per plane
        subsampled_coordinates = coordinates[rng.choice(neurons, 300, replace=False)]

        # Plot the points for the current z-plane
        ax.scatter(
            subsampled_coordinates[:, 0],
            subsampled_coordinates[:, 1],
            subsampled_coordinates[:, 2],
            marker=".",
            edgecolors="none",
            facecolors="black",
            alpha=0.2,
            s=15,
            zorder=i,
        )
        # Fit a plane to the points in the current z-plane
        X = subsampled_coordinates[:, :2]
        y = subsampled_coordinates[:, 2]
        reg = LinearRegression().fit(subsampled_coordinates[:, :2], y)
        xx, yy = np.meshgrid(
            np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 10),
            np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10),
        )
        zz = reg.intercept_ + reg.coef_[0] * xx + reg.coef_[1] * yy

        # Plot the fitted plane borders
        ax.plot(
            xx[0, :], yy[0, :], zz[0, :], color="black", linewidth=0.5, zorder=i
        )  # Top edge
        ax.plot(
            xx[-1, :], yy[-1, :], zz[-1, :], color="black", linewidth=0.5, zorder=i
        )  # Bottom edge
        ax.plot(
            xx[:, 0], yy[:, 0], zz[:, 0], color="black", linewidth=0.5, zorder=i
        )  # Left edge
        ax.plot(
            xx[:, -1], yy[:, -1], zz[:, -1], color="black", linewidth=0.5, zorder=i
        )  # Right edge

    # Three locations
    loc_1 = [-900, -550, unique_z_planes[6]]
    loc_2 = [-850, -400, unique_z_planes[0] - 3]
    loc_3 = [-900, -550, unique_z_planes[0]]

    line_kw = {"c": "black", "linewidth": 1.2}
    # # Plot the vertical \Delta line shifted to the left and down
    # shift_x = 220
    # ax.plot(
    #     [loc_1[0] - shift_x, loc_3[0] - shift_x],
    #     [loc_1[1], loc_3[1]],
    #     [loc_1[2], loc_3[2]],
    #     **line_kw,
    # )
    # for loc in [loc_1, loc_3]:
    #     ax.plot(
    #         [loc[0] - shift_x + 15, loc[0] - shift_x - 15],
    #         [loc[1], loc[1]],
    #         [loc[2], loc[2]],
    #         **line_kw,
    #     )
    # # Label delta
    # mid_x_shifted = ((loc_1[0] - shift_x + loc_3[0] - shift_x) / 2) - 20
    # mid_y_shifted = (loc_1[1] + loc_3[1]) / 2
    # mid_z_shifted = (loc_1[2] + loc_3[2]) / 2 + 10
    # ax.text(
    #     mid_x_shifted,
    #     mid_y_shifted,
    #     mid_z_shifted,
    #     s=r"$\Delta$",
    #     color="black",
    #     ha="right",
    #     fontsize=LABEL_FONTSIZE,
    # )
    #
    # # Add d line
    # shift_z = -25
    # ax.plot(
    #     [loc_3[0], loc_2[0]],
    #     [loc_3[1], loc_2[1]],
    #     [loc_3[2] + shift_z, loc_2[2] + shift_z],
    #     **line_kw,
    # )
    # for loc in [loc_2, loc_3]:
    #     ax.plot(
    #         [loc[0], loc[0]],
    #         [loc[1], loc[1]],
    #         [loc[2] + shift_z - 5, loc[2] + shift_z + 5],
    #         **line_kw,
    #     )
    # # Label d
    # mid_x = ((loc_3[0] + loc_2[0]) / 2) - 35
    # mid_y = (loc_3[1] + loc_2[1]) / 2
    # mid_z = loc_3[2] + shift_z + 2 - 3
    # ax.text(
    #     mid_x,
    #     mid_y,
    #     mid_z,
    #     s="d",
    #     color="black",
    #     va="bottom",
    #     fontsize=LABEL_FONTSIZE,
    # )

    # Add black line: depth
    opposite_corner_x = np.max(coordinates[:, 0]) + 20
    opposite_corner_y = np.max(coordinates[:, 1]) + 20
    top_planes_z = unique_z_planes[-2:]
    ax.plot(
        [opposite_corner_x, opposite_corner_x],
        [opposite_corner_y, opposite_corner_y],
        [top_planes_z[0], top_planes_z[-1]],
        c="black",
        linestyle="-",
        zorder=10,
        linewidth=linewidth,
    )
    for loc in [top_planes_z[0], top_planes_z[-1]]:
        ax.plot(
            [opposite_corner_x - 10, opposite_corner_x + 10],
            [opposite_corner_y, opposite_corner_y],
            [loc, loc],
            **line_kw,
        )
    ax.text3D(
        opposite_corner_x + 10,
        opposite_corner_y + 10,
        top_planes_z[1] + 20,
        s=rf"${int(delta_z)}\mu m$",
        color="black",
        va="top",
        ha="right",
        fontsize=TICK_FONTSIZE,
    )

    # ax.text(
    #     opposite_corner_x - 20,
    #     opposite_corner_y - 20,
    #     unique_z_planes[0] - 25,
    #     s="~8k V1 neurons",
    #     fontsize=LABEL_FONTSIZE,
    #     color="black",
    #     va="bottom",
    #     ha="right",
    # )

    ax.set_zlim(np.min(coordinates[:, 2]) - 60, np.max(coordinates[:, 2]) + 60)

    ax.invert_zaxis()
    ax.set_axis_off()

    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        filename,
        bbox_inches=figure.bbox_inches.from_bounds(0.32, 0.85, 2.4, 1.3),
        transparent=True,
        dpi=DPI,
    )
    # plot.save_figure(figure, filename=filename, dpi=10 * DPI)


def main():
    for mouse_id in ["B"]:
        print(f"\nPlotting mouse {mouse_id}...")
        plot_planes(
            coordinates=utils.get_neuron_coordinates(mouse_id=mouse_id),
            filename=PLOT_DIR / f"mouse{mouse_id}_neuron_coordinates.svg",
        )


if __name__ == "__main__":
    main()
