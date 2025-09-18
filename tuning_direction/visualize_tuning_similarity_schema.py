from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox
from matplotlib.offsetbox import AuxTransformBox
from matplotlib.patches import FancyArrow

from viv1t import data
from viv1t.utils import plot

plot.set_font()

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

PAPER_WIDTH = 5.1666  # width of the paper in inches
DPI = 400

DATA_DIR = Path("../data")
PLOT_DIR = Path("figures") / "tuning_similarity_schema"

DIRECTION_COLORMAPS = {
    0: "#1d4e31",  # 0
    1: "#EA0B13",  # 45
    2: "#FEBC16",  # 90
    3: "#1D4AA3",  # 135
    # 4: "#2ea45d",  # 180
    # 5: "#ff686e",  # 225
    # 6: "#ffdf90",  # 270
    # 7: "#718dc5",  # 315
}

FORMAT = "pdf"


def plot_arrows(fig: plt.Figure, angles: list[float] | np.ndarray, colors: list[str]):
    """Plots arrows at the bottom center of the figure with corresponding colors."""
    for i, (angle, color) in enumerate(zip(angles, colors)):
        # # Create arrow
        # arrow = FancyArrow(
        #     x=0,
        #     y=0,
        #     dx=0,
        #     dy=0.07,
        #     width=0.015,
        #     head_width=0.03,
        #     head_length=0.04,
        #     length_includes_head=True,
        #     facecolor=color,
        #     edgecolor="none",
        #     alpha=1,
        # )

        # Create perpendicular line
        line = Line2D(
            xdata=[-0.018, 0.018],
            ydata=[0, 0],
            color=color,
            linewidth=2.4,
            solid_capstyle="butt",
            solid_joinstyle="miter",
        )

        # Create a transformation that includes rotation and scaling
        transform_box = AuxTransformBox(
            fig.transFigure + mtransforms.Affine2D().rotate_deg(angle + 90)
        )
        # transform_box.add_artist(arrow)
        transform_box.add_artist(line)

        # Position the arrow at the bottom center
        ab = AnnotationBbox(
            offsetbox=transform_box,
            xy=(0.35 + i * 0.1, 0.15),
            frameon=False,
            box_alignment=(0.5, 0.5),
            xycoords="figure fraction",
            pad=0,
            bboxprops=dict(edgecolor="none"),
        )
        fig.add_artist(ab)


def add_plane_legend(ax, z_planes):
    # Add a vertical line connecting the bottom left corners of the deepest and 2nd planes
    line_x = [-5.8, -5.8]
    line_y = [-5.8, -5.8]
    line_z = [z_planes[0], z_planes[2]]
    ax.plot(
        line_x,
        line_y,
        line_z,
        color="black",
        linewidth=1.2,
        clip_on=False,
    )
    ax.text(
        -6.7,
        -5.8,
        ((z_planes[0] + z_planes[1]) / 2) + 0.5,
        r"$\Delta$",
        color="black",
        fontsize=LABEL_FONTSIZE,
        ha="center",
    )

    # Add a vertical line connecting the top right corners of the top 2 planes
    # line_x_right = [5.2, 5.23]
    # line_y_right = [5.2, 5.23]
    # line_z_right = [z_planes[1], z_planes[0]]
    # ax.plot(line_x_right, line_y_right, line_z_right, color="black", linewidth=1.0)
    # ax.text(
    #    5.2,
    #    5.2,
    #    z_planes[0] - 2,
    #    r"$40\mu m$",
    #    color="black",
    #    fontsize=TICK_FONTSIZE * 0.8,
    #    ha="center",
    # )

    # Add two parallel vertical lines going up from two random positions in the top plane
    line_kw = {"color": "black", "zorder": 100, "linewidth": 1.0}
    ax.plot([1, 1], [2, 2], [z_planes[-1], z_planes[-1] + 2], **line_kw)
    ax.plot([3, 3], [4, 4], [z_planes[-1], z_planes[-1] + 2], **line_kw)
    # Add a line connecting the tops of the two parallel vertical lines
    ax.plot([1, 3], [2, 4], [z_planes[-1] + 2, z_planes[-1] + 2], **line_kw)
    ax.text(
        2.1,
        2.8,
        z_planes[-1] + 2.3,
        s=r"$\it{d}$",
        color="black",
        fontsize=LABEL_FONTSIZE,
        ha="center",
    )


def plot_planes(filename: Path):
    figure, (ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=((2 / 3) * PAPER_WIDTH, 1.6),
        gridspec_kw={
            "wspace": 0.0,
            "hspace": 0.0,
            "left": 0.0,
            "right": 1,
            "top": 1.15,
            "bottom": 0,
        },
        dpi=DPI,
        subplot_kw={"projection": "3d"},
    )
    # ax1.view_init(elev=12, azim=-50, roll=-0.5)
    ax2.view_init(elev=12, azim=-50, roll=-0.5)
    ax3.view_init(elev=12, azim=-50, roll=-0.5)

    z_planes = np.linspace(-5, 5, 5)  # Adjust the range and number of planes as needed
    # Create a meshgrid for the x and y coordinates
    x = np.linspace(-5.3, 5.3, 5)  # Adjust the range as needed
    y = np.linspace(-5.3, 5.3, 5)  # Adjust the range as needed
    xx, yy = np.meshgrid(x, y)

    dot_alpha, dot_size = 0.6, 8

    # for i, z_plane in enumerate(z_planes):
    #    zz = np.full_like(xx, z_plane)

    #    # Plot the plane borders
    #    ax1.plot(
    #        xx[0, :], yy[0, :], zz[0, :], color="black", linewidth=0.5, zorder=i
    #    )  # Top edge
    #    ax1.plot(
    #        xx[-1, :], yy[-1, :], zz[-1, :], color="black", linewidth=0.5, zorder=i
    #    )  # Bottom edge
    #    ax1.plot(
    #        xx[:, 0], yy[:, 0], zz[:, 0], color="black", linewidth=0.5, zorder=i
    #    )  # Left edge
    #    ax1.plot(
    #        xx[:, -1], yy[:, -1], zz[:, -1], color="black", linewidth=0.5, zorder=i
    #    )  # Right edge

    #    # Add random dots in the plane
    #    num_dots = 80
    #    x_dots = np.random.uniform(np.min(x), np.max(x), num_dots)
    #    y_dots = np.random.uniform(np.min(y), np.max(y), num_dots)
    #    z_dots = np.full(num_dots, z_plane)

    #    ax1.scatter(
    #        x_dots,
    #        y_dots,
    #        z_dots,
    #        color="gray",
    #        s=dot_size,
    #        zorder=i + 1,
    #        alpha=dot_alpha,
    #        edgecolors="none",
    #    )

    border_linewidth = 0.8
    # ax2.set_title("Salt and Pepper", fontsize=TITLE_FONTSIZE, y=0.9)
    for i, z_plane in enumerate(z_planes):
        zz = np.full_like(xx, z_plane)

        # Plot the plane borders
        ax2.plot(
            xx[0, :],
            yy[0, :],
            zz[0, :],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Top edge
        ax2.plot(
            xx[-1, :],
            yy[-1, :],
            zz[-1, :],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Bottom edge
        ax2.plot(
            xx[:, 0],
            yy[:, 0],
            zz[:, 0],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Left edge
        ax2.plot(
            xx[:, -1],
            yy[:, -1],
            zz[:, -1],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Right edge

        # Add random dots in the plane
        num_dots = 80
        x_dots = np.random.uniform(np.min(x), np.max(x), num_dots)
        y_dots = np.random.uniform(np.min(y), np.max(y), num_dots)
        z_dots = np.full(num_dots, z_plane)
        dot_colors = np.random.choice(list(DIRECTION_COLORMAPS.values()), num_dots)

        ax2.scatter(
            x_dots,
            y_dots,
            z_dots,
            color=dot_colors,
            s=dot_size,
            zorder=i + 1,
            alpha=dot_alpha,
            edgecolors="none",
        )

    # ax3.set_title("Mini Columns", fontsize=TITLE_FONTSIZE, y=0.9)
    column_x = np.linspace(-4, 4, 3)  # x-coordinates for the columns
    column_y = np.linspace(-4, 4, 3)  # y-coordinates for the columns
    column_x, column_y = np.meshgrid(column_x, column_y)
    column_x = column_x.flatten()
    column_y = column_y.flatten()
    column_colors = np.random.choice(list(DIRECTION_COLORMAPS.values()), len(column_x))

    for i, z_plane in enumerate(z_planes):
        zz = np.full_like(xx, z_plane)

        # Plot the plane borders
        ax3.plot(
            xx[0, :],
            yy[0, :],
            zz[0, :],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Top edge
        ax3.plot(
            xx[-1, :],
            yy[-1, :],
            zz[-1, :],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Bottom edge
        ax3.plot(
            xx[:, 0],
            yy[:, 0],
            zz[:, 0],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Left edge
        ax3.plot(
            xx[:, -1],
            yy[:, -1],
            zz[:, -1],
            color="black",
            linewidth=border_linewidth,
            zorder=i,
        )  # Right edge

        num_dots_around_column = 5
        for j, (cx, cy) in enumerate(zip(column_x, column_y)):
            x_dots = cx + np.random.uniform(-0.5, 0.5, num_dots_around_column)
            y_dots = cy + np.random.uniform(-0.5, 0.5, num_dots_around_column)
            z_dots = np.full(num_dots_around_column, z_plane)
            ax3.scatter(
                x_dots,
                y_dots,
                z_dots,
                color=column_colors[j],
                s=dot_size,
                zorder=i + 1,
                alpha=dot_alpha,
                edgecolors="none",
            )

    # Add dots connecting the columns across planes with random offsets
    for j, (cx, cy) in enumerate(zip(column_x, column_y)):
        for z_start, z_end in zip(z_planes[:-1], z_planes[1:]):
            z_dots = np.linspace(z_start, z_end, 10)
            x_dots = cx + np.random.uniform(-0.5, 0.5, len(z_dots))
            y_dots = cy + np.random.uniform(-0.5, 0.5, len(z_dots))
            ax3.scatter(
                x_dots,
                y_dots,
                z_dots,
                color=column_colors[j],
                s=dot_size,
                alpha=dot_alpha,
                edgecolors="none",
            )

    # ax1.set_axis_off()
    # ax1.set_xlim(-4.7, 4.7)
    # ax1.set_ylim(-4.7, 4.7)
    # ax1.set_zlim(-5, 5)
    # add_plane_legend(ax1, z_planes)

    ax2.set_axis_off()
    ax2.set_xlim(-5.7, 5.7)
    ax2.set_ylim(-5.7, 5.7)
    ax2.set_zlim(-5.7, 5.7)
    # add_plane_legend(ax2, z_planes)

    ax3.set_axis_off()
    ax3.set_xlim(-5.7, 5.7)
    ax3.set_ylim(-5.7, 5.7)
    ax3.set_zlim(-5.7, 5.7)
    add_plane_legend(ax3, z_planes)

    # Plot arrows at the bottom center
    plot_arrows(
        figure,
        np.array(list(data.DIRECTIONS.keys()), dtype=int),
        DIRECTION_COLORMAPS.values(),
    )

    figure.text(
        x=0.5, y=0.9, s="v.s.", ha="center", va="center", fontsize=LABEL_FONTSIZE
    )
    plot.save_figure(figure, filename=filename, layout="none", dpi=DPI)


def plot_planes_legend10(filename: Path):
    figure = plt.figure(figsize=(PAPER_WIDTH / 2, 1.2))
    ax = figure.add_subplot(111, projection="3d")
    ax.view_init(elev=12, azim=-50, roll=-0.5)

    z_planes = np.linspace(-5, 5, 10)  # Adjust the range and number of planes to 10

    # Create a meshgrid for the x and y coordinates
    x = np.linspace(-5, 5, 5)  # Adjust the range as needed
    y = np.linspace(-5, 5, 5)  # Adjust the range as needed
    xx, yy = np.meshgrid(x, y)

    for i, z_plane in enumerate(z_planes):
        zz = np.full_like(xx, z_plane)

        # Determine the color for the plane borders
        color = "black"

        # Plot the plane borders with the determined color
        ax.plot(
            xx[0, :], yy[0, :], zz[0, :], color=color, linewidth=0.5, zorder=i
        )  # Top edge
        ax.plot(
            xx[-1, :], yy[-1, :], zz[-1, :], color=color, linewidth=0.5, zorder=i
        )  # Bottom edge
        ax.plot(
            xx[:, 0], yy[:, 0], zz[:, 0], color=color, linewidth=0.5, zorder=i
        )  # Left edge
        ax.plot(
            xx[:, -1], yy[:, -1], zz[:, -1], color=color, linewidth=0.5, zorder=i
        )  # Right edge

        # Add random grey dots in the plane
        num_dots = 80
        x_dots = np.random.uniform(np.min(x), np.max(x), num_dots)
        y_dots = np.random.uniform(np.min(y), np.max(y), num_dots)
        z_dots = np.full(num_dots, z_plane)

        ax.scatter(
            x_dots,
            y_dots,
            z_dots,
            color="grey",
            s=3,
            zorder=i + 1,
            alpha=0.5,
            edgecolors="none",
        )

    line_x_right = [5.2, 5.23]
    line_y_right = [5.2, 5.23]
    line_z_right = [z_planes[1], z_planes[0]]
    ax.plot(line_x_right, line_y_right, line_z_right, color="black", linewidth=1.0)
    ax.text(
        5.2,
        5.2,
        z_planes[0] - 2,
        r"$25\mu m$",
        color="black",
        fontsize=TICK_FONTSIZE * 0.8,
        ha="center",
    )

    ax.set_axis_off()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    plot.save_figure(figure, filename=filename, dpi=10 * DPI)


def main():
    np.random.seed(5)
    plot_planes(
        filename=PLOT_DIR / f"tuning_column_schema.{FORMAT}",
    )
    plot_planes_legend10(
        filename=PLOT_DIR / f"schema_legend10.{FORMAT}",
    )


if __name__ == "__main__":
    main()
