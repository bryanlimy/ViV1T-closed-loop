import os
import random
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

sns.set_style("ticks")
plt.style.use("seaborn-v0_8-deep")

matplotlib.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams["animation.embed_limit"] = 2**64

FPS = 30
DPI = 240
FONTSIZE = 9
MAX_FRAME = 300
SKIP = 50
VIDEO_MIN, VIDEO_MAX = 0, 255

PARAMS_PAD = 2
PARAMS_LENGTH = 3

plt.rcParams.update(
    {
        "mathtext.default": "regular",
        "xtick.major.pad": PARAMS_PAD,
        "ytick.major.pad": PARAMS_PAD,
        "xtick.major.size": PARAMS_LENGTH,
        "ytick.major.size": PARAMS_LENGTH,
    }
)


JET = matplotlib.colormaps.get_cmap("jet")
GRAY = matplotlib.colormaps.get_cmap("gray")
TURBO = matplotlib.colormaps.get_cmap("turbo")
GRAY2RGB = TURBO(np.arange(256))[:, :3]

TICK_FONTSIZE = 8
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 10


def set_font():
    font_path = os.getenv(
        "MATPLOTLIB_FONT", default=Path(__file__).parent / "Lexend-Regular.ttf"
    )
    if font_path is not None and os.path.exists(font_path):
        font_manager.fontManager.addfont(path=font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams.update({"font.family": [prop.get_name(), "DejaVu Sans"]})


def remove_spines(axis: Axes):
    """remove all spines"""
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["bottom"].set_visible(False)


def remove_top_right_spines(axis: Axes):
    """Remove the ticks and spines of the top and right axis"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def set_right_label(axis: Axes, label: str, fontsize: int = None):
    """Set y-axis label on the right-hand side"""
    right_axis = axis.twinx()
    kwargs = {"rotation": 270, "va": "center", "labelpad": 3}
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    right_axis.set_ylabel(label, **kwargs)
    right_axis.set_yticks([])
    remove_top_right_spines(right_axis)


def set_xticks(
    axis: Axes,
    ticks: np.ndarray | list,
    tick_labels: np.ndarray | list = None,
    label: str = None,
    tick_fontsize: int = None,
    label_fontsize: int = None,
    linespacing: float = 1.0,
    rotation: int = None,
    label_pad: int = 2,
    color: str = "black",
    va: str = "center",
    ha: str = "center",
):

    kwargs = {"fontsize": tick_fontsize, "linespacing": linespacing}
    if rotation is not None:
        kwargs["rotation"] = rotation
        kwargs["va"] = va
        kwargs["ha"] = ha

    axis.set_xticks(ticks, labels=tick_labels, **kwargs)
    if label is not None:
        axis.set_xlabel(
            label,
            fontsize=label_fontsize,
            color=color,
            linespacing=linespacing,
            labelpad=label_pad,
        )


def set_yticks(
    axis: Axes,
    ticks: np.ndarray | list,
    tick_labels: np.ndarray | list = None,
    label: str = "",
    tick_fontsize: int = None,
    label_fontsize: int = None,
    linespacing: float = 1.0,
    rotation: int = None,
    label_pad: int = 2,
    ha: str | None = None,
    va: str | None = None,
):
    kwargs = {"fontsize": tick_fontsize}
    if rotation is not None:
        kwargs["rotation"] = rotation
        if va is None:
            va = "center"
    if ha is not None:
        kwargs["horizontalalignment"] = ha
    if va is not None:
        kwargs["verticalalignment"] = va
    axis.set_yticks(ticks, labels=tick_labels, **kwargs)
    if label:
        axis.set_ylabel(
            label,
            fontsize=label_fontsize,
            linespacing=linespacing,
            labelpad=label_pad,
        )


def set_ticks_params(
    axis: Axes,
    length: float = PARAMS_LENGTH,
    pad: float = PARAMS_PAD,
    minor_length: float | None = 0.85 * PARAMS_LENGTH,
    color: str = "black",
    linewidth: float = 1.2,
):
    axis.tick_params(
        axis="both",
        which="major",
        length=length,
        pad=pad,
        colors=color,
        width=linewidth,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        length=length if minor_length is None else minor_length,
        pad=pad,
        colors=color,
        width=linewidth,
    )
    for ax in ["top", "bottom", "left", "right"]:
        axis.spines[ax].set_linewidth(linewidth)
        axis.spines[ax].set_color(color)


def get_p_value_asterisk(p_value: float) -> str:
    if p_value <= 0.001:
        text = "***"
    elif p_value <= 0.01:
        text = "**"
    elif p_value <= 0.05:
        text = "*"
    else:
        text = "n.s."
    return text


def add_p_value(
    ax: Axes,
    x0: float | np.ndarray,
    x1: float | np.ndarray,
    y: float | np.ndarray,
    p_value: float,
    fontsize: float = TICK_FONTSIZE,
    tick_length: float | np.ndarray = None,
    tick_linewidth: float = 1.0,
    text_pad: float = None,
):
    if p_value <= 0.001:
        text = "***"
    elif p_value <= 0.01:
        text = "**"
    elif p_value <= 0.05:
        text = "*"
    else:
        text = "n.s."

    ns = text == "n.s."

    ax.plot(
        [x0, x0, x1, x1],
        [y - tick_length, y, y, y - tick_length],
        color="black",
        linewidth=tick_linewidth,
        clip_on=False,
        solid_capstyle="butt",
        solid_joinstyle="miter",
    )
    ax.text(
        x=((x1 - x0) / 2) + x0,
        y=y + (1.5 * text_pad if ns else text_pad),
        s=text,
        ha="center",
        va="top",
        fontsize=fontsize - 2 if ns else fontsize,
        transform=ax.transData,
    )


def save_figure(
    figure: plt.Figure,
    filename: Path,
    dpi: int = 120,
    pad_inches: float = 0.01,
    layout: str = "constrained",
    close: bool = True,
):
    filename.parent.mkdir(parents=True, exist_ok=True)
    figure.set_layout_engine(layout=layout)
    figure.savefig(
        filename,
        dpi=dpi,
        transparent=True,
        pad_inches=pad_inches,
    )
    if close:
        plt.close(figure)


def get_color(model_name: str):
    match model_name.lower():
        case "recorded":
            color = "black"
        case "ln":
            color = "lightskyblue"
        case "fcnn":
            color = "dodgerblue"
        case "dwiseneuro" | "dn":
            color = "slateblue"
        case "viv1t" | "predicted" | "viv1t_causal":
            color = "limegreen"
        case "vivit":
            color = "indigo"
        case "random" | "chance" | "shuffle":
            color = "dimgrey"
        case _:
            # return a random color
            color = random.choice(list(matplotlib.colors.cnames.keys()))
            print(f"Unknown model name {model_name}. Set color to {color}.")
    return color


def get_zorder(model_name: str) -> int:
    match model_name.lower():
        case "recorded":
            zorder = 5
        case "ln":
            zorder = 1
        case "fcnn":
            zorder = 2
        case "dwiseneuro":
            zorder = 3
        case "vivit" | "viv1t" | "predicted" | "viv1t_causal":
            zorder = 6
        case "random" | "chance" | "shuffle":
            zorder = 1
        case _:
            zorder = 1
            print(f"Unknown model name {model_name}. Set zorder to {zorder}.")
    return zorder


def animate_attention_map(
    sample: dict[str, np.ndarray],
    filename: Path = None,
    alpha: float = 0.4,
    max_frame: int = 300,
    spatial_title: str = 'spatial "attention"',
    temporal_title: str = 'temporal "attention"',
    fps: int = 30,
    dpi: int = 240,
):
    """Animate attention map overlay over video frames"""
    turbo_color = TURBO(np.arange(256))[:, :3]

    # colormap
    mappable = cm.ScalarMappable(cmap=TURBO)
    mappable.set_clim(0, 1)

    figure_width, figure_height = 4.6, 2.3
    figure = plt.figure(
        figsize=(figure_width, figure_height), dpi=dpi, facecolor="white"
    )
    _, t, h, w = sample["video"].shape
    frames = np.arange(start=max_frame - t, stop=max_frame, step=1)
    frame_xticks = np.arange(frames[0], frames[-1] + 50, 50)
    temporal_attention_color = mappable.cmap(sample["temporal_attention"])

    behavior, pupil_center = sample["behavior"], sample["pupil_center"]

    get_width = lambda height: height * (w / h) * (figure_height / figure_width)
    # spatial attention
    spatial_height = 0.68
    ax1 = figure.add_axes(rect=(0.03, 0.25, get_width(spatial_height), spatial_height))
    # temporal attention
    ax2 = figure.add_axes(rect=(0.03, 0.10, get_width(spatial_height), 0.08))
    pos1, pos2 = ax1.get_position(), ax2.get_position()
    # inputs
    input_height = 0.35
    ax3 = figure.add_axes(
        rect=(0.68, pos1.y1 - input_height, get_width(input_height), input_height)
    )

    # add colorbar
    cbar_width, cbar_height = 0.008, 0.1
    cbar_ax = figure.add_axes(
        rect=(
            pos1.x1 + 0.01,
            pos1.y0,  # pos2.y0 + ((pos1.y1 - pos2.y0) / 2) - (cbar_height / 2),
            cbar_width,
            cbar_height,
        )
    )
    plt.colorbar(mappable, cax=cbar_ax, shrink=0.5)
    cbar_yticks = np.linspace(0, 1, 2, dtype=int)
    set_yticks(
        axis=cbar_ax,
        ticks=cbar_yticks,
        tick_labels=cbar_yticks,
        tick_fontsize=TICK_FONTSIZE,
    )
    set_ticks_params(cbar_ax, length=1.5, pad=1)

    text_kwargs = {
        "y": -0.14,
        "va": "center",
        "fontsize": TICK_FONTSIZE - 1,
        "transform": ax3.transAxes,
        "linespacing": 0.85,
    }

    def animate(i: int):
        for ax in [ax1, ax2, ax3]:
            ax.cla()

        frame = frames[i]

        # plot spatial attention map overlay on frame
        image = sample["video"][0, i]
        heatmap = sample["spatial_attention"][i]
        heatmap = turbo_color[np.uint8(255.0 * heatmap)] * 255.0
        heatmap = alpha * heatmap + (1 - alpha) * image[..., None]
        ax1.imshow(heatmap.astype(np.uint8), cmap=mappable.cmap, interpolation=None)
        ax1.set_title(spatial_title, pad=3, fontsize=TICK_FONTSIZE)
        ax1.grid(linewidth=0)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # plot temporal attention
        ax2.scatter(
            frames[:i],
            sample["temporal_attention"][:i],
            s=8,
            clip_on=False,
            edgecolor="none",
            c=temporal_attention_color[:i],
        )
        ax2.set_xlim(frames[0], frames[-1])
        ax2.set_ylim(0, 1.0)
        set_yticks(
            ax2,
            ticks=[0, 1],
            tick_labels=[0, 1],
            tick_fontsize=TICK_FONTSIZE,
        )
        ax2.set_title(temporal_title, pad=2, fontsize=TICK_FONTSIZE)
        ax2.set_xlim(frames[0], frames[-1])
        set_xticks(
            ax2,
            ticks=frame_xticks,
            tick_labels=frame_xticks.astype(int),
            label="movie frame",
            tick_fontsize=TICK_FONTSIZE,
            label_fontsize=TICK_FONTSIZE,
            label_pad=-1,
        )
        ax2.grid(visible=False, which="major")
        sns.despine(ax=ax2)
        set_ticks_params(ax2, length=2)

        # plot inputs
        ax3.imshow(image / 255.0, cmap="gray", interpolation=None, vmin=0, vmax=1)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.text(
            x=0.0,
            y=1.0,
            s=f"Frame {frame:03d}",
            ha="left",
            va="bottom",
            color="black",
            alpha=0.8,
            fontsize=TICK_FONTSIZE - 1,
            transform=ax3.transAxes,
        )
        if behavior is not None:
            ax3.text(
                x=0,
                s=f"{behavior[0, i]:.1e}\npupil size",
                ha="left",
                **text_kwargs,
            )
            ax3.text(
                x=0.45,
                s=f"{behavior[1, i]:.1e}\nspeed",
                ha="center",
                **text_kwargs,
            )
        if pupil_center is not None:
            ax3.text(
                x=1.0,
                s=f"({pupil_center[0, i]:.0f}, {pupil_center[1, i]:.0f})\npupil center",
                ha="right",
                **text_kwargs,
            )

    anim = FuncAnimation(figure, animate, frames=len(frames), interval=int(1000 / fps))

    if filename is not None:
        filename.parent.mkdir(parents=True, exist_ok=True)
        anim.save(filename, fps=fps, dpi=dpi, savefig_kwargs={"pad_inches": 0})

    return anim


def animate_stimulus(
    video: np.ndarray | torch.Tensor,
    response: np.ndarray | torch.Tensor,
    filename: Path,
    neuron: int | None = None,
    loss: torch.Tensor | np.ndarray | None = None,
    ds_max: float | torch.Tensor | np.ndarray = None,
    ds_mean: float | torch.Tensor | np.ndarray = None,
    max_response: float | torch.Tensor | np.ndarray = None,
    sum_response: float | torch.Tensor | np.ndarray = None,
    presentation_mask: np.ndarray | None = None,
    skip: int = SKIP,
):
    assert len(video.shape) == 4
    t, h, w = video.shape[1], video.shape[2], video.shape[3]

    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    if torch.is_tensor(response):
        response = response.detach().cpu().numpy()
    if presentation_mask is not None and torch.is_tensor(presentation_mask):
        presentation_mask = presentation_mask.cpu().numpy()

    f_w, f_h = 3.6, 2.3
    figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
    get_width = lambda height: height * (w / h) * (f_h / f_w)
    spatial_height = 0.68
    ax1 = figure.add_axes(rect=(0.12, 0.25, get_width(spatial_height), spatial_height))
    ax2 = figure.add_axes(rect=(0.12, 0.10, get_width(spatial_height), 0.08))

    x_ticks = np.array([0, t], dtype=int)
    min_value, max_value = 0, np.max(response)
    if ds_max is not None:
        max_value = max(max_value, ds_max)
    # max_value = ceil(max_value)

    if neuron is None:
        title = rf"Average $\Delta F/F$"
    else:
        title = rf"N{neuron:04d} $\Delta F/F$"
    if max_response is not None and sum_response is not None:
        title += f" (max: {max_response:.1f}, sum: {sum_response:.0f})"

    x = np.arange(t)

    imshow = ax1.imshow(
        np.random.rand(h, w),
        cmap="gray",
        aspect="equal",
        vmin=VIDEO_MIN,
        vmax=VIDEO_MAX,
    )
    pos = ax1.get_position()
    text = ax1.text(
        x=0,
        y=pos.y1 + 0.11,
        s="",
        ha="left",
        va="center",
        fontsize=FONTSIZE,
        transform=ax1.transAxes,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    line = ax2.plot(
        [],
        [],
        linewidth=2,
        color="forestgreen",
        zorder=1,
        clip_on=False,
    )[0]
    ax2.set_xlim(x_ticks[0], x_ticks[-1])
    set_xticks(axis=ax2, ticks=x_ticks, tick_labels=x_ticks, tick_fontsize=FONTSIZE)
    ax2.xaxis.set_minor_locator(MultipleLocator(10))
    y_ticks = np.array([min_value, max_value], dtype=np.float32)
    ax2.set_ylim(y_ticks[0], y_ticks[-1])
    set_yticks(
        axis=ax2,
        ticks=y_ticks,
        tick_labels=np.round(y_ticks, decimals=2),
        tick_fontsize=FONTSIZE,
    )
    ax2.text(
        x=x_ticks[0],
        y=max_value,
        s=title,
        ha="left",
        va="bottom",
        fontsize=FONTSIZE,
        transform=ax2.transData,
    )
    if ds_max is not None:
        ax2.axhline(
            y=ds_max,
            color="orangered",
            linewidth=1.2,
            linestyle="dashed",
            zorder=0,
            clip_on=False,
        )
    if ds_mean is not None:
        ax2.axhline(
            y=ds_mean,
            color="dodgerblue",
            linewidth=1.2,
            linestyle="dotted",
            zorder=0,
            clip_on=False,
        )
    if presentation_mask is not None:
        ax2.fill_between(
            x,
            y1=0,
            y2=max_value,
            where=presentation_mask,
            facecolor="#e0e0e0",
            edgecolor="none",
            zorder=-1,
        )
    set_ticks_params(axis=ax2, minor_length=None)
    sns.despine(ax=ax2, trim=True)

    title = f" (best loss: {loss:.2f})" if loss else ""

    def animate(frame: int):
        artists = [imshow]
        imshow.set_data(video[0, frame, :, :])
        text.set_text(f"Frame: {frame :03d}" + title)
        if frame >= skip:
            line.set_data(x[skip : frame + 1], response[: frame - skip + 1])
            artists.append(line)
        return artists

    anim = FuncAnimation(
        figure, func=animate, frames=t, interval=int(1000 / FPS), blit=True
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0})
    plt.close(figure)


def get_scaled_fonts(sizes, figure_width_latex_inches, figure_width_matplotlib_inches):
    scaling_factor = figure_width_latex_inches / figure_width_matplotlib_inches
    return [size / scaling_factor for size in sizes]
