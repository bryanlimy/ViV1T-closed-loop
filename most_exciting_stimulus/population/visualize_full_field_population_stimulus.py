from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib import axes
from matplotlib.figure import SubFigure
from matplotlib.gridspec import GridSpec
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import sem
from tqdm import tqdm

from viv1t import data
from viv1t.utils import h5
from viv1t.utils import plot

TICK_FONTSIZE = 7
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "svg"
PAPER_WIDTH = 5.1666  # width of the paper in inches

plot.set_font()

DATA_DIR = Path("../../data/sensorium")
PLOT_DIR = Path("figures") / "population_full_field_stimulus"


PATTERN_SIZE = 15
PADDING = 10  # number of frames to include before and after stimulus presentation

MIN, MAX = 0, 255.0
GREY_COLOR = (MAX - MIN) // 2

GRATING_COLOR = "black"
STATIC_NATURAL_COLOR = "midnightblue"
DYNAMIC_NATURAL_COLOR = "dodgerblue"
STATIC_GENERATED_COLOR = "orangered"
DYNAMIC_GENERATED_COLOR = "crimson"


def load_max_response(filename: Path, mouse_id: str) -> np.ndarray:
    max_responses = np.load(filename, allow_pickle=True)
    return max_responses[mouse_id]


def load_grating_result(output_dir: Path, mouse_id: str) -> pd.DataFrame | None:
    filename = output_dir / "most_exciting_stimulus" / "population" / "gratings.pkl"
    df = pd.read_pickle(filename)
    if mouse_id not in df.mouse.unique():
        return None
    # get most-exciting grating from the recorded response
    recorded_df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
    max_response = recorded_df.response.max()
    recorded_df = recorded_df[recorded_df.response == max_response]
    # get predicted response to the same directional grating
    predicted_df = df[
        (df.mouse == mouse_id)
        & (df.model == "predicted")
        & (df.direction == recorded_df.iloc[0].direction)
        & (df.wavelength == recorded_df.iloc[0].wavelength)
        & (df.frequency == recorded_df.iloc[0].frequency)
    ]
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    df["stimulus_type"] = "grating"
    df.drop(columns=["direction"], inplace=True)
    return df


def load_natural_image_result(output_dir: Path, mouse_id: str) -> pd.DataFrame | None:
    filename = output_dir / "most_exciting_stimulus" / "population" / "images.pkl"
    df = pd.read_pickle(filename)
    if mouse_id not in df.mouse.unique():
        return None
    # get most-exciting grating from the recorded response
    recorded_df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
    max_response = recorded_df.response.max()
    recorded_df = recorded_df[recorded_df.response == max_response]
    # get predicted response to the same image
    predicted_df = df[
        (df.mouse == mouse_id)
        & (df.model == "predicted")
        & (df.image == recorded_df.iloc[0].image)
    ]
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    df["stimulus_type"] = "image"
    df.drop(columns=["image"], inplace=True)
    return df


def load_video(mouse_id: str, trial_id: int) -> np.ndarray:
    video = np.load(
        DATA_DIR / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    video = video[..., : data.MAX_FRAME]
    video = np.round(video, decimals=0)
    video = rearrange(video, "h w t -> () t h w")
    return video


def load_natural_video_result(output_dir: Path, mouse_id: str) -> pd.DataFrame | None:
    filename = output_dir / "most_exciting_stimulus" / "population" / "videos.pkl"
    df = pd.read_pickle(filename)
    if mouse_id not in df.mouse.unique():
        return None
    # get most-exciting grating from the recorded response
    recorded_df = df[(df.mouse == mouse_id) & (df.model == "recorded")]
    max_response = recorded_df.response.max()
    recorded_df = recorded_df[recorded_df.response == max_response]
    # get predicted response to the same video clip
    predicted_df = df[
        (df.mouse == mouse_id)
        & (df.model == "predicted")
        & (df.video_id == recorded_df.iloc[0].video_id)
        & (df.frame_id == recorded_df.iloc[0].frame_id)
    ]
    del df
    # load video clip
    video_id = int(recorded_df.iloc[0].video_id)
    frame_id = int(recorded_df.iloc[0].frame_id)
    video_ids = data.get_video_ids(mouse_id=mouse_id)
    trial_ids = np.where(video_ids == video_id)[0]
    video = load_video(mouse_id=mouse_id, trial_id=trial_ids[0])
    video = video[0, frame_id : frame_id + PATTERN_SIZE, :, :]
    recorded_df["stimulus"] = [video.tolist()]
    predicted_df["stimulus"] = [video.tolist()]
    df = pd.concat([recorded_df, predicted_df], ignore_index=True)
    df.drop(columns=["video_id", "frame_id"], inplace=True)
    df["stimulus_type"] = "video"
    return df


def load_generated_result(
    output_dir: Path, experiment_name: str, stimulus_type: str, mouse_id: str
) -> pd.DataFrame:
    assert stimulus_type in ("mei", "mev")
    max_response = load_max_response(filename=Path("ViV1T.npz"), mouse_id=mouse_id)
    filename = (
        output_dir
        / "most_exciting_stimulus"
        / "population"
        / "generated"
        / "full_field"
        / ("static" if stimulus_type == "mei" else "dynamic")
        / experiment_name
        / f"mouse{mouse_id}"
        / "most_exciting"
        / "ckpt.pt"
    )
    ckpt = torch.load(filename, map_location="cpu")
    # compute population average
    responses = ckpt["response"].numpy()
    blank_size2 = int(np.floor((data.MAX_FRAME - PATTERN_SIZE) / 2))
    responses = responses[
        :, -(blank_size2 + PATTERN_SIZE + PADDING) : -(blank_size2 - PADDING)
    ]
    # responses = responses / max_response[:, None]
    # average response over population
    responses = np.mean(responses, axis=0)
    response = np.sum(responses[PADDING : PADDING + PATTERN_SIZE])
    video = ckpt["video"].detach().numpy()
    video = video[0, -(blank_size2 + PATTERN_SIZE) : -blank_size2]
    df = pd.DataFrame(
        {
            "mouse": [mouse_id],
            "model": ["predicted"],
            "response": [response],
            "raw_response": [responses],
            "stimulus": [video.tolist()],
            "stimulus_type": [stimulus_type],
        }
    )
    return df


def most_exciting_window(response: np.ndarray) -> tuple[int, int]:
    """
    For stimulus that has presentation longer than PATTERN_SIZE, return the
    beginning and ending index of the PATTERN_SIZE window that has the  most
    exciting response
    """
    frame_ids = np.arange(response.shape[0])
    response = sliding_window_view(response, window_shape=PATTERN_SIZE, axis=0)
    response = np.sum(response, axis=1)
    i = np.argmax(response)
    frame_ids = sliding_window_view(frame_ids, window_shape=PATTERN_SIZE, axis=0)
    return (frame_ids[i][0], frame_ids[i][-1])


def plot_stimulus(
    ax: axes.Axes,
    df: pd.DataFrame,
    figure: plt.Figure,
    imshow_kws: dict,
    dynamic: bool,
    title: str = None,
    y_label: str = None,
):
    assert len(df) == 1
    stimulus = np.array(df.iloc[0].stimulus, dtype=np.float32)
    if not dynamic:
        stimulus = stimulus[stimulus.shape[0] // 2]
    ax.imshow(stimulus[1] if dynamic else stimulus, **imshow_kws)
    ax.set_xticks([])
    ax.set_yticks([])
    plot.set_ticks_params(ax, linewidth=1.2)
    if dynamic:
        # plot stack of frames for dynamic stimuli
        pos = ax.get_position()
        x_offset = 0.06 * pos.width
        y_offset = 0.07 * pos.height
        for i in range(2):
            axis = figure.add_axes(
                (
                    pos.x0 + (i + 1) * x_offset,  # left
                    pos.y0 + (i + 1) * y_offset,  # bottom
                    pos.width,  # width
                    pos.height,  # height
                )
            )
            axis.imshow(stimulus[(i + 1) * 5], **imshow_kws)
            axis.set_xticks([])
            axis.set_yticks([])
            plot.set_ticks_params(axis, linewidth=1.2)
            axis.set_zorder(-i - 1)
            if i == 0:
                title_ax = axis
        # add an arrow connecting the top left corners of the axes with padding
        ax.annotate(
            "",
            xy=(
                pos.x0 + 2 * x_offset + 0.02 * pos.width,
                pos.y1 + 3 * y_offset + 0.1 * pos.height,
            ),  # arrow end
            xytext=(
                pos.x0 - 0.02 * pos.width,
                pos.y1 + 0.04 * pos.height,
            ),  # arrow start
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.25,head_width=0.15",
                color="black",
                lw=1.2,
            ),
            xycoords="subfigure fraction",
            textcoords="subfigure fraction",
        )
    if title is not None:
        h, w = stimulus.shape[1:] if dynamic else stimulus.shape
        ax.text(
            x=0.55 * w if dynamic else 0.5 * w,
            y=-0.28 * h,
            s=title,
            fontsize=LABEL_FONTSIZE,
            linespacing=0.85,
            transform=ax.transData,
            ha="center",
            va="bottom",
        )
    if y_label is not None:
        ax.set_ylabel(
            y_label,
            fontsize=LABEL_FONTSIZE,
            labelpad=1.5,
            linespacing=0.9,
            y=0.05,
            va="bottom",
            ha="left",
        )


def plot_response(
    ax: axes.Axes,
    df: pd.DataFrame,
    x_label: str = None,
    y_label: str = None,
    color: str = "black",
    max_value: float = None,
    show_tick_label: bool = False,
):
    assert len(df) == 1
    response = np.array(df.iloc[0].raw_response, dtype=np.float32)
    x = np.arange(response.shape[0])
    x_ticks = np.arange(0, response.shape[0] + 1, 5)
    match df.iloc[0].stimulus_type:
        case "grating":
            presentation_mask = np.ones_like(x)
            x0, x1 = most_exciting_window(response)
        case "image" | "mei":
            presentation_mask = np.zeros(response.shape[0])
            presentation_mask[PADDING : -(PADDING - 1)] = 1
            x0 = PADDING
            x1 = PADDING + PATTERN_SIZE
        case "video":
            presentation_mask = np.ones_like(x)
            x0, x1 = most_exciting_window(response)
        case "mev":
            presentation_mask = np.ones_like(x)
            x0 = PADDING
            x1 = PADDING + PATTERN_SIZE
        case _:
            raise ValueError(f"Unknown stimulus type: {df.iloc[0].stimulus_type}")
    if show_tick_label:
        x_tick_labels = x_ticks.astype(int)
    else:
        x_tick_labels = [""] * len(x_ticks)
    if max_value is None:
        max_value = np.max(response)
    ax.plot(
        x,
        response,
        linewidth=2,
        color=color,
        zorder=1,
        clip_on=False,
    )
    ax.set_xlim(0, len(x))
    plot.set_xticks(
        ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        label=x_label,
        label_pad=0,
    )
    # change different styling of y ticks based on its range
    exp = None
    if max_value >= 1:
        max_value = int(np.ceil(max_value))
        max_tick = str(max_value)
    elif 0.1 <= max_value < 1:
        max_value = 0.1 * np.ceil(max_value * 10)
        max_tick = f"{max_value:.01f}"
    else:
        exp = int(np.ceil(np.log10(max_value)))
        max_value = 10**exp
        max_tick = r"$10^{" + str(exp) + r"}$"
    ax.set_ylim(0, max_value)
    y_ticks = np.array([0, max_value])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=[0, max_tick],
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=TICK_FONTSIZE,
        ha="right",
    )
    if y_label is not None:
        ax.text(
            x=0,
            y=0.95 * max_value,
            s=y_label,
            va="bottom",
            ha="left",
            transform=ax.transData,
            fontsize=TICK_FONTSIZE,
        )
    # show presentation period
    ax.fill_between(
        x,
        y1=0,
        y2=max_value,
        where=presentation_mask,
        facecolor="#e0e0e0",
        edgecolor="none",
        zorder=-1,
    )
    # show sliding window to compute response sum
    axvline_kw = {
        "color": "black",
        "linewidth": 1.2,
        "linestyle": (0, (0.75, 0.75)),
        "clip_on": False,
    }
    ax.axvline(x0, **axvline_kw)
    ax.axvline(x1, **axvline_kw)

    sns.despine(ax=ax)
    ax.set_axisbelow(False)
    plot.set_ticks_params(ax, pad=1.5)
    if exp is not None:
        tick = ax.get_yaxis().get_major_ticks()[-1]
        tick.set_pad(-0.5)


def plot_mouse(
    df: pd.DataFrame,
    mouse_id: str,
    response_type: str,
    model_name: str,
    filename: Path,
) -> None:
    imshow_kws = {"aspect": "equal", "cmap": "gray", "vmin": 0, "vmax": 255}

    figure = plt.figure(figsize=((2 / 3) * PAPER_WIDTH, 2.3), dpi=DPI)

    column_width = 0.24
    stimulus_height = 0.3

    # grating
    top = 0.75
    left = 0.04
    gs1 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs1[0]),
        df=df[(df.stimulus_type == "grating") & (df.model == "recorded")],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=True,
        title="Most-exciting\ngrating",
        y_label=None,
    )
    gs2 = GridSpec(
        nrows=2,
        ncols=1,
        left=left,
        right=left + column_width,
        top=0.43,
        bottom=0.22,
        wspace=0,
        hspace=1.5,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs2[0]),
        df=df[(df.stimulus_type == "grating") & (df.model == "recorded")],
        y_label=r"Recorded $\Delta$F/F",
        color="black",
    )
    plot_response(
        ax=figure.add_subplot(gs2[1]),
        df=df[(df.stimulus_type == "grating") & (df.model == "predicted")],
        x_label="Time (frame)",
        y_label=r"Predicted $\Delta$F/F",
        color=plot.get_color(model_name),
        show_tick_label=True,
    )

    # most-exciting natural image
    left = 0.37
    top = 0.90
    gs3 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs3[0]),
        df=df[(df.stimulus_type == "image") & (df.model == "recorded")],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=False,
        title="Most-exciting\nnatural image",
        y_label=None,
    )
    gs4 = GridSpec(
        nrows=2,
        ncols=1,
        left=left,
        right=left + column_width,
        top=0.63,
        bottom=0.48,
        wspace=0,
        hspace=0.7,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs4[0]),
        df=df[(df.stimulus_type == "image") & (df.model == "recorded")],
        y_label="",
        color="black",
    )
    plot_response(
        ax=figure.add_subplot(gs4[1]),
        df=df[(df.stimulus_type == "image") & (df.model == "predicted")],
        x_label="",
        y_label="",
        color=plot.get_color(model_name),
    )

    # most-exciting generated image
    top = 0.345
    gs5 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs5[0]),
        df=df[(df.stimulus_type == "mei") & (df.model == "predicted")],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=False,
        title="Most-exciting\ngenerated image",
        y_label=None,
    )
    gs6 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=0.075,
        bottom=0.025,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs6[0]),
        df=df[(df.stimulus_type == "mei") & (df.model == "predicted")],
        x_label="",
        y_label="",
        color=plot.get_color(model_name),
    )

    # most-exciting natural video
    left = 0.71
    top = 0.90
    gs7 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs7[0]),
        df=df[(df.stimulus_type == "video") & (df.model == "recorded")],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=True,
        title="Most-exciting\nnatural video",
        y_label=None,
    )
    gs8 = GridSpec(
        nrows=2,
        ncols=1,
        left=left,
        right=left + column_width,
        top=0.63,
        bottom=0.48,
        wspace=0,
        hspace=0.7,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs8[0]),
        df=df[(df.stimulus_type == "video") & (df.model == "recorded")],
        y_label="",
        color="black",
    )
    plot_response(
        ax=figure.add_subplot(gs8[1]),
        df=df[(df.stimulus_type == "video") & (df.model == "predicted")],
        x_label="",
        y_label="",
        color=plot.get_color(model_name),
    )

    # most-exciting generated video
    top = 0.345
    gs9 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=top,
        bottom=top - stimulus_height,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_stimulus(
        ax=figure.add_subplot(gs9[0]),
        df=df[(df.stimulus_type == "mev") & (df.model == "predicted")],
        figure=figure,
        imshow_kws=imshow_kws,
        dynamic=True,
        title="Most-exciting\ngenerated video",
        y_label=None,
    )
    gs10 = GridSpec(
        nrows=1,
        ncols=1,
        left=left,
        right=left + column_width,
        top=0.075,
        bottom=0.025,
        wspace=0,
        hspace=0,
        figure=figure,
    )
    plot_response(
        ax=figure.add_subplot(gs10[0]),
        df=df[(df.stimulus_type == "mev") & (df.model == "predicted")],
        x_label="",
        y_label="",
        color=plot.get_color(model_name),
    )

    plot.save_figure(figure, filename=filename, dpi=DPI, layout="none")


def process_model(model_name: str, output_dir: Path):
    experiment_name = "003_cutoff_population"
    for mouse_id in ["B", "C"]:  # only mouse B and C have flashing images and gratings
        print(f"Plot mouse {mouse_id}...")
        grating_df = load_grating_result(output_dir=output_dir, mouse_id=mouse_id)
        if grating_df is None:
            continue
        natural_image_df = load_natural_image_result(
            output_dir=output_dir, mouse_id=mouse_id
        )
        if natural_image_df is None:
            continue
        natural_video_df = load_natural_video_result(
            output_dir=output_dir, mouse_id=mouse_id
        )
        mei_df = load_generated_result(
            output_dir=output_dir,
            experiment_name=experiment_name,
            stimulus_type="mei",
            mouse_id=mouse_id,
        )
        mev_df = load_generated_result(
            output_dir=output_dir,
            experiment_name=experiment_name,
            stimulus_type="mev",
            mouse_id=mouse_id,
        )
        df = pd.concat(
            [grating_df, natural_image_df, natural_video_df, mei_df, mev_df],
            ignore_index=True,
        )
        response_type = "most_exciting"
        filename = (
            PLOT_DIR / f"full_field_{response_type}_mouse{mouse_id}_stimulus.{FORMAT}"
        )
        plot_mouse(
            df=df,
            mouse_id=mouse_id,
            response_type=response_type,
            model_name=model_name,
            filename=filename,
        )
        print(f"Saved mouse {mouse_id} plot to {filename}.")

    print(f"Saved result to {PLOT_DIR}.")


def main():
    models = {
        "ViV1T": Path("../../runs/vivit/204_causal_viv1t"),
    }

    for model_name, output_dir in models.items():
        process_model(model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    main()
