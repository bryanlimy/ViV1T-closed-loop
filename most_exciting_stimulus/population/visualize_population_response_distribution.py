from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import rearrange
from matplotlib.ticker import MultipleLocator
from numpy import unravel_index
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import sem
from tqdm import tqdm

from viv1t import data
from viv1t.data import get_gabor_parameters
from viv1t.utils import h5
from viv1t.utils import plot
from viv1t.utils import utils

DATA_DIR = Path("../../data/sensorium")
PLOT_DIR = Path("figures") / "response_distribution"


TICK_FONTSIZE = 7
LABEL_FONTSIZE = 7
TITLE_FONTSIZE = 8

DPI = 400
FORMAT = "jpg"
PAPER_WIDTH = 5.1666  # width of the paper in inches
plot.set_font()

WINDOW_SIZE = 15  # slide window size to match flashing image presentation window


def load_grating_response(
    output_dir: Path,
    model: str,
    mouse_id: str,
    response_type: str,
    rng: np.random.Generator,
    pattern: int | None = None,
    window: int | None = None,
):
    block_size = 25  # drifting Gabor presentation window
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot find {filename}.")
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 4)[0]
    if not trial_ids.size:
        return None, None, None
    # responses shape (trial, neuron, time)
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]
    # separate response to each direction of BLOCK_SIZE
    responses = rearrange(
        responses,
        "trial neuron (block frame) -> (trial block) frame neuron",
        frame=block_size,
    )
    # get direction parameters for each frame
    parameters = np.array(
        [get_gabor_parameters(mouse_id, trial_id=trial_id) for trial_id in trial_ids],
        dtype=np.float32,
    )
    parameters = parameters[:, -num_frames:]
    parameters = rearrange(
        parameters,
        "trial (block frame) param -> (trial block) frame param",
        frame=block_size,
    )
    assert np.all(parameters == parameters[:, 0][:, None])
    parameters = parameters[:, 0, :]
    assert parameters.shape[0] == responses.shape[0]
    # find the minimum number of presentations in each Gabor parameter configuration
    unique_parameters, counts = np.unique(parameters, return_counts=True, axis=0)
    min_repeats = np.min(counts)
    # randomly select min_samples for each direction
    responses_ = np.zeros(
        (len(unique_parameters), min_repeats, block_size, num_neurons), dtype=np.float32
    )
    for i, parameter in enumerate(unique_parameters):
        index = np.where(np.all(parameters == parameter, axis=1))[0]
        index = rng.choice(index, size=min_repeats, replace=False)
        responses_[i] = responses[index]
    responses = responses_.copy()
    del responses_, unique_parameters, stimulus_ids, trial_ids, counts, min_repeats
    # average response over repeats
    responses = np.mean(responses, axis=1)
    responses = rearrange(responses, "pattern frame neuron -> pattern neuron frame")
    ############### find most-exciting/most-inhibiting response ###############
    responses = sliding_window_view(responses, window_shape=WINDOW_SIZE, axis=-1)
    # sum response over sliding window
    responses = np.sum(responses, axis=-1)
    if pattern is not None and window is not None:
        response = responses[pattern, :, window]
    else:
        # sum response over population
        sum_responses = np.sum(responses, axis=1)
        # select the most-exciting/most-inhibiting sliding window and pattern
        match response_type:
            case "most_exciting":
                pattern, window = unravel_index(
                    np.argmax(sum_responses), sum_responses.shape
                )
            case "most_inhibiting":
                pattern, window = unravel_index(
                    np.argmin(sum_responses), sum_responses.shape
                )
            case _:
                raise RuntimeError(f"Unknown response_type {response_type}")
        response = responses[pattern, :, window]
    df = pd.DataFrame(
        {
            "model": model,
            "stimulus_type": "grating",
            "mouse": mouse_id,
            "response": response,
        }
    )
    return df, pattern, window


def load_natural_image_response(
    output_dir: Path,
    model: str,
    mouse_id: str,
    response_type: str,
    natural_image_id: int | None = None,
):
    blank_size = 10
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 5)[0]
    if not trial_ids.size:
        return None, None
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    if not filename.is_file():
        raise FileNotFoundError(f"Cannot find {filename}.")
    # responses shape (trial, neuron, time)
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]
    image_ids = np.array(
        [
            data.get_flashing_image_parameters(mouse_id, trial_id=trial_id)
            for trial_id in trial_ids
        ],
        dtype=int,
    )
    image_ids = image_ids[:, -num_frames:]
    responses_ = {}
    for image_id in np.unique(image_ids):
        if image_id == -1:
            continue  # ignore blank screen
        for i in range(image_ids.shape[0]):
            if image_id not in image_ids[i]:
                continue
            indexes = np.where(image_ids[i] == image_id)[0]
            # requires presentation to be at least WINDOW_SIZE presentation window
            if len(indexes) < WINDOW_SIZE:
                continue
            start, end = indexes[0], indexes[-1] + 1
            # for some reason, the first and last frame of the presentation window
            # are at a much lower contrast
            if len(indexes) - WINDOW_SIZE == 1:
                start += 1
            elif len(indexes) - WINDOW_SIZE == 2:
                start += 1
                end -= 1
            # require presentation to have at least BLANK_SIZE blank screen
            # before and after the presentation
            assert end - start == WINDOW_SIZE
            start -= blank_size
            end += blank_size
            if start <= 0 or end >= num_frames:
                continue
            if image_id not in responses_:
                responses_[image_id] = []
            responses_[image_id].append(responses[i, :, start:end])
            del indexes, start, end
    # remove images that have less than 5 repeats
    image_ids, responses = [], []
    for image_id, response in responses_.items():
        if len(response) >= 5:
            response = np.mean(np.stack(response), axis=0)
            image_ids.append(image_id)
            responses.append(response)
        else:
            print(f"Image ID {image_id} only has {len(response)} repeats")
    responses = np.stack(responses)
    responses = rearrange(responses, "image neuron frame -> image frame neuron")
    del responses_, stimulus_ids, trial_ids
    ############### find most-exciting/most-inhibiting response ###############
    # select response during presentation window
    responses = responses[:, blank_size:-blank_size, :]
    # sum response over presentation window
    responses = np.sum(responses, axis=1)
    if natural_image_id is not None:
        assert natural_image_id in image_ids
        index = image_ids.index(natural_image_id)
        response = responses[index]
        image_id = natural_image_id
    else:
        # sum response over population
        sum_responses = np.sum(responses, axis=1)
        match response_type:
            case "most_exciting":
                index = np.argmax(sum_responses)
            case "most_inhibiting":
                index = np.argmin(sum_responses)
            case _:
                raise RuntimeError(f"Unknown response_type {response_type}")
        response = responses[index, :]
        image_id = image_ids[index]
    df = pd.DataFrame(
        {
            "model": model,
            "stimulus_type": "natural_image",
            "mouse": mouse_id,
            "response": response,
        }
    )
    return df, image_id


def load_natural_video_response(
    output_dir: Path,
    model: str,
    mouse_id: str,
    response_type: str,
    rng: np.random.Generator,
    natural_video_id: int | None = None,
    natural_frame_id: int | None = None,
):
    # load live_main and final_main test sets which are all natural movies
    tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
    trial_ids = np.where((tiers == "live_main") | (tiers == "final_main"))[0]
    video_ids = data.get_video_ids(mouse_id=mouse_id)[trial_ids]
    # responses shape (trial, neuron, time)
    filename = output_dir / "responses" / f"mouse{mouse_id}.h5"
    responses = np.stack(h5.get(filename, trial_ids=trial_ids))
    num_neurons, num_frames = responses.shape[1:]
    # group video by video IDs
    video_ids_ = np.unique(video_ids)
    responses_, min_repeat = [], np.inf
    for video_id in video_ids_:
        responses_.append(responses[video_ids == video_id])
        min_repeat = min(min_repeat, np.count_nonzero(video_ids == video_id))
    if min_repeat < 5:
        print(
            f"Only {min_repeat} repeats for mouse {mouse_id} from {output_dir} natural video"
        )
    # randomly select repeat so that all unique video IDs have the same number
    # of repeats
    responses = np.stack(
        [r[rng.choice(len(r), size=min_repeat, replace=False)] for r in responses_]
    )
    video_ids = video_ids_.copy()
    del responses_, video_ids_, trial_ids, tiers, min_repeat, video_id
    # average response over repeats
    responses = np.mean(responses, axis=1)
    ############### find most-exciting/most-inhibiting response ###############
    frame_ids = np.arange(num_frames) + (data.MAX_FRAME - num_frames)
    responses = sliding_window_view(responses, window_shape=WINDOW_SIZE, axis=-1)
    frame_ids = sliding_window_view(frame_ids, window_shape=WINDOW_SIZE, axis=-1)
    # sum response over sliding window
    responses = np.sum(responses, axis=-1)
    frame_ids = frame_ids[:, 0]
    if natural_video_id is not None and natural_frame_id is not None:
        assert natural_video_id in video_ids and natural_frame_id in frame_ids
        video_i = video_ids.tolist().index(natural_video_id)
        frame_i = frame_ids.tolist().index(natural_frame_id)
        response = responses[video_i, :, frame_i]
        video_id = natural_video_id
        frame_id = natural_frame_id
    else:
        # sum response over population
        sum_responses = np.sum(responses, axis=1)
        # select the most-exciting/most-inhibiting sliding window and pattern
        match response_type:
            case "most_exciting":
                video_i, frame_i = unravel_index(
                    np.argmax(sum_responses), sum_responses.shape
                )
            case "most_inhibiting":
                video_i, frame_i = unravel_index(
                    np.argmin(sum_responses), sum_responses.shape
                )
            case _:
                raise RuntimeError(f"Unknown response_type {response_type}")
        response = responses[video_i, :, frame_i]
        video_id = video_ids[video_i]
        frame_id = frame_ids[frame_i]
    df = pd.DataFrame(
        {
            "model": model,
            "stimulus_type": "natural_video",
            "mouse": mouse_id,
            "response": response,
        }
    )
    return df, video_id, frame_id


def load_generated_response(
    output_dir: Path,
    mouse_id: str,
    experiment_name: str,
    stimulus_type: str,
    response_type: str,
) -> pd.DataFrame:
    assert stimulus_type in ("generated_image", "generated_video")
    assert response_type in ("most_exciting", "most_inhibiting")
    filename = (
        output_dir
        / "most_exciting_stimulus"
        / "population"
        / "generated"
        / "full_field"
        / ("static" if stimulus_type == "generated_image" else "dynamic")
        / experiment_name
        / f"mouse{mouse_id}"
        / response_type
        / "ckpt.pt"
    )
    ckpt = torch.load(filename, map_location="cpu")
    # compute population average
    responses = ckpt["response"].numpy()
    blank_size2 = int(np.floor((data.MAX_FRAME - WINDOW_SIZE) / 2))
    responses = responses[:, -(blank_size2 + WINDOW_SIZE) : -(blank_size2)]
    # sum response over presentation window
    response = np.sum(responses, axis=1)
    df = pd.DataFrame(
        {
            "model": "predicted",
            "stimulus_type": stimulus_type,
            "mouse": mouse_id,
            "response": response,
        }
    )
    return df


def get_responses(
    model_name: str,
    output_dir: Path,
    experiment_name: str,
    response_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = []
    for mouse_id in tqdm(data.SENSORIUM_OLD, desc=f"{model_name} {response_type}"):
        grating_response, grating_pattern, grating_window = load_grating_response(
            output_dir=DATA_DIR,
            model="recorded",
            mouse_id=mouse_id,
            response_type=response_type,
            rng=rng,
        )
        if grating_response is not None:
            df.append(grating_response)
        grating_response, _, _ = load_grating_response(
            output_dir=output_dir,
            model="predicted",
            mouse_id=mouse_id,
            response_type=response_type,
            rng=rng,
            pattern=grating_pattern,
            window=grating_window,
        )
        if grating_response is not None:
            df.append(grating_response)
        del grating_response, grating_pattern, grating_window, _
        natural_image_response, natural_image_id = load_natural_image_response(
            output_dir=DATA_DIR,
            model="recorded",
            mouse_id=mouse_id,
            response_type=response_type,
        )
        if natural_image_response is not None:
            df.append(natural_image_response)
        natural_image_response, _ = load_natural_image_response(
            output_dir=output_dir,
            model="predicted",
            mouse_id=mouse_id,
            response_type=response_type,
            natural_image_id=natural_image_id,
        )
        if natural_image_response is not None:
            df.append(natural_image_response)
        del natural_image_response, natural_image_id, _
        natural_video_response, natural_video_id, natural_frame_id = (
            load_natural_video_response(
                output_dir=DATA_DIR,
                model="recorded",
                mouse_id=mouse_id,
                response_type=response_type,
                rng=rng,
            )
        )
        df.append(natural_video_response)
        natural_video_response, _, _ = load_natural_video_response(
            output_dir=output_dir,
            model="predicted",
            mouse_id=mouse_id,
            response_type=response_type,
            rng=rng,
            natural_video_id=natural_video_id,
            natural_frame_id=natural_frame_id,
        )
        df.append(natural_video_response)
        del natural_video_response, natural_video_id, natural_frame_id, _
        generated_image_response = load_generated_response(
            output_dir=output_dir,
            mouse_id=mouse_id,
            experiment_name=experiment_name,
            stimulus_type="generated_image",
            response_type=response_type,
        )
        df.append(generated_image_response)
        generated_video_response = load_generated_response(
            output_dir=output_dir,
            mouse_id=mouse_id,
            experiment_name=experiment_name,
            stimulus_type="generated_video",
            response_type=response_type,
        )
        df.append(generated_video_response)
        del generated_image_response, generated_video_response
    df = pd.concat(df, ignore_index=True)
    df.insert(loc=1, column="response_type", value=response_type)
    return df


def plot_distribution(df: pd.DataFrame, response_type: str, filename: Path) -> None:
    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(0.5 * PAPER_WIDTH, 1.5),
        dpi=DPI,
    )

    y_min = df[(df.response >= 0)].response.min()
    exp_min = int(np.floor(np.log10(y_min)))
    y1 = 10**exp_min
    exp_max = int(max(np.ceil(np.log10(df.response.max())), 0))
    y2 = 10**exp_max
    exp = np.linspace(exp_min, exp_max, exp_max - exp_min + 1)[::-1]
    y_ticks = 10**exp

    sns.violinplot(
        data=df,
        x="stimulus_type",
        y="response",
        hue="model",
        width=0.9,
        split=True,
        inner="quart",
        palette={"recorded": "grey", "predicted": "limegreen"},
        linewidth=1,
        linecolor="black",
        order=[
            "grating",
            "natural_image",
            "natural_video",
            "generated_image",
            "generated_video",
        ],
        log_scale=True,
        ax=ax,
    )
    sns.move_legend(
        ax,
        loc="lower right",
        bbox_to_anchor=(4.5, 10 ** (exp_min + 0.25)),
        bbox_transform=ax.transData,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.3,
        handlelength=0.7,
        labelspacing=0.08,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        labels=["Recorded", "Predicted"],
        fontsize=TICK_FONTSIZE,
    )
    x_tick_labels = (
        ["MEG", "MENI", "MENV", "MEGI", "MEGV"]
        if "most_exciting" in filename.stem
        else ["MIG", "MINI", "MINV", "MIGI", "MIGV"]
    )
    plot.set_xticks(
        axis=ax,
        ticks=np.arange(5),
        tick_labels=x_tick_labels,
        label=f"{response_type.replace('_', '-').capitalize()} stimulus",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
    )

    plot.set_yticks(
        axis=ax,
        ticks=y_ticks,
        tick_labels=[int(exp[i]) if i % 2 == 0 else "" for i in range(len(exp))],
        label="Sum response (log)",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=0,
        linespacing=0.9,
    )
    ax.text(x=-0.7, y=10 ** (exp_min - 1.5), s=r"$10^y$", fontsize=TICK_FONTSIZE)
    ax.set_ylim(y1, y2)
    plot.set_ticks_params(ax, length=3, pad=2, minor_length=2)
    sns.despine(ax=ax)
    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def plot_response(
    df: pd.DataFrame,
    model_name: str,
    rng: np.random.Generator,
    filename: Path,
):
    responses = []
    for stimulus_type in [
        "grating",
        "natural_image",
        "natural_video",
        "generated_image",
        "generated_video",
    ]:
        y_true = df[(df.model == "recorded") & (df.stimulus_type == stimulus_type)]

        if not y_true.empty:
            # y_true = y_true[y_true.response >= 1e-3]
            responses.append(y_true.groupby(by="mouse").response.apply("mean").values)
        y_pred = df[(df.model == "predicted") & (df.stimulus_type == stimulus_type)]
        if not y_pred.empty:
            # y_pred = y_pred[y_pred.response >= 1e-3]
            responses.append(y_pred.groupby(by="mouse").response.apply("mean").values)

    # statistical_tests(responses=responses)

    figure, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=((1 / 3) * PAPER_WIDTH, 2.3),
        dpi=DPI,
    )

    linewidth = 1.2

    x_ticks = [1, 2, 3, 4, 5]
    positions = [0.8, 1.2, 1.8, 2.2, 2.8, 3.2, 4, 5]

    color = plot.get_color(model_name)
    colors = [
        "black",
        color,
        "black",
        color,
        "black",
        color,
        color,
        color,
    ]
    height = [np.mean(r) for r in responses]
    error = [sem(r) for r in responses]
    linestyle = "-"
    bar_width = 0.3
    ax.bar(
        x=positions,
        height=height,
        yerr=error,
        width=bar_width,
        facecolor="none",
        edgecolor=colors,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=1,
        zorder=1,
        error_kw={"linewidth": 1.0, "zorder": 1},
        # clip_on=False,
    )
    max_value = max([np.max(r) for r in responses])
    max_value = np.ceil(max_value / 10) * 10
    max_value = max(max_value, 150)

    # add_p_value(
    #     ax=ax,
    #     response1=responses[0],
    #     response2=responses[1],
    #     position1=positions[0],
    #     position2=positions[1],
    #     max_value=max_value,
    # )
    # add_p_value(
    #     ax=ax,
    #     response1=responses[2],
    #     response2=responses[3],
    #     position1=positions[2],
    #     position2=positions[3],
    #     max_value=max_value,
    # )
    # add_p_value(
    #     ax=ax,
    #     response1=responses[4],
    #     response2=responses[5],
    #     position1=positions[4],
    #     position2=positions[5],
    #     max_value=max_value,
    # )

    scatter_kw = {
        "s": 30,
        "marker": ".",
        "alpha": 0.8,
        "zorder": 0,
        "facecolors": "none",
        "clip_on": False,
    }
    for i, (position, response) in enumerate(zip(positions, responses)):
        x = rng.normal(position, 0.0, size=len(response))
        match i:
            case 0:
                label = "Recorded"
            case 1:
                label = "Predicted"
            case _:
                label = ""
        ax.scatter(x, response, edgecolors=colors[i], **scatter_kw, label=label)

    # for text in legend.texts:
    #     text.set_y(-0.2)
    # ax.text(
    #     x=positions[0],
    #     y=0.86 * max_value,
    #     s=r"$N_{mice}=$" + str(df.mouse.nunique()),
    #     fontsize=TICK_FONTSIZE,
    #     va="top",
    #     ha="left",
    #     transform=ax.transData,
    # )

    xlim = [positions[0] - 0.3, positions[-1] + 0.2]
    ax.set_xlim(*xlim)
    x_tick_labels = (
        ["MEG", "MENI", "MENV", "MEGI", "MEGV"]
        if "most_exciting" in filename.stem
        else ["MIG", "MINI", "MINV", "MIGI", "MIGV"]
    )
    plot.set_xticks(
        axis=ax,
        ticks=x_ticks,
        tick_labels=x_tick_labels,
        tick_fontsize=TICK_FONTSIZE,
        linespacing=0.85,
        # rotation=0,
        # va="top",
    )

    y_ticks = np.array([0, max_value], dtype=int)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    plot.set_yticks(
        ax,
        ticks=y_ticks,
        tick_labels=y_ticks,
        label="Sum ΔF/F",
        tick_fontsize=TICK_FONTSIZE,
        label_fontsize=LABEL_FONTSIZE,
        label_pad=-6,
        rotation=90,
    )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(positions[0], max_value),
        ncols=1,
        fontsize=TICK_FONTSIZE,
        frameon=False,
        title="",
        alignment="left",
        handletextpad=0.2,
        handlelength=0.7,
        labelspacing=0.05,
        columnspacing=0,
        borderpad=0,
        borderaxespad=0,
        bbox_transform=ax.transData,
    )
    for lh in legend.legend_handles:
        lh.set_alpha(1)

    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plot.set_ticks_params(ax, pad=1)
    ax.tick_params(axis="y", length=3, pad=0, labelsize=TICK_FONTSIZE)
    # ax.tick_params(axis="y", which="minor", length=2.2)
    # ax.tick_params(axis="x", length=3, pad=1, labelsize=TICK_FONTSIZE)
    sns.despine(ax=ax)

    plot.save_figure(figure=figure, filename=filename, dpi=DPI)


def process_model(model_name: str, output_dir: Path, response_type: str):
    rng = np.random.default_rng(seed=1234)
    experiment_name = "003_cutoff_population"
    # df = get_responses(
    #     model_name=model_name,
    #     output_dir=output_dir,
    #     experiment_name=experiment_name,
    #     response_type=response_type,
    #     rng=rng,
    # )
    # df.to_pickle(f"{response_type}_response_distribution.pkl")
    df = pd.read_pickle(f"{response_type}_response_distribution.pkl")
    plot_distribution(
        df=df,
        response_type=response_type,
        filename=PLOT_DIR / f"{response_type}_response_distribution.{FORMAT}",
    )
    plot_response(
        df=df,
        model_name=model_name,
        rng=rng,
        filename=PLOT_DIR / f"{response_type}_response.{FORMAT}",
    )


def main():
    models = {
        "ViV1T": Path("../../runs/vivit/204_causal_viv1t"),
    }
    for response_type in ["most_exciting", "most_inhibiting"]:
        for model_name, output_dir in models.items():
            process_model(
                model_name=model_name,
                output_dir=output_dir,
                response_type=response_type,
            )


if __name__ == "__main__":
    main()
