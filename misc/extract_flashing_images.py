"""
Estimate the orientation, wavelength, and frequency of all unique
drifting gabor stimulus and save the results to OUTPUT_DIR
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from viv1t import data

DATA_DIR = Path("../data/sensorium")
META_DIR = data.METADATA_DIR / "ood_features" / "flashing_images"
PLOT_DIR = Path("figures") / "flashing_images"

BLOCK_SIZE = 15  # number of frames for each flashing image presentation
FPS = 30


# min and max pixel values
MIN, MAX = 0, 255
DPI = 240
LABEL_FONTSIZE = 12


def load_video(mouse_id: str, trial_id: int) -> np.ndarray:
    video = np.load(
        DATA_DIR / data.MOUSE_IDS[mouse_id] / "data" / "videos" / f"{trial_id}.npy"
    )
    num_frames = min(data.utils.find_nan(video[0, 0, :]), data.MAX_FRAME)
    video = rearrange(video[..., :num_frames], "h w t -> () t h w")
    video = video.astype(np.float32)
    video = np.round(video, decimals=0)
    return video


def normalize(x: np.ndarray, x_min: float = -1, x_max: float = 1):
    """Normalize the pixel range to [MIN, MAX]"""
    return (x - x_min) * (MAX - MIN) / (x_max - x_min) + MIN


def animate(video: np.ndarray, image_ids: np.ndarray, filename: Path):
    h, w = video.shape[2], video.shape[3]
    f_h, f_w = (h / 16) + (7 / h), w / 16
    a_w = 0.99
    a_h = (h / 16) / f_h

    figure = plt.figure(figsize=(f_w, f_h), dpi=DPI, facecolor="white")
    ax = figure.add_axes(rect=((1 - a_w) / 2, (1 - a_w) / 2, a_w, a_h))

    imshow = ax.imshow(
        np.random.rand(h, w), cmap="gray", aspect="equal", vmin=MIN, vmax=MAX
    )
    pos = ax.get_position()
    text1 = ax.text(
        x=0,
        y=pos.y1 + 0.12,
        s="",
        ha="left",
        va="center",
        fontsize=LABEL_FONTSIZE,
        transform=ax.transAxes,
    )
    text2 = ax.text(
        x=pos.x1,
        y=pos.y1 + 0.12,
        s="",
        ha="right",
        va="center",
        fontsize=LABEL_FONTSIZE,
        transform=ax.transAxes,
    )
    ax.grid(linewidth=0)
    ax.set_xticks([])
    ax.set_yticks([])

    def _animate(frame: int):
        artists = [imshow]
        imshow.set_data(video[0, frame, :, :])
        text1.set_text(f"Image ID {image_ids[frame]}")
        text2.set_text(f"Frame {frame:03d}")
        return artists

    anim = FuncAnimation(
        figure,
        func=_animate,
        frames=video.shape[1],
        interval=int(1000 / FPS),
        blit=True,
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    anim.save(filename, fps=FPS, dpi=DPI, savefig_kwargs={"pad_inches": 0.5})
    plt.close(figure)


def save_parameters(output_dir: Path, trial_ids: np.ndarray, image_ids: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    for trial_id in trial_ids:
        np.save(output_dir / f"{trial_id}.npy", image_ids)


def is_blank_frame(frame: np.ndarray) -> bool:
    frame = frame.flatten()
    return np.all(frame[0] == frame)


def extract_sequences(video: np.ndarray) -> list[np.ndarray]:
    """
    Group video into sequences of grey screen or image presentation

    For some reason, not all pixels have the same value during image
    presentation window, the first and last frames often have lower contrast
    than the rest of the pixels. Therefore, we have to manually check, frame by
    frame, when it is grey screen vs image.
    """
    assert video.ndim == 3
    num_frames = video.shape[0]
    is_blank = np.all(video == video[:, 0, 0][:, None, None], axis=(1, 2))
    frames = np.arange(num_frames, dtype=int)
    frames = np.split(frames, np.nonzero(np.diff(is_blank))[0] + 1)
    sequences = [video[frames[i]] for i in range(len(frames))]
    assert np.array_equal(video, np.concat(sequences, axis=0))
    return sequences


def extract_image_ids(
    video: np.ndarray,
    unique_images: list[np.ndarray],
    mouse_id: str,
    trial_id: int,
    plot: bool = False,
) -> (np.ndarray, list[np.ndarray]):
    assert video.shape[0] == 1
    video = video[0]
    sequences = extract_sequences(video)
    image_ids = []
    for sequence in sequences:
        frame = sequence[len(sequence) // 2]  # get the middle frame
        if is_blank_frame(frame):
            image_ids.extend([-1] * len(sequence))
        else:
            unique = True
            for image_id, unique_image in enumerate(unique_images):
                if np.allclose(unique_image, frame):
                    image_ids.extend([image_id] * len(sequence))
                    unique = False
                    break
            if unique:
                image_ids.extend([len(unique_images)] * len(sequence))
                unique_images.append(frame)
    image_ids = np.array(image_ids, dtype=int)
    print(f"Found {len(np.unique(image_ids)) - 1} unique images in trial {trial_id}")
    if plot:
        animate(
            video=rearrange(video, "t h w -> () t h w"),
            image_ids=image_ids,
            filename=PLOT_DIR / f"mouse{mouse_id}" / f"trial{trial_id:03d}.mp4",
        )
    return image_ids, unique_images


def estimate_mouse(mouse_id: str):
    # get flashing image trials
    stimulus_ids = data.get_stimulus_ids(mouse_id)
    trial_ids = np.where(stimulus_ids == 5)[0]
    if not trial_ids.size:
        return
    # store image and unique image ID
    unique_images = []
    video_ids = data.get_video_ids(mouse_id)
    for video_id in tqdm(np.unique(video_ids[trial_ids]), desc=f"Mouse {mouse_id}"):
        # get the first trial with the video_id
        trial_id = np.where(video_ids == video_id)[0][0]
        video = load_video(mouse_id=mouse_id, trial_id=trial_id)
        image_ids, unique_images = extract_image_ids(
            video=video,
            unique_images=unique_images,
            mouse_id=mouse_id,
            trial_id=trial_id,
            plot=True,
        )
        save_parameters(
            output_dir=META_DIR / f"mouse{mouse_id}",
            trial_ids=np.where(video_ids == video_id)[0],
            image_ids=image_ids,
        )
    print(f"Found {len(unique_images)} unique images in mouse {mouse_id}")


def main():
    for mouse_id in data.SENSORIUM_OLD:
        estimate_mouse(mouse_id)
    print(f"Saved flashing image parameters to {META_DIR}.")


if __name__ == "__main__":
    main()
