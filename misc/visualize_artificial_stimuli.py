from pathlib import Path

import matplotlib.pyplot as plt

from viv1t import data
from viv1t.utils import stimulus

VIDEO_H, VIDEO_W = 1080, 1920
MIN, MAX = 0, 255
GREY_COLOR = (MAX - MIN) // 2
FPS = 30
FORMAT = "svg"
output_path = Path("figures") / "artificial_stimuli"
output_path.mkdir(parents=True, exist_ok=True)

mouse_id = "A"
monitor_info = data.MONITOR_INFO[mouse_id]

stimulus_size = 40
mask = stimulus.create_circular_mask(
    stimulus_size=stimulus_size,
    center=(VIDEO_W // 2, VIDEO_H // 2),
    pixel_width=VIDEO_W,
    pixel_height=VIDEO_H,
    monitor_width=monitor_info["width"],
    monitor_height=monitor_info["height"],
    monitor_distance=monitor_info["distance"],
    num_frames=30,
    to_tensor=False,
)
left = (VIDEO_W // 2) - (VIDEO_H // 2)
right = (VIDEO_W // 2) + (VIDEO_H // 2)


# CLASSIC
center_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=10,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)

center_grating[~mask] = GREY_COLOR
for frame in [0, 5, 10]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(
        center_grating[0][frame][:, left:right],
        cmap="gray",
        vmin=MIN,
        vmax=MAX,
    )
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(10)
    plt.savefig(
        output_path / f"center_{frame}.{FORMAT}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

# INVERSE
center_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=10,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
center_grating[mask] = GREY_COLOR
for frame in [0, 5, 10]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(
        center_grating[0][frame][:, left:right],
        cmap="gray",
        vmin=MIN,
        vmax=MAX,
    )
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(10)
    plt.savefig(
        output_path / f"inverse_{frame}.{FORMAT}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

# ISO
center_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=10,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
surround_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=10,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
center_grating[mask] = surround_grating[mask]
for frame in [0, 5, 10]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(
        center_grating[0][frame][:, left:right],
        cmap="gray",
        vmin=MIN,
        vmax=MAX,
    )
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(10)
    plt.savefig(
        output_path / f"iso_{frame}.{FORMAT}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


# CROSS
center_grating = stimulus.create_full_field_grating(
    direction=0,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=10,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
surround_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=10,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
center_grating[mask] = surround_grating[mask]
for frame in [0, 5, 10]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(
        center_grating[0][frame][:, left:right],
        cmap="gray",
        vmin=MIN,
        vmax=MAX,
    )
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(10)
    plt.savefig(
        output_path / f"cross_{frame}.{FORMAT}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

# SHIFT
center_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=0,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
surround_grating = stimulus.create_full_field_grating(
    direction=90,
    cpd=0.04,
    cpf=2 / FPS,
    num_frames=30,
    height=VIDEO_H,
    width=VIDEO_W,
    phase=180,
    contrast=1,
    fps=FPS,
    to_tensor=False,
)
center_grating[mask] = surround_grating[mask]
for frame in [0, 5, 10]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(
        center_grating[0][frame][:, left:right],
        cmap="gray",
        vmin=MIN,
        vmax=MAX,
    )
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(10)
    plt.savefig(
        output_path / f"shift_{frame}.{FORMAT}",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()
