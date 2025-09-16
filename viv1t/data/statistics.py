import numpy as np
import torch

from viv1t.data.constants import STATISTICS_DIR


def compute_response_precision(std: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Compute response precision to standardize response for the
    Sensorium 2023 challenge.

    Reference:
    - https://github.com/sinzlab/neuralpredictors/blob/29206ece4ed20af57f7c6e5ee614e49da50dd8d5/neuralpredictors/data/transforms.py#L375
    """
    is_tensor = torch.is_tensor(std)
    if not is_tensor:
        std = torch.from_numpy(std)
    threshold = 0.01 * torch.nanmean(std)
    idx = std > threshold
    precision = torch.ones_like(std) / threshold
    precision[idx] = 1 / std[idx]
    return precision


def load_stats(mouse_id: str) -> dict[str, dict[str, np.ndarray]]:
    """
    Load data statistics of the training set for mouse mouse_id.
    The statistics are computed using the helper script in `data/compute_stats.py`.


    Note that the Sensorium 2023 challenge abnormally used the statistics of
    the trial, instead of trial and time, to standardize the data.
    See https://www.sensorium-competition.net/#announcements 2023-05-15 Problems.

    Args:
        mouse_id: str, path to the mouse directory
    Returns:
        stats: Dict[str, Dict[str, np.ndarray]], statistics of the data
    """
    stats_dir = STATISTICS_DIR / f"mouse{mouse_id}"
    assert stats_dir.is_dir(), f"Cannot find metadata directory for Mouse {mouse_id}."
    stat_names = ["max", "mean", "median", "min", "std"]
    load = lambda a, b: np.load(stats_dir / a / f"{b}.npy")
    stats = {
        data_name: {stat_name: load(data_name, stat_name) for stat_name in stat_names}
        for data_name in ["behavior", "pupil_center", "response", "video"]
    }
    return stats
