from typing import Literal

import numpy as np
import torch
from einops import rearrange

REDUCTION = Literal["sum", "mean"]
EPS = torch.finfo(torch.float32).eps
TENSOR = np.ndarray | torch.Tensor | list[np.ndarray] | list[torch.Tensor]


def sse(
    y_true: torch.Tensor, y_pred: torch.Tensor, reduction: REDUCTION = "mean"
) -> torch.Tensor:
    """sum squared error over frames and neurons"""
    loss = torch.sum(torch.square(y_true - y_pred), dim=(1, 2))
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def poisson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float | torch.Tensor = 1e-8,
    reduction: REDUCTION = "sum",
) -> torch.Tensor:
    """poisson loss over frames and neurons"""
    y_pred, y_true = y_pred + eps, y_true + eps
    loss = torch.sum(y_pred - y_true * torch.log(y_pred), dim=(1, 2))
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def correlation(
    y1: torch.Tensor | np.ndarray,
    y2: torch.Tensor | np.ndarray,
    dim: None | int | tuple[int] = -1,
    eps: torch.Tensor | float = 1e-8,
) -> torch.Tensor | np.ndarray:
    assert type(y1) == type(y2)
    match y1:
        case torch.Tensor():
            if dim is None:
                dim = tuple(range(y1.dim()))
            y1 = (y1 - y1.mean(dim=dim, keepdim=True)) / (
                y1.std(dim=dim, unbiased=False, keepdim=True) + eps
            )
            y2 = (y2 - y2.mean(dim=dim, keepdim=True)) / (
                y2.std(dim=dim, unbiased=False, keepdim=True) + eps
            )
            corr = torch.mean(y1 * y2, dim=dim)
        case np.ndarray():
            y1 = (y1 - y1.mean(axis=dim, keepdims=True)) / (
                y1.std(axis=dim, ddof=0, keepdims=True) + eps
            )
            y2 = (y2 - y2.mean(axis=dim, keepdims=True)) / (
                y2.std(axis=dim, ddof=0, keepdims=True) + eps
            )
            corr = np.mean(y1 * y2, axis=dim)
        case _:
            raise NotImplementedError(f"Unsupported type {type(y1)}")
    return corr


def single_trial_correlation(
    y_true: torch.Tensor | list[torch.Tensor],
    y_pred: torch.Tensor | list[torch.Tensor],
    per_neuron: bool = False,
) -> torch.Tensor:
    """
    Compute signal trial correlation and return average value across trials and neurons

    In the Sensorium 2023 challenge, the trial and time dimension are
    flattened before computing the correlation, i.e. (B, N, T) -> (B * T, N).
    https://github.com/ecker-lab/sensorium_2023/blob/c71a451e610ff1d0dacef20ba30812d045b1685c/sensorium/utility/scores.py#L87

    Here, we instead compute the single trial correlation over the time dimension
    for each trial and neuron, then compute the average over trials.

    Args:
        y_true: recorded responses in (B, N, T)
        y_pred: predicted responses in (B, N, T)
        per_neuron: return per_neuron correlation if True
    """
    assert type(y_true) == type(y_pred) and type(y_true[0]) == type(y_pred[0])
    match y_true:
        case torch.Tensor():
            assert y_true.shape == y_pred.shape and y_true.dim() == 3
            corr = correlation(y1=y_true, y2=y_pred, dim=2)
        case list():
            corr = torch.stack(
                [
                    correlation(y1=y_true[i], y2=y_pred[i], dim=1)
                    for i in range(len(y_true))
                ]
            )
        case _:
            raise NotImplementedError(f"Unsupported type {type(y_true)}")
    return torch.mean(corr, dim=0) if per_neuron else torch.mean(corr)


def challenge_correlation(
    y_true: torch.Tensor | list[torch.Tensor],
    y_pred: torch.Tensor | list[torch.Tensor],
    per_neuron: bool = False,
) -> torch.Tensor:
    """
    Compute the single trial correlation used in the Sensorium 2023 challenge

    In the Sensorium 2023 challenge, the trial and time dimension are
    flattened before computing the correlation, i.e. (B, N, T) -> (B * T, N).
    https://github.com/ecker-lab/sensorium_2023/blob/c71a451e610ff1d0dacef20ba30812d045b1685c/sensorium/utility/scores.py#L87

    Args:
        y_true: recorded responses in (B, N, T)
        y_pred: predicted responses in (B, N, T)
        per_neuron: return per_neuron correlation if True
    """
    assert type(y_true) == type(y_pred) and type(y_true[0]) == type(y_pred[0])
    match y_true:
        case torch.Tensor():
            y_true = rearrange(y_true, "b n t -> (b t) n")
            y_pred = rearrange(y_pred, "b n t -> (b t) n")
        case list():
            y_true = torch.vstack([rearrange(y, "n t -> t n") for y in y_true])
            y_pred = torch.vstack([rearrange(y, "n t -> t n") for y in y_pred])
        case _:
            raise NotImplementedError(f"Unsupported type {type(y_true)}")
    corr = correlation(y1=y_true, y2=y_pred, dim=0)
    return corr if per_neuron else torch.mean(corr)


def group_repeated_trials(
    y_true: torch.Tensor | list[torch.Tensor],
    y_pred: torch.Tensor | list[torch.Tensor],
    video_ids: torch.Tensor,
) -> (list[torch.Tensor], list[torch.Tensor]):
    """
    Separate trials into groups based on video_ids

    Returns:
        repeat_true: list of repeated responses (n_unique_video, n_repeats, n_neurons, n_frames)
        repeat_pred: list of repeated predictions (n_unique_video, n_repeats, n_neurons, n_frames)
    """
    unique_ids = torch.unique(video_ids)
    if len(unique_ids) == len(video_ids):
        print(f"Warning: all {len(video_ids)} trials are unique")
        return None, None
    repeat_true, repeat_pred = [], []
    for unique_id in unique_ids:
        idx = torch.where(video_ids == unique_id)[0]
        if len(idx) < 2:  # ignore stimulus with less than 2 repeats
            continue
        match y_true:
            case torch.Tensor():
                group_true = y_true[idx]
                group_pred = y_pred[idx]
            case list():
                group_true = torch.stack([y_true[i] for i in idx])
                group_pred = torch.stack([y_pred[i] for i in idx])
            case _:
                raise NotImplementedError(f"Unsupported type {type(y_true)}")
        repeat_true.append(group_true)
        repeat_pred.append(group_pred)
    return repeat_true, repeat_pred


def correlation_to_average(
    y_true: torch.Tensor | list[torch.Tensor],
    y_pred: torch.Tensor | list[torch.Tensor],
    video_ids: torch.Tensor,
    per_neuron: bool = False,
) -> torch.Tensor:
    repeat_true, repeat_pred = group_repeated_trials(y_true, y_pred, video_ids)
    if repeat_true is None or len(repeat_true) == 0:
        return 0.0
    assert len(repeat_true) == len(repeat_pred)
    num_videos, num_neurons = len(repeat_true), repeat_true[0].shape[1]
    corr = torch.full((num_videos, num_neurons), fill_value=np.nan, dtype=torch.float32)
    for i in range(num_videos):
        corr[i] = correlation(
            y1=torch.mean(repeat_true[i], dim=0),
            y2=torch.mean(repeat_pred[i], dim=0),
            dim=1,
        )
    corr = torch.mean(corr, dim=0)  # average over unique videos
    return corr if per_neuron else torch.mean(corr)


def get_cc_abs(mean_true: torch.Tensor, mean_pred: torch.Tensor) -> torch.Tensor:
    assert mean_true.shape == mean_pred.shape
    num_neurons = mean_true.shape[0]
    pairs = rearrange(torch.stack([mean_true, mean_pred]), "b n t -> n b t")
    cov = torch.stack([torch.cov(pairs[n])[0, 1] for n in range(num_neurons)])
    var_true = torch.var(mean_true, dim=1)
    var_pred = torch.var(mean_pred, dim=1)
    cc_abs = cov / (torch.sqrt(var_true * var_pred) + 1e-8)
    return cc_abs


def get_cc_norm(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    assert y_true.shape == y_pred.shape
    # response in format (num. repeats, num. neurons, num. frames)
    num_repeats, num_neurons, num_frames = y_true.shape
    mean_true = torch.mean(y_true, dim=0)
    mean_pred = torch.mean(y_pred, dim=0)
    cov = torch.sum(
        (mean_true - torch.mean(mean_true, dim=1, keepdim=True))
        * (mean_pred - torch.mean(mean_pred, dim=1, keepdim=True)),
        dim=1,
    ) / (num_frames - 1)
    sp = (
        torch.var(torch.sum(y_true, dim=0), dim=1)
        - torch.sum(torch.var(y_true, dim=2), dim=0)
    ) / (num_repeats * (num_repeats - 1))
    cc_norm = cov / (torch.sqrt(torch.var(mean_pred, dim=1) * sp) + 1e-8)
    invalid = sp <= 1e-4
    cc_norm[invalid] = torch.nan
    return cc_norm


def normalized_correlation(
    y_true: torch.Tensor | list[torch.Tensor],
    y_pred: torch.Tensor | list[torch.Tensor],
    video_ids: torch.Tensor,
    per_neuron: bool = False,
) -> torch.Tensor:
    """
    Normalized correlation from Schoppe et al. 2016 and Wang et al. 2023
    Normalize average correlation over repeated trials by the upper bound of
    achievable performance given neural variability.
    - Schoppe et al. 2016 https://www.frontiersin.org/articles/10.3389/fncom.2016.00010/full
    - Wang et al. 2023 https://www.biorxiv.org/content/10.1101/2023.03.21.533548v2
    """
    repeat_true, repeat_pred = group_repeated_trials(y_true, y_pred, video_ids)
    if repeat_true is None or len(repeat_true) == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    assert len(repeat_true) == len(repeat_pred)
    num_videos, num_neurons = len(repeat_true), repeat_true[0].shape[1]
    cc_norm = torch.full(
        (num_videos, num_neurons), fill_value=np.nan, dtype=torch.float32
    )
    for i in range(num_videos):
        cc_norm[i] = get_cc_norm(y_true=repeat_true[i], y_pred=repeat_pred[i])
    cc_norm = torch.nanmean(cc_norm, dim=0)  # average over unique videos
    return cc_norm if per_neuron else torch.nanmean(cc_norm)


@torch.inference_mode()
def compute_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, video_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Metrics to compute as part of training and validation step"""
    metrics = dict()
    metrics["msse"] = sse(
        y_true=y_true,
        y_pred=y_pred,
        reduction="mean",
    )
    metrics["poisson_loss"] = poisson_loss(
        y_true=y_true,
        y_pred=y_pred,
        reduction="mean",
    )
    metrics["single_trial_correlation"] = single_trial_correlation(
        y_true=y_true,
        y_pred=y_pred,
    )
    metrics["correlation"] = challenge_correlation(
        y_true=y_true,
        y_pred=y_pred,
    )
    metrics["correlation_to_average"] = correlation_to_average(
        y_true=y_true,
        y_pred=y_pred,
        video_ids=video_ids,
    )
    metrics["normalized_correlation"] = normalized_correlation(
        y_true=y_true,
        y_pred=y_pred,
        video_ids=video_ids,
    )
    return {f"metrics/{k}": v.cpu() for k, v in metrics.items()}
