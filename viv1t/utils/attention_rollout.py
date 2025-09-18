from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torch import nn

from viv1t.model.core.vivit import ParallelAttentionBlock
from viv1t.model.core.vivit import Transformer


class Recorder(nn.Module):
    def __init__(self, transformer: Transformer):
        super(Recorder, self).__init__()
        self.transformer = transformer
        self.cache = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False

    @staticmethod
    def _find_modules(module: nn.Module, type: nn.Module):
        return [m for m in module.modules() if isinstance(m, type)]

    def _hook(self, _, inputs: torch.Tensor, outputs: torch.Tensor):
        self.cache.append(outputs.cpu().detach().clone())

    def _register_hook(self):
        modules = self._find_modules(self.transformer, ParallelAttentionBlock)
        for module in modules:
            handle = module.softmax.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.transformer

    def clear(self):
        self.cache.clear()
        torch.cuda.empty_cache()

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        mouse_id: str,
    ):
        """
        Return softmax scaled dot product outputs from ViT/V1T

        Returns:
            outputs: torch.Tensor, the output of the core
            attentions: torch.Tensor, the softmax scaled dot product outputs in
                format (batch size, num blocks, num heads, num patches, num patches)
        """
        assert not self.ejected, "recorder has been ejected, cannot be used anymore"
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        outputs = self.transformer(
            inputs=inputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        attentions = None
        if len(self.cache) > 0:
            attentions = torch.stack(self.cache, dim=1)
        return outputs, attentions


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())


def attention_rollout(attention: torch.Tensor):
    """
    Apply Attention rollout from https://arxiv.org/abs/2005.00928 to a single
    sample of softmax attention
    Code examples
    - https://keras.io/examples/vision/probing_vits/#method-ii-attention-rollout
    - https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    """
    assert attention.dim() == 4
    # take max values of attention heads
    attention, _ = torch.max(attention, dim=1)

    # to account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(attention.size(1))
    aug_att_mat = attention + residual_att
    aug_att_mat = aug_att_mat / torch.sum(aug_att_mat, dim=-1, keepdim=True)

    # recursively multiply the weight matrices
    rollout = torch.zeros_like(aug_att_mat)
    rollout[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        rollout[n] = torch.matmul(aug_att_mat[n], rollout[n - 1])

    return rollout[-1]


def fold_spatial_attention(
    patch_attentions: np.ndarray,
    frame_size: Tuple[int, int],
    patch_size: int,
    stride: int,
    padding: Tuple[int, int, int, int, int, int],
):
    """
    Fold the spatial attention value per patch to the original video frame dimension.

    Args:
        patch_attentions: the attention value per patch in shape (B, P) where P is
            the number of spatial patches
        frame_size: the original frame size in (H, W)
        patch_size: patch size used to tokenize the video frame
        stride: patch stride used to tokenize the video frame
        padding: padding used to tokenize the video frame in
            (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    if not torch.is_tensor(patch_attentions):
        patch_attentions = torch.from_numpy(patch_attentions)
    assert patch_attentions.dim() == 2 and len(padding) == 6
    output_size = (
        frame_size[0] + padding[2] + padding[3],
        frame_size[1] + padding[0] + padding[1],
    )
    # stretch the attention value in each patch to (patch_size, patch_size)
    patch_attentions = repeat(
        patch_attentions, "b n -> b n ph pw", ph=patch_size, pw=patch_size
    )
    patch_attentions = rearrange(patch_attentions, "b n ph pw -> b (ph pw) n")
    frame_attention = F.fold(
        patch_attentions,
        output_size=output_size,
        kernel_size=patch_size,
        stride=stride,
    )
    # normalize by the of times when a pixel was part of a patch
    # note that F.fold sum the values over overlapping patches
    ones = F.fold(
        torch.ones_like(patch_attentions),
        output_size=output_size,
        kernel_size=patch_size,
        stride=stride,
    )
    frame_attention = frame_attention / ones
    frame_attention = frame_attention[:, 0, :, :]
    # remove zero padding
    frame_attention = frame_attention[
        :, padding[2] : -padding[3], padding[0] : -padding[1]
    ]
    max_attention = torch.amax(frame_attention, dim=(1, 2), keepdim=True)
    frame_attention = frame_attention / max_attention
    return frame_attention.numpy()


def spatial_attention_rollout(attentions: torch.Tensor):
    """
    Apply attention rollout over the spatial patches and return the attention
    value per patch.
    """
    assert attentions.dim() == 5
    batch_size = attentions.size(0)
    attention_values = torch.zeros(
        (batch_size, attentions.shape[-1]),
        dtype=attentions.dtype,
        device=attentions.device,
    )
    for i in range(batch_size):
        rollout = attention_rollout(attentions[i])
        attention_values[i] = torch.mean(rollout, dim=0)
    return attention_values


def temporal_attention_rollout(attentions: torch.Tensor):
    """
    Attention rollout a sample of softmax attentions from the temporal Transformer
    """
    assert attentions.dim() == 5
    batch_size = attentions.size(0)
    heatmaps = torch.zeros(
        (batch_size, attentions.shape[-1]),
        dtype=attentions.dtype,
        device=attentions.device,
    )
    for i in range(batch_size):
        rollout = attention_rollout(attentions[i])
        rollout = torch.mean(rollout, dim=0)
        heatmaps[i] = rollout
    heatmaps = torch.mean(heatmaps, dim=0)
    heatmaps = normalize(heatmaps)
    return heatmaps
