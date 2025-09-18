from typing import Any
from typing import Tuple

import torch
from torch import nn

from viv1t.model.core.core import Core
from viv1t.model.core.core import register


@register("random")
class RandomCore(Core):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        verbose: int = 0,
    ):
        """
        Behavior mode (--core_behavior_mode)
            0: do not include behavior
            1: concat behavior with visual input
            2: concat behavior and pupil center with visual input
        """
        super(RandomCore, self).__init__(args, input_shape=input_shape, verbose=verbose)
        self.hidden_size = 32
        self.output_shape = (
            self.hidden_size,
            input_shape[1],
            input_shape[2] - 8,
            input_shape[3] - 8,
        )
        self.weight = nn.Parameter(torch.rand(1))

    def regularizer(self):
        return 0.0

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, _, t, h, w = inputs.shape
        outputs = torch.rand(
            *(b, self.hidden_size, t, h - 8, w - 8), device=inputs.device
        )
        return outputs + self.weight - self.weight
