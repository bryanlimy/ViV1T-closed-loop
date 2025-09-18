from math import floor
from typing import Any
from typing import Tuple

import torch
from torch import nn

from viv1t.model.core.core import Core
from viv1t.model.core.core import register


@register("large_random")
class LargeRandomCore(Core):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        verbose: int = 0,
    ):
        super(LargeRandomCore, self).__init__(
            args,
            input_shape=input_shape,
            verbose=verbose,
        )
        self.hidden_size = 32
        self.output_shape = (
            self.hidden_size,
            input_shape[1],
            input_shape[2] - 8,
            input_shape[3] - 8,
        )
        self.weight = nn.Parameter(torch.rand(1))
        memory = torch.cuda.get_device_properties(0).total_memory
        memory = memory / 4 * 0.6
        self.filler = nn.Parameter(torch.rand(floor(memory)))
        self.filler.requires_grad_(False)

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
        return outputs * self.weight
