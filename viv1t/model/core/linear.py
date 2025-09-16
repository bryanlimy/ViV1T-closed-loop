from typing import Any
from typing import Tuple

import torch
from einops import rearrange
from torch import nn

from viv1t.model.core.core import Core
from viv1t.model.core.core import register
from viv1t.model.modules import ConcatBehaviors


@register("linear")
class LinearCore(Core):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        verbose: int = 0,
    ):
        super(LinearCore, self).__init__(args, input_shape=input_shape, verbose=verbose)
        self.input_shape = input_shape
        self.behavior_mode = args.core_behavior_mode

        self.concat_behaviors = ConcatBehaviors(input_shape, self.behavior_mode)
        input_shape = self.concat_behaviors.output_shape

        c, _, h, w = input_shape
        self.register_parameter("weight", nn.Parameter(torch.randn(c, h, w)))
        self.register_parameter("bias", nn.Parameter(torch.randn(c, h, w)))

        self.register_buffer("reg_scale", torch.tensor(0.0))

        self.output_shape = input_shape

    def regularizer(self):
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, _, t, _, _ = inputs.shape
        outputs = inputs
        outputs = self.concat_behaviors(outputs, behaviors, pupil_centers)
        outputs = rearrange(outputs, "b c t h w -> (b t) c h w")
        outputs = self.weight * outputs + self.bias
        outputs = rearrange(outputs, "(b t) c h w -> b c t h w", b=b, t=t)
        return outputs
