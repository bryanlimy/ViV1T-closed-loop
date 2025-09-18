from typing import Any

import numpy as np
import torch
from einops import rearrange
from einops import reduce
from einops.layers.torch import Rearrange
from einops.layers.torch import Reduce
from torch import nn

from viv1t.model.readout.readout import Readout
from viv1t.model.readout.readout import Readouts
from viv1t.model.readout.readout import register


class LinearReadout(Readout):
    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: np.ndarray | torch.Tensor,
        mean_responses: torch.Tensor = None,
    ):
        super(LinearReadout, self).__init__(
            args,
            input_shape=input_shape,
            mouse_id=mouse_id,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
        self.linear = nn.Linear(input_shape[0], len(neuron_coordinates))

    def forward(
        self,
        inputs: torch.Tensor,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        b, _, t, _, _ = inputs.shape
        outputs = inputs
        outputs = reduce(outputs, "b c t h w -> (b t) c", "max")
        outputs = self.linear(outputs)
        outputs = rearrange(outputs, "(b t) n -> b n t", b=b, t=t)
        return outputs


@register("linear")
class LinearReadouts(Readouts):
    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        neuron_coordinates: dict[str, np.ndarray | torch.Tensor],
        mean_responses: dict[str, torch.Tensor] = None,
    ):
        super(LinearReadouts, self).__init__(
            args,
            readout=LinearReadout,
            input_shape=input_shape,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
