import math
from typing import Any

import numpy as np
import torch
from einops import rearrange
from torch import nn

from viv1t.model.readout.readout import Readout
from viv1t.model.readout.readout import Readouts
from viv1t.model.readout.readout import register


class ConvReadout(Readout):
    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: np.ndarray | torch.Tensor,
        mean_responses: torch.Tensor = None,
    ):
        super(ConvReadout, self).__init__(
            args,
            input_shape=input_shape,
            mouse_id=mouse_id,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
        self.layer = nn.Sequential(
            nn.Dropout1d(p=args.readout_dropout),
            nn.Conv1d(
                in_channels=input_shape[0],
                out_channels=math.ceil(self.num_neurons / args.readout_groups)
                * args.readout_groups,
                kernel_size=(1,),
                groups=args.readout_groups,
                bias=True,
            ),
        )
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(
        self,
        inputs: torch.Tensor,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        outputs = self.pool(inputs)
        outputs = rearrange(outputs, "b c t 1 1 -> b c t")
        outputs = self.layer(outputs)  # (B, N, T)
        outputs = outputs[:, : self.num_neurons]  # (B, N, T)
        return outputs


@register("conv")
class ConvReadouts(Readouts):
    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        neuron_coordinates: dict[str, np.ndarray | torch.Tensor],
        mean_responses: dict[str, torch.Tensor] = None,
    ):
        super(ConvReadouts, self).__init__(
            args,
            readout=ConvReadout,
            input_shape=input_shape,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
