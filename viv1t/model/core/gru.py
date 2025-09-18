"""
Code reference: GRU module from https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/rnn_modules/gru_module.py
"""

from typing import Any
from typing import Tuple

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell
    https://github.com/sinzlab/Sinz2018_NIPS/blob/master/nips2018/architectures/cores.py
    """

    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        pad_input: bool = True,
    ):
        super().__init__()
        self.input_shape = input_shape

        input_kern = args.core_rnn_input_kernel
        rec_kern = args.core_rnn_kernel
        input_channels = self.input_shape[0]
        rec_channels = args.core_rnn_size

        input_padding = input_kern // 2 if pad_input else 0
        rec_padding = rec_kern // 2

        self.rec_channels = rec_channels
        self._shrinkage = 0 if pad_input else input_kern - 1

        self.register_buffer("gamma_rec", torch.tensor(args.core_rnn_reg))
        self.reset_gate_input = nn.Conv2d(
            input_channels, rec_channels, input_kern, padding=input_padding
        )
        self.reset_gate_hidden = nn.Conv2d(
            rec_channels, rec_channels, rec_kern, padding=rec_padding
        )

        self.update_gate_input = nn.Conv2d(
            input_channels, rec_channels, input_kern, padding=input_padding
        )
        self.update_gate_hidden = nn.Conv2d(
            rec_channels, rec_channels, rec_kern, padding=rec_padding
        )

        self.out_gate_input = nn.Conv2d(
            input_channels, rec_channels, input_kern, padding=input_padding
        )
        self.out_gate_hidden = nn.Conv2d(
            rec_channels, rec_channels, rec_kern, padding=rec_padding
        )

        self.apply(self.init_conv)
        self.register_parameter("_prev_state", None)

        self.output_shape = (
            rec_channels,
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def init_state(self, input_: torch.Tensor):
        batch_size, _, *spatial_size = input_.data.size()
        state_size = [batch_size, self.rec_channels] + [
            s - self._shrinkage for s in spatial_size
        ]
        prev_state = nn.Parameter(torch.zeros(*state_size, device=input_.device))
        return prev_state

    def bias_l1(self):
        return (
            self.reset_gate_hidden.bias.abs().mean() / 3
            + self.update_gate_hidden.weight.abs().mean() / 3
            + self.out_gate_hidden.bias.abs().mean() / 3
        )

    def regularizer(self):
        return self.gamma_rec * self.bias_l1()

    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor = None):
        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(inputs)

        update = self.update_gate_input(inputs) + self.update_gate_hidden(prev_state)
        update = F.sigmoid(update)

        reset = self.reset_gate_input(inputs) + self.reset_gate_hidden(prev_state)
        reset = F.sigmoid(reset)

        out = self.out_gate_input(inputs) + self.out_gate_hidden(prev_state * reset)
        h_t = F.tanh(out)
        new_state = prev_state * (1 - update) + h_t * update

        return new_state


class GRU(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, int, int, int]):
        """
        A GRU module for video data to add between the core and the readout.
        Receives as input the output of a 3Dcore. Expected dimensions:
            - (batch, channels, num. frames, height, width) or
            - (channels, num. frames, height, width)
        The input is fed sequentially to a convolutional GRU cell, based on the
        frames channel. The output has the same dimensions as the input.
        """
        super(GRU, self).__init__()
        self.input_shape = input_shape
        self.conv_gru_cell = ConvGRUCell(args, input_shape=input_shape)
        self.output_shape = self.conv_gru_cell.output_shape

    def regularizer(self):
        return self.conv_gru_cell.regularizer()

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass definition based on https://github.com/sinzlab/Sinz2018_NIPS/blob/3a99f7a6985ae8dec17a5f2c54f550c2cbf74263/nips2018/architectures/cores.py#L556
        Modified to also accept 4 dimensional inputs (assuming no batch
        dimension is provided).
        """
        match inputs.ndim:
            case 4:
                batch = False
                inputs = rearrange(inputs, "c t h w -> 1 c t h w")
            case 5:
                batch = True
            case _:
                raise RuntimeError(
                    "input to GRU must has 4 (unbatched) or 5 (batched) dimensions."
                )
        states = []
        hidden = None
        time_index = 2
        for frame in range(inputs.shape[time_index]):
            slice_channel = [
                frame if time_index == i else slice(None) for i in range(inputs.ndim)
            ]
            hidden = self.conv_gru_cell(inputs[slice_channel], hidden)
            states.append(hidden)
        outputs = torch.stack(states, time_index)
        if not batch:
            outputs = rearrange(outputs, "1 c t h w -> c t h w")
        return outputs
