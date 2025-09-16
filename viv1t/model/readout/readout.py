_READOUTS = dict()

from typing import Any

import numpy as np
import torch
from torch import nn


def register(name):
    def add_to_dict(fn):
        global _READOUTS
        _READOUTS[name] = fn
        return fn

    return add_to_dict


class Readout(nn.Module):
    """Basic readout module for a single rodent"""

    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: np.ndarray | torch.Tensor,
        mean_responses: np.ndarray | torch.Tensor | None = None,
    ):
        super(Readout, self).__init__()
        self.mouse_id = mouse_id
        self.input_shape = input_shape
        self.output_shape = (len(neuron_coordinates), input_shape[1])
        self.register_buffer("reg_scale", torch.tensor(0.0))

    @property
    def num_neurons(self):
        """Number of neurons to output"""
        return self.output_shape[0]

    def initialize(self, *args: Any, **kwargs: Any):
        pass

    def regularizer(self, reduction: str):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(
        self,
        inputs: torch.Tensor,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        raise NotImplementedError("Readout forward method not implemented.")


class Readouts(nn.ModuleDict):
    """Dictionary of Readout modules to store Mouse ID: Readout module pairs"""

    def __init__(
        self,
        args: Any,
        input_shape: tuple[int, int, int, int],
        neuron_coordinates: dict[str, np.ndarray | torch.Tensor],
        mean_responses: dict[str, np.ndarray | torch.Tensor] | None = None,
        readout: Readout = Readout,
    ):
        super(Readouts, self).__init__()
        self.input_shape = input_shape
        for mouse_id in neuron_coordinates.keys():
            self.add_module(
                name=mouse_id,
                module=readout(
                    args,
                    input_shape=input_shape,
                    mouse_id=mouse_id,
                    neuron_coordinates=neuron_coordinates[mouse_id],
                    mean_responses=(
                        None if mean_responses is None else mean_responses[mouse_id]
                    ),
                ),
            )

    def regularizer(self, mouse_id: str, reduction: str = "sum"):
        return self[mouse_id].regularizer(reduction=reduction)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        return self[mouse_id](
            inputs, shifts=shifts, behaviors=behaviors, pupil_centers=pupil_centers
        )


def get_readouts(args) -> Readouts:
    readouts = args.readout.lower()
    if not readouts in _READOUTS.keys():
        raise NotImplementedError(f"Readout {readouts} has not been implemented.")
    return _READOUTS[readouts]
