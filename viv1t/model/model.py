import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
import torchinfo
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from viv1t.data import estimate_mean_response
from viv1t.model.core import get_core
from viv1t.model.modules import OutputActivation
from viv1t.model.readout import get_readouts
from viv1t.model.shifter import MLPShifters
from viv1t.utils import utils
from viv1t.utils import yaml


class Model(nn.Module):
    def __init__(
        self,
        args: Any,
        neuron_coordinates: dict[str, torch.Tensor],
        mean_responses: dict[str, torch.Tensor] = None,
        autocast: Literal["auto", "disable", "enable"] = "disable",
        verbose: int = 0,
    ):
        super(Model, self).__init__()
        self.input_shapes = args.input_shapes
        self.output_shapes = args.output_shapes
        self.verbose = verbose

        self.clip_grad = args.clip_grad if hasattr(args, "clip_grad") else 0
        # store up to 100,000 gradient norms from the past
        self.grad_idx = 0
        self.grad_norms = np.zeros(100000, dtype=np.float32)

        self.add_module(
            name="core",
            module=get_core(args)(
                args,
                input_shape=self.input_shapes["video"],
                verbose=self.verbose,
            ),
        )

        if args.shifter_mode:
            self.add_module(
                "shifters",
                module=MLPShifters(
                    args,
                    input_shapes=self.input_shapes,
                    mouse_ids=list(self.output_shapes.keys()),
                ),
            )
        else:
            self.shifters = None

        self.add_module(
            name="readouts",
            module=get_readouts(args)(
                args,
                input_shape=self.core.output_shape,
                neuron_coordinates=neuron_coordinates,
                mean_responses=mean_responses,
            ),
        )

        if args.transform_output == 2 and args.output_mode not in (0, 4):
            args.output_mode = 4
            print(f"Overwrite to sigmoid output since normalized response is expected.")

        self.add_module(
            name="output_activation",
            module=OutputActivation(
                output_mode=args.output_mode,
                neuron_coordinates=neuron_coordinates,
            ),
        )
        self.set_autocast(autocast=autocast, device=args.device)

    @property
    def device(self) -> torch.device:
        """return the device that the model parameters is on"""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """return the dtype that the model parameters"""
        return next(self.parameters()).dtype

    def load_pretrained_core(self, pretrained_dir: Path):
        """Load trained core module weights"""
        device = self.device
        ckpt_dir = pretrained_dir / "ckpt"
        ckpt_info = yaml.load(ckpt_dir / "info.yaml")
        pretrained_weights = torch.load(
            ckpt_dir / "model.pt", map_location=device, weights_only=True
        )
        # select only core module weights
        pretrained_weights = {
            k: v for k, v in pretrained_weights.items() if k.startswith("core.")
        }
        # restore parameters that have the same key
        current_weights = self.state_dict()
        weights = {k: v for k, v in pretrained_weights.items() if k in current_weights}
        current_weights.update(weights)
        self.load_state_dict(current_weights)
        if self.verbose:
            print(
                f"\nLoaded pretrained core from {pretrained_dir} with "
                f"{len(weights)}/{len(current_weights)} parameters "
                f"(epoch: {ckpt_info['epoch']}, "
                f"correlation: {ckpt_info['best_corr']:.04f}).\n"
            )

    def get_parameters(self):
        """Return a list of parameters for torch.optim.Optimizer"""
        params = []
        params.extend(self.core.get_parameters())
        if self.shifters is not None and len(list(self.shifters.parameters())) > 0:
            params.append({"params": self.shifters.parameters(), "name": "shifters"})
        params.append({"params": self.readouts.parameters(), "name": "readouts"})
        if len(list(self.output_activation.parameters())) > 0:
            params.append(
                {"params": self.output_activation.parameters(), "name": "activation"}
            )
        return params

    def set_autocast(
        self,
        autocast: Literal["auto", "disable", "enable"],
        device: torch.device,
    ):
        # set core computation precision to bfloat16 if supported
        device = device.type
        support_bf16 = utils.support_bf16(device)
        if autocast == "enable" and not support_bf16:
            print(f"Device {device} does not support torch.bfloat16.")
        if support_bf16 and autocast in ("auto", "enable"):
            print("Enable torch.autocast in bfloat16.")
            self.autocast = torch.autocast(device, dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

    def clip_grad_norm(self) -> float | np.ndarray:
        """Clip the gradient norm if clip_grad is set to non-zero value

        If clip_grad is set to -1, then max_norm is set to the 10th percentile
        of past gradient norms based on AutoClip (https://arxiv.org/abs/2007.14469).
        If clip_grad is set to a positive value, then set max_norm to it.
        """
        max_norm = torch.inf
        if self.clip_grad == -1 and self.grad_idx > 0:
            max_norm = np.quantile(self.grad_norms[: self.grad_idx], q=0.1)
        elif self.clip_grad > 0:
            max_norm = self.clip_grad
        total_norm = clip_grad_norm_(self.parameters(), max_norm=max_norm).item()
        # log gradient norm
        self.grad_norms[self.grad_idx % len(self.grad_norms)] = total_norm
        self.grad_idx += 1
        return total_norm

    def compile(self):
        """Compile submodules that support torch.compile"""
        self.core.compile()

    def regularizer(self, mouse_id: str):
        reg = 0
        if not self.core.frozen:
            reg += self.core.regularizer()
        reg += self.readouts.regularizer(mouse_id=mouse_id)
        if self.shifters is not None:
            reg += self.shifters.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        activate: bool = True,
    ):
        dtype = inputs.dtype
        with self.autocast:
            core_outputs = self.core(
                inputs=inputs,
                mouse_id=mouse_id,
                behaviors=behaviors,
                pupil_centers=pupil_centers,
            )
        core_outputs = core_outputs.to(dtype)
        shifts, t = None, core_outputs.size(2)
        if self.shifters is not None:
            shifts = self.shifters(
                behaviors=behaviors[..., -t:],
                pupil_centers=pupil_centers[..., -t:],
                mouse_id=mouse_id,
            )
        outputs = self.readouts(
            core_outputs,
            mouse_id=mouse_id,
            shifts=shifts,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        if activate:
            outputs = self.output_activation(outputs, mouse_id=mouse_id)
        return outputs, core_outputs.detach()


def log_model_info(
    args,
    model: Model,
    filename: Path,
    device: torch.device = "cpu",
) -> int:
    random_tensor = lambda size: torch.rand((1, *size), device=device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model_info = torchinfo.summary(
            model=model,
            input_data={
                "inputs": random_tensor(args.input_shapes["video"]),
                "behaviors": random_tensor(args.input_shapes["behavior"]),
                "pupil_centers": random_tensor(args.input_shapes["pupil_center"]),
            },
            col_names=["input_size", "output_size", "num_params"],
            col_width=30,
            depth=8,
            device=device,
            verbose=0,
            mouse_id=args.mouse_ids[0],
        )
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(str(model_info))
    if args.verbose > 2:
        print(str(model_info))
    return model_info.trainable_params


def get_model(args, ds: dict[str, DataLoader], model_info: bool = False) -> Model:
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in ds.items()
        },
        mean_responses=estimate_mean_response(ds),
        autocast=args.autocast,
        verbose=args.verbose,
    )
    if hasattr(args, "pretrained_core") and args.pretrained_core is not None:
        model.load_pretrained_core(pretrained_dir=args.pretrained_core)
    if args.core_lr == 0:
        model.core.freeze()
    if model_info:
        args.trainable_params = log_model_info(
            args,
            model=model,
            filename=args.output_dir / "model.txt",
        )
    return model
