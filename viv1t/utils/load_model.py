from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from viv1t import data
from viv1t.checkpoint import Checkpoint
from viv1t.model import Model
from viv1t.utils import utils


def load_model(
    args: Any,
    sensorium_dir: Path | None = None,
    evaluate: bool = False,
    autocast: Literal["auto", "disable", "enable"] = "disable",
    compile: bool = False,
    grad_checkpointing: int = 0,
) -> (Model, dict[str, DataLoader]):
    """
    Load a trained model and evaluate the model on the validation set
    if evaluate is set.
    """
    model_args = deepcopy(args)
    if hasattr(model_args, "mouse_ids"):
        del model_args.mouse_ids
    utils.load_args(model_args)
    model_args.crop_frame = data.MAX_FRAME
    model_args.grad_checkpointing = grad_checkpointing
    model_args.verbose = 1
    train_ds, val_ds, _ = data.get_training_ds(
        model_args,
        data_dir=model_args.data_dir,
        mouse_ids=model_args.mouse_ids,
        batch_size=model_args.batch_size,
        device=args.device,
    )
    model = Model(
        model_args,
        neuron_coordinates={
            mouse_id: train_ds[mouse_id].dataset.neuron_coordinates
            for mouse_id in model_args.mouse_ids
        },
        autocast=autocast,
    )
    checkpoint = Checkpoint(model_args, model=model)
    checkpoint.restore(force=True)
    if hasattr(args, "precision"):
        match args.precision:
            case "16":
                print(f"Perform inference in float16")
                model = model.to(torch.float16)
            case "bf16":
                print(f"Perform inference in bfloat16")
                model = model.to(torch.bfloat16)
            case "32":
                print(f"Perform inference in float32")
                model = model.to(torch.float32)
    model = model.to(args.device)
    if compile:
        model.compile()
    # load training and validation set
    if sensorium_dir is not None and sensorium_dir.is_dir():
        model_args.data_dir = sensorium_dir
    if evaluate:
        val_result = utils.evaluate(args, ds=val_ds, model=model, progress_bar=True)
        val_corr = val_result["correlation"]["average"]
        print(f"Validation correlation: {val_corr:.04f}\n")
        best_corr = checkpoint.get_best_corr()
        assert np.allclose(
            best_corr, val_corr, rtol=0, atol=1e-3
        ), f"Mismatch best_corr ({best_corr:.04f}) and val_corr ({val_corr:.04f})."
    return model, train_ds
