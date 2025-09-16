from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from viv1t.utils import yaml


class Checkpoint:
    """Checkpoint manager for saving and loading model, optimizer, and scheduler"""

    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None,
    ):
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self.ckpt_dir = args.output_dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model_ckpt = self.ckpt_dir / "model.pt"
        self.optimizer_ckpt = self.ckpt_dir / "optimizer.pt"
        self.scheduler_ckpt = self.ckpt_dir / "scheduler.pt"
        self.ckpt_info = self.ckpt_dir / "info.yaml"
        self.verbose = args.verbose

        # store the non-compiled model state_dict keys
        self.model_state_dict_keys = self._model.state_dict().keys()

    def get_model_state_dict(self) -> dict[str, torch.Tensor]:
        """Return model state_dict without torch.compile _orig_mod. prefix"""
        self._model = self._model.to("cpu")
        state_dict = self._model.state_dict()
        clean_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "_orig_mod." in k:
                k = k.replace("_orig_mod.", "")
            clean_state_dict[k] = v
        assert self.model_state_dict_keys == clean_state_dict.keys()
        return clean_state_dict

    def save(self, value: float | np.ndarray, epoch: int):
        """Save checkpoints for model, optimizer and scheduler."""
        if self._optimizer is not None and hasattr(self._optimizer, "eval"):
            self._optimizer.eval()
        # save model weights
        torch.save(self.get_model_state_dict(), f=self.model_ckpt)
        # save checkpoint information
        yaml.save(self.ckpt_info, data={"epoch": epoch, "best_corr": float(value)})
        # save optimizer parameters
        if self._optimizer is not None:
            torch.save(self._optimizer.state_dict(), f=self.optimizer_ckpt)
        # save scheduler parameters
        if self._scheduler is not None:
            torch.save(self._scheduler.state_dict(), f=self.scheduler_ckpt)
        if self.verbose:
            print(f"Checkpoint saved to {self.ckpt_dir}.")

    def get_best_corr(self) -> float:
        """Return the best correlation value from the checkpoint"""
        assert self.ckpt_info.is_file(), f"Cannot find checkpoint in {self.ckpt_dir}."
        info = yaml.load(self.ckpt_info)
        return info["best_corr"]

    def restore(
        self,
        force: bool = False,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> int:
        """
        Load model in self.checkpoint if exists and return the epoch number
        Args:
            force: bool, raise an error if checkpoint is not found.
            load_optimizer: bool, load optimizer and scaler (if exists) from checkpoint.
            load_scheduler: bool, load scheduler from checkpoint.
        Return:
            epoch: int, the number of epoch the model has been trained for,
                return 0 if checkpoint does not exist.
        """
        epoch = 0
        load_kwargs = {"map_location": "cpu", "weights_only": True}
        if self.ckpt_dir.is_dir() and self.ckpt_info.is_file():
            ckpt_info = yaml.load(self.ckpt_info)
            device = self._model.device
            model_ckpt = torch.load(self.model_ckpt, **load_kwargs)
            # it is possible that the checkpoint only contains part of a model
            # hence we update the current state_dict of the model instead of
            # directly calling model.load_state_dict(ckpt)
            current = self._model.state_dict()
            weights = {k: v for k, v in model_ckpt.items() if k in current}
            current.update(weights)
            missing_keys, unexpected_keys = self._model.load_state_dict(current)
            if missing_keys or unexpected_keys:
                print(
                    f"Warning: {missing_keys} missing keys and {unexpected_keys} "
                    f"unexpected keys when restoring from checkpoint."
                )
            self._model = self._model.to(device)
            # load optimizer parameters if exists
            if load_optimizer and self.optimizer_ckpt.is_file():
                optimizer_ckpt = torch.load(self.optimizer_ckpt, **load_kwargs)
                self._optimizer.load_state_dict(optimizer_ckpt)
            # load scheduler parameters if exists
            if load_scheduler and self.scheduler_ckpt.is_file():
                scheduler_ckpt = torch.load(self.scheduler_ckpt, **load_kwargs)
                self._scheduler.load_state_dict(scheduler_ckpt)
            epoch = ckpt_info["epoch"]
            if self.verbose:
                print(
                    f"\nLoaded {len(weights)}/{len(current)} parameters from {self.model_ckpt}"
                    f" (epoch: {epoch}, correlation: {ckpt_info['best_corr']:.04f}).\n"
                )
        elif force:
            raise FileNotFoundError(f"Cannot find checkpoint in {self.ckpt_dir}.")
        return epoch
