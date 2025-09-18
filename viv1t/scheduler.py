from typing import Any
from typing import Literal

import torch
from torch.optim import Optimizer

from viv1t.checkpoint import Checkpoint


class Scheduler:
    def __init__(
        self,
        args,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "max",
        warmup_factor: float = 0.1,
        warmup_epochs: int = 0,
        reduce_factor: float = 0.3,
        reduce_patient: int = 5,
        max_reduce: int = 2,
        min_lr: float = None,
    ):
        """
        Linear warmup and reduce on plateau learning rate scheduler
        Args:
            args: argparse parameters.
            optimizer: torch.optim, optimizer.
            mode: 'min' or 'max', compare objective by minimum or maximum
            warmup_factor: the initial learning rate is set to warmup_factor * lr
            warmup_epochs: the number of epochs to warmup (linearly increase)
            reduce_factor: the factor by which the learning rate will be reduced
            reduce_patient: the number of epochs to wait before reducing the
                learning rate
            max_reduce: the maximum number of learning rate reductions before
                early stopping
            min_lr: the minimum learning rate, use finfo.eps if None
        """
        self._optimizer = optimizer

        self.mode = mode
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs - 1

        self.reduce_factor = reduce_factor
        self.reduce_patient = reduce_patient
        self.max_reduce = max_reduce

        self.min_lr = torch.finfo(torch.float32).eps if min_lr is None else min_lr

        self.num_reduce = 0  # number of consecutive learning rate reductions
        self.lr_wait = 0  # number of epochs since last learning rate reduction

        self.was_nan = False  # the value from the previous step was NaN

        assert 0 <= self.warmup_factor <= 1.0
        assert 0 <= self.reduce_factor <= 1.0

        # get target learning rate for all parameter groups
        self.target_lr = {
            param_group["name"]: param_group["lr"]
            for param_group in optimizer.param_groups
        }
        self.target_lr["default"] = optimizer.defaults["lr"]
        # calculate the initial learning rate for all parameter groups
        self.init_lr = {k: self.warmup_factor * v for k, v in self.target_lr.items()}

        match mode:
            case "max":
                self.best_value = 0
            case "min":
                self.best_value = torch.inf
            case _:
                raise ValueError(f"mode must be either min or max.")
        self.best_epoch = 0
        self.verbose = args.verbose
        self._checkpoint = None
        # ignore these attributes when saving and loading state_dict
        self._ignore = ("_optimizer", "_checkpoint", "_ignore", "verbose")

    def set_checkpoint(self, checkpoint: Checkpoint):
        self._checkpoint = checkpoint

    def state_dict(self):
        """State dict for Scheduler"""
        return {k: v for k, v in self.__dict__.items() if k not in self._ignore}

    def load_state_dict(self, state_dict: dict[str, Any]):
        for k in self._ignore:
            state_dict.pop(k, None)
        self.__dict__.update(state_dict)

    def is_better(self, value: float | torch.Tensor) -> bool:
        if self.mode == "min":
            return value < self.best_value
        else:
            return value > self.best_value

    def _increase_lr(self, init_lr: float, target_lr: float, epoch: int):
        """Linearly increase learning rate from init_lr to target_lr"""
        return epoch * ((target_lr - init_lr) / self.warmup_epochs) + init_lr

    def warmup_lr(self, epoch: int):
        """Warm-up learning rate from warmup_factor * lr to lr"""
        if epoch > self.warmup_epochs:
            return
        for i, param_group in enumerate(self._optimizer.param_groups):
            name = param_group["name"]
            new_lr = self._increase_lr(self.init_lr[name], self.target_lr[name], epoch)
            param_group["lr"] = new_lr
            if self.verbose:
                print(f"Warmup {param_group['name']} learning rate to {new_lr:.04e}.")
        # update default learning rate
        self._optimizer.defaults["lr"] = self._increase_lr(
            self.init_lr["default"], self.target_lr["default"], epoch
        )

    def _reduce_lr(self, lr: float):
        return max(self.reduce_factor * lr, self.min_lr)

    def reduce_lr(self):
        """Reduce the learning rates for each param_group by the defined factor"""
        for i, param_group in enumerate(self._optimizer.param_groups):
            param_group["lr"] = self._reduce_lr(param_group["lr"])
            if self.verbose:
                print(
                    f"Reduce {param_group['name']} learning rate to "
                    f"{param_group['lr']:.04e} (num. reduce: {self.num_reduce})."
                )
        # update default learning rate
        self._optimizer.defaults["lr"] = self._reduce_lr(self._optimizer.defaults["lr"])

    def step(self, value: float | torch.Tensor, epoch: int) -> int:
        """Scheduler step function
        Args:
            value: value to compare with best_value
            epoch: current epoch
        Returns:
            exit_code: 0 - continue training, 1 - early stop training.
        """
        if torch.isnan(value):
            # when NaN value is reported, restore from checkpoint if exists,
            # otherwise terminate training
            if self.verbose:
                print(f"NaN correlation reported at epoch {epoch}.")
            if self.best_epoch > 0 and not self.was_nan:
                self._checkpoint.restore()
                self.was_nan = True
                return 0
            else:
                return 1

        improved = self.is_better(value)
        if improved:
            self.best_value, self.best_epoch = value, epoch
            self.lr_wait, self.num_reduce = 0, 0
            self.was_nan = False
            # self._checkpoint.save(value=value, epoch=epoch)

        if epoch <= self.warmup_epochs:
            self.warmup_lr(epoch)
        elif not improved:
            if self.lr_wait < self.reduce_patient - 1:
                self.lr_wait += 1
            elif self.num_reduce < self.max_reduce:
                self.num_reduce += 1
                self._checkpoint.restore()
                self.reduce_lr()
                self.lr_wait = 0
            else:
                if self.verbose:
                    print(f"\nNo improvement after {self.num_reduce} LR reductions.")
                return 1
        return 0
