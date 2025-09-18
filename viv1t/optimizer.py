from typing import Any

import torch
import torch.optim
from schedulefree import AdamWScheduleFree
from torch import nn

from viv1t.scheduler import Scheduler


def get_optimizer(
    args: Any, model: nn.Module
) -> (torch.optim.Optimizer, Scheduler | None):
    if args.schedule_free:
        optimizer = AdamWScheduleFree(
            params=model.get_parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            warmup_steps=args.adam_warmup_steps,
            r=args.adam_r,
            weight_lr_power=args.adam_weight_lr_power,
        )
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(
            params=model.get_parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        scheduler = Scheduler(args, optimizer=optimizer, mode="max")
    return optimizer, scheduler
