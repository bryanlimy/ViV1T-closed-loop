import traceback
from typing import List

import torch

from viv1t import data
from viv1t.criterions import Criterion
from viv1t.criterions import get_criterion
from viv1t.model import Model


def fit(
    mouse_ids: List[str],
    model: Model,
    criterion: Criterion,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    num_iterations: int = 5,
    verbose: int = 1,
):
    success = False
    try:
        model = model.to(device)
        random_tensor = lambda *size: torch.rand(size, device=device)
        for _ in range(num_iterations):
            for mouse_id in mouse_ids:
                y_pred, _ = model(
                    random_tensor(batch_size, *model.input_shapes["video"]),
                    mouse_id=mouse_id,
                    behaviors=random_tensor(
                        batch_size, *model.input_shapes["behavior"]
                    ),
                    pupil_centers=random_tensor(
                        batch_size, *model.input_shapes["pupil_center"]
                    ),
                )
                y_true = random_tensor(*y_pred.shape)
                loss = criterion(
                    y_true=y_true,
                    y_pred=y_pred,
                    mouse_id=mouse_id,
                    batch_size=batch_size,
                )
                loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        success = True
    except RuntimeError:
        if verbose:
            print(f"Error at batch size: {batch_size}\n{traceback.format_exc()}")
    return success


def estimate_batch_size(args) -> int:
    """
    Calculate the maximum micro bath size that can fill the GPU memory if
    CUDA device is set.

    Return 0 if a micro batch size is found, 1 if OOM at micro batch size of 1.
    """
    if hasattr(args, "micro_batch_size") and args.micro_batch_size:
        assert args.micro_batch_size <= args.batch_size
        return 0

    device = args.device
    if "cuda" not in device.type or args.batch_size == 1:
        args.micro_batch_size = args.batch_size
        return 0

    if args.verbose:
        print(f"\nEstimate micro batch size...")

    train_ds, val_ds, test_ds = data.get_training_ds(
        args,
        data_dir=args.data_dir,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=device,
        num_workers=0,
    )
    mouse_ids = list(train_ds.keys())
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in train_ds.items()
        },
        mean_responses=data.estimate_mean_response(train_ds),
        autocast=args.autocast,
    )
    model = model.to(device)
    if args.compile:
        model.compile()

    args.core_lr = args.lr if args.core_lr is None else args.core_lr
    optimizer = torch.optim.AdamW(
        params=model.get_parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    criterion = get_criterion(args, ds=train_ds)
    del train_ds, val_ds, test_ds

    current, previous, exit_code = 1, None, 0
    while True:
        success = fit(
            mouse_ids=mouse_ids,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=current,
            device=device,
            verbose=args.verbose,
        )
        if success:
            if current >= args.batch_size:
                args.micro_batch_size = args.batch_size
                break
            previous, current = current, current + (1 if current < 8 else 2)
            if current >= args.batch_size:
                current = args.batch_size
        elif current == 1:
            print("OOM at micro batch size of 1.")
            args.micro_batch_size, exit_code = 0, 1
            break
        else:
            # if the previously successful micro_batch_size is larger than 8,
            # then we set micro_batch_size to previous - 2 as to avoid being
            # too closed to the GPU memory limit.
            args.micro_batch_size = previous - 2 if previous > 8 else previous
            break
    del model, optimizer, criterion
    torch.cuda.empty_cache()
    if args.micro_batch_size > 0 and args.verbose:
        print(f"Set micro batch size to {args.micro_batch_size}.")
    return exit_code
