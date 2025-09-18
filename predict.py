"""
Predict the train, validation, live main, live bonus, final main and final
bonus sets and store the predicted responses into h5 files
"""

import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t.data import ROCHEFORT_LAB
from viv1t.data import SENSORIUM_OLD
from viv1t.data import get_submission_ds
from viv1t.data import get_training_ds
from viv1t.model import Model
from viv1t.utils import h5
from viv1t.utils import utils
from viv1t.utils.load_model import load_model

SKIP = 50  # skip the first 50 frames from each trial


@torch.inference_mode()
def inference(
    args,
    ds: DataLoader,
    model: Model,
    mouse_id: str,
    save_dir: Path,
    device: torch.device = "cpu",
    desc: str = "",
):
    assert (
        ds.batch_size == 1
    ), f"batch size must be 1 due to trials with varying duration in the test sets."
    model = model.to(device)
    model.train(False)
    predictions = {}
    i_transform_response = ds.dataset.i_transform_response
    for batch in tqdm(ds, desc=desc, disable=args.verbose < 2):
        trial_id = batch["trial_id"][0].item()
        t = batch["video"].shape[2] - SKIP
        prediction, _ = model(
            inputs=batch["video"].to(device, model.dtype),
            mouse_id=mouse_id,
            behaviors=batch["behavior"].to(device, model.dtype),
            pupil_centers=batch["pupil_center"].to(device, model.dtype),
        )
        prediction = prediction.to("cpu", torch.float32)
        predictions[trial_id] = i_transform_response(prediction[0, :, -t:])
    h5.write(save_dir / f"mouse{mouse_id}.h5", predictions)


def main(args):
    if not args.output_dir.is_dir():
        raise FileNotFoundError(f"Cannot find {args.output_dir}.")

    args.device = utils.get_device(args.device)
    if args.mouse_ids is None:
        del args.mouse_ids
    utils.load_args(args)

    # set batch size to 1 due to different trial length in the test set
    args.batch_size = 1
    # use all frames
    args.crop_frame = -1
    # inference all data
    args.limit_data = None

    model, _ = load_model(args, evaluate=True, compile=args.compile)

    train_ds, val_ds, _ = get_training_ds(
        args,
        data_dir=args.data_dir,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )
    _, test_ds = get_submission_ds(
        args,
        data_dir=args.data_dir,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ignore mouse dataset where labels are not available
    mice_with_labels = SENSORIUM_OLD + ROCHEFORT_LAB
    filter = lambda d: {k: v for k, v in d.items() if k in mice_with_labels}

    if args.ds_tier:
        ds_tiers = [args.ds_tier]
    else:
        ds_tiers = [
            "train",
            "validation",
            "live_main",
            "live_bonus",
            "final_main",
            "final_bonus",
        ]
    save_dir = args.output_dir / "responses"

    for ds_tier in ds_tiers:
        match ds_tier:
            case "train":
                ds = train_ds
            case "validation":
                ds = val_ds
            case "live_main":
                ds = filter(test_ds["live_main"])
            case "live_bonus":
                ds = filter(test_ds["live_bonus"])
            case "final_main":
                ds = filter(test_ds["final_main"])
            case "final_bonus":
                ds = filter(test_ds["final_bonus"])
            case _:
                raise KeyError(f"Unknown dataset tier {ds_tier}")

        print(f"\nInference {ds_tier} set.")
        for mouse_id, mouse_ds in ds.items():
            inference(
                args,
                ds=mouse_ds,
                model=model,
                mouse_id=mouse_id,
                save_dir=save_dir,
                device=args.device,
                desc=f"Mouse {mouse_id}",
            )

    print(f"Save prediction to {save_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument(
        "--ds_tier",
        type=str,
        default="",
        choices=[
            "",
            "train",
            "validation",
            "live_main",
            "live_bonus",
            "final_main",
            "final_bonus",
        ],
    )
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=str,
        default=None,
        help="Mouse to use for training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of workers for DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "use the best available device if --device is not specified.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="torch.compile the model"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["32", "bf16"],
        default="32",
        help="Precision to use for inference, both model weights and input data would be converted.",
    )
    parser.add_argument("--verbose", type=int, default=2)
    main(parser.parse_args())
