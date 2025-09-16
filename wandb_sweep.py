import argparse
import traceback
from functools import partial
from pathlib import Path

import wandb

import train as trainer
from viv1t.utils.utils import get_timestamp


def get_sweep_name(sweep_id: str) -> str:
    api = wandb.Api()
    sweep = api.sweep(f"bryanlimy/sensorium2023/{sweep_id}")
    return sweep.name


def get_run_number(sweep_id: str) -> int:
    """the run number is the number of runs in the sweep (which includes the current run)"""
    api = wandb.Api()
    runs = api.runs(path="bryanlimy/sensorium2023", filters={"sweep": sweep_id})
    return len(runs)


def get_run_name(run_id: str) -> str:
    """
    run name is the timestamp followed by the run ID, which is the display name
    on wandb and the output_dir name.
    """
    return f"{get_timestamp()}-{run_id}"


class Args:
    def __init__(
        self,
        run_name: str,
        config: wandb.Config,
        data_dir: Path,
        output_dir: Path,
        wandb_group: str,
        num_workers: int,
        verbose: int = 1,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir / wandb_group / run_name

        self.wandb = wandb_group
        self.wandb_id = None
        self.num_workers = num_workers
        self.verbose = verbose

        self.mouse_ids = None
        self.limit_data = None
        self.micro_batch_size = 0
        self.device = ""
        self.seed = 1234
        self.deterministic = False
        self.compile = True
        self.autocast = "auto"
        self.clear_output_dir = False

        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)


def experiment(
    sweep_id: str,
    data_dir: Path,
    output_dir: Path,
    num_workers: int,
):
    try:
        sweep_name = get_sweep_name(sweep_id)
        run = wandb.init(group=sweep_name)
        run.name = get_run_name(run.id)  # update run name with run ID
        run_args = Args(
            run_name=run.name,
            config=run.config,
            data_dir=data_dir,
            output_dir=output_dir,
            wandb_group=sweep_name,
            num_workers=num_workers,
        )
        trainer.main(run_args, wandb_sweep=True)
    except Exception as e:
        print(f"\nExperiment error: {e}\n{traceback.format_exc()}\n")
        wandb.finish(exit_code=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="/home/storage/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument("--output_dir", type=Path, default="/home/storage/runs")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])

    sweep_args = parser.parse_args()

    try:
        wandb.agent(
            sweep_id=f"bryanlimy/sensorium2023/{sweep_args.sweep_id}",
            function=partial(
                experiment,
                sweep_id=sweep_args.sweep_id,
                data_dir=sweep_args.data_dir,
                output_dir=sweep_args.output_dir,
                num_workers=sweep_args.num_workers,
            ),
            count=1,
        )
    except Exception as e:
        print(f"\nError in wandb.agent: {e}\n{traceback.format_exc()}")
