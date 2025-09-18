import argparse
import os
import pickle
import traceback
from argparse import RawTextHelpFormatter
from socket import gethostname
from time import sleep
from typing import Any
from typing import Dict

import ray
import wandb
from hpo import get_search_space
from ray import air
from ray import tune
from ray.air import FailureConfig
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from wandb.util import generate_id

import train as trainer
from viv1t.utils import utils

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
os.environ["RAY_AIR_REENABLE_DEPRECATED_SYNC_TO_HEAD_NODE"] = "1"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"


def trial_name_creator(trial: tune.experiment.trial.Trial):
    return f"{trial.trial_id}"


def trial_dirname_creator(trial: tune.experiment.trial.Trial):
    return f"{utils.get_timestamp()}-{trial.trial_id}"


class Args:
    def __init__(
        self,
        config: Dict[str, Any],
        data_dir: str,
        output_dir: str,
        epochs: int,
        wandb_group: str,
        wandb_id: str,
        num_workers: int,
        verbose: int = 1,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.epochs = epochs
        self.wandb = wandb_group
        self.wandb_id = wandb_id
        self.num_workers = num_workers
        self.verbose = verbose

        self.mouse_ids = None
        self.limit_data = None
        self.restore = None
        self.micro_batch_size = 0
        self.device = ""
        self.seed = 1234
        self.deterministic = False
        self.backend = None
        self.precision = None
        self.clear_output_dir = False

        for key, value in config.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        # hyperparameters that are in power of 2 settings
        for key in ["batch_size"]:
            setattr(self, key, 2 ** getattr(self, key))

        # hyperparameters that are in log scale settings
        for key in ["lr", "weight_decay", "core_lr", "core_weight_decay"]:
            setattr(self, key, 10 ** float(getattr(self, key)))


def trial(config, wandb_group: str, data_dir: str, epochs: int, num_workers: int):
    hostname, failed = gethostname(), 1
    sleep(10)
    # make log to ray tune to avoid error
    session.report({"best_corr": 0, "epoch": 0})
    try:
        run_id = f"{session.get_trial_id()}-{generate_id()}"
        args = Args(
            config=config,
            data_dir=data_dir,
            wandb_id=run_id,
            output_dir=session.get_trial_dir(),
            epochs=epochs,
            wandb_group=wandb_group,
            num_workers=num_workers,
        )
        failed = trainer.main(args, ray_sweep=True)
    except Exception as e:
        print(f"\nError from {hostname}: {e}\n{traceback.format_exc()}")
        if wandb.run is not None:
            wandb.finish(exit_code=failed)
    sleep(10)


def main(args):
    # set directory for local worker and head nodes
    output_dir = os.path.join(args.storage_path, args.wandb_group)
    os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = output_dir

    # look for the existing cluster and connect to it
    ray.init(include_dashboard=False)

    trainable = tune.with_resources(
        tune.with_parameters(
            trial,
            wandb_group=args.wandb_group,
            data_dir=args.data_dir,
            epochs=args.epochs,
            num_workers=args.num_workers,
        ),
        resources={"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial},
    )

    metric, mode = "best_corr", "max"
    search_space, points_to_evaluate = get_search_space(args.search_space)

    search_algo = OptunaSearch(
        space=search_space,
        metric=metric,
        mode=mode,
        points_to_evaluate=points_to_evaluate,
    )
    if args.restore is not None:
        assert os.path.isdir(args.restore)
        search_algo.restore_from_dir(args.restore)
        print(
            f"\nRestore search algorithm checkpoint {args.restore}\n"
            f"\tcompleted trials: {len(search_algo._completed_trials)}\n"
            f"\tbest_value: {search_algo._ot_study.best_value:.04f}\n"
        )

    progress_reporter = CLIReporter(
        metric_columns=["epoch", "elapse", "best_corr"],
        parameter_columns=["lr", "output_mode"],
        max_progress_rows=10,
        max_error_rows=3,
        max_column_length=15,
        max_report_frequency=30,
        print_intermediate_tables=True,
        sort_by_metric=True,
    )

    tuner = tune.Tuner(
        trainable,
        run_config=air.RunConfig(
            name=utils.get_timestamp(),
            failure_config=FailureConfig(max_failures=1),
            sync_config=None,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=0, checkpoint_at_end=False
            ),
            progress_reporter=progress_reporter,
            verbose=2,
        ),
        tune_config=tune.TuneConfig(
            search_alg=search_algo,
            scheduler=ASHAScheduler(
                time_attr="epoch",
                max_t=args.epochs + 1,
                grace_period=min(args.epochs, args.grace_period),
                reduction_factor=args.reduction_factor,
            ),
            num_samples=args.num_trials,
            metric=metric,
            mode=mode,
            reuse_actors=False,
            trial_name_creator=trial_name_creator,
            trial_dirname_creator=trial_dirname_creator,
        ),
    )

    results = tuner.fit()

    with open(os.path.join(output_dir, "results.pkl"), "wb") as file:
        pickle.dump(results, file)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--search_space", type=str, required=True)
    parser.add_argument("--restore", type=str, help="existing HPO run to restore from.")
    parser.add_argument(
        "--num_trials", type=int, required=True, help="number of HPO trials to perform."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/storage/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--storage_path",
        type=str,
        default="/home/storage/runs",
        help="RunConfig storage path.",
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=2,
        help="reduction factor for ASHA. a reduction factor of 4 means that "
        "only 1/4 of all trials are kept each time they are reduced",
    )
    parser.add_argument(
        "--grace_period",
        type=int,
        default=50,
        help="grace period before early stopping in ASHA.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--wandb_group", type=str, required=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of workers for PyTorch DataLoader",
    )
    parser.add_argument("--cpus_per_trial", type=int, default=16)
    parser.add_argument("--gpus_per_trial", type=int, default=1)

    main(parser.parse_args())
