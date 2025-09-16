import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from viv1t.data import utils
from viv1t.data.constants import *
from viv1t.data.statistics import compute_response_precision
from viv1t.data.statistics import load_stats

TENSOR = np.ndarray | torch.Tensor
SAMPLE = dict[str, int | str | TENSOR]


def get_mouse_ids(args):
    """Retrieve the mouse IDs when args.mouse_ids is not provided"""
    if args.mouse_ids is None:
        # by default, train on all Sensorium 2023 mice.
        args.mouse_ids = SENSORIUM_OLD + SENSORIUM_NEW
    # check if mouse_id exists
    for mouse_id in args.mouse_ids:
        assert mouse_id in MOUSE_IDS.keys()
        mouse_dir = args.data_dir / MOUSE_IDS[mouse_id]
        assert (
            mouse_dir.exists()
        ), f"Cannot find data folder for mouse {mouse_id} in {mouse_dir}."


def set_neuron_idx(args):
    """Set the neuron idx for mouse, randomly select neuron_ids if limit_neurons is set"""
    args.neuron_idx = {}
    for mouse_id in args.mouse_ids:
        num_neurons = utils.get_num_neurons(mouse_id)
        if hasattr(args, "limit_neurons") and args.limit_neurons is not None:
            rng = np.random.RandomState(args.seed)
            neuron_idx = rng.choice(
                num_neurons, size=min(num_neurons, args.limit_neurons), replace=False
            )
            if args.verbose:
                print(f"Limit mouse {mouse_id} to {len(neuron_idx)} neurons.")
        else:
            neuron_idx = np.arange(num_neurons)
        args.neuron_idx[mouse_id] = np.sort(neuron_idx)


def load_mouse_metadata(mouse_id: str, data_dir: Path):
    """
    Load the relevant metadata of a specific mouse
    Args:
        mouse_id: str, mouse ID
        data_dir: Path, path to the data directory
    Returns:
        metadata: Dict[str, Any]
            num_neurons: int, number of neurons
            neuron_coordinates: np.ndarray, (x, y, z) coordinates of each
                neuron in the cortex
            neuron_ids: np.ndarray, neuron IDs
            tiers: np.ndarray, `train`, `validation`, `live_main`,
                `live_bonus`, `final_main`, `final_bonus`, `test`, `none`
            video_ids: np.ndarray, unique video IDs of the stimuli
            statistics: t.Dict[str, t.Dict[str, np.ndarray]], the
                statistics (min, max, median, mean, std) of the training data
    """
    mouse_dir = data_dir / MOUSE_IDS[mouse_id]
    if not mouse_dir.is_dir():
        utils.unzip(filename=mouse_dir.with_suffix(".zip"), unzip_dir=data_dir)
    meta_dir = mouse_dir / "meta"

    tiers = utils.get_tier_ids(data_dir, mouse_id=mouse_id)

    neuron_dir = meta_dir / "neurons"
    neuron_coordinates = utils.get_neuron_coordinates(mouse_id=mouse_id)
    neuron_ids = np.load(neuron_dir / "unit_ids.npy").astype(int)

    video_ids = utils.get_video_ids(mouse_id)
    stimulus_ids = utils.get_stimulus_ids(mouse_id)

    return {
        "num_neurons": len(neuron_coordinates),
        "neuron_coordinates": neuron_coordinates,
        "neuron_ids": neuron_ids,
        "tiers": tiers,
        "video_ids": video_ids,
        "stimulus_ids": stimulus_ids,
        "stats": load_stats(mouse_id),
    }


class MovieDataset(Dataset):
    """
    MoveDataset class for loading data from a single mouse

    Notable attributes:
    - tier: str, the tier of the dataset
    - video_stats: Dict[str, np.ndarray], the statistics of the video data
    - response_stats: Dict[str, np.ndarray], the statistics of the response data
    - behavior_stats: Dict[str, np.ndarray], the statistics of the behavior data
    - pupil_center_stats: Dict[str, np.ndarray], the statistics of the pupil center data
    - num_neurons: int, number of neurons
    - neuron_coordinates: np.ndarray, (x, y, z) coordinates of each neuron
    - neuron_ids: np.ndarray, neuron IDs
    - trials: np.ndarray, the trial IDs
    - max_frame: int, the maximum number of frames the dataset returns
    - hidden_response: bool, whether the recorded responses are hidden (zeros)
    """

    def __init__(
        self,
        tier: str,
        data_dir: Path,
        mouse_id: str,
        transform_input: int,
        transform_output: int,
        crop_frame: int = -1,
        center_crop: float = 1.0,
        limit_data: int | None = None,
        neuron_idx: np.ndarray | list[int] | None = None,
        random_seed: int = 1234,
        verbose: int = 0,
    ):
        """
        Construct Movie Dataset

        Args:
            tier: one of 'train', 'validation', 'live_test_main', 'live_test_bonus',
             'final_test_main', 'final_test_bonus'
            data_dir: path to where all data are stored
            mouse_id: the mouse ID
            transform_input: input transformation using statistics from training set
                0: no transformation
                1: standardize input
                2: normalize input
            transform_output: output transformation using statistics from training set
                0: no transformation
                1: standardize output
                2: normalize output
            crop_frame: number of frames to take from each trial, set to -1
                to use all available frames
            limit_data: limit the number of samples, set None to use all
                available trials
        """
        self.tier = tier
        assert transform_input in (0, 1, 2) and transform_output in (0, 1, 2)
        self.transform_input, self.transform_output = transform_input, transform_output
        assert crop_frame == -1 or crop_frame > 50
        self.crop_frame = crop_frame
        assert 0.0 < center_crop <= 1.0
        self.center_crop = center_crop

        self.verbose = verbose

        self.mouse_id = mouse_id
        self.mouse_dir = data_dir / MOUSE_IDS[mouse_id]
        metadata = load_mouse_metadata(mouse_id=mouse_id, data_dir=data_dir)

        self.max_frame = MAX_FRAME
        assert self.crop_frame <= self.max_frame
        self.eps = torch.finfo(torch.float32).eps

        self.num_neurons = metadata["num_neurons"]
        self.neuron_ids = metadata["neuron_ids"]
        self.neuron_coordinates = torch.from_numpy(metadata["neuron_coordinates"])
        self.store_stats(metadata["stats"])

        self.neuron_idx = neuron_idx
        if self.neuron_idx is None:
            self.neuron_idx = list(range(self.num_neurons))
        else:
            self.select_neurons()

        self.tiers = metadata["tiers"]
        self.select_trials(
            metadata["tiers"], limit_data=limit_data, random_seed=random_seed
        )
        self.video_ids = torch.from_numpy(metadata["video_ids"])
        self.stimulus_ids = torch.from_numpy(metadata["stimulus_ids"])

        # get data dimensions
        sample = utils.load_trial(self.mouse_dir, trial_id=self.trial_ids[0])
        self.video_shape = sample["video"].shape[-2:]
        self.num_channels = sample["video"].shape[0]
        self.hidden_response = not np.any(sample["response"])

        del sample, metadata

        if self.center_crop < 1:
            self.prepare_center_crop()

    def prepare_center_crop(self):
        in_h, in_w = self.video_shape
        crop_h = int(in_h * self.center_crop)
        crop_w = int(in_w * self.center_crop)
        crop_scale = self.center_crop
        h_pixels = torch.linspace(-crop_scale, crop_scale, crop_h)
        w_pixels = torch.linspace(-crop_scale, crop_scale, crop_w)
        mesh_y, mesh_x = torch.meshgrid(h_pixels, w_pixels, indexing="ij")
        # grid_sample uses (x, y) coordinates
        grid = torch.stack((mesh_x, mesh_y), dim=2)
        self.grid = grid.unsqueeze(0)
        self.resize = transforms.Resize(size=(in_h, in_w), antialias=False)

    def select_neurons(self):
        # update neuron related information based on neuron_idx
        self.num_neurons = len(self.neuron_idx)
        self.neuron_ids = self.neuron_ids[self.neuron_idx]
        self.neuron_coordinates = self.neuron_coordinates[self.neuron_idx]
        self.response_precision = self.response_precision[self.neuron_idx]
        self.response_stats = {
            k: v[self.neuron_idx] for k, v in self.response_stats.items()
        }

    def select_trials(
        self, tiers: np.ndarray, limit_data: int | None = None, random_seed: int = 1234
    ):
        trial_ids = np.where(tiers == self.tier)[0]
        assert trial_ids.size, f"No trial for mouse {self.mouse_id} {self.tier} set."
        trial_ids = trial_ids.astype(np.int32)
        if limit_data is not None and len(trial_ids) > limit_data:
            # randomly select limit_data number of samples in the training set
            rng = np.random.default_rng(seed=random_seed)
            trial_ids = rng.choice(trial_ids, size=limit_data, replace=False)
            trial_ids = np.sort(trial_ids)
            if self.verbose:
                print(
                    f"Limit mouse {self.mouse_id} {self.tier} to "
                    f"{limit_data} samples."
                )
        self.trial_ids = torch.from_numpy(trial_ids)

    def __len__(self):
        return len(self.trial_ids)

    def store_stats(self, stats: dict[str, dict[str, np.ndarray]]):
        self.video_stats = {k: torch.from_numpy(v) for k, v in stats["video"].items()}
        self.response_stats = {
            k: torch.from_numpy(v)[:, None] for k, v in stats["response"].items()
        }
        self.behavior_stats = {
            k: torch.from_numpy(v)[:, None] for k, v in stats["behavior"].items()
        }
        self.pupil_center_stats = {
            k: torch.from_numpy(v)[:, None] for k, v in stats["pupil_center"].items()
        }
        self.response_precision = compute_response_precision(self.response_stats["std"])

    def crop_image(self, video: TENSOR):
        if self.center_crop < 1:
            video = F.grid_sample(video, self.grid, mode="nearest", align_corners=True)
            video = self.resize(video)
        return video

    def transform_video(self, video: torch.Tensor):
        stats = self.video_stats
        match self.transform_input:
            case 1:
                video = (video - stats["mean"]) / (stats["std"] + self.eps)
            case 2:
                video = (video - stats["min"]) / (stats["max"] - stats["min"])
        if self.center_crop < 1:
            video = self.crop_image(video)
        return video

    def i_transform_videos(self, videos: torch.Tensor):
        assert videos.ndim == 5
        stats = self.video_stats
        match self.transform_input:
            case 1:
                videos = videos * (stats["std"] + self.eps) + stats["mean"]
            case 2:
                videos = videos * (stats["max"] - stats["min"]) + stats["min"]
        return videos

    def transform_behavior(self, behavior: torch.Tensor):
        stats = self.behavior_stats
        match self.transform_input:
            case 1:
                behavior = behavior / stats["std"]
            case 2:
                behavior = (behavior - stats["min"]) / (stats["max"] - stats["min"])
        return behavior

    def i_transform_behaviors(self, behaviors: torch.Tensor):
        assert behaviors.ndim == 3
        stats = self.behavior_stats
        match self.transform_input:
            case 1:
                behaviors = behaviors * stats["std"]
            case 2:
                behaviors = behaviors * (stats["max"] - stats["min"]) + stats["min"]
        return behaviors

    def transform_pupil_center(self, pupil_center: torch.Tensor):
        stats = self.pupil_center_stats
        match self.transform_input:
            case 1:
                pupil_center = (pupil_center - stats["mean"]) / (
                    stats["std"] + self.eps
                )
            case 2:
                pupil_center = (pupil_center - stats["min"]) / (
                    stats["max"] - stats["min"]
                )
        return pupil_center

    def i_transform_pupil_centers(self, pupil_centers: torch.Tensor):
        assert pupil_centers.ndim == 3
        stats = self.pupil_center_stats
        match self.transform_input:
            case 1:
                pupil_centers = (
                    pupil_centers * (stats["std"] + self.eps) + stats["mean"]
                )
            case 2:
                pupil_centers = (
                    pupil_centers * (stats["max"] - stats["min"]) + stats["min"]
                )
        return pupil_centers

    def transform_response(self, response: torch.Tensor):
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response * self.response_precision
            case 2:
                response = (response - stats["min"]) / (stats["max"] - stats["min"])
            case _:
                # ensure response is non-negative
                response = torch.clamp(response, min=0)
        return response

    def i_transform_response(self, response: torch.Tensor):
        stats = self.response_stats
        match self.transform_output:
            case 1:
                response = response / self.response_precision
            case 2:
                response = response * (stats["max"] - stats["min"]) + stats["min"]
        return response

    def load_sample(self, trial_id: int | str | torch.Tensor, to_tensor: bool = False):
        """
        Load sample from disk and apply transformation

        The Sensorium 2023 challenge only consider the first 300 frames, even
        though some trials have more than 300 frames
        """
        sample = utils.load_trial(
            self.mouse_dir, trial_id=trial_id, to_tensor=to_tensor
        )
        # crop to max frames if trial is longer
        t = sample["duration"] = min(self.max_frame, sample["duration"])
        sample["video"] = sample["video"][:, :t]
        sample["response"] = sample["response"][self.neuron_idx, :t]
        sample["behavior"] = sample["behavior"][:, :t]
        sample["pupil_center"] = sample["pupil_center"][:, :t]

        sample["video"] = self.transform_video(sample["video"])
        sample["behavior"] = self.transform_behavior(sample["behavior"])
        sample["pupil_center"] = self.transform_pupil_center(sample["pupil_center"])
        sample["response"] = self.transform_response(sample["response"])
        return sample

    def random_crop(self, sample: dict[str, TENSOR], crop_frame: int):
        """randomly crop the trial to crop_frame in training set"""
        start, frame_diff = 0, sample["video"].shape[1] - crop_frame
        if self.tier == "train" and frame_diff > 0:
            start = np.random.randint(0, frame_diff)
        sample["video"] = sample["video"][:, start : start + crop_frame, ...]
        sample["response"] = sample["response"][:, start : start + crop_frame]
        sample["behavior"] = sample["behavior"][:, start : start + crop_frame]
        sample["pupil_center"] = sample["pupil_center"][:, start : start + crop_frame]

    def __getitem__(self, idx: int | torch.Tensor, to_tensor: bool = True):
        """Return a sample

        Returns
            sample
                video: the movie stimulus in (C, T, H, W)
                response: the corresponding response in (N, T)
                behavior: pupil size and locomotive speed in (2, T)
                pupil_center:  pupil center(x, y) coordinates  in (2, T)
                mouse_id: the mouse ID
                trial_id:  the trial ID
                video_id: the video ID
                stimulus_id: the stimulus ID
        """
        trial_id = self.trial_ids[idx]
        sample = self.load_sample(trial_id, to_tensor=to_tensor)
        if self.crop_frame != -1 and sample["duration"] > self.crop_frame:
            self.random_crop(sample, crop_frame=self.crop_frame)
        sample["mouse_id"] = self.mouse_id
        sample["trial_id"] = trial_id
        sample["video_id"] = self.video_ids[trial_id]
        sample["stimulus_id"] = self.stimulus_ids[trial_id]
        sample["tier"] = str(self.tiers[trial_id])
        del sample["duration"]
        return sample


def get_training_ds(
    args,
    data_dir: Path,
    mouse_ids: list[str],
    batch_size: int = 1,
    val_batch_size: int | None = None,
    device: torch.device = torch.device("cpu"),
    num_workers: int = None,
):
    """
    Get DataLoaders for training
    Args:
        args
        data_dir: path to directory where the zip files are stored
        mouse_ids: mouse IDs to extract
        batch_size: batch size of the DataLoaders
        device: torch device
        num_workers: number of workers for DataLoader, use args.num_workers if None.
    Return:
        train_ds: DataLoaders of the training sets.
        val_ds: DataLoaders of the validation sets.
        test_ds: DataLoaders of the main and bonus live test set if they are available.
    """
    dataset_kwargs = {
        "data_dir": data_dir,
        "transform_input": args.transform_input,
        "transform_output": args.transform_output,
        "center_crop": args.center_crop if hasattr(args, "center_crop") else 1.0,
        "random_seed": args.seed,
        "verbose": args.verbose,
    }
    loader_kwargs = utils.get_dataloader_kwargs(
        args, device=device, num_workers=num_workers
    )
    if val_batch_size is None:
        val_batch_size = batch_size

    train_ds, val_ds, test_ds = {}, {}, {}
    for mouse_id in mouse_ids:
        dataset_kwargs["neuron_idx"] = (
            args.neuron_idx[mouse_id] if hasattr(args, "neuron_idx") else None
        )
        train_ds[mouse_id] = DataLoader(
            MovieDataset(
                tier="train",
                mouse_id=mouse_id,
                crop_frame=args.crop_frame,
                limit_data=args.limit_data,
                **dataset_kwargs,
            ),
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_ds[mouse_id] = DataLoader(
            MovieDataset(tier="validation", mouse_id=mouse_id, **dataset_kwargs),
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if mouse_id in SENSORIUM_OLD:
            # batch size must be 1 for the test sets
            if not test_ds:
                test_ds = {"live_main": {}, "live_bonus": {}}
            test_ds["live_main"][mouse_id] = DataLoader(
                MovieDataset(tier="live_main", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            test_ds["live_bonus"][mouse_id] = DataLoader(
                MovieDataset(tier="live_bonus", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
        elif mouse_id in ROCHEFORT_LAB:
            if not test_ds:
                test_ds = {"live_main": {}}
            test_ds["live_main"][mouse_id] = DataLoader(
                MovieDataset(tier="live_main", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )

    utils.set_shapes(args, ds=train_ds)
    args.max_frame = train_ds[mouse_ids[0]].dataset.max_frame

    args.train_sizes = {m: len(ds.dataset) for m, ds in train_ds.items()}

    return train_ds, val_ds, test_ds


def get_submission_ds(
    args,
    data_dir: Path,
    mouse_ids: list[str],
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
    num_workers: int = None,
):
    """
    Get DataLoaders for submission
    Args:
        args
        data_dir: path to directory where the zip files are stored
        mouse_ids: mouse IDs to extract
        batch_size: batch size of the DataLoaders
        device: torch device
        num_workers:  number of workers for DataLoader
    Return:
        val_ds: Dict[str, DataLoader], dictionary of DataLoaders of the
            validation sets where keys are the mouse IDs.
        test_ds: Dict[str, Dict[str, DataLoader]]
            live_main: dictionary of DataLoaders of the live main test sets
                where keys are the mouse IDs.
            live_bonus: dictionary of DataLoaders of the live bonus test sets
                where keys are the mouse IDs.
            final_main: dictionary of DataLoaders of the final main test sets
                where keys are the mouse IDs.
            final_bonus: dictionary of DataLoaders of the final bonus test sets
                where keys are the mouse IDs.
    """
    dataset_kwargs = {
        "data_dir": data_dir,
        "transform_input": args.transform_input,
        "transform_output": args.transform_output,
        "center_crop": args.center_crop if hasattr(args, "center_crop") else 1.0,
        "random_seed": args.seed,
        "verbose": args.verbose,
    }
    loader_kwargs = utils.get_dataloader_kwargs(
        args, device=device, num_workers=num_workers
    )

    val_ds = {}
    test_ds = {"live_main": {}, "live_bonus": {}, "final_main": {}, "final_bonus": {}}
    for mouse_id in mouse_ids:
        dataset_kwargs["neuron_idx"] = (
            args.neuron_idx[mouse_id] if hasattr(args, "neuron_idx") else None
        )
        val_ds[mouse_id] = DataLoader(
            MovieDataset(tier="validation", mouse_id=mouse_id, **dataset_kwargs),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if mouse_id in SENSORIUM:
            test_ds["live_main"][mouse_id] = DataLoader(
                MovieDataset(tier="live_main", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            test_ds["live_bonus"][mouse_id] = DataLoader(
                MovieDataset(tier="live_bonus", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            test_ds["final_main"][mouse_id] = DataLoader(
                MovieDataset(tier="final_main", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            test_ds["final_bonus"][mouse_id] = DataLoader(
                MovieDataset(tier="final_bonus", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
        elif mouse_id in ROCHEFORT_LAB:
            if not test_ds:
                test_ds = {"live_main": {}}
            test_ds["live_main"][mouse_id] = DataLoader(
                MovieDataset(tier="live_main", mouse_id=mouse_id, **dataset_kwargs),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )

    utils.set_shapes(args, ds=val_ds)
    args.max_frame = val_ds[mouse_ids[0]].dataset.max_frame

    return val_ds, test_ds
