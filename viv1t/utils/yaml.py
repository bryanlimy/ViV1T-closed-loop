from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

yaml.representer.ignore_aliases = lambda *_data: True


def array2py(data: dict):
    """
    Recursively replace np.ndarray and torch.Tensor variables in data with
    Python integer, float or list.
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cpu().numpy().tolist()
        elif isinstance(v, torch.device):
            data[k] = v.type
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
            data[k] = float(v)
        elif isinstance(v, np.integer):
            data[k] = int(v)
        elif isinstance(v, list) and isinstance(v[0], Path):
            data[k] = [str(p) for p in v]
        elif isinstance(v, Path):
            data[k] = str(v)
        elif isinstance(v, dict):
            array2py(data[k])


def load(filename: Path):
    """Load yaml file"""
    with open(filename, "r") as file:
        data = yaml.load(file)
    return data


def save(filename: Path, data: dict):
    """Save data dictionary to yaml file"""
    assert type(data) == dict
    array2py(data)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as file:
        yaml.dump(data, file)


def update(filename: Path, data: dict):
    """Update json file with filename with items in data"""
    content = {}
    if filename.is_file():
        content = load(filename)
    for k, v in data.items():
        content[k] = v
    save(filename, data=content)
