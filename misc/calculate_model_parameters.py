"""
Calculate the number of parameters in a given model checkpoint
"""

from pathlib import Path

import torch


def process_model(model_name: str, filename: Path):
    ckpt = torch.load(filename, map_location="cpu")
    if model_name == "DwiseNeuro":
        ckpt = ckpt["nn_state_dict"]
    size = 0
    for k, v in ckpt.items():
        size += v.size().numel()
    print(f"{model_name} number of parameters: {size}")


def main():
    models = {
        "LN": Path("../runs/fCNN/036_linear_fCNN/ckpt/model.pt"),
        "fCNN": Path("../runs/fCNN/038_fCNN/ckpt/model.pt"),
        "DwiseNeuro": Path("../runs/lRomul/model-017-0.262910.pth"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t/ckpt/model.pt"),
    }
    for model_name, filename in models.items():
        process_model(model_name=model_name, filename=filename)


if __name__ == "__main__":
    main()
