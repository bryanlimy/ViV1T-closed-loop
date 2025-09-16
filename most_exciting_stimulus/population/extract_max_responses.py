from pathlib import Path

import numpy as np
from tqdm import tqdm

from viv1t import data
from viv1t.utils import h5

DATA_DIR = Path("../../data/sensorium")


def get_max_response(output_dir: Path, mouse_id: str) -> np.ndarray:
    tiers = data.get_tier_ids(data_dir=DATA_DIR, mouse_id=mouse_id)
    trial_ids = np.where(tiers != "none")[0]
    responses = h5.get(
        filename=output_dir / "responses" / f"mouse{mouse_id}.h5",
        trial_ids=trial_ids,
    )
    max_response = np.stack([np.max(response, axis=1) for response in responses])
    max_response = np.max(max_response, axis=0)
    return max_response


def main():
    models = {
        "recorded": DATA_DIR,
        "ViV1T": Path("../../runs/vivit/204_causal_viv1t"),
    }
    for model_name, output_dir in models.items():
        max_responses = {}
        for mouse_id in tqdm(data.SENSORIUM_OLD, desc=model_name):
            max_responses[mouse_id] = get_max_response(
                output_dir=output_dir, mouse_id=mouse_id
            )
        filename = f"{model_name}.npz"
        np.savez_compressed(filename, **max_responses)
        print(f"Saved max responses from {output_dir} to {filename}.")


if __name__ == "__main__":
    main()
