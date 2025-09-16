from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from viv1t import data
from viv1t.utils import plot
from viv1t.utils import utils

plot.set_font()

TICK_FONTSIZE = 6
LABEL_FONTSIZE = 7
TITLE_FONTSIZE = 7

DPI = 400
FORMAT = "pdf"
PAPER_WIDTH = 5.1666  # width of the paper in inches

BLOCK_SIZE = 25

H_INTERVAL = 25  # horizontal distance interval
V_INTERVAL = 50  # vertical distance interval
SI_THRESHOLD = 0.2

DATA_DIR = Path("../data/sensorium")


def compute_distance(a: np.ndarray, b: np.ndarray) -> (float, float):
    """
    compute horizontal (Euclidean) and vertical (absolute) distance between
    two 3D points
    """
    assert a.shape == b.shape == (3,)
    h = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    v = abs(a[2] - b[2])
    return h, v


def compute_tuning_similarity(
    save_dir: Path,
    reference: Path,
    mouse_id: str,
    tuning_type: str,
) -> pd.DataFrame:
    tuning = utils.load_tuning(save_dir, mouse_id=mouse_id)
    tuning_curves = tuning["tuning_curve"]
    if tuning_type == "orientation":
        tuning_curves = (tuning_curves[:, :4] + tuning_curves[:, 4:]) / 2
    neuron_coordinates = data.get_neuron_coordinates(mouse_id=mouse_id)

    # filter out non-selective neurons based on OSI of recorded neurons
    selective_neurons = utils.get_selective_neurons(
        save_dir=reference,
        mouse_id=mouse_id,
        threshold=SI_THRESHOLD,
        tuning_type=tuning_type,
    )

    depths = np.unique(neuron_coordinates[:, 2])  # use neurons from all depths
    print(f"consider depths: {depths}")
    neurons = np.where(np.isin(neuron_coordinates[:, 2], depths))[0]

    selective_neurons = np.intersect1d(selective_neurons, neurons)

    # compute correlation coefficient of tuning curves for all neuron pairs
    tuning_similarities = np.corrcoef(tuning_curves, dtype=np.float32)

    # get all combinations of selective neuron pairs
    neuron_pairs = np.array(list(combinations(np.sort(selective_neurons), r=2)))
    print(
        f"number of neurons: {len(selective_neurons)}, "
        f"number of pairs: {len(neuron_pairs)}"
    )

    results = defaultdict(lambda: defaultdict(list))
    for n1, n2 in tqdm(neuron_pairs, desc=f"mouse {mouse_id}"):
        delta_h, delta_v = compute_distance(
            neuron_coordinates[n1], neuron_coordinates[n2]
        )
        if delta_h >= 7 * H_INTERVAL:
            continue
        delta_h = H_INTERVAL * (round(delta_h / H_INTERVAL) + 1)
        delta_v = V_INTERVAL * round(delta_v / V_INTERVAL)
        results[delta_v][delta_h].append(tuning_similarities[n1, n2])

    df = pd.DataFrame(results).sort_index(axis=0).sort_index(axis=1)
    # df = df.map(np.stack)
    df = df.map(lambda a: [] if np.isnan(a).any() else np.stack(a))

    print(
        f"number of pairs in each (column, d) vertical and (row, Δ) horizontal "
        f"distance group:\n{df.map(len)}"
    )

    df = pd.melt(
        df.reset_index(),
        id_vars=["index"],
        var_name="v",
        value_name="tuning_similarity",
    )
    df = df.explode("tuning_similarity")
    df.rename(columns={"index": "d"}, inplace=True)
    df["tuning_similarity"] = df["tuning_similarity"].astype(np.float32)

    return df


def estimate_tuning_similarity(
    output_dir: Path,
    mouse_ids: list[str],
    tuning_type: str,
) -> pd.DataFrame:
    assert output_dir.exists(), f"{output_dir} does not exist."
    tuning_similarities = []
    for i, mouse_id in enumerate(mouse_ids):
        print(f"\nProcessing mouse {mouse_id} {tuning_type}...")
        tuning_similarity = compute_tuning_similarity(
            save_dir=output_dir,
            reference=data.METADATA_DIR,
            # reference=output_dir,
            # reference=Path("../runs/vivit/204_causal_viv1t"),
            mouse_id=mouse_id,
            tuning_type=tuning_type,
        )
        tuning_similarity.insert(loc=0, column="mouse", value=mouse_id)
        tuning_similarities.append(tuning_similarity)
    tuning_similarities = pd.concat(tuning_similarities, ignore_index=True)
    tuning_similarities.insert(loc=1, column="tuning_type", value=tuning_type)
    return tuning_similarities


def main():
    models = {
        "recorded": data.METADATA_DIR,
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }
    mouse_ids = ["B", "C", "E"]
    tuning_types = ["direction", "orientation"]

    for model_name, output_dir in models.items():
        print(f"\nProcess {model_name} responses from {output_dir}...")
        tuning_similarities = []
        for tuning_type in tuning_types:
            tuning_similarity = estimate_tuning_similarity(
                output_dir=output_dir,
                mouse_ids=mouse_ids,
                tuning_type=tuning_type,
            )
            tuning_similarities.append(tuning_similarity)
        tuning_similarities = pd.concat(tuning_similarities, ignore_index=True)
        filename = output_dir / "tuning" / "tuning_similarity.parquet"
        tuning_similarities.to_parquet(filename)
        print(f"Saved tuning similarity to {filename}")


if __name__ == "__main__":
    main()
