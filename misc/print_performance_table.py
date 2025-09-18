"""
Helper script to print the performance (e.g. single trial correlation,
normalized correlation) of each model in a LaTeX table.

Please first run misc/evaluate_response.py to compute the metrics on a specific model.
"""

from pathlib import Path

import numpy as np
from scipy.stats import sem

from viv1t.data import SENSORIUM_OLD
from viv1t.data import STIMULUS_IDS
from viv1t.utils import yaml


def make_cell(result: list[float] | np.ndarray) -> str:
    # return (
    #     r"\begin{tabular}{@{}c@{}}$"
    #     + f"{np.mean(result):.03f}"
    #     + r"$ \\ $("
    #     + f"{sem(result):.03f}"
    #     + r")$\end{tabular}"
    # )
    return (
        r"\makecell{$"
        + f"{np.mean(result):.03f}"
        + r"$ \\ $("
        + f"{sem(result):.03f}"
        + r")$}"
    )


def print_table(models: dict[str, Path], metric: str):
    for model_name, output_dir in models.items():
        statement = f"{model_name}\n"
        results = yaml.load(output_dir / "evaluation_type.yaml")
        averages = []
        for stimulus_name in STIMULUS_IDS.keys():
            correlations = [
                results[stimulus_name][metric][mouse_id]
                for mouse_id in results[stimulus_name][metric].keys()
                if mouse_id in SENSORIUM_OLD
            ]
            statement += " & " + make_cell(correlations)
            averages.append(results[stimulus_name][metric]["average"])
        statement += " & " + r"\makecell{$" + f"{np.mean(averages):.03f}" + r"$}"
        statement += "\n"
        print(statement)


def main():
    models = {
        "LN": Path("../runs/fCNN/036_linear_fCNN"),
        "fCNN no Behavior": Path("../runs/fCNN/042_fCNN_no_behavior"),
        "fCNN": Path("../runs/fCNN/038_fCNN"),
        "DwiseNeuro": Path("../runs/lRomul"),
        "ViV1T no Behavior": Path("../runs/vivit/206_causal_viv1t_no_behavior"),
        "ViV1T": Path("../runs/vivit/204_causal_viv1t"),
    }

    print("Normalized correlation")
    print_table(models, metric="normalized_correlation")

    print("--------------------------------\n\nSingle trial correlation")
    print_table(models, metric="correlation")

    print("--------------------------------\n\nCorrelation to average")
    print_table(models, metric="correlation_to_average")


if __name__ == "__main__":
    main()
