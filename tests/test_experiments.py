import re
import subprocess
from typing import Any, Dict, List

import pytest

EXPECTED_RESULTS = [
    {"config": "cfgs/1nn-example.yaml", "split": "biased_usual", "result": 0.84},
    {"config": "cfgs/1nn-example.yaml", "split": "biased_mirrored", "result": 0.84},
    {"config": "cfgs/1nn-example.yaml", "split": "unbiased", "result": 0.80},
    # KNN examples
    {"config": "cfgs/knn-example.yaml", "split": "biased_usual", "result": 0.88},
    {"config": "cfgs/knn-example.yaml", "split": "biased_mirrored", "result": 0.87},
    {"config": "cfgs/knn-example.yaml", "split": "unbiased", "result": 0.81},
    # Random Forest examples
    {"config": "cfgs/randomforest-example.yaml", "split": "biased_usual", "result": 0.97},
    {"config": "cfgs/randomforest-example.yaml", "split": "biased_mirrored", "result": 0.98},
    {"config": "cfgs/randomforest-example.yaml", "split": "unbiased", "result": 0.90},
]


@pytest.fixture()
def expected_results() -> List[Dict[str, Any]]:
    return EXPECTED_RESULTS


def _test_baseline(expected_results: List[Dict[str, Any]], config: str):
    target_results = list(filter(lambda x: x["config"] == config, expected_results))
    for metadata in target_results:
        cfg, split, expected_accuracy = metadata["config"], metadata["split"], metadata["result"]
        result = subprocess.run(
            ["vibnet", "baselines", "--cfg", cfg, "--split", split], stdout=subprocess.PIPE, text=True
        )
        stdout = result.stdout
        match = re.search(r"Balanced accuracy: (\d+\.\d+)", stdout)
        if match:
            accuracy = float(match.group(1))
            assert accuracy == pytest.approx(expected_accuracy, rel=1e-6), f"Failed for cfg={cfg} and split={split}"
        else:
            raise Exception(f"Accuracy value not found in output for cfg={cfg} and split={split}")


def test_1nn_baseline(expected_results: List[Dict[str, Any]]):
    _test_baseline(expected_results, "cfgs/1nn-example.yaml")


def test_knn_baseline(expected_results: List[Dict[str, Any]]):
    _test_baseline(expected_results, "cfgs/knn-example.yaml")


def test_randomforest_baseline(expected_results: List[Dict[str, Any]]):
    _test_baseline(expected_results, "cfgs/randomforest-example.yaml")
