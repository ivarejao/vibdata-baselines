import os
import pickle
from typing import Any, Dict

from .create import compute_combinations

__all__ = ["compute_combinations", "load_combinations"]


def load_combinations(dataset_config: Dict[str, Any]):
    def is_filter_transform(transf: Dict[str, Any]):
        parameters = transf.get("parameters", None)
        return transf.get("name") == "FilterByValue" and parameters.get("on_field") == "sample_rate"

    # Identify which dataset is being used
    dataset_name = dataset_config["name"]
    transforms = dataset_config["deep"]["transforms"]
    filter_transform = [str(transf["parameters"]["values"]) for transf in transforms if is_filter_transform(transf)]
    suffix = "_" + "_".join(filter_transform) if len(filter_transform) > 0 else ""
    # Then load the respectively file with the data division for multi round cross validation
    file_name = f"{dataset_name}{suffix}_rounds_combinations.pkl"
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    print(f"Loading Multi Round combinations from {file_path}")
    return pickle.load(open(file_path, "rb"))
