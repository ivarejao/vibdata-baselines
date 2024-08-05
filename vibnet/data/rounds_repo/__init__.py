import os
import pickle
from typing import Any, Dict

__all__ = ["load_combinations", "is_multi_round_dataset"]


def _get_combinations_file_name(dataset_config: Dict[str, Any]) -> str:
    def is_filter_transform(transf: Dict[str, Any]):
        parameters = transf.get("parameters", None)
        return transf.get("name") == "FilterByValue" and parameters.get("on_field") == "sample_rate"

    dataset_name = dataset_config["name"]
    transforms = dataset_config["deep"]["transforms"]
    filter_transform = [str(transf["parameters"]["values"]) for transf in transforms if is_filter_transform(transf)]
    suffix = "_" + "_".join(filter_transform) if len(filter_transform) > 0 else ""
    file_name = f"{dataset_name}{suffix}_rounds_combinations.pkl"
    return file_name


def load_combinations(dataset_config: Dict[str, Any]):
    # Identify which dataset is being used
    file_name = _get_combinations_file_name(dataset_config)
    # Then load the respectively file with the data division for multi round cross validation
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Combinations for dataset {dataset_config['name']} not found")
    print(f"Loading Multi Round combinations from {file_path}")
    return pickle.load(open(file_path, "rb"))


def is_multi_round_dataset(dataset_config: Dict[str, Any]):
    basedir = os.path.dirname(__file__)
    combinations_files = set(
        f
        for f in os.listdir(basedir)
        if os.path.isfile(os.path.join(basedir, f)) and f.endswith("_rounds_combinations.pkl")
    )
    supposed_dataset_file = _get_combinations_file_name(dataset_config)
    return supposed_dataset_file in combinations_files
