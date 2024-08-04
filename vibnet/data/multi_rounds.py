import os
import pickle
from time import time
from pathlib import Path
from itertools import product, combinations

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

import vibnet.data.rounds_repo as round_pkg
from vibnet.config import ConfigSklearn
from vibnet.cli.common import group_class, set_deterministic
from vibnet.utils.sklearn import TrainDataset

__all__ = ["compute_combinations"]

TOTAL_SPLITS = 30


def picker(combs, n, seed):
    rng = np.random.default_rng(seed)

    def equal_folds(x1, x2):
        sample_x_folds = set(x1)
        sample_y_folds = set(x2)
        return len(sample_x_folds.intersection(sample_y_folds))

    selected_combs = [combs[0]]
    while len(selected_combs) < n:
        new_comb = combs[rng.choice(range(len(combs)))]
        if any([equal_folds(new_comb, selected) != 0 for selected in selected_combs]):
            continue
        else:
            selected_combs.append(new_comb)
    return selected_combs


def print_combinations(combs, dataset_name="CWRU", categorical_groups=False):
    if dataset_name == "PU":
        CLASS_DEF = {26: "N", 27: "O", 28: "I", 29: "R"}
        CONDITION_DEF = {"1000_15.0_0.7": "0", "1000_25.0_0.1": "1", "1000_25.0_0.7": "2", "400_25.0_0.7": "3"}
    elif dataset_name == "CWRU":
        CLASS_DEF = {0: "N", 1: "O", 2: "I", 3: "R"}
        CONDITION_DEF = {"0": "0", "1": "1", "2": "2", "3": "3"}
    elif dataset_name == "MFPT":
        CLASS_DEF = {23: "N", 25: "O", 24: "I"}

        class foo(dict):
            def __getitem__(self, key):
                return str(key)

        CONDITION_DEF = foo()

    print("Total combs: ", len(combs))
    folds = set()
    for r, c in enumerate(combs):
        print("round: ", r)
        for f, fold_groups in enumerate(c):
            print("fold: ", f, end=" -> ")
            for gp in fold_groups:
                if not categorical_groups:
                    gp = CLASS_DEF[int(gp.split(" ")[0])] + " " + CONDITION_DEF[gp.split(" ")[1]]
                print(f"{gp}", end=", ")
            print(" ", end="")
            fold_repr = "-".join(list(map(str, fold_groups)))
            if fold_repr in folds:
                print("REPEATED")
            else:
                print(f"=> {len(folds)}")
            folds.add(fold_repr)
        print()


def compute_combinations(y, groups, n_splits, n_repeats, seed):
    initial_states = {label: np.unique(groups[y == label]).tolist() for label in np.unique(y)}

    def custom_sort_key(gp):
        condition = gp.split(" ")[-1]
        returned = (
            f"{int(condition):07}"
            if "_" not in condition
            else "_".join([f"{float(c):05.2f}" for c in condition.split("_")])
        )
        return returned

    def englobe_all_data(round_comb):
        matrix = np.array(round_comb)
        n_groups = matrix.shape[0]
        n_labels = matrix.shape[1]
        return all([np.unique(matrix[:, col]).size == n_groups for col in range(n_labels)])

    initial_states = {label: sorted(groups, key=custom_sort_key) for label, groups in initial_states.items()}
    folds = list(product(*initial_states.values()))
    print("Total combinations of folds:", len(folds))

    start = time()
    round_combs_generator = combinations(folds, r=n_splits)
    round_combinations = list(round_combs_generator)
    print("Total combinations between folds", len(round_combinations))
    end = time()

    print("Time to generate combinations: {:.2f} seconds".format(end - start))
    valid_combs = list(filter(englobe_all_data, tqdm(round_combinations, desc="Filtering combinations")))
    print("Combinations that englobe all data from dataset:", len(valid_combs))

    choosen_round_combs = picker(valid_combs, n_repeats, seed)
    return choosen_round_combs


def is_filter_transform(transf):
    parameters = transf.get("parameters", None)
    return transf.get("name") == "FilterByValue" and parameters.get("on_field") == "sample_rate"


def main(cfg: Path, clear_cache: bool):
    data_divisions_repo = os.path.dirname(round_pkg.__file__)
    print(data_divisions_repo)

    config = ConfigSklearn(cfg)
    seed = config.get_yaml().get("seed", None)
    if clear_cache:
        config.clear_cache()
    set_deterministic(seed)

    dataset_config = config["dataset"]
    dataset_name = config["dataset"]["name"]

    dataset = config.get_deepdataset()
    # TODO: improve this form
    filter_transform = [
        str(transf["parameters"]["values"])
        for transf in config.get_yaml()["dataset"]["deep"]["transforms"]
        if is_filter_transform(transf)
    ]
    suffix = "_" + "_".join(filter_transform) if len(filter_transform) > 0 else ""

    groups = group_class(dataset_config)(dataset=dataset, config=config, custom_name=config["run_name"]).groups()
    categorical_groups = pd.Categorical(groups).codes  # noqa F841

    train_dataset = TrainDataset(dataset)
    _, y = train_dataset, train_dataset.targets

    n_splits = int(np.unique(groups).shape[0] / np.unique(y).shape[0])
    n_repeats = np.ceil(TOTAL_SPLITS / n_splits).astype(int)
    print("Per round splits: ", n_splits)
    print("Number of repeats: ", n_repeats)

    rounds_combinations = compute_combinations(y, groups, n_splits, n_repeats, seed)

    file_name = f"{dataset_name}{suffix}_rounds_combinations.pkl"
    pickle.dump(rounds_combinations, open(os.path.join(data_divisions_repo, file_name), "wb"))
    print(f"Multi Rounds data division saved in {file_name}")

    print("Multi Round combinations choosen: ")
    print_combinations(rounds_combinations, dataset_name, categorical_groups=False)
