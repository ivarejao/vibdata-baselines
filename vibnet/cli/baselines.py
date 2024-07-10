from datetime import datetime
import os
import pickle
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from vibdata.deep.DeepDataset import DeepDataset

import vibnet.data.group_dataset as groups_module
from vibnet.config import ConfigSklearn
from vibnet.cli.common import Split
from vibnet.data.group_dataset import GroupMirrorBiased

def classifier_biased(cfg: ConfigSklearn, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
    seed = cfg["seed"]
    num_folds = len(set(groups))

    cv_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    clf = cfg.get_estimator(grid_serch_cv=cv_inner)

    y_pred = cross_val_predict(clf, inputs, labels, cv=cv_outer)

    return y_pred


def classifier_predefined(cfg: ConfigSklearn, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
    cv_outer = LeaveOneGroupOut()
    cv_inner = LeaveOneGroupOut()

    clf = cfg.get_estimator(grid_serch_cv=cv_inner)

    fit_params = {"classifier__groups": groups} if "params_grid" in cfg else None

    y_pred = cross_val_predict(clf, inputs, labels, groups=groups, cv=cv_outer, fit_params=fit_params)

    return y_pred

def round_cv(cfg: ConfigSklearn, inputs: List[int], labels: List[int], groups: List[int], metainfo: pd.DataFrame) -> List[int]:

    from vibnet.data.round_cv import NewLogo, RepeteadNewLogo
    from sklearn.model_selection import cross_validate

    NUM_REPEATS = 30
    actual_datetime = datetime.now()

    # num_folds_per_dataset = {"CWRU": lambda df: df["load"].nunique(), 
    #        "MFPT": lambda df: df[["load", "label"]].drop_duplicates().groupby("label").count()["load"].mean(), 
    #        "PU": lambda df: df[["radial_force_n", "rotation_hz", "load_nm"]].drop_duplicates().shape[0]}
    # folds_per_round = num_folds_per_dataset[cfg["dataset"]["name"]](metainfo)

    folds_per_round = np.unique(groups).shape[0] / np.unique(labels).shape[0]  # folds_per_round = total_groups / total_labels = num_conditions
    num_repeats = np.ceil(NUM_REPEATS / folds_per_round).astype(int)
    print("Num_repeat: ", num_repeats)
    print("Folds per round: ", folds_per_round)

    cv_outer = RepeteadNewLogo(n_repeats=num_repeats, random_state=cfg["seed"], y=labels, groups=groups)
    cv_inner = NewLogo(shuffle=True, random_state=cfg["seed"], combinations=cv_outer.combinations)

    clf = cfg.get_estimator(grid_serch_cv=cv_inner)

    fit_params = {"classifier__groups": groups} if "params_grid" in cfg else None

    # Currently, the `cross_val_predict` does not support Cross-Validation with Rounds
    # https://github.com/scikit-learn/scikit-learn/issues/16135#issue-550525513
    stats = cross_validate(clf, inputs, labels, groups=groups, cv=cv_outer, fit_params=fit_params, scoring=["f1_macro", "balanced_accuracy"],
                           return_indices=True, error_score="raise", verbose=2)

    print("Num splits: ", len(stats["indices"]["test"]))
    # TODO: Change format back to csv
    filename = f"results-baselines-{cfg.config.get('run_name', None)}-{actual_datetime.isoformat()}.pkl"

    # pd.DataFrame(stats).to_csv(filename, index=False)
    pickle.dump(stats, open(filename, "wb"))

    print(f"Mean F1 macro: {np.mean(stats['test_f1_macro']):.2f}")
    print(f"Mean Balanced accuracy: {np.mean(stats['test_balanced_accuracy']):.2f}")
    print(f"Stats saved in {filename}")

    # return y_pred


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int]) -> None:
    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    print(f"{classification_report(y_true, y_pred, target_names=labels)}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.2f}")


def configure_wandb(run_name: str, cfg: ConfigSklearn, cfg_path: str, groups: List[int], split: Split) -> None:
    wandb.login(key=os.environ["WANDB_KEY"])
    config = {
        "model": cfg["model"]["name"],
        "folds": len(set(groups)),
        "split": split.value,
    }
    config["params_grid"] = cfg["params_grid"] if "params_grid" in cfg else None

    wandb.init(
        # Set the project where this run will be logged
        project=os.environ["WANDB_PROJECT"],
        # Track essentials hyperparameters and run metadata
        config=config,
        # Set the name of the experiment
        name=run_name,
    )
    # Add configuration file into the wandb log
    wandb.save(cfg_path, policy="now")


def main(cfg: Path, split: Split):
    load_dotenv()
    config = ConfigSklearn(cfg)
    # config.clear_cache()

    dataset_name = config["dataset"]["name"]
    dataset = config._get_dataset_deep()

    # group_obj = getattr(groups_module, "Group" + dataset_name)(dataset=dataset, config=config)
    # groups = group_obj.groups()
    from vibnet.data.round_cv import GroupRepeatedCWRU48k
    import pandas as pd
    group_obj = GroupRepeatedCWRU48k(dataset=dataset, config=config, custom_name=config["run_name"])
    groups = group_obj.groups()
    groups_str = groups.copy()  # TODO: remove
    groups = pd.Categorical(groups).codes

    # if split is Split.biased_mirrored:
    #     groups = GroupMirrorBiased(dataset=dataset, config=config, custom_name=config["run_name"]).groups(groups)

    # configure_wandb(config["run_name"], config, cfg, groups, split)

    X, y = config.get_dataset()
    # from vibnet.data.round_cv import _compute_combinations
    # CLASS_DEF = [-1, "I", "R", "O"]
    # combs = [c for c in _compute_combinations(y, groups)]
    # print("Total combs: ", len(combs))
    # for i in range(4):
    #     print("round: ", i)
    #     c = combs[i]
    #     for i in range(4):
    #         print("fold: ", i, end="-> ")
    #         for key, item in c.items():
    #             gp = item[i]
    #             gp = CLASS_DEF[int(gp.split(" ")[0])] + " " + gp.split(" ")[1]
    #             print(f"{gp}", end=", ")
    #         print()
    #     print()
    round_cv(config, X, y, groups, metainfo=dataset.get_metainfo())



    # data = {"y_pred":y_pred, "y_true":np.concatenate([y,] * num_repeats), "groups":np.concatenate([groups,] * num_repeats)}
    # data["round"] = np.array([i for i in range(num_repeats) for _ in range(len(y))])
    # results = pd.DataFrame(data).to_csv("results.csv", index=False)
    # if split is Split.biased_usual:
    #     y_pred = classifier_biased(config, X, y, groups)
    # else:
    #     y_pred = classifier_predefined(config, X, y, groups)
    
    # results(dataset, y, y_pred)
