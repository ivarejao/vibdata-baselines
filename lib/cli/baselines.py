import json
import os
from typing import List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from vibdata.deep.DeepDataset import DeepDataset

import lib.data.group_dataset as groups_module
from lib.config import Config

classifiers = ["randomforest", "knn"]


# TODO: This class exists for compatibility with `Config`
# Must be removed in the future
@dataclass
class Args:
    cfg: Path
    biased: bool
    unbiased: bool


def get_features(dataset: DeepDataset) -> tuple[List[int], List[int]]:
    X = [sample["signal"][0] for sample in dataset]
    y = [sample["metainfo"]["label"] for sample in dataset]

    return X, y


def randomforest_classifier_biased(cfg: Config, inputs: List[List[float]], labels: List[int], groups: List[int]) -> \
        List[int]:
    seed = cfg["seed"]
    parameters = cfg["params_grid"]

    num_folds = len(set(groups))

    model = RandomForestClassifier(random_state=seed)

    cv_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    clf = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="balanced_accuracy",
        cv=cv_inner,
        n_jobs=-1,
    )

    y_pred = cross_val_predict(clf, inputs, labels, cv=cv_outer)

    return y_pred


def randomforest_classifier_unbiased(cfg: Config, inputs: List[List[float]], labels: List[int], groups: List[int]) -> \
        List[int]:
    seed = cfg["seed"]
    parameters = cfg["params_grid"]

    model = RandomForestClassifier(random_state=seed)

    cv_inner = LeaveOneGroupOut()
    clf = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="balanced_accuracy",
        cv=cv_inner,
        n_jobs=-1,
    )

    cv_outer = LeaveOneGroupOut()
    y_pred = cross_val_predict(clf, inputs, labels, groups=groups, cv=cv_outer, fit_params={"groups": groups})

    return y_pred


def knn_classifier_biased(cfg: Config, inputs: List[List[float]], labels: List[int], groups: List[int]) -> List[int]:
    seed = cfg["seed"]
    n_neighbors = cfg["n_neighbors"]

    num_folds = len(set(groups))

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    y_pred = cross_val_predict(model, inputs, labels, cv=cv)

    return y_pred


def knn_classifier_unbiased(cfg: Config, inputs: List[List[float]], labels: List[int], groups: List[int]) -> List[int]:
    n_neighbors = cfg["n_neighbors"]

    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    cv = LeaveOneGroupOut()

    y_pred = cross_val_predict(model, inputs, labels, groups=groups, cv=cv)

    return y_pred


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int], biased: bool, dataset_name: str) -> None:
    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Show the results in terminal
    print(f"{classification_report(y_true, y_pred, target_names=labels)}")
    print(f"Balanced accuracy: {balanced_accuracy:.2f}")

    # Save the results in a .json file
    bias = "biased" if biased else "unbiased"
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report['balanced accuracy'] = balanced_accuracy
    directory = "results_baselines/" + bias
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = dataset_name + ".json"
    full_path = os.path.join(directory, file_name)
    with open(full_path, 'w') as json_file:
        json.dump(report, json_file, indent=4)
    print(f"\nThe file .json was saved in {full_path}.")


def configure_wandb(run_name: str, cfg: Config, cfg_path: str, groups: List[int], args) -> None:
    wandb.login(key=os.environ["WANDB_KEY"])
    config = {
        "model": cfg["model"]["name"],
        "folds": len(set(groups)),
        "biased": args.biased,
        "unbiased": args.unbiased,
    }
    if "params_grid" in cfg:
        config["params_grid"] = cfg["params_grid"]
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


def main(cfg: Path, biased: bool):
    args = Args(cfg=cfg, biased=biased, unbiased=not biased)
    load_dotenv()
    cfg_path = cfg
    cfg = Config(cfg_path, args=args)

    dataset_name = cfg["dataset"]["name"]
    dataset = cfg.get_dataset()
    run_name = cfg["run_name"]

    model = cfg["model"]["name"]

    if model not in classifiers:
        raise Exception(f"Undefined Classifier. Must be {classifiers}.")

    group_obj = getattr(groups_module, "Group" + dataset_name)(dataset=dataset, config=cfg)
    groups = group_obj.groups()

    configure_wandb(run_name, cfg, str(cfg_path), groups, args)

    X, y = get_features(dataset)

    if args.biased:
        y_pred = eval(f"{model}_classifier_biased")(cfg, X, y, groups)
    elif args.unbiased:
        y_pred = eval(f"{model}_classifier_unbiased")(cfg, X, y, groups)
    else:
        raise Exception("Undefined Classifier. Biased or unbiased must be selected.")

    results(dataset, y, y_pred, biased, run_name)
