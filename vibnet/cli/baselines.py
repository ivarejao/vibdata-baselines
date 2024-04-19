import os
from typing import List
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from vibdata.deep.DeepDataset import DeepDataset

import vibnet.data.group_dataset as groups_module
from vibnet.config import Config
from vibnet.cli.common import Split
from vibnet.data.group_dataset import GroupMirrorBiased


def get_features(dataset: DeepDataset) -> (List[int], List[int]):
    X = [sample["signal"][0] for sample in dataset]
    y = [sample["metainfo"]["label"] for sample in dataset]

    return X, y


def classifier_biased(cfg: Config, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
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


def classifier_predefined(cfg: Config, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
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


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int]) -> None:
    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    print(f"{classification_report(y_true, y_pred, target_names=labels)}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.2f}")


def configure_wandb(run_name: str, cfg: Config, cfg_path: str, groups: List[int], split: Split) -> None:
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        # Set the project where this run will be logged
        project=os.environ["WANDB_PROJECT"],
        # Track essentials hyperparameters and run metadata
        config={
            "model": cfg["model"]["name"],
            "folds": len(set(groups)),
            "params_grid": cfg["params_grid"],
            "split": split.value,
        },
        # Set the name of the experiment
        name=run_name,
    )
    # Add configuration file into the wandb log
    wandb.save(cfg_path, policy="now")


def main(cfg: Path, split: Split):
    load_dotenv()
    config = Config(cfg)

    dataset_name = config["dataset"]["name"]
    dataset = config.get_dataset()

    group_obj = getattr(groups_module, "Group" + dataset_name)(dataset=dataset, config=config)
    groups = group_obj.groups()

    if split is Split.biased_mirrored:
        groups = GroupMirrorBiased(dataset=dataset, config=config, custom_name=config["run_name"]).groups(groups)

    configure_wandb(dataset_name, config, cfg, groups, split)

    X, y = get_features(dataset)
    if split is Split.biased_usual:
        y_pred = classifier_biased(config, X, y, groups)
    else:
        y_pred = classifier_predefined(config, X, y, groups)

    results(dataset, y, y_pred)
