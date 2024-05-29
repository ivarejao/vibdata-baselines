import csv
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneGroupOut, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from vibdata.deep.DeepDataset import DeepDataset

import vibnet.data.group_dataset as groups_module
from vibnet.cli.common import Split
from vibnet.config import Config
from vibnet.data.group_dataset import GroupMirrorBiased


def get_features(dataset: DeepDataset) -> (List[int], List[int]):
    X = [sample["signal"][0] for sample in dataset]
    y = [sample["metainfo"]["label"] for sample in dataset]

    return X, y


def classifier_biased(cfg: Config, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
    model_name = cfg["model"]["name"]
    seed = cfg["seed"]
    num_folds = len(set(groups))

    if model_name == "randomforest":
        parameters = cfg["params_grid"]
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        model = GridSearchCV(
            estimator=RandomForestClassifier(random_state=seed),
            param_grid=parameters,
            scoring="balanced_accuracy",
            cv=cv_inner,
            n_jobs=-1,
        )

    elif model_name == "knn":
        n_neighbors = cfg["n_neighbors"]
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

    else:
        raise Exception(f"Unexpected model_name {model_name}.")

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    scores = cross_validate(model, inputs, labels, scoring=('accuracy', 'f1_macro', 'balanced_accuracy'), cv=cv)

    return scores


def classifier_predefined(cfg: Config, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
    model_name = cfg["model"]["name"]
    seed = cfg["seed"]
    cv = LeaveOneGroupOut()

    if model_name == "randomforest":
        parameters = cfg["params_grid"]
        cv_inner = LeaveOneGroupOut()

        model = GridSearchCV(
            estimator=RandomForestClassifier(random_state=seed),
            param_grid=parameters,
            scoring="balanced_accuracy",
            cv=cv_inner,
            n_jobs=-1,
        )

        scores = cross_validate(model, inputs, labels, groups=groups,
                                scoring=('accuracy', 'f1_macro', 'balanced_accuracy'),
                                cv=cv, params={"groups": groups})

    elif model_name == "knn":
        n_neighbors = cfg["n_neighbors"]
        model = KNeighborsClassifier(n_neighbors=n_neighbors)

        scores = cross_validate(model, inputs, labels, groups=groups,
                                scoring=('accuracy', 'f1_macro', 'balanced_accuracy'),
                                cv=cv)
    else:
        raise Exception(f"Unexpected model_name {model_name}.")

    return scores


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int], split: Split, file_name: str,
            model_name: str) -> None:
    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Show the results in terminal
    print(f"{classification_report(y_true, y_pred, target_names=labels)}")
    print(f"Balanced accuracy: {balanced_accuracy:.2f}")

    # Save the results in a .json file
    bias = split.name
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    report['balanced accuracy'] = balanced_accuracy
    directory = "results_baselines/" + model_name + "/" + bias
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_file_name = file_name + ".json"
    full_path = os.path.join(directory, full_file_name)
    with open(full_path, 'w') as json_file:
        json.dump(report, json_file, indent=4)
    print(f"\nThe file .json was saved in {full_path}.")


def configure_wandb(run_name: str, cfg: Config, cfg_path: str, groups: List[int], split: Split) -> None:
    wandb.login(key=os.environ["WANDB_KEY"])
    config = {
        "model": cfg["model"]["name"],
        "folds": len(set(groups)),
        "split": split.value
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


def save_scores(scores, split, model_name, file_name):
    df_scores = pd.DataFrame(scores)

    bias = split.name
    filepath = Path("results_baselines/" + model_name + "/" + bias + '/' + file_name + ".csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)


def main(cfg: Path, split: Split):
    load_dotenv()
    config = Config(cfg)

    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    dataset = config.get_dataset()

    group_obj = getattr(groups_module, "Group" + dataset_name)(dataset=dataset, config=config)
    groups = group_obj.groups()

    if split is Split.biased_mirrored:
        groups = GroupMirrorBiased(
            dataset=dataset, config=config, custom_name=config["run_name"]
        ).groups(groups)

    configure_wandb(dataset_name, config, str(cfg), groups, split)

    X, y = get_features(dataset)

    if split is Split.biased_usual:
        scores = classifier_biased(config, X, y, groups)
    else:
        scores = classifier_predefined(config, X, y, groups)

    save_scores(scores, split, model_name, config["run_name"].lower())
