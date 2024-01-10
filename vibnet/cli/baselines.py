import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import typer
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
)
from vibdata.deep.DeepDataset import DeepDataset

import vibnet.data.group_dataset as groups_module
from vibnet.config import Config

app = typer.Typer(pretty_exceptions_show_locals=False)


# TODO: This class exists for compatibility with `Config`
# Must be removed in the future
@dataclass
class Args:
    cfg: Path
    biased: bool
    unbiased: bool


def get_features(dataset: DeepDataset) -> (List[int], List[int]):
    X = np.empty([len(dataset), 9])
    y = np.empty([len(dataset)], dtype=np.int8)

    for i, sample in enumerate(dataset):
        features = []
        for feature in sample["features"].values():
            features.append(feature)
        X[i] = features
        y[i] = sample["metainfo"]["label"].iloc[0]

    return X, y


def classifier_biased(
    cfg: Config, inputs: List[int], labels: List[int], groups: List[int]
) -> List[int]:
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


def classifier_unbiased(
    cfg: Config, inputs: List[int], labels: List[int], groups: List[int]
) -> List[int]:
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
    y_pred = cross_val_predict(
        clf, inputs, labels, groups=groups, cv=cv_outer, fit_params={"groups": groups}
    )

    return y_pred


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int]) -> None:
    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    print(f"{classification_report(y_true, y_pred, target_names=labels)}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.2f}")


def configure_wandb(
    run_name: str, cfg: Config, cfg_path: str, groups: List[int], args
) -> None:
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        # Set the project where this run will be logged
        project=os.environ["WANDB_PROJECT"],
        # Track essentials hyperparameters and run metadata
        config={
            "model": cfg["model"]["name"],
            "folds": len(set(groups)),
            "params_grid": cfg["params_grid"],
            "biased": args.biased,
            "unbiased": args.unbiased,
        },
        # Set the name of the experiment
        name=run_name,
    )
    # Add configuration file into the wandb log
    wandb.save(cfg_path, policy="now")


@app.command()
def main(
    cfg: Path = typer.Option(help="Config file"),
    biased: bool = typer.Option(False, help="Use biased classifier"),
):
    args = Args(cfg=cfg, biased=biased, unbiased=not biased)
    load_dotenv()
    cfg_path = cfg
    cfg = Config(cfg_path, args=args)

    dataset_name = cfg["dataset"]["name"]
    dataset = cfg.get_dataset()

    group_obj = getattr(groups_module, "Group" + dataset_name)(
        dataset=dataset, config=cfg
    )
    groups = group_obj.groups()

    configure_wandb(dataset_name, cfg, cfg_path, groups, args)

    X, y = get_features(dataset)

    if args.biased:
        y_pred = classifier_biased(cfg, X, y, groups)
    elif args.unbiased:
        y_pred = classifier_unbiased(cfg, X, y, groups)
    else:
        raise Exception("Undefined classifier")

    results(dataset, y, y_pred)