import os
import warnings
from typing import List
from pathlib import Path
from datetime import datetime

import numpy as np
import wandb
import pandas as pd
from rich import print
from scipy import stats
from dotenv import load_dotenv
from sklearn.dummy import check_random_state
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_val_predict
from vibdata.deep.DeepDataset import DeepDataset

from vibnet.config import ConfigSklearn
from vibnet.cli.common import Split, is_logged, group_class, set_deterministic
from vibnet.data.rounds_repo import load_combinations, is_multi_round_dataset
from vibnet.data.group_dataset import GroupMirrorBiased

# Suppress FutureWarning from scikit-learnig `fit_params` deprecated
warnings.simplefilter(action="ignore", category=FutureWarning)


def classifier_biased(
    cfg: ConfigSklearn, inputs: List[int], labels: List[int], groups: List[int], seed=None
) -> pd.DataFrame:
    if seed is None:
        seed = cfg["seed"]
    num_folds = len(set(groups))

    cv_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    clf = cfg.get_estimator(grid_serch_cv=cv_inner)

    y_pred = cross_val_predict(clf, inputs, labels, cv=cv_outer)

    results = pd.DataFrame({"y_true": labels, "y_pred": y_pred})

    idx_arr = np.arange(len(labels))
    folds_idx = np.sum(
        [
            np.isin(idx_arr, test_indices) * fold
            for fold, (_, test_indices) in enumerate(cv_outer.split(inputs, labels))
        ],
        axis=0,
    )

    # For compability with report function
    results["round"] = 0
    results["fold"] = folds_idx
    return results


def classifier_predefined(cfg: ConfigSklearn, inputs: List[int], labels: List[int], groups: List[int]) -> pd.DataFrame:
    cv_outer = LeaveOneGroupOut()
    cv_inner = LeaveOneGroupOut()

    clf = cfg.get_estimator(grid_serch_cv=cv_inner)

    fit_params = {"classifier__groups": groups} if "params_grid" in cfg else None

    y_pred = cross_val_predict(clf, inputs, labels, groups=groups, cv=cv_outer, fit_params=fit_params)

    results = pd.DataFrame({"y_true": labels, "y_pred": y_pred})
    # For compability with report function
    results["round"] = 0
    results["fold"] = groups - groups.min()

    return results


def classifier_multi_round(
    cfg: ConfigSklearn,
    inputs: List[int],
    labels: List[int],
    groups: List[int],
    split: Split,
    dataset: DeepDataset,
) -> pd.DataFrame:
    results = []
    combinations = load_combinations(cfg["dataset"])
    rng = check_random_state(cfg["seed"])  # Only used for biased_usual split

    for r, round_groups in enumerate(combinations):
        print("Round: ", r)
        # Get the new grops combinations and define the folds division in this round
        current_group = np.sum([np.isin(groups, fold_groups) * i for i, fold_groups in enumerate(round_groups)], axis=0)
        if split.value == Split.biased_usual:
            round_results = classifier_biased(cfg, inputs, labels, current_group, seed=rng)
        elif split.value == Split.biased_mirrored:
            current_group = GroupMirrorBiased(dataset=dataset, config=cfg, custom_name=cfg["run_name"]).groups(
                current_group
            )
            round_results = classifier_predefined(cfg, inputs, labels, current_group)
        elif split.value == Split.unbiased:
            round_results = classifier_predefined(cfg, inputs, labels, current_group)

        round_results["round"] = r
        results.append(round_results)

    results = pd.concat(results)

    return results


def report(dataset: DeepDataset, results: pd.DataFrame) -> None:
    def stats_report(scores):
        mean = scores.mean()
        std = scores.std()
        inf, sup = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(scores)))
        return (mean, std, inf, sup)

    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    metrics = ["balanced_accuracy", "f1_macro"]
    scores = {}
    scores["balanced_accuracy"] = np.array(
        [balanced_accuracy_score(group["y_true"], group["y_pred"]) for _, group in results.groupby("round")]
    )

    scores["f1_macro"] = np.array(
        [f1_score(group["y_true"], group["y_pred"], average="macro") for _, group in results.groupby("round")]
    )

    if results.index.get_level_values("round").nunique() == 1:
        print(f"{classification_report(results['y_true'], results['y_pred'], target_names=labels)}")

    for metric in metrics:
        if len(scores[metric]) == 1:
            print(f"{metric.capitalize()}: {scores[metric][0]:.2f}")
        else:
            mean, std, inf, sup = stats_report(scores[metric])
            print(f"{metric.capitalize()}: {mean:.2f} ± {std:.2f} [{inf:.2f}, {sup:.2f}]")


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


def main(cfg: Path, split: Split, clear_cache: bool):
    load_dotenv()
    actual_datetime = datetime.now()
    config = ConfigSklearn(cfg)
    if clear_cache:
        config.clear_cache()

    seed = config.get_yaml().get("seed", None)
    if seed is not None:
        set_deterministic(seed)
    else:
        print("[bold red]Alert![/bold red] Seed was not set!")

    dataset = config._get_dataset_deep()

    groups = group_class(config["dataset"])(dataset=dataset, config=config, custom_name=config["run_name"]).groups()
    groups = pd.Categorical(groups).codes

    configure_wandb(config["run_name"], config, cfg, groups, split)
    X, y = config.get_dataset()

    if is_multi_round_dataset(config["dataset"]):
        results = classifier_multi_round(config, X, y, groups, split, dataset)
    elif split == Split.biased_usual:
        results = classifier_biased(config, X, y, groups)
    else:
        if split == Split.biased_mirrored:
            groups = GroupMirrorBiased(dataset=dataset, config=config, custom_name=config["run_name"]).groups(groups)
        results = classifier_predefined(config, X, y, groups)

    results = results.set_index(["round", "fold"])
    results.sort_index(inplace=True)
    filename = f"results-baselines-{config.config.get('run_name', None)}-{actual_datetime.isoformat()}.csv"
    print(f"Saving csv at [bold green]{filename}[/bold green]")
    results.to_csv(filename)
    report(dataset, results)

    if is_logged():
        table = wandb.Table(data=results)
        wandb.log({"Results": table})
        wandb.save(str(cfg))
