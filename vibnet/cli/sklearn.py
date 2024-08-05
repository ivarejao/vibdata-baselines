import os
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime

import numpy as np
import wandb
import pandas as pd
import numpy.typing as npt
from rich import print
from scipy import stats
from sklearn.dummy import check_random_state
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, cross_validate
from vibdata.deep.DeepDataset import DeepDataset

from vibnet.config import ConfigSklearn
from vibnet.utils.sklearn import TrainDataset
from vibnet.data.rounds_repo import load_combinations, is_multi_round_dataset
from vibnet.data.group_dataset import GroupMirrorBiased

from .common import Split, is_logged, group_class, wandb_login, set_deterministic


def report(results: pd.DataFrame) -> None:
    def stats_report(scores):
        mean = scores.mean()
        std = scores.std()
        inf, sup = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(scores)))
        return (mean, std, inf, sup)

    agg_results = results.groupby("round", as_index=True).agg("mean")
    print(agg_results) if agg_results.shape[0] > 1 else print(results)

    for test_metric in [c for c in agg_results.columns if "test_" in c]:
        metric_name = test_metric.replace("test_", "")
        if agg_results.shape[0] == 1:
            print(
                f"{metric_name.capitalize()}: {agg_results[test_metric].mean():.2f}"
            )  # Does not need this mean, its only to help print the actual value instead of a pd.Series
        else:
            mean, std, inf, sup = stats_report(agg_results[test_metric])
            print(f"{metric_name.capitalize()}: {mean:.2f} ± {std:.2f} [{inf:.2f}, {sup:.2f}]")


def classifier_multi_rounds(
    dataset: DeepDataset,
    cfg: ConfigSklearn,
    split: Split,
    cross_validate_args: Dict[str, Any],
    base_groups: npt.ArrayLike,
) -> List[Dict]:
    results = []
    combinations = load_combinations(cfg["dataset"])
    rng = check_random_state(cfg["seed"])  # Only used for biased_usual split

    for r, round_groups in enumerate(combinations):
        print("Round: ", r)
        current_group = np.sum(
            [np.isin(base_groups, fold_groups) * i for i, fold_groups in enumerate(round_groups)], axis=0
        )
        n_splits = np.unique(current_group).size

        if split == Split.biased_usual:
            cross_validate_args["cv"] = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
        else:
            if split == Split.biased_mirrored:
                cross_validate_args["groups"] = GroupMirrorBiased(
                    dataset=dataset, config=cfg, custom_name=cfg["run_name"]
                ).groups(current_group)
            elif split == Split.unbiased:
                cross_validate_args["groups"] = current_group
            cross_validate_args["cv"] = LeaveOneGroupOut()
            # This will be only used if the config["train_split"] are set
            # to a GroupCrossValidator, e.g. LeaveOneGroupOut
            cross_validate_args["fit_params"] = {"classifier__groups": cross_validate_args["groups"]}

        # Run round
        round_results = cross_validate(**cross_validate_args)
        round_results["round"] = [
            r,
        ] * n_splits
        round_results["fold"] = list(range(n_splits))
        results.append(round_results)

    return results


def main(cfg: Path, split: Split, clear_cache: bool):
    actual_datetime = datetime.now()

    try:
        wandb_login()
    except RuntimeError as e:
        print(f"[bold red]Alert![/bold red] Running without wandb: {e}")

    config = ConfigSklearn(cfg)
    seed = config.get_yaml().get("seed", None)
    if clear_cache:
        config.clear_cache()
    if seed is not None:
        set_deterministic(seed)
    else:
        print("[bold red]Alert![/bold red] Seed was not set!")

    dataset = config.get_deepdataset()

    # Create common args for cross validation
    pipeline = config.get_estimator()
    cross_validate_args = {
        "estimator": pipeline,
        "scoring": ["accuracy", "f1_macro", "balanced_accuracy"],
        "verbose": 4,
    }
    unbiased_groups = group_class(config["dataset"])(
        dataset=dataset, config=config, custom_name=config["run_name"]
    ).groups()
    unbiased_groups = pd.Categorical(unbiased_groups).codes

    train_dataset = TrainDataset(dataset)
    cross_validate_args.update({"X": train_dataset, "y": train_dataset.targets})

    # Define specific params based on the type of split
    if is_multi_round_dataset(config["dataset"]):
        results = classifier_multi_rounds(dataset, config, split, cross_validate_args, unbiased_groups)
    else:
        if split is Split.biased_usual:
            num_folds = len(set(unbiased_groups))
            cross_validate_args["cv"] = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        else:
            if split is Split.biased_mirrored:
                cross_validate_args["groups"] = GroupMirrorBiased(
                    dataset=dataset, config=config, custom_name=config["run_name"]
                ).groups(unbiased_groups)
            else:
                cross_validate_args["groups"] = unbiased_groups
            cross_validate_args["cv"] = LeaveOneGroupOut()
            # This will be only used if the config["train_split"] are set
            # to a GroupCrossValidator, e.g. LeaveOneGroupOut
            cross_validate_args["fit_params"] = {"classifier__groups": cross_validate_args["groups"]}

        # Run cross validation
        num_folds = len(set(unbiased_groups))
        results = cross_validate(**cross_validate_args)
        results["round"] = [
            0,
        ] * num_folds  # For compability with report
        results["fold"] = list(range(num_folds))

    df = pd.DataFrame(results) if isinstance(results, dict) else pd.concat([pd.DataFrame(d) for d in results])
    df = df.set_index(["round", "fold"])
    filename = f"results-{config.config.get('run_name', None)}-{actual_datetime.isoformat()}.csv"
    print(f"Saving csv at [bold green]{filename}[/bold green]")
    df.to_csv(filename)
    report(df)

    if is_logged():
        project_name = os.environ["WANDB_PROJECT"]
        run = wandb.init(
            project=project_name,
            name="Results",
            group=config.group_name,
        )
        table = wandb.Table(data=df)

        run.log({"Results": table})
        run.save(str(cfg))
