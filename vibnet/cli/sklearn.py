import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from rich import print
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

import wandb
from vibnet.config import ConfigSklearn
from vibnet.utils.sklearn import TrainDataset

from .common import is_logged, group_class, wandb_login, set_deterministic


def main(cfg: Path):
    actual_datetime = datetime.now()

    try:
        wandb_login()
    except RuntimeError as e:
        print(f"[bold red]Alert![/bold red] Running without wandb: {e}")

    config = ConfigSklearn(cfg)
    seed = config.get_yaml().get("seed", None)
    if seed is not None:
        set_deterministic(seed)
    else:
        print("[bold red]Alert![/bold red] Seed was not set!")

    dataset_name = config["dataset"]["name"]
    dataset = config.get_deepdataset()
    group_obj = group_class(dataset_name)(dataset=dataset, config=config)
    groups = group_obj.groups().reshape(-1).astype(int)

    pipeline = config.get_estimator()
    cross_validate_args = {
        "estimator": pipeline,
        "groups": groups,
        "fit_params": {
            "classifier__groups": groups,
        },
        "scoring": ["accuracy", "f1_macro", "balanced_accuracy"],
        "verbose": 4,
        "cv": LeaveOneGroupOut(),
    }
    if config.is_deep_learning:
        train_dataset = TrainDataset(dataset)
        cross_validate_args.update({"X": train_dataset, "y": train_dataset.targets})
    else:
        X, y = config.get_dataset()
        cross_validate_args.update({"X": X, "y": y})

    results = cross_validate(**cross_validate_args)

    df = pd.DataFrame(results)
    df = df.rename_axis("fold", axis="index")
    print(df)
    filename = f"results-{actual_datetime.isoformat()}.csv"
    print(f"Saving csv at [bold green]{filename}[/bold green]")
    df.to_csv(filename)

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
