import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich import print
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

import wandb
from vibnet.config import ConfigSklearn
from vibnet.utils.sklearn import TrainDataset

from .common import group_class, is_logged, set_deterministic, wandb_login


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
    dataset = config.get_dataset()
    group_obj = group_class(dataset_name)(dataset=dataset, config=config)
    groups = group_obj.groups().reshape(-1)

    estimator = config.get_estimator()
    dataset = TrainDataset(dataset)
    cv = LeaveOneGroupOut()
    results = cross_validate(
        estimator,
        dataset,
        dataset.targets,
        groups=groups,
        cv=cv,
        scoring=["accuracy", "f1_macro", "balanced_accuracy"],
        verbose=4,
    )

    df = pd.DataFrame(results)
    df = df.rename_axis("fold", axis="index")
    print(df)
    filename = f"results-{actual_datetime.isoformat()}.csv"
    print(f"Saving csv at [bold green]{filename}[/bold green]")
    df.to_csv(filename)

    if is_logged():
        project_name = os.environ["WANDB_PROJECT"]
        run = wandb.init(project=project_name, name="Results")
        table = wandb.Table(data=df)
        run.log({"Results": table})
