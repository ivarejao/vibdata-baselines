from datetime import datetime
from pathlib import Path

import pandas as pd
from rich import print
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

from vibnet.config import ConfigSklearn
from vibnet.utils.sklearn import TrainDataset

from .common import group_class, set_deterministic, wandb_login


def main(cfg: Path):
    actual_datetime = datetime.now()

    config = ConfigSklearn(cfg)
    seed = config.get_yaml().get("seed", None)
    if seed is not None:
        set_deterministic(seed)

    try:
        wandb_login()
    except RuntimeError as e:
        print(f"[bold red]Alert![/bold red] Running without wandb: {e}")

    dataset_name = config["dataset"]["name"]
    dataset = config.get_dataset()
    group_obj = group_class(dataset_name)(dataset=dataset, config=config)
    groups = group_obj.groups()

    estimator = config.get_estimator()
    dataset = TrainDataset(dataset)
    cv = LeaveOneGroupOut()
    results = cross_validate(
        estimator,
        dataset,
        dataset.targets,
        groups=groups,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
    )

    df = pd.DataFrame(results)
    print(df)
    filename = f"results-{actual_datetime.isoformat()}.csv"
    print(f"Saving csv at [bold green]{filename}[/bold green]")
    df.to_csv(filename)
