import os
import shutil
from os import PathLike
from typing import Any, Type, Optional
from datetime import datetime

import yaml
import numpy as np
import torch
import vibdata.raw as datasets
import vibdata.deep.signal.transforms as deep_transforms
from sklearn import model_selection as cross_validators
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, BaseCrossValidator
from vibdata.deep.DeepDataset import DeepDataset, convertDataset

import vibnet.data.resampling as resampler_pkg
from vibnet.schema import load_config
from vibnet.models.M5 import M5
from vibnet.utils.sklearn import SingleSplit, TrainDataset, VibnetEstimator, VibnetStandardScaler
from vibnet.models.Resnet1d import resnet18, resnet34
from vibnet.models.Alexnet1d import alexnet

__all__ = ["ConfigSklearn"]

RESAMPLING_DATASETS = {"IMS", "XJTU"}


def _get_model_class(name: str):
    # Import tsai models inside match case to prevent slowdown when not using these
    # models as they execute tkinter and matplotlib during import
    match name:
        case "alexnet":
            return alexnet
        case "resnet18":
            return resnet18
        case "resnet34":
            resnet34
        case "xresnet18":
            from tsai.models.XResNet1d import xresnet1d18

            return xresnet1d18
        case "m5":
            return M5
        case "resnet18-tsai":
            from tsai.models.ResNet import ResNet

            return ResNet
        case _:
            raise RuntimeError(f"Unknown model {name}")


def _get_sklearn_class(name: str) -> Type[BaseEstimator]:
    match name:
        case "randomforest":
            return RandomForestClassifier
        case "knn":
            return KNeighborsClassifier
        case _:
            raise RuntimeError(f"Unknown model {name}")


class ConfigSklearn:
    def __init__(self, config_path: str | PathLike, args=None):
        cfg_type, config = load_config(config_path)
        self.config: dict[str, Any] = config
        self.config_type: str = cfg_type

        self.dataset: DeepDataset | None = None

        dataset_name = self.config["dataset"]["name"]
        model_name = self.config["model"]["name"]
        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M")
        self.group_name_ = f"{dataset_name}/{model_name} [{now_str}]"

    def clear_cache(self):
        deep_root_dir = os.path.join(
            self.config["dataset"]["deep"]["root"], self.config.get("run_name", self.config["dataset"]["name"])
        )
        if os.path.exists(deep_root_dir):
            print("!! Removing cache deep", deep_root_dir)
            shutil.rmtree(deep_root_dir)
        group_path = os.path.join(self.config["dataset"]["groups_dir"], "groups_" + self.config["run_name"] + ".npy")
        if os.path.exists(group_path):
            print("!! Removing cache groups", group_path)
            os.remove(group_path)

    @property
    def group_name(self) -> Optional[str]:
        return self.group_name_

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config

    def get_yaml(self) -> dict[str, Any]:
        return self.config

    def _get_dataset_deep(self) -> DeepDataset:
        if self.dataset is not None:
            return self.dataset

        dataset_name = self.config["dataset"]["name"]

        # Get raw root_dir
        raw_root_dir = self.config["dataset"]["raw"]["root"]
        raw_dataset_package = getattr(datasets, dataset_name)
        raw_dataset_module = getattr(raw_dataset_package, dataset_name)
        raw_dataset = getattr(raw_dataset_module, dataset_name + "_raw")(raw_root_dir, download=True)

        deep_root_dir = os.path.join(self.config["dataset"]["deep"]["root"], self.config.get("run_name", dataset_name))
        # Get the transforms to be appliede
        transforms_config = self.config["dataset"]["deep"]["transforms"]
        transforms = deep_transforms.Sequential(
            [getattr(deep_transforms, t["name"])(**t["parameters"]) for t in transforms_config]
        )
        # Convert the raw dataset to deepdataset
        convertDataset(dataset=raw_dataset, transforms=transforms, dir_path=deep_root_dir)
        dataset = DeepDataset(deep_root_dir)
        # Resample the dataset if is needed
        if dataset_name in RESAMPLING_DATASETS:
            resampler = getattr(resampler_pkg, "Resampler" + dataset_name)()
            dataset = resampler.resample(dataset)
        self.dataset = dataset
        return self.dataset

    def _add_scaler(self, estimator: BaseEstimator) -> Pipeline:
        if self.is_deep_learning:
            scaler = VibnetStandardScaler(verbose=True)
        else:
            scaler = StandardScaler()

        pipeline = Pipeline([("scaler", scaler), ("classifier", estimator)])
        return pipeline

    def _get_estimator_deep(self) -> VibnetEstimator:
        dataset = self.get_dataset()
        wrapped_dataset = TrainDataset(dataset)
        targets = wrapped_dataset.targets
        num_classes = np.unique(targets).size
        estimator_parameters = {"num_classes": num_classes}

        model_config = self.config["model"]
        estimator_parameters["module"] = _get_model_class(model_config["name"])
        estimator_parameters.update({"module__" + k: v for k, v in model_config["parameters"].items()})
        estimator_parameters.update({f"module__{model_config['output_param']}": num_classes})

        optimizer_config = self.config["optimizer"]
        estimator_parameters["optimizer"] = getattr(torch.optim, optimizer_config["name"])
        estimator_parameters.update({"optimizer__" + k: v for k, v in optimizer_config["parameters"].items()})

        if "lr_scheduler" in self.config:
            lr_scheduler_config = self.config["lr_scheduler"]
            estimator_parameters["lr_scheduler"] = getattr(torch.optim.lr_scheduler, lr_scheduler_config["name"])
            estimator_parameters.update({f"lr_scheduler__{k}": v for k, v in lr_scheduler_config["parameters"].items()})
        # Extra parameters
        max_epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]

        train_split = self.config.get("train_split", None)
        if train_split is not None:
            if isinstance(train_split, int):
                train_split = SingleSplit(train_split)
            elif isinstance(train_split, dict):
                split_type = train_split["type"]
                params = train_split.get("parameters", {})
                try:
                    cv_class = getattr(cross_validators, split_type)
                    if not issubclass(cv_class, BaseCrossValidator):
                        raise ValueError(f"{split_type} is not a valid cross validation")
                except Exception:
                    raise ValueError(f"{split_type} is not a valid cross validation")
                cv = cv_class(**params)
                train_split = SingleSplit(cv)

        precision = self.config.get("precision", None)
        fast_dev_run = self.config.get("fast_dev_run", False)
        verbose = self.config.get("verbose", False)
        num_workers = self.config.get("num_workers", 0)

        project_name = os.environ.get("WANDB_PROJECT", None)
        run_name = self.config.get("run_name", None)
        run_name = run_name if project_name is not None else None

        dataset_name = self.config["dataset"]["name"]
        model_name = self.config["model"]["name"]

        trainer_parameters = (
            {f"trainer__{param}": value for param, value in self.config["trainer"].items()}
            if "trainer" in self.config
            else {}
        )

        estimator = VibnetEstimator(
            wandb_project=project_name,
            wandb_name=run_name,
            wandb_group=f"{dataset_name}/{model_name}",
            max_epochs=max_epochs,
            iterator_train__batch_size=batch_size,
            iterator_train__num_workers=num_workers,
            iterator_valid__batch_size=batch_size,
            train_split=train_split,
            precision=precision,
            fast_dev_run=fast_dev_run,
            verbose=verbose,
            devices=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            **estimator_parameters,
            **trainer_parameters,
        )
        self.group_name_ = estimator.group_name
        return estimator

    def _get_estimator_ml(self, gs_cv: BaseCrossValidator) -> BaseEstimator:
        model_config = self.config["model"]
        model_class = _get_sklearn_class(model_config["name"])
        model_class_parameters = model_config.get("parameters", {})
        estimator = model_class(**model_class_parameters)

        parameter_grid = self.config.get("params_grid", None)
        if parameter_grid is not None:
            estimator = GridSearchCV(
                estimator=estimator,
                param_grid=parameter_grid,
                scoring="balanced_accuracy",
                cv=gs_cv,
                n_jobs=-1,
            )

        return estimator

    def _get_dataset_ml(self) -> tuple[np.ndarray, np.ndarray]:
        deepdataset = self._get_dataset_deep()
        traindataset = TrainDataset(deepdataset, standardize=True)

        X = []
        length = len(traindataset)
        for i in range(length):
            feats, _ = traindataset[i]
            feats = feats.cpu().numpy().flatten()
            X.append(feats)

        X = np.vstack(X)
        y = traindataset.targets
        return X, y

    @property
    def is_deep_learning(self) -> bool:
        return self.config_type == "deep"

    def get_estimator(self, grid_serch_cv=StratifiedKFold(10, shuffle=False)) -> BaseEstimator:
        if self.is_deep_learning:
            estimator = self._get_estimator_deep()
        else:
            estimator = self._get_estimator_ml(grid_serch_cv)

        pipeline = self._add_scaler(estimator)
        return pipeline

    def get_dataset(self) -> DeepDataset | tuple[np.ndarray, np.ndarray]:
        if self.is_deep_learning:
            return self._get_dataset_deep()
        else:
            return self._get_dataset_ml()

    def get_deepdataset(self) -> DeepDataset:
        return self._get_dataset_deep()
