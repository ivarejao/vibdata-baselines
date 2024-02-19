import os
from typing import Any, Type, Optional
from pathlib import Path
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
from vibnet.models.M5 import M5
from vibnet.models.model import Model
from vibnet.utils.sklearn import SingleSplit, TrainDataset, VibnetEstimator, VibnetStandardScaler
from vibnet.models.Resnet1d import resnet18, resnet34
from vibnet.models.Alexnet1d import alexnet

__all__ = ["Config", "ConfigSklearn"]

_DEEP_MODELS = ["alexnet", "resnet18", "resnet34", "xresnet18", "m5", "resnet18-tsai"]

RESAMPLING_DATASETS = {"IMS", "XJTU"}


class Config:
    def __init__(self, config_path, args=None):
        self.config = {}
        self.load(config_path)

        # Override the configuration with the cli args
        # if args:
        #     self.args = args
        #     self.config["epochs"] = self.config["epochs"] if args.epochs is None else self.args.epochs
        #     self.config["optimizer"]["parameters"]["lr"] = (
        #         self.config["optimizer"]["parameters"]["lr"] if args.lr is None else self.args.lr
        #     )
        #     self.config["dataset"]["name"] = (
        #         self.config["dataset"]["name"] if args.dataset is None else self.args.dataset
        #     )
        #     self.config["batch_size"] = self.config["batch_size"] if args.dataset is None else self.args.batch_size

        self.setup_model()

    def load(self, path):
        with open(path, "r") as file:
            config_str = file.read()
        self.config = yaml.load(config_str, Loader=yaml.FullLoader)

    def setup_model(self):
        name = self.config["model"]["name"]
        parameters = self.config["model"]["parameters"]
        output_param_name = self.config["model"]["output_param"]

        # TODO: improve it by only calling one time this funcion
        dataset: DeepDataset = self.get_dataset()
        parameters[output_param_name] = len(dataset.get_labels())

        self.model_constructor = Model(name, **parameters)

    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config

    def get_yaml(self):
        return self.config

    def get_optimizer(self, model_parameters, **kwargs):
        opt_params = self.config["optimizer"]["parameters"]
        # Override the params
        for key, value in kwargs.items():
            # Just ensure that the key exists in optimizer params
            if key in opt_params:
                opt_params[key] = value

        return getattr(torch.optim, self.config["optimizer"]["name"])(model_parameters, **opt_params)

    def get_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        lr_schedulers = []
        for scheduler in self.config["lr_scheduler"]:
            lr_schedulers.append(
                getattr(torch.optim.lr_scheduler, scheduler["name"])(optimizer, **scheduler["parameters"])
            )
        return lr_schedulers

    def get_device(self):
        """
        If cuda device is avalaible, get the last one, as it more likely to not been used by other users
        """
        if torch.cuda.is_available():
            dev_number = torch.cuda.device_count()
            device = torch.device(f"cuda:{dev_number-1}")
        else:
            device = torch.device("cpu")
        return device

    def get_model(self, **kwargs) -> torch.nn.Module:
        # Besides the default parameters given when Config is instantiated, these can be override, passing as kwargs
        # into this method
        return self.model_constructor.new(**kwargs)

    def get_dataset(self):
        dataset_name = self.config["dataset"]["name"]

        # Get raw root_dir
        raw_root_dir = self.config["dataset"]["raw"]["root"]
        raw_dataset_package = getattr(datasets, dataset_name)
        raw_dataset_module = getattr(raw_dataset_package, dataset_name)
        raw_dataset = getattr(raw_dataset_module, dataset_name + "_raw")(raw_root_dir, download=True)

        deep_root_dir = os.path.join(self.config["dataset"]["deep"]["root"], dataset_name)
        # Get the transforms to be applied
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
        return dataset


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
    def __init__(self, config_path: str | Path, args=None):
        self.config = {}
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.dataset: DeepDataset | None = None

        dataset_name = self.config["dataset"]["name"]
        model_name = self.config["model"]["name"]
        now = datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M")
        self.group_name_ = f"{dataset_name}/{model_name} [{now_str}]"

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

        deep_root_dir = os.path.join(self.config["dataset"]["deep"]["root"], dataset_name)
        # Get the transforms to be applied
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

    def _get_estimator_ml(self) -> BaseEstimator:
        model_config = self.config["model"]
        model_class = _get_sklearn_class(model_config["name"])
        model_class_parameters = model_config.get("parameters", {})
        estimator = model_class(**model_class_parameters)

        parameter_grid = self.config.get("params_grid", None)
        if parameter_grid is not None:
            cv = StratifiedKFold(10, shuffle=False)
            estimator = GridSearchCV(
                estimator,
                param_grid=parameter_grid,
                cv=cv,
                n_jobs=-5,
                scoring="f1_macro",
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
        return self.config["model"]["name"] in _DEEP_MODELS

    def get_estimator(self) -> BaseEstimator:
        if self.is_deep_learning:
            estimator = self._get_estimator_deep()
        else:
            estimator = self._get_estimator_ml()

        pipeline = self._add_scaler(estimator)
        return pipeline

    def get_dataset(self) -> DeepDataset | tuple[np.ndarray, np.ndarray]:
        if self.is_deep_learning:
            return self._get_dataset_deep()
        else:
            return self._get_dataset_ml()

    def get_deepdataset(self) -> DeepDataset:
        return self._get_dataset_deep()
