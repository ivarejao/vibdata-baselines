import logging
import numbers
import sys
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Literal, Optional, Type

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies import Strategy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset, Subset
from vibdata.deep.DeepDataset import DeepDataset

from vibnet.utils.dataloaders import BalancedDataLoader, get_targets
from vibnet.utils.lightning import VibnetModule
from vibnet.utils.MemeDataset import MemeDataset


def _no_lightning_logs(func: Callable):
    @wraps(func)
    def wrapper_func(*args, **kwargs) -> Any:
        lightning_logger = logging.getLogger("lightning.pytorch")
        current_level = lightning_logger.level
        lightning_logger.setLevel(logging.ERROR)

        return_value = func(*args, **kwargs)

        lightning_logger.setLevel(current_level)
        return return_value

    return wrapper_func


class TrainDataset(MemeDataset):
    """Dataset that can be splitted into a subset. Useful for folding"""

    def __getitem__(self, idx: int | np.ndarray[np.int64] | list[int]):
        if isinstance(idx, numbers.Integral):  # works with int, np.int64, ...
            X, y = super().__getitem__(idx)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y)

        return Subset(self, idx)

    @property
    def ndim(self):
        return 2  # Compatibility work around

    @property
    def shape(self):
        return len(self), 2  # Compatibility work around


class PredictDataset(Dataset):
    """Useful for test fold"""

    def __init__(self, ds: Dataset | DeepDataset):
        if isinstance(ds, DeepDataset):
            ds = TrainDataset(ds)
        self.ds = ds

    def __getitem__(self, idx: int):
        entry = self.ds[idx]
        array = None
        match entry:
            case (X, _):
                array = X
            case X:
                array = X

        if torch.is_tensor(array):
            return array.type(torch.float32)
        else:
            return torch.tensor(array, dtype=torch.float32)

    def __len__(self):
        return len(self.ds)


class SingleSplit:
    """CV splitter wrapper. Split dataset only once"""

    def __init__(self, cv):
        if isinstance(cv, numbers.Integral):
            self.cv = StratifiedKFold(cv)
        else:
            self.cv = cv

    def __call__(self, dataset, y=None, groups=None) -> tuple[np.ndarray, np.ndarray]:
        """Return train and test indexes"""
        X = np.zeros([len(y), 1])  # fake dataset

        for train_index, test_index in self.cv.split(X, y, groups):
            # Uses only the first split
            return train_index, test_index


class VibnetEstimator(BaseEstimator, ClassifierMixin):
    _kwargs_prefixes = ["optimizer__", "iterator_train__", "iterator_valid__"]
    _run_name_counter = defaultdict(lambda: 0)

    def __init__(
        self,
        module: Type[nn.Module],
        num_classes: int,
        wandb_project: str,
        wandb_name: str,
        optimizer: Type[Optimizer] = Adam,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        iterator_train: Type[DataLoader] = BalancedDataLoader,
        iterator_valid: Type[DataLoader] = DataLoader,
        train_split: Optional[SingleSplit] = None,
        accelerator: Literal["cpu", "gpu"] = "cpu",
        devices: int | Literal["auto"] = "auto",
        precision: Optional[
            Literal[64, 32, 16] | Literal["64", "32", "16", "bf16-mixed"]
        ] = None,
        max_epochs: int = 1,
        fast_dev_run: int | bool = False,
        strategy: str | Strategy = "auto",
        **kwargs,
    ):
        super().__init__()

        self.module = module
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.iterator_train = iterator_train
        self.iterator_valid = iterator_valid
        self.train_split = train_split
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.max_epochs = max_epochs
        self.fast_dev_run = fast_dev_run
        self.strategy = strategy

        self.module_: Optional[VibnetModule] = None
        self.trainer_: Optional[L.Trainer] = None
        self.logger_: Optional[WandbLogger] = None

        self.wandb_project = wandb_project
        self.wandb_name = wandb_name

        for k, v in kwargs.items():
            if not any(k.startswith(p) for p in VibnetEstimator._kwargs_prefixes):
                raise TypeError("'{k}' is not a valid argument")
            setattr(self, k, v)

    def _optimizer_params(self) -> dict[str, Any]:
        return self._params_prefix("optimizer__")

    def _iterator_train_params(self) -> dict[str, Any]:
        return self._params_prefix("iterator_train__")

    def _iterator_valid_params(self) -> dict[str, Any]:
        return self._params_prefix("iterator_valid__")

    def _params_prefix(self, prefix: str) -> dict[str, Any]:
        params = {}
        for attr in dir(self):
            if attr.startswith(prefix):
                params[attr.removeprefix(prefix)] = getattr(self, attr)
        return params

    def get_params(self, deep=True) -> dict[str, Any]:
        out: dict[str, Any] = super().get_params(deep=deep)
        for attr in dir(self):
            if any(attr.startswith(p) for p in VibnetEstimator._kwargs_prefixes):
                out[attr] = getattr(self, attr)
        return out

    def _create_module(self) -> VibnetModule:
        return VibnetModule(
            network=self.module(),
            loss_fn=self.loss_fn,
            optimizer_class=self.optimizer,
            optimizer_params=self._optimizer_params(),
            num_classes=self.num_classes,
        )

    def _create_trainer(self) -> L.Trainer:
        return L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            max_epochs=self.max_epochs,
            fast_dev_run=self.fast_dev_run,
            strategy=self.strategy,
            enable_progress_bar=False,
        )

    def _dataloaders(
        self, X: DeepDataset | TrainDataset | Subset
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """train/validation dataloaders"""

        if isinstance(X, DeepDataset):
            dataset = TrainDataset(X)
        else:
            dataset = X

        self.classes_ = np.unique(get_targets(dataset))

        if self.train_split is None:
            params = self._iterator_train_params()
            dataloader = self.iterator_train(dataset, **params)
            return dataloader, None

        targets = get_targets(dataset)
        train_index, valid_index = self.train_split(X, targets)
        train_dataset, valid_dataset = dataset[train_index], dataset[valid_index]

        train_params = self._iterator_train_params()
        train_dl = self.iterator_train(train_dataset, **train_params)

        valid_params = self._iterator_valid_params()
        valid_dl = self.iterator_valid(valid_dataset, **valid_params)

        return train_dl, valid_dl

    def _create_logger(self) -> WandbLogger:
        name = self.enumerated_run_name(self.wandb_name)
        return WandbLogger(project=self.wandb_project, name=name)

    @_no_lightning_logs
    def fit(self, X: DeepDataset | TrainDataset | Subset, y=None, **fit_params):
        model = self._create_module()
        trainer = self._create_trainer()

        train_dl, valid_dl = self._dataloaders(X)

        if valid_dl is None:
            trainer.fit(model, train_dl)
        else:
            trainer.fit(model, train_dl, valid_dl)

        self.trainer_ = trainer
        self.module_ = model

        # Default strategy duplicates the process and run two equal scripts after
        # training. To prevent erros the clone process must exit early with status 0.
        try:
            if self.strategy == "ddp" and not self.module_.trainer.is_global_zero:
                sys.exit(0)
        except RuntimeError:
            # Some strategies (e.g. "ddp_spawn") don't attach trainer object
            pass

        return self

    def _create_predict_trainer(self):
        return L.Trainer(
            accelerator=self.accelerator,
            # Multi GPU in .predict can be error prone
            devices=1 if self.accelerator == "gpu" else "auto",
            precision=self.precision,
            enable_progress_bar=False,
        )

    @_no_lightning_logs
    def predict(self, X: DeepDataset | TrainDataset | MemeDataset | Subset):
        check_is_fitted(self)
        dataset = PredictDataset(X)
        trainer = self._create_predict_trainer()
        valid_params = self._iterator_valid_params()
        dataloader = self.iterator_valid(dataset, **valid_params)
        predictions = trainer.predict(self.module_, dataloader)
        predicted_labels = torch.concat([p.argmax(axis=1) for p in predictions])
        return predicted_labels.cpu().numpy()

    @staticmethod
    def enumerated_run_name(prefix: str) -> str:
        counter = VibnetEstimator._run_name_counter[prefix]
        VibnetEstimator._run_name_counter[prefix] += 1
        return f"{prefix}-{counter:02d}"
