import sys
import logging
import numbers
import tempfile
import warnings
from types import EllipsisType
from typing import Any, Type, Literal, Callable, Iterator, Optional
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime
from tempfile import TemporaryDirectory
from functools import wraps
from collections import defaultdict

import numpy as np
import torch
import wandb
import lightning as L
from torch import nn
from tqdm.auto import tqdm
from torch.optim import Adam, Optimizer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from vibdata.deep.DeepDataset import DeepDataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import Strategy
from lightning.pytorch.loggers.wandb import WandbLogger

from vibnet.utils.lightning import VibnetModule
from vibnet.utils.dataloaders import BalancedDataLoader, get_targets
from vibnet.utils.sklearn_dataset import SklearnDataset

_run_name_counter = defaultdict(lambda: 0)


def add_index(prefix: str) -> str:
    global _run_name_counter
    counter = _run_name_counter[prefix]
    _run_name_counter[prefix] = counter + 1
    name = f"{prefix}-{counter:02d}"
    return name


def _no_lightning_logs(func: Callable):
    @wraps(func)
    def wrapper_func(*args, **kwargs) -> Any:
        lightning_logger = logging.getLogger("lightning.pytorch")
        current_level = lightning_logger.level
        lightning_logger.setLevel(logging.ERROR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="lightning")
            return_value = func(*args, **kwargs)

        lightning_logger.setLevel(current_level)
        return return_value

    return wrapper_func


class _SklearnCompatibleDataset:
    @property
    def ndim(self):
        return 2  # Compatibility work around

    @property
    def shape(self):
        return len(self), 2  # Compatibility work around


class TrainDataset(SklearnDataset, _SklearnCompatibleDataset):
    """Dataset that can be splitted into a subset. Useful for folding"""

    def __init__(self, src_dataset: DeepDataset, standardize: bool = True):
        super().__init__(src_dataset, standardize=True)

    def __getitem__(self, idx: int | np.ndarray[np.int64] | list[int] | tuple[np.ndarray[np.int64], EllipsisType]):
        if isinstance(idx, numbers.Integral):  # works with int, np.int64, ...
            X, y = super().__getitem__(idx)
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y)

        if isinstance(idx, (np.ndarray, list)):
            indices = idx
        elif isinstance(idx, tuple):
            try:
                possible_indices, possible_ellipsis = idx
            except Exception:
                size = len(idx)
                raise ValueError(f"Tuple must be of size 2, not {size}")

            if possible_ellipsis != ...:
                raise ValueError(f"Second element of tuple must be ellipsis, not {possible_ellipsis}")
            elif not isinstance(possible_indices, (np.ndarray, list)):
                t = type(possible_indices)
                raise ValueError(f"First element of tuple must be ndarray or list, not {t}")

            indices = possible_indices
        else:
            t = type(idx)
            raise ValueError(f"Indices of type {t} not supported")

        return Subset(self, indices)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(self)
        for i in range(n):
            yield self[i]


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

        for train_index, test_index in self.cv.split(X, y, groups=groups):
            # Uses only the first split
            return train_index, test_index


class VibnetEstimator(BaseEstimator, ClassifierMixin):
    _kwargs_prefixes = [
        "module__",
        "optimizer__",
        "iterator_train__",
        "iterator_valid__",
        "trainer__",
        "lr_scheduler__",
    ]
    _group_name = {}

    def __init__(
        self,
        module: Type[nn.Module],
        num_classes: int,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_group: Optional[str] = None,
        optimizer: Type[Optimizer] = Adam,
        lr_scheduler: Type[LRScheduler] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        iterator_train: Type[DataLoader] = BalancedDataLoader,
        iterator_valid: Type[DataLoader] = DataLoader,
        train_split: Optional[SingleSplit] = None,
        accelerator: Literal["cpu", "gpu"] = "cpu",
        devices: int | Literal["auto"] = "auto",
        precision: Optional[Literal[64, 32, 16] | Literal["64", "32", "16", "bf16-mixed"]] = None,
        max_epochs: int = 1,
        fast_dev_run: int | bool = False,
        strategy: str | Strategy = "auto",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.module = module
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
        self.verbose = verbose

        self.module_: Optional[VibnetModule] = None
        self.logger_: Optional[WandbLogger] = None

        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.wandb_group = wandb_group

        # Replace this XOR-like expression?
        if (wandb_project is None) != (wandb_name is None):
            raise RuntimeError("Both wandb_project and wandb_name must be set or both must be None")

        try:
            self.group_name = VibnetEstimator._group_name[self.wandb_group]
        except KeyError:
            now = datetime.now()
            now_str = now.strftime("%d/%m/%Y %H:%M")
            if self.wandb_group is not None:
                group_name = f"{self.wandb_group} [{now_str}]"
            else:
                group_name = f"{now_str}"
            self.group_name = group_name
            VibnetEstimator._group_name[self.wandb_group] = group_name

        for k, v in kwargs.items():
            if not any(k.startswith(p) for p in VibnetEstimator._kwargs_prefixes):
                raise TypeError("'{k}' is not a valid argument")
            setattr(self, k, v)

    def _trainer_params(self) -> dict[str, Any]:
        return self._params_prefix("trainer__")

    def _module_params(self) -> dict[str, Any]:
        return self._params_prefix("module__")

    def _optimizer_params(self) -> dict[str, Any]:
        return self._params_prefix("optimizer__")

    def _lr_scheduler_params(self) -> dict[str, Any]:
        return self._params_prefix("lr_scheduler__")

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
        params = {
            "network": self.module(**self._module_params()),
            "loss_fn": self.loss_fn,
            "optimizer_class": self.optimizer,
            "optimizer_params": self._optimizer_params(),
            "num_classes": self.num_classes,
        }
        if self.lr_scheduler is not None:
            params["lr_scheduler_class"] = self.lr_scheduler
            params["lr_scheduler_params"] = self._lr_scheduler_params()
        return VibnetModule(**params)

    def _create_trainer(self) -> L.Trainer:
        trainer_params = {
            "accelerator": self.accelerator,
            "devices": self.devices,
            "precision": self.precision,
            "max_epochs": self.max_epochs,
            "fast_dev_run": self.fast_dev_run,
            "strategy": self.strategy,
            "enable_progress_bar": self.verbose,
            "logger": self._create_logger(),
            "deterministic": True,
            "callbacks": [self._create_model_checkpoint()],
        }
        trainer_params.update(self._trainer_params())
        return L.Trainer(**trainer_params)

    def _dataloaders(
        self, X: DeepDataset | TrainDataset | Subset, groups=None
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
        train_index, valid_index = self.train_split(X, targets, groups=groups)
        train_dataset, valid_dataset = dataset[train_index], dataset[valid_index]

        train_params = self._iterator_train_params()
        train_dl = self.iterator_train(train_dataset, **train_params)

        valid_params = self._iterator_valid_params()
        valid_dl = self.iterator_valid(valid_dataset, **valid_params)

        return train_dl, valid_dl

    def _create_logger(self) -> Optional[WandbLogger]:
        if self.wandb_project is None:
            return None
        save_dir = tempfile.mkdtemp(prefix="vibnet_run-")
        name = add_index(self.wandb_name)

        return WandbLogger(
            project=self.wandb_project,
            name=name,
            save_dir=save_dir,
            group=self.group_name,
        )

    def _create_model_checkpoint(self) -> ModelCheckpoint:
        run_id = self.group_name.replace("/", "-").replace(" ", "-").lower()
        checkpoint = ModelCheckpoint(
            dirpath="checkpoints/{}".format(run_id),
            monitor="val/f1",
            save_last=True,
            mode="max",
            filename=add_index("sample_epoch={epoch:02d}-val_f1={val/f1:.2f}-fold"),
            save_top_k=1,
            save_on_train_epoch_end=True,
            # Prevent format filename into a directory because of the slash in `val/f1`
            auto_insert_metric_name=False,
        )
        checkpoint.CHECKPOINT_NAME_LAST = add_index("last-fold")  # Just to keep the same pattern
        return checkpoint

    @_no_lightning_logs
    def fit(self, X: DeepDataset | TrainDataset | Subset, y=None, groups=None, **fit_params):
        model = self._create_module()
        trainer = self._create_trainer()

        train_dl, valid_dl = self._dataloaders(X, groups=groups)

        if valid_dl is None:
            trainer.fit(model, train_dl)
        else:
            trainer.fit(model, train_dl, valid_dl)

        if self.wandb_project is not None:
            wandb.finish()

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
        trainer_params = {
            "accelerator": self.accelerator,
            "devices": 1 if self.accelerator == "gpu" else "auto",
            "precision": self.precision,
            "enable_progress_bar": self.verbose,
            "deterministic": True,
        }
        new_params = self._trainer_params()
        # Check if multple devices are set and override to 1 as
        # multiple devices on predictions are error prone
        if "devices" in new_params and isinstance(new_params["devices"], list) and len(new_params["devices"]) > 1:
            new_params["devices"] = 1 if self.accelerator == "gpu" else "auto"
        trainer_params.update(new_params)
        return L.Trainer(**trainer_params)

    @_no_lightning_logs
    def predict_proba(self, X: DeepDataset | TrainDataset | SklearnDataset | Subset) -> np.ndarray[float]:
        check_is_fitted(self)
        dataset = PredictDataset(X)
        trainer = self._create_predict_trainer()
        valid_params = self._iterator_valid_params()
        dataloader = self.iterator_valid(dataset, **valid_params)
        predictions = trainer.predict(self.module_, dataloader)
        predictions = [p.softmax(axis=1).to(torch.float32).cpu().numpy() for p in predictions]
        return np.vstack(predictions)

    def predict(self, X: DeepDataset | TrainDataset | SklearnDataset | Subset):
        probabilities = self.predict_proba(X)
        predicted_labels = probabilities.argmax(axis=1)
        return predicted_labels


class ScaledDataset(Dataset, _SklearnCompatibleDataset):
    def __init__(self, base_dataset: TrainDataset | Subset, tmp_dir: TemporaryDirectory):
        self.base_dataset = base_dataset
        self.tmp_dir = tmp_dir
        self.zip_path = Path(tmp_dir.name) / "scaled_signals.zip"

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(idx, numbers.Integral):
            return Subset(self, idx)

        _, target = self.base_dataset[idx]

        with np.load(self.zip_path) as npz:
            signal = npz[str(idx)]

        signal = torch.tensor(signal, dtype=torch.float32)
        return signal, target


class VibnetStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, in_channels: int = 1, verbose: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.verbose = verbose

    def fit(self, X: TrainDataset, y=None, **fit_params):
        numel = [0] * self.in_channels
        sums = [0] * self.in_channels
        for signal, _ in X:
            for idx, channel in enumerate(signal):
                numel[idx] += int(channel.numel())
                sums[idx] += float(channel.sum())
        means = [s / n for s, n in zip(sums, numel)]

        diff_sums = [0] * self.in_channels
        for signal, _ in X:
            for idx, channel in enumerate(signal):
                mean_diff = (channel - means[idx]) ** 2
                diff_sums[idx] += mean_diff.sum()
        stds = np.sqrt([d / (n - 1) for d, n in zip(diff_sums, numel)])

        self.means_ = np.array(means)
        self.stds_ = stds

        return self

    def transform(self, X: TrainDataset) -> ScaledDataset:
        check_is_fitted(self)

        tmp_dir = TemporaryDirectory(dir=Path("."), prefix="scaled_dataset-")
        path = Path(tmp_dir.name)

        with ZipFile(path / "scaled_signals.zip", mode="w") as zipf:
            idx = 0
            iterator = tqdm(X, desc="Scaling signals") if self.verbose else X
            for signal, _ in iterator:
                signal = (signal - self.means_) / self.stds_
                with zipf.open(f"{idx}.npy", "w") as npy_file:
                    np.save(npy_file, signal, allow_pickle=False)
                idx += 1

        dataset = ScaledDataset(base_dataset=X, tmp_dir=tmp_dir)
        return dataset
