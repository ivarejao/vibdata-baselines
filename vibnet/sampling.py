import random
from argparse import Namespace

import numpy as np
import torch
import numpy.typing as npt
from torch.utils.data import Subset, Dataset, DataLoader, BatchSampler, SequentialSampler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

import vibnet.data.group_dataset as groups_module
from vibnet.config import Config
from vibnet.utils.dataloaders import BalancedBatchSampler
from vibnet.utils.MemeDataset import MemeDataset


class DataSampling:
    def __init__(self, original_dataset: Dataset, config: Config, args: Namespace) -> None:

        group_obj = getattr(groups_module, "Group" + config["dataset"]["name"])(dataset=original_dataset,
                                                                                config=config)
        groups = group_obj.groups()

        self.dataset = MemeDataset(original_dataset)

        # Define the folds
        if args.biased:
            num_folds = len(set(groups))
            stratKFold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config['seed'])
            X, y = zip(*self.dataset)
            self.folds = [fold_ids for _, fold_ids in stratKFold.split(X=X, y=y)]

        elif args.unbiased:
            logo = LeaveOneGroupOut()
            # Split the dataset into folds based on a conditon
            self.folds = [fold_ids for _, fold_ids in logo.split(self.dataset, groups=groups)]

        else:
            raise Exception("Classifier ( Biased / Unbiased ) Undefined")

        # TODO: Reimplement these methods `get_labels` and `get_labels_name`
        # calling the respective methods directly from MemeDataset as it will
        # inherinthe from DeepDataset
        self.labels = self.dataset.dataset.get_labels()
        self.labels_name = self.dataset.dataset.get_labels_name()
        self.config = config

        # Create generator for the dataloaders
        self.generator = torch.Generator()
        # Fix the seed for the dataloaders generator
        self.generator.manual_seed(self.config["seed"])
        self.initialized = False

    def split(self, test_fold: int, with_val_set=True):
        self.current_fold = test_fold

        num_folds = len(self.folds)
        # Define test ids
        self.test_ids = self.folds[test_fold]
        if with_val_set:
            # Define validation fold
            val_fold = (test_fold + 1) % num_folds
            self.val_ids = self.folds[val_fold]
        else:
            val_fold = test_fold
        # Set the remaining folds to training
        train_folds = set(range(num_folds)).difference(set([test_fold, val_fold]))
        self.train_ids = np.concatenate([self.folds[i] for i in train_folds])

        # Set the flag
        self.initialized = True

    def is_initialized(self) -> bool:
        return self.initialized

    def get_trainloader(self) -> DataLoader:
        return self._get_dataloader(self.train_ids)

    def get_valloader(self) -> DataLoader:
        if not hasattr(self, "val_ids"):
            raise ValueError("The validation wasnt set at this split")
        return self._get_dataloader(self.val_ids)

    def get_testloader(self) -> DataLoader:
        return self._get_dataloader(self.test_ids, test=True)

    def _get_dataloader(self, samples_ids, test: bool = False) -> DataLoader:
        subset = Subset(self.dataset, samples_ids)
        if test:
            # TODO: Check how the sampler should behaviour in this case
            subset_sampler = BatchSampler(
                sampler=SequentialSampler(subset), batch_size=self.config["batch_size"], drop_last=False
            )
        else:
            subset_sampler = BalancedBatchSampler(
                labels=self.arange_labels(subset),
                n_classes=len(self.labels),
                n_samples=int(self.config["batch_size"] / len(self.labels)),
            )
        return DataLoader(
            subset, batch_sampler=subset_sampler, worker_init_fn=self.seed_worker, generator=self.generator
        )

    # Getter and Setters
    def get_labels(self) -> npt.NDArray[np.int_]:
        return self.labels

    def get_labels_name(self) -> npt.NDArray[np.str_]:
        return self.labels_name

    def get_num_folds(self) -> np.int32:
        return len(self.folds)

    def get_fold(self) -> np.int32:
        return self.current_fold

    # Static methods
    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @staticmethod
    def arange_labels(dataset):
        return [y for (x, y) in dataset]