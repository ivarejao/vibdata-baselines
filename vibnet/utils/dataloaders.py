import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, default_collate
from torch.utils.data.sampler import Sampler

from vibnet.utils.sklearn_dataset import SklearnDataset


def get_targets(dataset: SklearnDataset | Subset) -> np.ndarray[np.int64]:
    if isinstance(dataset, SklearnDataset):
        return dataset.targets
    elif isinstance(dataset, Subset):
        targets = get_targets(dataset.dataset)
        return targets[dataset.indices]
    elif hasattr(dataset, "base_dataset"):
        return get_targets(dataset.base_dataset)
    else:
        ds_type = type(dataset)
        raise TypeError(f"Dataset of type {ds_type} is not supported")


class BalancedSampler(Sampler):
    def __init__(self, dataset: SklearnDataset | Subset, random_state=None):
        labels = get_targets(dataset)
        self.labels = labels.reshape(-1)
        self.unique_labels: np.ndarray[int] = np.unique(labels)
        self.n_labels = self.unique_labels.size

        self.length = self.n_labels * (labels.size // self.n_labels)
        self.rng = np.random.default_rng(random_state)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        indexes = np.arange(self.labels.size)

        indexes_by_label = []
        for label in self.unique_labels:
            mask = self.labels == label
            label_indexes = indexes[mask]
            sampled_indexes = self.rng.choice(label_indexes, size=len(self) // self.n_labels)
            indexes_by_label.append(sampled_indexes)

        batch_indexes = np.vstack(indexes_by_label).T.reshape(-1)
        return iter(batch_indexes)


def unsqueeze_collate(batch):
    if isinstance(batch, list):
        return default_collate(batch)

    signal, target = batch
    return signal.unsqueeze(dim=0), target


class BalancedDataLoader(DataLoader):
    def __init__(self, dataset: SklearnDataset | Subset, drop_last=False, sampler=None, **kwargs):
        sampler = BalancedSampler(dataset, random_state=10)
        super().__init__(dataset, sampler=sampler, drop_last=False, **kwargs)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
