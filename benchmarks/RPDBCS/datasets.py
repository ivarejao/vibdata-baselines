from typing import Iterable, List, Union
import torch
from vibdata.datahandler.base import RawVibrationDataset
from vibdata.datahandler.transforms.TransformDataset import PickledDataset
from vibdata.datahandler.transforms.signal import FilterByValue, StandardScaler, Split, FFT, asType, ReshapeSingleChannel, SelectFields, toBinaryClassification, NormalizeSampleRate
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import hashlib
import os
import pickle


class TransformsDataset(torch.utils.data.IterableDataset):
    def __init__(self, D: torch.utils.data.IterableDataset, transforms) -> None:
        super().__init__()
        self.D = D
        self.transforms = transforms

    def __iter__(self):
        for d in self.D:
            if(hasattr(self.transforms, 'transform')):
                yield self.transforms.transform(d)
            else:
                yield self.transforms(d)

    def __getitem__(self, i):
        d = self.D[i]
        if(hasattr(self.transforms, 'transform')):
            return self.transforms.transform(d)
        else:
            return self.transforms(d)


class ConcatenateDataset(torch.utils.data.IterableDataset):
    def __init__(self, datasets: Iterable[PickledDataset], map_labels: List[List]):
        super().__init__()
        self.datasets = datasets

        if(map_labels is None):
            labels = [d.metainfo['label'].astype(int) for d in self.datasets]
        else:
            labels = []
            for d, mapl in zip(self.datasets, map_labels):
                L = d.metainfo['label'].copy()
                idxs = [(L == i, ml) for i, ml in enumerate(mapl) if i != ml]
                for idx, ml in idxs:
                    L[idx] = ml
                labels.append(L)
        self.labels = np.hstack(labels)
        self.num_labels = len(np.unique(self.labels))
        group_ids = [d.metainfo['index'] for d in self.datasets]
        for i, g in enumerate(group_ids[1:]):
            g += max(group_ids[i])+1
        self.group_ids = np.hstack(group_ids)

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, i: int):
        for k, d in enumerate(self.datasets):
            if(i < len(d)):
                break
            i -= len(d)
        ret = d[i]
        ret['X'] = ret['signal']
        del ret['signal']
        y = ret['label']
        del ret['label']
        ret['domain'] = k
        return ret, y

    def __iter__(self):
        for k, d in enumerate(self.datasets):
            for i in range(len(d)):
                ret = d[i]
                ret['X'] = ret['signal']
                del ret['signal']
                y = ret['label']
                del ret['label']
                ret['domain'] = k
                yield ret, y

    def getLabels(self):
        return self.labels

    def getDomains(self) -> np.ndarray:
        domains = np.empty(len(self), dtype=int)
        k = 0
        for i, d in enumerate(self.datasets):
            domains[k:k+len(d)] = i
            k += len(d)
        return domains

    def numLabels(self) -> int:
        return self.num_labels

    def getInputSize(self) -> int:
        return self[0][0]['X'].shape[-1]


class AppendDataset(torch.utils.data.IterableDataset):
    def __init__(self, D: torch.utils.data.IterableDataset, data_to_append: dict) -> None:
        super().__init__()
        self.D = D
        self.data_to_append = data_to_append
        if(hasattr(D, 'labels')):
            self.labels = D.labels

    def __iter__(self):
        for i, (X, y) in enumerate(self.D):
            for k, v in self.data_to_append.items():
                X[k] = v[i]
            yield X, y

    def __getitem__(self, i):
        X, y = self.D[i]
        for k, v in self.data_to_append.items():
            X[k] = v[i]
        return X, y

    def __len__(self):
        return len(self.D)


COMMON_TRANSFORMERS = [
    StandardScaler(on_field='signal', type='all'),
    NormalizeSampleRate(97656),
    Split(6100*2),
    FFT(),
    #    StandardScaler(on_field='signal', type='all'),
    asType(np.float32, on_field='signal'),
    # toBinaryClassification(),
    SelectFields('signal', ['label', 'index'])]


CWRU_TRANSFORMERS = COMMON_TRANSFORMERS
#  [
# FilterByValue(on_field='file_name', values=["97.mat", "105.mat", "118.mat", "130.mat", "169.mat", "185.mat",
#                                             "197.mat", "209.mat", "222.mat", "234.mat"]),
# FilterByValue(on_field='axis', values='DE'),
# ] + COMMON_TRANSFORMERS

MFPT_TRANSFORMERS = [
    FilterByValue(on_field='dir_name', values=["1 - Three Baseline Conditions",
                                               "3 - Seven More Outer Race Fault Conditions",
                                               "4 - Seven Inner Race Fault Conditions"]),
    # FilterByValue(on_field='file_name', remove=True,
    #               values=['1 - Three Baseline Conditions/baseline_2.mat', '1 - Three Baseline Conditions/baseline_2.mat']),
] + COMMON_TRANSFORMERS

SEU_TRANSFORMERS = [FilterByValue(on_field='channel', values=1)] + COMMON_TRANSFORMERS

PU_TRANSFORMERS = COMMON_TRANSFORMERS
# PU_TRANSFORMERS = [
#     FilterByValue(on_field='rotation_hz', values=[15])
# ] + COMMON_TRANSFORMERS

RPDBCS_TRANSFORMERS = [SelectFields('signal', ['label', 'index'])]
