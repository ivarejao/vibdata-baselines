from typing import Iterable, List, Union
import torch
from vibdata.datahandler.base import RawVibrationDataset
from vibdata.datahandler.transforms.TransformDataset import PickledDataset
from vibdata.datahandler.transforms.signal import FilterByValue, StandardScaler, Split, FFT, asType, SelectFields, toBinaryClassification, NormalizeSampleRate
from vibdata.datahandler.transforms.signal import Sampling
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
import hashlib
import os
import pickle


class TransformsDataset(torch.utils.data.IterableDataset):
    """
    Just transforms a Pytorch dataset. 
    Note that samples are transformed only when they are iterated and there is no caching.
    """

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

    def __len__(self):
        return len(self.D)


class ConcatenateDataset(torch.utils.data.IterableDataset):
    """
    Concatenates multiple datasets in order to construct a Pytorch dataset.
    """

    def __init__(self, datasets: Iterable[PickledDataset], map_labels: List[List] = None):
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
        """ This class automatically assigns a domain number for each dataset.
        This method return the domain number for each sample.

        Returns:
            np.ndarray: a numpy array of the same size as the concatenated datasets.
        """
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
    """
    DEPRECATED
    """

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


# Common transformers used by most datasets
COMMON_TRANSFORMERS = [
    NormalizeSampleRate(97656),
    Split(6101*2),
    FFT(discard_first_points=1),
    StandardScaler(on_field='signal', type='all'),
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
#     Sampling(0.5) # Useful for testing
# ] + COMMON_TRANSFORMERS

RPDBCS_TRANSFORMERS = [StandardScaler(on_field='signal', type='all'),
                       SelectFields('signal', ['label', 'index'])]

IMS_TRANSFORMERS = [
    Sampling(0.2)  # Even with this, it did not work.
] + COMMON_TRANSFORMERS


def custom_xjtu_filter(data: dict):
    """
    Adapts the output of :class:`XJTU_raw`, which outputs a two columns dataframe for each signal,
     into the standard format, which is a 1-d numpy array for each signal.

    TODO: Move this to the XJTU_raw class
    """
    metainfo: pd.DataFrame = data['metainfo']
    mask = metainfo['fault'].notna()
    metainfo = metainfo[mask].copy()
    signal = data['signal'][mask]
    signal = np.hstack(signal).T
    metainfo['label'] = pd.factorize(metainfo['fault'])[0]
    metainfo = pd.DataFrame(metainfo.values.repeat(2, axis=0),
                            columns=metainfo.columns)

    return {'signal': signal,
            'metainfo': metainfo}


XJTU_TRANSFORMERS = [
    FilterByValue(on_field='intensity', values=[0, 100]),
    custom_xjtu_filter,
] + COMMON_TRANSFORMERS
