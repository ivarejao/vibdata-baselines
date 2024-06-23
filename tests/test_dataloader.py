import itertools

import numpy as np
import pytest
from vibdata.raw import CWRU_raw
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRatePoly

from vibnet.utils.dataloaders import BalancedSampler, BalancedDataLoader
from vibnet.utils.sklearn_dataset import SklearnDataset


@pytest.fixture
def cwru() -> SklearnDataset:
    """Returns a CWRU dataset"""

    ds = CWRU_raw("./cache/raw", download=True)
    convertDataset(ds, [SplitSampleRate(), NormalizeSampleRatePoly(97656)], dir_path="./cache/deep")
    dataset = DeepDataset("./cache/deep")
    return SklearnDataset(dataset)


def test_sampler(cwru: SklearnDataset):
    sampler = BalancedSampler(cwru, random_state=0)
    for i, sampler_index in enumerate(itertools.islice(sampler, 4)):
        _, y = cwru[sampler_index]
        assert i == y


def test_dataloader(cwru: SklearnDataset):
    dataloader = BalancedDataLoader(cwru, batch_size=20)
    _, y = next(iter(dataloader))
    unique, counts = np.unique(y, return_counts=True)
    assert all(unique == [0, 1, 2, 3])
    assert all(counts == [5, 5, 5, 5])
