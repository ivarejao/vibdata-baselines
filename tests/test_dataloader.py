import itertools

import numpy as np
import pytest
from vibdata.raw import CWRU_raw
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRatePoly

from vibnet.utils.dataloaders import BalancedSampler, BalancedDataLoader
from vibnet.utils.MemeDataset import MemeDataset


@pytest.fixture
def cwru() -> MemeDataset:
    """Returns a CWRU dataset"""

    ds = CWRU_raw("./cache/raw", download=True)
    convertDataset(ds, [SplitSampleRate(), NormalizeSampleRatePoly(97656)], dir_path="./cache/deep")
    dataset = DeepDataset("./cache/deep")
    return MemeDataset(dataset)


def test_sampler(cwru: MemeDataset):
    sampler = BalancedSampler(cwru, random_state=0)
    for i, sampler_index in enumerate(itertools.islice(sampler, 4)):
        _, y = cwru[sampler_index]
        assert i == y


def test_dataloader(cwru: MemeDataset):
    dataloader = BalancedDataLoader(cwru, batch_size=20)
    _, y = next(iter(dataloader))
    unique, counts = np.unique(y, return_counts=True)
    assert all(unique == [0, 1, 2, 3])
    assert all(counts == [5, 5, 5, 5])
