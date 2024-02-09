import numpy as np
import pytest
from vibdata.raw import CWRU_raw
from torch.utils.data import Subset
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRatePoly

from vibnet.utils.sklearn import TrainDataset
from vibnet.utils.MemeDataset import MemeDataset


@pytest.fixture
def cwru() -> MemeDataset:
    """Returns a CWRU dataset"""

    ds = CWRU_raw("./cache/raw", download=True)
    convertDataset(ds, [SplitSampleRate(), NormalizeSampleRatePoly(97656)], dir_path="./cache/deep")
    dataset = DeepDataset("./cache/deep")
    return MemeDataset(dataset)


def test_targets(cwru: MemeDataset):
    targets = cwru.targets
    assert len(cwru) == len(targets)
    assert all(targets == cwru.dataset.metainfo["label"])


def test_subset(cwru: MemeDataset):
    cwru = TrainDataset(cwru.dataset, standardize=True)
    try:
        x, y = cwru[0]
    except Exception:
        pytest.fail("Dataset not returning tuple on integer index")

    indexes = np.arange(100)
    subset = cwru[indexes]
    assert isinstance(subset, Subset)
    assert len(subset) == len(indexes)
