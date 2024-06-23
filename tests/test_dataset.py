import numpy as np
import pytest
from vibdata.raw import CWRU_raw
from torch.utils.data import Subset
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRatePoly

from vibnet.utils.sklearn import TrainDataset
from vibnet.utils.sklearn_dataset import SklearnDataset


@pytest.fixture
def cwru() -> SklearnDataset:
    """Returns a CWRU dataset"""

    ds = CWRU_raw("./cache/raw", download=True)
    convertDataset(ds, [SplitSampleRate(), NormalizeSampleRatePoly(97656)], dir_path="./cache/deep")
    dataset = DeepDataset("./cache/deep")
    return SklearnDataset(dataset)


def test_targets(cwru: SklearnDataset):
    targets = cwru.targets
    assert len(cwru) == len(targets)
    assert all(targets == cwru.dataset.metainfo["label"])


def test_subset(cwru: SklearnDataset):
    cwru = TrainDataset(cwru.dataset, standardize=True)
    try:
        x, y = cwru[0]
    except Exception:
        pytest.fail("Dataset not returning tuple on integer index")

    indexes = np.arange(100)
    subset = cwru[indexes]
    assert isinstance(subset, Subset)
    assert len(subset) == len(indexes)
