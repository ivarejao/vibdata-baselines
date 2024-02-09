import pytest
from torch import nn
from vibdata.raw import CWRU_raw
from torch.utils.data import Subset
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRatePoly

from vibnet.utils.sklearn import SingleSplit, TrainDataset, VibnetEstimator


class Net(nn.Module):
    def __init__(self, out_dim=4):
        super().__init__()

        self.cov = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3, padding=1),
            nn.Conv1d(32, 64, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3, padding=1),
            nn.Conv1d(64, 128, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3, padding=1),
            nn.Conv1d(128, 128, 5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3, padding=1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 15, 64 * 15),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64 * 15, 32 * 15),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32 * 15, out_dim),
        )

    def forward(self, input):
        input = self.cov(input)
        input = self.fc(input)
        return input


@pytest.fixture
def cwru() -> TrainDataset:
    """Returns a CWRU dataset"""

    ds = CWRU_raw("./cache/raw", download=True)
    convertDataset(ds, [SplitSampleRate(), NormalizeSampleRatePoly(97656)], dir_path="./cache/deep")
    dataset = DeepDataset("./cache/deep")
    return TrainDataset(dataset)


def test_split(cwru: TrainDataset):
    net = VibnetEstimator(Net, 4, "", "", train_split=SingleSplit(5))
    dl_train, dl_valid = net._dataloaders(cwru)
    ds_train: Subset = dl_train.dataset
    ds_valid: Subset = dl_valid.dataset

    assert isinstance(ds_train, Subset)
    assert isinstance(ds_valid, Subset)

    total_size = len(ds_train) + len(ds_valid)
    assert total_size == len(cwru), "Dataset size after split is different"

    train_indices = set(ds_train.indices)
    valid_indices = set(ds_valid.indices)
    assert train_indices.isdisjoint(valid_indices), "Train and validation sets are not disjoint"


def test_parameters(cwru: TrainDataset):
    net = VibnetEstimator(
        Net,
        4,
        "",
        "",
        train_split=SingleSplit(5),
        module__out_dim=4,
        optimizer__lr=2,
        iterator_train__batch_size=16,
        iterator_valid__batch_size=32,
    )

    assert net._module_params()["out_dim"] == 4
    assert net._optimizer_params()["lr"] == 2
    assert net._iterator_train_params()["batch_size"] == 16
    assert net._iterator_valid_params()["batch_size"] == 32
