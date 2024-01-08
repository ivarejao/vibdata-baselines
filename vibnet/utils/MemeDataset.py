import pandas as pd
from torch.utils.data import Dataset
from vibdata.deep.DeepDataset import DeepDataset


class MemeDataset(Dataset):
    def __init__(self, src_dataset: DeepDataset, standardize=False):
        self.dataset = src_dataset
        self.standardize = standardize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        ret = self.dataset[idx]
        X = ret["signal"]
        X = X.astype("float64")  # Force the array type
        if isinstance(ret["metainfo"], pd.Series):
            y = ret["metainfo"]["label"]
        else:
            y = ret["metainfo"]["label"].values.reshape(-1, 1)

        if self.standardize:
            y -= self.dataset.metainfo["label"].min()
        y = y.astype("int")  # Force the array type
        return X, y
