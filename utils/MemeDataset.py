import pandas as pd
from torch.utils.data import Dataset


class MemeDataset(Dataset):
    def __init__(self, src_dataset, standardize=False):
        self.dataset = src_dataset
        self.standardize = standardize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        ret = self.dataset[idx]
        X = ret["signal"]
        if isinstance(ret["metainfo"], pd.Series):
            y = ret["metainfo"]["label"]
        else:
            y = ret["metainfo"]["label"].values.reshape(-1, 1)

        if self.standardize:
            y -= self.dataset.metainfo["label"].min()
        return X, y
