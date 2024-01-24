import numpy as np
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
        X = np.array(X, dtype=float)  # Force the array type
        if isinstance(ret["metainfo"], pd.Series):
            y = ret["metainfo"]["label"]
        else:
            y = ret["metainfo"]["label"].values.reshape(-1, 1)

        if self.standardize:
            y -= self.dataset.metainfo["label"].min()
        y = y.astype("int")  # Force the array type
        return X, y

    @property
    def targets(self) -> np.ndarray[np.int64]:
        """Dataset targets

        Returns:
            Numpy array with dataset labels
        """
        targets = np.array(self.dataset.metainfo["label"], dtype=np.int64)
        if self.standardize:
            targets -= targets.min()
        return targets
