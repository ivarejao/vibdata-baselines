from torch.utils.data import Dataset
import pandas as pd

class MemeDataset(Dataset):
    
    def __init__(self, src_dataset):
        self.dataset = src_dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx : int):
        ret = self.dataset[idx]
        X = ret['signal']
        if isinstance(ret['metainfo'], pd.Series):
            y = ret['metainfo']['label']
        else:
            y = ret['metainfo']['label'].values.reshape(-1, 1)
        return X, y