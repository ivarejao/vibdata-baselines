import torch
from datahandler.datahandler import RawVibrationDataset
import numpy as np
from datahandler.transforms.signal import FilterByValue, Transform


class DLDB_Dataset(torch.utils.data.IterableDataset):
    def __init__(self, raw_dataset: RawVibrationDataset, transforms=None):
        dataset = transforms.transform(raw_dataset.asSimpleForm())
        self.signal_data: np.ndarray = dataset['signal']
        self.labels: np.ndarray = dataset['label']
        self.num_labels = len(np.unique(self.labels))
        # if('label_names' in dataset):
        #     self.labels_names = dataset['label_names']
        # else:
        #     self.labels_names = raw_dataset.getLabelsNames()

    def __getitem__(self, i):
        return self.signal_data[i], self.labels[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __len__(self):
        return len(self.signal_data)

    def getX(self) -> np.ndarray:
        return self.signal_data

    def getLabels(self) -> np.ndarray:
        return self.labels

    def numLabels(self) -> int:
        return self.num_labels


class _OneSampleOneLabel(Transform):
    def transform(self, data):
        data = data.copy()
        metainfo = data['metainfo'].copy(deep=False)
        metainfo['label'] = np.arange(len(metainfo))
        data['metainfo'] = metainfo
        data['label_names'] = np.arange(len(metainfo))
        return data


CWRU_TRANSFORMERS = [
    FilterByValue(on_field='file_name', values=["97.mat", "105.mat", "118.mat", "130.mat", "169.mat", "185.mat",
                                                "197.mat", "209.mat", "222.mat", "234.mat"]),
    FilterByValue(on_field='axis', values='DE'),
]

MFPT_TRANSFORMERS = [FilterByValue(on_field='dir_name', values=["1 - Three Baseline Conditions",
                                                                "3 - Seven More Outer Race Fault Conditions",
                                                                "4 - Seven Inner Race Fault Conditions"]),
                     FilterByValue(on_field='file_name', remove=True,
                                   values=['1 - Three Baseline Conditions/baseline_2.mat', '1 - Three Baseline Conditions/baseline_2.mat']),
                     _OneSampleOneLabel()
                     ]

SEU_TRANSFORMERS = [FilterByValue(on_field='channel', values=1)]
