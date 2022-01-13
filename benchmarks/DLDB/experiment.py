from benchmarks.DLDB.models import CNN_1d, MLP, Alexnet1d, LeNet1d
from datahandler.CWRU import CWRU_raw
from datahandler.MFPT import MFPT_raw
from datahandler.SEU import SEU_raw
from sklearn.model_selection import cross_validate, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from datahandler.transforms.signal import Split, FFT, MinMaxScaler, StandardScaler, asType, SelectFields, ReshapeSingleChannel
import numpy as np
from .datasets import MFPT_TRANSFORMERS, CWRU_TRANSFORMERS, DLDB_Dataset, SEU_TRANSFORMERS
import pandas as pd


def createClassifiers(dataset: DLDB_Dataset):
    import torch
    import skorch

    archs = [Alexnet1d, LeNet1d, MLP, CNN_1d]

    num_classes = dataset.numLabels()
    lr_scheduler = skorch.callbacks.LRScheduler(torch.optim.lr_scheduler.StepLR,
                                                step_size=9, gamma=0.1)
    net_params = {
        'device': 'cuda',
        'criterion': torch.nn.CrossEntropyLoss,
        'max_epochs': 0,
        'batch_size': 64,
        'train_split': None,  # skorch.dataset.ValidSplit(9, stratified=True),
        'optimizer': torch.optim.Adam,
        'callbacks': [lr_scheduler]
    }

    optimizer_params = {
        'weight_decay': 1e-5,
        'lr': 1e-3
    }
    optimizer_params = {"optimizer__"+key: v for key, v in optimizer_params.items()}

    for arch in archs:
        net = skorch.NeuralNetClassifier(module=arch, module__out_channel=num_classes,
                                         **net_params)
        yield (arch.__name__, net)


METRICS = ['accuracy', 'f1_macro']
DATASETS = {
    'cwru': (CWRU_raw, CWRU_TRANSFORMERS),
    'mfpt': (MFPT_raw, MFPT_TRANSFORMERS),
    'seu': (SEU_raw, SEU_TRANSFORMERS)
}
DOMAINS = {'frequency': [Split(1024), FFT()],
           'time': [Split(1024)]}
AUGMENTATIONS = {'none': None}
NORMALIZATIONS = {'mean-std': StandardScaler(on_field='signal', type='row'),
                  '0-1': MinMaxScaler(on_field='signal', type='row'),
                  '-1-1': MinMaxScaler(on_field='signal', type='row', feature_range=(-1, 1))}

CLASSIFIERS = createClassifiers


def getTransforms(dataset, domain, norm_type):
    _, T = DATASETS[dataset]
    T = list(T)  # make a copy
    T += list(DOMAINS[domain])
    T += [NORMALIZATIONS[norm_type]]

    T += [asType(np.float32, on_field='signal'),
          ReshapeSingleChannel(),
          SelectFields('signal', 'label')]
    return make_pipeline(*T)


def loadDataset(dataset_name, root_dir, domain, norm_type, download=True):
    transforms = getTransforms(dataset_name, domain, norm_type)
    Dcls, _ = DATASETS[dataset_name]
    return DLDB_Dataset(Dcls(root_dir=root_dir, download=download), transforms=transforms)


def DLDB_benchmark(data_root_dir):
    Results = []
    metrics_names = [f'{t}_{m}' for m in METRICS for t in ['train', 'test']]
    metrics_names.append('fit_time')
    for domain in DOMAINS:
        for norm_type in NORMALIZATIONS:
            for dname in DATASETS:
                dataset = loadDataset(dname, data_root_dir, domain=domain, norm_type=norm_type)
                for clf_name, clf in CLASSIFIERS(dataset):
                    res = _experiment(dataset, clf)
                    scores = [res[m].mean() for m in metrics_names]
                    Results.append([clf_name, dname, norm_type, domain]+scores)

    return pd.DataFrame(Results,
                        columns=['classifier_name', 'dataset_name', 'norm_type', 'domain']+metrics_names)


def _experiment(dataset, classifier):
    X = dataset.getX()
    y = dataset.getLabels()
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=40)
    results = cross_validate(classifier, X, y, cv=sampler, scoring=METRICS, return_train_score=True)
    return results
