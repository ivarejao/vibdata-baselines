import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data.dataset import Subset
from skorch_extra.callbacks import EstimatorEpochScoring
import skorch
from skorch.callbacks.base import Callback
from typing import Callable, List, Tuple
from tqdm import tqdm
import os
from sklearn.pipeline import make_pipeline
from vibdata.datahandler.transforms.TransformDataset import PickledDataset, transform_and_saveDataset


class DomainValidSplit:
    def __init__(self, random_state=None) -> None:
        self.random_state = random_state

    def __call__(self, dataset, y=None, groups=None):
        domain = np.array([x['domain'] for x, _ in dataset], dtype=int)
        group_ids = np.array([x['index'] for x, _ in dataset], dtype=int)
        sampler = StratifiedGroupKFold(10, shuffle=True, random_state=self.random_state)
        uniq_domains = np.unique(domain)
        train_idxs = []
        test_idxs = []
        for d in uniq_domains:
            d_idxs = np.where(domain == d)[0]
            y_d = y[d_idxs]
            train_idxs_d, test_idxs_d = next(sampler.split(y_d, y_d, group_ids[d_idxs]))
            train_idxs.append(d_idxs[train_idxs_d])
            test_idxs.append(d_idxs[test_idxs_d])

        train_idxs = np.hstack(train_idxs)
        test_idxs = np.hstack(test_idxs)

        return Subset(dataset, train_idxs), Subset(dataset, test_idxs)


class EstimatorDomainScoring(EstimatorEpochScoring):
    def __init__(self, i, domains, estimator, metric='f1_macro', name='score', lower_is_better=False, use_caching=False, on_train=False, **kwargs):
        """
        Args:
            i (int): the desired domain.
            domains (array of ints): array of same size as the dataset, telling the domain each sample belongs.
        """
        super().__init__(estimator, metric, name, lower_is_better, use_caching, on_train, **kwargs)
        self.i = i
        self.domains = domains

    def get_test_data(self, dataset_train: Subset, dataset_valid: Subset):
        dtrain = self.domains[dataset_train.indices]
        dvalid = self.domains[dataset_valid.indices]
        idxs_train = dtrain == self.i
        idxs_valid = dvalid == self.i
        X, Y, P = super().get_test_data(dataset_train, dataset_valid)
        Xtr, Xv = X
        Xtr = Subset(Xtr, np.where(idxs_train)[0])
        Xv = Subset(Xv, np.where(idxs_valid)[0])
        return (Xtr, Xv), Y, P  # FIXME when Y is not None.


class metric_domain:
    """
    Makes any metric to only be used on samples of a certain domain.
    """

    def __init__(self, i, metric: Callable) -> None:
        self.i = i
        self.metric = metric

    def __call__(self, net, X, y):
        yp = net.predict(X)
        domain = np.array([x['domain'] for x, _ in X], dtype=int)
        mask = domain == self.i
        assert(mask.sum() > 0)
        return self.metric(y[mask], yp[mask])


class ExternalDatasetScoringCallback(Callback):
    def __init__(self, X, Y, metrics: List[Tuple], classifier_adapter=lambda net: net, classifier_adapter__kwargs={}) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.classifier_adapter = classifier_adapter
        self.metrics = metrics
        self.best_scores = [np.PINF if lower_is_better else np.NINF for _, _, lower_is_better in metrics]
        self.classifier_adapter__kwargs = classifier_adapter__kwargs

    def on_epoch_end(self, net: skorch.NeuralNet, **kwargs):
        def is_better(v0, v_current, lower_is_better):
            if(lower_is_better):
                return v_current < v0
            return v_current > v0

        clf = self.classifier_adapter(net, **self.classifier_adapter__kwargs)
        yp = clf.predict(self.X)
        for i, (mname, m, lower_is_better) in enumerate(self.metrics):
            value = m(self.Y, yp)
            net.history.record(mname, value)
            better = is_better(self.best_scores[i], value, lower_is_better)
            if(better):
                self.best_scores[i] = value
            net.history.record(mname+"_best", better)


def loadTransformDatasets(data_root_dir, datasets: List[Tuple], cache_dir=None) -> List[PickledDataset]:
    datasets_transformed = []
    for dname, dataset_raw_cls, params, transforms in tqdm(datasets, desc="Loading and transforming datasets"):
        dataset_raw = dataset_raw_cls(data_root_dir, download=True, **params)
        data_dir = os.path.join(cache_dir, dname)
        D = transform_and_saveDataset(dataset_raw, make_pipeline(*transforms), data_dir, batch_size=6000)
        print(">>>Length of %s: %d" % (dname, len(D)))
        datasets_transformed.append((dname, D))
    return datasets_transformed
