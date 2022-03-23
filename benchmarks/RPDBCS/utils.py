import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data.dataset import Subset
from skorch_extra.callbacks import EstimatorEpochScoring
import skorch
from skorch.callbacks.base import Callback
from typing import Any, Callable, Dict, Iterable, List, Tuple
from tqdm import tqdm
import os
from sklearn.pipeline import make_pipeline
from vibdata.datahandler.transforms.TransformDataset import PickledDataset, transform_and_saveDataset
from pytorch_balanced_sampler.sampler import BalancedDataLoader


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
        assert(Y[0] is None)  # FIXME when Y is not None.
        Xtr, Xv = X
        Xtr = Subset(Xtr, np.where(idxs_train)[0])
        Xv = Subset(Xv, np.where(idxs_valid)[0])

        return (Xtr, Xv), Y, P


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
        print(f"Transforming {dname}...")
        dataset_raw = dataset_raw_cls(data_root_dir, download=True, **params)
        data_dir = os.path.join(cache_dir, dname)
        D = transform_and_saveDataset(dataset_raw, transforms, data_dir, batch_size=6000)
        print(">>>Length of %s: %d" % (dname, len(D)))
        datasets_transformed.append((dname, D))
    return datasets_transformed


class SplitLosses(torch.nn.Module):
    def __init__(self, losses_list: List, lambs: List[float] = None,
                 split_x_list: List = None, split_y_list: List = None):
        super().__init__()
        self.losses_list = losses_list
        if(split_y_list is None):
            self.split_y_list = [None]*len(losses_list)
        else:
            self.split_y_list = split_y_list
        if(split_x_list is None):
            self.split_x_list = [None]*len(losses_list)
        else:
            self.split_x_list = split_x_list
        if(lambs is None):
            lambs = [1.0]*len(losses_list)
        self.lambs = lambs
        assert(len(lambs) == len(losses_list))

    def forward(self, X: Tuple, Y):
        def _process_spl(spl, X):
            if(spl is None):
                xs = X
            else:
                if(isinstance(spl, int)):
                    xs = X[spl]
                else:
                    xs = tuple([X[i] for i in spl])
            return xs

        l = torch.tensor(0.0, device=X[0].device)
        ret: Dict[str, Any] = {}
        for lossf, splx, sply, lamb in zip(self.losses_list, self.split_x_list, self.split_y_list, self.lambs):
            if(isinstance(lossf, tuple)):
                loss_name, lossf = lossf
            else:
                loss_name = None

            ys = _process_spl(sply, Y)
            xs = _process_spl(splx, X)

            lll = lossf(xs, ys)
            if(isinstance(lll, dict)):
                lll = lll['loss']
            l += lamb*lll
            if(loss_name is not None):
                ret[loss_name] = lll

        if(len(ret) != 0):
            ret['loss'] = l
            return ret
        return l


class MacroAvgLoss(torch.nn.Module):
    def __init__(self, base_loss: torch.nn.Module, positive_label=0, Y_index=None) -> None:
        super().__init__()
        self.positive_label = positive_label
        self.base_loss = base_loss
        self.Y_index = Y_index

    @staticmethod
    def _maskTuple(X, mask):
        if(isinstance(X, tuple)):
            return tuple([x[mask] for x in X])
        return X[mask]

    def forward(self, X, Y):
        if(self.Y_index is None):
            assert(not isinstance(Y, tuple))
            ylabels = Y
        else:
            ylabels = Y[self.Y_index]

        pos_labels_mask = ylabels == self.positive_label
        neg_labels_mask = ylabels != self.positive_label

        if(pos_labels_mask.sum() >= 1):
            posX = MacroAvgLoss._maskTuple(X, pos_labels_mask)
            posY = MacroAvgLoss._maskTuple(Y, pos_labels_mask)
            loss_pos = self.base_loss(posX, posY)
        else:
            loss_pos = 0.0

        if(neg_labels_mask.sum() >= 1):
            negX = MacroAvgLoss._maskTuple(X, neg_labels_mask)
            negY = MacroAvgLoss._maskTuple(Y, neg_labels_mask)
            loss_neg = self.base_loss(negX, negY)
        else:
            loss_neg = 0.0

        if(isinstance(loss_pos, dict)):
            return {k: (loss_neg[k]+vpos)/2 for k, vpos in loss_pos.items()}
        return (loss_neg+loss_pos)/2


class DomainBalancedDataLoader(BalancedDataLoader):
    @staticmethod
    def _get_labels_domain(dataset):
        labels = BalancedDataLoader._get_labels(dataset)
        domains = DomainBalancedDataLoader._get_domain(dataset)
        return labels
        # m = labels.max()
        # return labels + (m+1)*domains

    @staticmethod
    def _get_domain(dataset):
        if isinstance(dataset, torch.utils.data.Subset):
            domains = DomainBalancedDataLoader._get_domain(dataset.dataset)
            return domains[dataset.indices]

        """
        Guesses how to get the domains.
        """
        if hasattr(dataset, 'getDomains'):
            return dataset.getDomains()
        if hasattr(dataset, 'domains'):
            return dataset.domains
        raise NotImplementedError("DomainBalancedDataLoader: Domains were not found!")

    def __init__(self, dataset, batch_size=1, num_workers=0, collate_fn=None, pin_memory=False, worker_init_fn=None, circular_list=True, shuffle=False, random_state=None, **kwargs):
        super().__init__(dataset, batch_size, num_workers, collate_fn, pin_memory,
                         worker_init_fn, callback_get_label=DomainBalancedDataLoader._get_labels_domain,
                         circular_list=circular_list, shuffle=shuffle, random_state=random_state, **kwargs)
