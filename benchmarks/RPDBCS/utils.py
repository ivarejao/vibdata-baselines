from collections import defaultdict
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
    """
    This class is used as a callback for splitting dataset into train and valid dataset when using skorch.
    It splits the training dataset into `cv_size` stratified folds with respect to labels and domains, that is,
    each fold contains the same proportion found in the whole dataset for domains and labels.
    """

    def __init__(self, cv_size=10, random_state=None) -> None:
        self.random_state = random_state
        self.cv_size = cv_size

    def __call__(self, dataset, y=None, groups=None):
        domain = np.array([x['domain'] for x, _ in dataset], dtype=int)
        group_ids = np.array([x['index'] for x, _ in dataset], dtype=int)

        # We are using StratifiedGroupKFold here, but we will only get the first fold.
        sampler = StratifiedGroupKFold(self.cv_size, shuffle=True, random_state=self.random_state)
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
    """
    This an adaptation of `EstimatorEpochScoring` for selecting a single domain for using the EstimatorEpochScoring.
    It just runs the EstimatorEpochScoring for a single specified domain of choice.
    """

    def __init__(self, i, domains, estimator, metric='f1_macro', name='score', lower_is_better=False, use_caching=False, on_train=False, **kwargs):
        """
        Args:
            i (int): the desired domain.
            domains (array of ints): array of same size as the dataset, telling what domain each sample belongs.
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
    Makes any metric to be only used on samples of a certain domain.
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
    """
    A skoch callback for testing the neural net on an external dataset, at each epoch.
    """

    def __init__(self, X, Y, metrics: List[Tuple], classifier_adapter=lambda net: net, classifier_adapter__kwargs={}) -> None:
        """

        Args:
            X (numpy array, pytorch tensor or pytorch dataset): 
            Y (numpy array, pytorch tensor or None): 
            metrics (List[Tuple]): a list of tuples of three elements, where the first element is any arbitrary metric name, the second element is the metric (a callable with two parameters)
                and the third element is a boolean telling if `lower_is_better`. Example: 
                >>> metrics=[("valid/accuracy", accuracy_cb, False)
            classifier_adapter: parameter for changing the prediction of the original network. 
            It is a callback function that receives the `skorch.NeuralNet` and returns a sklearn estimator.
                Note that `skorch.NeuralNet` is already a sklearn estimator, so the user don't need to use this parameters, 
                unless the original network prediction needs some adaptation.
            Defaults to lambdanet:net.
            classifier_adapter__kwargs: passed to the `classifier_adapter` callback function. Defaults to {}.
        """
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
    """Loads and transforms datasets. 

    Args:
        data_root_dir (str): path to the datasets.
        datasets (List[Tuple]): List of `RawVibrationDataset`.
        cache_dir (str, optional): path to a directory, to use as cache for the transformed datasets. Defaults to None.

    Returns:
        List[PickledDataset]: _description_
    """
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
    """When the network arch has multiple outputs and/or there are multiple output labels (e.g, domain label and class label),
    and/or there are multiple losses, this class is useful to split each loss into their respective input/output.
    Example:
    >>> loss1 = torch.nn.CrossEntropyLoss()
    ... loss2 = pytorch_metric_learning.losses.TripletMarginLoss()
    ... loss_combined = SplitLosses(losses_list=[loss1, loss2], split_x_list=[0,1], lambs=[0.5,1.0])
    This example combines cross entropy loss and triplet loss by applying cross entropy to the first output (index 0) 
    and triplet loss to the second output (index 1). Then, the first loss is multiplied by 0.5 while the second is multiplied by 1.0. 
    Note the the network arch (torch.nn.Module) should output (.forward method) a tuple of two tensors.
    """

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
    """It applies a specified loss twice, one for the positive label and one for the negative label, and then averages it.
    The negative label is all other labels not designated as positive by the user.
    Note: the negative labels are passed to the loss in their original form, that is, they are NOT converted to a single integer value.
    """

    def __init__(self, base_loss: torch.nn.Module, positive_label=0, Y_index=None) -> None:
        """
        Args:
            base_loss (torch.nn.Module): Any pytorch loss
            positive_label (int, optional): The integer defining which label is the positive label. Defaults to 0.
            Y_index (_type_, optional): If the dataset output multiple columns labels (example: a domain vector and a class label vector), 
                this parameter indicates which column to consider. None means there is a single column. Defaults to None.
        """
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
            if(isinstance(loss_neg, dict)):
                return {k: (loss_neg[k]+vpos)/2 for k, vpos in loss_pos.items()}
            return {k: (loss_neg+vpos)/2 for k, vpos in loss_pos.items()}
        if(isinstance(loss_neg, dict)):
            return {k: (loss_pos+vneg)/2 for k, vneg in loss_neg.items()}
        return (loss_neg+loss_pos)/2


class DomainBalancedDataLoader(BalancedDataLoader):
    """
    A dataloader that balances both labels and domains.

    """
    @staticmethod
    def _get_labels_domain(dataset):
        labels = BalancedDataLoader._get_labels(dataset)
        domains = DomainBalancedDataLoader._get_domain(dataset)
        m = labels.max()
        return labels + (m+1)*domains

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

