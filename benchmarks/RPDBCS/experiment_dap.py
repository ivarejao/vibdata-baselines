from typing import Dict, Iterable, List, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from skorch.callbacks.base import Callback
from .datasets import CWRU_TRANSFORMERS, PU_TRANSFORMERS, SEU_TRANSFORMERS, MFPT_TRANSFORMERS, RPDBCS_TRANSFORMERS, ConcatenateDataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, GroupShuffleSplit, ShuffleSplit, PredefinedSplit, cross_val_predict
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from vibdata.datahandler import CWRU_raw, PU_raw, MFPT_raw
from tqdm import tqdm
import numpy as np
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_balanced_sampler.sampler import BalancedDataLoader
from adabelief_pytorch import AdaBelief
from torch import nn
import skorch
from skorch_extra.netbase import NeuralNetTransformer, NeuralNetBase
from skorch.callbacks import EpochScoring, LRScheduler
from skorch.dataset import ValidSplit
import torch
from sklearn.metrics import f1_score
from domain_adap import DomainAdapLoss, DomainAdapCallbackScore
import wandb
from wandb.sdk import wandb_run
from skorch.callbacks import WandbLogger
from datetime import datetime
from .models.GradRev_models import DomainAdapNet, DomainAdapNetConv


CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')
RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)
# torch.manual_seed(RANDOM_STATE)
# torch.cuda.manual_seed(RANDOM_STATE)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False


class ValidGroupSplit(ValidSplit):
    def _check_cv_float(self):
        assert(self.stratified == False)
        return GroupShuffleSplit(test_size=self.cv, random_state=self.random_state)


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


class WandbLoggerExtended(WandbLogger):
    def __init__(self, wandb_run: wandb_run.Run, keys_ignored=None, prefix=""):
        self.wandb_run = wandb_run
        super().__init__(wandb_run, save_model=False, keys_ignored=keys_ignored)
        self.prefix = prefix

    def on_epoch_end(self, net, **kwargs):
        def rename_key_value(key):
            if(key == 'valid_loss'):
                return 'valid/loss'
            elif(key == 'train_loss'):
                return 'train/loss'
            return key
        """Log values from the last history step and save best model"""
        hist = net.history[-1]
        keys_kept = skorch.callbacks.logging.filter_log_keys(hist, keys_ignored=self.keys_ignored_)
        logged_vals = {self.prefix+rename_key_value(k): hist[k] for k in keys_kept}
        self.wandb_run.log(logged_vals)


class MyNet(NeuralNetBase):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss_unreduced = super().get_loss(y_pred, y_true, X=X, training=training)
        sample_weight = X['sample_weight']
        sample_weight = sample_weight.to(device=loss_unreduced.device)
        return (sample_weight * loss_unreduced).mean()


class MyDomainAdaptationClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, net: skorch.NeuralNet):
        super().__init__()
        self.net = net

    def fit(self, X, y, **kwargs):
        self.net.fit(X, y, **kwargs)
        return self

    # def predict_proba(self, X):
    #     _, ypc = self.net.forward(X)
    #     return ypc

    def predict(self, X):
        _, ypc = self.net.forward(X)
        return ypc.argmax(dim=1)


def _f1macro(yt, yp): return f1_score(yt, yp, average='macro')


def _minf1score(yt, yp): return f1_score(yt, yp, average=None).min()


# def _mostpredictclass(yt, yp): return np.unique(yp, return_counts=True)[1].max()/len(yp)

def createCallbacks1(dataset: ConcatenateDataset, dataset_names):
    global WANDB_RUN

    lr_scheduler = LRScheduler(torch.optim.lr_scheduler.StepLR,
                               step_size=40, gamma=0.1)
    callbacks = []
    for t in ['train', 'valid']:
        on_train = t == 'train'
        f1macro_cl_callback = EpochScoring(DomainAdapCallbackScore(_f1macro, on_class=True),
                                           lower_is_better=False, on_train=on_train,
                                           name='%s/cl_f1score' % t)
        f1macro_da_callback = EpochScoring(DomainAdapCallbackScore(_f1macro, on_class=False),
                                           lower_is_better=False, on_train=on_train,
                                           name='%s/da_f1score' % t)
        minfscore_da_callback = EpochScoring(DomainAdapCallbackScore(_minf1score, on_class=False),
                                             lower_is_better=False, on_train=on_train,
                                             name='%s/da_min-f1score' % t)
        minfscore_cl_callback = EpochScoring(DomainAdapCallbackScore(_minf1score, on_class=True),
                                             lower_is_better=False, on_train=on_train,
                                             name='%s/min-f1score_cl' % t)

        # mostpredict_cl_callback = EpochScoring(DomainAdapCallbackScore(_mostpredictclass, on_class=True),
        #                                        lower_is_better=False, on_train=on_train,
        #                                        name='mostpredicted_cl_%s' % t)
        callbacks += [f1macro_cl_callback, f1macro_da_callback]
        callbacks += [minfscore_cl_callback, minfscore_da_callback]
    for d, dname in zip(dataset.datasets, dataset_names):
        if(dname is None):
            continue
        fmacro_external = ExternalDatasetScoringCallback(d.getX(), d.getLabels(),
                                                         metrics=[("valid/f1score_"+dname, _f1macro, False)],
                                                         classifier_adapter=MyDomainAdaptationClassifier)
        callbacks.append(fmacro_external)
    callbacks.append(WandbLoggerExtended(WANDB_RUN))
    return callbacks


def createClassifiers(dataset: ConcatenateDataset, dataset_names) -> Iterable:
    num_classes = dataset.numLabels()

    default_net_params = {
        'device': 'cuda',
        'criterion': DomainAdapLoss, 'criterion__alpha': 1.0,  # 'criterion__reduction': 'none',
        'max_epochs': 20,
        'batch_size': 256,
        'train_split': ValidGroupSplit(0.1, stratified=False),
        # 'iterator_train': BalancedDataLoader, 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False, 'iterator_train__random_state': RANDOM_STATE,
    }

    default_module_params = {
        'n_domains': len(dataset.datasets)-1, 'n_classes': num_classes,
        'output_size': 32, 'input_size': dataset.getInputSize(),
    }
    default_module_params = {"module__"+key: v for key, v in default_module_params.items()}
    default_module_params['module'] = DomainAdapNet

    default_optimizer_params = {'weight_decay': 1e-4, 'lr': 1e-4,
                                'eps': 1e-16, 'betas': (0.9, 0.999),
                                'weight_decouple': False, 'rectify': False,
                                'print_change_log': False}
    default_optimizer_params = {"optimizer__"+key: v for key, v in default_optimizer_params.items()}
    default_optimizer_params['optimizer'] = AdaBelief

    module_params = default_module_params.copy()
    module_params['module__gradient_rev_lambda'] = 1.0
    net_params = default_net_params.copy()
    net_params['callbacks'] = createCallbacks1(dataset, dataset_names)
    name = "MLP_GradRev"
    net = MyNet(**net_params, **module_params, **default_optimizer_params)
    clf = MyDomainAdaptationClassifier(net)
    yield (name, clf)

    module_params = default_module_params.copy()
    module_params['module__gradient_rev_lambda'] = 0.0
    net_params = default_net_params.copy()
    net_params['callbacks'] = createCallbacks1(dataset, dataset_names)
    name = "MLP"
    net = MyNet(**net_params, **module_params, **default_optimizer_params)
    clf = MyDomainAdaptationClassifier(net)

    yield (name, clf)

    module_params = default_module_params.copy()
    module_params['module__gradient_rev_lambda'] = 1.0
    module_params['module'] = DomainAdapNetConv
    net_params = default_net_params.copy()
    net_params['callbacks'] = createCallbacks1(dataset, dataset_names)
    name = "IJCNN2020net_GradRev"
    net = MyNet(**net_params, **module_params, **default_optimizer_params)
    clf = MyDomainAdaptationClassifier(net)
    yield (name, clf)

    module_params = default_module_params.copy()
    module_params['module__gradient_rev_lambda'] = 0.0
    module_params['module'] = DomainAdapNetConv
    net_params = default_net_params.copy()
    net_params['callbacks'] = createCallbacks1(dataset, dataset_names)
    name = "IJCNN2020net"
    net = MyNet(**net_params, **module_params, **default_optimizer_params)
    clf = MyDomainAdaptationClassifier(net)
    yield (name, clf)


CLASSIFIER_NAMES = ['MLP_GradRev', 'MLP', 'IJCNN2020net_GradRev', 'IJCNN2020net']
# CLASSIFIER_NAMES = ['IJCNN2020net_GradRev','IJCNN2020net']

DATASETS = [
    ('cwru', CWRU_raw, CWRU_TRANSFORMERS),
    ('mfpt', MFPT_raw, MFPT_TRANSFORMERS),
    # ('seu', SEU_raw, SEU_TRANSFORMERS),
    ('pu', PU_raw, PU_TRANSFORMERS),
    # ('rpdbcs', RPDBCS_raw, RPDBCS_TRANSFORMERS)
]


def _loadTransformDatasets(data_root_dir, cache_dir=None):
    datasets_transformed = []
    for dname, dataset_raw_cls, transforms in tqdm(DATASETS, desc="Loading and transforming datasets"):
        dataset_raw = dataset_raw_cls(data_root_dir, download=True)
        dataset = TransformDataset(dataset_raw, transforms, cache_dir=cache_dir)
        datasets_transformed.append((dname, dataset))

    return datasets_transformed


def _log_config():
    wandb.config.update({
        'Datasets': [name for name, _, _ in DATASETS]
    })


def run_experiment(data_root_dir, cache_dir="/tmp/sigdata_cache"):
    global WANDB_RUN

    # Transform datasets
    datasets_transformed = _loadTransformDatasets(data_root_dir, cache_dir)
    datasets_len = [len(d) for _, d in datasets_transformed]
    datasets_names = [name for name, _ in datasets_transformed]

    # Concatenate datasets
    datasets_concat = ConcatenateDataset([d for _, d in datasets_transformed], None)
    X = datasets_concat.getX()
    Yc = datasets_concat.getLabels()
    Yd = datasets_concat.getDomains()
    groups_ids = datasets_concat.group_ids

    # Sample weights
    weights_cl = np.unique(Yc, return_counts=True)[1]
    weights_cl = weights_cl.sum()/weights_cl
    weights_cl = weights_cl[Yc]
    weights_d = np.unique(Yd, return_counts=True)[1]
    weights_d = weights_d.sum()/weights_d
    weights_d = weights_d[Yd]
    sample_weights = (weights_cl+weights_d)/2

    Results = []
    # One-against-All
    with tqdm(total=len(datasets_transformed)*len(CLASSIFIER_NAMES)) as pbar:
        for i, (target_dname, _) in tqdm(enumerate(datasets_transformed)):
            pbar.set_description(target_dname)

            # Remove dataset i from training dataset
            test_idxs = -np.ones(len(datasets_concat))
            begin_idx = sum(datasets_len[:i])
            end_idx = begin_idx + datasets_len[i]
            test_idxs[begin_idx:end_idx] = 0
            sampler = PredefinedSplit(test_idxs)

            target_datasets_names = [None]*len(datasets_names)
            target_datasets_names[i] = datasets_names[i]

            # train/test split
            train_idxs, test_idxs = next(sampler.split(X, Yc))
            randomstate = np.random.RandomState(RANDOM_STATE)
            randomstate.shuffle(train_idxs)
            Yd_train, Yc_train = Yd[train_idxs], Yc[train_idxs]
            Yc_test = Yc[test_idxs]
            X_train, X_test = X[train_idxs], X[test_idxs]
            groups_ids_train = groups_ids[train_idxs]
            sample_weights_train = sample_weights[train_idxs]

            # Maps domains values to {0,1,2,3,...}
            _, newYd = np.unique(Yd_train, return_inverse=True)
            Ytrain = np.stack([newYd, Yc_train]).T

            clfs = createClassifiers(datasets_concat, target_datasets_names)
            source_datasets_names = ",".join(datasets_names[:i]+datasets_names[i+1:])
            for clf_name in CLASSIFIER_NAMES:
                WANDB_RUN = wandb.init(project="domain_adap_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                                       job_type=source_datasets_names)
                WANDB_RUN.config.update({'estimator_name': clf_name})
                with WANDB_RUN:
                    clf_name2, clf = next(clfs)
                    assert(clf_name == clf_name2)

                    # R = cross_validate(clf, X, Y, cv=sampler, scoring=METRICS, return_train_score=True)
                    # scores = [R[m].mean() for m in metrics_names]

                    # Train and test
                    clf.fit({'X': X_train, 'sample_weight': sample_weights_train},
                            Ytrain, groups=groups_ids_train)
                    # clf.fit(X_train, Ytrain)
                    Yp = clf.predict(X_test)
                    score = f1_score(Yc_test, Yp, average='macro')
                    Results.append([clf_name, datasets_names[:i]+datasets_names[i+1:], target_dname]+[score])

                    pbar.update()

    Results = pd.DataFrame(Results,
                           columns=['classifier_name', 'source_dataset', 'target_dataset'] + ['f1_macro'])
    return Results
