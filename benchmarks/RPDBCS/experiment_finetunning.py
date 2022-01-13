import os
from typing import Any, List, Sequence, Tuple
from sklearn import metrics
import skorch
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, EpochScoring
from torch.nn.modules.loss import TripletMarginLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, TensorDataset
from vibdata.datahandler.transforms.signal import SelectFields
from .datasets import CWRU_TRANSFORMERS, PU_TRANSFORMERS, SEU_TRANSFORMERS, MFPT_TRANSFORMERS, RPDBCS_TRANSFORMERS, ConcatenateDataset, AppendDataset, TransformsDataset
from .models.RPDBCS2020Net import RPDBCS2020Net, MLP6_backbone, CNN5, SuperFast_backbone, FastRPDBCS2020Net
from vibdata.datahandler import CWRU_raw, PU_raw, MFPT_raw, RPDBCS_raw
from vibdata.datahandler.transforms.TransformDataset import PickledDataset, transform_and_saveDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from skorch_extra.netbase import NeuralNetBase, NeuralNetClassifier
from skorch_extra.callbacks import EstimatorEpochScoring, LoadEndState
from pytorch_balanced_sampler.sampler import BalancedDataLoader
from adabelief_pytorch import AdaBelief
from torch import nn
import torch
from .experiment_dap import ExternalDatasetScoringCallback, WandbLoggerExtended
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import MeanReducer, DoNothingReducer
import wandb
from datetime import datetime

RANDOM_STATE = 42
CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')

VIBNET_BB_FPATH = '/tmp/vibnet_backbone.pt'


class MLP6Classifier(nn.Module):
    def __init__(self, n_classes, input_size, encode_size, activation_function=nn.PReLU(), load_bb_weights_fpath=None) -> None:
        super().__init__()
        if(load_bb_weights_fpath is not None):
            self.backbone = torch.load(load_bb_weights_fpath)
        else:
            self.backbone = RPDBCS2020Net(input_size=input_size, output_size=encode_size,
                                          activation_function=activation_function)
        self.classif_head = nn.Linear(encode_size, n_classes)

    def forward(self, X, **kwargs):
        return self.classif_head(self.backbone(X))


class MLP6ClassifierPerDomain(nn.Module):
    def __init__(self, n_classes, input_size: int, encode_size: int, activation_function=nn.PReLU()) -> None:
        super().__init__()
        self.backbone = RPDBCS2020Net(input_size=input_size, output_size=encode_size,
                                      activation_function=activation_function)
        self.max_nclasses = max(n_classes)
        self.n_classes = n_classes
        self.n_heads = len(n_classes)
        self.classif_heads = nn.Linear(encode_size, self.max_nclasses*self.n_heads)
        self.mask = torch.zeros((self.n_heads, self.max_nclasses), dtype=torch.float32)
        for i, nc in enumerate(self.n_classes):
            self.mask[i, :nc] = 1.0
        self.transform_mode = False

    def forward(self, X, domain, **kwargs):
        X = self.backbone(X)
        if(self.transform_mode):
            return X
        X = self.classif_heads(X).reshape(-1, self.n_heads, self.max_nclasses)
        X = X[:, domain]
        X = X * self.mask[domain].to(X.device)
        X = X[torch.arange(len(X)), domain]
        return X


class MyNet(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss_unreduced = super().get_loss(y_pred, y_true, X=X, training=training)
        if(not isinstance(X, dict)):
            return loss_unreduced.mean()
        sample_weight = X['sample_weight'].to(device=loss_unreduced.device)
        return (sample_weight * loss_unreduced).mean()

    def transform(self, X):
        self.module_.transform_mode = True
        ret = self.predict_proba(X)
        self.module_.transform_mode = False
        return ret


DEFAULT_NETPARAMS = {
    'device': 'cuda',
    'criterion': nn.CrossEntropyLoss,  # 'criterion__reduction': 'none',
    'max_epochs': 100,
    'batch_size': 128,
    'train_split': ValidSplit(cv=0.1, stratified=True, random_state=RANDOM_STATE),
    'iterator_train': BalancedDataLoader,
    # 'iterator_train__num_workers': 6, 'iterator_train__pin_memory': True
    'iterator_train__random_state': RANDOM_STATE,
    'iterator_train__shuffle': True,  # 'iterator_train__prefetch_factor': 22,
    # 'iterator_valid__num_workers': 4, 'iterator_valid__pin_memory': True,
    # 'iterator_valid__prefetch_factor': 32
}

DEFAULT_OPTIM_PARAMS = {'weight_decay': 1e-4, 'lr': 1e-3,
                        'eps': 1e-16, 'betas': (0.9, 0.999),
                        'weight_decouple': False, 'rectify': False,
                        'print_change_log': False}
DEFAULT_OPTIM_PARAMS = {"optimizer__"+key: v for key, v in DEFAULT_OPTIM_PARAMS.items()}
DEFAULT_OPTIM_PARAMS['optimizer'] = AdaBelief


def commonCallbacks() -> List[skorch.callbacks.Callback]:
    from tempfile import mkdtemp

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=mkdtemp(),
                                                      monitor='valid_loss_best',
                                                      f_params='best_epoch_params.pt',
                                                      f_history=None, f_optimizer=None, f_criterion=None)
    loadbest_net_callback = LoadEndState(checkpoint_callback, delete_checkpoint=True)
    earlystop_cb = EarlyStopping('valid_loss', patience=40)
    lr_scheduler_cb = skorch.callbacks.LRScheduler(torch.optim.lr_scheduler.StepLR,
                                                   step_size=20, gamma=0.5)
    return [checkpoint_callback, loadbest_net_callback, earlystop_cb, lr_scheduler_cb]


def _f1macro(yt, yp): return f1_score(yt, yp, average='macro')


def _tripletloss(yt, xe):
    xe = torch.cuda.FloatTensor(xe)
    yt = torch.cuda.IntTensor(yt)
    xe = xe-xe.mean()
    xe = xe/xe.std()
    tripletloss_func = TripletMarginLoss(margin=1.0, reducer=MeanReducer(), triplets_per_anchor=1)

    D = TensorDataset(xe, yt)
    dl = DataLoader(D, batch_size=1024, shuffle=True, drop_last=True)

    bs = 1024
    n = len(xe)//bs
    loss = 0.0
    for xe_i, yt_i in dl:
        try:
            loss += tripletloss_func.forward(xe_i, yt_i)
        except:
            loss += 0.5
    return loss/n


def _positives_triplets_loss(yt, xe):
    xe = torch.cuda.FloatTensor(xe)
    yt = torch.cuda.IntTensor(yt)
    xe = xe-xe.mean()
    xe = xe/xe.std()
    tripletloss_func = TripletMarginLoss(margin=1/1024, reducer=DoNothingReducer(), triplets_per_anchor=1)

    D = TensorDataset(xe, yt)
    dl = DataLoader(D, batch_size=1024, shuffle=True, drop_last=True)
    n = len(xe)//1024

    loss = 0.0
    for xe_i, yt_i in dl:
        try:
            Ls = tripletloss_func.forward(xe_i, yt_i)['loss']['losses']
            loss += (Ls > 0).sum()/len(Ls)
        except:
            loss += 0.5
    return loss/n


def createVibnet(dataset: ConcatenateDataset, dataset_names, add_data: Tuple[PickledDataset, str] = None):
    class _transform_out:
        """
        Renames data
        """

        def __init__(self, i) -> None:
            self.i = i

        def __call__(self, data):
            data['X'] = data['signal']
            del data['signal']
            label = data['label']
            del data['label']
            data['domain'] = self.i
            return data, label

    class transform_classifier_cb:
        def __init__(self, net, **kwargs) -> None:
            self.net = net

        def predict(self, X):
            return self.net.transform(X)

    global WANDB_RUN

    num_classes = [len(np.unique(d.metainfo['label'])) for d in dataset.datasets]

    module_params = {
        'n_classes': num_classes,
        'encode_size': 16, 'input_size': dataset.getInputSize(),
    }
    module_params = {"module__"+key: v for key, v in module_params.items()}
    module_params['module'] = MLP6ClassifierPerDomain

    callbacks = commonCallbacks()
    callbacks.append(EpochScoring('f1_macro', lower_is_better=False, on_train=False,
                                  name='valid/f1_macro'))
    callbacks.append(EpochScoring('f1_macro', lower_is_better=False, on_train=True,
                                  name='train/f1_macro'))

    if(add_data is not None):
        d, dname = add_data
        labels = d.metainfo['label']
        sampler = StratifiedShuffleSplit(n_splits=1, test_size=6000, random_state=RANDOM_STATE)
        _, test_idxs = next(sampler.split(labels, labels))
        d = TransformsDataset(Subset(d, test_idxs), _transform_out(-1))
        labels = labels[test_idxs]
        d.labels = labels
        ext_metrics = [("valid/tripletloss_"+dname, _tripletloss, True),
                       ("valid/non-neg_triplets_"+dname, _positives_triplets_loss, True)]
        score_external = ExternalDatasetScoringCallback(d, labels, classifier_adapter=transform_classifier_cb,
                                                        metrics=ext_metrics)
        callbacks.append(score_external)

    callbacks.append(WandbLoggerExtended(WANDB_RUN))

    net_params = DEFAULT_NETPARAMS.copy()
    net_params['criterion__reduction'] = 'none'
    net_params['callbacks'] = callbacks
    name = "MLP6-Vibnet"
    clf = MyNet(**net_params, **module_params, **DEFAULT_OPTIM_PARAMS)
    return (name, clf)


def createFineTNet(dataset: PickledDataset, finetunning_on=True):
    num_classes = len(np.unique(dataset.metainfo['label']))

    module_params = {
        'n_classes': num_classes,
        'load_bb_weights_fpath': VIBNET_BB_FPATH if finetunning_on else None,
        'encode_size': 16, 'input_size': dataset[0]['signal'].shape[-1],
    }
    module_params = {"module__"+key: v for key, v in module_params.items()}
    module_params['module'] = MLP6Classifier

    # callbacks = [EstimatorEpochScoring(QuadraticDiscriminantAnalysis, 'f1_macro',
    #                                    name='QDA_f1_macro', lower_is_better=False)]
    callbacks = [
        EpochScoring('f1_macro', False, on_train=False, name='valid/f1_macro'),
        EpochScoring('f1_macro', False, on_train=True, name='train/f1_macro')] + commonCallbacks()

    callbacks.append(WandbLoggerExtended(WANDB_RUN))

    net_params = DEFAULT_NETPARAMS.copy()
    net_params.update({
        'iterator_train': BalancedDataLoader,  # 'iterator_train__num_workers': 0, 'iterator_train__pin_memory': False,
        'iterator_train__random_state': RANDOM_STATE,
        'callbacks': callbacks,
        # 'callbacks': [EpochScoring('f1_macro', False)],
    })
    # del net_params['criterion__reduction']
    # net_params['callbacks'] = createCallbacks1(dataset, dataset_names)
    name = "MLP6-finetunned"
    clf = NeuralNetClassifier(**net_params, **module_params, **DEFAULT_OPTIM_PARAMS,
                              cache_dir=None)
    return (name, clf)


DATASETS = [
    # ('cwru', CWRU_raw, {}, CWRU_TRANSFORMERS),
    # ('mfpt', MFPT_raw, {}, MFPT_TRANSFORMERS),
    # ('seu', SEU_raw, SEU_TRANSFORMERS),
    ('pu', PU_raw, {}, PU_TRANSFORMERS),
    ('rpdbcs', RPDBCS_raw, {'frequency_domain': True}, RPDBCS_TRANSFORMERS),
]


def _loadTransformDatasets(data_root_dir, cache_dir=None) -> List[PickledDataset]:
    # ddd = CWRU_raw(data_root_dir)
    # labels = ddd.getMetaInfo()['label']
    datasets_transformed = []
    for dname, dataset_raw_cls, params, transforms in tqdm(DATASETS, desc="Loading and transforming datasets"):
        dataset_raw = dataset_raw_cls(data_root_dir, download=True, **params)
        data_dir = os.path.join(cache_dir, dname)
        D = transform_and_saveDataset(dataset_raw, make_pipeline(*transforms), data_dir)
        datasets_transformed.append((dname, D))
    return datasets_transformed


def _fmacro_perdomain(Yc, Yd, Ycp):
    results = []
    for d in range(len(np.unique(Yd))):
        idxs = Yd == d
        score = f1_score(Yc[idxs], Ycp[idxs], average='macro')
        results.append(score)
    return results


def _transform_output(data: dict):
    """
    Renames data
    """
    label = data['label']
    del data['label']
    data['X'] = data['signal']
    del data['signal']
    return data, label


def run_single_experiment(datasets_concat: ConcatenateDataset, datasets_names, Dtarget: PickledDataset, dname: str,
                          sampler, d_percentage, finetunning_on=True, train_vibnet=True) -> Sequence[Tuple]:
    global WANDB_RUN

    configs = {'target_dataset': dname, 'd_percentage': d_percentage, 'finetunning_on': finetunning_on}

    if(finetunning_on and train_vibnet):
        source_names = "+".join(datasets_names)
        Yc = datasets_concat.getLabels()
        Yd = datasets_concat.getDomains()
        # groups_ids = datasets_concat.group_ids
        WANDB_RUN = wandb.init(project="finetunning_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                               job_type='vibnet-train')
        WANDB_RUN.config.update({'source_datasets': source_names})
        WANDB_RUN.config.update(configs)

        with WANDB_RUN:
            vibnet_name, vibnet = createVibnet(datasets_concat, datasets_names,
                                               add_data=(Dtarget, dname))

            # Sample weights
            weights_cl = np.unique(Yc, return_counts=True)[1]
            weights_cl = weights_cl.sum()/weights_cl
            weights_cl = weights_cl[Yc]
            weights_d = np.unique(Yd, return_counts=True)[1]
            weights_d = weights_d.sum()/weights_d
            weights_d = weights_d[Yd]
            # sample_weights = (weights_cl+weights_d)/2
            sample_weights = weights_d

            final_dataset = AppendDataset(datasets_concat, {'sample_weight': sample_weights})
            # vibnet.fit({'sample_weight': sample_weights, 'X': X, 'domain': Yd}, Yc)
            vibnet.fit(final_dataset, Yc)
        torch.save(vibnet.module_.backbone, VIBNET_BB_FPATH)
        vibnet = None
    elif(finetunning_on == False):
        source_names = "none"
    else:
        source_names = "+".join(datasets_names)

    WANDB_RUN = wandb.init(project="finetunning_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                           job_type='finetunning')
    WANDB_RUN.config.update({'source_datasets': source_names})
    WANDB_RUN.config.update(configs)

    with WANDB_RUN:
        netname, net = createFineTNet(Dtarget, finetunning_on=finetunning_on)

        Yc = Dtarget.metainfo['label']
        group_ids = Dtarget.metainfo['index']
        Dtarget_transf = TransformsDataset(Dtarget, _transform_output)
        Dtarget_transf.labels = Yc

        assert(len(np.unique(group_ids)) > 1)
        train_idxs, test_idxs = next(sampler.split(Yc, Yc, group_ids))
        train_idxs = train_idxs[:int(len(train_idxs)*d_percentage)]
        Yc_train, Yc_test = Yc[train_idxs], Yc[test_idxs]
        Dtarget_train = Subset(Dtarget_transf, train_idxs)
        Dtarget_test = Subset(Dtarget_transf, test_idxs)
        net.fit(Dtarget_train, Yc_train)

    Results = []

    Ycp_test = net.predict(Dtarget_test)
    score = f1_score(Yc_test, Ycp_test, average='macro')
    Results.append((netname, source_names, dname, "test", 'fmacro', score, d_percentage, finetunning_on))

    Ycp_train = net.predict(Dtarget_train)
    score = f1_score(Yc_train, Ycp_train, average='macro')
    Results.append((netname, source_names, dname, "train", 'fmacro', score, d_percentage, finetunning_on))

    return Results


def run_experiment(data_root_dir, cache_dir="/tmp/sigdata_cache"):
    def calculate_metrics(Yc, Yd, Ycp):
        """
        Calculate metrics and add them to the Results list.
        """
        fmacro = f1_score(Yc, Ycp, average='macro')
        if(Yd is None):
            return [fmacro]
        scs = _fmacro_perdomain(Yc, Yd, Ycp)
        assert(len(scs) == len(np.unique(Yd)))
        return [fmacro] + scs

    global WANDB_RUN

    sampler = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    # Transform datasets
    datasets_transformed = _loadTransformDatasets(data_root_dir, cache_dir)
    datasets_names = [name for name, _ in datasets_transformed]

    # All datasets
    Results = []

    d_percentage_list = [1.0]
    finetunning_on_options = [True]

    with tqdm(total=len(datasets_transformed) * len(d_percentage_list) * len(finetunning_on_options)) as pbar:
        for finetunning_on in [True, False]:  # [True, False]:
            for i, (dname, Dtarget) in enumerate(datasets_transformed):
                train_vibnet = True
                for d_percentage in d_percentage_list:
                    pbar.set_description(dname)

                    datasets_names_sources = datasets_names[:i] + datasets_names[i+1:]
                    datasets_concat = ConcatenateDataset([d for j, (_, d) in enumerate(datasets_transformed) if j != i],
                                                         None)

                    R = run_single_experiment(datasets_concat, datasets_names_sources, Dtarget, dname,
                                              sampler, d_percentage, finetunning_on, train_vibnet=train_vibnet)
                    Results += R
                    pbar.update()
                    train_vibnet = False

    columns = ['classifier_name', 'source_dataset', 'target_dataset', 'test_sample', 'metric_name',
               'value', 'd_percentage', 'finetunning_on']
    Results = pd.DataFrame(Results, columns=columns)
    return Results
