import os
from typing import Any, Iterable, List, Sequence, Tuple, Union
import skorch
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, EpochScoring, PassthroughScoring
from torch.nn.modules.loss import TripletMarginLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, TensorDataset
from vibdata.datahandler.transforms.signal import SelectFields
from .datasets import CWRU_TRANSFORMERS, PU_TRANSFORMERS, SEU_TRANSFORMERS, MFPT_TRANSFORMERS, RPDBCS_TRANSFORMERS, ConcatenateDataset, AppendDataset, TransformsDataset
from .models.RPDBCS2020Net import BigRPDBCS2020Net, RPDBCS2020Net, MLP6_backbone, CNN5, SuperFast_backbone, FastRPDBCS2020Net
from vibdata.datahandler import CWRU_raw, PU_raw, MFPT_raw, RPDBCS_raw
from vibdata.datahandler.transforms.TransformDataset import PickledDataset, transform_and_saveDataset
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from skorch_extra.netbase import NeuralNetBase, NeuralNetClassifier, NeuralNetTransformer
from skorch_extra.callbacks import EstimatorEpochScoring, ExtendedEpochScoring, LoadEndState, _get_labels
from pytorch_balanced_sampler.sampler import BalancedDataLoader
from adabelief_pytorch import AdaBelief
from torch import nn
import torch
from .experiment_dap import ValidGroupSplit, WandbLoggerExtended
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import MeanReducer, DoNothingReducer
import wandb
from datetime import datetime
from pathlib import Path
from .utils import DomainValidSplit, EstimatorDomainScoring, metric_domain, ExternalDatasetScoringCallback, loadTransformDatasets

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')

DATASETS = [
    ('mfpt', MFPT_raw, {}, MFPT_TRANSFORMERS),
    ('cwru', CWRU_raw, {}, CWRU_TRANSFORMERS),
    # ('seu', SEU_raw, SEU_TRANSFORMERS),
    ('pu', PU_raw, {}, PU_TRANSFORMERS),
    ('rpdbcs', RPDBCS_raw, {'frequency_domain': True}, RPDBCS_TRANSFORMERS),
]


class MetricNet(nn.Module):
    def __init__(self, n_classes, input_size, encode_size, activation_function=nn.PReLU(), load_bb_weights_fpath=None) -> None:
        super().__init__()
        if(load_bb_weights_fpath is not None):
            self.backbone = torch.load(load_bb_weights_fpath)
        else:
            self.backbone = RPDBCS2020Net(input_size=input_size, output_size=encode_size,
                                          activation_function=activation_function)

    def forward(self, X, **kwargs):
        return self.backbone(X)


class MLP6ClassifierPerDomain(nn.Module):
    def __init__(self, n_classes, input_size: int, encode_size: int, activation_function=nn.PReLU()) -> None:
        super().__init__()
        self.backbone = RPDBCS2020Net(input_size=input_size, output_size=encode_size,
                                      activation_function=activation_function, random_state=RANDOM_STATE)
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


class NetPerDomain(NeuralNetTransformer):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        return super().get_loss(y_pred, (y_true, X['domain']), X=X, training=training)

    def transform(self, X):
        self.module_.transform_mode = True
        ret = self.predict_proba(X)
        self.module_.transform_mode = False
        return ret


class LossPerDomain(nn.Module):
    def __init__(self, base_loss, **kwargs) -> None:
        super().__init__()
        self.base_loss = base_loss(**kwargs)

    def forward(self, yp, Y):
        yt, domain = Y

        uniq_domains = torch.unique(domain)
        loss = 0.0
        # losses = []
        for d in uniq_domains:
            idxs = domain == d
            ypi, yti = yp[idxs], yt[idxs]
            l = self.base_loss.forward(ypi, yti)
            loss += l
            # losses.append(l)

        # loss = torch.stack(losses).max()
        return loss/len(uniq_domains)


DEFAULT_NETPARAMS = {
    'device': 'cuda',
    'criterion': nn.CrossEntropyLoss,  # 'criterion__reduction': 'none',
    'max_epochs': 100,
    'batch_size': 256,
    'train_split': ValidGroupSplit(cv=0.1, stratified=True, random_state=RANDOM_STATE),
    'iterator_train': BalancedDataLoader,
    # 'iterator_train__num_workers': 6, 'iterator_train__pin_memory': True
    # 'iterator_train__random_state': RANDOM_STATE,
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


def commonCallbacks(to_save_fpath: Union[str, Path] = None) -> List[skorch.callbacks.Callback]:
    from tempfile import mkdtemp

    if(isinstance(to_save_fpath, str) and to_save_fpath is not None):
        to_save_fpath = Path(to_save_fpath)

    checkpoint_callback = skorch.callbacks.Checkpoint(dirname=mkdtemp(),
                                                      monitor='valid_loss_best',
                                                      f_params='best_epoch_params.pt',
                                                      f_history=None, f_optimizer=None, f_criterion=None)
    loadbest_net_callback = LoadEndState(checkpoint_callback, delete_checkpoint=True)
    earlystop_cb = EarlyStopping('valid_loss', patience=50)
    scheduler_policy = torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_scheduler_cb = skorch.callbacks.LRScheduler(scheduler_policy, monitor='valid_loss',
                                                   patience=20, factor=0.1)

    callbacks = [checkpoint_callback, loadbest_net_callback, earlystop_cb, lr_scheduler_cb]
    if(to_save_fpath is not None):
        save_model_cb = skorch.callbacks.TrainEndCheckpoint(f_params=to_save_fpath.name, dirname=str(to_save_fpath.parent),
                                                            f_optimizer=None, f_criterion=None, f_history=None)
        callbacks += [save_model_cb]

    return callbacks


def _tripletloss(yt, xe):
    xe2 = torch.cuda.FloatTensor(xe)
    yt2 = torch.cuda.IntTensor(yt)
    xe2 = xe2-xe2.mean()
    xe2 = xe2/xe2.std()
    tripletloss_func = TripletMarginLoss(margin=0.5, reducer=MeanReducer(), triplets_per_anchor='all')

    D = TensorDataset(xe2, yt2)
    D.labels = yt
    dl = BalancedDataLoader(D, batch_size=256, shuffle=True)

    loss = 0.0
    n = 0
    for xe_i, yt_i in dl:
        try:
            loss += tripletloss_func.forward(xe_i, yt_i)
        except:
            loss += 0.5
        n += 1
    return loss/n


def _positives_triplets_loss(yt, xe):
    xe = torch.cuda.FloatTensor(xe)
    yt = torch.cuda.IntTensor(yt)
    xe = xe-xe.mean()
    xe = xe/xe.std()
    tripletloss_func = TripletMarginLoss(margin=1/1024, reducer=DoNothingReducer(), triplets_per_anchor='all')

    D = TensorDataset(xe, yt)
    D.labels = yt
    dl = BalancedDataLoader(D, batch_size=256, shuffle=True)

    n = 0
    loss = 0.0
    for xe_i, yt_i in dl:
        try:
            Ls = tripletloss_func.forward(xe_i, yt_i)['loss']['losses']
            loss += (Ls > 0).sum()/len(Ls)
        except:
            loss += 0.5
        n += 1
    return loss/n


def _pipelinedQDA(yt, xe):
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    clf = QuadraticDiscriminantAnalysis()
    result = cross_val_score(clf, xe, yt, scoring='f1_macro', cv=sampler)
    return result[0]


def createVibnet(dataset: ConcatenateDataset, dataset_names: Iterable[str],
                 lr, encode_size, margin,
                 add_data: Tuple[PickledDataset, str] = None):
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

    global WANDB_RUN

    num_classes = [len(np.unique(d.metainfo['label'])) for d in dataset.datasets]

    module_params = {
        'n_classes': num_classes,
        'encode_size': encode_size, 'input_size': dataset.getInputSize(),
    }
    module_params = {"module__"+key: v for key, v in module_params.items()}
    module_params['module'] = MetricNet

    callbacks = commonCallbacks('saved_models/tripletnet/vibnet_%s.pt' % "-".join(dataset_names))
    # callbacks.append(EstimatorEpochScoring(QuadraticDiscriminantAnalysis(), 'f1_macro',
    #                                        name='QDA_f1_macro', lower_is_better=False, use_caching=False))
    # domains = np.unique(dataset.getDomains())  # WARNING: relies on the order
    for i, dname in enumerate(dataset_names):
        if(dname is None):
            continue

        # fmacro_d_cb = EpochScoring(_f1macro_domain(domains[i]), lower_is_better=False,
        #                            name='valid/f1_macro_%s' % dname)
        # callbacks.append(fmacro_d_cb)
        cb = EpochScoring(metric_domain(i, _tripletloss), lower_is_better=True,
                          name="valid/triplet_loss_%s" % dname, on_train=False)
        callbacks.append(cb)
        cb = EstimatorDomainScoring(i, dataset.getDomains(),
                                    QuadraticDiscriminantAnalysis(), name='f1_macro_%s' % dname,
                                    metric='f1_macro', lower_is_better=False)
        callbacks.append(cb)

    if(add_data is not None):
        d, dname = add_data
        labels = d.metainfo['label']
        if(len(labels) > 6000):
            sampler = StratifiedShuffleSplit(n_splits=1, test_size=6000, random_state=RANDOM_STATE)
            _, test_idxs = next(sampler.split(labels, labels))
            d = Subset(d, test_idxs)
            labels = labels[test_idxs]
        d = TransformsDataset(d, _transform_out(-1))
        d.labels = labels
        # ext_metrics = [("valid/tripletloss_"+dname, _tripletloss, True),
        #                ("valid/non-neg_triplets_"+dname, _positives_triplets_loss, True)]
        ext_metrics = [("valid/f1_macro_"+dname, _pipelinedQDA, False)]
        score_external = ExternalDatasetScoringCallback(d, labels,  # classifier_adapter=transform_classifier_cb,
                                                        metrics=ext_metrics)
        callbacks.append(score_external)

    callbacks.append(WandbLoggerExtended(WANDB_RUN))

    net_params = DEFAULT_NETPARAMS.copy()
    # net_params['criterion__reduction'] = 'none'
    net_params.update({'criterion': LossPerDomain,
                       'criterion__base_loss': TripletMarginLoss, 'criterion__reducer': MeanReducer(),
                       'criterion__margin': margin, 'criterion__triplets_per_anchor': 'all'
                       })
    net_params['callbacks'] = callbacks
    net_params['train_split'] = DomainValidSplit()
    # net_params['train_split'] = ValidSplit(cv=0.1, stratified=True, random_state=RANDOM_STATE)

    optimizer_params = DEFAULT_OPTIM_PARAMS.copy()
    optimizer_params['optimizer__lr'] = lr

    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in net_params.items()})
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in module_params.items()})
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in optimizer_params.items()})

    name = "Tripletloss-Vibnet"
    clf = NetPerDomain(**net_params, **module_params, **optimizer_params)
    return (name, clf)


def run_single_experiment(datasets_concat: ConcatenateDataset, datasets_names,
                          Dtarget: PickledDataset, dname: str,
                          lr, encode_size, margin, domain_balance_mode: str):
    global WANDB_RUN

    configs = {'domain_balance_mode': domain_balance_mode}

    source_names = "+".join(datasets_names)
    Yc = datasets_concat.getLabels()
    Yd = datasets_concat.getDomains()
    groups_ids = datasets_concat.group_ids
    WANDB_RUN = wandb.init(project="finetunning_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                           job_type='vibnet-train')
    WANDB_RUN.config.update({'source_datasets': source_names})
    WANDB_RUN.config.update(configs)

    with WANDB_RUN:
        r = np.random.randint(100000)
        torch.manual_seed(r)
        torch.cuda.manual_seed(r)
        np.random.seed(r)
        vibnet_name, vibnet = createVibnet(datasets_concat, datasets_names,
                                           lr=lr, encode_size=encode_size, margin=margin,
                                           add_data=(Dtarget, dname))

        # # Sample weights
        # weights_cl = np.unique(Yc, return_counts=True)[1]
        # weights_cl = weights_cl.sum()/weights_cl
        # weights_cl = weights_cl[Yc]
        # weights_d = np.unique(Yd, return_counts=True)[1]
        # weights_d = weights_d.sum()/weights_d
        # weights_d = weights_d[Yd]
        # # sample_weights = (weights_cl+weights_d)/2
        # sample_weights = weights_d

        # final_dataset = AppendDataset(datasets_concat, {'sample_weight': sample_weights})
        if(domain_balance_mode == 'batch'):
            vibnet.fit(datasets_concat, Yc+Yd*10, groups=groups_ids)
        elif(domain_balance_mode == 'loss'):
            vibnet.fit(datasets_concat, Yc, groups=groups_ids)
        else:
            raise ValueError()
    vibnet = None


def run_experiment(data_root_dir, cache_dir="/tmp/sigdata_cache"):
    # Transform datasets
    datasets_transformed = loadTransformDatasets(data_root_dir, DATASETS, cache_dir)
    datasets_names = [name for name, _ in datasets_transformed]

    domain_balance_mode_list = ['loss']
    lr_list = [1e-3]
    encode_size_list = [8, 12, 16, 20, 24, 32, 48]
    margin_list = [0.5]
    reruns = 2

    n_exps = len(datasets_transformed)*len(margin_list)*len(lr_list)*len(encode_size_list)*len(domain_balance_mode_list)

    with open('log.txt', 'w') as f:
        pass

    with tqdm(total=n_exps*reruns) as pbar:
        for rerun_id in range(reruns):
            for i, (dname, Dtarget) in enumerate(datasets_transformed):
                for margin in margin_list:
                    for encode_size in encode_size_list:
                        for lr in lr_list:
                            for domain_balance_mode in domain_balance_mode_list:
                                pbar.set_description("margin=%.2f enc_size=%d lr=%f dmode=%s" %
                                                     (margin, encode_size, lr, domain_balance_mode))

                                datasets_names_sources = datasets_names[:i] + datasets_names[i+1:]
                                datasets_concat = ConcatenateDataset([d for j, (_, d) in enumerate(datasets_transformed) if j != i],
                                                                     None)

                                run_single_experiment(datasets_concat, datasets_names_sources,
                                                      Dtarget, dname,
                                                      lr=lr, encode_size=encode_size, margin=margin, domain_balance_mode=domain_balance_mode)
                                pbar.update()
                                with open('log.txt', 'a') as f:
                                    log_params = (rerun_id, dname, margin, encode_size, lr, domain_balance_mode)
                                    f.write("Finished run %d | dataset %s | margin %.2f | encode_size %d | lr %.5f | dbalance_mode %s\n"
                                            % log_params)


"""
TODO:
- Testar sem normalização do RPDBCS
- Testar normalizar por sample=4096
- ver questão do sample_rate ser muito alto (94000) para um numero de pontos baixo (6100)
"""
