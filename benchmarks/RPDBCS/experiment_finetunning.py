import os
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
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
from .experiment_dap import ExternalDatasetScoringCallback, ValidGroupSplit, WandbLoggerExtended
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

VIBNET_BB_FPATH = '/tmp/vibnet_backbone.pt'


DATASETS = [
    ('mfpt', MFPT_raw, {}, MFPT_TRANSFORMERS),
    ('cwru', CWRU_raw, {}, CWRU_TRANSFORMERS),
    # ('seu', SEU_raw, SEU_TRANSFORMERS),
    ('pu', PU_raw, {}, PU_TRANSFORMERS),
    ('rpdbcs', RPDBCS_raw, {'frequency_domain': True}, RPDBCS_TRANSFORMERS),
]


class MLP6Classifier(nn.Module):
    def __init__(self, n_classes, input_size, encode_size, activation_function=nn.PReLU(), load_bb_weights_fpath=None) -> None:
        super().__init__()
        if(load_bb_weights_fpath is not None):
            self.backbone = torch.load(load_bb_weights_fpath)
        else:
            self.backbone = RPDBCS2020Net(input_size=input_size, output_size=encode_size,
                                          activation_function=activation_function, random_state=RANDOM_STATE)
        self.classif_head = nn.Linear(encode_size, n_classes)

    def forward(self, X, **kwargs):
        return self.classif_head(self.backbone(X))


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


class MyNet(NeuralNetClassifier):
    def transform(self, X):
        self.module_.transform_mode = True
        ret = self.predict_proba(X)
        self.module_.transform_mode = False
        return ret


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


def _f1macro(yt, yp): return f1_score(yt, yp, average='macro')


class _f1macro_domain:
    def __init__(self, i) -> None:
        self.i = i

    def __call__(self, net, X, y) -> Any:
        yp = net.predict(X)
        domain = np.array([x['domain'] for x, _ in X], dtype=int)
        mask = domain == self.i
        assert(mask.sum() > 0)
        return f1_score(y[mask], yp[mask], average='macro')


def _tripletloss(yt, xe):
    xe = torch.cuda.FloatTensor(xe)
    yt = torch.cuda.IntTensor(yt)
    xe = xe-xe.mean()
    xe = xe/xe.std()
    tripletloss_func = TripletMarginLoss(margin=0.3, reducer=MeanReducer(), triplets_per_anchor=1)

    D = TensorDataset(xe, yt)
    dl = DataLoader(D, batch_size=1024, shuffle=True)

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
    tripletloss_func = TripletMarginLoss(margin=1/1024, reducer=DoNothingReducer(), triplets_per_anchor=1)

    D = TensorDataset(xe, yt)
    dl = DataLoader(D, batch_size=1024, shuffle=True)

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
    for i, dname in enumerate(dataset_names):
        if(dname is None):
            continue

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


def createFineTNet(dataset: PickledDataset, encode_size, finetunning_on=True):
    global WANDB_RUN
    num_classes = len(np.unique(dataset.metainfo['label']))

    module_params = {
        'n_classes': num_classes,
        'load_bb_weights_fpath': VIBNET_BB_FPATH if finetunning_on else None,
        'encode_size': encode_size, 'input_size': dataset[0]['signal'].shape[-1],
    }
    module_params = {"module__"+key: v for key, v in module_params.items()}
    module_params['module'] = MetricNet

    callbacks = [EstimatorEpochScoring(QuadraticDiscriminantAnalysis(), 'f1_macro',
                                       name='QDA_f1_macro', lower_is_better=False, use_caching=False)]
    # callbacks = [
    #     EpochScoring('f1_macro', False, on_train=False, name='valid/f1_macro'),
    #     EpochScoring('f1_macro', False, on_train=True, name='train/f1_macro')]

    callbacks += commonCallbacks()

    callbacks.append(WandbLoggerExtended(WANDB_RUN))

    net_params = DEFAULT_NETPARAMS.copy()
    net_params.update({
        'callbacks': callbacks,
        'criterion': TripletMarginLoss,
        'criterion__margin': 0.5, 'criterion__triplets_per_anchor': 'all', 'criterion__reducer': MeanReducer()
    })
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in net_params.items()})
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in module_params.items()})
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in DEFAULT_OPTIM_PARAMS.items()})
    name = "TripletNet-finetunned"
    # clf = NeuralNetClassifier(**net_params, **module_params, **DEFAULT_OPTIM_PARAMS,
    #                           cache_dir=None)
    clf = NeuralNetTransformer(**net_params, **module_params, **DEFAULT_OPTIM_PARAMS,
                               cache_dir=None)
    clf = Pipeline([('metricnet', clf),
                    ('clf', RandomForestClassifier(n_estimators=300, min_impurity_decrease=1e-5, n_jobs=-1))])
    return (name, clf)


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
        groups_ids = datasets_concat.group_ids
        WANDB_RUN = wandb.init(project="finetunning_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                               job_type='vibnet-train')
        WANDB_RUN.config.update({'source_datasets': source_names})
        WANDB_RUN.config.update(configs)

        with WANDB_RUN:
            torch.manual_seed(RANDOM_STATE)
            torch.cuda.manual_seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)
            vibnet_name, vibnet = createVibnet(datasets_concat, datasets_names,
                                               add_data=(Dtarget, dname))

            # final_dataset = AppendDataset(datasets_concat, {'sample_weight': sample_weights})
            vibnet.fit(datasets_concat, Yc, groups=groups_ids)
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
    netname, net = createFineTNet(Dtarget, finetunning_on=finetunning_on)

    with WANDB_RUN:
        torch.manual_seed(RANDOM_STATE)
        torch.cuda.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        Yc = Dtarget.metainfo['label']
        group_ids = Dtarget.metainfo['index']
        Dtarget_transf = TransformsDataset(Dtarget, _transform_output)
        Dtarget_transf.labels = Yc

        assert(len(np.unique(group_ids)) > 1)
        train_idxs, test_idxs = next(sampler.split(Yc, Yc, group_ids))
        train_idxs = train_idxs[:int(len(train_idxs)*d_percentage)]

        assert(len(train_idxs) > 2)

        Yc_train, Yc_test = Yc[train_idxs], Yc[test_idxs]
        assert(len(np.unique(Yc_train)) > 1)
        Dtarget_train = Subset(Dtarget_transf, train_idxs)
        Dtarget_test = Subset(Dtarget_transf, test_idxs)
        # net.fit(Dtarget_train, Yc_train, groups=group_ids[train_idxs])
        net.fit(Dtarget_train, Yc_train, metricnet__groups=group_ids[train_idxs])

    Results = []

    # if(isinstance(net, NeuralNetTransformer)):
    #     clf = make_pipeline(net,
    #                         RandomForestClassifier(n_estimators=300, min_impurity_decrease=1e-5, n_jobs=-1))
    # else:
    #     clf = net
    Ycp_test = net.predict(Dtarget_test)
    score = f1_score(Yc_test, Ycp_test, average='macro')
    Results.append((netname, source_names, dname, "test", 'fmacro', score, d_percentage, finetunning_on))

    Ycp_train = net.predict(Dtarget_train)
    score = f1_score(Yc_train, Ycp_train, average='macro')
    Results.append((netname, source_names, dname, "train", 'fmacro', score, d_percentage, finetunning_on))

    return Results


def run_experiment(data_root_dir, cache_dir="/tmp/sigdata_cache"):
    sampler = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    # sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)

    # Transform datasets
    datasets_transformed = loadTransformDatasets(data_root_dir, DATASETS, cache_dir)
    datasets_names = [name for name, _ in datasets_transformed]

    # All datasets
    Results = []

    d_percentage_list = [0.1, 0.25, 0.5, 1.0]
    finetunning_on_options = [True, False]

    with tqdm(total=len(datasets_transformed) * len(d_percentage_list) * len(finetunning_on_options)) as pbar:
        for finetunning_on in finetunning_on_options:
            for i, (dname, Dtarget) in enumerate(datasets_transformed):
                train_vibnet = True
                for d_percentage in d_percentage_list:
                    if(d_percentage < 0.3 and len(Dtarget) < 500):
                        continue
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


"""
TODO:
- Testar sem normalização do RPDBCS
- Testar normalizar por sample=4096
- ver questão do sample_rate ser muito alto (94000) para um numero de pontos baixo (6100)
"""
