from sklearn.base import BaseEstimator, ClassifierMixin
import os
from typing import Any, Callable, Iterable, List, Sequence, Tuple, Union
import skorch
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping, EpochScoring, PassthroughScoring
from torch.nn.modules.loss import TripletMarginLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset, TensorDataset
from vibdata.datahandler.transforms.signal import SelectFields
from .datasets import COMMON_TRANSFORMERS, CWRU_TRANSFORMERS, IMS_TRANSFORMERS, PU_TRANSFORMERS, SEU_TRANSFORMERS, MFPT_TRANSFORMERS, RPDBCS_TRANSFORMERS, XJTU_TRANSFORMERS,\
    ConcatenateDataset, AppendDataset, TransformsDataset
from .models.RPDBCS2020Net import RPDBCS2020Net, RPDBCS2020NetWithMixStyle
from vibdata.datahandler import CWRU_raw, PU_raw, MFPT_raw, RPDBCS_raw, IMS_raw, UOC_raw, XJTU_raw
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
from .utils import DomainValidSplit, EstimatorDomainScoring, ExternalDatasetScoringCallback, loadTransformDatasets, SplitLosses, MacroAvgLoss, DomainBalancedDataLoader
from domain_adap import LossPerDomain, DomainRecognizerCallback
from .coral import CoralLoss
from pathlib import Path
from barlow_twins import BarlowTwins

"""
Experiments for domain adaptation using Coral and Barlow Twins
"""

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
CURRENT_TIME = datetime.now().strftime('%b%d_%H-%M-%S')

VIBNET_BB_FPATH = '/tmp/vibnet_backbone.pt'
MODEL_SAVE_DIR = Path('saved_models/multiple_coral_losses/')


# Defining the datasets and transformers to use.
# Tuple: (name, dataset class, params to be passed to __init__, transformers)
DATASETS = [
    ('mfpt', MFPT_raw, {}, MFPT_TRANSFORMERS),
    ('cwru', CWRU_raw, {}, CWRU_TRANSFORMERS),
    # ('seu', SEU_raw, SEU_TRANSFORMERS),
    ('XJTU', XJTU_raw, {}, XJTU_TRANSFORMERS),
    # ('IMS', IMS_raw, {}, IMS_TRANSFORMERS),
    ('UOC', UOC_raw, {}, COMMON_TRANSFORMERS),
    ('pu', PU_raw, {}, PU_TRANSFORMERS),
    ('rpdbcs', RPDBCS_raw, {'frequency_domain': True}, RPDBCS_TRANSFORMERS),
]


class RPDBCS2020NetLoadable(nn.Module):
    def __init__(self, input_size, encode_size, activation_function=nn.PReLU(),
                 load_bb_weights_fpath=None, single_output=True) -> None:
        super().__init__()
        if(load_bb_weights_fpath is not None):
            self.backbone = torch.load(load_bb_weights_fpath)
        else:
            self.backbone = RPDBCS2020Net(input_size=input_size, output_size=encode_size,
                                          activation_function=activation_function, single_output=single_output)

    def forward(self, X, **kwargs):
        return self.backbone(X)


class MetricNetPerDomain(nn.Module):
    """
    This arch has a head for each domain.
    """

    def __init__(self, backbone, n_domains, input_size, encode_size, activation_function=nn.PReLU(), load_bb_weights_fpath: str = None,
                 head_encode_size=8, all_outputs=True, **kwargs) -> None:
        """

        Args:
            backbone (class of nn.Module): must have `input_size` and `output_size` as parameters of `__init__`
            n_domains (int): number of distinct domains.
            input_size (int): backbone input size.
            encode_size (int): output size of the backbone.
            activation_function (torch.nn.Module, optional): activation function. Defaults to nn.PReLU().
            load_bb_weights_fpath (str, optional): path to load the backbone, instead of constructing a new one. Defaults to None.
            head_encode_size (int, optional): encode_size of each domain head. Defaults to 8.
            all_outputs (bool, optional): If true, it will output the backbone output alongside with the heads output. Defaults to True.
        """
        super().__init__()
        if(load_bb_weights_fpath is not None):
            self.backbone = torch.load(load_bb_weights_fpath)
        else:
            self.backbone = backbone(input_size=input_size, output_size=encode_size, **kwargs)
        self.head_encode_size = head_encode_size
        self.heads_reg = nn.Sequential(activation_function,
                                       nn.Linear(encode_size, head_encode_size*n_domains))
        self.n_domains = n_domains
        self.transform_mode = False
        self.all_outputs = all_outputs

    def forward(self, X, domain, **kwargs):
        X2, X1 = self.backbone(X)
        # X2 = self.backbone(X)
        if(self.transform_mode):
            return X2
        Xc = self.heads_reg(X2).reshape(-1, self.n_domains, self.head_encode_size)
        Xc = Xc[torch.arange(len(Xc)), domain]

        if(self.all_outputs):
            return Xc, X2, X1
        return Xc


class NeuralNetDomainAdapter(NeuralNetTransformer):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        return super().get_loss(y_pred, (X['domain'], y_true), X=X, training=training)

    def transform(self, X):
        self.module_.transform_mode = True
        ret = self.predict_proba(X)
        self.module_.transform_mode = False
        return ret


DEFAULT_NETPARAMS = {
    'device': 'cuda',
    'max_epochs': 100,
    'batch_size': 256,
    'train_split': ValidGroupSplit(cv=0.1, stratified=True, random_state=RANDOM_STATE),
    'iterator_train': BalancedDataLoader,
    'iterator_train__num_workers': 4, 'iterator_train__pin_memory': True,
    # 'iterator_train__random_state': RANDOM_STATE,
    'iterator_train__shuffle': True,
    'iterator_valid__num_workers': 4, 'iterator_valid__pin_memory': True,
}

DEFAULT_OPTIM_PARAMS = {'weight_decay': 1e-4, 'lr': 1e-3,
                        'eps': 1e-16, 'betas': (0.9, 0.999),
                        'weight_decouple': False, 'rectify': False,
                        'print_change_log': False}
DEFAULT_OPTIM_PARAMS = {"optimizer__"+key: v for key, v in DEFAULT_OPTIM_PARAMS.items()}
DEFAULT_OPTIM_PARAMS['optimizer'] = AdaBelief


def commonCallbacks(to_save_fpath: Union[str, Path] = None) -> List[skorch.callbacks.Callback]:
    """Builds common callbacks:
    - The `skorch.callbacks.Checkpoint` callback, which saves the model every epoch in which the valid_loss is improved compared with the best one.
    - The `skorch_extra.callbacks.LoadEndState` callback, which loads the weights of the best epoch, after training.
    - The `EarlyStopping` callback, which stops training after 50 epochs of no improvement.
    - The `LRScheduler` callback, which multiplies the learning rate by 0.1 every 20 epochs.
    - The `TrainEndCheckpoint` callback, which saves the model in a specified path, if parameter `to_save_fpath` is not None.

    Args:
        to_save_fpath (Union[str, Path], optional): path to save the model after training. Defaults to None.

    Returns:
        List[skorch.callbacks.Callback]: a list of callbacks.
    """
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

    callbacks = [checkpoint_callback, loadbest_net_callback, earlystop_cb,
                 lr_scheduler_cb]
    if(to_save_fpath is not None):
        save_model_cb = skorch.callbacks.TrainEndCheckpoint(f_params=to_save_fpath.name, dirname=str(to_save_fpath.parent),
                                                            f_optimizer=None, f_criterion=None, f_history=None)
        callbacks += [save_model_cb]

    return callbacks


def _pipelinedQDA(yt, xe):
    sampler = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    clf = QuadraticDiscriminantAnalysis(tol=1e-10)
    result = cross_val_score(clf, xe, yt, scoring='f1_macro', cv=sampler)
    return result[0]


SAVE_COUNT = 0


def createVibnet(dataset: ConcatenateDataset, dataset_names: Iterable[str],
                 lr, encode_size, margin, lamb=1.0,
                 add_data: Tuple[PickledDataset, str] = None,
                 backbone=RPDBCS2020Net):
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

    global WANDB_RUN, SAVE_COUNT

    ## Network arch parameters ##
    module_params = {
        'n_domains': len(dataset_names),
        'encode_size': encode_size, 'input_size': dataset.getInputSize(),
        'all_outputs': True,
        'backbone': backbone,
        'single_output': False
    }
    module_params = {"module__"+key: v for key, v in module_params.items()}
    module_params['module'] = MetricNetPerDomain
    #############################

    saved_model_fname = 'vibnet_%d.pt' % SAVE_COUNT
    callbacks = commonCallbacks(MODEL_SAVE_DIR/saved_model_fname)
    append_or_write = 'w' if SAVE_COUNT == 0 else 'a'
    SAVE_COUNT += 1

    #### Adding callbacks ####
    for i, dname in enumerate(dataset_names):
        # Estimates f1_macro for each domain
        cb = EstimatorDomainScoring(i, dataset.getDomains(),
                                    QuadraticDiscriminantAnalysis(tol=1e-10), name='f1_macro_head_%s' % dname,
                                    metric='f1_macro', lower_is_better=False, use_transform=False, on_train=False)
        callbacks.append(cb)

    # Classifier the uses the net.transform() to recognize domains.
    cb = DomainRecognizerCallback(dataset.getDomains(),
                                  QuadraticDiscriminantAnalysis(tol=1e-10), name='f1_macro_dreg',
                                  metric='f1_macro', lower_is_better=False, use_transform=True)
    callbacks.append(cb)

    if(add_data is not None):
        d, dname = add_data
        labels = d.metainfo['label']
        if(len(labels) > 6000):  # testing on at most 6000 samples, otherwise, the validation will take too much long.
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

    ### Coral loss for each pair of domains ###
    '''
    uniq_domains = np.unique(dataset.getDomains())
    for i in range(len(uniq_domains)):
        for j in range(i+1, len(uniq_domains)):
            callbacks.append(PassthroughScoring('valid/coralloss_%d_%d' %
                             (uniq_domains[i], uniq_domains[j]), on_train=False))
            callbacks.append(PassthroughScoring('train/coralloss_%d_%d' %
                             (uniq_domains[i], uniq_domains[j]), on_train=True))
    '''

    losses_names = ['coralloss', 'clf_loss', 'meanloss', 'covloss', 'bw_loss']
    for lname in losses_names:
        callbacks.append(PassthroughScoring('valid/%s' % lname, on_train=False))
        callbacks.append(PassthroughScoring('train/%s' % lname, on_train=True))
    callbacks.append(WandbLoggerExtended(WANDB_RUN))
    ###########Adding Callbacks END#############

    net_params = DEFAULT_NETPARAMS.copy()

    ### Criterion parameters ###
    bwloss = SplitLosses(losses_list=[BarlowTwins(lamb=1.0, alpha=1.0, use_positive=False, reducer=MeanReducer())],
                         split_y_list=[0]  # 0: domain. 1: class
                         )
    coralloss = CoralLoss()
    losses_list = [
        ('bw_loss', MacroAvgLoss(bwloss, Y_index=1)),  # 1: class
        ('clf_loss', LossPerDomain(TripletMarginLoss(margin=0.5, reducer=MeanReducer(), triplets_per_anchor='all'))),
        ('coralloss1', MacroAvgLoss(coralloss, Y_index=1))
    ]
    net_params.update({
        'criterion': SplitLosses,
        'criterion__losses_list': losses_list,
        'criterion__split_y_list': [(0, 1), (0, 1), (0, 1)],  # 0: domain. 1: class.
        'criterion__split_x_list': [1, 0, 1],  # 2: convnet output (size=6080). 1: backbone output. 0: heads output
        # We can monitor coralloss (or any other loss) without optimizing it by setting lamb=0.0.
        'criterion__lambs': [0.0, 1.0, lamb]
    })
    # net_params.update({
    #     'criterion': TripletMarginLoss,
    #     'criterion__margin': 0.5, 'criterion__triplets_per_anchor': 'all', 'criterion__reducer': MeanReducer()
    # })
    ############################
    net_params['callbacks'] = callbacks
    net_params['train_split'] = DomainValidSplit()
    net_params['iterator_train'] = DomainBalancedDataLoader
    # net_params['train_split'] = ValidSplit(cv=0.1, stratified=True, random_state=RANDOM_STATE)

    optimizer_params = DEFAULT_OPTIM_PARAMS.copy()
    optimizer_params['optimizer__lr'] = lr

    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in net_params.items()})
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in module_params.items()})
    WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in optimizer_params.items()})
    WANDB_RUN.config.update({'loss function': 'multiple coral loss',
                             'dropout_on_heads': True})

    name = "Tripletloss-Vibnet"
    clf = NeuralNetDomainAdapter(**net_params, **module_params, **optimizer_params, cache_dir=None)

    with open(MODEL_SAVE_DIR/'metainfo.txt', append_or_write) as f:
        f.write("%s:%s\n%s\n\n%s\n\n%s" % (saved_model_fname,
                                           '-'.join(dataset_names),
                                           str(net_params),
                                           str(module_params),
                                           str(optimizer_params)
                                           ))

    return (name, clf)


def createFineTNet(dataset, encode_size, finetunning_on=True) -> Tuple[str, NeuralNetBase]:
    global WANDB_RUN
    # num_classes = len(np.unique(dataset.metainfo['label']))

    module_params = {
        # 'n_classes': num_classes,
        'load_bb_weights_fpath': VIBNET_BB_FPATH if finetunning_on else None,
        'encode_size': encode_size, 'input_size': 6100,  # dataset[0]['signal'].shape[-1],
        'single_output': False
    }
    module_params = {"module__"+key: v for key, v in module_params.items()}
    module_params['module'] = RPDBCS2020NetLoadable

    callbacks = [EstimatorEpochScoring(QuadraticDiscriminantAnalysis(), 'f1_macro',
                                       name='QDA_f1_macro', lower_is_better=False, use_caching=False)]
    callbacks += commonCallbacks()
    callbacks.append(WandbLoggerExtended(WANDB_RUN))

    net_params = DEFAULT_NETPARAMS.copy()
    net_params['callbacks'] = callbacks
    net_params.update({
        'criterion': SplitLosses,
        'criterion__losses_list': [TripletMarginLoss(margin=0.5, triplets_per_anchor='all', reducer=MeanReducer())],
        'criterion__split_x_list': [0]
    })
    if(WANDB_RUN is not None):
        WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in net_params.items()})
        WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in module_params.items()})
        WANDB_RUN.config.update({"net_param__%s" % k: v for k, v in DEFAULT_OPTIM_PARAMS.items()})
    name = "TripletNet-finetunned"
    # clf = NeuralNetClassifier(**net_params, **module_params, **DEFAULT_OPTIM_PARAMS,
    #                           cache_dir=None)
    clf = NeuralNetTransformer(**net_params, **module_params, **DEFAULT_OPTIM_PARAMS,
                               cache_dir=None)
    clf = Pipeline([('metricnet', clf),
                    ('clf', RandomForestClassifier(n_estimators=300, min_impurity_decrease=1e-6, n_jobs=6))])
    return (name, clf)


def _transform_output(data: dict):
    """
    Transforms the dict output into a tuple output format that pytorch normally accepts.
    """
    label = data['label']
    del data['label']
    data['X'] = data['signal']
    del data['signal']
    return data, label


def run_single_experiment(datasets_concat: ConcatenateDataset, datasets_names, Dtarget: Tuple, dname: str,
                          d_percentage, finetunning_on=True, train_vibnet=True, lamb=1.0) -> Sequence[Tuple]:
    global WANDB_RUN

    configs = {'target_dataset': dname, 'd_percentage': d_percentage, 'finetunning_on': finetunning_on}

    (Dtarget_train, Yc_train, groups_ids_train), (Dtarget_test, Yc_test) = Dtarget

    if(finetunning_on and train_vibnet):
        source_names = "+".join(datasets_names)
        Yc = datasets_concat.getLabels()
        # Yd = datasets_concat.getDomains()
        groups_ids = datasets_concat.group_ids
        WANDB_RUN = wandb.init(project="coral_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                               job_type='vibnet-train')
        WANDB_RUN.config.update({'source_datasets': source_names})
        WANDB_RUN.config.update(configs)

        with WANDB_RUN:
            torch.manual_seed(RANDOM_STATE)
            torch.cuda.manual_seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)
            vibnet_name, vibnet = createVibnet(datasets_concat, datasets_names, lr=1e-3, encode_size=32, margin=0.5, lamb=lamb,
                                               add_data=None, backbone=RPDBCS2020Net)

            # final_dataset = AppendDataset(datasets_concat, {'sample_weight': sample_weights})
            vibnet.fit(datasets_concat, Yc, groups=groups_ids)
        torch.save(vibnet.module_.backbone, VIBNET_BB_FPATH)
        vibnet = None
    elif(finetunning_on == False):
        source_names = "none"
    else:
        source_names = "+".join(datasets_names)

    Results = []

    ### Zero epochs testing ###
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    WANDB_RUN = None
    netname, net = createFineTNet(Dtarget_train, finetunning_on=finetunning_on, encode_size=32)
    net['metricnet'].initialize()
    net['clf'].fit(net['metricnet'].transform(Dtarget_train), Yc_train)

    Ycp_test = net.predict(Dtarget_test)
    score = f1_score(Yc_test, Ycp_test, average='macro')
    Results.append((netname, source_names, dname, "test", 'fmacro', score,
                   d_percentage, finetunning_on, False, lamb))

    Ycp_train = net.predict(Dtarget_train)
    score = f1_score(Yc_train, Ycp_train, average='macro')
    Results.append((netname, source_names, dname, "train", 'fmacro', score,
                   d_percentage, finetunning_on, False, lamb))
    ###########################

    WANDB_RUN = wandb.init(project="coral_vibnet", entity="lucashsmello", group=CURRENT_TIME,
                           job_type='finetunning')
    WANDB_RUN.config.update({'source_datasets': source_names})
    WANDB_RUN.config.update(configs)

    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    netname, net = createFineTNet(Dtarget_train, finetunning_on=finetunning_on, encode_size=32)

    with WANDB_RUN:
        net.fit(Dtarget_train, Yc_train, metricnet__groups=groups_ids_train)

    Ycp_test = net.predict(Dtarget_test)
    score = f1_score(Yc_test, Ycp_test, average='macro')
    Results.append((netname, source_names, dname, "test", 'fmacro', score,
                   d_percentage, finetunning_on, True, lamb))

    Ycp_train = net.predict(Dtarget_train)
    score = f1_score(Yc_train, Ycp_train, average='macro')
    Results.append((netname, source_names, dname, "train", 'fmacro', score,
                   d_percentage, finetunning_on, True, lamb))

    return Results


def run_experiment(data_root_dir, cache_dir="/tmp/sigdata_cache"):
    # Generates 4 folds, but only tests in the first one.
    sampler = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)

    # Transform datasets
    datasets_transformed = loadTransformDatasets(data_root_dir, DATASETS, cache_dir)
    datasets_names = [name for name, _ in datasets_transformed]

    # In fine-tunning, d_percentage parameter removes a percentage of the training dataset. The test set remains intact.
    d_percentage_list = [1.0, 0.5, 0.25]  # All parameters of this list will be tested

    lambds = [1.0, 0.0, 10.0]  # All lambs to be tested. Multiples the domain adaptation loss by lamb.

    # If True, the vibnet will be trained on source datasets and its weights
    #  will be used to fine tune a neural network on the target dataset.
    # If False, vibnet is not trained and the network trained in the target dataset will be initilized normally (random weights).
    finetunning_on_options = [True, False]

    Results = []
    with tqdm(total=len(datasets_transformed) * len(d_percentage_list) * len(finetunning_on_options) * len(lambds)) as pbar:
        for finetunning_on in finetunning_on_options:
            for i, (dname, Dtarget) in enumerate(datasets_transformed):
                if(dname != 'rpdbcs'):
                    pbar.update()
                    continue

                for lamb in lambds:
                    train_vibnet = True
                    for d_percentage in d_percentage_list:
                        if(d_percentage*len(Dtarget) < 150):
                            pbar.update()
                            continue
                        ### Preparing dataset ###
                        Yc = Dtarget.metainfo['label']
                        group_ids = Dtarget.metainfo['index']

                        # Transforming the dict output into a tuple output that pytorch normally accepts.
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
                        ##########################

                        pbar.set_description(dname)

                        datasets_names_sources = datasets_names[:i] + datasets_names[i+1:]
                        # Concatanates all source datasets.
                        datasets_concat = ConcatenateDataset([d for j, (_, d) in enumerate(datasets_transformed) if j != i],
                                                             None)

                        R = run_single_experiment(datasets_concat, datasets_names_sources, ((Dtarget_train, Yc_train, group_ids[train_idxs]), (Dtarget_test, Yc_test)), dname,
                                                  d_percentage=d_percentage, finetunning_on=finetunning_on, train_vibnet=train_vibnet, lamb=lamb)
                        Results += R
                        pbar.update()
                        train_vibnet = False
                    if(finetunning_on == False):
                        break

    columns = ['classifier_name', 'source_dataset', 'target_dataset', 'test_sample', 'metric_name',
               'value', 'd_percentage', 'finetunning_on', 'is_trained', 'lamb']
    Results = pd.DataFrame(Results, columns=columns)
    return Results


"""
TODO:
- Testar sem normalização do RPDBCS
- Testar normalizar por sample=4096
- ver questão do sample_rate ser muito alto (94000) para um numero de pontos baixo (6100)
"""


"""
04/02/2022
Há varias formas de se usar o Coral loss:
- Em cada cabeça da rede neural ou só no backbone
- Calcular coral loss para todos os pares de dominios de source e target

16/03/2022
- Testar dropout na ultima camada
- Testar coral loss nas primeiras camadas (com peso menor)
- Separar uma parte da arquitetura para ser especializada em um dominio, e outra para ser genérica. 
- https://github.com/anh-ntv/STEM_iccv21
- Seria bom balancear os batches
"""
