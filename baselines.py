import os
from argparse import ArgumentParser, Namespace
from typing import List

import essentia.standard
import numpy as np
import wandb
from dotenv import load_dotenv
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, LeaveOneGroupOut
from tqdm import tqdm
from vibdata.deep.DeepDataset import DeepDataset

import lib.data.group_dataset as groups_module
from lib.config import Config


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--cfg", help="Config file", required=True)

    classifier = parser.add_mutually_exclusive_group(required=True)
    classifier.add_argument('--biased', help='Use biased classifier', action='store_true')
    classifier.add_argument('--unbiased', help='Use unbiased classifier', action='store_true')

    args = parser.parse_args()
    return args


def folds_by_groups(groups: List[int]) -> int:
    return len(set([fold for fold in groups]))


def extract_features(dataset: DeepDataset) -> (List[int], List[int]):
    X = np.empty([len(dataset), 9])
    y = np.empty([len(dataset)], dtype=np.int8)

    for i, sample in tqdm(enumerate(dataset), desc="Extracting features", unit="samples"):

        signal = sample['signal'][0]

        sampleRate32 = sample['metainfo']['sample_rate'].astype('float32')
        signal32 = signal.astype('float32')

        envelope = essentia.standard.Envelope(sampleRate=sampleRate32, applyRectification=False)
        signal_envelope = envelope(signal32)

        # Kurtosis
        X[i][0] = kurtosis(signal)

        # Root Mean Square (RMS)
        X[i][1] = np.sqrt(sum(np.square(signal)) / len(signal))

        # Standard Deviation
        X[i][2] = np.std(signal)

        # Mean
        X[i][3] = np.mean(signal)

        # Log Attack Time
        logAttackTime = essentia.standard.LogAttackTime(sampleRate=sampleRate32)
        X[i][4] = logAttackTime(signal_envelope)[0]

        # Temporal Decrease
        decrease = essentia.standard.Decrease(range=((len(signal32) - 1) / sampleRate32))
        X[i][5] = decrease(signal32)

        # Temporal Centroid
        centroid = essentia.standard.Centroid(range=((len(signal32) - 1) / sampleRate32))
        X[i][6] = centroid(signal_envelope)

        # Effective Duration
        effective = essentia.standard.EffectiveDuration(sampleRate=sampleRate32)
        X[i][7] = effective(signal_envelope)

        # Zero Crossing Rate
        zeroCrossingRate = essentia.standard.ZeroCrossingRate()
        X[i][8] = zeroCrossingRate(signal32)

        # Labels (Targets)
        y[i] = sample['metainfo']['label']

    return X, y


def classifier_biased(cfg: Config, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
    seed = cfg['seed']
    parameters = cfg['params_grid']

    num_folds = folds_by_groups(groups)

    model = RandomForestClassifier(random_state=seed)

    cv_outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    clf = GridSearchCV(estimator=model, param_grid=parameters, scoring="balanced_accuracy",
                       cv=cv_inner, n_jobs=-1)

    y_pred = cross_val_predict(clf, inputs, labels, cv=cv_outer)

    return y_pred


def classifier_unbiased(cfg: Config, inputs: List[int], labels: List[int], groups: List[int]) -> List[int]:
    seed = cfg['seed']
    parameters = cfg['params_grid']

    model = RandomForestClassifier(random_state=seed)

    cv_inner = LeaveOneGroupOut()
    clf = GridSearchCV(estimator=model, param_grid=parameters, scoring="balanced_accuracy",
                       cv=cv_inner, n_jobs=-1)

    cv_outer = LeaveOneGroupOut()
    y_pred = cross_val_predict(clf, inputs, labels, groups=groups, cv=cv_outer,
                               fit_params={"groups": groups})

    return y_pred


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int]) -> None:
    labels = dataset.get_labels_name()
    labels = [label if label is not np.nan else "NaN" for label in labels]

    print(f'{classification_report(y_true, y_pred, target_names=labels)}')
    print(f'Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.2f}')


def configure_wandb(run_name: str, cfg: Config, cfg_path: str, groups: List[int], args) -> None:
    wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        # Set the project where this run will be logged
        project=os.environ["WANDB_PROJECT"],
        # Track essentials hyperparameters and run metadata
        config={
            "model": cfg["model"]["name"],
            "folds": folds_by_groups(groups),
            "params_grid": cfg["params_grid"],
            "biased": args.biased,
            "unbiased": args.unbiased
        },
        # Set the name of the experiment
        name=run_name,
    )
    # Add configuration file into the wandb log
    wandb.save(cfg_path, policy="now")


def main():
    load_dotenv()
    args = parse_args()
    cfg_path = args.cfg
    cfg = Config(cfg_path, args=args)

    dataset_name = cfg['dataset']['name']
    dataset = cfg.get_dataset()

    group_obj = getattr(groups_module, "Group" + dataset_name)(dataset=dataset, config=cfg)
    groups = group_obj.groups()

    configure_wandb(dataset_name, cfg, cfg_path, groups, args)

    X, y = extract_features(dataset)

    if args.biased:
        y_pred = classifier_biased(cfg, X, y, groups)
    elif args.unbiased:
        y_pred = classifier_unbiased(cfg, X, y, groups)
    else:
        raise Exception("Undefined classifier")

    results(dataset, y, y_pred)


if __name__ == "__main__":
    main()
