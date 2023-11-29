from argparse import ArgumentParser, Namespace
from typing import List

import essentia.standard
import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from tqdm import tqdm
from vibdata.deep.DeepDataset import DeepDataset

from lib.config import Config


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # parser.add_argument("--dataset", help="The dataset name", required=True)
    parser.add_argument("--cfg", help="Config file", required=True)
    args = parser.parse_args()
    return args


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


def classifier_biased(cfg: Config, inputs: List[int], labels: List[int]) -> List[int]:
    model = RandomForestClassifier()

    parameters = cfg['params_grid']

    cv_outer = StratifiedKFold(n_splits=5, shuffle=True)
    cv_inner = StratifiedKFold(n_splits=3)

    clf = GridSearchCV(estimator=model, param_grid=parameters, cv=cv_inner, n_jobs=-1)
    y_pred = cross_val_predict(clf, inputs, labels, cv=cv_outer)

    return y_pred


def results(dataset: DeepDataset, y_true: List[int], y_pred: List[int]) -> None:
    print(f'{classification_report(y_true, y_pred, target_names=dataset.get_labels_name())}')
    print(f'Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.2f}')


def main():
    args = parse_args()
    cfg_path = args.cfg
    cfg = Config(cfg_path, args=args)

    dataset = cfg.get_dataset()
    X, y = extract_features(dataset)
    y_pred = classifier_biased(cfg, X, y)

    results(dataset, y, y_pred)


if __name__ == "__main__":
    main()
