import os
import sys
import random
from typing import Dict, List, Tuple
from pathlib import Path
from argparse import Namespace, ArgumentParser

import yaml
import numpy as np
import torch
import pandas as pd
import torch.optim as optim
import vibdata.raw as raw
from torch import nn
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import classification_report, balanced_accuracy_score
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import Sequential, StandardScaler, SplitSampleRate, NormalizeSampleRate

import wandb
from models.model import Model
from utils.report import ReportDict
from utils.dataloaders import BalancedBatchSampler, seed_worker
from utils.MemeDataset import MemeDataset

BATCH_SIZE = 32
LEARNING_RATE = 1e-1
NUM_EPOCHS = 120
OUTPUT_ROOT = Path("./output")

# This program implements an experiment designed in the paper `An experimental
# methodology to evaluate machine learning methods for fault diagnosis based on
# vibration signals`.
# The experiment is executed by a Repeated Nested Cross Validation, and the
# scheme is described below:
# The dataset D is divided into 4 folds, where each fold consist of samples
# recorded with a specific `fault_size`. Then in the outer loop of the
# cross-validation, the test fold is leaved out where the other 3 folds are used
# in the inner loop. Inside the inner loop, the fold division are kept and then
# one fold are choosed for the validation set, and the others two to training.
# Then, for each set of hyperparameters, the model is trained and valideted in
# the set division, and the best set of hyperparaeters are used.
# This process is reapeated until all folds in the inner loop
# are used as a validation set.


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--epochs", help="Number of epochs", type=int)
    parser.add_argument("--run", help="Name of the experiment", required=True)
    parser.add_argument("--lr", help="Learning rate", type=float)
    parser.add_argument("--pretrained", help="Pretrained", action="store_true")
    parser.add_argument("--model", help="Model used", default="alexnet")
    args = parser.parse_args()
    return args


def def_hyperparams(args: Namespace) -> None:
    global LEARNING_RATE, NUM_EPOCHS
    if args.lr:
        LEARNING_RATE = args.lr
    if args.epochs:
        NUM_EPOCHS = args.epochs


def main():
    args = parse_args()
    def_hyperparams(args)

    configure_wandb(args.run)
    # Fix seed in order to make deterministic
    make_deterministic()
    dataset = load_dataset()

    # Load the models
    with open("cfgs/models.yaml", "r") as f:
        models_cfgs = yaml.safe_load(f)

    # Create net
    models = {m["name"]: Model(m["name"], **m["parameters"]) for m in models_cfgs["models"]}
    # model = Model(model_name=args.model, pretrained=args.pretrained, out_channel=4)
    experiment(models, dataset, args.run)
    wandb.finish()


def configure_wandb(run_name: str) -> None:
    # Retrieve global variables
    global LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE
    wandb.login(key="e510b088c4a273c87de176015dc0b9f0bc30934e")
    wandb.init(
        # Set the project where this run will be logged
        project="vibdata-deeplearning-timedomain",
        # Track hyperparameters and run metadata
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "arquitecture": "resnet-18",
        },
        # Set the name of the experiment
        name=run_name,
    )


def make_deterministic():
    # Fix seed
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    np.random.seed(SEED)
    random.seed(SEED)

    # CUDA convolution determinism
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Set cubLAS enviroment variable to guarantee a deterministc behaviour in multiple streams
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def experiment(param_grid: Dict[str, Model], original_dataset: torch.utils.data.Dataset, run_name: str) -> nn.Module:
    # Retrive hyperparameters
    global BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, ROUNDS
    print("Beginning training")
    # Create the directory where logs will be stored
    output_dir = OUTPUT_ROOT / run_name
    os.makedirs(output_dir, exist_ok=True)

    report_predicitons = ReportDict(["real_label", "predicted", "fold", "dataset"])

    criterion = nn.CrossEntropyLoss()

    logo = LeaveOneGroupOut()

    groups = np.array([ret["metainfo"]["load"] for ret in original_dataset])

    dataset = MemeDataset(original_dataset)

    folds = [fold_ids for _, fold_ids in logo.split(dataset, groups=groups)]
    num_folds = len(folds)
    # ---
    # Cross validation
    # ---
    for test_fold, test_ids in enumerate(folds):
        print("".center(30, "="))
        print(f"Fold {test_fold}".center(30, "="))
        print("".center(30, "="))

        get_labels = lambda dataset: [y for (x, y) in dataset]  # noqa E731

        #
        # Divide the dataset into train, validation and test sets
        #
        # Generator in order to fix the ramdoness
        SEED = 42
        g = torch.Generator()
        g.manual_seed(SEED)

        # Testing set
        # ---
        testset = Subset(dataset, test_ids)
        test_sampler = BalancedBatchSampler(
            labels=get_labels(testset), n_classes=len(LABELS), n_samples=int(BATCH_SIZE / len(LABELS))
        )
        test_loader = DataLoader(testset, batch_sampler=test_sampler, worker_init_fn=seed_worker, generator=g)

        # Validation
        # ---
        val_fold = (test_fold + 1) % num_folds
        val_ids = folds[val_fold]
        # torch.nn.functional.one_hot(labels).sum(dim=0)
        valset = Subset(dataset, val_ids)
        val_sampler = BalancedBatchSampler(
            labels=get_labels(valset), n_classes=len(LABELS), n_samples=int((BATCH_SIZE / len(LABELS)))
        )
        val_loader = DataLoader(valset, batch_sampler=val_sampler, worker_init_fn=seed_worker, generator=g)
        # val_loader = DataLoader(valset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

        # Training set
        # ---
        train_folds = set(range(num_folds)).difference(set([test_fold, val_fold]))
        train_ids = np.concatenate([folds[i] for i in train_folds])
        trainset = Subset(dataset, train_ids)
        train_sampler = BalancedBatchSampler(
            labels=get_labels(trainset), n_classes=len(LABELS), n_samples=int(BATCH_SIZE / len(LABELS))
        )
        train_loader = DataLoader(trainset, batch_sampler=train_sampler, worker_init_fn=seed_worker, generator=g)
        # train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

        # Instantiate the model
        net = Model("Resnet18", out_channel=4).new()
        net.apply(Model.reset_weights)

        # TODO: Implement an initializer weights method
        # Create the optimizer with the learning rate candidate
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

        # Create the net to be saved
        best_val_loss = sys.maxsize
        model_path = output_dir / f"model-fold-{test_fold}.pth"
        # ---
        # Train model
        # ---
        print("Training".center(30, "="))
        for epoch in range(NUM_EPOCHS):
            # Train the net
            train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.float().cuda()
                labels = labels.cuda()

                # Zero the graidients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = net(inputs)

                # breakpoint()  # Check labels and output
                # Compute loss
                loss = criterion(outputs, labels)

                # Do the backpropagation
                loss.backward()

                # Update the weights
                optimizer.step()

                # Convert from one-hot to indexing
                outputs = torch.argmax(outputs, dim=1)
                # Return to the normalized labels
                outputs += LABELS.min()
                # breakpoint()  # Check labels and output

                train_loss += loss  # * inputs.size(0)

            # Validate
            train_loss = train_loss / (i + 1)
            val_loss, _ = evaluate(net, val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), model_path)

            wandb.log(
                {
                    f"{test_fold}_train_loss": train_loss,
                    f"{test_fold}_val_loss": val_loss,
                }
            )
            print(f"[{epoch + 1 : 3d}] train_loss: {train_loss:5.3f} | val_loss: {val_loss:5.3f}")

        # ---
        # Test model
        # ---
        print("Testing".center(30, "="))
        # Retrieve the best model
        net = Model("Resnet18", out_channel=4).new()
        net.load_state_dict(torch.load(model_path))

        # Evaluate with the testset
        _, test_predicitons = evaluate(net, test_loader, report=True)
        test_size = len(test_predicitons["real_label"])
        report_predicitons.update(
            test_predicitons,
            fold=[
                test_fold,
            ]
            * test_size,
            dataset=[
                "test",
            ]
            * test_size,
        )
        wandb.run.summary[f"{test_fold}_balanced_accuracy"] = (
            balanced_accuracy_score(test_predicitons["real_label"], test_predicitons["predicted"]) * 100
        )

    # Save the predicitons
    fpath = output_dir / "predictions.csv"
    os.makedirs(output_dir, exist_ok=True)
    y_df = pd.DataFrame(report_predicitons)
    y_df.to_csv(fpath, index=False)

    pd.DataFrame(dict(label_name=LABELS_NAME, id=LABELS)).to_csv(output_dir / "labels_name.csv", index=False)
    wandb.run.summary["total_balanced_accuracy"] = (
        balanced_accuracy_score(report_predicitons["real_label"], report_predicitons["predicted"]) * 100
    )

    print("The end")

    return net


def evaluate(model: nn.Module, evalloader: DataLoader, report=False) -> Tuple[float, dict]:
    eval_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    macro_output = []
    macro_label = []

    # Disable autograd
    with torch.no_grad():
        for batch_id, (data, labels) in enumerate(evalloader):
            # Normalize labels
            labels -= torch.min(labels)
            # Move to gpu
            if torch.cuda.is_available():
                data = data.float().cuda()
                labels = labels.cuda()

            output = model(data)
            loss = criterion(output, labels)
            eval_loss += loss.item()

            # Convert from one-hot to indexing
            output = torch.argmax(output, dim=1)
            # Return to the normalized labels
            output += LABELS.min()

            macro_output.append(output.cpu().numpy())
            macro_label.append(labels.cpu().numpy())

    eval_loss = eval_loss / (batch_id + 1)

    if report:
        # Creates table from classification report and log it
        data = {"predicted": np.concatenate(macro_output), "real_label": np.concatenate(macro_label)}
        class_report = classification_report(
            data["real_label"],
            data["predicted"],
            target_names=LABELS_NAME,
            zero_division=0,
        )
        print(class_report)
        print(
            "\nBalanced Accuracy: {:.3f}%".format(balanced_accuracy_score(data["real_label"], data["predicted"]) * 100)
        )
        return eval_loss, data
    else:
        return eval_loss, np.NAN


def load_dataset() -> DeepDataset:
    MAX_SAMPLE_RATE = 97656

    transforms = Sequential(
        [SplitSampleRate(), NormalizeSampleRate(MAX_SAMPLE_RATE), StandardScaler(on_field="signal")]
    )

    dataset_name = "CWRU_std_scaler"
    raw_dir = Path("data/raw_datasets")
    deep_dir = Path("data/deep_datasets")

    # Load the raw_dataset
    raw_dataset = raw.CWRU.CWRU.CWRU_raw(raw_dir, download=True)
    global LABELS_NAME, LABELS
    LABELS = np.array(raw_dataset.getLabels())
    LABELS_NAME = np.array(raw_dataset.getLabels(as_str=True))

    # Convert the dataset into a DeepDataset and save it on root_dir/<dataset_name> path
    convertDataset(raw_dataset, transforms, deep_dir / dataset_name)
    # Instantiate the deep dataset and load the transforms to be applied
    dataset = DeepDataset(deep_dir / dataset_name)
    return dataset


if __name__ == "__main__":
    main()
