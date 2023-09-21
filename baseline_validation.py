import os
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
from sklearn.metrics import classification_report, balanced_accuracy_score
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import LeaveOneGroupOut
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import Sequential, SplitSampleRate, NormalizeSampleRate

import wandb
from models.model import Model
from utils.report import ReportDict
from utils.MemeDataset import MemeDataset

BATCH_SIZE = 16
LEARNING_RATE = 0.1
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
    global LEARNING_RATE, NUM_EPOCHS
    wandb.login(key="e510b088c4a273c87de176015dc0b9f0bc30934e")
    wandb.init(
        # Set the project where this run will be logged
        project="vibdata-deeplearning-timedomain",
        # Track hyperparameters and run metadata
        config={"learning_rate": LEARNING_RATE, "epochs": NUM_EPOCHS, "arquitecture": "alexnet-1d"},
        # Set the name of the experiment
        name=run_name,
    )


def make_deterministic():
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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

    # ---
    # Cross validation
    # ---
    for fold_id, (train_ids, test_ids) in enumerate(logo.split(dataset, groups=groups)):
        print("".center(30, "="))
        print(f"Fold {fold_id}".center(30, "="))
        print("".center(30, "="))

        # Create test and train dataloader
        test_sampler = SubsetRandomSampler(test_ids)
        test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

        train_sampler = SubsetRandomSampler(train_ids)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

        # Instantiate the model
        net = Model("Resnet18", out_channel=4).new()
        net.apply(Model.reset_weights)

        # TODO: Implement an initializer weights method
        # Create the optimizer with the learning rate candidate
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        # ---
        # Train model
        # ---
        print("Training".center(30, "="))
        for epoch in range(NUM_EPOCHS):
            train_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.float().cuda()
                labels = labels.cuda()

                # inputs, labels = data['signal'], data['metainfo']['label'].values.reshape(-1, 1)
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
                # train_acc += torch.sum(torch.eq(labels, outputs)) / len(labels)

            epoch_loss = train_loss / (i + 1)
            wandb.log(
                {
                    f"{fold_id}_loss": epoch_loss,
                }
            )
            print(f"[{epoch + 1 : 3d}] loss: {epoch_loss:.3f}")

        print("Testing".center(30, "="))
        _, test_predicitons = evaluate(net, test_loader)
        test_size = len(test_ids)
        report_predicitons.update(
            test_predicitons,
            fold=[
                fold_id,
            ]
            * test_size,
            dataset=[
                "test",
            ]
            * test_size,
        )

        # Save the model
        model_path = output_dir / f"model-fold-{fold_id}.pth"
        torch.save(net.state_dict(), model_path)

    # Save the predicitons
    fpath = output_dir / "predictions.csv"
    os.makedirs(output_dir, exist_ok=True)
    y_df = pd.DataFrame(report_predicitons)
    y_df.to_csv(fpath, index=False)

    pd.DataFrame(dict(label_name=LABELS_NAME, id=LABELS)).to_csv(output_dir / "labels_name.csv", index=False)

    print("The end")

    return net


def evaluate(model: nn.Module, evalloader: DataLoader) -> Tuple[float, dict]:
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    macro_output = []
    macro_label = []

    for data, labels in evalloader:
        # Normalize labels
        labels -= torch.min(labels)
        # Move to gpu
        if torch.cuda.is_available():
            data = data.float().cuda()
            labels = labels.cuda()

        output = model(data)
        loss = criterion(output, labels)
        test_loss += loss.item() * data.size(0)

        # Convert from one-hot to indexing
        output = torch.argmax(output, dim=1)
        # Return to the normalized labels
        output += LABELS.min()

        macro_output.append(output.cpu().numpy())
        macro_label.append(labels.cpu().numpy())

    test_loss = test_loss / len(evalloader)

    # Creates table from classification report and log it
    data = {"predicted": np.concatenate(macro_output), "real_label": np.concatenate(macro_label)}
    class_report = classification_report(
        data["real_label"],
        data["predicted"],
        target_names=LABELS_NAME,
        zero_division=0,
    )
    print(class_report)
    print("\nBalanced Accuracy: {:3f}%".format(balanced_accuracy_score(data["real_label"], data["predicted"])))
    return test_loss, data


def load_dataset() -> DeepDataset:
    MAX_SAMPLE_RATE = 97656

    transforms = Sequential([SplitSampleRate(), NormalizeSampleRate(MAX_SAMPLE_RATE)])

    dataset_name = "CWRU"
    raw_dir = Path("../datasets")
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
