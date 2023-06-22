import os
from typing import Tuple
from pathlib import Path
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import pandas as pd
import torch.optim as optim
import vibdata.raw as raw
from torch import nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from sklearn.model_selection import KFold
from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import Sequential, SplitSampleRate, NormalizeSampleRate

import wandb
from models.model import Model
from utils.MemeDataset import MemeDataset

BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
OUTPUT_ROOT = Path("./output")


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

    # Create net
    model = Model(model_name=args.model, pretrained=args.pretrained, out_channel=4)
    experiment(model, MemeDataset(dataset), args.run)
    wandb.finish()


def configure_wandb(run_name: str) -> None:
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


def experiment(model: Model, dataset: torch.utils.data.Dataset, run_name: str) -> nn.Module:
    # Retrive hyperparameters
    global BATCH_SIZE, LEARNING_RATE
    print("Beginning training")
    # Create the directory where logs will be stored
    output_dir = OUTPUT_ROOT / run_name
    os.makedirs(output_dir, exist_ok=True)

    param_grid = {"learning_rate": [1e-1, 1e-3, 3e-4]}
    report_predicitons = {"real_label": [], "predicted": [], "fold": []}

    criterion = nn.CrossEntropyLoss()

    # TODO: Change shuffle to True
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (remaining_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"Training fold {fold}")

        # Create test dataloader
        test_sampler = SubsetRandomSampler(test_ids)
        test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=BATCH_SIZE)

        val_generator = torch.Generator().manual_seed(42)  # Generator used to fix the random sample
        # Create subset
        train_ids, val_ids = random_split(remaining_ids, [0.7, 0.3], generator=val_generator)
        train_sampler = SubsetRandomSampler(train_ids)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

        results = {
            "learning_rate": [],
            "accuracy": [],
        }  # Variable that will store the results for each param
        for lr in param_grid["learning_rate"]:
            net = model.new()
            # Update the optimizer with the learning rate candidate
            LEARNING_RATE = lr
            optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
            print(f"Fold: {fold}\n" f"Learning rate: {lr}\n", "---")

            # Train
            for epoch in range(NUM_EPOCHS):
                train_loss = 0.0
                train_acc = 0.0
                for i, (inputs, labels) in enumerate(train_loader, 0):
                    inputs = inputs.float().cuda()
                    labels = labels.cuda()

                    # inputs, labels = data['signal'], data['metainfo']['label'].values.reshape(-1, 1)

                    optimizer.zero_grad()
                    outputs = net(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Convert from one-hot to indexing
                    outputs = torch.argmax(outputs, dim=1)
                    # Return to the normalized labels
                    outputs += LABELS.min()
                    # print(f"LABELS SIZE: {labels.shape} OUTPUTS SIZE: {outputs.shape}")

                    train_loss += loss.item() * inputs.size(0)
                    train_acc += torch.sum(torch.eq(labels, outputs)) / len(labels)

                train_acc = (train_acc / i) * 100
                epoch_loss = train_loss / i
                wandb.log({f"{fold}_{lr}_loss": epoch_loss, f"{fold}_{lr}_accuracy": train_acc})
                print(f"[{epoch + 1 : 3d}] loss: {epoch_loss:.3f} | acc: {train_acc:.3f}")
            # Validate the model
            val_sampler = SubsetRandomSampler(val_ids)
            val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=BATCH_SIZE)

            print("Validation".center(30, "="))
            val_loss, val_acc, _ = evaluate(net, val_loader)
            results["learning_rate"].append(lr)
            results["accuracy"].append(val_acc)

        # Evaluate with the testset
        net = model.new()
        # Define the trainset, now with the validatioset
        train_sampler = SubsetRandomSampler(remaining_ids)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
        # Get the best param
        best_param_index = results["accuracy"].index(max(results["accuracy"]))
        lr = results["learning_rate"][best_param_index]
        LEARNING_RATE = lr
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
        # Train with the model with it
        for epoch in range(NUM_EPOCHS):
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.float().cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print("Testing".center(30, "="))
        print(f"Best Param: {lr}")
        _, test_acc, test_predicitons = evaluate(net, test_loader)

        # Update the preds variable
        for key in test_predicitons:
            report_predicitons[key].extend(test_predicitons[key])
        report_predicitons["fold"].extend(
            [
                fold,
            ]
            * len(test_predicitons["real_label"])
        )

    # Save the predicitons
    fpath = output_dir / "predictions.csv"
    os.makedirs(output_dir, exist_ok=True)
    y_df = pd.DataFrame(report_predicitons)
    y_df.to_csv(fpath, index=False)

    pd.DataFrame(dict(label_name=LABELS_NAME, id=LABELS)).to_csv(output_dir / "labels_name.csv", index=False)

    print("The end")

    return net


def evaluate(model: nn.Module, evalloader: DataLoader) -> Tuple[float, float, dict]:
    # print("Testing")
    test_loss = 0.0
    correct, total = 0, 0

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
        for o, l in zip(output, labels):
            if o == l:
                correct += 1
            total += 1

        macro_output.append(output.cpu().numpy())
        macro_label.append(labels.cpu().numpy())

    test_loss = test_loss / len(evalloader)
    test_acc = correct / total * 100

    # Creates table from classification report and log it
    data = {"predicted": np.concatenate(macro_output), "real_label": np.concatenate(macro_label)}
    class_report = classification_report(
        data["real_label"],
        data["predicted"],
        target_names=LABELS_NAME,
        zero_division=0,
    )
    print(class_report)
    return (test_loss, test_acc, data)


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
