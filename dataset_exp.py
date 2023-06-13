from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRate, Sequential
import vibdata.raw as raw
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from utils.MemeDataset import MemeDataset
import sklearn
from skorch import NeuralNetClassifier
import models
import numpy as np
import torch.optim as optim
from torch import nn
import wandb
from argparse import ArgumentParser, Namespace
from typing import Tuple
import pandas as pd
import os

BATCH_SIZE = 32
LEARNING_RATE = 0.01    
NUM_EPOCHS = 100
OUTPUT_DIR = Path('./output')

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--epochs', help='Number of epochs', type=int)
    parser.add_argument('--run', help='Name of the experiment', required=True)
    parser.add_argument('--lr', help='Learning rate', type=float)
    parser.add_argument('--pretrained', help='Pretrained', action='store_true')
    parser.add_argument('--model', help='Model used', default='alexnet')
    args = parser.parse_args()
    return args

def def_hyperparams(args : Namespace) -> None:
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
    
    train_loader, test_loader = load_train_test(MemeDataset(dataset))
    
    # Create net
    from models import Resnet1d, Alexnet1d
    if args.model == 'alexnet':
        net = Alexnet1d.alexnet(pretrained=args.pretrained, out_channel=4)    
    else:
        net = Resnet1d.resnet50(pretrained=args.pretrained, out_channel=4)    
    
    model = train(net, train_loader, test_loader)
    # Test and save file
    testing(model, test_loader, args.run)
    wandb.finish()

def configure_wandb(run_name : str) -> None:
    wandb.login(key="e510b088c4a273c87de176015dc0b9f0bc30934e")
    wandb.init(
        # Set the project where this run will be logged
        project="vibdata-deeplearning-timedomain",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "arquitecture": "alexnet-1d"
        },
        # Set the name of the experiment
        name=run_name
    )
    
def make_deterministic():
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
def train(net : nn.Module, train_loader : DataLoader, test_loader) -> nn.Module:
    print("Beginning training")


    if torch.cuda.is_available():
        net = net.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    print("Begin")
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            
            inputs = inputs.float().cuda()
            labels = labels.cuda()

            #inputs, labels = data['signal'], data['metainfo']['label'].values.reshape(-1, 1)
            
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Convert from one-hot to indexing 
            outputs = torch.argmax(outputs, dim=1)
            # Return to the normalized labels
            outputs += LABELS.min()
            #print(f"LABELS SIZE: {labels.shape} OUTPUTS SIZE: {outputs.shape}")

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(torch.eq(labels,outputs)) / len(labels)
    
        
        test_loss, test_acc = testing(net, test_loader)
        train_acc = (train_acc/i) * 100
        epoch_loss = train_loss/i
        wandb.log({"loss":epoch_loss, "accuracy":train_acc, "testing_accuracy": test_acc})
        print(f'[{epoch + 1 : 3d}] loss: {epoch_loss:.3f} | acc: {train_acc:.3f} | test_acc: {test_acc:.3f}')
        
                    
    print("Finished Training")
    return net

def testing(model : nn.Module, testloader : DataLoader, run_name: str = None) -> Tuple[float, float]:
    #print("Testing")
    test_loss = 0.0
    correct, total = 0,0
    
    criterion = nn.CrossEntropyLoss()
    
    macro_output = []
    macro_label = []
    
    for data, labels in testloader:
        # Normalize labels
        labels -= torch.min(labels)
        # Move to gpu
        if torch.cuda.is_available():
            data = data.float().cuda()
            labels = labels.cuda()
        
        output = model(data)
        loss = criterion(output,labels)
        test_loss += loss.item() * data.size(0)
        
        # Convert from one-hot to indexing 
        output = torch.argmax(output, dim=1)
        # Return to the normalized labels
        output += LABELS.min()
        for o,l in zip(output,labels):
            if o == l:
                correct += 1
            total += 1
        
        macro_output.append(output.cpu().numpy())
        macro_label.append(labels.cpu().numpy())
        
    test_loss = test_loss/len(testloader)
    test_acc = correct/total*100

    # Creates table from classification report and log it
    data = {'classe_predita' : np.concatenate(macro_output), 'classe_real' : np.concatenate(macro_label)}
    class_report = sklearn.metrics.classification_report(
        data['classe_real'], data['classe_predita'],
        target_names=LABELS_NAME, output_dict=True,
        zero_division=0,
    )
    # Fix the accuracy column to be compatible to other lines
    class_report['accuracy'] = {'precision':0, 'recall':0, 'f1-score':class_report['accuracy'], 'support': class_report['macro avg']['support'] }
    df = pd.DataFrame(class_report)
    # Column type is numpy._str, therefore, convert columns name to native string
    df.columns = df.columns.astype(str)
    table_report = wandb.Table(dataframe=df, allow_mixed_types=True)

    wandb.log({'classification_report': table_report})
    wandb.log({'class_report_str': \
        sklearn.metrics.classification_report(
        data['classe_real'], data['classe_predita'],
        target_names=LABELS_NAME, zero_division=0)
    })

    if run_name:
        # Save the predicitons 
        fpath = OUTPUT_DIR / (run_name + '.csv')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        y_df = pd.DataFrame(data)
        y_df['labels_name'] = y['classe_real'].apply(lambda x : LABELS_NAME[x]) 
        y_df.to_csv(fpath, index=False)
        
        # Log the classification_report  
        wandb.sklearn.plot_class_proportions(y_train, y_test, ['dog', 'cat', 'owl'])
    return test_loss, test_acc

    
def load_train_test(dataset : DeepDataset):
    test_size = 0.3
    print("Splitting train and test")
    trainset, testset = random_split(dataset, [1-test_size, test_size]) 
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    return trainloader, testloader

    
def load_dataset() -> DeepDataset:
    MAX_SAMPLE_RATE = 97656

    transforms = Sequential([ 
        SplitSampleRate(),
        NormalizeSampleRate(MAX_SAMPLE_RATE)
    ])
    
    dataset_name = "CWRU"
    raw_dir = Path('../datasets')
    deep_dir = Path('data/deep_datasets')
    
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
