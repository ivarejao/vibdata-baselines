from vibdata.deep.DeepDataset import DeepDataset, convertDataset
from vibdata.deep.signal.transforms import SplitSampleRate, NormalizeSampleRate, Sequential
import vibdata.raw as raw
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from utils.MemeDataset import MemeDataset
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier
import models
import numpy as np
from sklearn.metrics import classification_report
import torch.optim as optim
from torch import nn
import wandb

LEARNING_RATE = 0.1    
NUM_EPOCHS = 100

def main():

    configure_wandb()
    # Fix seed in order to make deterministic
    make_deterministic()
    dataset = load_dataset()
    # Get the labels from dataset
    CLASSES : np.array = dataset.metainfo['label'].unique()
    train_loader, test_loader = load_train_test(MemeDataset(dataset))
    model = train(train_loader, CLASSES)
    testing(model, test_loader)
    wandb.finish()

def configure_wandb() -> None:
    wandb.login(key="e510b088c4a273c87de176015dc0b9f0bc30934e")
    wandb.init(
        # Set the project where this run will be logged
        project="vibdata-deeplearning-timedomain",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "arquitecture": "alexnet-1d"
        }
    )
    
def make_deterministic():
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
def train(train_loader : DataLoader, classes : np.array) -> nn.Module:
    print("Beginning training")

    from models import Alexnet1d
    net = Alexnet1d.alexnet(pretrained=False, out_channel=4)

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
            
            print(f"LABELS SIZE: {labels.shape} OUTPUTS SIZE: {outputs.shape}")

            train_loss += loss.item() * inputs.size(0)
            train_acc += torch.sum(torch.eq(labels,outputs)) / len(labels)
    
        print(f'[{epoch + 1 : 3d}] loss: {train_loss/i:.3f} | acc: {train_acc/i:.3f}')
        
        wandb.log({"loss":train_loss/i, "acc":train_acc/i})
        
                    
    print("Finished Training")
    return net

def testing(model : nn.Module, testloader : DataLoader):
    print("Testing")
    test_loss = 0.0
    correct, total = 0,0
    
    criterion = nn.CrossEntropyLoss()
    
    for data, labels in testloader:
        # Move to gpu
        if torch.cuda.is_available():
            data = data.float().cuda()
            labels = labels.cuda()
        
        output = model(data)
        for o,l in zip(torch.argmax(output,axis = 1),labels):
            if o == l:
                correct += 1
            total += 1
        loss = criterion(output,labels)
        test_loss += loss.item() * data.size(0)
        
    print(f'Testing Loss: {test_loss/len(testloader) :.5}')
    print(f'Correct Predictions: {correct}/{total}')

    
def load_train_test(dataset : DeepDataset):
    BATCH_SIZE = 32
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

    # Convert the dataset into a DeepDataset and save it on root_dir/<dataset_name> path
    convertDataset(raw_dataset, transforms, deep_dir / dataset_name)
    # Instantiate the deep dataset and load the transforms to be applied
    dataset = DeepDataset(deep_dir / dataset_name)
    return dataset


if __name__ == "__main__":
    main()