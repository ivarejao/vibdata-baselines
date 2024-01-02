import torch
import torch.nn as nn


class NaiveTime(nn.Module):
    def __init__(self, num_units=10, num_outputs=10):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 2, kernel_size=32)
        self.pool0 = nn.MaxPool1d(32)
        self.conv1 = nn.Conv1d(2, 4, kernel_size=32)
        self.pool1 = nn.MaxPool1d(32)

        # Define the nonlinear layer
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(376, num_outputs)

    def forward(self, X, **kwargs):
        X = self.pool0(self.nonlin(self.conv0(X)))
        X = self.pool1(self.nonlin(self.conv1(X)))
        X = torch.flatten(X, 1)
        X = self.nonlin(X)
        X = self.output(X)
        return X
