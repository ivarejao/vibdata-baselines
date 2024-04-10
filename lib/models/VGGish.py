import torch
import torch.nn as nn


class VGGish(nn.Module):
    def __init__(self, out_channel, softmax, **kwargs):
        super().__init__()
        self.softmax = softmax
        self.fc = nn.Linear(128, out_channel, dtype=torch.float32)

    def forward(self, x, **kwargs):
        x = x.squeeze(1).squeeze(1)
        if self.softmax:
            return nn.functional.softmax(self.fc(x), dim=1)
        return self.fc(x)
