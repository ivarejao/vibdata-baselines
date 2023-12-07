import numpy as np
import torch
import torch.nn as nn

class VGGish(nn.Module):
    def __init__(self, channels, out_channel, classifier=False):
        super().__init__()
        self.channels = channels
        self.classifier = classifier
        self.out_channel = out_channel
        
        model = torch.hub.load(
            "harritaylor/torchvggish",
            "vggish",
            postprocess=False,
            preprocess=False,
            pretrained=True,
        )
        self.conv = model

        if classifier:
            self.fc = nn.Linear(128 * channels, out_channel)

    def forward(self, x, **kwargs):
        input = x
        Xt: np.ndarray | None = None
        
        for i in range(self.channels):
            x = self.conv.forward()
            Xt = x

        return Xt if not self.classifier else self.fc(nn.functional.relu(Xt))