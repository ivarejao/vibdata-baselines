import torch.nn as nn
from torchvision.models import resnet18 as rn18


class BCSResNet18(nn.Module):

    def __init__(self, out_channel: int = 4, pretrained=True):
        super().__init__()
        self.pretrained = pretrained
        self.conv = rn18(weights="IMAGENET1K_V1" if pretrained else None)
        self.conv.fc = nn.Identity()
        self.fc = nn.Linear(512, out_channel)

    def forward(self, input):
        input = self.conv(input)
        input = self.fc(nn.functional.relu(input))
        return input


def resnet18(pretrained=False, **kwargs):
    model = BCSResNet18(pretrained=pretrained, **kwargs)
    return model
