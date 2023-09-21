import torch
import torch.nn as nn
from xgboost import XGBClassifier

from .Resnet1d import resnet18, resnet34
from .Alexnet1d import alexnet

models = {
    "Alexnet": alexnet,
    "Resnet18": resnet18,
    "Resnet34": resnet34,
    "XGBClassifier": XGBClassifier,
}


class Model:
    def __init__(self, model_name: str = "Alexnet", **kwargs):
        self.model_name = model_name
        self.key_values = kwargs

    def new(self):
        net = models[self.model_name](**self.key_values)
        if torch.cuda.is_available() and hasattr(net, "cuda"):
            net = net.cuda()
        return net

    @classmethod
    def weights_init(cls, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight.data)

    @classmethod
    def reset_weights(cls, m):
        """
        Try resetting model weights to avoid
        weight leakage.
        """
        for layer in m.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
