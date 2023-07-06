import torch
from xgboost import XGBClassifier

from .Resnet1d import resnet18
from .Alexnet1d import alexnet

models = {
    "Alexnet": alexnet,
    "Resnet50": resnet18,
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
