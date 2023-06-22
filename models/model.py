import torch
from xgboost import XGBClassifier

from .Resnet1d import resnet50
from .Alexnet1d import alexnet

models = {
    "alexnet": alexnet,
    "resnet": resnet50,
    "xgb": XGBClassifier,
}


class Model:
    def __init__(self, model_name: str = "alexnet", **kwargs):
        self.model_name = model_name
        self.key_values = kwargs

    def new(self):
        net = models[self.model_name](**self.key_values)
        if torch.cuda.is_available() and hasattr(net, "cuda"):
            net = net.cuda()
        return net
