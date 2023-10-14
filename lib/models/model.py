import torch
import torch.nn as nn
from xgboost import XGBClassifier

from .Resnet1d import resnet18, resnet34
from .Alexnet1d import alexnet

models = {
    "alexnet": alexnet,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "xgbclassifier": XGBClassifier,
}


class Model:
    def __init__(self, model_name: str = "Alexnet", **kwargs):
        self.model_name = model_name
        self.key_values = kwargs

    def new(self, device: torch.device = None, **kwargs):
        net = models[self.model_name](**self.key_values, **kwargs)
        if torch.cuda.is_available() and hasattr(net, "cuda"):
            net = net.cuda() if not device else net.to(device)
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

    @classmethod
    def initialize_layers(cls, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)

    @classmethod
    def bias_init(cls, m):
        if isinstance(m, nn.Linear) and (m.out_features == 4):
            # CWRU_dataset dist
            dataset_dist = torch.Tensor(
                [0.09204470742932282, 0.30046022353714663, 0.3037475345167653, 0.3037475345167653]
            )
            m.bias.data = dataset_dist.cuda()
