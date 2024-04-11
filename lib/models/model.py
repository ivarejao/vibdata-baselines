import torch
import torch.nn as nn
from tsai.models.XResNet1d import xresnet1d18
from lib.models.Resnet1d import resnet18 as resnet1d18
from lib.models.ResNet import resnet18
from .VGGish import VGGish

models = {
    "resnet18": resnet18,
    "resnet1d18": resnet1d18,
    "xresnet18": xresnet1d18,
    "vggish": VGGish,
}


class Model:
    def __init__(self, model_name: str = "Alexnet", **kwargs):
        self.model_name = model_name
        self.key_values = kwargs

    def new(self, **kwargs):
        net = models[self.model_name](**self.key_values, **kwargs)
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
