from torch import nn
import skorch
import numpy as np

class DomainAdapLoss(nn.Module):
    def __init__(self, alpha=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.cl_criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.da_criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, preds, y):
        ypd, ypc = preds
        yd, yc = y[:, 0], y[:, 1]
        da_loss = self.da_criterion(ypd, yd)
        cl_loss = self.cl_criterion(ypc, yc)

        return cl_loss + self.alpha*da_loss


class DomainAdapCallbackScore:
    def __init__(self, scorer, on_class=True) -> None:
        self.scorer = scorer
        self.on_class = on_class

    def __call__(self, net: skorch.NeuralNet, X, y):
        ypd, ypc = net.forward(X)
        if(self.on_class):
            yp = ypc.argmax(dim=1)
            yt = y[:, 1]
        else:
            yp = ypd.argmax(dim=1)
            yt = y[:, 0]
        return self.scorer(yt, yp.detach().numpy())
