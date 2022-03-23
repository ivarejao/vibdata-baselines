from torch import nn
import skorch
import numpy as np
import torch
from skorch_extra.callbacks import EstimatorEpochScoring
from torch.utils.data.dataset import Subset


class DomainAdapLoss(nn.Module):
    """
    This loss requires the output of a domain recognizer and the output of a classifier.
    """

    def __init__(self, alpha=1.0,
                 cl_criterion=nn.CrossEntropyLoss(), da_criterion=nn.CrossEntropyLoss(),
                 macroavg_clloss_perdomain=False):
        super().__init__()
        self.alpha = alpha
        self.cl_criterion = cl_criterion
        self.da_criterion = da_criterion
        self.macroavg_clloss_perdomain = macroavg_clloss_perdomain

    def forward(self, preds, y):
        ypd, ypc = preds
        # yd, yc = y[:, 0], y[:, 1]
        yd, yc = y

        uniq_domains = torch.unique(yd)

        if(self.macroavg_clloss_perdomain):
            cl_loss = 0.0
            for d in uniq_domains:
                idxs = yd == d
                ypi, yti = ypc[idxs], yc[idxs]
                l = self.cl_criterion.forward(ypi, yti)
                cl_loss += l

            cl_loss /= len(uniq_domains)
        else:
            cl_loss = self.cl_criterion(ypc, yc)
        da_loss = self.da_criterion(ypd, yd)

        return {'loss': cl_loss + self.alpha*da_loss,
                'cl_loss': cl_loss, 'da_loss': da_loss}


class LossPerDomain(nn.Module):
    """
    This is a macro average of a loss over domains.
    """

    def __init__(self, base_loss, **kwargs) -> None:
        super().__init__()
        if(len(kwargs) > 0):
            self.base_loss = base_loss(**kwargs)
        else:
            self.base_loss = base_loss

    def forward(self, yp, Y):
        domain, yt = Y  # parece que estava errado antes.

        uniq_domains = torch.unique(domain)
        loss = 0.0
        for d in uniq_domains:
            idxs = domain == d
            ypi, yti = yp[idxs], yt[idxs]
            l = self.base_loss.forward(ypi, yti)
            loss += l

        return loss/len(uniq_domains)


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

class DomainRecognizerCallback(EstimatorEpochScoring):
    def __init__(self, domains, estimator, metric='f1_macro', name='score', lower_is_better=False, use_caching=False, on_train=False, **kwargs):
        """
        Args:
            domains (array of ints): array of same size as the dataset, telling the domain each sample belongs.
        """
        super().__init__(estimator, metric, name, lower_is_better, use_caching, on_train, **kwargs)
        self.domains = domains

    def get_test_data(self, dataset_train: Subset, dataset_valid: Subset):
        X, Y, P = super().get_test_data(dataset_train, dataset_valid)
        dtrain = self.domains[dataset_train.indices]
        dvalid = self.domains[dataset_valid.indices]
        return X, (dtrain, dvalid), P