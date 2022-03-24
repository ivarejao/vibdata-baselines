from typing import Tuple
import torch
import numpy as np
from itertools import combinations

# Source: https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py


# class CoralLoss(torch.nn.Module):
#     def __init__(self, clf_loss, target_domain_id, lamb=1.0):
#         super().__init__()
#         self.clf_loss = clf_loss
#         self.lamb = lamb
#         self.target_domain_id = target_domain_id

#     def forward(self, Xpred, Y):
#         domain, yt = Y
#         Xtarget, Xsrc = Xpred
#         # Xtarget = Xpred[domain == self.target_domain_id]
#         source_domain_mask = domain != self.target_domain_id
#         # Xsrc = Xpred[source_domain_mask]
#         yt_src = yt[source_domain_mask]
#         domain_src = domain[source_domain_mask]

#         coralloss = coral(Xsrc, Xtarget)
#         clfloss = self.clf_loss(Xsrc, (domain_src, yt_src))
#         print(coralloss)
#         return {'loss': clfloss,# + self.lamb*coralloss,
#                 'coral_loss': coralloss, 'clf_loss': clfloss}

class CoralLoss(torch.nn.Module):
    def __init__(self, clf_loss=None, lamb=1.0):
        super().__init__()
        self.clf_loss = clf_loss
        self.lamb = lamb

    def forward(self, Xpred: Tuple[torch.Tensor, torch.Tensor], Y: Tuple):
        domain, _ = Y
        if(self.clf_loss is not None or isinstance(Xpred, tuple)):
            X, Xc = Xpred
        else:
            X = Xpred

        uniq_domains = torch.unique(domain)
        losses = {}
        avg_coralloss = torch.tensor(0.0, device=X.device)
        avg_meanloss = torch.tensor(0.0, device=X.device)
        avg_covloss = torch.tensor(0.0, device=X.device)
        for di, dj in combinations(uniq_domains, 2):
            Xi = X[domain == di]
            Xj = X[domain == dj]
            covloss, mean_loss = coral(Xi, Xj)
            coralloss = covloss + mean_loss
            # losses['coralloss_%d_%d' % (di, dj)] = coralloss
            avg_coralloss += coralloss
            avg_meanloss += mean_loss
            avg_covloss += covloss
        if(len(losses) != 0):  # is zero when only samples of a single domain are present
            avg_coralloss /= len(losses)
            avg_meanloss /= len(losses)
            avg_covloss /= len(losses)

        if(self.clf_loss is not None):
            clfloss = self.clf_loss(Xc, Y)
            losses['clf_loss'] = clfloss
        else:
            clfloss = 0.0
        losses.update({'loss': clfloss + self.lamb * avg_coralloss,
                       'coralloss': avg_coralloss, 'meanloss': avg_meanloss, 'covloss': avg_covloss})
        return losses


def coral(source, target):
    d = source.data.shape[1]
    mean_s = torch.mean(source, 0, keepdim=True)
    mean_t = torch.mean(target, 0, keepdim=True)

    # source covariance
    xm = mean_s - source
    xc = xm.t() @ xm

    # target covariance
    xmt = mean_t - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))

    mean_diff = (mean_s-mean_t).pow(2).mean()

    return loss/(4*d*d), mean_diff
