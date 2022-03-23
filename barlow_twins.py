from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers.do_nothing_reducer import DoNothingReducer
import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class BarlowTwins(BaseMetricLossFunction):
    def __init__(self, alpha=1.0, add_loss_func=None, lamb=1.0, reducer=DoNothingReducer(), use_positive=True, **kwargs):
        super().__init__(reducer=reducer, **kwargs)
        self.alpha = alpha
        self.add_loss_func = add_loss_func
        self.lamb = lamb
        self.use_positive = use_positive

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb=None, ref_labels=None):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        if(len(torch.unique(labels)) == 1):
            ret = {
                "loss": {
                    "losses": 0.0,
                    "indices": None,
                    "reduction_type": "already_reduced",
                },
                "invariance_loss": {
                    "losses": 0.0,
                    "indices": None,
                    "reduction_type": "already_reduced",
                },
                "redundancy_loss": {
                    "losses": 0.0,
                    "indices": None,
                    "reduction_type": "already_reduced",
                }
            }
            return ret

        a, p, a2, n = lmu.convert_to_pairs(indices_tuple, labels)
        if(self.use_positive):
            o = p
        else:
            a = a2
            o = n
        emb_a = embeddings[a]
        emb_p = embeddings[o]
        emb_a = (emb_a-emb_a.mean(0))/emb_a.std()
        emb_p = (emb_p-emb_p.mean(0))/emb_p.std()
        c = torch.mm(emb_a.T, emb_p)
        c.div_(emb_a.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
        off_diag = off_diagonal(c).pow_(2).mean()
        loss = on_diag + self.alpha * off_diag

        if(self.add_loss_func is not None):
            second_loss = self.add_loss_func.forward(embeddings, labels, indices_tuple)
            loss *= self.lamb
            loss += second_loss
            ret_add = {
                'second_loss': {
                    'losses': second_loss,
                    'indices': None,
                    "reduction_type": "already_reduced"
                }
            }
        else:
            ret_add = {}

        ret = {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            },
            "invariance_loss": {
                "losses": on_diag,
                "indices": None,
                "reduction_type": "already_reduced",
            },
            "redundancy_loss": {
                "losses": off_diag,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
        ret.update(ret_add)

        return ret

    def _sub_loss_names(self):
        return ["loss", "invariance_loss", "redundancy_loss"]

    def get_default_reducer(self):
        return DoNothingReducer()
