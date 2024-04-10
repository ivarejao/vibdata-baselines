import numpy as np
import torch

from sklearn.base import BaseEstimator


class VGGish(BaseEstimator):
    def __init__(self, preprocess=False, postprocess=False) -> None:
        model = torch.hub.load(
            "harritaylor/torchvggish", "vggish", postprocess=postprocess, preprocess=preprocess, pretrained=True, verbose=True
        )
        model.eval()
        model.train(False)
        model.requires_grad_(False)
        self.vgg = model
        self.preprocess = preprocess
        self.postprocess = postprocess

    def fit(self, X: np.ndarray, y=None, **kwargs):
        return self

    def transform(self, X, fs=None) -> torch.Tensor:
        if self.preprocess:
            if fs is None:
                raise ValueError("Sample rate ('fs') cannot be 'None' if the preprocessing step is set to 'True'")
            Xt: torch.Tensor = self.vgg.forward(X[0], fs=fs).to(device="cpu")
            return Xt.detach()

        X = torch.Tensor(np.array(X))
        Xt = self.vgg.forward(X.unsqueeze(1)).to(device="cpu")
        return Xt

    def fit_transform(self, X, y=None):
        return self.transform(X)
