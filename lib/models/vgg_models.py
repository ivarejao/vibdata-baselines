import numpy as np
import torch

from sklearn.base import BaseEstimator
from skorch.helper import SliceDict
from utils.transforms import resize

class VGGish(BaseEstimator):
    def __init__(self, channels: int) -> None:
        self.channels = channels
        
        model = torch.hub.load('harritaylor/torchvggish', 'vggish', postprocess=False, preprocess=False, pretrained=True)
        model.eval()
        model.train(False)
        model.requires_grad_(False)
        self.vgg = model
        
        super().__init__()
        
    def fit(self, X : np.ndarray, y=None, **kwargs):        
        return self

    def transform(self, X, **kwargs):
        # x = torch.Tensor(X['x']) if isinstance(X, SliceDict) else torch.Tensor(X)
        
        output = []
        for channel in range(self.channels):
            j = 0
            for x in X:
                x = torch.Tensor(x)
                # print("[DEBUG] x.squeeze:", x.squeeze().shape)
                image = torch.Tensor(np.array([resize(x.squeeze().numpy(), [96, 64])]))
                # print("[DEBUG] image:", image.shape)
                Xt = self.vgg.forward(image.unsqueeze(0)).to(device='cpu')
                data = Xt.squeeze().numpy()
                output.append(data)
        
        output = np.concatenate([output], axis=0)
        # print("[DEBUG] Output:", output.shape)
        return output
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

