import numpy as np
import torch

from sklearn.base import BaseEstimator
from skorch.helper import SliceDict

from utils.transforms import to_spectrogram

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
        x = torch.Tensor(X['x']) if isinstance(X, SliceDict) else torch.Tensor(X)
        
        output = []
        for channel in range(self.channels):
            j = 0
            data = None
            batch_size = 400
            while j < len(x):
                x_ = x[j:j+min(batch_size, len(x)-j)].to(device='cpu').unsqueeze(1)
                images = to_spectrogram(x_.squeeze())
                Xt = self.vgg.forward(images).to(device='cpu')
                data = Xt.numpy() if data is None else np.concatenate((data, Xt), axis=0)
                j += batch_size
            output.append(data)
        
        output = np.concatenate(output, axis=0)
        if len(output.shape) > 2:
            output = np.moveaxis(output, [0, 1], [1, 0])
            
        if isinstance(X, SliceDict):
            return SliceDict(x=torch.Tensor(output), hc=X['hc'], y=X['y'])
        return torch.Tensor(output)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
