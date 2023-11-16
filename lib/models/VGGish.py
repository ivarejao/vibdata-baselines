import numpy as np
import torch
import torch.nn as nn

class VGGish(nn.Module):
    def __init__(self, channels, out_channel, classifier=False):
        super().__init__()
        self.channels = channels
        self.classifier = classifier
        self.out_channel = out_channel
        
        model = torch.hub.load(
            "harritaylor/torchvggish",
            "vggish",
            postprocess=False,
            preprocess=False,
            pretrained=True,
        )
        self.conv = model

        if classifier:
            self.fc = nn.Linear(128 * channels, out_channel)

    def forward(self, x, **kwargs):
        input = x
        Xt: np.ndarray | None = None
        
        for i in range(self.channels):
            images = to_spectrogram(input[:, i])
            x = self.conv.forward(images)
            Xt = x

        return Xt if not self.classifier else self.fc(nn.functional.relu(Xt))

def signal_to_image(array, entry):
        sample_rate = entry

        sample_frequencies, time_segments, spectro = spectrogram(
            array,
            fs=sample_rate,
            nperseg=sample_rate,
            window=('tukey', 0.25),
            scaling='spectrum',
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        sample_frequencies = sample_frequencies[1:]
        spectro = 2 * spectro[1:]

        mask = (sample_frequencies > 0.8) & (sample_frequencies < 130)
        norm_factor = unit_conversion_factor(SensorUnit.volt, 100.0) / (2 * np.pi)

        spectro *= norm_factor / sample_frequencies[:, np.newaxis]
        spectro = np.clip(spectro, 0.0, 1.0)

        spectrogram_image = spectro[mask]
        spectrogram_image = cv.resize(spectrogram_image, [96, 64])

        return spectrogram_image
    
def to_spectrogram(array):

    images = []
    for signal in array:
        signal = signal.detach().cpu().numpy()
        image = signal_to_image(signal, signal.shape[-1])
        images.append(image)
 
    images = np.expand_dims(np.array(images), axis=1)
    return torch.tensor(np.array(images)).to(array.device)