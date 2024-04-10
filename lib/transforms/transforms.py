import numpy as np
import pandas as pd
import pywt

import vibdata.deep.signal.transforms as deep_transforms
# from rpdbcs.experimental.fft import SensorUnit, unit_conversion_factor

from lib.utils.transforms import resize
from lib.models.vgg_models import VGGish

from scipy.signal import resample_poly, spectrogram
from tftb.processing import PseudoWignerVilleDistribution as PWV
from matplotlib.image import imsave
from PIL.Image import fromarray


class Resize(deep_transforms.Transform):
    def __init__(self, size: list = []) -> None:
        super().__init__()
        self.size = size

    def transform(self, data):
        data = data.copy()

        resized = []
        for s in data["signal"]:
            image = np.array([resize(np.squeeze(s), self.size)]).squeeze()
            resized.append(image)
        data["signal"] = resized

        return data


class VGGishTransformation(deep_transforms.Transform):
    def __init__(self, preprocess=False, postprocess=False) -> None:
        super().__init__()
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.model = VGGish(preprocess=preprocess, postprocess=postprocess)

    def transform(self, data):
        data = data.copy()

        fs = data["metainfo"].iloc[0]["sample_rate"]
        data["signal"] = self.model.transform(data["signal"], fs=fs)

        signals = len(data["signal"])
        info = data["metainfo"]
        if signals != len(info):
            data["metainfo"] = pd.DataFrame(np.repeat(info.values, signals, axis=0), columns=info.columns)

        return data


class Luciano(deep_transforms.Transform):
    def __init__(
        self,
        size: list = None,
        transforms: list = ["scalogram", "pseudo_wigner_ville", "spectrogram"],
        params: dict = None,
        image_examples: int = 0,
    ) -> None:
        super().__init__()
        self.labels = {}
        self.size = size

        transforms_dict = {
            "scalogram": Scalogram,
            "pseudo_wigner_ville": PseudoWignerVille,
            "spectrogram": SpectrogramCustom,
        }
        self.transforms = [(i, transforms_dict[i]) for i in transforms]

        self.params = params

        self.image_examples = image_examples

    def _execute_transform(self, data, transform: deep_transforms.Transform, kwargs: dict):
        transformed_signal = transform(**kwargs).transform(data)
        if len(transformed_signal["signal"].shape) > 3:
            transformed_signal["signal"] = np.squeeze(transformed_signal["signal"], 1)
        return transformed_signal

    def transform(self, data):
        output = []
        label: int = data["metainfo"].iloc[0]["label"]

        print_image = label not in self.labels or (label in self.labels and self.labels[label] < self.image_examples)

        for name, t in self.transforms:
            if not print_image and self.image_examples > 0:
                transformed_data = {"signal": np.zeros([1] + self.size), "metadata": []}
            else:
                transformed_data = self._execute_transform(data, t, {"size": self.size, **self.params.get(name)})
                if self.image_examples > 0:
                    item = self.labels[label] if label in self.labels else 0
                    imsave(f"{name}_{label}_{item}.png", transformed_data["signal"][0])

            output.append(transformed_data["signal"])

        while len(output) < 3:
            output.append(np.zeros([len(output[0])] + self.size))
        data["signal"] = np.swapaxes(np.array(output), 0, 1)

        if print_image and self.image_examples > 0:
            if label not in self.labels:
                self.labels.update({label: 1})
            else:
                self.labels[label] += 1

        return data


class SpectrogramCustom(deep_transforms.Transform):
    def __init__(
        self,
        size: list = [224, 224],
        resolution: float = 1 / 8,
        set_nperseg: bool = False,
        set_noverlap: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.resolution = resolution
        self.set_nperseg = set_nperseg
        self.set_noverlap = set_noverlap

    def transform(self, data):
        data = data.copy()

        sample_rate = data["metainfo"].iloc[0]["sample_rate"]
        nperseg = round(sample_rate / self.resolution)

        output = []
        for signal in data["signal"]:
            sample_frequencies, time_segments, spectro = spectrogram(
                signal,
                fs=sample_rate,
                nperseg=nperseg if self.set_nperseg else None,
                noverlap=nperseg // 8 if self.set_noverlap else None,
                window=("tukey", 0.25),
                scaling="spectrum",
                detrend=False,
                return_onesided=True,
                mode="magnitude",
            )
            spectro *= 2

            # Remove frequency 0
            sample_frequencies = sample_frequencies[1:]
            spectro = spectro[1:]

            # norm_factor = unit_conversion_factor(SensorUnit.volt, 100.0) / (2 * np.pi)
            # spectro *= norm_factor / sample_frequencies[:, np.newaxis]
            spectro /= sample_frequencies[:, np.newaxis]
            spectro = np.clip(spectro, 0, 1.0)

            image = np.array([resize(spectro, self.size)]) if self.size else spectro
            output.append(image)

        data["signal"] = np.array(output)
        return data


class Scalogram(deep_transforms.Transform):
    def __init__(self, size: list = [224, 224], scales: int = 128, wavelet: str = "morl") -> None:
        super().__init__()
        self.size = size
        self.scales = scales
        self.wavelet = wavelet

    def transform(self, data):
        data = data.copy()

        scales = np.linspace(1, self.scales)  # 32, 64, 128

        # isso vai servir para conferir se está mostrando os picos
        # rotation_freq = data["metainfo"].iloc[0]["rotation_hz"]
        # velocidade do motor -> rotation_hz
        # harmonicas: multiplos da carga que devem aparecer o pico

        sample_rate = data["metainfo"].iloc[0]["sample_rate"]

        output = []
        for s in data["signal"]:
            # testar chapeu mexicano
            coefficients, _ = pywt.cwt(s, scales, self.wavelet, sampling_period=1 / sample_rate)
            amplitudes = abs(coefficients)

            image = np.array([resize(amplitudes, self.size)]) if self.size else amplitudes
            output.append(image)

        data["signal"] = np.array(output)
        return data


class PseudoWignerVille(deep_transforms.Transform):
    def __init__(self, size: list = [224, 224], n_fbins: int = 6_000) -> None:
        super().__init__()
        self.size = size
        self.n_fbins = n_fbins

    def transform(self, data):
        data = data.copy()

        sample_rate = data["metainfo"].iloc[0]["sample_rate"]

        # se ficar pesado, usar downsampling
        # downsampling_factor = int(sample_rate / 260)  # alterar esse número para ver mudanças (original: 260)
        # new_sample_rate = sample_rate / downsampling_factor

        output = []
        for signal in data["signal"]:
            # array_resampled = resample_poly(signal, 1, downsampling_factor)
            # timestamps = np.arange(array_resampled.size) / new_sample_rate
            # pwv = PWV(array_resampled, timestamps=timestamps, n_fbins=7000)

            timestamps = np.arange(signal.size) / sample_rate
            pwv = PWV(signal, timestamps=timestamps, n_fbins=self.n_fbins)  # ir testando n_fbins

            tf, ts, freqs = pwv.run()

            tf_positive = tf[: len(tf) // 2]  # Remove negative frequencies
            # tf_positive *= 2  # ver se faz diferença esse x2
            tf_positive = np.abs(tf_positive)

            image = np.array([resize(tf_positive, self.size)]) if self.size else tf_positive
            output.append(image)

        data["signal"] = np.array(output)
        return data
