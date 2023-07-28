import random

import numpy as np

from npyx.corr import acg as make_acg
from npyx.datasets import preprocess_template, resample_acg

from .dataset_init import calc_snr


class GaussianNoise(object):
    """Adds Gaussian noise to the waveform"""

    def __init__(self, p=0.3, std=0.1):
        self.p = p
        self.std = std

    def __call__(self, sample):
        if self.p <= np.random.rand():
            return sample
        noise = np.random.normal(0, self.std * np.std(sample), size=sample.shape)
        return sample + noise


class SelectWave(object):
    """Selects a single wave from the multi channel waveform"""

    def __init__(self, window=3, n_channels=10):
        self.window = window
        self.n_channels = n_channels

    def __call__(self, sample):
        sample = sample.reshape(self.n_channels, -1)
        sample = sample[
            sample.shape[0] // 2 - self.window : sample.shape[0] // 2 + self.window, :
        ]
        snrs = np.array([calc_snr(wave) for wave in sample])
        good_mask = snrs > 20

        if not np.any(good_mask):
            return preprocess_template(sample[sample.shape[0] // 2, :])
        snrs = snrs[good_mask]
        sample = sample[good_mask, :]
        # chosen = np.random.choice(np.arange(len(sample)), p=snrs / np.sum(snrs))
        chosen = np.random.choice(np.arange(len(sample)))
        sample = sample[chosen, :]

        return preprocess_template(sample)
