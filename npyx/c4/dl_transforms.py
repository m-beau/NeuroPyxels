import random

import numpy as np
from npyx.corr import acg as make_acg
from npyx.datasets import preprocess_template, resample_acg

from .dataset_init import BIN_SIZE, WIN_SIZE

ACG_LEN = int(WIN_SIZE / BIN_SIZE // 2)


WAVEFORM_SAMPLES = 90
N_CHANNELS = 4


class ConformWaveform(object):
    def __init__(self):
        pass

    def __call__(self, sample, spikes=None):
        istuple = isinstance(sample, (tuple, list))
        if istuple:
            data_point, label = sample
        else:
            data_point = sample
        data_point = np.squeeze(data_point)

        wf = data_point.reshape(N_CHANNELS, WAVEFORM_SAMPLES)
        wf = preprocess_template(wf)
        new_data_point = wf.ravel().copy().reshape(1, -1).astype("float32")
        if istuple:
            transformed_sample = (new_data_point, label)
            return (
                (transformed_sample, spikes)
                if spikes is not None
                else transformed_sample
            )
        return (new_data_point, spikes) if spikes is not None else new_data_point


class SwapChannels(object):
    """
    Swaps the indices of even and odd channels, mimicking the biological
    scenario in which the probe was oriented in the same way along the longitudinal axis
    but in the opposite way along the dorsoventral axis.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample, spikes=None):
        istuple = isinstance(sample, (tuple, list))
        if self.p <= np.random.rand():
            return (sample, spikes) if spikes is not None else sample

        if istuple:
            data_point, label = sample
        else:
            data_point = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use a waveform dataset
        if len(data_point.ravel()) == N_CHANNELS * WAVEFORM_SAMPLES:
            wvf = data_point
            acg = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        wvf = wvf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)
        channels = np.arange(N_CHANNELS)
        evens = channels[::2]
        odds = channels[1::2]
        if len(evens) == len(odds):
            new_channels = np.array(list(zip(odds, evens))).ravel()
        else:
            new_channels = np.array(list(zip(odds, evens))).ravel()
            new_channels = np.append(new_channels, evens[-1])
        new_wvf = wvf[new_channels, :].ravel().copy()
        new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")

        if istuple:
            transformed_sample = (new_data_point, label)
            return (
                (transformed_sample, spikes)
                if spikes is not None
                else transformed_sample
            )
        return (new_data_point, spikes) if spikes is not None else new_data_point


class VerticalReflection(object):
    """
    Reverses the indices of the waveform channels,
    mimicking the  scenario in which the probe was oriented in the same way along
    the dorsoventral axis but in the opposite way along the longitudinal axis.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample, spikes=None):
        istuple = isinstance(sample, tuple)
        if self.p <= np.random.rand():
            return (sample, spikes) if spikes is not None else sample
        if istuple:
            data_point, label = sample
        else:
            data_point = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use a waveform dataset
        if len(data_point.ravel()) == N_CHANNELS * WAVEFORM_SAMPLES:
            wvf = data_point
            acg = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        new_wvf = wvf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)[::-1].ravel().copy()
        new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")

        if istuple:
            transformed_sample = (new_data_point, label)
            return (
                (transformed_sample, spikes)
                if spikes is not None
                else transformed_sample
            )
        return (new_data_point, spikes) if spikes is not None else new_data_point


class GaussianNoise(object):
    """Adds random Gaussian noise to the image."""

    def __init__(self, p=0.3, eps_multiplier=1):
        self.eps = eps_multiplier
        self.p = p

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
            acg = data_point
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        wvf_std = np.std(wvf)
        acg_std = np.std(acg)

        new_wvf = wvf + np.random.normal(0, self.eps * wvf_std, wvf.shape)

        if label in ["PkC_ss", 5]:
            new_acg = acg
        else:
            new_acg = acg + np.random.normal(0, self.eps * acg_std, acg.shape)
        new_acg = np.clip(new_acg, 0, None)
        new_acg = np.nan_to_num(new_acg)
        assert (
            new_acg.shape[0] == ACG_LEN
        ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = (
            np.concatenate((new_wvf, new_acg)).reshape(1, -1).astype("float32")
        )
        transformed_sample = (new_data_point, label)
        return transformed_sample, spikes


class DeleteSpikes(object):
    """Deletes a random portion of the spikes in an ACG"""

    def __init__(
        self,
        p=0.3,
        deletion_prob=0.1,
        acg_scaling=None,
    ):
        self.p = p
        self.deletion_prob = deletion_prob
        self.acg_scaling = acg_scaling

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]

        new_spikes = spikes[
            np.random.choice(
                [0, 1],
                size=(spikes.shape[0]),
                p=[self.deletion_prob, 1 - self.deletion_prob],
            ).astype(bool)
        ]
        new_acg = make_acg(".npyx_placeholder", 4, BIN_SIZE, WIN_SIZE, train=new_spikes)

        if self.acg_scaling is not None:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :]) / self.acg_scaling
        else:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :])
            new_acg = np.clip(new_acg / np.max(new_acg), 0, 10)
            new_acg = np.nan_to_num(new_acg)

        assert (
            new_acg.shape[0] == ACG_LEN
        ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
        transformed_sample = (new_data_point, label)
        return transformed_sample, new_spikes


class MoveSpikes(object):
    """Randomly moves the spikes in a spike train by a maximum amount"""

    def __init__(self, p=0.3, max_shift=10, acg_scaling=None):
        self.p = p
        self.max_shift = int(np.ceil(max_shift))  # To work with RandAugment behavior
        self.acg_scaling = acg_scaling

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]

        random_moving = np.random.choice(
            np.arange(-self.max_shift, self.max_shift), size=spikes.shape[0]
        )
        new_spikes = (spikes + random_moving).astype(int)
        new_acg = make_acg(".npyx_placeholder", 4, BIN_SIZE, WIN_SIZE, train=new_spikes)

        if self.acg_scaling is not None:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :]) / self.acg_scaling
        else:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :])
            new_acg = np.clip(new_acg / np.max(new_acg), 0, 10)
            new_acg = np.nan_to_num(new_acg)

        assert (
            new_acg.shape[0] == ACG_LEN
        ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
        transformed_sample = (new_data_point, label)
        return transformed_sample, new_spikes


class AddSpikes(object):
    """Adds a random amount of spikes (in percentage) to the spike list and recomputes the ACG"""

    def __init__(self, p=0.3, max_addition=0.1, acg_scaling=None):
        self.p = p
        self.max_addition = max_addition
        self.acg_scaling = acg_scaling

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]

        random_addition = np.random.randint(
            low=spikes[0],
            high=spikes[-1],
            size=int(spikes.shape[0] * self.max_addition),
        )
        new_spikes = np.unique(np.concatenate((spikes, random_addition)))
        new_acg = make_acg(".npyx_placeholder", 4, BIN_SIZE, WIN_SIZE, train=new_spikes)

        if self.acg_scaling is not None:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :]) / self.acg_scaling
        else:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :])
            new_acg = np.clip(new_acg / np.max(new_acg), 0, 10)
            new_acg = np.nan_to_num(new_acg)

        assert (
            new_acg.shape[0] == ACG_LEN
        ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
        transformed_sample = (new_data_point, label)
        return transformed_sample, new_spikes


class HorizontalCompression(object):
    """Compress or expand the signal horizontally by a given factor."""

    def __init__(self, p=0.3, max_compression_factor=0.6):
        self.max_compression_factor = max_compression_factor
        self.p = p

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
            acg = data_point
        elif len(data_point.ravel()) == N_CHANNELS * WAVEFORM_SAMPLES:
            wvf = data_point.reshape(N_CHANNELS, WAVEFORM_SAMPLES)
            acg = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES].reshape(
                N_CHANNELS, WAVEFORM_SAMPLES
            )
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        used_factor = np.random.choice(
            np.linspace(0.1, self.max_compression_factor, 5), size=1
        )

        factor = 1 + used_factor if np.random.choice([0, 1]) == 1 else 1 - used_factor
        if len(acg) > 0:
            new_acg = np.interp(
                np.arange(0, len(acg), factor), np.arange(len(acg)), acg
            )
            if len(new_acg) != ACG_LEN:
                diff = len(new_acg) - ACG_LEN
                if diff > 0:  # Crop
                    crop_left = diff // 2
                    crop_right = diff - crop_left
                    new_acg = new_acg[crop_left:-crop_right]
                else:  # Pad
                    pad_left = -diff // 2
                    pad_right = -diff - pad_left
                    new_acg = np.pad(new_acg, (pad_left, pad_right), mode="reflect")
        else:
            new_acg = np.array([])

        if len(wvf) > 0:
            new_wvf_shape = (N_CHANNELS, int(np.ceil(wvf.shape[1] / factor)))
            new_wvf = np.zeros(new_wvf_shape)
            for i in range(N_CHANNELS):
                new_wvf[i] = np.interp(
                    np.arange(0, wvf.shape[1], factor), np.arange(wvf.shape[1]), wvf[i]
                )
            if new_wvf.shape[1] != WAVEFORM_SAMPLES:
                diff = new_wvf.shape[1] - WAVEFORM_SAMPLES
                if diff > 0:  # Crop
                    crop_left = diff // 2
                    crop_right = diff - crop_left
                    new_wvf = new_wvf[:, crop_left:-crop_right]
                else:  # Pad
                    pad_left = -diff // 2
                    pad_right = -diff - pad_left
                    new_wvf = np.pad(
                        new_wvf, ((0, 0), (pad_left, pad_right)), mode="reflect"
                    )
            new_wvf = new_wvf.ravel().copy()
        else:
            new_wvf = np.array([])

        if len(wvf) > 0 and len(acg) > 0:
            assert (
                new_acg.shape[0] == ACG_LEN
            ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = (
            np.concatenate((new_wvf, new_acg)).reshape(1, -1).astype("float32")
        )
        transformed_sample = (new_data_point, label)
        return transformed_sample, spikes


class ConstantShift(object):
    """Randomly shift the signal up or down by a given scalar amount."""

    def __init__(self, p=0.3, scalar=0.1):
        self.scalar = scalar
        self.p = p

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
            acg = data_point
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        acg_amount = self.scalar * np.mean(acg)
        wvf_amount = self.scalar * np.mean(wvf)

        new_acg = (
            (acg_amount + acg) if np.random.choice([0, 1]) == 1 else (acg - acg_amount)
        )
        new_acg = np.clip(new_acg, 0, None)
        new_acg = np.nan_to_num(new_acg)
        new_wvf = (
            (wvf_amount + wvf) if np.random.choice([0, 1]) == 1 else (wvf - wvf_amount)
        )

        assert (
            new_acg.shape[0] == ACG_LEN
        ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = (
            np.concatenate((new_wvf, new_acg)).reshape(1, -1).astype("float32")
        )
        transformed_sample = (new_data_point, label)
        return transformed_sample, spikes


class DeleteChannels(object):
    """Randomly delete some channels in the recording.

    Args:
        p (float): Probability of deleting a channel.
        n_channels (int): Number of channels to delete.
    """

    def __init__(self, p=0.3, n_channels=1):
        self.p = p
        self.n_channels = int(np.ceil(n_channels))  # To work with RandAugment behavior

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use a waveform dataset
        if len(data_point.ravel()) == N_CHANNELS * WAVEFORM_SAMPLES:
            wvf = data_point
            acg = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        wvf = wvf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)
        deleted_channels = np.random.choice(
            np.arange(wvf.shape[0]),
            size=self.n_channels,
            replace=False,
        )
        noise = np.random.rand(wvf.shape[1]) * np.std(wvf[0, :])
        new_wvf = wvf.copy()
        new_wvf[deleted_channels, :] = noise
        new_wvf = new_wvf.ravel()
        new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")

        transformed_sample = (new_data_point, label)
        return transformed_sample, spikes


class NewWindowACG(object):
    """Recomputes the given acg with a different bin_size and window_size."""

    def __init__(
        self,
        p=0.3,
        magnitude_change=3,
        acg_scaling=None,
    ):
        self.p = p
        self.magnitude_change = magnitude_change
        self.acg_scaling = acg_scaling

    def __call__(self, sample, spikes):
        if self.p <= np.random.rand():
            return sample, spikes
        data_point, label = sample
        data_point = np.squeeze(data_point)

        # Handle the case when we only use an ACG Dataset
        if len(data_point.ravel()) == ACG_LEN:
            wvf = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]

        new_acg = make_acg(
            None,
            4,
            (0.5 * self.magnitude_change),
            (100 * self.magnitude_change),
            train=spikes,
        )

        if self.acg_scaling is not None:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :]) / self.acg_scaling
        else:
            new_acg = resample_acg(new_acg[len(new_acg) // 2 :])
            new_acg = np.clip(new_acg / np.max(new_acg), 0, 10)
            new_acg = np.nan_to_num(new_acg)

        assert (
            new_acg.shape[0] == ACG_LEN
        ), f"ACG length is different than expected. Got {new_acg.shape[0]} instead of {ACG_LEN}"
        new_data_point = np.concatenate((wvf, new_acg)).reshape(1, -1).astype("float32")
        transformed_sample = (new_data_point, label)
        return transformed_sample, spikes


class PermuteChannels(object):
    """Randomly permutes some channels in the recording.

    Args:
        p (float): Probability of applying the permutation.
        n_channels (int): Number of channels to permute.
    """

    def __init__(self, p=0.3, n_channels=1):
        self.p = p
        self.n_channels = int(np.ceil(n_channels))  # To work with RandAugment behavior
        assert self.n_channels <= N_CHANNELS // 2, "Too many channels to permute"

    def __call__(self, sample, spikes=None):
        istuple = isinstance(sample, tuple)
        if self.p <= np.random.rand():
            return (sample, spikes) if spikes is not None else sample
        if istuple:
            data_point, label = sample
        else:
            data_point = sample
        data_point = np.squeeze(data_point)
        # Handle the case when we only use a waveform dataset
        if len(data_point.ravel()) == N_CHANNELS * WAVEFORM_SAMPLES:
            wvf = data_point
            acg = np.array([])
        else:
            wvf = data_point[: N_CHANNELS * WAVEFORM_SAMPLES]
            acg = data_point[N_CHANNELS * WAVEFORM_SAMPLES :]

        wvf = wvf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)
        permuted_channels = np.random.choice(
            np.arange(wvf.shape[0]), size=self.n_channels * 2, replace=False
        )

        new_wvf = wvf.copy()
        new_wvf[permuted_channels[: self.n_channels]] = wvf[
            permuted_channels[self.n_channels :]
        ]
        new_wvf[permuted_channels[self.n_channels :]] = wvf[
            permuted_channels[: self.n_channels]
        ]

        new_wvf = new_wvf.ravel()
        new_data_point = np.concatenate((new_wvf, acg)).reshape(1, -1).astype("float32")

        if istuple:
            transformed_sample = (new_data_point, label)
            return (
                (transformed_sample, spikes)
                if spikes is not None
                else transformed_sample
            )
        return (new_data_point, spikes) if spikes is not None else new_data_point


class CustomCompose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, spikes=None):
        for t in self.transforms:
            sample, spikes = t(sample, spikes)
        return sample, spikes

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def fixmatch_augment_pool():
    return [
        (
            AddSpikes,
            1,
            0.6,
        ),
        (DeleteSpikes, 1, 0.6),
        (ConstantShift, 1, 0.4),
        (GaussianNoise, 1, 2),
        (MoveSpikes, 1, 30),
        (DeleteChannels, 1, 5),
        (NewWindowACG, 1, 3),
        (
            PermuteChannels,
            1,
            5,
        ),
    ]


def waveform_augment_pool():
    return [
        (DeleteChannels, 1, 5),
        (
            PermuteChannels,
            1,
            5,
        ),
    ]


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, sample, spikes):
        # First we apply the soft transforms
        sample, spikes = SwapChannels(p=0.6)(sample, spikes)
        sample, spikes = VerticalReflection(p=0.6)(sample, spikes)

        # Then choose transforms to apply from the augment_pool
        ops = random.choices(self.augment_pool, k=self.n)
        for op, p, max_magn in ops:
            v = np.random.randint(1, self.m)
            v = v / 10  #! To correct for behaviour of our custom transforms
            if random.random() < 0.5:
                augmentation = op(p, max_magn * v)
                sample, spikes = augmentation(sample, spikes)
        data_point, label = sample
        return data_point, spikes


class RandTrans(object):
    def __init__(self, n, m, acg=True):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool() if acg else waveform_augment_pool()

    def __call__(self, sample, spikes):
        # First we apply the soft transforms
        sample, spikes = SwapChannels(p=0.6)(sample, spikes)
        sample, spikes = VerticalReflection(p=0.6)(sample, spikes)

        # Then choose transforms to apply from the augment_pool
        ops = random.choices(self.augment_pool, k=self.n)
        for op, p, max_magn in ops:
            v = np.random.randint(1, self.m)
            v = v / 10  #! To correct for behaviour of our custom transforms
            if random.random() < 0.5:
                augmentation = op(p, max_magn * v)
                sample, spikes = augmentation(sample, spikes)
        return sample, spikes
