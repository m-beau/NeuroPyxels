import random

import npyx
import numpy as np
try:
    import torch
except ImportError:
    pass

from npyx.corr import acg as make_acg
from npyx.datasets import resample_acg
from scipy.signal import fftconvolve


def fast_acg3d(
    spike_times,
    win_size,
    bin_size,
    fs=30000,
    num_firing_rate_bins=10,
    smooth=250,
    cut=None,
):
    if cut is not None:
        cut = min(cut, len(spike_times))
        spike_times = spike_times[:cut]

    assert fs > 0.0
    bin_size = np.clip(bin_size, 1000 * 1.0 / fs, 1e8)  # in milliseconds
    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    winsize_bins = 2 * int(0.5 * win_size * 1.0 / bin_size) + 1  # Both in millisecond
    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1
    time_axis = np.linspace(-win_size / 2, win_size / 2, num=winsize_bins)
    spike_counts = np.zeros(
        (num_firing_rate_bins, len(time_axis))
    )  # Counts number of occurences of spikes in a given bin in time axis
    times = np.zeros(num_firing_rate_bins, dtype=np.int64)  # total occurence

    # Samples per bin
    samples_per_bin = int(np.ceil(fs / (1000 / bin_size)))

    # Convert times_1 and times_2 (which are in units of fs to units of bin_size)
    spike_times = np.floor(spike_times / samples_per_bin).astype(np.int64)

    # Convert times_1 into a binary spike train
    max_indices = int(np.ceil(max(spike_times[-1], spike_times[-1]) + 1))
    spiketrain = np.zeros(max_indices, dtype=bool)
    spiketrain[spike_times] = True

    bin_size_seconds = bin_size / 1000.0
    intervals = np.searchsorted(spike_times, np.arange(max_indices))
    firing_rate = 1 / (
        (bin_size_seconds) * (spike_times[intervals] - spike_times[intervals - 1])
    )
    firing_rate = np.nan_to_num(firing_rate)

    # Smooth the firing rate with a convolution if requested
    if type(smooth) in [int, float] and smooth > 0:
        kernel_size = int(np.ceil(smooth / bin_size))
        half_kernel_size = kernel_size // 2
        kernel = np.ones(kernel_size) / kernel_size

        # Pad the input firing rate array
        padded_firing_rate = np.pad(firing_rate, pad_width=kernel_size, mode="edge")

        # Convolve the padded firing rate with the kernel, use cupy if available as this is an expensive operation
        smoothed_firing_rate = fftconvolve(padded_firing_rate, kernel, mode="valid")

        # Trim the edges of the smoothed firing rate array to remove the padded values
        trimmed_smoothed_firing_rate = smoothed_firing_rate[
            half_kernel_size:-half_kernel_size
        ]

        firing_rate = trimmed_smoothed_firing_rate

    # Get firing rate quantiles
    quantile_bins = np.linspace(0, 1, num_firing_rate_bins + 3)[1:-1]
    firing_rate_bins = np.quantile(firing_rate[spike_times], quantile_bins)

    # Find the bin number for each spike based on its firing rate
    current_firing_rate = firing_rate[spike_times]
    current_firing_rate_bin_number = np.searchsorted(
        firing_rate_bins, current_firing_rate
    )
    current_firing_rate_bin_number[current_firing_rate_bin_number == 0] = 1
    current_firing_rate_bin_number[
        current_firing_rate_bin_number == len(firing_rate_bins)
    ] = (len(firing_rate_bins) - 1)

    # Calculate spike counts for each firing rate bin
    bin_indices = np.arange(num_firing_rate_bins)
    spike_counts = np.zeros((num_firing_rate_bins, len(time_axis)))
    for bin_number in bin_indices:
        bin_spikes = spike_times[current_firing_rate_bin_number == bin_number + 1]
        start = bin_spikes + np.ceil(time_axis[0] / bin_size)
        stop = start + len(time_axis)
        mask = (
            (start >= 0)
            & (stop < len(spiketrain))
            & (bin_spikes >= spike_times[0])
            & (bin_spikes < spike_times[-1])
        )
        masked_start = start[mask].astype(int)
        masked_stop = stop[mask].astype(int)

        spike_counts[bin_number, :] = np.sum(
            [
                spiketrain[masked_start[i] : masked_stop[i]]
                for i in range(len(masked_start))
            ],
            axis=0,
        )

        times[bin_number] += np.sum(mask)

    acg_3d = spike_counts / (np.ones((len(time_axis), num_firing_rate_bins)) * times).T
    # Divison by zero cases will return nans, so we fix this
    acg_3d = np.nan_to_num(acg_3d)
    # remove bin 0, which will always be 1
    acg_3d[:, acg_3d.shape[1] // 2] = 0

    return firing_rate_bins, acg_3d


class SubselectPeriod(object):
    """
    Subselect spikes from only a certain fraction of the recording.
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, spikes, sample):
        if self.p <= np.random.rand():
            return spikes, sample

        recording_portions = np.array([int(spikes[-1] * 0.33), int(spikes[-1] * 0.66)])
        portion_to_use = np.random.randint(0, 3)

        original_spikes = spikes.copy()
        if portion_to_use == 0:
            spikes = spikes[spikes < recording_portions[0]]
        elif portion_to_use == 1:
            spikes = spikes[
                (spikes >= recording_portions[0]) & (spikes < recording_portions[1])
            ]
        else:
            spikes = spikes[spikes >= recording_portions[1]]

        return (original_spikes, sample) if len(spikes) == 0 else (spikes, None)


class DownsampleSpikes(object):
    """Downsamples the spikes that will constitue an ACG to a given number"""

    def __init__(self, n=10000, p=0.3):
        self.n = n
        self.p = p

    def __call__(self, spikes, sample):
        if self.p <= np.random.rand():
            # Return the original ACG if we do not downsample
            return spikes, sample

        if len(spikes) > self.n:
            spikes = spikes[: self.n]

        # Do not return an ACG if we downsample the spikes
        return spikes, None


class DeleteSpikes(object):
    """Deletes a random portion of the spikes that will constitute an ACG"""

    def __init__(
        self,
        p=0.3,
        deletion_prob=0.1,
    ):
        self.p = p
        self.deletion_prob = deletion_prob

    def __call__(self, spikes, sample):
        if self.p <= np.random.rand() or sample is not None:
            return spikes, sample

        mask = np.random.rand(spikes.shape[0]) > self.deletion_prob

        return spikes[mask], None


class RandomJitter(object):
    """Randomly moves the spikes in a spike train by a maximum amount"""

    def __init__(self, p=0.3, max_shift=10):
        self.p = p
        self.max_shift = int(np.ceil(max_shift))  # To work with RandAugment behavior

    def __call__(self, spikes, sample):
        if self.p <= np.random.rand() or sample is not None:
            return spikes, sample

        random_moving = np.random.randint(
            -self.max_shift, self.max_shift, size=spikes.shape[0]
        )
        return (spikes + random_moving).astype(int), None


class AddSpikes(object):
    """Adds a random amount of spikes (in percentage) to the spike list and recomputes the ACG"""

    def __init__(self, p=0.3, max_addition=0.1):
        self.p = p
        self.max_addition = max_addition

    def __call__(self, spikes, sample):
        if self.p <= np.random.rand() or sample is not None:
            return spikes, sample
        random_addition = np.random.randint(
            low=spikes[0],
            high=spikes[-1],
            size=int(spikes.shape[0] * self.max_addition),
        )

        return np.unique(np.concatenate((spikes, random_addition))), None


class Make3DACG(object):
    """Comptues the 3D acg after a set of transformations has been applied."""

    def __init__(self, bin_size, window_size, normalise=True, cut=False, log_acg=False):
        self.bin_size = bin_size
        self.window_size = window_size
        self.normalise = normalise
        self.cut = cut
        self.log_acg = log_acg

    def __call__(self, spikes, sample):
        # Handle the case where there was no transformation applied so the original ACG is returned
        if sample is not None:
            sample = sample.reshape(10, int(self.window_size / self.bin_size + 1))
            if self.normalise:
                sample = sample / sample.max(axis=(0, 1), keepdims=True)
            if self.cut:
                sample = sample[:, sample.shape[1] // 2 :]

            return None, np.nan_to_num(sample)

        # First correct the spiketrain after the transformations to enforce the rp
        duplicates_m = npyx.spk_t.duplicates_mask(spikes, enforced_rp=1)
        spikes = spikes[~duplicates_m]
        # Compute the ACG
        try:
            _, acg_3d = fast_acg3d(
                spikes,
                bin_size=self.bin_size,
                win_size=self.window_size,
            )
        except IndexError:
            try:
                _, acg_3d = npyx.corr.crosscorr_vs_firing_rate(
                    spikes,
                    spikes,
                    bin_size=self.bin_size,
                    win_size=self.window_size,
                )
            except IndexError:
                acg_3d = np.zeros((10, int(self.window_size / self.bin_size + 1)))

        # Normalise the ACG
        if self.normalise:
            acg_3d = acg_3d / acg_3d.max(axis=(0, 1), keepdims=True)
        acg_3d = np.nan_to_num(acg_3d)

        if self.log_acg:
            acg_3d, _ = npyx.corr.convert_acg_log(
                acg_3d, self.bin_size, self.window_size
            )

        if self.cut:
            acg_3d = acg_3d[:, acg_3d.shape[1] // 2 :]

        return None, acg_3d
