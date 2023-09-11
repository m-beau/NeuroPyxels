# -*- coding: utf-8 -*-
"""
2021-4-22
Authors: @fededagos, @agolajko

Functions needed to extract temporal and waveform features from Neuropixels recordings.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import ticker
from scipy import ndimage
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.stats import skew
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

from .corr import acg
from .datasets import NeuronsDataset, preprocess_template
from .gl import get_units
from .inout import chan_map
from .spk_t import trn_filtered
from .spk_wvf import wvf_dsmatch

# pylint: disable=unsupported-binary-operation

FEATURES = [
    "label",
    "dataset",
    "unit",
    "mfr",
    "mifr",
    "med_isi",
    "mode_isi",
    "prct5ISI",
    "entropy",
    "CV2_mean",
    "CV2_median",
    "CV",
    "IR",
    "Lv",
    "LvR",
    "LcV",
    "SI",
    "SKW",
    "acg_burst_vs_mfr",
    "acg_oscill_vs_mfr",
    "relevant_channel",
    "any_somatic",
    "max_peaks",
    "trough_voltage",
    "trough_t",
    "peak_voltage",
    "peak_t",
    "repolarisation_t",
    "depolarisation_t",
    "peak_50_width",
    "trough_50_width",
    "onset_t",
    "onset_amp",
    "wvf_width",
    "peak_trough_ratio",
    "tau_recovery",
    "multiplicative_a_recovery",
    "repolarisation_slope",
    "depolarisation_slope",
    "spatial_decay_24um",
    "dendritic_comp_amp",
]

WAVEFORM_AMPLITUDE_FEATURES = [
    "peak_voltage",
    "trough_voltage",
    "onset_amp",
    "multiplicative_a_recovery",
]

CORRELATED_FEATURES = [
    "mifr",
    "CV2_mean",
    "med_isi",
    "prct5ISI",
    "Lv",
    "IR",
    "mode_isi",
    "wvf_width",
]

CORRESPONDENCE = {
    5: "PkC_cs",
    4: "PkC_ss",
    3: "MFB",
    2: "MLI",
    1: "GoC",
    0: "GrC",
    -1: "unlabelled",
}

####################################################
# Temporal features
####################################################


def acg_burst_vs_mfr(train, mfr, sampling_rate=30_000):
    """
    It computes the autocorrelogram of the spike train, smooths it, and then computes the ratio of the
    maximum value of the autocorrelogram in the first 4 ms to the mean firing rate, and does the same outside
    the bursting period. Note that outside the bursting period we find the maximum oscillation with peak detection
    to avoid mistakenly labeling as a peak the decreasing part of the initial burst.

    Args:
      train: the spike train of the neuron
      mfr: mean firing rate

    Returns:
      The ratio of the maximum of the smoothened ACG in the first 4 ms and the mfr, and the same ratio but
      computed outside the bursting period.
    """
    # Compute the ACG and take only half to cut some processing
    autocorr = acg(None, 0, 0.2, 80, fs=sampling_rate, train=train)
    autocorr = autocorr[len(autocorr) // 2 :]

    # Smoothen the acg to avoid too crazy values for the feature even for bursty neurons
    smooth_acg = ndimage.gaussian_filter1d(autocorr, np.std(autocorr) / 10)

    # If the acg was empty this will return all nans, so we fix this behaviour
    smooth_acg = np.nan_to_num(smooth_acg, nan=0)

    # Define the burst period as 0-4 ms
    burst_delimiter = int(4 / 0.2)

    acg_peak_idxes, _ = find_peaks(
        smooth_acg, height=0.5 * np.std(smooth_acg), distance=len(smooth_acg) // 20
    )
    idxes_after_burst = acg_peak_idxes[acg_peak_idxes > burst_delimiter]

    max_burst = np.max(smooth_acg[:burst_delimiter])

    max_oscillation = (
        np.max(smooth_acg[idxes_after_burst]) if len(idxes_after_burst) > 0 else mfr
    )

    return (
        max_burst / mfr,
        max_oscillation / mfr,
    )


def compute_isi(train, quantile=0.02, fs=30_000, rp_seconds=1e-3):
    """
    Input: spike times in samples and returns ISI of spikes that pass through
        exclusion quantile
    Operations: if quantile is given the given quantile from the end of the ISI will be
                discarded
    Returns: isi in s"""

    isi_ = np.diff(train).astype(np.int64)

    if quantile is not None:
        mask = (isi_ >= (rp_seconds * fs)) & (isi_ <= np.quantile(isi_, 1 - quantile))
        isi_ = isi_[mask]
    return isi_ / fs


def entropy_log_isi(isint):
    """
    It takes a list of interspike intervals, bins them logarithmically, smooths the resulting histogram,
    and returns the Shannon entropy of the smoothed histogram

    Args:
      isint: the interspike intervals

    Returns:
      The entropy of the smoothed and binned ISI histogram.
    """

    # Note: the benefits of computing the entropy of the ISI histogram in log bins are detailed in
    # Dorval, 2011: https://doi.org/10.3390/e13020485

    log_bins = np.logspace(np.log10(isint.min()), np.log10(isint.max()), 200)
    num, _ = np.histogram(isint, log_bins)
    normalised_hist = num / np.sum(num)
    sigma = 2
    smooth_log_isi = ndimage.gaussian_filter1d(normalised_hist, sigma)

    # Remove 0 values (result of smoothing) to avoid taking the log of 0
    smooth_log_isi = smooth_log_isi[smooth_log_isi > 0]

    # Finally return the Shannon entropy of the smoothed and binned ISI histogram
    return (-smooth_log_isi * np.log2(smooth_log_isi)).sum()


def compute_isi_features(isint):
    """
    `compute_isi_features` takes a list of interspike intervals (ISIs) and returns a list of features
    that describe the distribution of ISIs.
    Most of these features are described in the supplementary materials of van Dijck et al., 2013:
    https://doi.org/10.1371/journal.pone.0057669

    Args:
      isint: interspike interval

    Returns:
      A list of features
    """

    # Instantaneous frequencies were calculated for each interspike interval as the reciprocal of the isi;
    # mean instantaneous frequency as the arithmetic mean of all values.
    mfr = 1 / np.mean(isint)
    mifr = np.mean(1.0 / isint)

    # Median inter-spike interval distribution
    med_isi = np.median(isint)

    # Mode of inter-spike interval distribution
    num, bins = np.histogram(isint, bins=np.linspace(isint.min(), isint.max(), 200))
    mode_isi = bins[np.argmax(num)]

    # Burstiness of firing: 5th percentile of inter-spike interval distribution
    prct5ISI = np.percentile(isint, 5)

    # Entropy of inter-spike interval distribution
    entropy = entropy_log_isi(isint)

    # Compute offset ISIs once for use in later calcuations
    sum_isi_offset = isint[1:] + isint[:-1]
    diff_isi_offset = isint[1:] - isint[:-1]
    prod_isi_offset = isint[1:] * isint[:-1]

    # Average coefficient of variation for a sequence of 2 ISIs
    # Relative difference of adjacent ISIs
    CV2_mean = np.mean(2 * np.abs(diff_isi_offset) / sum_isi_offset)
    CV2_median = np.median(
        2 * np.abs(diff_isi_offset) / sum_isi_offset
    )  # (Holt et al., 1996)

    # Coefficient of variation
    CV = np.std(isint) / np.mean(isint)

    # Instantaneous irregularity >> equivalent to the difference of the log ISIs
    inst_irregularity = np.mean(np.abs(np.log(isint[1:] / isint[:-1])))

    # NOTE: Hard-coded values from now on come from the supplementary materials of
    # van Dijck et al., 2013: https://doi.org/10.1371/journal.pone.0057669
    # In most cases, they ensure that the statistic is equal to 1 for a Poisson process.

    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    local_variation = 3 * np.mean(
        np.ones((len(isint) - 1)) - (4 * prod_isi_offset) / (sum_isi_offset**2)
    )

    # Revised Local Variation, with R the refractory period in the same unit as isint
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    R = 0.8  # ms
    revised_local_variation = 3 * np.mean(
        (np.ones((len(isint) - 1)) - (4 * prod_isi_offset) / (sum_isi_offset**2))
        * (np.ones((len(isint) - 1)) + (4 * R / sum_isi_offset))
    )

    # Coefficient of variation of the log ISIs
    log_CV = np.std(np.log10(isint)) * 1.0 / np.mean(np.log10(isint))

    # Geometric average of the rescaled cross correlation of ISIs
    SI_index = -np.mean(0.5 * np.log10((4 * prod_isi_offset) / (sum_isi_offset**2)))

    # Skewness of the inter-spikes intervals distribution
    skewness = skew(isint)

    return [
        mfr,
        mifr,
        med_isi,
        mode_isi,
        prct5ISI,
        entropy,
        CV2_mean,
        CV2_median,
        CV,
        inst_irregularity,
        local_variation,
        revised_local_variation,
        log_CV,
        SI_index,
        skewness,
    ]


#############################################
# Waveform features
#############################################


def cross_zero_t(waveform, t1, t2):
    """
    Find the first time that the waveform crosses zero between t1 and t2, and return the time and the
    value of the waveform at that time

    Args:
      waveform: the waveform to be analyzed
      t1: the start of the window to search for the zero-crossing
      t2: the end of the window to search for the zero-crossing


    Returns:
      The time of the first zero crossing and the value of the waveform at that time.
    """

    cross_relative_t = np.where(np.diff(np.sign(waveform[t1:t2])))[0]
    cross_t = cross_relative_t[0] + t1 if cross_relative_t.size > 0 else t2
    cross_t = cross_t.astype("int16")

    return cross_t, waveform[cross_t + 1]


def plot_debug_peaks(wvf, all_idxes, all_values, margin=0.8, onset=0.2):
    """
    It plots the waveform, the margin, the onset and offset, and the peaks.
    Used for debug purposes in conjunction with the "detect_peaks" function.

    Args:
      wvf: the waveform
      all_idxes: the indexes of the peaks
      all_values: the values of the peaks
      margin: how many standard deviations above/below the mean to consider a peak
      onset: the fraction of the waveform to ignore at the beginning and end.
    """
    plt.plot(wvf)
    plt.fill_between(
        np.arange(len(wvf)),
        np.zeros_like(wvf) + margin * np.std(wvf),
        np.zeros_like(wvf) - margin * np.std(wvf),
        alpha=0.4,
    )
    for i, _ in enumerate(all_idxes):
        plt.plot(all_idxes[i], all_values[i], "x", color="red")
    plt.axvspan(0, len(wvf) // (1 / onset), alpha=0.4)
    plt.axvspan(int(len(wvf) * (1 - onset)), len(wvf), alpha=0.4)
    plt.show()


def find_repolarisation(wvf, all_idxes, all_values):
    """
    If the minimum value is the last value, or the value after the minimum is not positive, then find the
    first peak after the minimum and add it to the list of peaks

    Args:
      wvf: the waveform
      all_idxes: the indexes of all the peaks
      all_values: the values of the peaks

    Returns:
      The new indices and values of all the peaks in the waveform, including the newly found repolarisation.
    """
    abs_min_rel_idx = np.argmin(all_values)
    if abs_min_rel_idx == len(all_values) - 1 or all_values[abs_min_rel_idx + 1] < 0:
        abs_min_idx = all_idxes[abs_min_rel_idx]
        peak_after_trough_idxes, _ = find_peaks(wvf[abs_min_idx:])
        if len(peak_after_trough_idxes) > 0:
            peak_after_trough_idx = peak_after_trough_idxes[0] + abs_min_idx
            peak_after_trough_value = wvf[peak_after_trough_idx]
            all_idxes = np.append(all_idxes, peak_after_trough_idx)
            all_values = np.append(all_values, peak_after_trough_value)

            # Sort peaks by time again
            sorting_idx = np.argsort(all_idxes)
            all_idxes = all_idxes[sorting_idx]
            all_values = all_values[sorting_idx]

    return all_idxes, all_values


def detect_peaks(wvf, margin=0.8, onset=0.2, plot_debug=False):
    """Custom peak detection algorithm. Based on scipy.signal.find_peaks.
    Args:
        wvf (np.ndarray): Waveform to detect peaks in.
        margin (float): Margin around baseline around which peaks are ignored, in std units.
        onset (float): Percentage of waveform to ignore before first and last peak.

    Returns:
        all_idxes (np.ndarray): Indices of all peaks.
        all_values (np.ndarray): Values of all peaks.
    """

    # Determine which peaks to ignore using provided margin
    peak_height = margin * np.std(wvf)

    # Find peaks using scipy. We also apply a distance criterion to avoid finding too close peaks in artefactual waveforms.
    max_idxes, _ = find_peaks(wvf, height=peak_height, distance=len(wvf) // 20)
    min_idxes, _ = find_peaks(-wvf, height=peak_height, distance=len(wvf) // 20)
    max_values = wvf[max_idxes]
    min_values = wvf[min_idxes]

    # Concatenate negative and positive peaks
    all_idxes = np.concatenate((max_idxes, min_idxes))
    all_values = np.concatenate((max_values, min_values))

    # Sort peaks by time
    sorting_idx = np.argsort(all_idxes)
    all_idxes = all_idxes[sorting_idx]
    all_values = all_values[sorting_idx]

    # Exclude peaks found before the provided onset time
    mask = (all_idxes < len(wvf) // (1 / onset)) | (
        all_idxes > int(len(wvf) * (1 - onset))
    )
    all_idxes = all_idxes[~mask]
    all_values = all_values[~mask]

    if len(all_values) != 0:
        # Ensure that we have a positive peak coming after the most negative one, otherwise find it
        all_idxes, all_values = find_repolarisation(wvf, all_idxes, all_values)

    if plot_debug:
        plot_debug_peaks(wvf, all_idxes, all_values, margin=margin, onset=onset)

    return all_idxes, all_values


def wvf_width(waveform, peak_time, trough_time):
    """
    The function `wvf_width` takes a waveform, a peak time, and a trough time, and returns the width
    of the waveform, the trough time, the peak time, and the trough value

    Args:
      waveform: the waveform
      peak_time: the time of the peak of the waveform
      trough_time: the time of the trough

    Returns:
      The width of the waveform.
    """

    # use the last returned value for plotting only

    return np.abs(peak_time - trough_time)


def pt_ratio(waveform, peak_time, trough_time):
    """
    Calculates the absolute value of the peak to trough ratio.

    Args:
      waveform: the waveform
      peak_time: the time of the peak of the wave
      trough_time: the time of the trough

    Returns:
      The absolute value of the ratio of the peak time to the trough time.
    """

    return np.abs(waveform[peak_time] / waveform[trough_time])


def trough_onset_t(waveform, trough_time):
    """
    It finds the last time before the trough that the waveform crossed a threshold of 5% of the trough
    value

    Args:
      waves: the array of voltage values
      trough_time: the time of the trough

    Returns:
      The last crossing of the onset voltage and the onset voltage.
    """

    onset_v = waveform[trough_time] * 0.05

    # now we find the last crossing of this value before the peak
    before_peak = waveform[:trough_time]
    crossings = np.where(np.diff(np.sign(before_peak - onset_v)))

    # if there are no crossings, this means the recording was always above
    # or below the value, hence the crossing can be said to have happened
    # before the start of the recording, which we approximate by 0

    last_cross = 0 if crossings[0].size == 0 else crossings[-1][0]
    return last_cross, onset_v


def peak_offset_t(waveform, peak_time):
    """
    It finds the last time after the peak when the waveform relaxed to 5% of the peak value.

    Args:
      waveform: the array of voltage values
      peak_time: the time of the last peak

    Returns:
      The last crossing of the offset voltage and the offset voltage.
    """
    # get the wvf peaks
    offset_v = waveform[peak_time] * 0.05

    # now we find the last crossing of this value
    # get section after the last peak
    after_peak = waveform[peak_time:]
    # now we have list of crossings where
    crossings = np.where(np.diff(np.sign(after_peak - offset_v)))

    # if there are crossings, get the first of these
    # if there are no crossings, return the shape of the wvf as we know
    # the last crossing happened after our time window
    if len(crossings[0]):
        last_cross = peak_time + crossings[0][0]
    else:
        last_cross = waveform.shape[0]

    return last_cross, offset_v


def repol_10_90_t(waveform, peak_time, trough_time):
    """
    Find the 10th and 90th percentile values of the upslope of the waveform, then find the closest
    points to those values on the waveform. It is an approximation of the repolarisation time.

    Args:
      waveform: the waveform you want to measure
      peak_time: the time of the peak of the waveform
      trough_time: the time of the trough

    Returns:
      the time of the 10% and 90% repolarization points, the value of the waveform at those points, and
    the difference between the two times.
    """

    # get the section of the slope we need
    upslope = waveform[trough_time:peak_time]

    # Here they are inverted in the numpy call because we are dealing with negative values!
    perc_10 = np.percentile(upslope, 90)
    perc_90 = np.percentile(upslope, 10)

    # now need to find where the before and after cross two points happened
    # get the points where the wave reaches the percentile values
    cross_10 = trough_time + np.where(np.diff(np.sign(upslope - perc_10)))[0][0]
    cross_90 = trough_time + np.where(np.diff(np.sign(upslope - perc_90)))[0][0]

    crosses_10 = np.array([cross_10, cross_10 + 1])
    # ensure the value is positive for the first crossing
    # crosses_10 = crosses_10[waves[crosses_10] > 0]
    crosses_90 = np.array([cross_90, cross_90 + 1])
    # from these crossing points find the closer one to the value
    close_10 = crosses_10[np.argmin(np.abs(waveform[crosses_10]) - perc_10)]
    close_90 = crosses_90[np.argmin(np.abs(waveform[crosses_90]) - perc_90)]

    return (
        np.array([close_10, close_90]),
        np.array([waveform[close_10], waveform[close_90]]),
        close_90 - close_10,
    )


def depol_10_90_t(waveform, peak_time, trough_time):
    """
    Find the points where the waveform crosses 10% and 90% of the trough value, and return the time
    difference between them. This is an approximation of depolarisation time.

    Args:
      waveform: the waveform of the spike
      peak_time: the time of the peak of the waveform
      trough_time: the time of the trough

    Returns:
      the time of the 10% and 90% depolarization, the value of the waveform at those times, and the
    difference between the two times.
    """

    trough_value = waveform[trough_time]
    prev_peak_t = trough_onset_t(waveform, trough_time)[0]

    perc_10 = 0.1 * trough_value
    perc_90 = 0.9 * trough_value

    # now need to find where the before and after cross two points happened
    # get the section of the slope we need
    downslope = waveform[prev_peak_t : trough_time + 1]
    # get the points where the wave reaches the percentile values
    cross_10 = prev_peak_t + np.where(np.diff(np.sign(downslope - perc_10)))[0][0]
    cross_90 = prev_peak_t + np.where(np.diff(np.sign(downslope - perc_90)))[0][0]

    crosses_10 = np.array([cross_10, cross_10 + 1])
    crosses_90 = np.array([cross_90, cross_90 + 1])
    # from these crossing points find the closer one to the value
    close_10 = crosses_10[np.argmin(np.abs(waveform[crosses_10]) - perc_10)]
    close_90 = crosses_90[np.argmin(np.abs(waveform[crosses_90]) - perc_90)]

    return (
        np.array([close_10, close_90]),
        np.array([waveform[close_10], waveform[close_90]]),
        close_90 - close_10,
    )


def depol_slope(waveform, trough_time):
    """
    It fits a line to the downslope of the trough (from the half width),
    and returns the fit coefficients and the fitted line.

    Args:
      waveform: The waveform we want to fit
      trough_time: The time of the trough

    Returns:
      The fit coefficients of the depolarisation slope, and the fitted line
    """

    # Normalise the waveform to correct for amplitude effects!
    scaling = np.max(np.abs(waveform))
    waveform = waveform / scaling

    # Find trough onset and 50% of trough value
    trough_value = waveform[trough_time]
    trough_onset = trough_onset_t(waveform, trough_time)[0]
    perc_50 = 0.5 * trough_value

    # Define the section of the waveform we want to fit
    downslope = waveform[trough_onset : trough_time + 1]
    # Get the points where the wave reaches 50% of the trough
    cross_50 = trough_onset + np.where(np.diff(np.sign(downslope - perc_50)))[0][0]

    x_fit = np.arange(cross_50, trough_time + 1)
    y_fit = waveform[cross_50 : trough_time + 1]

    coeff = np.polyfit(x_fit, y_fit, deg=1)

    # Return the fit coefficients and the fitted line
    return (
        coeff,
        (coeff[0] * x_fit + coeff[1]) * scaling,
    )


def pos_half_width(waveform, peak_time, trough_time):
    """
    Find the repolarisation half width (in time) of the waveform.

    Args:
      waveform: the waveform
      peak_time: the time of the peak of the action potential
      trough_time: the time of the trough

    Returns:
      The start and end of the half width, the 50% value and the half width time
    """

    peak_value = waveform[peak_time]
    perc_50 = 0.5 * peak_value

    # look for crossings from cross_time to end
    start_interval = cross_zero_t(waveform, trough_time, peak_time)[0].astype(np.int64)

    # If the cross time found is the peak value, it means that the peak is negative! (i.e. the repolarisation is not very prominent)
    if start_interval == peak_time:
        return peak_time, peak_time, 0, 0

    end_interval = waveform.shape[0]

    current_slope = waveform[start_interval:end_interval]
    # get the real time when the crossings happened, not just relative time
    cross_start = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][0]
    )
    cross_end = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][-1]
    )
    return cross_start, cross_end, perc_50, cross_end - cross_start


def neg_half_width(waveform, peak_time, trough_time):
    """
    Find the depolarisation half width (in time) of the waveform.

    Args:
      waveform: the waveform
      peak_time: the time of the peak of the action potential
      trough_time: the time of the trough

    Returns:
      The start and end of the half width, the 50% value and the half width time
    """

    trough_value = waveform[trough_time]
    perc_50 = 0.5 * trough_value

    # get the half width for the first peak
    # look for crossings from 0 to cross_time

    start_interval = 0
    end_interval = cross_zero_t(waveform, trough_time, peak_time)[0].astype(np.int64)

    current_slope = waveform[start_interval:end_interval]
    # get the real time when the crossings happened, not just relative time
    cross_start = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][0]
    )
    cross_end = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][-1]
    )
    return cross_start, cross_end, perc_50, cross_end - cross_start


def tau_end_slope(waveform, peak_time, trough_time):
    """
    It fits an exponential to the end of the waveform, and returns the fit, the mean squared error, and
    the time constant

    Args:
      waves: the waveform
      peak_time: the time of the peak of the action potential
      trough_time: The time of the trough of the action potential

    Returns:
      The fit, the mean squared error, and the time constant of the fit.
    """

    # Normalise the waveform to correct for amplitude effects!
    scaling = np.max(np.abs(waveform))
    waveform = waveform / scaling

    y = waveform[peak_time : peak_time + 1000]
    x = np.arange(0, len(y))

    # # We are fitting an exponential of the type y = a * exp(-b * x)
    # # By taking the log we can perform a linear fit: ln(y) = ln(a) + b * x
    # # Where b = 1/tau

    def exp_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit through scipy
    try:
        params, _ = sp.optimize.curve_fit(
            exp_func, x, y, check_finite=True, maxfev=10000
        )
    except RuntimeError:
        return np.zeros_like(x), np.inf, np.nan, np.nan

    a, b, c = params

    fit = exp_func(x, a, b, c)

    mse = mean_squared_error(y, fit)

    return fit * scaling, mse, 1 / b, a * scaling


def interp_wave(waveform, multi=100, axis=-1):
    """
    It takes a waveform and interpolates it by a factor of `multi` along the `axis` dimension

    Args:
      waveform: the waveform to be interpolated
      multi: the number of times to interpolate the waveform. Defaults to 100
      axis: The axis along which to interpolate.

    Returns:
      The interpolated waveform.
    """

    wvf_len = waveform.shape[1] if waveform.ndim == 2 else waveform.shape[0]
    interp_fn = interp1d(np.linspace(0, wvf_len - 1, wvf_len), waveform, axis=axis)

    return interp_fn(np.linspace(0, wvf_len - 1, wvf_len * multi))


def repol_slope(waveform, peak_time, trough_time):
    """
    It fits a line to the upslope from the trough (up to the half width),
    and returns the fit coefficients and the fitted line.

    Args:
      waveform: The waveform we want to fit
      trough_time: The time of the trough

    Returns:
      The fit coefficients of the depolarisation slope, and the fitted line
    """

    # Normalise the waveform to correct for amplitude effects!
    scaling = np.max(np.abs(waveform))
    waveform = waveform / scaling

    # Get the section of the slope we need
    upslope = waveform[trough_time:peak_time]

    trough_value = waveform[trough_time]

    perc_50 = 0.5 * trough_value

    cross_50 = trough_time + np.where(np.diff(np.sign(upslope - perc_50)))[0][0]

    x_fit = np.arange(trough_time, cross_50 + 1)
    y_fit = waveform[trough_time : cross_50 + 1]

    coeff = np.polyfit(x_fit, y_fit, deg=1)

    # Return the fit coefficients and the fitted line
    return (
        coeff,
        (coeff[0] * x_fit + coeff[1]) * scaling,
    )


def recover_chanmap(partial_chanmap: np.ndarray) -> np.ndarray:
    """
    Given a 1D array containing an incomplete channelmap (of only x coordinates),
    fill in the information by adding the y coordinates.
    Works only for Neuropixels 1.0 exp_funcprobes.

    Args:
      partial_chanmap (np.ndarray): the incomplete chanmap that you have.

    Returns:
      a chanmap with the y coordinates of the channels.
    """
    assert (
        partial_chanmap.shape[1] == 1
    ), "Are you sure you provided an incomplete chanmap?"
    probe_type = None

    if set(partial_chanmap.astype(int).ravel().tolist()) == {
        11,
        27,
        43,
        59,
    }:
        probe_type = "1.0"
    if set(partial_chanmap.astype(int).ravel().tolist()) == {0, 32}:
        probe_type = "2.0"

    assert probe_type in [
        "1.0",
        "2.0",
    ], "Chanmap provided is not supported. Only Neuropixels 1.0 and 2.0 probes are supported."

    if probe_type == "1.0":
        y_chanmap = [x * 10 for x in range(len(partial_chanmap) + 1) if x % 2 == 0]
    else:
        y_chanmap = [x * 7.5 for x in range(len(partial_chanmap) + 1) if x % 2 == 0]
    y_chanmap = np.array(list(zip(y_chanmap, y_chanmap))).ravel()
    # Behviour changes if we are starting with a pair or with an odd channel
    if int(partial_chanmap.ravel()[0]) in {59, 43, 32}:
        y_chanmap = np.array(y_chanmap[1 : len(partial_chanmap) + 1])
    else:
        y_chanmap = np.array(y_chanmap[: len(partial_chanmap)])

    return np.concatenate((partial_chanmap, y_chanmap.reshape(-1, 1)), axis=1)


def dendritic_component(waveform_2d, peak_chan, somatic_mask):
    dendritic_waves = waveform_2d[~somatic_mask]
    max_dendritic_amp = np.ptp(dendritic_waves, axis=1)

    if (max_dendritic_amp < 30).all():
        return 0

    return np.max(max_dendritic_amp) / np.ptp(waveform_2d[peak_chan])


def chan_spread(all_wav, peak_chan, chanmap):
    """
    It takes in the waveforms, the peak channel, and the channel map and returns the ratio of the mean
    amplitude of the closest channels to the peak channel.
    Note: Assumes that the provided channelmap comes from Neuropixels 1.0 probes.

    Args:
      all_wav: the waveforms for all channels
      peak_chan: the channel with the largest amplitude
      chanmap: a numpy array of shape (n_channels, 2) where the first column is the x-coordinate and the
    second column is the y-coordinate of each channel.

    Returns:
      The ratio of the mean amplitude of the closest channels to the peak channel
    """

    assert (
        chanmap.shape[0] == all_wav.shape[0]
    ), f"Waveforms and chanmap shapes do not match {all_wav.shape} vs {chanmap.shape}"

    # First find the Euclidean distance of all channels from peak chan (L2 norm)
    distances = np.linalg.norm(chanmap - chanmap[peak_chan], axis=1)

    # Find the clossest channels to the peak chan
    # TODO Extend compatible probe types.
    closest = (distances < 26) & (
        distances != 0
    )  # 25.5 is the exact distance between peak and closest channel in 1.0 probes

    # Get the amplitude on all channels
    all_max_amp = np.ptp(all_wav, axis=1)

    # Get mean of closest values and return ratio with respect to peak channel
    mean_closest = all_max_amp[closest].mean()

    return mean_closest / np.ptp(all_wav[peak_chan])


def healthy_waveform(waveform_1d, peaks_values):
    """
    Determine if waveform looks healthy.
    Works for somatic and dendritic waveforms, and also fat spikes.

    - at least 1 waveform amplitude > 30uV
    - count peaks from peak detect - should be between 2 and 5.
    """

    if len(peaks_values) < 2 or len(peaks_values) > 5:
        return False

    return np.ptp(waveform_1d) >= 30


def is_somatic(peaks_v, wvf_std):
    """
    Assures that the waveform is somatic and can be used in further processing.
    For this we need to have at least a through followed by a peak in the waveform we found.
    Alternative, the repolarisation (if negative) should be within the std of the waveform.
    """

    peak_signs = np.sign(peaks_v).astype(np.int32).tolist()
    peak_signs = str(peak_signs)
    pattern = "-1, 1"
    negative_extremum = (pattern in peak_signs and len(peaks_v) == 2) or (
        np.abs(np.max(peaks_v)) < np.abs(np.min(peaks_v))
    )
    repolarising = pattern in peak_signs or (np.abs(peaks_v[-1]) < wvf_std)

    return repolarising and negative_extremum


def detect_peaks_2d(waveform2d, channels):
    """
    For each channel, we detect peaks in the waveform, check if it has an healthy shape, and
    finally if it is somatic or not. Returns the candidate somatic and non-somatic helthy channels,
    along with a boolean mask for the somatic channels and an int for the max number of peaks detected (used later as a feature.)

    Args:
      waveform2d: 2D array of waveforms, shape (n_channels, n_samples)
      channels: list of channel numbers

    Returns:
      candidate_channel_somatic: list of somatic channels
      candidate_channel_non_somatic: list of non-somatic channels
      somatic_mask: boolean mask for somatic channels
      max_peaks: max number of peaks detected
    """
    candidate_channel_somatic = []
    candidate_channel_non_somatic = []
    somatic_mask = np.zeros(waveform2d.shape[0], dtype=bool)
    max_peaks = []

    considered_waveforms = waveform2d[channels]
    for i, (channel, wf) in enumerate(zip(channels, considered_waveforms)):
        _, peak_v = detect_peaks(wf, margin=0.8)
        if healthy_waveform(wf, peak_v):
            max_peaks.append(len(peak_v))
            if is_somatic(peak_v, np.std(wf)):
                somatic_mask[i] = True
                candidate_channel_somatic.append(channel)
            else:
                candidate_channel_non_somatic.append(channel)
        else:
            continue

    return (
        candidate_channel_somatic,
        candidate_channel_non_somatic,
        somatic_mask,
        max(max_peaks, default=0),
    )


def filter_out_waves(waveform_2d, peak_channel, max_chan_lookaway=21):
    """
    Filter out waveforms that are not within a certain range of the peak channel and have a
    peak-to-peak amplitude less than 30 microV counts

    Args:
      waveform_2d: the waveform data, as a 2D array of shape (n_channels, n_samples)
      peak_channel: the channel with the highest amplitude
      max_chan_lookaway: The maximum number of channels away from the peak channel to consider. Defaults
    to 16

    Returns:
      The channels that have a peak-to-peak amplitude greater than 30.
    """
    wf_channels = np.arange(len(waveform_2d))

    # Determine which channels to consider
    chan_slice = slice(
        max(0, peak_channel - max_chan_lookaway),
        min(peak_channel + max_chan_lookaway + 1, waveform_2d.shape[0]),
    )
    considered_waveforms = waveform_2d[chan_slice]
    considered_channels = wf_channels[chan_slice]

    return considered_channels[np.ptp(considered_waveforms, axis=1) > 30]


def find_relevant_waveform(
    waveform_2d, candidate_channel_somatic, candidate_channel_non_somatic, somatic_mask
):
    """
    If there are any somatic waveforms, return the best one. Otherwise, if there are any dendritic
    waveforms, return the best one flipped. Otherwise, return None.

    Args:
      waveform_2d: a 2D array of shape (n_channels, n_samples)
      candidate_channel_somatic: A list of channels that have a somatic waveform
      candidate_channel_non_somatic: list of channels that have good, healthy waveforms but are not somatic
      somatic_mask: A boolean mask of the channels that are somatic.

    Returns:
      The relevant waveform, whether it is somatic or not, and the channel it was found on.
    """
    if somatic_mask.any() == True:
        candidate_waveforms = waveform_2d[candidate_channel_somatic]
        relevant_channel = candidate_channel_somatic[
            np.argmax(np.ptp(candidate_waveforms, axis=1))
        ]
        return waveform_2d[relevant_channel], True, relevant_channel

    if somatic_mask.any() == False and candidate_channel_non_somatic:
        # All good waveforms found are dendritic, so we return the best one flipped
        candidate_waveforms = waveform_2d[candidate_channel_non_somatic]
        relevant_channel = candidate_channel_non_somatic[
            np.argmax(np.ptp(candidate_waveforms, axis=1))
        ]
        return -waveform_2d[relevant_channel], False, relevant_channel

    # No relevant waveform found if we reach this point
    return None, None, None


def find_relevant_peaks(peak_t, peak_v, wvf_std):
    """
    Given two arrays of peak times and peak values from detect_peaks,
    finds the first trough in the array which is followed by a peak to use in the extraction
    of all the next features. Returns the time of the relevant peak and trough.

    Args:
        - peak_t: array of peak times
        - peak_v: array of peak values

    Returns:
        - relevant_trough_t: time of relevant trough
        - relevant_peak_t: time of the relevant peak
    """

    assert len(peak_t) >= 2, "Number of peaks must be at least 2!"

    if len(peak_t) == 2:
        return peak_t[0], peak_t[1]

    signs = np.sign(peak_v).astype(np.int32)
    trough_mask = signs == -1
    first_trough_idx = np.where(trough_mask)[0][0]

    # Ensure that a peak follows the trough we found
    valid = False
    while not valid and first_trough_idx < len(peak_t) - 1:
        if (
            np.sign(peak_v[first_trough_idx]) == -1
            and (np.sign(peak_v[first_trough_idx + 1]) == 1)
            or (np.abs(peak_v[first_trough_idx + 1]) < wvf_std)
        ):
            first_trough_t = peak_t[first_trough_idx]
            first_peak_t = peak_t[first_trough_idx + 1]
            valid = True
        else:
            first_trough_idx += 1

    return first_trough_t, first_peak_t


def extract_single_channel_features(
    relevant_waveform, plot_debug=False, interp_coeff=100
):
    """
    It takes a waveform and returns a list of features that describe the waveform

    Args:
        relevant_waveform: the waveform to be analyzed

    Returns:
        The return is a list of the features that are being extracted from the waveform.
    """

    # First interpolates the waveform for higher precision
    relevant_waveform = interp_wave(relevant_waveform, interp_coeff)

    peak_times, peak_values = detect_peaks(relevant_waveform, plot_debug=plot_debug)

    if len(peak_times) < 2 or np.all(peak_values < 0):
        return [0] * 15

    first_trough_t, first_peak_t = find_relevant_peaks(
        peak_times, peak_values, 0.8 * np.std(relevant_waveform)
    )

    neg_v = relevant_waveform[first_trough_t]
    neg_t = first_trough_t

    pos_v = relevant_waveform[first_peak_t]
    pos_t = first_peak_t

    # pos 10-90
    _, _, pos_10_90_t = repol_10_90_t(relevant_waveform, first_peak_t, first_trough_t)

    # neg 10-90
    _, _, neg_10_90_t = depol_10_90_t(relevant_waveform, first_peak_t, first_trough_t)

    # pos half width
    _, _, _, pos50 = pos_half_width(relevant_waveform, first_peak_t, first_trough_t)

    # neg half width
    _, _, _, neg50 = neg_half_width(relevant_waveform, first_peak_t, first_trough_t)

    # Trough onset time and amplitude
    onset_t, onset_amp = trough_onset_t(relevant_waveform, first_trough_t)

    # wvf duration
    wvfd = wvf_width(relevant_waveform, first_peak_t, first_trough_t)

    # peak-to-trough ratio
    ptr = pt_ratio(relevant_waveform, first_peak_t, first_trough_t)

    # recovery time constant
    try:
        _, _, tau, a_coeff = tau_end_slope(
            relevant_waveform, first_peak_t, first_trough_t
        )
    except np.linalg.LinAlgError:
        tau = 0

    # repolarisation slope coefficients
    repol_coeff, _ = repol_slope(relevant_waveform, first_peak_t, first_trough_t)

    # depolarisation slope coefficients
    depol_coeff, _ = depol_slope(relevant_waveform, first_trough_t)

    # Multiply slope coefficients by 100 (and divide tau) to undo interpolation effect and obtain meaningful values
    tau, repol_coeff, depol_coeff = tau / 100, repol_coeff * 100, depol_coeff * 100

    return [
        neg_v,
        neg_t,
        pos_v,
        pos_t,
        pos_10_90_t,
        neg_10_90_t,
        pos50,
        neg50,
        onset_t,
        onset_amp,
        wvfd,
        ptr,
        tau,
        a_coeff,
        repol_coeff[0],
        depol_coeff[0],
    ]


def extract_spatial_features(
    waveform_2d, peak_chan, relevant_channel, somatic_mask, chanmap
):
    """
    - peak_chan: channel from which the 1D waveworm features will be extracted
    """

    # ratio between max amplitude among 4/5 closest channels at 23.61 um and max amplitude on peak channel
    chanmap = np.array(chanmap)
    spatial_spread_ratio = chan_spread(waveform_2d, peak_chan, chanmap)
    # set to 1 if relevant waveform is dendritic already (because normalized)
    max_dendritic_voltage = (
        dendritic_component(waveform_2d, relevant_channel, somatic_mask)
        if somatic_mask.any()
        else 1
    )

    return spatial_spread_ratio, max_dendritic_voltage


def waveform_features(
    waveform_2d: np.ndarray,
    peak_channel: int,
    chanmap: np.ndarray,
    interp_coeff: int = 100,
    plot_debug: bool = False,
    waveform_type="relevant",
    _clip_size=(1e-3, 2e-3),
    fs=30000,
) -> list:
    """
    > Given a 2D array of waveforms, the channel with the peak, and a boolean flag for whether to plot
    debug figures, return a list of features.

    Args:
      waveform_2d (np.ndarray): The waveform used to compute features, of shape (n_channels, n_samples)
      peak_channel (int): the channel that has the largest amplitude in the waveform
      chanmap (np.ndarray): the channel map of the probe
      plot_debug (bool): If True, plots the waveform and the features extracted from it. Defaults to
    False
    """

    if waveform_type not in ["relevant", "flipped", "peak"]:
        raise NotImplementedError(
            f"Waveform type {waveform_type} not implemented. Please choose one of the following: 'relevant', 'flipped', 'peak'"
        )

    high_amp_channels = filter_out_waves(waveform_2d, peak_channel)

    (
        candidate_channel_somatic,
        candidate_channel_non_somatic,
        somatic_mask,
        max_peaks,
    ) = detect_peaks_2d(waveform_2d, high_amp_channels)

    peak_waveform = waveform_2d[peak_channel, :]
    if waveform_type == "relevant":
        # First find working waveform
        # If None is found features cannot be extracted
        relevant_waveform, somatic, relevant_channel = find_relevant_waveform(
            waveform_2d,
            candidate_channel_somatic,
            candidate_channel_non_somatic,
            somatic_mask,
        )
        if relevant_waveform is None:
            return [peak_channel, *[0] * 18]
    elif waveform_type == "flipped":
        relevant_waveform = preprocess_template(
            peak_waveform,
            clip_size=_clip_size,
            normalize=False,
            original_sampling_rate=fs,
        )
        relevant_channel = peak_channel
        somatic = False
    elif waveform_type == "peak":
        relevant_waveform = peak_waveform
        relevant_channel = peak_channel
        somatic = False

    peak_channel_features = extract_single_channel_features(
        relevant_waveform, plot_debug, interp_coeff
    )
    spatial_features = (
        extract_spatial_features(
            waveform_2d, peak_channel, relevant_channel, somatic_mask, chanmap
        )
        if chanmap is not None
        else [0, 0]
    )
    return [
        int(relevant_channel),
        int(somatic),
        max_peaks,
        *peak_channel_features,
        *spatial_features,
    ]


def waveform_features_json(dp: str, unit: int, plot_debug: bool = False) -> list:
    """
    Given a path to a recording and a unit number, return a list of features for that unit.
    Wrapper function for waveform_features.

    Args:
      dp (str): path to the .dat file
      unit (int): int
      plot_debug (bool): if True, will plot the waveform and the peak channel. Defaults to False
    """

    assert os.path.exists(dp), f"Provided path {dp} does not exist"

    _, waveform_2d, _, peak_channel = wvf_dsmatch(
        dp,
        unit,
        verbose=False,
        again=True,
        save=True,
    )

    chanmap = chan_map(probe_version="1.0")

    wvf_feats = waveform_features(waveform_2d.T, peak_channel, chanmap, plot_debug)

    return [dp, unit] + wvf_feats


def plot_all_features(waveform, normalise=True, dp=None, unit=None, label=None):
    wvf = interp_wave(waveform)
    if normalise:
        wvf = wvf / np.max(np.abs(wvf))

    peak_t, peak_v = detect_peaks(wvf)

    plt.ion()
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.plot(wvf, linewidth=2, alpha=0.8)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    try:
        trough_time, peak_time = find_relevant_peaks(peak_t, peak_v, 0.8 * np.std(wvf))

        onset = trough_onset_t(wvf, trough_time)
        offset = peak_offset_t(wvf, peak_time)
        zero_time = cross_zero_t(wvf, trough_time, peak_time)
        pos_hw = pos_half_width(wvf, peak_time, trough_time)
        neg_hw = neg_half_width(wvf, peak_time, trough_time)
        end_slope, _, tau, _ = tau_end_slope(wvf, peak_time, trough_time)
        rep_slope, rep_fit = repol_slope(wvf, peak_time, trough_time)
        dep_slope, dep_fit = depol_slope(wvf, trough_time)
        wvf_dur = wvf_width(wvf, peak_time, trough_time)
        ptrat = pt_ratio(wvf, peak_time, trough_time)

        # get the negative and positive peaks for the PtR
        neg_t = trough_time
        neg_v = wvf[trough_time]
        pos_v = wvf[peak_time]
        pos_t = peak_time

        ax.plot(peak_t, peak_v, "*", color="red", markersize=10)
        ax.plot(onset[0], onset[1], "rx")
        ax.plot(offset[0], offset[1], "rx")

        ax.plot(zero_time[0], 0, "rx")
        ax.plot([neg_t, neg_t], [0, neg_v], linewidth=1, c="black", linestyle="dotted")
        ax.plot([pos_t, pos_t], [0, pos_v], linewidth=1, c="black", linestyle="dotted")
        ax.plot([neg_t, pos_t], [0, 0], linewidth=1, c="black", linestyle="dotted")

        ax.plot(
            np.arange(peak_time, peak_time + len(end_slope)),
            end_slope,
            linewidth=3,
            linestyle="dotted",
            color="red",
        )
        ax.plot(
            neg_t + np.arange(0, rep_fit.shape[0]),
            rep_fit,
            linewidth=3,
            linestyle="dotted",
            color="red",
        )
        ax.plot(
            (neg_t + np.arange(0, dep_fit.shape[0])) - dep_fit.shape[0],
            dep_fit,
            linewidth=3,
            linestyle="dotted",
            color="red",
        )

        ax.plot([pos_hw[0], pos_hw[1]], [pos_hw[2], pos_hw[2]], color="purple")
        ax.plot([neg_hw[0], neg_hw[1]], [neg_hw[2], neg_hw[2]], color="purple")

        # Plot and add label to the waveform duration
        ax.plot(
            np.linspace(neg_t, pos_t, wvf_dur + 1),
            np.linspace(neg_v + 0.1 * ylim[0], neg_v + 0.1 * ylim[0], wvf_dur + 1),
        )
        ax.text(
            pos_t + 0.01 * xlim[1],
            neg_v + 0.1 * ylim[0],
            f"wvf duration: {np.round(wvf_dur/3000, 2)}",
        )

        # NOTE: we are multiplying the slope values by 100 to undo the interpolation effect and obtain meaningful values!
        # For the same reason we divide tau

        ax.text(
            end_slope[-1] + 0.1 * xlim[1],
            end_slope[-1],
            f"tau recovery: {np.round(tau/100,2)}",
        )
        ax.text(
            neg_t + 0.1 * xlim[1],
            neg_v - 0.1 * ylim[0],
            f"rep slope: {np.round(rep_slope[0]* 100,2)}",
        )
        ax.text(
            neg_t - 0.2 * xlim[1],
            neg_v - 0.1 * ylim[0],
            f"dep slope: {np.round(dep_slope[0] * 100,2)}",
        )

        ax.text(
            pos_t + 0.05 * xlim[1],
            0.3 * ylim[0],
            f"Peak/trough ratio: {np.abs(np.round(ptrat,2))}",
        )
    except (AssertionError, UnboundLocalError, IndexError):
        ax.text(
            xlim[0] + (xlim[1] - xlim[0]) / 2,
            ylim[0] + (ylim[1] - ylim[0]) / 2,
            "Waveform features could not be computed for this unit.",
            horizontalalignment="center",
            verticalalignment="center",
            style="italic",
            bbox={"facecolor": "red", "alpha": 0.4, "pad": 10},
        )
    ax.set_ylim(ylim[0] + 0.1 * ylim[0], ylim[1] + 0.1 * ylim[1])

    ax.set_xlabel("ms")
    if normalise:
        ax.set_ylabel("Arbitrary units")
    else:
        ax.set_ylabel("\u03bcV")

    ticks = ticker.FuncFormatter(lambda x, pos: "{0:.2f}".format(x / 3000))
    ax.xaxis.set_major_formatter(ticks)
    if dp is not None:
        fig.suptitle(f"{dp} cell {unit}. {label}.")
    fig.tight_layout()


def temporal_features(all_spikes, sampling_rate=30_000) -> list:
    """
    It takes a list of spike times for each neuron, and returns a list of features that describe the
    temporal structure of the spike trains

    Args:
      all_spikes: a list of spike times for the given neuron.

    Returns:
      the features of the inter-spike interval, in a list
    """

    all_spikes = np.hstack(np.array(all_spikes))

    isi_block_clipped = compute_isi(all_spikes, fs=sampling_rate)

    isi_features = compute_isi_features(isi_block_clipped)

    acg_burst, acg_oscillation = acg_burst_vs_mfr(
        all_spikes, isi_features[0], sampling_rate
    )

    return [*isi_features, acg_burst, acg_oscillation]


def temporal_features_wrap(dp, unit, use_or_operator=True, use_consecutive=False):
    """
    High level function for getting the temporal features of a unit from a dataset at dp.

    Arguments:
        - use_or_operator: if True, if a chunk (10sec default) belongs to at least one 30s chunk
          passing both false positive and negative filters, it is considered good.
          Else, the 3 overlapping 30sec chunks must all pass both thresholds.
        - use_consecutive: if True, uses the longest period with consecutive good chunks.
            Else, uses a concatenation of all good chunks.
    """
    # get the train quality for each unit in the list of units
    # pass the spike times for that unit
    # accumulate all features for passed units
    # return matrix of features

    # units can be either a single integer or a list or np array of units

    unit_spikes, _ = trn_filtered(
        dp,
        unit,
        use_or_operator=use_or_operator,
        use_consecutive=use_consecutive,
        enforced_rp=0.2,
    )

    tf = temporal_features(unit_spikes) if len(unit_spikes) > 1 else np.zeros(15)
    return [str(dp), unit] + list(tf)


def check_json_file(json_path: str) -> None:
    """
    It checks that the files and units specified in the json file exist

    Args:
      json_f (str): Path to a json file containing datasets and units information.
    """
    with open(json_path, encoding="utf-8") as f:
        json_f = json.load(f)

    datasets = {}
    any_not_found = False
    for ds in tqdm(
        json_f.values(), desc="Checking provided json file", position=0, leave=False
    ):
        dp = Path(ds["dp"])
        datasets[dp.name] = ds
        if not dp.exists():
            print(f"\033[31;1m{dp} doesn't exist!!\033[0m")
            any_not_found = True
            continue
        units = ds["units"]
        units_m = np.isin(units, get_units(dp))
        if not all(units_m):
            print(f"\033[31;1m{np.array(units)[~units_m]} not found in {dp}!\033[0m")
            any_not_found = True
    if any_not_found:
        raise ValueError("Some files were not found")


def feature_extraction_json(
    json_path,
    check_json=True,
    save_path=None,
    use_unlabelled=False,
    ignore_exceptions=False,
    _debug=False,
):
    """
    It takes a json file containing paths to all the recordings and extracts the features.

    Params:

        json_path:  path to the json file that contains the data
        check_json: if True, will check that the files in the json file exist and that the units in
            the json file can all be found.
        save_path: where to save the csv file

    Returns:

        A dataframe with all the features for all the cells in the json file.
        This will include:
            - neuron information (optolabel, data_path, unit)
            - temporal features (15)
            - waveform features (18)
    """

    with open(json_path, encoding="utf-8") as f:
        json_f = json.load(f)

    if check_json:
        check_json_file(json_path)

    columns = FEATURES

    feat_df = pd.DataFrame(columns=columns)
    # Wrapping everything inside an exception block just to make sure we are returning the dataframe even if the results are partial.
    try:
        for ds in tqdm(
            json_f.values(),
            desc="Extracting features from datasets",
            position=0,
            leave=True,
        ):
            dp = Path(ds["dp"])
            optolabel = ds["ct"]
            if optolabel == "PkC":
                optolabel = "PkC_ss"
            if use_unlabelled:
                units = get_units(dp, "good").tolist()
                ss = []
                cs = []
            else:
                units = ds["units"]
                ss = ds["ss"]
                cs = ds["cs"]
            for u in tqdm(
                units + ss + cs,
                desc=f"Extracting features from {dp.name}",
                position=1,
                leave=False,
            ):
                if u in units:
                    label = "unlabelled" if use_unlabelled else optolabel
                elif u in ss:
                    label = "PkC_ss"
                elif u in cs:
                    label = "PkC_cs"
                try:
                    wvf_features = waveform_features_json(dp, u, plot_debug=_debug)
                    tmp_features = temporal_features_wrap(dp, u)
                    curr_feat = (
                        [label] + wvf_features[:2] + tmp_features[2:] + wvf_features[2:]
                    )
                    feat_df = feat_df.append(
                        dict(zip(columns, curr_feat)), ignore_index=True
                    )
                except Exception as e:
                    exc_type, _, exc_tb = sys.exc_info()
                    print(
                        f"Something went wrong for the feature computation of neuron {u} in {dp}"
                    )
                    print(f"{exc_type} at line {exc_tb.tb_lineno}: {e}")

                    if not ignore_exceptions:
                        raise

                    curr_feat = np.zeros(len(columns))
                    feat_df = feat_df.append(
                        dict(zip(columns, curr_feat)), ignore_index=True
                    )
    finally:
        if save_path is not None:
            today = date.today().strftime("%b-%d-%Y")
            feat_df.to_csv(f"{save_path}{today}_all_features.csv")

    return feat_df


def h5_feature_extraction(
    dataset_path: str,
    save_path=None,
    fix_chanmap=True,
    ignore_exceptions=False,
    quality_check=True,
    labels_only=True,
    _debug=False,
    _n_channels=21,
    _central_range=82,
    _label=None,
    _sampling_rate=30_000,
    _use_chanmap=True,
    _wvf_type="relevant",
    _clip_size=(1e-3, 2e-3),
):
    """
    It takes a NeuronsDataset instance coming from an h5 dataset and extracts the features.

    Params:

        dataset_path (str):  path to the h5 dataset
        save_path: where to save the csv file

    Returns:

        A dataframe with all the features for all the cells in the dataset.
        This will include:
            - neuron information (optolabel, data_path, unit)
            - temporal features (15)
            - waveform features (18)
    """
    if _label is None:
        _label = "optotagged_label"
    columns = FEATURES

    feat_df = pd.DataFrame(columns=columns)

    if isinstance(dataset_path, NeuronsDataset):
        dataset = dataset_path
    else:
        dataset = NeuronsDataset(
            dataset_path,
            quality_check=quality_check,
            normalise_wvf=False,
            n_channels=_n_channels,
            central_range=_central_range,
            _use_amplitudes=True,
            _label=_label,
        )

    if labels_only:
        dataset.make_labels_only()

    for i in tqdm(range(len(dataset)), desc="Extracting features"):
        dp = "/".join(dataset.info[i].split("/")[:-1])
        unit = int(dataset.info[i].split("/")[-1])
        label = CORRESPONDENCE[dataset.targets[i]]
        waveform = dataset.wf[i].reshape(dataset._n_channels, dataset._central_range)
        spike_train = dataset.spikes_list[i]
        # Recover the channelmap
        if _use_chanmap:
            try:
                if isinstance(dataset_path, NeuronsDataset):
                    chanmap = dataset.chanmap_list[i]
                    chanmap = np.array(chanmap)
                    if chanmap.shape[0] == 0:
                        # In this case the chanmap is invalid and we need to discard it
                        chanmap = None
                    elif waveform.shape[0] > chanmap.shape[0]:
                        # If this happened, the waveform was tiled and we need to recover
                        # the original one to match the chanmap
                        waveform = waveform[waveform.shape[0] - chanmap.shape[0] :, :]

                else:
                    chanmap_path = f"datasets/{dataset.info[i]}/channelmap"
                    with h5py.File(dataset_path, "r") as hdf5_file:
                        chanmap = hdf5_file[chanmap_path][(...)]
                        if fix_chanmap:
                            chanmap = recover_chanmap(chanmap)
            except KeyError:
                chanmap = None
        else:
            chanmap = None
        try:
            wvf_features = waveform_features(
                waveform,
                dataset._n_channels // 2,
                chanmap,
                plot_debug=_debug,
                waveform_type=_wvf_type,
                _clip_size=_clip_size,
                fs=_sampling_rate,
            )
            tmp_features = temporal_features(spike_train, _sampling_rate)

            curr_feat = [label, dp, unit, *tmp_features, *wvf_features]
            feat_df = feat_df.append(dict(zip(columns, curr_feat)), ignore_index=True)

        except Exception as e:
            exc_type, _, exc_tb = sys.exc_info()
            print(
                f"Something went wrong for the feature computation of neuron {i} (unit {unit} in {dp})"
            )
            print(f"{exc_type} at line {exc_tb.tb_lineno}: {e}")
            if not ignore_exceptions:
                raise

            curr_feat = np.zeros(len(columns))[3:].tolist()
            discarded_info = [label, dp, unit]
            curr_feat = discarded_info + curr_feat
            feat_df = feat_df.append(dict(zip(columns, curr_feat)), ignore_index=True)
    feat_df = feat_df.infer_objects()
    if save_path is None:
        save_path = os.getcwd()
        today = date.today().strftime("%b-%d-%Y")
        feat_df.to_csv(os.path.join(save_path, f"{today}_all_features.csv"))
    else:
        feat_df.to_csv(save_path)

    return feat_df


def get_unusable_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the index of unusable features
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    features_only = df.iloc[:, 2:]
    bad_idx = []
    for i, row in features_only.iterrows():
        value, count = np.unique(row.to_numpy(), return_counts=True)
        zeros = count[value == 0]
        if zeros.size > 0 and zeros > 5:
            bad_idx.append(i)
    bad_idx += df.index[df.isna().any(axis=1)].tolist()
    return np.unique(bad_idx)


def prepare_classification(
    df: pd.DataFrame, bad_idx=None, drop_cols=None
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the dataframe for classification.
    """

    if drop_cols is None:
        drop_cols = [
            "label",
            "dataset",
            "unit",
            "relevant_channel",
            "any_somatic",
            "max_peaks",
        ]
    if bad_idx is not None:
        df = df.drop(index=bad_idx)
        df = df.reset_index(drop=True)

    df = df.infer_objects()

    X, y = df.drop(columns=drop_cols, axis=1), df["label"]

    return X, y
