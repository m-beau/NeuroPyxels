"""
2021-4-22
Author: @agolajko

All functions needed for extracting temporal and waveform features from
Neuropixel recordings.


Functions in this doc to be used with TODO [insert file and function name]
for extracting the waveform and temporal features.


Example usage:
    TODO


"""
from tqdm import tqdm

import numpy as np
from pathlib import Path
from npyx.utils import peakdetect
from npyx.spk_wvf import wvf, wvf_dsmatch, get_depthSort_peakChans
from npyx.spk_t import trn_filtered
from npyx.plot import plot_wvf, plot_ccg, plt_wvf, quickplot_n_waves
import matplotlib.pyplot as plt
from scipy.stats import iqr, skew, norm
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import scipy.optimize as opt
from itertools import compress
from sklearn.metrics import mean_squared_error
from scipy import ndimage
from psutil import virtual_memory as vmem
from datetime import date

import pandas as pd
import json
import matplotlib.ticker as ticker
from npyx.inout import chan_map, read_metadata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from npyx.corr import (
    ccg,
    StarkAbeles2009_ccg_significance,
    ccg_sig_stack,
    gen_sfc,
    scaled_acg,
)
from npyx.gl import get_units, get_npyx_memory


#############################################
# Waveform features


def mean_start_subtract(waves, first_values=25, axes=1):
    """subtract the mean of the first 25 values from the wvf"""
    if len(waves.shape) == 1:
        waves = waves.reshape(1, -1)
    return waves - np.mean(waves[:, :first_values], axis=axes).reshape(-1, 1)


def pos_peak_amp_time(waves, axes=1):

    """
    Input: array of single vector or matrix
    Return: two vectors with the amp size and the time points
    """

    if len(waves.shape) == 1:
        waves = waves.reshape(1, -1)

    return np.max(waves, axis=axes), np.argmax(waves, axis=axes)


def neg_peak_amp_time(waves, axes=1):

    """
    Input: array of single vector or matrix
    Return: two vectors with the amp size and the time points
    """
    if len(waves.shape) == 1:
        waves = waves.reshape(1, -1)

    return np.min(waves, axis=axes), np.argmin(waves, axis=axes)


def cross_times(waves, peak_time, trough_time):
    """
    find where the wvf 'crosses' 0 between the most negative and the next peak

    """

    # crossing between the two peaks
    # find the two max peaks
    #    peak_t, peak_v = detect_peaks(waves.T)

    # get the most negative peak
    #    min_amp_arg = np.argmin(peak_v)
    # get the peak following it

    #    max_amp_arg = min_amp_arg +1
    #    min_amp = peak_v[min_amp_arg]
    #    max_amp = peak_v[max_amp_arg]
    #    pos_t = peak_t[max_amp_arg]
    #    neg_t = peak_t[min_amp_arg]

    # new start

    crossing1 = np.array([0,])

    crossing1 = (
        trough_time + np.where(np.diff(np.sign(waves[trough_time:peak_time])))[0][0]
    )
    crossing1 = crossing1.astype("int16")

    return crossing1, waves[crossing1 + 1]


# old start
#    _, pos_t = pos_peak_amp_time(waves)
#    _, neg_t = neg_peak_amp_time(waves)
#
#    # find the two time points on each side of the crossing
#    crossing1,crossing2 = np.zeros((2,waves.shape[0]))
#    for idx, wave in enumerate(waves):
#        if neg_t[idx] < pos_t[idx]:
#            crossing1[idx] = neg_t[idx] + np.where(np.diff(np.sign(waves[idx,neg_t[idx]:pos_t[idx]])))[0][0]
#            crossing2[idx] = crossing1[idx] + 1
#        else:
#            peak_t, peak_v = detect_peaks(waves.T)
#            min_amp_arg = np.argmin(peak_v)
#            # get the peak following it
#            if peak_t.shape[0] == 2 and peak_v[0] > peak_v[1] :
#                max_amp_arg = min_amp_arg-1
#
#                min_amp = peak_v[min_amp_arg]
#                max_amp = peak_v[max_amp_arg]
#                prev_peak_t = peak_t[max_amp_arg]
#                curr_peak_t = peak_t[min_amp_arg]
#
#            else:
#                max_amp_arg = min_amp_arg +1
#
#                min_amp = peak_v[min_amp_arg]
#                max_amp = peak_v[max_amp_arg]
#                curr_peak_t = peak_t[max_amp_arg]
#                prev_peak_t = peak_t[min_amp_arg]
#
##            max_amp_arg = min_amp_arg +1
##            min_amp = peak_v[min_amp_arg]
##            max_amp = peak_v[max_amp_arg]
##            pos_t = peak_t[max_amp_arg]
##            neg_t = peak_t[min_amp_arg]
#
##            crossing1[idx] = pos_t[idx] + np.where(np.diff(np.sign(waves[idx,pos_t[idx]:neg_t[idx]])))[0][0]
#            crossing1[idx] = neg_t + np.where(np.diff(np.sign(waves[idx,neg_t:pos_t])))[0][0]
#            crossing2[idx] = crossing1[idx] + 1
# old end


# now we have an array with the two times on each side of the zero cross
# find the value that is take on the wave and interpolate for the value
#    crossing1,crossing2 = crossing1.astype('int16'),crossing2.astype('int16')
# breakpoint()
#    return crossing1[0], waves[0,crossing1+1][0]


def detect_peaks(wvf, margin=0.8, onset=0.2, plot_debug=False):
    """Custom peak detection algorithm. Based on scipy.signal.find_peaks.
    Args:
        wvf (np.array): Waveform to detect peaks in.
        margin (float): Margin around baseline around which peaks are ignored, in std units.
        onset (float): Percentage of waveform to ignore before first and last peak.
    
    Returns:
        all_idxes (np.array): Indices of all peaks.
        all_values (np.array): Values of all peaks.
    """

    # Determine which peaks to ignore using provided margin
    peak_height = margin * np.std(wvf)

    # Find peaks using scipy. We also apply a distance criterion to avoid finding too close peaks in artefactual waveforms.
    max_idxes, _ = find_peaks(wvf, height=peak_height, distance=len(wvf) // 10)
    min_idxes, _ = find_peaks(-wvf, height=peak_height, distance=len(wvf) // 10)
    max_values = wvf[max_idxes]
    min_values = wvf[min_idxes]

    all_idxes = np.concatenate((max_idxes, min_idxes))
    all_values = np.concatenate((max_values, min_values))

    sorting_idx = np.argsort(all_idxes)
    all_idxes = all_idxes[sorting_idx]
    all_values = all_values[sorting_idx]

    mask = (all_idxes < len(wvf) // (1 / onset)) | (
        all_idxes > int(len(wvf) * (1 - onset))
    )
    all_idxes = all_idxes[~mask]
    all_values = all_values[~mask]

    if plot_debug:
        plt.plot(wvf)
        plt.fill_between(
            np.arange(len(wvf)),
            np.zeros_like(wvf) + margin * np.std(wvf),
            np.zeros_like(wvf) - margin * np.std(wvf),
            alpha=0.4,
        )
        for i, _ in enumerate(all_idxes):
            plt.plot(all_idxes[i], all_values[i], "x", color="red")
            plt.axvspan(0, len(wvf) // (1 / onset), alpha=0.2)
            plt.axvspan(int(len(wvf) * (1 - onset)), len(wvf), alpha=0.2)
        plt.show()

    return all_idxes, all_values


# def detect_peaks(wvf, outliers_dev=1.5):

#     """
#     Detect peaks happening on a wvf, with a filter applied in terms of std to
#     contrain peaks under threshold
#     Steps:
#             - find all the possible peaks with a noisy algorithm
#             - find the median of the wvf shape and the interquartile range
#             - if a peak happens outside 'middle'  band append it to list
#     Input: wvf
#     Returns: list of peak times and peak values
#     """

#     detected_peaks = []
#     xs = []
#     ys = []
#     # detect peaks using existing non precise method
#     for lk in list(range(3, 60, 3)):
#         detected_peaks.append(peakdetect(wvf, lookahead=lk))
#     # check if detected_peaks is empty
#     # it is usually only empty if the wvf vector is fully flat
#     # meaning all the values are the same
#     # flatten list of x,y coordinates
#     detected_peaks = [x for x in detected_peaks[0] if x != []]
#     detected_peaks = [item for items in detected_peaks for item in items]

#     #   there are some cases where the passed wvf value is blank zeros
#     #   this is a hardware fault, when a single channel might blank
#     #   can take care of this by checking if this condition holds and returning -1
#     if not detected_peaks:
#         # print('No peaks detected, passed -1,-1 as output')
#         return -1, -1
#     # get the iqr of the wvf and the median
#     # define window outisde of which the peaks can happen
#     iqr_ = iqr(wvf) * outliers_dev
#     _med = np.median(wvf)
#     lower = _med - iqr_
#     upper = _med + iqr_
#     # check for each peak if it is happening outside of middle window
#     # if yes, append it
#     # convert to np array
#     # find the entries where each condition is passed and make a matrix of them
#     dp = np.array(detected_peaks)
#     filtered_peaks = np.vstack((dp[dp[:, 1] <= lower], dp[upper <= dp[:, 1]]))
#     # if the largest positive peak is not in the list add it manually
#     x_new, y_new = filtered_peaks[:, 0], filtered_peaks[:, 1]
#     if np.max(wvf) not in y_new:
#         y_new = np.hstack((y_new, np.max(wvf)))
#         x_new = np.hstack((x_new, np.argmax(wvf)))
#     x_sorted = np.argsort(x_new)
#     x_new = x_new[x_sorted].astype("int")
#     y_new = y_new[x_sorted].astype("int")

#     return x_new, y_new


def wvf_width(wave, peak_time, trough_time):
    """
    time between the max positive and min negative peak
    """

    # use the last returned value for plotting only

    return (
        np.abs(peak_time - trough_time),
        trough_time,
        peak_time,
        wave[trough_time],
    )


def pt_ratio(wave, peak_time, trough_time):
    """
    ratio between the amplitude of the most positive ad most negaitve peak
    simply take the absolute values of the most negative and most positive points
    along the recording and find their ratio
    """

    # peak_t, peak_v = detect_peaks(wave)

    # neg_id = np.argmin(peak_v)
    # neg_v = peak_v[neg_id]

    # # get positive peak after the negative peak
    # if neg_id + 1 < peak_t.shape:
    #     pos_v = peak_v[neg_id + 1]
    # else:
    #     pos_v = peak_v[np.argmax(peak_v)]

    return np.abs(wave[peak_time] / wave[trough_time])


def onset_amp_time(waves, peak_time, trough_time):

    """
    Input: single wvf shape
    Return: time when the 5% of the first (negative or pos) peak was reached

    """

    onset_v = waves[trough_time] * 0.05

    # now we find the last crossing of this value before the peak
    before_peak = waves[:trough_time]
    crossings = np.where(np.diff(np.sign(before_peak - onset_v)))

    # if there are no crossings, this means the recording was always above
    # or below the value, hence the crossing can be said to have happened
    # before the start of the recording, which we approximate by 0

    if crossings[0].size == 0:
        last_cross = 0
    else:
        last_cross = crossings[-1][0]

    return last_cross, onset_v


def end_amp_time(waves, peak_time, trough_time):
    """
    Return: time when the last positive peak first reaches it's 5% value
            after the peak
    """
    # get the wvf peaks
    onset_v = waves[peak_time] * 0.05

    # now we find the last crossing of this value
    # get section after the last peak
    after_peak = waves[peak_time:]
    # now we have list of crossings where
    crossings = np.where(np.diff(np.sign(after_peak - onset_v)))

    # if there are crossings, get the first of these
    # if there are no crossings, return the shape of the wvf as we know
    # the last crossing happened after our time window
    if len(crossings[0]):
        last_cross = peak_time + crossings[0][0]
    else:
        last_cross = waves.shape[0]

    return last_cross, onset_v


def pos_10_90(waves, peak_time, trough_time):

    """
    (used by pos_decay_t, Orange line on features plot)
    Input: wvf
    Return: - times when the pos peak was at 10% and 90%
            - values at 10 and 90%
            - time it took to get from 10-90%
    """

    # get the wvf peaks
    #    peak_t, peak_v = detect_peaks(waves)

    # get the most negative peak
    #    min_amp_arg = np.argmin(peak_v)
    # get the peak following it

    prev_peak_t = trough_time
    curr_peak_t = peak_time
    peak_value = waves[peak_time]

    #    if peak_t.shape[0] == 2 and peak_v[0] > peak_v[1] :
    #        max_amp_arg = min_amp_arg-1
    #
    #        min_amp = peak_v[min_amp_arg]
    #        max_amp = peak_v[max_amp_arg]
    #        prev_peak_t = peak_t[max_amp_arg]
    #        curr_peak_t = peak_t[min_amp_arg]
    #
    #    else:
    #        max_amp_arg = np.argmax(peak_v)
    ##        max_amp_arg = min_amp_arg +1
    #
    #        min_amp = peak_v[min_amp_arg]
    #        max_amp = peak_v[max_amp_arg]
    #        curr_peak_t = peak_t[max_amp_arg]
    #        prev_peak_t = peak_t[min_amp_arg]

    #    perc_10 = 0.1 * max_amp
    #    perc_90 = 0.9 * max_amp

    perc_10, perc_90 = 0.1 * peak_value, 0.9 * peak_value

    # now need to find where the before and after cross two points happened
    # get the section of the slope we need
    upslope = waves[prev_peak_t:curr_peak_t]
    # get the points where the wave reaches the percentile values
    cross_10 = prev_peak_t + np.where(np.diff(np.sign(upslope - perc_10)))[0][0]
    cross_90 = prev_peak_t + np.where(np.diff(np.sign(upslope - perc_90)))[0][0]

    crosses_10 = np.array([cross_10, cross_10 + 1])
    # ensure the value is positive for the first crossing
    crosses_10 = crosses_10[waves[crosses_10] > 0]
    crosses_90 = np.array([cross_90, cross_90 + 1])
    # from these crossing points find the closer one to the value
    close_10 = crosses_10[np.argmin(np.abs(waves[crosses_10]) - perc_10)]
    close_90 = crosses_90[np.argmin(np.abs(waves[crosses_90]) - perc_90)]

    return (
        np.array([close_10, close_90]),
        np.array([waves[close_10], waves[close_90]]),
        close_90 - close_10,
    )


def pos_decay_t(wave, peak_time, trough_time):
    """
    time between when the positive peak happened and when 10% happened
    (Orange line on features plot)
    """
    # get the wvf peaks
    peak_value = wave[peak_time]

    pos_t = peak_time[np.argmax(peak_value)]
    perc_10_t = pos_10_90(wave)[0][0]

    return pos_t - perc_10_t


def neg_10_90(waves, peak_time, trough_time):
    """
    Calculate time it took to go from negative 10%-90% of peak
    Input: wvf
    Return: - times when the neg peak was at 10% and 90%
            - values at 10 and 90%
            - time it took to get from 10-90%

    """

    through_value = waves[trough_time]
    prev_peak_t = onset_amp_time(waves, peak_time, trough_time)[0]

    perc_10 = 0.1 * through_value
    perc_90 = 0.9 * through_value

    # now need to find where the before and after cross two points happened
    # get the section of the slope we need
    downslope = waves[prev_peak_t : trough_time + 1]
    # get the points where the wave reaches the percentile values
    cross_10 = prev_peak_t + np.where(np.diff(np.sign(downslope - perc_10)))[0][0]
    cross_90 = prev_peak_t + np.where(np.diff(np.sign(downslope - perc_90)))[0][0]
    # print(cross_10, cross_90)
    crosses_10 = np.array([cross_10, cross_10 + 1])
    crosses_90 = np.array([cross_90, cross_90 + 1])
    # from these crossing points find the closer one to the value
    close_10 = crosses_10[np.argmin(np.abs(waves[crosses_10]) - perc_10)]
    close_90 = crosses_90[np.argmin(np.abs(waves[crosses_90]) - perc_90)]

    return (
        np.array([close_10, close_90]),
        np.array([waves[close_10], waves[close_90]]),
        close_90 - close_10,
    )


def neg_decay_t(wave, peak_time, trough_time):

    """negative decay time: time between the negative peak amplitude, and when it was at 10%
    (Green line on features plot)"""

    perc_10_t = neg_10_90(wave)[0][0]

    return trough_time - perc_10_t


def depol_slope(waves, peak_time, trough_time):
    """
    regression fitted from the first 30 observations before the negative peak

    - find the negative peak
    - fit regression to the previous  10% of dots between this and the baseline
    """

    trough_value = waves[trough_time]

    # Normalise the waveform to correct for amplitude effects!
    waves = waves / np.max(np.abs(waves))

    # find number of points between negative and positive peaks

    all_dots_peak = np.abs((trough_time - waves.shape[0] // 10) - trough_time)
    dots_pos_neg_20 = (0.2 * all_dots_peak).astype(np.int64)
    fit_slope = waves[trough_time - dots_pos_neg_20 : trough_time]
    coeff = np.polyfit(
        np.linspace(0, dots_pos_neg_20 - 1, dots_pos_neg_20), fit_slope, deg=1
    )
    # fit a slope with the new parameters
    # all data points between two peaks
    # multiply by first coeff and add the second one
    return (
        coeff,
        coeff[0] * np.linspace(0, dots_pos_neg_20, dots_pos_neg_20 + 1) + coeff[1],
        trough_time,
        trough_value,
    )


# def peaks_order(waves):

#     """
#     Return: the positive and negative peaks, and an indicator for their order
#     """

#     # get the wvf peaks
#     peak_t, peak_v = detect_peaks(waves)

#     min_amp_arg_id = np.argmin(peak_v)
#     min_amp_arg = peak_t[min_amp_arg_id]
#     # get the peak following it

#     max_amp_arg_id = np.argmax(peak_v)
#     max_amp_arg = peak_t[max_amp_arg_id]

#     # if there are only two peaks and the positive is before the negative
#     if peak_t.shape[0] == 2 and peak_v[0] > peak_v[1]:
#         peak_order_flipped = 1
#     else:
#         peak_order_flipped = 0

#     min_amp = peak_v[min_amp_arg_id]
#     max_amp = peak_v[max_amp_arg_id]

#     # return, most negative peak time and value, most positive peak time and value, and order
#     return min_amp_arg, min_amp, max_amp_arg, max_amp, peak_order_flipped


def pos_half_width(waves, peak_time, trough_time):
    """
    Give the the half width time for the positive peak
    Input: wvf
    Return: - start time
            - end time
            - 50% value
            - duration
    """

    # regradless of the order of the peaks, return the half width time for the
    # the positive peak
    peak_value = waves[peak_time]
    perc_50 = 0.5 * peak_value

    # look for crossings from cross_time to end
    start_interval = cross_times(waves, peak_time, trough_time)[0].astype(np.int64)
    end_interval = waves.shape[0]

    current_slope = waves[start_interval:end_interval]
    # get the real time when the crossings happened, not just relative time
    cross_start = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][0]
    )
    cross_end = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][-1]
    )
    return cross_start, cross_end, perc_50, cross_end - cross_start


# old
# get the wvf peaks
#    peak_t, peak_v = detect_peaks(waves)
#
#    min_amp_arg = np.argmin(peak_v)
#    # get the peak following it
#
#    max_amp_arg = min_amp_arg +1
#    prev_peak_t = peak_t[min_amp_arg]
#    curr_peak_t = peak_t[max_amp_arg]
#
##    max_amp_arg = np.argmax(peak_v)
#    max_amp= peak_v[max_amp_arg]
##    curr_peak_t = peak_t[max_amp_arg]
##    prev_peak_t = peak_t[max_amp_arg-1]
#
#    perc_50 = 0.5 * max_amp
#    # find the interval we need start from 0 crossing to end
#
#    # start looking for the crossing from the onset.  Maybe faster without it?
#    # also need this in case there are other crossings that might happen at
#    # other times of the recording
#
#    # depending on the orientation of the peaks, find the corresponding
#    # pos half-width
#    start_interval = cross_times(waves)[0].astype(np.int64)
#    end_interval = end_amp_time(waves)[0]
#    current_slope = waves[start_interval:end_interval]
#    # get the real time when the crossings happened, not just relative time
#    cross_start = start_interval + np.where(np.diff(np.sign(current_slope- perc_50)))[0][0]
#    cross_end = start_interval + np.where(np.diff(np.sign(current_slope- perc_50)))[0][-1]
##    breakpoint()
#    return cross_start, cross_end,perc_50, cross_end - cross_start


def neg_half_width(waves, peak_time, trough_time):
    """
    Give the the half width time for the neg peak
    Input: wvf
    Return: - start time
            - end time
            - 50% value
            - duration
    """

    # regradless of the order of the peaks, return the half width time for the
    # the positive peak
    trough_value = waves[trough_time]
    perc_50 = 0.5 * trough_value

    # get the half width for the first peak
    # look for crossings from 0 to cross_time

    start_interval = 0
    end_interval = cross_times(waves, peak_time, trough_time)[0].astype(np.int64)

    current_slope = waves[start_interval:end_interval]
    # get the real time when the crossings happened, not just relative time
    cross_start = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][0]
    )
    cross_end = (
        start_interval + np.where(np.diff(np.sign(current_slope - perc_50)))[0][-1]
    )
    return cross_start, cross_end, perc_50, cross_end - cross_start


def tau_end_slope(waves, peak_time, trough_time):
    """
    find the last peak
    get 30 values after last peak
    fit exponential to these values
    get parameter of exponential
    """

    last_peak = peak_time
    slope_obs = waves[last_peak + 100 : last_peak + 1600]
    slope_correct = slope_obs - np.min(slope_obs) + 1
    logged = np.log(slope_correct)
    #    print(slope_obs.shape, logged.shape)
    K, A_log = np.polyfit(np.linspace(0, 1499, 1500), logged, 1)
    A = np.exp(A_log)
    fit = slope_obs[-1] + A * np.exp(K * np.linspace(0, 1499, 1500)) - 1

    mse = mean_squared_error(slope_obs, fit)

    return fit, mse, 1 / K


def interp_wave(wave, multi=100, axis=-1):
    """
    linearly upsample waves so there are overall multi times more points
    We need this for more precise estimates to when time points happened
    Input: wvf and upsampling multiplier
    Retun: upsampled wvf
    """
    if wave.ndim == 2:
        wvf_len = wave.shape[1]
    else:
        wvf_len = wave.shape[0]

    interp_fn = interp1d(np.linspace(0, wvf_len - 1, wvf_len), wave, axis=axis)

    new_wave = interp_fn(np.linspace(0, wvf_len - 1, wvf_len * multi))

    return new_wave


def repol_slope(waves, peak_time, trough_time):
    """
    regression fitted to the first 30 observations from the negative peak

    - find the negative peak
    - fit regression to the next  10% of dots between this and the next peak
    """

    trough_value = waves[trough_time]

    # Normalise the waveform to correct for amplitude effects!
    waves = waves / np.max(np.abs(waves))

    # find number of points between negative and positive peaks

    all_dots_peak = np.abs(peak_time - trough_time)
    dots_pos_neg_20 = (0.5 * all_dots_peak).astype(np.int64)
    fit_slope = waves[trough_time : trough_time + dots_pos_neg_20]
    coeff = np.polyfit(
        np.linspace(0, dots_pos_neg_20 - 1, dots_pos_neg_20), fit_slope, deg=1
    )
    # fit a slope with the new parameters
    # all data points between two peaks
    # multiply by first coeff and add the second one
    return (
        coeff,
        coeff[0] * np.linspace(0, dots_pos_neg_20, dots_pos_neg_20 + 1) + coeff[1],
        trough_time,
        trough_value,
    )


def recovery_slope(waves, peak_time, trough_time):
    """
    fit a regression to observations after the last pos peak
    """

    peak_value = waves[peak_time]

    # Normalise the waveform to correct for amplitude effects!
    waves = waves / np.max(np.abs(waves))

    #    peak_time = peak_t[np.argmax(peak_v)].astype(np.int64)
    #    peak_value = np.max(peak_v).astype(np.int64)

    # find number of points between negative and positive peaks

    all_dots_peak = waves.shape[0] - peak_time
    dots_pos_neg_20 = (0.2 * all_dots_peak).astype(np.int64)
    fit_slope = waves[peak_time : peak_time + dots_pos_neg_20]

    coeff = np.polyfit(
        np.linspace(0, dots_pos_neg_20 - 1, dots_pos_neg_20), fit_slope, deg=1
    )
    # fit a slope with the new parameters
    # all data points between two peaks
    # multiply by first coeff and add the second one
    return (
        coeff,
        coeff[0] * np.linspace(0, dots_pos_neg_20, dots_pos_neg_20 + 1) + coeff[1],
        peak_time,
        peak_value,
    )

def previous_peak_wrap(waves, chan_path, unit, n_chans=20):

    dpnm = get_npyx_memory(chan_path)
    max_chan_path = list(dpnm.glob(f"dsm_{unit}_peakchan*"))[0]
    max_chan = int(np.load(max_chan_path))

    pbp, max_pbp = previous_peak(waves, max_chan, n_chans)

    return pbp, max_pbp

def previous_peak(waves, max_chan, n_chans=20):
    """
    takes as input 384x82 matrix

    - take a matrix of waves along the probe
    - find the peak chan
    - find channels +- 20 from the peak chan
    - find the ratio between the most negative peak and the peak preceding it
    """

    # loop through each wave
    # detect_peaks
    # find most negative peak
    # check if there is a peak before the most negative one

    if max_chan <= n_chans - 1:
        bounds = (0, max_chan + n_chans + 1)
    elif max_chan > 384 - n_chans - 1:
        bounds = (max_chan - n_chans, 384)
    else:
        bounds = (max_chan - n_chans, max_chan + n_chans + 1)

    bound_waves = waves[bounds[0] : bounds[1] + 1]
    no_baseline_values = bound_waves.shape[1] // 4
    max_wav = waves[max_chan]
    mean_s = np.mean(max_wav[np.r_[0:no_baseline_values, -no_baseline_values:-1]])
    std_s = np.std(max_wav[np.r_[0:no_baseline_values, -no_baseline_values:-1]])
    #    breakpoint()
    pbp = np.zeros((bound_waves.shape[0]))
    # first n values to calculate mean and std
    for ids, wav in enumerate(bound_waves):
        peak_t, peak_v = detect_peaks(wav)
        if len(peak_t) == 0 or isinstance(
            peak_t, int
        ):  # To account when no peaks are found in both new and old implementation
            pbp[ids] = 0
        else:
            peak_v = (peak_v - mean_s) / std_s
            neg_t = np.argmin(peak_v)
            neg_v = peak_t[neg_t]
            # if the time when the negative peak happened is not 0
            # meaning this is not the first peak that happed
            # and hence there might be a 'previous peak'
            if neg_t != 0:
                pbp[ids] = np.abs(peak_v[neg_t - 1])
            else:
                pbp[ids] = 0
    # find the max amd argmax of pbp
    argmax_pbp = np.argmax(pbp)
    max_pbp = pbp[argmax_pbp]

    return pbp, max_pbp


def consecutive_peaks_amp(mean_waves: np.array) -> tuple:

    """
    Takes all channels corresponding to a single wave as input.
    Returns the maximum channel and the max amplitude.
    Max amplitude is calculated by first finding the most negative peak,
    then the most positive peak that happens in the time after the first neg
    peak is also found.
    This is in place to add an extra filter so we know the amplitudes we are
    using happen in the 'correct' order.

    Input:
        mean_waves -- np.arrray of waveforms
    Return:
        max_channel -- the channel where the consecuitive amplitude is maximised
        max_amplitude -- value of the max amplitude
        all_amps -- neg to next positive peak amplitude for all waves
    """
    middle_value = int(mean_waves.shape[0] * 40 / 82)
    mid_range = int(mean_waves.shape[0] * 15 / 82)

    truncated_waves = np.zeros_like(mean_waves.T)
    loc_min_val = np.argmin(mean_waves, axis=0)
    # truncate the waves so we can look at the amplitudes of the neg and next peak
    for idx, row in enumerate(mean_waves.T):
        truncated_waves[idx, loc_min_val[idx] :] = row[loc_min_val[idx] :]
    #    for idx, row in enumerate(mean_waves.T):
    #        truncated_waves[idx, middle_value:middle_value + mid_range] = row[middle_value:middle_value + mid_range]
    truncated_waves = truncated_waves.T
    #    breakpoint()
    return (
        np.argmax(np.ptp(truncated_waves, axis=0)),
        np.max(np.ptp(truncated_waves, axis=0)),
        np.ptp(truncated_waves, axis=0),
    )


def chan_dist(chan, chanmap):
    """
    given a chan and a chan map, will return 384 long vector of distances
    """
    current_loc = chanmap[chan, 1:]
    coordinates = chanmap[:, 1:] - current_loc
    # first we find all the distances for each channel
    all_dist = np.sqrt(np.sum(np.square(coordinates), axis=1))
    # next the positive or a negative value is assigned to each
    # depending on which side of the peak chan it is
    sign_mask = np.ones((384))
    sign_mask[:chan] = -1

    return all_dist * sign_mask


def in_distance_surface(dp, dist_limit=2000):
    """

    Return all units from a recording that have a peak channel that is within
    a certain distance from the 'surface' of the brain. This is in order to
    filter out neurons that are so deep as to be in the nuclei.
    """

    # get the peak channels for each unit sorted by channel location
    peak_chan_depth = get_depthSort_peakChans(dp, units=[], quality="good")

    surface_chan = peak_chan_depth[0, 1]

    # get the chanmap to work with
    chanmap = chan_map(dp)

    # get the channel distances from the highest channel
    # TODO check the orientation of the probe in chanmap and peak_chan_depth is same
    chan_dist_from_top_unit = chan_dist(surface_chan, chanmap)

    # filter for distances that are within 2mm, so 2000micrometer from the top unit
    chan_in_bounds = chanmap[np.abs(chan_dist_from_top_unit) < dist_limit][:, 0]

    # get the units that are on channels within the 2mm boundary
    select_chan_in_bound = peak_chan_depth[:, 0][
        np.isin(peak_chan_depth[:, 1], chan_in_bounds)
    ]

    return select_chan_in_bound


def chan_spread_wrap(all_wav, chan_path, unit, n_chans=20, probe_v="1.0_staggered"):

    # search for the file that has the given peak chan
    dpnm = get_npyx_memory(chan_path)
    max_chan_path = list(dpnm.glob(f"dsm_{unit}_peakchan*"))[0]
    max_chan = int(np.load(max_chan_path))

    chanmap = chan_map(chan_path)
    probe_v = read_metadata(chan_path)["probe_version"]

    dists, p2p, dist_p2p, sort_dist_p2p, spread = chan_spread(all_wav, max_chan, chanmap, n_chans, probe_v)

    return max_chan, dists, p2p, dist_p2p, sort_dist_p2p, spread

def chan_spread(all_wav, max_chan, chanmap, n_chans=20, probe_v="1.0_staggered"):
    """
    - take a 82*384 matrix
    - find the peak chan by looking at the amplitude differnce on all chans
    - calculate distance from peak chan to all other chans, pythagoras, chan_map
    - find the max amp of all chans
    - make a plot with the distance from peak chan on x axis and amplitude on y
    -
    """

    # find peak chan
    # find difference between min and max on all chans
    # use peal-to-peak

    #    p2p = np.ptp(all_wav, axis = 1)

    # find the most negative peak and the peak after that to
    # get the distance between peak and trough

    _, _, p2p = consecutive_peaks_amp(all_wav.T)
    
    # find the distance of all channels from peak chan
    dists = chan_dist(max_chan, chanmap)

    dist_p2p = np.vstack((dists, p2p)).T

    # I want a distribution of amplitudes with the x axis being distance
    # so sort the matrix so the first column is sorted
    # plot the second column
    sort_dist = np.argsort(dists)
    sort_dist_p2p = np.vstack((dists[sort_dist], p2p[sort_dist])).T

    if max_chan < n_chans + 1:
        bounds = (0, max_chan + n_chans + 1)
    elif max_chan > 384 - n_chans - 1:
        bounds = (max_chan - n_chans, 384)
    else:
        bounds = (max_chan - n_chans, max_chan + n_chans + 1)
    # bound_dist = dists[bounds[0] : bounds[1] + 1]
    # bound_p2p = p2p[bounds[0] : bounds[1] + 1]
    # bound_dist_p2p = dist_p2p[bounds[0] : bounds[1] + 1]
    sort_dist_p2p = sort_dist_p2p[bounds[0] : bounds[1] + 1]
    # get the chanel maximum peak-to-peak distance from the channels
    # at chan_spread_dist appart from the peak chan

    # add separate conditions for what distance to look at depending
    # on the probe version being used

    # get the probe version
    if probe_v in ["3A", "1.0_staggered"]:
        chan_spread_dist = 25.61249695

    elif probe_v == "2.0_singleshank":
        chan_spread_dist = 15

    #    if chan_spread_dist == 25.6: chan_spread_dist = 25.61249695
    vals_at_25 = sort_dist_p2p[:, 1][
        np.round(sort_dist_p2p[:, 0]) == np.round(chan_spread_dist)
    ]
    spread = np.max(vals_at_25)

    return dists, p2p, dist_p2p, sort_dist_p2p, spread


def wvf_shape(wave):
    # check if the wvf shape is one that we need
    # either first large negative peak and then a smaller pos
    # first pos peak, then largest neg peak and small positive peak
    # largest first positive peak and smaller neg peak is not good

    peak_t, peak_v = detect_peaks(wave)
    # start_mean, end_mean = True, True
    peak_trough, not_flat, peak_order = (False,) * 3

    # TODO GET RID OF HARD-CODED NUMBERS
    if np.max(peak_v) - np.min(peak_v) > 15:
        peak_trough = True

    # Checking that the peaks are at least 3 std away from the baseline (i.e. the waveform is not just noise)
    if np.max([np.abs(np.min(wave)), np.abs(np.max(wave))]) > 3 * np.std(
        wave[: len(wave) // 6]
    ):
        not_flat = True

    if np.argmax(peak_v) > np.argmin(peak_v):
        peak_order = True

    return True  # peak_trough  and peak_order and not_flat


def chan_plot(waves, peak_chan, n_chans=20):

    """
    find the peak chan and the n_chans on either side
    plot the mean wvf of the channels
    return object that can be plotted with
    """

    plot_waves = waves[peak_chan - n_chans : peak_chan + n_chans + 2]
    #    print(plot_waves.shape)
    fig, ax = plt.subplots(n_chans + 1, 2)

    for i in range(plot_waves.shape[0]):
        # find where to have the plot depending on

        y_axis = (i // 2) % (plot_waves.shape[0] + 1)
        x_axis = i % 2
        ax[y_axis, x_axis].plot(plot_waves[i])
        #   plt.fill_between(np.arange(0,82), wave_form-deviations,wave_form+deviations, color = 'paleturquoise')
        # ax[y_axis, x_axis].set_title(f'{peak_chan-n_chans+i}', y= 1)
        ax[y_axis, x_axis].text(0, 0, f"{peak_chan-n_chans+i}")
        ax[y_axis, x_axis].set_xticks([])
        ax[y_axis, x_axis].set_yticks([])
        # ax[y_axis, x_axis].set_xlim(500)
        ax[y_axis, x_axis].set_ylim(np.min(plot_waves), np.max(plot_waves))

    ax[10, 0].set_xticks([0, 40, 82])
    ax[10, 0].set_xticklabels([-1.365, 0, 1.365])
    ax[10, 0].set_yticks([np.min(plot_waves), 0, np.max(plot_waves)])
    ax[10, 0].set_yticklabels(
        [np.round(np.min(plot_waves), 2), 0, np.round(np.max(plot_waves), 2)]
    )
    ax[10, 0].set_xlabel("ms")
    ax[10, 0].set_ylabel(r"$\mu$V")

    fig.suptitle(f"{n_chans} chans on either side of the peak chan {peak_chan}")
    fig.set_size_inches(10, 20)
    # fig.tight_layout()
    # fig.subplots_adjust(top =0.9)
    plt.show()


####################################################
# Temporal features
def compute_isi(train, quantile=0.02, fs=30_000):

    """
    Input: spike times in samples and returns ISI of spikes that pass through
        exclusion quantile
    Operations: if quantile is given the given quantile from the ISI will be
                discarded, returning the spikes
    Returns: isi in s"""

    isi_ = np.diff(train).astype(np.int64)
    # isi_ = np.asarray(diffs, dtype='float64')
    if quantile is not None:
        m = (isi_ >= np.quantile(isi_, quantile)) & (
            isi_ <= np.quantile(isi_, 1 - quantile)
        )
        isi_ = isi_[m]
        return isi_ / fs
    else:
        return isi_ / fs


def compute_entropy_dorval(isint):

    """
    Dorval2007:
    Using logqrithmic ISIs i.e. ISIs binned in bISI(k) =ISI0 * 10**k/κ with k=1:Klog.
    Classical entropy estimation from Shannon & Weaver, 1949.
    Spike entropy characterizes the regularity of firing (the higher the less regular)
    """

    ## Compute entropy as in Dorval et al., 2009
    # 1) Pisi is the logscaled discrete density of the ISIs for a given unit
    # (density = normalized histogram)
    # right hand side of k_th bin is bISI(k) =ISI0 * 10**k/κ with k=1:Klog
    # where ISI0 is smaller than the smallest ISI -> 0.01
    # and Klog is picked such that bISI(Klog) is larger than the largest ISI -> 300 so 350
    # K is the number of bins per ISI decade.

    # Entropy can be thought as a measurement of the sharpness of the histogram peaks,
    # which is directly related with a better defined structural information

    #    ISI0 = 0.1
    #    Klog = 350
    #    K = 200

    #    try:
    # binsLog = ISI0 * 10 ** (np.arange(1, Klog + 1, 1) * 1. / K)
    binsLog = 200
    num, bins = np.histogram(isint, binsLog)
    histy, histx = num * 1.0 / np.sum(num), bins[1:]
    sigma = (1.0 / 6) * np.std(histy)
    Pisi = ndimage.gaussian_filter1d(histy, sigma)

    #    except ValueError:
    #        binsLog = 200
    #        num, bins = np.histogram(isint, binsLog)
    #        histy, histx = num * 1. / np.sum(num), bins[1:]
    #        sigma = (1. / 6) * np.std(histy)
    #        Pisi = ndimage.gaussian_filter1d(histy, sigma)

    # Remove 0 values
    non0vals = Pisi > 0
    Pisi = Pisi[non0vals]

    entropy = 0

    for i in range(len(Pisi)):
        entropy += -Pisi[i] * np.log2(Pisi[i])

    return entropy


def compute_isi_features(isint):

    # Instantaneous frequencies were calculated for each interspike interval as the reciprocal of the isi;
    # mean instantaneous frequency as the arithmetic mean of all values.
    mfr = 1 / np.mean(isint)
    mifr = np.mean(1.0 / isint)

    # Median inter-spike interval distribution
    # medISI = np.median(isint)
    med_isi = np.median(isint)

    # Mode of inter-spike interval distribution
    # Why these values for the bins?
    num, bins = np.histogram(isint, bins=np.arange(0, 0.1 + 0.001, 0.001))
    mode_isi = bins[np.argmax(num)]

    # Burstiness of firing: 5th percentile of inter-spike interval distribution
    prct5ISI = np.percentile(isint, 5)

    # Entropy of inter-spike interval distribution
    entropy = compute_entropy_dorval(isint)

    # Average coefficient of variation for a sequence of 2 ISIs
    # Relative difference of adjacent ISIs
    CV2_mean = np.mean(2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1]))
    CV2_median = np.median(
        2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1])
    )  # (Holt et al., 1996)

    # Coefficient of variation
    # Checked!
    CV = np.std(isint) / np.mean(isint)

    # Instantaneous irregularity >> equivalent to the difference of the log ISIs
    # Checked!
    IR = np.mean(np.abs(np.log(isint[1:] / isint[:-1])))

    # # Local Variation
    # Checked!
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    Lv = 3 * np.mean(
        np.ones((len(isint) - 1))
        - (4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2)
    )

    # Revised Local Variation, with R the refractory period in the same unit as isint
    # Checked!
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    R = 0.8  # ms
    LvR = 3 * np.mean(
        (
            np.ones((len(isint) - 1))
            - (4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2)
        )
        * (np.ones((len(isint) - 1)) + (4 * R / (isint[:-1] + isint[1:])))
    )

    # Coefficient of variation of the log ISIs
    # Checked!
    LcV = np.std(np.log10(isint)) * 1.0 / np.mean(np.log10(isint))

    # Geometric average of the rescaled cross correlation of ISIs
    # Checked!
    SI = -np.mean(
        0.5 * np.log10((4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2))
    )

    # Skewness of the inter-spikes intervals distribution
    # Checked!
    SKW = skew(isint)

    # Entropy not included
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
        IR,
        Lv,
        LvR,
        LcV,
        SI,
        SKW,
    ]


def healthy_waveform(waveform_1d, peaks_values):
    """
    Define if waveform looks healthy.
    Works for somatic and dendritic waveforms, and also fat spikes.
    
    - at least 1 waveform amplitude > 30uV
    - count peaks from peak detect - should be between 2 and 5.
    """

    if len(peaks_values) < 2 or len(peaks_values) > 5:
        return False

    if np.ptp(waveform_1d) < 30:
        return False

    return True


def is_somatic(peaks_v):
    """ 
    Assures that the waveform is somatic and can be used in further processing.
    For this we need to have at least a through followed by a peak in the waveform we found.
    """

    peak_signs = np.sign(peaks_v).astype(np.int32).tolist()
    peak_signs = str(peak_signs)
    pattern = "-1, 1"

    return pattern in peak_signs


def find_relevant_waveform(waveform_2d, peak_channel, max_chan_lookaway=16):
    """
    Input:
        - 2D waveform (n channels x t samples)
        
    Algorithm:
    0 - if healthy_waveform returns True
        else: returns NaNs
    1 - get waveforms on channels with high enough peak to peak amplitude (30uV)
    2 - check if these waveforms are somatic (biphasic) or axonal (triphasic) or axonal-terminal (quadriphasic)
        2.0 - if healthy_waveform returns True_
        2.1 - if is_somatic returns True
    3 - among waveforms detected as somatic, take channel with max amplitude
    3 - set boolean as True
    
    2bis - no somatic waveform is found
    3bis - among waveforms detected as non somatic, take channel with max amplitude
    4bis - flip 1D waveform vertically
    5bis - set boolean as False
    
    Returns:
    - 1D waveform on relevant channel (either somatic or dendritic)
    - boolean, True if somatic, False if dendritic
    """
    # Initialise utility variables
    any_somatic = False
    candidate_channel_somatic = []
    candidate_channel_non_somatic = []
    wf_channels = np.arange(len(waveform_2d))

    # Determine which channels to consider
    chan_slice = slice(max(0, peak_channel - max_chan_lookaway), min(peak_channel + max_chan_lookaway + 1, waveform_2d.shape[0]))
    considered_waveforms = waveform_2d[chan_slice]
    considered_channels = wf_channels[chan_slice]

    # Filter away low amplitude channels
    high_amp_waveforms = considered_waveforms[np.ptp(considered_waveforms, axis=1) > 30]
    high_amp_channels = considered_channels[np.ptp(considered_waveforms, axis=1) > 30]

    # Scan remaining channels for candidate somatic waveforms
    for channel, wf in zip(high_amp_channels, high_amp_waveforms):
        _, peak_v = detect_peaks(wf)
        if healthy_waveform(wf, peak_v):
            if is_somatic(peak_v):
                any_somatic = True
                candidate_channel_somatic.append(channel)
            else:
                candidate_channel_non_somatic.append(channel)
        else:
            continue

    if any_somatic == True:
        candidate_waveforms = waveform_2d[candidate_channel_somatic]
        relevant_channel = candidate_channel_somatic[
            np.argmax(np.ptp(candidate_waveforms, axis=1))
        ]
        return waveform_2d[relevant_channel], True, relevant_channel

    elif any_somatic == False and len(candidate_channel_non_somatic) > 0:
        # All good waveforms found are dendritic, so we return the best one flipped
        candidate_waveforms = waveform_2d[candidate_channel_non_somatic]
        relevant_channel = candidate_channel_non_somatic[
            np.argmax(np.ptp(candidate_waveforms, axis=1))
        ]
        return -waveform_2d[relevant_channel], False, relevant_channel

    else:
        # No relevant waveform found
        return None, None, None


def find_relevant_peaks(peak_t, peak_v):
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

    if len(peak_t) == 2:
        return peak_t[0], peak_t[1]

    signs = np.sign(peak_v).astype(np.int32)
    trough_mask = signs == -1
    first_trough_idx = np.where(trough_mask)[0][0]
    first_trough_t = peak_t[first_trough_idx]
    first_peak_t = peak_t[first_trough_idx + 1]

    return first_trough_t, first_peak_t


def extract_peak_channel_features(relevant_waveform):
    """
    It takes a waveform and returns a list of features that describe the waveform
    
    Args:
        relevant_waveform: the waveform to be analyzed
    
    Returns:
        The return is a list of the features that are being extracted from the waveform.
    """

    # First interpolates the waveform for higher precision
    relevant_waveform = interp_wave(relevant_waveform)

    peak_times, peak_values = detect_peaks(relevant_waveform)

    first_trough_t, first_peak_t = find_relevant_peaks(peak_times, peak_values)

    neg_v = relevant_waveform[first_trough_t]
    neg_t = first_trough_t

    pos_v = relevant_waveform[first_peak_t]
    pos_t = first_peak_t

    # pos 10-90
    _, _, pos_10_90_t = pos_10_90(relevant_waveform, first_peak_t, first_trough_t)

    # neg 10-90
    _, _, neg_10_90_t = neg_10_90(relevant_waveform, first_peak_t, first_trough_t)

    # pos half width
    _, _, _, pos50 = pos_half_width(relevant_waveform, first_peak_t, first_trough_t)

    # neg half width
    _, _, _, neg50 = neg_half_width(relevant_waveform, first_peak_t, first_trough_t)

    # onset amp and time
    onset_t, onset_amp = onset_amp_time(relevant_waveform, first_peak_t, first_trough_t)

    # wvf duration
    wvfd, _, _, _ = wvf_width(relevant_waveform, first_peak_t, first_trough_t)

    # pt ratio
    ptr = pt_ratio(relevant_waveform, first_peak_t, first_trough_t)

    # recovery slope
    coeff1, _, _, _ = recovery_slope(relevant_waveform, first_peak_t, first_trough_t)

    # repolarisation slope
    coeff2, _, _, _ = repol_slope(relevant_waveform, first_peak_t, first_trough_t)

    # depolarisation slope
    coeff3, _, _, _ = depol_slope(relevant_waveform, first_peak_t, first_trough_t)

    # Multiply slope coefficients by 100 to undo interpolation effect and obtain meaningful values
    coeff1, coeff2, coeff3 = coeff1 * 100, coeff2 * 100, coeff3 * 100

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
        coeff1[0],
        coeff2[0],
        coeff3[0],
    ]


def extract_spatial_features_wrap(waveform_2d, somatic, dp, unit):

    _, _, _, _, _, spatial_spread_ratio = chan_spread_wrap(
        waveform_2d, dp, unit
    )  # ratio between max amplitude among 4/5 closest channels at 23.61 um and max amplitude on peak channel

    _, max_dendritic_voltage = (
        previous_peak_wrap(waveform_2d, dp, unit) if somatic else (1, 1)
    )  # set to 1 if relevant waveform is dendritic already (because normalized)

    return spatial_spread_ratio, max_dendritic_voltage

def extract_spatial_features(waveform_2d, peak_chan, somatic):
    """
    - peak_chan: channel from which the 1D waveworm features will be extracted
    """

    n_chans = 20
    probe_v = "1.0_staggered"
    chanmap = chan_map(probe_version="1.0")

    # ratio between max amplitude among 4/5 closest channels at 23.61 um and max amplitude on peak channel

    _, _, _, _, spatial_spread_ratio = chan_spread(
        waveform_2d, peak_chan, chanmap, n_chans, probe_v
    )  

    # set to 1 if relevant waveform is dendritic already (because normalized)
    _, max_dendritic_voltage = (
        previous_peak(waveform_2d, peak_chan, n_chans) if somatic else (1, 1)
    )  

    return spatial_spread_ratio, max_dendritic_voltage


def new_waveform_features(waveform_2d, peak_channel, plot_debug=False):
    """
    It takes a datapoint and unit, finds the relevant waveform, and extracts features from it
    
    Args:
      dp: the datapoint
      unit: the unit number
      plot_debug: If True, plots the waveforms and the relevant channel. Defaults to False
    
    Returns:
      A numpy array of the following:
        [dp, unit, relevant_channel, *peak_channel_features, *spatial_features]
    """

    # First find working waveform
    relevant_waveform, somatic, relevant_channel = find_relevant_waveform(
        waveform_2d.T, peak_channel
    )

    # If None is found features cannot be extracted
    if relevant_waveform is None:
        return [peak_channel, *[0] * 16]

    if plot_debug:
        info = f'Original peak: {peak_channel} relevant channel: {relevant_channel}. {"Somatic." if somatic else "Dendritic."}'
        plot = quickplot_n_waves(
            waveform_2d,
            info,
            relevant_channel,
            24,
            custom_text=[
                (
                    healthy_waveform(wf, detect_peaks(wf)[1])
                    and is_somatic(detect_peaks(wf)[1])
                )
                for wf in waveform_2d.T[relevant_channel - 12 : relevant_channel + 12]
            ],
        )
        plt.show()
        print(
            "\n +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ \n"
        )
        return None

    peak_channel_features = extract_peak_channel_features(relevant_waveform)
    spatial_features = extract_spatial_features(waveform_2d.T, peak_channel, somatic)

    return [
        int(relevant_channel),
        *peak_channel_features,
        *spatial_features,
    ]

def waveform_features_wrap(dp, unit, plot_debug=False):

    _, waveform_2d, _, peak_channel = wvf_dsmatch(
        dp, unit, verbose=False, again=False, save=True,
    )

    wvf_feats = new_waveform_features(waveform_2d, peak_channel, plot_debug)

    return [str(dp), unit] + wvf_feats

def waveform_features(all_waves, dpath, peak_chan, unit):
    # return: list of all features for a unit

    # get the negative peak
    best_wave = all_waves[peak_chan].reshape(1, -1)
    best_wave -= np.mean(best_wave[:20])
    best_wave = interp_wave(best_wave).reshape(-1)

    if not wvf_shape(best_wave):
        return list(np.zeros(17))
    # get positive peak

    peak_t, peak_v = detect_peaks(best_wave)
    neg_id = np.argmin(peak_v)
    neg_v = peak_v[neg_id]
    neg_t = peak_t[neg_id]

    if neg_id + 1 <= peak_t.shape:
        pos_v = peak_v[neg_id + 1]
        pos_t = peak_t[neg_id + 1]

    # pos 10-90
    _, _, pos_10_90_t = pos_10_90(best_wave)

    # neg 10-90
    _, _, neg_10_90_t = neg_10_90(best_wave)

    # pos half width
    _, _, _, pos50 = pos_half_width(best_wave)

    # neg half width
    _, _, _, neg50 = neg_half_width(best_wave)

    # onset amp and time
    onset_t, onset_amp = onset_amp_time(best_wave)

    # wvf duration
    wvfd, _, _, _ = wvf_width(best_wave)

    # pt ratio
    ptr = pt_ratio(best_wave)

    # rec slope
    coeff1, _, _, _ = recovery_slope(best_wave)

    # repol slope
    coeff2, _, _, _ = repol_slope(best_wave)

    # chan spread
    _, _, _, _, _, chans = chan_spread_wrap(all_waves, dpath, unit)

    # backprop spread
    _, backp_max = previous_peak_wrap(all_waves, dpath, unit)

    ret_arr = [
        unit,
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
        coeff1[0],
        coeff2[0],
        chans,
        backp_max,
    ]
    return ret_arr


def plot_all(one_wave, dp=None, unit=None, label=None):
    #    one_wave = one_wave[0]
    wvf = interp_wave(one_wave)
    wvf -= np.mean(wvf[:2000])
    wvf = wvf / np.max(np.abs(wvf))
    peak_t, peak_v = detect_peaks(wvf)

    trough_time, peak_time = find_relevant_peaks(peak_t, peak_v)

    onset = onset_amp_time(wvf, peak_time, trough_time)
    offset = end_amp_time(wvf, peak_time, trough_time)
    # positive_line = pos_10_90(wvf, peak_time, trough_time)
    # negative_line = neg_10_90(wvf, peak_time, trough_time)
    zero_time = cross_times(wvf, peak_time, trough_time)
    pos_hw = pos_half_width(wvf, peak_time, trough_time)
    neg_hw = neg_half_width(wvf, peak_time, trough_time)
    # end_slope, mse_fit, tau = tau_end_slope(wvf, peak_time, trough_time)
    rec_slope = recovery_slope(wvf, peak_time, trough_time)
    rep_slope = repol_slope(wvf, peak_time, trough_time)
    dep_slope = depol_slope(wvf, peak_time, trough_time)
    wvf_dur = wvf_width(wvf, peak_time, trough_time)
    ptrat = pt_ratio(wvf, peak_time, trough_time)

    # get the negative and positive peaks for the PtR
    neg_t = trough_time
    neg_v = wvf[trough_time]
    pos_v = wvf[peak_time]
    pos_t = peak_time

    #    print(tau)
    #    print(ptrat)
    plt.ion()
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.plot(wvf, linewidth=2, alpha=0.8)
    ax.plot(peak_t, peak_v, "*", color="red", markersize=10)
    ax.plot(onset[0], onset[1], "rx")
    ax.plot(offset[0], offset[1], "rx")

    ax.plot(zero_time[0], 0, "rx")
    ax.plot([neg_t, neg_t], [0, neg_v], linewidth=1, c="black", linestyle="dotted")
    ax.plot([pos_t, pos_t], [0, pos_v], linewidth=1, c="black", linestyle="dotted")
    ax.plot([neg_t, pos_t], [0, 0], linewidth=1, c="black", linestyle="dotted")

    # ax.plot(
    #     [positive_line[0][0], positive_line[0][1]],
    #     [positive_line[1][1], positive_line[1][1]],
    # )
    # ax.plot(
    #     [negative_line[0][0], negative_line[0][1]],
    #     [negative_line[1][1], negative_line[1][1]],
    # )

    ax.plot(
        rec_slope[2] + np.arange(0, rec_slope[1].shape[0]),
        rec_slope[1],
        linewidth=3,
        linestyle="dotted",
        color="red",
    )
    ax.plot(
        rep_slope[2] + np.arange(0, rep_slope[1].shape[0]),
        rep_slope[1],
        linewidth=3,
        linestyle="dotted",
        color="red",
    )
    ax.plot(
        (dep_slope[2] + np.arange(0, dep_slope[1].shape[0])) - dep_slope[1].shape[0],
        dep_slope[1],
        linewidth=3,
        linestyle="dotted",
        color="red",
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot([pos_hw[0], pos_hw[1]], [pos_hw[2], pos_hw[2]], color="purple")
    ax.plot([neg_hw[0], neg_hw[1]], [neg_hw[2], neg_hw[2]], color="purple")
    ax.plot(
        np.linspace(wvf_dur[1], wvf_dur[2], wvf_dur[0] + 1),
        np.linspace(
            wvf_dur[3] + 0.1 * ylim[0], wvf_dur[3] + 0.1 * ylim[0], wvf_dur[0] + 1
        ),
    )

    ax.text(
        wvf_dur[2] + 0.01 * xlim[1],
        wvf_dur[3] + 0.1 * ylim[0],
        f"wvf duration: {np.round(wvf_dur[0]/3000, 2)}",
    )

    # NOTE: we are multiplying the slope values by 100 to undo the interpolation effect and obtain meaningful values!

    ax.text(
        rec_slope[2] + 0.1 * xlim[1],
        rec_slope[3],
        f"rec slope: {np.round(rec_slope[0][0]* 100,2)}",
    )
    ax.text(
        rep_slope[2] + 0.1 * xlim[1],
        rep_slope[3] - 0.1 * ylim[0],
        f"rep slope: {np.round(rep_slope[0][0]* 100,2)}",
    )
    ax.text(
        dep_slope[2] - 0.2 * xlim[1],
        dep_slope[3] - 0.1 * ylim[0],
        f"dep slope: {np.round(dep_slope[0][0] * 100,2)}",
    )

    # ax.text(peaks[0][-1]+500, peaks[1][-1]-15, f"MSE of fit is {np.round(mse_fit,2)} \n tau: {np.round(tau, 2)}")

    ax.text(
        pos_t + 0.05 * xlim[1],
        0.3 * ylim[0],
        f"Peak/trough ratio: {np.abs(np.round(ptrat,2))}",
    )
    ax.set_ylim(ylim[0] + 0.1 * ylim[0], ylim[1] + 0.1 * ylim[1])

    ax.set_xlabel("ms")
    ax.set_ylabel("Arbitrary units")

    ticks = ticker.FuncFormatter(lambda x, pos: "{0:.2f}".format(x / 3000))
    ax.xaxis.set_major_formatter(ticks)
    if dp is None:
        pass
    else:
        fig.suptitle(f"{dp} cell {unit}. {label}.")
    fig.tight_layout()


def chan_spread_bp_plot(dp, unit, n_chans=20):
    """
    Generates a plot with number of channels and the channels on left
    and backprop and chan spread on the right
    Input: datapath and unit (drift and shift matched datasets for now)
    Returns: plot
    """
    dpnm = get_npyx_memory(dp)
    curr_fil = (
        dpnm / f"dsm_{unit}_all_waves_100-82_regular_False300-FalseNone-FalseNone.npy"
    )
    if curr_fil.is_file():

        if n_chans % 2 != 0:
            n_chans += 1
        all_waves_unit_x = np.load(curr_fil)
        backp, true_bp = previous_peak_wrap(all_waves_unit_x.T, dp, unit, n_chans)
        csp_x = chan_spread_wrap(all_waves_unit_x.T, dp, unit, n_chans)
        peak_chan = csp_x[0]
        print(peak_chan)
        plt.figure()
        plt.plot(csp_x[4][:, 0], csp_x[4][:, 1])
        plt.show()
        fig, ax = plt.subplots(3, 1)
        if backp.shape[0] == 42:

            ax[0].plot(csp_x[4][:, 0], backp)
        else:
            n_chans_edge = backp.shape[0]
            ax[0].plot(
                np.linspace(0, n_chans_edge - 1, n_chans_edge).astype("int"), backp
            )
        ax[0].set_ylabel("Z-scored value \n  of the previous peak")

        ax[1].plot(csp_x[4][:, 0], csp_x[4][:, 1])
        ax[1].set_xlabel("Distance from peak chan (micrometer)")
        ax[1].set_ylabel("Amplitude of  negative \n to positive peak")
        ax[1].title.set_text(
            f"Amplitude of most negative peak to next peak for all chans {csp_x[5]}"
        )

        t_all = np.round(82 / 30, 2)
        t_xaxis = np.linspace(-t_all / 2, t_all, 82)
        t_xaxis = np.hstack(
            (
                np.linspace(-t_all / 2, 0, 41),
                np.linspace(t_all / 82, t_all / 2 + t_all / 82, 41),
            )
        )
        ax[2].plot(t_xaxis, all_waves_unit_x.T[csp_x[0]])
        ax[2].set_xlabel("ms")
        ax[2].set_ylabel(r"$\mu$V")

        ax[2].title.set_text(f"Wvf of Peak chan {csp_x[0]} ")
        fig.tight_layout()

        if csp_x[0] > n_chans // 2:
            to_plot = n_chans // 2
        else:
            to_plot = csp_x[0] - 1
        chan_plot(all_waves_unit_x.T, csp_x[0], to_plot)


def temp_feat(all_spikes):
    """
    Input: spike times IN SAMPLES
    Returns: list of features
    """
    all_spikes = np.hstack(np.array(all_spikes))

    isi_block_clipped = compute_isi(all_spikes)

    return compute_isi_features(isi_block_clipped)


def temporal_features_wrap(dp, unit, use_or_operator=True, use_consecutive=False):
    """
    High level function for getting the temporal features of a unit from a dataset at dp.

    Parameters:
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

    if len(unit_spikes) > 1:
        tf = temp_feat(unit_spikes)
    else:
        tf = np.zeros(15)

    return [str(dp), unit] + list(tf)


def wvf_feat(dp, units):
    """
    High level function for getting the wvf features from a single (integer) or
    set of units (list of units) from a dp dataset.
    """
    if isinstance(units, (int, np.int16, np.int32, np.int64)):
        units = [units]

    all_ft_list = []
    spike_clusters = np.load(Path(dp, "spike_clusters.npy"))
    for unit in units:

        #       the below 10_000 spike value was chosen by testing what the lowest
        #       value is that still works
        len_unit_spks = len(spike_clusters[spike_clusters == unit])
        if len_unit_spks > 10_000:
            #           if there is enough ram then the process is much faster
            #           How much RAM is available and how much is needed
            #           number of spikes * number of channels per spike * number of datapoints per chan (including padding)
            #            ram_needed =  len_unit_spks * 384 *182
            #            ram_available = vmem().available
            #            if ram_needed < ram_available:
            #                # if there is enough ram to store all the spikes in memory, FAST
            #                mean_pc, extracted_waves, _, max_chan = wvf_dsmatch(dp,unit, verbose=False, again=False,fast =True, save = True)
            #            else:
            #                # not enough RAM to store all spikes in memory, Slow
            #                mean_pc, extracted_waves, _, max_chan = wvf_dsmatch(dp,unit, verbose=False, again=False,fast =False, save = True)
            mean_pc, extracted_waves, _, max_chan = wvf_dsmatch(
                dp, unit, verbose=False, again=False, save=True
            )
            curr_feat = waveform_features(extracted_waves.T, dp, max_chan, unit)
            curr_feat.insert(0, dp)
            all_ft_list.append(curr_feat)
        else:
            all_ft_list.append([dp] + [0] * 17)
            mean_pc = np.zeros(82)

    return all_ft_list, mean_pc


def temp_wvf_feat(dp, units):
    """
    get all the temporal and wvf features for the given units
    for all unnits:
        # get temp features
        # get wvf features
        # if one of the features is null, the unit is unclassifiable
    make matrix from vectors
    """

    if np.isscalar(units):
        if units == int(units):
            units = [int(units)]
        else:
            raise TypeError(
                "Only ints, list of ints or ints disguised as floats allowed"
            )
    all_feats = []
    for unit in units:
        t_feat = temp_feat(dp, unit)[0]
        w_feat, mean_wvf = wvf_feat(dp, unit)
        all_feat = t_feat + w_feat[0][2:]
        all_feats.append(all_feat)
    #    all_feats = np.array((all_feats))
    return all_feats, mean_wvf


def get_pca_weights(all_acgs_matrix, n_components=5, show=False, titl="WVF"):
    """
    Input: matrix with all the normalised acgs, size: n x m
    Return: matrix n x no_pca_feat
    """

    X = StandardScaler().fit_transform(all_acgs_matrix)
    pca2 = PCA(n_components=n_components)
    projected = pca2.fit_transform(X)
    # show two plots
    #   - first one with the PCA features for the number of components
    #   - second is the variance explained per PC
    if show:

        plt.figure()
        plt.scatter(projected[:, 0], projected[:, 1])
        plt.title(f"Projection of first and second principal components for {titl}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        pca_comp = pca2.components_
        plt.ion()
        plt.figure()
        line_objects = plt.plot(pca_comp.T)
        plt.title(
            f"First {n_components} PC features, {titl} vector len {all_acgs_matrix.shape[1]}"
        )
        plt.legend(line_objects, tuple(np.arange(n_components)), title="PCA features")

        exp_var = np.round(np.sum(pca2.explained_variance_ratio_), 3)

        plt.figure()
        plt.plot(np.arange(n_components), pca2.explained_variance_ratio_)
        plt.xticks(np.arange(n_components), np.arange(n_components))
        plt.yticks(
            pca2.explained_variance_ratio_, np.round(pca2.explained_variance_ratio_, 2)
        )
        plt.title(
            f"{titl} first {n_components} PCs and the variance explained by each,\n overall explain {exp_var} of variance, {titl} vector len is {all_acgs_matrix.shape[1]}"
        )

    return projected, pca2


def gen_ss_cs(recs_fn, show=False):

    # load JSON file with all recording info
    with open(recs_fn) as json_handle:
        recs = json.loads(json_handle.read())

    chan_range = [0, 383]
    # images_folder.mkdir(exist_ok=True)

    # loop over all recordings

    new_json = recs
    for i, ds in list(recs.items()):
        dp = ds["dp"]

        # dp = "/media/npyx/ssd2/ago/optotag/recordings/PkC/18-08-30_YC001_probe1"
        # create the main folder for the images to be saved
        dpnm = get_npyx_memory(dp)
        ss_cs_folder = dpnm / "ss_cs"
        ss_cs_folder.mkdir(exist_ok=True, parents=True)

        #%% Find CCGs with long pause (at least 5ms)
        ctx_units = get_units(dp, quality="good", chan_range=chan_range).tolist()
        cbin = 0.5
        cwin = 100
        pause_dur = 5
        n_consec_bins = int(pause_dur // cbin)
        # Use the same 'name' keyword to ensure that ccg stack is saved to routines memory
        ccg_sig_05_100, ccg_sig_u, sfc = ccg_sig_stack(
            dp,
            ctx_units,
            ctx_units,
            cbin=cbin,
            cwin=cwin,
            name="ctx-ctx",
            p_th=0.02,
            n_consec_bins=n_consec_bins,
            sgn=-1,
            fract_baseline=4.0 / 5,
            W_sd=10,
            test="Poisson_Stark",
            again=False,
            ret_features=True,
        )

        try:
            sfc1 = gen_sfc(
                dp,
                corr_type="cs_pause",
                metric="amp_z",
                cbin=cbin,
                cwin=cwin,
                p_th=0.02,
                n_consec_bins=n_consec_bins,
                fract_baseline=4.0 / 5,
                W_sd=10,
                test="Poisson_Stark",
                again=False,
                againCCG=False,
                units=ctx_units,
                name="ctx-ctx",
            )[0]
            df = pd.DataFrame(columns=["unit", "ss", "cs"])
            for j in sfc1.index:
                cs, ss = (
                    sfc1.loc[j, "uSrc":"uTrg"]
                    if sfc1.loc[j, "t_ms"] > 0
                    else sfc1.loc[j, "uSrc":"uTrg"][::-1]
                )
                plot_ccg(
                    dp,
                    [cs, ss],
                    0.5,
                    100,
                    normalize="Hertz",
                    saveFig=True,
                    as_grid=True,
                    saveDir=ss_cs_folder,
                )
                # c=ccg(dp, [cs, ss], 0.5, 100, normalize='Counts')[0,1,:]
                # StarkAbeles2009_ccg_significance(c, 0.5, 0.02, 10, -1, 10, True, True, True)
                df = df.append({"unit": cs, "ss": 0, "cs": 1}, ignore_index=True)
                df = df.append({"unit": ss, "ss": 1, "cs": 0}, ignore_index=True)

                df.drop_duplicates(inplace=True)
                df.sort_values(by=["unit"], inplace=True)
                df = df.astype(np.int64)

                df.to_csv(Path(ss_cs_folder, "SS-CS-table.csv"))
                ss_list = np.unique(df.unit[df.ss == 1].values)
                cs_list = np.unique(df.unit[df.cs == 1].values)
                new_json[i]["ss"] = ss_list.tolist()
                new_json[i]["cs"] = cs_list.tolist()
        #        breakpoint()
        except AssertionError:
            print(f"Recording {Path(dp).parts[-1]} can't be processed")
        except IndexError:
            print(f"Recording {Path(dp).parts[-1]} can't be processed")

    with open(recs_fn, "w") as json_handle:
        json.dump(new_json, json_handle, indent=2)


def process_all(recs_fn, show=False, again=False):

    """
    Function for processing downloaded files and returning a pandas dataframe
    as the output, with all the 'good units' processed with features
    Input: pass a JSON file as argument, needs to have certain strucutre
    returns: dataframe with all the features for all units

    """

    with open(recs_fn) as json_handle:
        recs = json.loads(json_handle.read())

    # directory for all files

    # proc_data
    # features
    # acg
    # wvf

    all_feat = []
    for i, ds in list(recs.items())[:]:
        print(f"/nProcessing dataset {ds['dp']}...")
        data_root = get_npyx_memory(ds["dp"])
        features_folder = data_root / "features"
        acg_folder = data_root / "acg"
        wvf_folder = data_root / "wvf"

        features_folder.mkdir(exist_ok=True)
        acg_folder.mkdir(exist_ok=True)
        wvf_folder.mkdir(exist_ok=True)
        # loop over all datasets

        # get all the good units
        ds["dp"] = Path(ds["dp"])
        good_units = get_units(ds["dp"], quality="good")

        #    good_units= ds['units']
        #        print(ds['units'])
        spike_clusters = np.load(Path(ds["dp"], "spike_clusters.npy"))

        cell_type = Path(ds["dp"]).parts[-2]
        rec_name = Path(ds["dp"]).parts[-1]
        features_filename = rec_name + "_" + cell_type + ".csv"
        features_file = features_folder / features_filename
        rec_feat = []

        if not features_file.is_file() or again:
            # if the file doesn't yet exist, need to run it to create it
            run_save_features = True
        else:
            #           if the saved file exists, check if the number of lines in there
            #           is the same as the number of good_units

            num_units_saved = pd.read_csv(
                features_file, delimiter=",", header=None
            ).shape[0]
            num_good_units = good_units.shape[0]
            if num_units_saved == num_good_units:
                run_save_features = False
            else:
                run_save_features = True

        within_bounds = in_distance_surface(ds["dp"])

        print("Extracting waveform and temporal features...")
        pbar = tqdm(total=len(good_units))
        for unit in good_units:
            pbar.update(1)
            acg_filename = (
                rec_name
                + "_"
                + "acg"
                + "_"
                + "_"
                + str(unit)
                + "_"
                + cell_type
                + ".npy"
            )
            wvf_filename = (
                rec_name
                + "_"
                + "wvf"
                + "_"
                + "_"
                + str(unit)
                + "_"
                + cell_type
                + ".npy"
            )
            mean_wvf_path = wvf_folder / wvf_filename
            acg_path = acg_folder / acg_filename
            in_bound = unit in within_bounds
            in_bound = np.multiply(in_bound, 1)
            #            print(unit)
            #       get the features for the current unit
            if run_save_features or not mean_wvf_path.is_file():
                curr_feat, mean_wvf = temp_wvf_feat(ds["dp"], unit)
                curr_feat[0].append(in_bound)
                rec_feat.append(curr_feat[0])
                np.save(mean_wvf_path, mean_wvf, allow_pickle=False)
                #        breakpoint()
                #       get the wvf shape for the current unit
                #       by this point all the wvf have been wvf_dsmatched
                #       meaning there is no need to be super precise whether to use fast or not
                #       it will just be retrieved from memory
                #       need to add condition to only run wvf_dsmatch if the file is not yet
                #       in wvf_folder

                #        if not mean_wvf_path.is_file():
                #            np.save(mean_wvf_path, mean_wvf, allow_pickle = False)

                #       get the scaled acg

                sc_acg = scaled_acg(ds["dp"], unit)[0][0]

                if not acg_path.is_file():
                    np.save(acg_path, sc_acg, allow_pickle=False)

            all_feat.append(rec_feat)

            #   save the matrix with the features as a csv for each dataset
            if run_save_features:
                np.savetxt(features_file, rec_feat, delimiter=", ", fmt="% s")

    wvf_files = []
    acg_files = []

    print("Computing PCA features across datasets...")
    for i, ds in list(recs.items())[:]:
        #        data_root = Path('/home/npyx/projects/optotag/proc_data')
        data_root = get_npyx_memory(ds["dp"])
        features_folder = data_root / "features"
        acg_folder = data_root / "acg"
        wvf_folder = data_root / "wvf"

        # loop over all datasets

        # get all the good units

        cell_type = Path(ds["dp"]).parts[-2]
        rec_name = Path(ds["dp"]).parts[-1]
        features_filename = rec_name + "_" + cell_type + ".csv"
        features_file = features_folder / features_filename

        good_units = get_units(ds["dp"], quality="good")

        for unit in good_units:
            acg_filename = acg_folder / str(
                rec_name
                + "_"
                + "acg"
                + "_"
                + "_"
                + str(unit)
                + "_"
                + cell_type
                + ".npy"
            )
            wvf_filename = wvf_folder / str(
                rec_name
                + "_"
                + "wvf"
                + "_"
                + "_"
                + str(unit)
                + "_"
                + cell_type
                + ".npy"
            )

            if (wvf_filename).is_file():
                wvf_files.append(wvf_filename)

            if (acg_filename).is_file():
                acg_files.append(acg_filename)

    wvf_files = np.array(wvf_files)
    acg_files = np.array(acg_files)

    # load all the wvf files to matrix
    load_wvf = []
    for i in wvf_files:
        load_wvf.append(np.load(i, allow_pickle=False))

    all_wvfs = np.vstack(load_wvf)

    # load all the acg files to matrix
    load_acg = []
    for i in acg_files:
        load_acg.append(np.load(i, allow_pickle=False))

    all_acgs = np.vstack(load_acg)

    # find where either of the matrices are 0 or inf valued and filter these vectors
    # from both
    zero_rows_acg = np.where(np.sum(all_acgs, axis=1) == 0)[0].tolist()
    zero_rows_wvf = np.where(np.sum(all_wvfs, axis=1) == 0)[0].tolist()

    inf_rows_acg = np.unique(np.where(np.isinf(all_acgs))[0]).tolist()
    inf_rows_wvf = np.unique(np.where(np.isinf(all_wvfs))[0]).tolist()

    nan_rows_acg = np.unique(np.where(np.isnan(all_acgs))[0]).tolist()
    nan_rows_wvf = np.unique(np.where(np.isnan(all_wvfs))[0]).tolist()

    # get the rows where either the wvf or the acg are zeros or inf values
    excluded_rows = np.unique(
        np.hstack(
            (
                zero_rows_acg,
                zero_rows_wvf,
                inf_rows_acg,
                inf_rows_wvf,
                nan_rows_wvf,
                nan_rows_acg,
            )
        )
    ).astype(np.int32)

    mask = np.ones(all_acgs.shape[0], np.bool)
    mask[excluded_rows] = 0

    masked_acgs = all_acgs[mask]
    masked_wvfs = all_wvfs[mask]

    masked_acg_files = acg_files[mask]
    masked_wvf_files = wvf_files[mask]

    # push all the processed wvf and acg through a PCA
    # and get the first few principal components

    acg_projected, acg_pca = get_pca_weights(masked_acgs, show=show, titl="ACG")

    wvf_projected, wvf_pca = get_pca_weights(masked_wvfs, show=show, titl="WVF")
    # all shifted to -1100, don't look like wvf at all

    # now that we have the pca of some of the projected pcas
    # need to make a new vector that can be then merged back with the dataframe

    ### MANUAL FILTERING here
    ### THERE ARE some clear outlier wvf visible in the pca space that I am removing
    ### I checked these items being filtered out by hand and they can be thrown out

    ### this will need to be reviewed after more data is added
    ## also Ideally I want to find out why these wvf turn out so noisy

    ### Also, should I not be looking at the ACG PCA space as well?
    ### YES, TODO after more files are added

    pca_filter_mask = np.zeros(wvf_projected.shape[0], dtype="bool")
    pca_filter_mask[(wvf_projected[:, 0] < 5) & (wvf_projected[:, 1] > -10)] = 1

    masked_acgs = masked_acgs[pca_filter_mask]
    masked_wvfs = masked_wvfs[pca_filter_mask]

    masked_acg_files = masked_acg_files[pca_filter_mask]
    masked_wvf_files = masked_wvf_files[pca_filter_mask]

    # need to take all the saved features
    # load them in in the order that the wvf_pca is
    # add them to a ddataframe
    # join the dataframe with the wvf and acg PCA
    # remove the rows where the features, wvf or pca are 0
    cols1 = [
        "file",
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
        "skw",
        "neg voltage",
        "neg time",
        "pos voltage",
        "pos time",
        "pos 10-90 time",
        "neg 10-90 time",
        "pos 50%",
        "neg 50%",
        "onset time",
        "onset voltage",
        "wvf duration",
        "peak/trough ratio",
        "recovery slope",
        "repolarisation slope",
        "chan spread",
        "backprop",
        "under2mm",
    ]

    all_feat_df = pd.DataFrame()

    for i, ds in list(recs.items())[:]:
        #        data_root = Path('/home/npyx/projects/optotag/proc_data')
        data_root = get_npyx_memory(ds["dp"])
        features_folder = data_root / "features"
        acg_folder = data_root / "acg"
        wvf_folder = data_root / "wvf"

        # loop over all datasets

        # get all the good units

        cell_type = Path(ds["dp"]).parts[-2]
        rec_name = Path(ds["dp"]).parts[-1]
        features_filename = rec_name + "_" + cell_type + ".csv"
        #    print(features_filename)
        features_file = features_folder / features_filename

        curr_features = pd.read_csv(features_file, delimiter=",", header=None)

        all_feat_df = pd.concat([all_feat_df, curr_features])

    all_feat_df = all_feat_df.set_index(np.arange(0, all_feat_df.shape[0], 1))
    all_feat_df.columns = cols1
    # append the wvf and acg vectors to this dataframe
    # need to only append the PCA to the rows where the acg and wvf are not bad
    # hence need to know where they are good, and then only append the pca of the good ones back

    # create a matrix of zeros, then replace the rows where the pca is good by overlaying a mask

    wvf_pca_zeros = np.zeros((mask.shape[0], wvf_projected.shape[1]))
    wvf_pca_zeros[mask] = wvf_projected

    acg_pca_zeros = np.zeros((mask.shape[0], acg_projected.shape[1]))
    acg_pca_zeros[mask] = acg_projected

    # join these two matrices to the all_feat_df matrix
    # are these in the same order as that dataframe?
    # yes they are now, after making sure the loading was ordered

    # append matrix to dataframe
    # create two new dataframes from it and append to original one

    wvf_pca_df = pd.DataFrame(
        wvf_pca_zeros,
        columns=["wvf_pca0", "wvf_pca1", "wvf_pca2", "wvf_pca3", "wvf_pca4"],
    )
    acg_pca_df = pd.DataFrame(
        acg_pca_zeros,
        columns=["acg_pca0", "acg_pca1", "acg_pca2", "acg_pca3", "acg_pca4"],
    )

    # append acg and wvf df to the main df
    # so finally we have a single df with the wvf, temporal, wvf pca and acg pca
    all_feat_df = pd.concat([all_feat_df, wvf_pca_df, acg_pca_df], axis=1)

    ct_dict = {0: "PkC", 1: "MLI", 2: "GrC", 3: "MFB", 4: "GoC"}
    ct_dict2 = {"PkC": 0, "MLI": 1, "GrC": 2, "MFB": 3, "GoC": 4}
    ct_colors = {
        0: "#755baf",
        1: "#d63737",
        2: "#f1871b",
        3: "#72b257",
        4: "#6197a1",
        5: "#ddd5d3",
    }

    # need to add cell type data
    # create new list of NaN values
    # loop over all the files and if there are ones that

    cell_type_tagged = []

    for i, ds in list(recs.items())[:]:
        #        data_root = Path('/home/npyx/projects/optotag/proc_data')
        data_root = get_npyx_memory(ds["dp"])
        features_folder = data_root / "features"
        acg_folder = data_root / "acg"
        wvf_folder = data_root / "wvf"

        # loop over all datasets

        # get all the good units

        cell_type = Path(ds["dp"]).parts[-2]
        rec_name = Path(ds["dp"]).parts[-1]
        features_filename = rec_name + "_" + cell_type + ".csv"
        features_file = features_folder / features_filename

        good_units = get_units(ds["dp"], quality="good")

        for i in good_units:
            if i in ds["units"]:
                #                cell_type_tagged.append(ct_dict2[ds['ct']])
                cell_type_tagged.append(ds["ct"])
            else:
                cell_type_tagged.append(np.nan)

    cell_type_tagged_df = pd.DataFrame(cell_type_tagged, columns=["optaged"])

    all_feat_df = pd.concat([all_feat_df, cell_type_tagged_df], axis=1)
    return all_feat_df


def add_ss_cs_labels(recs_fn):
    """
    Add the existing labels from JSON file to the df
    Input: takes JSON file with keys for SS and CS categories
    Returns: df with added categories for ss and cs labels in the 'optaged' column
    """

    # load JSON file with all recording info
    with open(recs_fn) as json_handle:
        recs = json.loads(json_handle.read())

    # get the dataframe for all the recording features

    features_df = process_all(recs_fn)

    # look through the recs dictionary
    # find the ss and cs units there
    # add a corresponding label to the features_df.optaged

    for i, ds in list(recs.items()):

        if "ss" in ds:
            for unit in ds["ss"]:
                features_df.loc[
                    (features_df["file"] == ds["dp"]) & (features_df["unit"] == unit),
                    "optaged",
                ] = "ss"

        if "cs" in ds:
            for unit in ds["cs"]:
                features_df.loc[
                    (features_df["file"] == ds["dp"]) & (features_df["unit"] == unit),
                    "optaged",
                ] = "cs"

    return features_df


def filter_df(dfram):
    """
    Return: dataframe with 0 valued rows for the wvf features and temporal features filtered out
    """
    all_conds = (
        np.isclose(np.sum(dfram.iloc[:, -6:-1], axis=1), 0)
        | np.isclose(np.sum(dfram.iloc[:, -11:-6], axis=1), 0)
        | np.isclose(np.sum(dfram.iloc[:, 16:-12], axis=1), 0)
        | np.isclose(np.sum(dfram.iloc[:, 2:17], axis=1), 0)
        | np.array(dfram.iloc[:, :-1].isnull().any(axis=1)).flatten()
        | np.isclose(dfram.iloc[:, -12], 0)
    )

    return dfram[~all_conds]


def feature_extraction(json_path, check_json=True, save_path=None):
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

    with open(json_path) as f:
        json_f = json.load(f)

    DSs = {}
    any_not_found = False
    if check_json:
        for ds in tqdm(
            json_f.values(), desc="Checking provided json file", position=0, leave=True
        ):
            dp = Path(ds["dp"])
            DSs[dp.name] = ds
            if not dp.exists():
                print(f"\033[31;1m{dp} doesn't exist!!\033[0m")
                any_not_found = True
                continue
            units = ds["units"]
            units_m = np.isin(units, get_units(dp))
            if not all(units_m):
                print(
                    f"\033[31;1m{np.array(units)[~units_m]} not found in {dp}!\033[0m"
                )
                any_not_found = True
    if any_not_found:
        raise ValueError("Some files were not found")

    columns = [
        "label",
        "dp",
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
        "relevant_channel",
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
        "recovery_slope",
        "repolarisation_slope",
        "depolarisation_slope",
        "spatial_decay_24um",
        "dendritic_comp_amp",
    ]

    feat_df = pd.DataFrame(columns=columns)

    for ds in tqdm(json_f.values(), desc="Extracting features", position=0, leave=True):
        dp = Path(ds["dp"])
        optolabel = ds["ct"]
        if optolabel == "PkC":
            optolabel = "PkC_ss"
        units = ds["units"]
        ss = ds["ss"]
        cs = ds["cs"]
        for u in units + ss + cs:
            if u in units:
                label = optolabel
            elif u in ss:
                label = "PkC_ss"
            elif u in cs:
                label = "PkC_cs"
            wvf_features = waveform_features_wrap(dp, u)
            tmp_features = temporal_features_wrap(dp, u)
            curr_feat = [label] + wvf_features[:2] + tmp_features[2:] + wvf_features[2:]
            feat_df = feat_df.append(dict(zip(columns, curr_feat)), ignore_index=True)

    if save_path is not None:
        today = date.today().strftime("%b-%d-%Y")
        feat_df.to_csv(f"{save_path}{today}_all_features.csv")

    return feat_df
