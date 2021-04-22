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

import numpy as np
from pathlib import Path
from npyx.utils import peakdetect
from npyx.io import chan_map
from npyx.spk_wvf import wvf, wvf_dsmatch
from npyx.plot import plot_wvf
import matplotlib.pyplot as plt
from scipy.stats import iqr, skew, norm
from scipy.interpolate import interp1d
import scipy.optimize as opt
from itertools import compress
from sklearn.metrics import mean_squared_error
from scipy import ndimage


#############################################
# Waveform features


def mean_start_subtract(waves,first_values = 25, axes = 1):
    """ subtract the mean of the first 25 values from the wvf"""
    if len(waves.shape) == 1:
        waves = waves.reshape(1,-1)
    return waves - np.mean(waves[:,:first_values], axis = axes).reshape(-1,1)

def pos_peak_amp_time(waves, axes = 1):

    """
    Input: array of single vector or matrix
    Return: two vectors with the amp size and the time points
    """

    if len(waves.shape)==1:
        waves = waves.reshape(1,-1)

    return np.max(waves, axis =axes), np.argmax(waves,axis = axes)

def neg_peak_amp_time(waves, axes= 1):

    """
    Input: array of single vector or matrix
    Return: two vectors with the amp size and the time points
    """
    if len(waves.shape)==1:
        waves = waves.reshape(1,-1)

    return np.min(waves, axis =axes), np.argmin(waves, axis = axes)

def cross_times(waves, axis = 1):
    """
    find where the wvf 'crosses' 0 for first and second time

    """

    # crossing between the two peaks
    # find the two max peaks

    if len(waves.shape)==1:
        waves = waves.reshape(1,-1)

    _, pos_t = pos_peak_amp_time(waves)
    _, neg_t = neg_peak_amp_time(waves)

    # find the two time points on each side of the crossing
    crossing1,crossing2 = np.zeros((2,waves.shape[0]))
    for idx, wave in enumerate(waves):
        if neg_t[idx] < pos_t[idx]:
            crossing1[idx] = neg_t[idx] + np.where(np.diff(np.sign(waves[idx,neg_t[idx]:pos_t[idx]])))[0][0]
            crossing2[idx] = crossing1[idx] + 1
        else:
            crossing1[idx] = pos_t[idx] + np.where(np.diff(np.sign(waves[idx,pos_t[idx]:neg_t[idx]])))[0][0]
            crossing2[idx] = crossing1[idx] + 1

    # now we have an array with the two times on each side of the zero cross
    # find the value that is take on the wave and interpolate for the value
    crossing1,crossing2 = crossing1.astype('int16'),crossing2.astype('int16')
    return crossing1[0], waves[0,crossing1+1][0]


def detect_peaks(wvf, outliers_dev=1.5):

    """
    Detect peaks happening on a wvf, with a filter applied in terms of std to
    contrain peaks under threshold
    Steps:
            - find all the possible peaks with a noisy algorithm
            - find the median of the wvf shape and the interquartile range
            - if a peak happens outside 'middle'  band append it to list
    Input: wvf
    Returns: list of peak times and peak values
    """


    detected_peaks = []
    xs = []
    ys = []
    # detect peaks using existing non precise method
    for lk in list(range(3, 60, 3)):
        detected_peaks.append(peakdetect(wvf, lookahead=lk))
    # check if detected_peaks is empty
    # it is usually only empty if the wvf vector is fully flat
    # meaning all the values are the same
    # flatten list of x,y coordinates
    detected_peaks = [x for x in detected_peaks[0] if x != []]
    detected_peaks = [item for items in detected_peaks for item in items]
#    breakpoint()
#   there are some cases where the passed wvf value is blank zeros
#   this is a hardware fault, when a single channel might blank
#   can take care of this by checking if this condition holds and returning -1
    if not detected_peaks:
        #print('No peaks detected, passed -1,-1 as output')
        return -1,-1
    # get the iqr of the wvf and the median
    # define window outisde of which the peaks can happen
    iqr_ = iqr(wvf) * outliers_dev
    _med = np.median(wvf)
    lower = _med - iqr_
    upper = _med + iqr_
    # check for each peak if it is happening outside of middle window
    # if yes, append it
        # convert to np array
        # find the entries where each condition is passed and make a matrix of them
    dp = np.array(detected_peaks)
    filtered_peaks = np.vstack((dp[dp[:,1]<=lower ], dp[upper <=dp[:,1]]))
    # if the largest positive peak is not in the list add it manually
    x_new, y_new = filtered_peaks[:,0], filtered_peaks[:,1]
    if np.max(wvf) not in y_new:
        y_new = np.hstack((y_new, np.max(wvf)))
        x_new = np.hstack(( x_new, np.argmax(wvf)))
    x_sorted = np.argsort(x_new)
    x_new = x_new[x_sorted].astype('int')
    y_new = y_new[x_sorted].astype('int')

    return x_new, y_new

def wvf_duration(wave):
    """
    time between the max positive and min negative peak
    """
    peak_t, peak_v = detect_peaks(wave)


    pos_t = np.argmax(peak_v)
    neg_t = np.argmin(peak_v)
    neg_v = np.min(peak_v)

    # use the last returned value for plotting only

    return np.abs(peak_t[pos_t] - peak_t[neg_t]), peak_t[neg_t], peak_t[pos_t], wave[peak_t[neg_t]]-30

def pt_ratio(wave):
    """
    ratio between the amplitude of the most positive ad most negaitve peak
    simply take the absolute values of the most negative and most positive points
    along the recording and find their ratio
    """

    peak_t, peak_v = detect_peaks(wave)

    neg_id = np.argmin(peak_v)
    neg_v = peak_v[neg_id]

    # get positive peak after the negative peak
    if neg_id + 1 <= peak_t.shape:
        pos_v = peak_v[neg_id + 1]
    else:
        return np.inf

    return pos_v/neg_v

def onset_amp_time(waves, axes = 1):

    """
    Input: single wvf shape
    Return: time when the 5% of the first (negative or pos) peak was reached

    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)
    onset_v = peak_v[0]*0.05

    # now we find the last crossing of this value before the peak
    before_peak = waves[:peak_t[0]]
    crossings = np.where(np.diff(np.sign(before_peak - onset_v)))

    # if there are no crossings, this means the recording was always above
    # or below the value, hence the crossing can be said to have happened
    # before the start of the recording, which we approximate by 0

    if crossings[0].size == 0:
        last_cross = 0
    else:
        last_cross = crossings[-1][0]

    return last_cross, onset_v

def end_amp_time(waves, axis = 1):
    """
    Return: time when the last positive peak first reaches it's 5% value
    """
    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)
    onset_v = peak_v[-1]*0.05

    # now we find the last crossing of this value 
    # get section after the last peak
    after_peak= waves[peak_t[-1]:]
    # now we have list of crossings where 
    crossings = np.where(np.diff(np.sign(after_peak-onset_v)))

    # if there are crossings, get the first of these
    # if there are no crossings, return the shape of the wvf as we know
    # the last crossing happened after our time window
    if len(crossings[0]):
        last_cross = peak_t[-1] + crossings[0][0]
    else:
        last_cross = waves.shape[0]

    return last_cross, onset_v

def pos_10_90(waves, axes = 1):

    """
    Input: wvf
    Return: - times when the pos peak was at 10% and 90%
            - values at 10 and 90%
            - time it took to get from 10-90%
    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)


    min_amp_arg = np.argmin(peak_v)
    min_amp = peak_v[min_amp_arg]
    curr_peak_t = peak_t[min_amp_arg+1]
    prev_peak_t = peak_t[min_amp_arg]
    max_amp = peak_v[min_amp_arg+1]


    perc_10 = 0.1 * max_amp
    perc_90 = 0.9 * max_amp

    # now need to find where the before and after cross two points happened
    # get the section of the slope we need
    upslope = waves[prev_peak_t:curr_peak_t]
    # get the points where the wave reaches the percentile values
    cross_10 = prev_peak_t + np.where(np.diff(np.sign(upslope - perc_10)))[0][0]
    cross_90 = prev_peak_t + np.where(np.diff(np.sign(upslope - perc_90)))[0][0]

    crosses_10 = np.array([cross_10, cross_10+1])
    # ensure the value is positive for the first crossing
    crosses_10 = crosses_10[waves[crosses_10]> 0]
    crosses_90 = np.array([cross_90, cross_90+1])
    # from these crossing points find the closer one to the value
    close_10 = crosses_10[np.argmin(np.abs(waves[crosses_10]) - perc_10)]
    close_90 = crosses_90[np.argmin(np.abs(waves[crosses_90]) - perc_90)]

    return np.array([close_10, close_90]), np.array([waves[close_10], waves[close_90]]), close_90 - close_10

def pos_decay_t(wave):
    """
    time between when the positive peak happened and  when 10% happened
    """
    # get the wvf peaks
    peak_t, peak_v = detect_peaks(wave)

    pos_t = peak_t[np.argmax(peak_v)]
    perc_10_t = pos_10_90(wave)[0][0]

    return pos_t-perc_10_t



def fall_time(waves):
    """
    Calculate time it took to go from negative 10%-90% of peak
    Input: wvf
    Return: - times when the neg peak was at 10% and 90%
            - values at 10 and 90%
            - time it took to get from 10-90%

    """
    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)

    min_amp_arg= np.argmin(peak_v)
    min_amp= peak_v[min_amp_arg]
    curr_peak_t = peak_t[min_amp_arg]
    prev_peak_t = onset_amp_time(waves)[0]

    perc_10 = 0.1 * min_amp
    perc_90 = 0.9 * min_amp

   # now need to find where the before and after cross two points happened
    # get the section of the slope we need
    downslope = waves[prev_peak_t:curr_peak_t+1]
    # get the points where the wave reaches the percentile values
    cross_10 = prev_peak_t + np.where(np.diff(np.sign(downslope - perc_10)))[0][0]
    cross_90 = prev_peak_t + np.where(np.diff(np.sign(downslope - perc_90)))[0][0]
    #print(cross_10, cross_90)
    crosses_10 = np.array([cross_10, cross_10+1])
    crosses_90 = np.array([cross_90, cross_90+1])
    # from these crossing points find the closer one to the value
    close_10 = crosses_10[np.argmin(np.abs(waves[crosses_10]) - perc_10)]
    close_90 = crosses_90[np.argmin(np.abs(waves[crosses_90]) - perc_90)]

    return np.array([close_10, close_90]), np.array([waves[close_10], waves[close_90]]), close_90 - close_10

def neg_decay_t(wave):

    "negative decay time: time between the negative peak amplitude, and when it was at 10%"

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(wave)

    neg_t = peak_t[np.argmin(peak_v)]
    perc_10_t = fall_time(wave)[0][0]

    return neg_t-perc_10_t

def pos_half_width(waves, axes = 1):
    """
    Give the the half width time for the positive peak
    Input: wvf
    Return: - start time
            - end time
            - 50% value
            - duration
    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)

    max_amp_arg = np.argmax(peak_v)
    max_amp= peak_v[max_amp_arg]
    curr_peak_t = peak_t[max_amp_arg]
    prev_peak_t = peak_t[max_amp_arg-1]

    perc_50 = 0.5 * max_amp
    # find the interval we need start from 0 crossing to end

    # start looking for the crossing from the onset.  Maybe faster without it?
    # also need this in case there are other crossings that might happen at
    # other times of the recording
    start_interval = cross_times(waves)[0].astype(int)
    end_interval = end_amp_time(waves)[0]
    current_slope = waves[start_interval:end_interval]
    # get the real time when the crossings happened, not just relative time
    cross_start = start_interval + np.where(np.diff(np.sign(current_slope- perc_50)))[0][0]
    cross_end = start_interval + np.where(np.diff(np.sign(current_slope- perc_50)))[0][-1]

    return cross_start, cross_end,perc_50, cross_end - cross_start

def neg_half_width(waves):
    """
    Give the the half width time for the neg peak
    Input: wvf
    Return: - start time
            - end time
            - 50% value
            - duration
    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)

    min_amp_arg = np.argmin(peak_v)
    min_amp= peak_v[min_amp_arg]
    curr_peak_t = peak_t[min_amp_arg]
    prev_peak_t = onset_amp_time(waves)[0]

    perc_50 = 0.5 * min_amp
    # find the interval we need start from 0 crossing to end

    start_interval = prev_peak_t
    end_interval = cross_times(waves)[0].astype(int)
    current_slope = waves[start_interval:end_interval]
    cross_start, cross_end = start_interval + np.where(np.diff(np.sign(current_slope- perc_50)))[0]

    return cross_start, cross_end,perc_50, cross_end - cross_start



def tau_end_slope(waves, axis = 1):
    """
    find the last peak
    get 30 values after last peak
    fit exponential to these values
    get parameter of expoential
    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)
    last_peak = peak_t[-1]
    slope_obs = waves[last_peak+100:last_peak+1600]
    slope_correct = slope_obs - np.min(slope_obs)+1
    logged = np.log(slope_correct)
#    print(slope_obs.shape, logged.shape)
    K, A_log = np.polyfit(np.linspace(0,1499,1500), logged, 1)
    A = np.exp(A_log)
    fit = slope_obs[-1]+A*np.exp(K*np.linspace(0,1499,1500))-1

    mse = mean_squared_error(slope_obs,fit)

    return fit, mse, 1/K

def interp_wave(wave, multi = 100, axis=-1):
    """
    linearly upsample waves so there are overall multi times more points
    We need this for more precise estimates to when time points happened
    Input: wvf and upsampling multiplier
    Retun: upsampled wvf
    """

    wvf_len = wave.shape[1]
    interp_fn = interp1d(np.linspace(0,wvf_len-1, wvf_len), wave,axis=axis)

    new_wave = interp_fn(np.linspace(0,wvf_len-1,wvf_len*multi))

    return new_wave

def repol_slope(waves):
    """
    regression fitted to the first 30 observations from the last peak

    - find the negative peak
    - fit regression to the next  10% of dots between this and the next peak
    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)

    neg_t = peak_t[np.argmin(peak_v)].astype(int)
    neg_v = np.min(peak_v).astype(int)

    # find number of points between negative and positive peaks

    pos_t = peak_t[np.argmax(peak_v)]

    all_dots_peak = np.abs(pos_t - neg_t)
    dots_pos_neg_20 = (0.3*all_dots_peak).astype(int)
    fit_slope = waves[neg_t:neg_t+dots_pos_neg_20]
    coeff = np.polyfit(np.linspace(0,dots_pos_neg_20-1, dots_pos_neg_20), fit_slope, deg=1)
    # fit a slope with the new parameters
    # all data points between two peaks
    # multiply by first coeff and add the second one
    return coeff, coeff[0]*np.linspace(0,dots_pos_neg_20,dots_pos_neg_20+1)+coeff[1], neg_t, neg_v


def recovery_slope(waves):
    """
    fit a regression to observations after the last pos peak
    """

    # get the wvf peaks
    peak_t, peak_v = detect_peaks(waves)

    pos_t = peak_t[np.argmax(peak_v)].astype(int)
    pos_v = np.max(peak_v).astype(int)

    # find number of points between negative and positive peaks


    all_dots_peak = waves.shape[0] - pos_t
    dots_pos_neg_20 = (0.2*all_dots_peak).astype(int)
    fit_slope = waves[pos_t:pos_t+dots_pos_neg_20]

    coeff = np.polyfit(np.linspace(0,dots_pos_neg_20-1, dots_pos_neg_20), fit_slope, deg=1)
    # fit a slope with the new parameters
    # all data points between two peaks
    # multiply by first coeff and add the second one
    return coeff, coeff[0]*np.linspace(0,dots_pos_neg_20,dots_pos_neg_20+1)+coeff[1], pos_t, pos_v
def previous_peak(waves, chan_path, unit):
    """
    takes as input 384x82 matrix

    - take a matrix of waves along the probe
    - find the peak chan
    - find channels +- 20 from the peak chan
    - get the value of the previous peak in standard deviations compared
        to baseline period at the start of the recording
    """

    # loop through each wave
    # detect_peaks
    # find most negative peak
    # check if there is a peak before the most negative one
    max_chan_path = list(Path(chan_path/'routinesMemory').glob(f'dsm_{unit}_peakchan*'))[0]
    max_chan = int(np.load(max_chan_path))
    if max_chan < 19:
        bounds = (0, max_chan+21)
    elif max_chan > 365:
        bounds = (max_chan-20, 384)
    else:
        bounds = (max_chan-20, max_chan+21)

    bound_waves = waves[bounds[0]:bounds[1]+1]
#    bound_waves = waves.T[bounds[0]:bounds[1]+1]

    pbp = np.zeros((bound_waves.shape[0]))
    # first n values to calculate mean and std
    no_baseline_values = bound_waves.shape[1]//4
    #breakpoint()

    # get the max_chan waveform
    max_wave = waves[max_chan]
    # get the mean of the start of this wave and the std at the start
    # these measurements will be use for getting the backprop values
    mean_s = np.mean(max_wave[np.r_[0:no_baseline_values, -no_baseline_values:-1]])
    std_s = np.std(max_wave[np.r_[0:no_baseline_values,-no_baseline_values:-1]])
    for ids, wav in enumerate(bound_waves):
        peak_t, peak_v  = detect_peaks(wav)
#       check if the values passed from detect_peaks are useful
#       in some cases the wvf passed to detect_peaks is flat, this is due to hardware error
#       these cases have to be taken care of manually, hence the following if statements
        if (isinstance(peak_t, int)) & (isinstance(peak_v,int)):
            pbp[ids] = 0
        else:
         #   breakpoint()
            peak_v = (peak_v-mean_s)/std_s
            neg_t = np.argmin(peak_v)
            neg_v = peak_t[neg_t]
            # if the time when the negative peak happened is not 0
            # meaning this is not the first peak that happed
            # and hence there might be a 'previous peak'
            if neg_t != 0:
                pbp[ids] = np.abs(peak_v[neg_t -1])
            else:
                pbp[ids] =0
  #  breakpoint()
    # find the max amd argmax of pbp
    argmax_pbp = np.argmax(pbp)
    max_pbp = pbp[argmax_pbp]
    # get the values before the peak
    before_max = pbp[:argmax_pbp]
    # get the values after
    after_max = pbp[argmax_pbp+1:]
    # find the crossing points with the fn below

    if len(before_max) > 0 and len(after_max) > 0:
        before_half_chan = np.where(before_max < max_pbp * 0.5)[0][-1]
        after_half_chan = np.where(after_max < max_pbp * 0.5)[0][0]
#        print(before_half_chan, after_half_chan, argmax_pbp)
        spread = argmax_pbp - before_half_chan + after_half_chan - 1
    else:
        spread = 0
#    breakpoint()

    # quantify the backpropagation
    # binary feature, if the peak is larger than some value
    # continuous feature, purely the value of the peak
    # what can I normalise by?
    # if there are more than 3 channels where the values are over 50% of max
    # and if there are 

#    if max_pbp > backprop_std:
#        yes_backp = 1
#    else:
#        yes_backp = 0

    return pbp, spread, max_pbp

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
    middle_value = int(mean_waves.shape[0]*40/82)
    mid_range = int(mean_waves.shape[0]*15/82)

    truncated_waves = np.zeros_like(mean_waves.T)
    loc_min_val = np.argmin(mean_waves,axis = 0)
    # truncate the waves so we can look at the amplitudes of the neg and next peak
#    for idx, row in enumerate(mean_waves.T):
#        truncated_waves[idx, loc_min_val[idx]:] = row[loc_min_val[idx]:]
    for idx, row in enumerate(mean_waves.T):
        truncated_waves[idx, middle_value:middle_value + mid_range] = row[middle_value:middle_value + mid_range]
    truncated_waves = truncated_waves.T
    breakpoint()
    return np.argmax(np.ptp(truncated_waves,axis=0)), np.max(np.ptp(truncated_waves,axis=0)), np.ptp(truncated_waves, axis = 0)

def chan_dist(chan, chanmap):
    """
    given a chan and a chan map, will return 384 long vector of distances
    """
    current_loc = chanmap[chan,1:]
    coordinates = chanmap[:,1:] - current_loc
    # first we find all the distances for each channel
    all_dist = np.sqrt(np.sum(np.square(coordinates), axis = 1))
    # next the positive or a negative value is assigned to each
    # depending on which side of the peak chan it is
    sign_mask = np.ones((384))
    sign_mask[:chan] = -1
    breakpoint()
    return all_dist * sign_mask



def chan_spread(all_wav, chan_path, unit):
    """
    - take a 82*384 matrix
    - find the peak chan by looking at the amplitude differnce on all chans
    - calculate distance from peak chan to all other chans, pythagoras, chan_map
    - find the max amp of all chans
    - make a plot with the distance from peak chan on x axis and amplitude on y
    -
    """

    # find the most negative peak and the peak after that to
    # get the distance between peak and trough

    _,_, p2p = consecutive_peaks_amp(all_wav.T)

    # search for the file that has the given peak chan 
    max_chan_path = list(Path(chan_path/'routinesMemory').glob(f'dsm_{unit}_peakchan*'))[0]
    max_chan = int(np.load(max_chan_path))

    chanmap = chan_map(chan_path)
    # find the distance of all channels from peak chan
    dists = chan_dist(max_chan, chanmap)

    dist_p2p = np.vstack((dists, p2p)).T

    # I want a distribution of amplitudes with the x axis being distance
    # so sort the matrix so the first column is sorted
    # plot the second column 
    sort_dist = np.argsort(dists)
    sort_dist_p2p = np.vstack((dists[sort_dist], p2p[sort_dist])).T
    breakpoint()
    if max_chan < 21:
        bounds = (0, max_chan+21)
    elif max_chan > 363:
        bounds = (max_chan-20, 384)
    else:
        bounds = (max_chan-20, max_chan+21)

    bound_dist = dists[bounds[0]:bounds[1]+1]
    bound_p2p = p2p[bounds[0]:bounds[1]+1]
    bound_dist_p2p = dist_p2p[bounds[0]:bounds[1]+1]
    sort_dist_p2p = sort_dist_p2p[bounds[0]:bounds[1]+1]
    spread = np.var(bound_p2p)
    # find the number of channels between the two half widths

    argmax_peak = np.argmax(sort_dist_p2p[:,1])
    max_peak = np.max(sort_dist_p2p[:,1])
    before_max = sort_dist_p2p[:,1][:argmax_peak+1]
    after_max  = sort_dist_p2p[:,1][argmax_peak:]
    before_boolean = np.where(before_max < max_peak * 0.5)
    after_boolean = np.where(after_max < max_peak * 0.5)
    if len(before_boolean[0]) > 0 or len(after_boolean[0]) > 0:
    #    print(before_max < max_peak * 0.5)
        if len(before_boolean[0]) > 0:
            before_half_chan = before_boolean[0][-1]
        else:
            before_half_chan = 0
        if len(after_boolean[0]) > 0:
            after_half_chan = after_boolean[0][0]
        else:
            after_half_chan = 0
        spread = argmax_peak - before_half_chan + after_half_chan - 1
    else:
        spread = 0
    return max_chan, dists, p2p, dist_p2p, sort_dist_p2p, spread


def wvf_shape(wave):
    # check if the wvf shape is one that we need
    # either first large negative peak and then a smaller pos
    # first pos peak, then largest neg peak and small positive peak
    # largest first positive peak and smaller neg peak is not good

    peak_t, peak_v = detect_peaks(wave)
   # start_mean, end_mean = True, True
    peak_trough, most_neg, start_thres, end_thres, peak_order = (False,) * 5

    # first get the length of the waveform so we can determine length of the end

    if wave.shape[0] == 82:
        end_section = 10
    elif wave.shape[0] == 8200:
        end_section = 1000
    else:
        end_section = int(wave.shape[0] * 10/82)
    #TODO review this hard condition
    if np.max(peak_v) - np.min(peak_v) > 15:
        peak_trough = True

    if np.abs(np.min(peak_v)) > np.max(peak_v):
        most_neg = True

    if np.all(np.abs(wave[:end_section]) < 20):
        start_thres = True

    #if np.all(np.abs(wave[-1000:]) < 30):
    end_thres = True

    if np.argmax(peak_v) > np.argmin(peak_v):
        peak_order = True

    #if np.abs(np.mean(wave[:20])) < 2.5:
     #   start_mean = True

    #if np.abs(np.mean(wave[-20:])) < 2.5:
     #   end_mean = True

    return peak_trough and most_neg and start_thres and end_thres and peak_order



def chan_plot(waves, peak_chan, n_chans):

    '''
    find the peak chan and the n_chans on either side
    plot the mean wvf of the channels
    return object that can be plotted with
    '''

    plot_waves = waves[peak_chan-n_chans:peak_chan+n_chans+2]
    print(plot_waves.shape)
    fig,ax  = plt.subplots(n_chans+1,2)

    for i in range(plot_waves.shape[0]):
        # find where to have the plot depending on 

        y_axis = (i//2)%(plot_waves.shape[0]+1 )
        x_axis = i%2
        ax[y_axis, x_axis].plot(plot_waves[i])
    #   plt.fill_between(np.arange(0,82), wave_form-deviations,wave_form+deviations, color = 'paleturquoise')
        #ax[y_axis, x_axis].set_title(f'{peak_chan-n_chans+i}', y= 1)
        ax[y_axis, x_axis].text(0,0,f'{peak_chan-n_chans+i}')
        ax[y_axis, x_axis].set_xticks([])
        ax[y_axis, x_axis].set_yticks([])
        #ax[y_axis, x_axis].set_xlim(500)
        ax[y_axis, x_axis].set_ylim(np.min(plot_waves),np.max(plot_waves))

    ax[10,0].set_xticks([0,40,82])
    ax[10,0].set_xticklabels([-1.365, 0, 1.365])
    ax[10,0].set_yticks([np.min(plot_waves),0, np.max(plot_waves)])
    ax[10,0].set_yticklabels([np.round(np.min(plot_waves),2),0,np.round( np.max(plot_waves),2)])
    ax[10,0].set_xlabel('ms')
    ax[10,0].set_ylabel(r'$\mu$V')

    fig.suptitle(f'{n_chans} chans on either side of the peak chan {peak_chan}')
    fig.set_size_inches(10,20)
    #fig.tight_layout()
    #fig.subplots_adjust(top =0.9)
    plt.show()






####################################################
# Temporal features

def gaussian_cut(x, a, mu, sigma, x_cut):
    g = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    g[x < x_cut] = 0
    return g


def curve_fit_(x, num, p1):
    pop_t = opt.curve_fit(gaussian_cut, x, num, p1, maxfev=10000)
    return pop_t




def ampli_fit_gaussian_cut(x, n_bins):
    # inputs: vector we want to estimate where the missing values start
    # inputs: number of bins
    # returns: ???

    # make inputs into numpy array
    a = np.asarray(x, dtype='float64')
    # get a histogram of the data, with the  number of entries in each bin
    # and the bin edges
    num, bins = np.histogram(a, bins=n_bins)
    # bin bottom bracket with the most entries
    # this can return more than one value for the mode
    mode_seed = bins[np.where(num == max(num))]
    #mode_seed = bins[np.argmax(num)]
    # find the bin width 
    bin_steps = np.diff(bins[0:2])[0]
    #get the mean values of each bin
    x = bins[0:len(bins) - 1] + bin_steps / 2
    # get the value of the start of the first bin
    next_low_bin = x[0] - bin_steps
    #next_low = bins[0] - bin_steps/2

    # now we make more bins so they go all the way to 0
    add_points = np.arange(start =  0, stop=next_low_bin, step=bin_steps)
    #add_points = np.arange(start=next_low_bin, stop=0, step=-bin_steps)
    #add_points = np.flipud(add_points)
    # concatenate the new bin midpoints with the old ones
    x = np.concatenate([add_points, x])
    zeros = np.zeros((len(add_points), 1))
    zeros = zeros.reshape(len(zeros), )
    # concatenate the old number of bin elements with 0 for the new bins
    num = np.concatenate([zeros, num])

    # if there is  more than one mode of the  distribution, mean them  
    if len(mode_seed) > 1:
        mode_seed = np.mean(mode_seed)

    # return: max, new mod, std for non nan values, and first percentile
    p0 = [np.max(num), mode_seed, np.nanstd(a), np.percentile(a, 1)]
    p0 = np.asarray(p0, dtype='float64')

    # Curve fit
    popt = curve_fit_(x, num, p0)
    p0 = popt[0]

    return x, p0


def gaussian_amp_est(x, n_bins):
#    breakpoint()
    try:
        x1, p0 = ampli_fit_gaussian_cut(x, n_bins)
        n_fit = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], x_cut=p0[3])
        min_amp = p0[3]
        n_fit_no_cut = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], x_cut=0)
        percent_missing = int(round(100 * norm.cdf((min_amp - p0[1]) / p0[2]), 0))

    except RuntimeError:
        x1, p0, min_amp, n_fit, n_fit_no_cut, percent_missing = None, None, None, None, None, np.nan

    return x1, p0, min_amp, n_fit, n_fit_no_cut, percent_missing


def estimate_bins(x, rule):

    n = len(x)
    maxi = max(x)
    mini = min(x)

    # Freedman-Diaconis rule
    if rule == 'Fd':

        data = np.asarray(x, dtype=np.float_)
        iqr_ = iqr(data, scale="raw", nan_policy="omit")
        n = data.size
        bw = (2 * iqr_) / np.power(n, 1 / 3)
        datmin= min(data)
        datmax = max(data)
        datrng = datmax - datmin
        bins = int(datrng/bw + 1)

        # q75, q25 = np.percentile(x, [75, 25])
        # iqr_ = q75 - q25
        # print('iqr', iqr_)
        # h = 2 * iqr_ * (n ** (-1/3))
        # print('h', h)
        # b = int(round((maxi-mini)/h, 0))

        return bins

    # Square-root choice
    elif rule == 'Sqrt':
        b = int(np.sqrt(n))
        return b


def compute_isi(train, quantile = 0.02):

    """
    Input: spike times in samples and returns ISI of spikes that pass through
        exclusion quantile
    Operations: if quantile is given the given quantile from the ISI will be
                discarded, returning the spikes
    Returns: isi in s """

    diffs = np.diff(train)
    isi_ = np.asarray(diffs, dtype='float64')
    if quantile:
        isi_ = isi_[(isi_ >= np.quantile(isi_, quantile)) & (isi_ <= np.quantile(isi_, 1 - quantile))]
        return isi_/30_000
    else:
        return isi_/30_000



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
    histy, histx = num * 1. / np.sum(num), bins[1:]
    sigma = (1. / 6) * np.std(histy)
    Pisi = ndimage.gaussian_filter1d(histy, sigma)

#    except ValueError:
#        binsLog = 200
#        num, bins = np.histogram(isint, binsLog)
#        histy, histx = num * 1. / np.sum(num), bins[1:]
#        sigma = (1. / 6) * np.std(histy)
#        Pisi = ndimage.gaussian_filter1d(histy, sigma)

    # Remove 0 values
    non0vals = (Pisi > 0)
    Pisi = Pisi[non0vals]

    entropy = 0

    for i in range(len(Pisi)):
        entropy += -Pisi[i] * np.log2(Pisi[i])

    return entropy



def compute_isi_features(isint):

    # Instantaneous frequencies were calculated for each interspike interval as the reciprocal of the isi;
    # mean instantaneous frequency as the arithmetic mean of all values.
    mifr = np.mean(1./isint)

    # Median inter-spike interval distribution
    # medISI = np.median(isint)
    med_isi = np.median(isint)

    # Mode of inter-spike interval distribution
    # Why these values for the bins?
    num, bins = np.histogram(isint, bins=np.linspace(0, 0.1, 100))
    mode_isi = bins[np.argmax(num)]

    # Burstiness of firing: 5th percentile of inter-spike interval distribution
    prct5ISI = np.percentile(isint, 5)

    # Entropy of inter-spike interval distribution
    entropy = compute_entropy_dorval(isint)

    # Average coefficient of variation for a sequence of 2 ISIs
    # Relative difference of adjacent ISIs
    CV2_mean = np.mean(2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1]))
    CV2_median = np.median(2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1]))  # (Holt et al., 1996)

    # Coefficient of variation
    # Checked!
    CV = np.std(isint) / np.mean(isint)

    # Instantaneous irregularity >> equivalent to the difference of the log ISIs
    # Checked!
    IR = np.mean(np.abs(np.log(isint[1:] / isint[:-1])))

    # # Local Variation
    # Checked!
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    Lv = 3 * np.mean(np.ones((len(isint) - 1)) - (4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2))

    # Revised Local Variation, with R the refractory period in the same unit as isint
    # Checked!
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2701610/pdf/pcbi.1000433.pdf
    R = 0.8  # ms
    LvR = 3 * np.mean((np.ones((len(isint) - 1)) - (4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2)) *
                      (np.ones((len(isint) - 1)) + (4 * R / (isint[:-1] + isint[1:]))))

    # Coefficient of variation of the log ISIs
    # Checked!
    LcV = np.std(np.log10(isint)) * 1. / np.mean(np.log10(isint))

    # Geometric average of the rescaled cross correlation of ISIs
    # Checked!
    SI = -np.mean(0.5 * np.log10((4 * isint[:-1] * isint[1:]) / ((isint[:-1] + isint[1:]) ** 2)))

    # Skewness of the inter-spikes intervals distribution
    # Checked!
    SKW = skew(isint)

    # Entropy not included
    return mifr, med_isi, mode_isi, prct5ISI, entropy, CV2_mean, CV2_median, CV, IR, Lv, LvR, LcV, SI, SKW


def waveform_features(all_waves, dpath,  peak_chan, unit):
    # return: list of all features for a unit

    # get the negative peak
    best_wave = all_waves[peak_chan].reshape(1,-1)
    best_wave -= np.mean(best_wave[:20])
    best_wave = interp_wave(best_wave).reshape(-1)
    peak_t, peak_v = detect_peaks(best_wave)
    neg_id = np.argmin(peak_v)
    neg_v = peak_v[neg_id]
    neg_t = peak_t[neg_id]

    if not wvf_shape(best_wave):
        return np.zeros(17)
    # get positive peak
    if neg_id + 1 <= peak_t.shape:
        pos_v = peak_v[neg_id + 1]
        pos_t = peak_t[neg_id + 1]
    # pos 10-90
    _,_,pos_10_90_t = pos_10_90(best_wave)

    # neg 10-90
    _,_, neg_10_90_t = fall_time(best_wave)

    # pos half width
    _,_,_,pos50 = pos_half_width(best_wave)

    # neg half width
    _,_,_, neg50 = neg_half_width(best_wave)

    # onset amp and time
    onset_t, onset_amp = onset_amp_time(best_wave)

    # wvf duration
    wvfd,_,_,_ = wvf_duration(best_wave)

    # pt ratio
    ptr = pt_ratio(best_wave)

    # rec slope
    coeff1, _, _, _ = recovery_slope(best_wave)

    # repol slope
    coeff2, _, _, _ = repol_slope(best_wave)

    # chan spread
    _,_,_,_,_, chans = chan_spread(all_waves, dpath, unit)

    #backprop spread
    _, bp_spread, backp_max =  previous_peak(all_waves, dpath, unit)

    ret_arr = [unit, neg_v, neg_t, pos_v, pos_t,pos_10_90_t,
        neg_10_90_t, pos50, neg50, onset_t, onset_amp, wvfd, ptr, coeff1[0], coeff2[0], chans, backp_max]
    return ret_arr

def plot_all(one_wave):
#    one_wave = one_wave[0]

    peaks = detect_peaks(one_wave)
    onset = onset_amp_time(one_wave)
    offset = end_amp_time(one_wave)
    positive_line = pos_10_90(one_wave)
    negative_line = fall_time(one_wave)
    zero_time = cross_times(one_wave)
    pos_hw = pos_half_width(one_wave)
    neg_hw = neg_half_width(one_wave)
    end_slope, mse_fit, tau = tau_end_slope(one_wave)
    rec_slope = recovery_slope(one_wave)
    rep_slope = repol_slope(one_wave)
    wvf_dur = wvf_duration(one_wave)
    ptrat  = pt_ratio(one_wave)

    print(tau)
    print(ptrat)
    plt.ion()
    fig, ax = plt.subplots(1)
    ax.plot(one_wave)
    ax.plot(peaks[0], peaks[1], 'rx')
    ax.plot(onset[0], onset[1], 'rx')
    ax.plot(offset[0], offset[1], 'rx')
    ax.plot(positive_line[0], positive_line[1],linewidth=3)
    ax.plot(negative_line[0], negative_line[1], linewidth = 3)
    ax.plot(zero_time[0], 0, 'rx')
    ax.plot(np.linspace(peaks[0][-1]+100, peaks[0][-1]+1600,1500 ), end_slope,linewidth=3)
    ax.plot([pos_hw[0], pos_hw[1]],[pos_hw[2], pos_hw[2]] , linewidth=3)
    ax.plot([neg_hw[0], neg_hw[1]],[neg_hw[2], neg_hw[2]], linewidth=3)
    ax.plot(np.linspace(wvf_dur[1], wvf_dur[2], wvf_dur[0]+1), np.linspace(wvf_dur[3], wvf_dur[3], wvf_dur[0]+1))
    ax.text(wvf_dur[2]+100, wvf_dur[3],f'wvf duration: {np.round(wvf_dur[0]/3000, 2)}' )
    ax.text(rec_slope[2], rec_slope[3]+30, f"rec slope: {np.round(rec_slope[0][0],2)}")
    ax.text(rep_slope[2]+300, rep_slope[3], f"rep slope: {np.round(rep_slope[0][0],2)}")
    ax.text(peaks[0][-1]+500, peaks[1][-1]-15, f"MSE of fit is {np.round(mse_fit,2)} \n tau: {np.round(tau, 2)}")
    ax.text(zero_time[0]+500,-30,f"Peak/trough ration: {np.abs(np.round(ptrat,2))}")
    ax.set_xlabel('ms')
    ax.set_ylabel(r'$\mu$ V')
    ax.set_xticks([0,4000, 8200])
    ax.set_xticklabels([-1.365, 0, 1.365])
    fig.suptitle('Wvf features for an MLI unit ')



def chan_spread_bp_plot(dp, unit):
    """
    Generates a plot with number of channels and the channels on left
    and backprop and chan spread on the right
    Input: datapath and unit (drift and shift matched datasets for now)
    Returns: plot
    """
    curr_fil = dp/'routinesMemory'/f'dsm_{i}_all_waves_100-82_regular_False300-FalseNone-FalseNone.npy'
    if curr_fil.is_file():

        all_waves_unit_x = np.load(curr_fil)
        backp, bp_spread, true_bp =  previous_peak(all_waves_unit_x.T, dpath21, i)
     #   print(i, true_bp, bp_spread)
        csp_x = chan_spread(all_waves_unit_x.T,dpath21, i)
        ct = str(dpath2).split('/')[7]
        peak_chan = csp_x[0]

    #    recording  = str(dpath2).split('/')[8]
        fig,ax = plt.subplots(3,1)
        if backp.shape[0] == 42:
    #        ax[0].plot(np.linspace(-20,21,42).astype('int'),backp)

            ax[0].plot(csp_x[4][:,0],backp)
        else:
            n_chans = backp.shape[0] 
            ax[0].plot(np.linspace(0, n_chans-1, n_chans).astype('int'), backp)
        #ax[0].set_xlabel('Distance from peak chan (micrometer)')
        ax[0].set_ylabel('Z-scored value \n  of the previous peak')
        #ax[0].axvline(21, color = 'red')
        ax[0].title.set_text(f'Z-scored value of previous peak to most negative peak, \n 20 chan from peak chan in each direction {ct} unit {i}')

        ax[1].plot(csp_x[4][:,0],csp_x[4][:,1])
        #ax[1].plot(np.linspace(-20,21,42).astype('int'),csp_x[4][:,1])
        ax[1].set_xlabel('Distance from peak chan (micrometer)')
        ax[1].set_ylabel('Amplitude of  negative \n to positive peak')
        ax[1].title.set_text(f'Amplitude of most negative peak to next peak for all chans {csp_x[5]}')

        t_all = np.round(82/30, 2)
        t_xaxis = np.linspace(-t_all/2, t_all, 82)
        t_xaxis = np.hstack((np.linspace(-t_all/2,0,41),np.linspace(t_all/82, t_all/2+t_all/82,41)))
        ax[2].plot(t_xaxis, all_waves_unit_x.T[csp_x[0]])
        ax[2].set_xlabel('ms')
        ax[2].set_ylabel(r'$\mu$V')

        ax[2].title.set_text(f'Wvf of Peak chan {csp_x[0]} ')
        fig.tight_layout()


