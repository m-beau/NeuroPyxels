# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os.path as op

from IPython.core.debugger import set_trace as breakpoint

opj=op.join
from pathlib import Path, PosixPath, WindowsPath

from joblib import Memory

cachedir = Path(op.expanduser("~")) / ".NeuroPyxels"
cache_memory = Memory(cachedir, verbose=0)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import iqr, norm

from npyx.gl import check_periods, get_npyx_memory, get_units
from npyx.inout import read_metadata
from npyx.utils import (
    assert_float,
    assert_int,
    docstring_decorator,
    cache_validation_again,
    npa,
    smooth,
    thresh_consec,
)

def ids(dp, unit, sav=True, verbose=False, periods='all', again=False, enforced_rp=-1):
    '''
    ********
    routine from routines_spikes
    computes spike indices (1, Nspikes) - int64, in samples
    ********

    - dp (string): DataPath to the Neuropixels dataset.
    - u (int): unit index
    - ret (bool - default False): if True, train returned by the routine.
      If False, by definition of the routine, drawn to global namespace.
    - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
    - periods = 'all' or [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
    - again: boolean, if True recomputes data from source files without checking routines memory.
    - enforced_rp: float, enforced refractory period - if 2 spikes are separated by less than enforced_rp ms, the first one only is kept.
                   By default -1, does not remove any spikes (unlike trn).
    '''

    dp = Path(dp)
    assert unit in get_units(dp), 'WARNING unit {} not found in dataset {}!'.format(unit, dp)
    # Search if the variable is already saved in dp/routinesMemory
    dpnm = get_npyx_memory(dp)

    fn=f'ids{unit}_{enforced_rp}.npy'
    if op.exists(Path(dpnm,fn)) and not again:
        if verbose: print("File {} found in routines memory.".format(fn))
        indices = np.asarray(np.load(Path(dpnm,fn)), dtype='int64')
    # if not, compute it
    else:
        if verbose: print(f"File {fn} not found in routines memory. Will be computed from source files.")
        if not (assert_int(unit)|assert_float(unit)): raise TypeError(f'WARNING unit {unit} type ({type(unit)}) not handled!')
        assert unit in get_units(dp), f'WARNING unit {unit} not found in dataset {dp}!'
        # if assert_multi(dp):
        #     ds_table = get_ds_table(dp)
        #     if ds_table.shape[0]>1: # If merged dataset
        #         spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
        #         indices = np.nonzero(spike_clusters==unit)[0].ravel()
        #     else:
        #         ds_i, unt = get_dataset_id(unit)
        #         spike_clusters = np.load(Path(ds_table.loc['dp'][ds_i],"spike_clusters.npy"), mmap_mode='r')
        #         indices = np.nonzero(spike_clusters==unt)[0].ravel()
        # else:

        spike_clusters = np.load(dp/"spike_clusters.npy", mmap_mode='r')
        indices = np.nonzero(spike_clusters==unit)[0].ravel()

        # Save it
        if sav:
            np.save(dpnm/fn, indices)
            
    # Optional selection of spies without duplicates
    dp_source = npyx.merger.get_source_dp_u(dp, unit)[0]
    fs=read_metadata(dp_source)["highpass"]['sampling_rate']
    train = trn(dp, unit, again=again, enforced_rp=-1)
    duplicates_m = duplicates_mask(train, enforced_rp, fs)
    train = train[~duplicates_m]
    indices = indices[~duplicates_m]

    # Optional selection of a section of the recording.
    # Always computed because cannot reasonably be part of file name.
    periods = check_periods(periods)
    if not isinstance(periods, str): # check_periods ensures that it should be 'all' if it is a string
        sec_bool=np.zeros(len(train)).astype(bool)
        for section in periods:
            sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
        indices=indices[sec_bool]

    return indices

def load_amplitudes(dp, unit, verbose=False, periods='all', again=False, enforced_rp=-1):
    f'''Load unit amplitudes
    i.e. kilosort template scaling factor for each spike.
    
    for parameters see doc of ids:
    {ids.__doc__}'''
    dp = Path(dp)
    unit_ids = ids(dp, unit, True, verbose, periods, again, enforced_rp)
    amps = np.load(dp/'amplitudes.npy')[unit_ids]
    
    return amps

@cache_memory.cache(cache_validation_callback=cache_validation_again)
def trn(dp, unit, sav=True, verbose=False, periods='all', again=False, enforced_rp=0):
    '''
    ********
    routine from routines_spikes
    computes spike train (1, Nspikes) - int64, in samples
    ********

    - dp (string): DataPath to the Neuropixels dataset.
    - u (int): unit index
    - ret (bool - default False): if True, train returned by the routine.
    if False, by definition of the routine, drawn to global namespace.
    - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
    - periods: list [[t1,t2], [t3,t4],...] (in seconds) or 'all' for all periods.
    - again: boolean, if True recomputes data from source files without checking routines memory.
    - enforced_rp: float, enforced refractory period (ms)- if 2 spikes are separated by less than enforced_rp ms, the first one only is kept.
                   By default 0, only removed pure duplicates (they happen!).
    '''

    # Search if the variable is already saved in dp/routinesMemory
    dpnm = get_npyx_memory(dp)
    dp_source = npyx.merger.get_source_dp_u(dp, unit)[0]
    fs=read_metadata(dp_source)['highpass']['sampling_rate']

    fn=f'trn{unit}_{enforced_rp}+.npy'
    if (dpnm/fn).exists() and not again:
        if verbose: print("File {} found in routines memory.".format(fn))
        try: train = np.load(dpnm/fn) # handling of weird allow_picke=True error
        except: pass

    # if not, compute it
    if 'train' not in locals(): # handling of weird allow_picke=True error when using joblib multiprocessing
        if verbose: print(f"File {fn} not found in routines memory. Will be computed from source files.")
        if not (assert_int(unit)|assert_float(unit)): raise TypeError(f'WARNING unit {unit} type ({type(unit)}) not handled!')
        assert unit in get_units(dp), f'WARNING unit {unit} not found in dataset {dp}!'

        spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
        spike_samples = np.load(Path(dp,'spike_times.npy'), mmap_mode='r')
        train = spike_samples[spike_clusters==unit].ravel()

        # Filter out spike duplicates (spikes following an ISI shorter than enforced_rp)
        # by default, only pure duplicates (yeah they happen!!)
        assert len(train)!=0, f'unit {unit} not found in spike_clusters.npy - probably a merger bug.'
        duplicates_m = duplicates_mask(train, enforced_rp, fs)
        train = train[~duplicates_m]
        # Save it
        if sav:
            np.save(dpnm/fn, train)

    # Optional selection of a section of the recording.
    # Always computed because cannot reasonably be part of file name.
    periods = check_periods(periods)
    periods = periods * fs # convert from seconds to samples
    if not isinstance(periods, str): # check_periods ensures that it should be 'all' if it is a string
        sec_bool = np.zeros(len(train)).astype(bool)
        for section in periods:
            sec_bool = sec_bool|(train>=section[0])&(train<=section[1])
        train=train[sec_bool]

    return train

def duplicates_mask(t, enforced_rp=0, fs=30000):
    '''
    - t: in samples,sampled at fs Hz
    - enforced_rp: in ms'''
    duplicate_m = np.append([False], np.diff(t)<=enforced_rp*fs/1000)
    return duplicate_m

def enforce_rp(t, enforced_rp=0, fs=30000):
    '''
    Enforce a refractory period of enforced_rp ms to a spike train.
    Every spike FOLLOWING a spike by less than enforced_rp ms is dropped.
    - t: np.array in samples, sampled at fs Hz
    - enforced_rp: float, in ms'''
    duplicate_m = duplicates_mask(t, enforced_rp, fs)
    return t[~duplicate_m]

def isi(dp, unit, enforced_rp=0, sav=True, verbose=False, periods='all', again=False):
    '''
    ********
    routine from routines_spikes
    computes spike inter spike intervals (1, Nspikes-1) - float64, in milliseconds
    ********

    - dp (string): DataPath to the Neuropixels dataset.
    - u (int): unit index
    - ret (bool - default False): if True, train returned by the routine.
      If False, by definition of the routine, drawn to global namespace.
    - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
    '''
    t=trn(dp, unit, sav, verbose, periods, again, enforced_rp)
    return np.diff(t) if len(t)>1 else None

def inst_cv2(t):
    '''
    Arguments:
        - t: (nspikes,) np array, spike times in any unit

    Returns:
        - cv2: (nspikes-2,) array, instantaneous cv2
    '''

    isint = np.diff(t)

    cv2 = 2 * np.abs(isint[1:] - isint[:-1]) / (isint[1:] + isint[:-1])

    return cv2

@cache_memory.cache
def mean_firing_rate(t, exclusion_quantile=0.005, fs=30000):
    """
    - t: array of time stamps, in samples
    - exclusion_quantile: float, quantiles beyond which we exclude too long interspike intervals
    - fs: sampling frequency, in Hz
    """
    isint = np.diff(t) if len(t)>1 else None
    if isint is None: return 0
    # Remove large outliers
    quantile_high = np.quantile(isint, 1-exclusion_quantile)
    #quantil_low = np.quantile(isint, exclusion_quantile)
    isint = isint[(isint <= quantile_high)]
    isint = isint/fs
    return np.round(1./np.mean(isint),2)

def mfr(dp=None, U=None, exclusion_quantile=0.005, enforced_rp=0,
        periods='all', again=False, train=None, fs=None):
    '''
    Computes the mean firing rate of a unit.

    2 ways to use it:
     - regular npyx mfr(dp, u) syntax - will grab the spike times and fs from files
     - custom mfr(train=array, fs=sampling_frequency) - for any spike train

    Arguments:
        - dp: str, datapath
        - U: int or list, unit or list of units
        - exclusion_quantile: float, quantiles beyond which we exclude interspike intervals (very short or very long)
        - enforced_rp: float, enforced refractory period - minimum time in ms separating consecutive spikes of the train - if sorther than that, the 1st spike of the couple is kept.
        - periods: 'all' or [(t1,t2), ...] - periods in s to consider for calculation
        - again=False
        - train=None
        - fs=None

    Returns:
        - mfr: float or np array, mean firing rates of unit(s) in Hz
    '''
    if train is not None:
        assert fs is not None, 'you need to provide a sampling frequency!'
        train=np.asarray(train)
        assert train.ndim==1
        return mean_firing_rate(train, exclusion_quantile, fs)

    U=npa([U]).flatten()
    MFR=[]
    for u in U:
        t=trn(dp, u, periods=periods, again=again, enforced_rp=enforced_rp)
        dp_source = npyx.merger.get_source_dp_u(dp, u)[0]
        fs=read_metadata(dp_source)['highpass']['sampling_rate']
        MFR.append(mean_firing_rate(t, exclusion_quantile, fs))

    return MFR[0] if len(U)==1 else npa(MFR)

def binarize(X, bin_size, fs, rec_len=None):
    '''Function to turn a spike train (array of time stamps)
       into a binarized spike train (array of 0 or 1
                                     of length rec_len with a given bin_size.).
       - X: spike train (array of time stamps, in samples sampled at fs Hertz)
       - bin_size: size of binarized spike train bins, in milliseconds.
       - rec_len: length of the recording, in SAMPLES. If not provided, time of the last spike.
       - fs: sampling frequency, in Hertz.'''

    # Process bin_size
    bin_size = int(np.ceil(fs*bin_size/1000))  # Conversion ms->samples

    # Process rec_len
    if rec_len is None:
        rec_len=X[-1]

    # Binarize spike train
    Xb = np.histogram(X, bins=np.arange(0, rec_len, bin_size))[0]

    # Decrease array size as much as possible
    for encode in [32,16,8]:
        Xb1=Xb.astype(f'int{encode}')
        if not np.all(Xb==Xb1):
            break
        Xb=Xb1

    return Xb

@cache_memory.cache(cache_validation_callback=cache_validation_again)
def trnb(dp, u, b, periods='all', again=False):
    '''
    ********
    Computes binarized spike train (1, Nspikes) - int64, in samples
    ********

    - dp (string): DataPath to the Neuropixels dataset.
    - u (int): unit index
    - bin_size: size of binarized spike train bins, in milliseconds.
    '''
    dp_source = npyx.merger.get_source_dp_u(dp, u)[0]
    fs=read_metadata(dp_source)['highpass']['sampling_rate']
    assert b>=1000/fs
    t = trn(dp, u, enforced_rp=1, periods=periods, again=again)
    t_end = np.load(Path(dp,'spike_times.npy'), mmap_mode='r').ravel()[-1]
    return binarize(t, b, fs, t_end)

def get_firing_periods(dp, u, b=1, sd=1000, th=0.02, again=False, train=None, fs=None, t_end=None):
    '''
    Arguments:
        - t: array of spike times, in samples
        - t_end: recording end time, in samples
        - b: float, bin size i.e. temporal resolution of presence periods, in ms | Default 1
        - sd: float, standard deviation of gaussian smoothing window, in ms | Default 1000
        - th: threshold to define presence, in fraction of mean firing rate
        - fs: sampling rate of spike times, in Hz
    '''
    assert 1<sd<10000
    assert 0<=th<1
    sav=False
    if train is None:
        sav=True
        dpnm = get_npyx_memory(dp)
        fn=f'firing_periods_{u}_{b}_{sd}_{th}.npy'
        if op.exists(Path(dpnm,fn)) and not again:
            return np.load(Path(dpnm,fn))
        t = trn(dp, u, enforced_rp=1, again=again)
        dp_source = npyx.merger.get_source_dp_u(dp, u)[0]
        fs=read_metadata(dp_source)['highpass']['sampling_rate']
        t_end = np.load(Path(dp,'spike_times.npy'), mmap_mode='r').ravel()[-1]
    else:
        assert fs is not None, "You need to provide the sampling rate of the provided train!"
        assert t_end is not None, "You need to provide the end time 't_end' of recorded train that you provide!"
        t=np.asarray(train)
        assert t.ndim==1

    periods = firing_periods(t, fs, t_end, b=b, sd=sd, th=th)

    if sav:
        np.save(Path(dpnm,fn), periods)

    return periods

def firing_periods(t, fs, t_end, b=1, sd=1000, th=0.02,
                   again=False, dp=None, u=None, return_smoothed_rate = False):
    '''
    Arguments:
        - t: array of spike times, in samples
        - fs: sampling rate of spike times, in Hz
        - t_end: recording end time, in samples
        - b: float, bin size i.e. temporal resolution of presence periods, in ms | Default 1. minimum 0.01
        - sd: float, standard deviation of gaussian smoothing window, in ms | Default 1000
        - th: threshold to define presence, in fraction of mean firing rate
    Returns:
        - periods, in samples
    '''
    sav=False
    if u is not None and not return_smoothed_rate:
        sav=True
        assert dp is not None
        #assert len(trn(dp,u,0))==len(t), 'There seems to be a mismatch between the provided spike trains and the unit index.'
        fn=f'firing_periods_{u}_{b}_{sd}_{th}.npy'
        dpnm = get_npyx_memory(dp)
        if op.exists(Path(dpnm,fn)) and not again:
            return np.load(Path(dpnm,fn))

    tbs = inst_firing_rate(t, fs, t_end, b, sd, again) # result is inst. firing rate in Hz - speed bottleneck

    assert 0<=th
    fr_th = mean_firing_rate(t, 0.005, fs)*th
    periods = thresh_consec(tbs, fr_th, sgn=1, n_consec=0, exclude_edges=False, only_max=False, ret_values=False)
    if not any(periods): periods=[[0,len(tbs)-1]]

    periods = (np.array(periods)*((b/1000)*fs)).astype(np.int64) # conversion from bins to samples

    if sav: np.save(Path(dpnm,fn), periods)

    if return_smoothed_rate:
        return periods, tbs

    return periods

@cache_memory.cache(cache_validation_callback=cache_validation_again)
def inst_firing_rate(t, fs, t_end, b=1, sd=1000, again=False):
    assert 1<sd<100000
    assert t.ndim==1
    t     = np.asarray(t)

    assert b>=10/fs
    tb    = binarize(t, b, fs, t_end)
    sd    = int(sd/b) # convert from ms to bin units
    b_s   = b/1000 # bin seconds
    tbs   = smooth(tb, 'gaussian', sd=sd)/b_s # result is inst. firing rate in Hz - speed bottleneck

    return tbs

def find_stable_recording_period(t, fs, t_end, target_period = 30,
                                 b=1000, sd=10000, minimum_fr = 0.4,
                                 return_rate=False):
    
    """
    Finds a locally optimal recording period (in terms of stability i.e. low std) of at least 'target_period' seconds.
    Arguments:
        - t: array of spike times, in samples
        - fs: sampling rate of spike times, in Hz
        - t_end: recording end time, in samples
        - target_period: float, minimum length of stable period, in seconds | Default 30.
                        If the algorithm does not find any period lasting at least this amount of time
                        for at least minimum_fr fraction of the mean firing rate,
                        it ties to decrease this period.
        - b: float, bin size i.e. temporal resolution of presence periods, in ms | Default 1
        - sd: float, standard deviation of gaussian smoothing window, in ms | Default 1000
        - th: threshold to define presence, in fraction of mean firing rate
    Returns:
        - periods: (2,) array of stable period on/offset, in samples
    """
    if not np.any(t):
        if return_rate:
            return np.array([np.nan, np.nan]), np.nan
        return np.array([np.nan, np.nan])       

    target_period = target_period*fs

    # find candidate periods
    fr_th = 0.9
    periods, tbs = firing_periods(t, fs, t_end, b, sd, th=fr_th, return_smoothed_rate=True)
    periods_t = np.diff(periods, axis=1).ravel()
    while not np.any(periods_t>target_period):
        periods, tbs = firing_periods(t, fs, t_end, b, sd, th=fr_th, return_smoothed_rate=True)
        periods_t = np.diff(periods, axis=1).ravel()
        fr_th -= 0.05
        if fr_th<=minimum_fr: # do not tolerate less then x% of mfr
            target_period -= 2
            fr_th = 0.9
            if target_period<=1:
                print("Could not find a long enough stable recording period!")
                break

    # if none was found, return nans
    if not np.any(periods_t>target_period):
        if return_rate:
            return np.array([np.nan, np.nan]), tbs
        return np.array([np.nan, np.nan])
    
    # find the most stable period
    candidate_periods = periods[periods_t>target_period]
    variances = np.zeros(len(candidate_periods))
    for i, p in enumerate(candidate_periods):
        period_ms = p * 1000 / fs
        period_bins = (period_ms / b).astype(int)
        variances[i] = np.std(tbs[period_bins[0]:period_bins[1]])
    
    most_stable_period = candidate_periods[np.argmin(variances)]

    most_stable_period_trimmed = np.array([most_stable_period[0], most_stable_period[0]+target_period])

    if return_rate:
        return most_stable_period_trimmed, tbs
    return most_stable_period_trimmed


def train_quality(dp, unit, period_m=[0,20],
                  fp_chunk_span=3, fp_chunk_size = 10,
                  fn_chunk_span = 3, fn_chunk_size = 10,
                  use_or_operator = True,
                  violations_ms = 0.8, fp_threshold = 0.05, fn_threshold = 0.05,
                  again = False, save = True, verbose = False, plot_debug = False,
                  enforced_rp = 0):
    """
    Subselect spike times which meet two criteria:
        low number of 'missed spikes' (false negatives)
        and low number of 'extra spikes' (false positives).
        
    The recording within period_m is split in fp/fn_chunk_size seconds chunks,
    and fp/fn_chunk_span chunks are used to estimate the fp and fn rates.
    (e.g. 3 10s chunks means that the rates are estimated on 30s chunks, overlapping by 10s).

    - False negative rate estimation: for checking which sections of the recording
        have too many spikes missing, by looking if the section is
        approximately Gaussian. If the time section of the recording
        has too much of the Gaussian distribution cut off (>5%) the section
        has to be discarded
    - False positive rate estimation: check if there are not too many spikes
        occuring in the the refractory period of the autocorrelogram.
        If there are too many spikes in the ACG the section of the
        recording will be discarded.
        
    Finally, the spikes belonging to the intersection (use_or_operator=False) or union (use_or_operator=True)
    of the chunks with low enough fp/fn rates are returned.
    
    E.g. if fp_chunk_span=3 and fp_chunk_size=10 anf use_or_operator=False,
    for a given 10s "chunk_k" to be considered, ALL 3*10=30s chunk triplets
    (chunk_k-2,chunk_k-1,chunk_k), (chunk_k-1,chunk_k,chunk_k+1) AND (chunk_k,chunk_k+1,chunk_k+2)
    fp rate estimations must be below fp_threshold.
    
    ***********
    
    Arguments:
        - dp: str, path to dataset.
        - unit: int/float, unit index.
        - period_m: [t1, t2] list of floats in minutes, period to consider
        - fp_chunk_span: int, number of recording chunks to concatenate to estimate fp rate.
        - fp_chunk_size: int, size of recording chunks used to estimate fp rate.
        
        - fn_chunk_span: int, number of chunks to concatenate to estimate fp rate.
        - fn_chunk_size: int, size of recording chunks used to estimate fn rate.
        
        - use_or_operator: bool, whether to use the union (True) or intersection (False)
                        of stitched chunks (fp/fn_chunk_span)
                        to state that a chunk passed the fp/fn threshold or not.
                        E.g. if 
        
        - violations_ms: float, width of window in ms used to estimate refractory period violations (center of autocorrelogram)
        - fp_threshold: [0-1] float, false positive rate (ratio of refractory periods violations/mean firing rate) threshold
        - fn_threshold: [0-1] float, false negative rate (AUC of gaussian fit missing) threshold
        
        - again: bool, whether to recompute trn_filtered or simply load it from npyxMemory
        - save: bool, whether to save result to npyxMemory for future fast reloading
        - verbose: bool, whether to print extra information for debugging purposes.
        - enforced_rp: float, enforced refractory period in ms (if 2 spikes are closer than enforced_rp ms, only the first one  is kept.)
    
    Returns:
        - good_spikes_m: mask of spikes belonging to the intersection of the chunks with low enough fp/fn rates.
        - good_fp_start_end: list of [start, end] of chunks with low enough fp rate (in seconds).
        - good_fn_start_end: list of [start, end] of chunks with low enough fn rate (in seconds).
    """
    
    dp = Path(dp)

    # Hard-coded parameters
    c_bin = 0.2
    c_win = 100
    n_bins_acg_baseline=80 # from start and end of acg window
    n_spikes_threshold = 300
    fs = 30_000
    title = f"{unit}, {dp.name}" # for plot_debug


    # check that the passed values make sense
    assert isinstance(dp, (str, PosixPath, WindowsPath)),\
        'Provide a string or a pathlib object as the source directory'
    assert assert_int(unit), 'Unit provided should be an int'
    assert assert_int(fp_chunk_span), 'fp_chunk_span provided should be an int'
    assert assert_int(fp_chunk_size), 'fp_chunk_size provided should be an int'
    assert assert_int(fn_chunk_span), 'fn_chunk_span provided should be an int'
    assert assert_int(fn_chunk_size), 'fn_chunk_size provided should be an int'

    assert fp_chunk_size >= 1, "ACG window length needs to be larger than 1 sec"
    assert fn_chunk_size >= 1, "Gaussian window length needs to be larger than 1 sec"
    assert fp_chunk_span >= 1, "ACG chunk size needs to be larger than 1 "
    assert fn_chunk_span >= 1, "Gaussian chunk size needs to be larger than 1 "

    dpnm = get_npyx_memory(dp)
    
    # Load data
    unit_amp = load_amplitudes(dp, unit, verbose, 'all', again, enforced_rp)
    
    unit_train = trn(dp, unit, enforced_rp=enforced_rp, again=again, verbose=verbose)/fs

    if period_m is None:
        period_m = [0, unit_train[-1]//60]
    period_s=[period_m[0]*60, period_m[1]*60]

    # Attempt to reload if precomputed
    fn=(f"trn_qual_{unit}_{str(period_m).replace(' ','')}"
        f"_{str(fp_chunk_span)}_{str(fp_chunk_size)}_{str(fn_chunk_span)}_{str(fn_chunk_size)}"
        f"_{str(violations_ms)}_{str(fp_threshold)}_{str(fn_threshold)}_{enforced_rp}.npy")
    fn_spikes = "spikes_"+fn
    if (dpnm/fn).exists() and (dpnm/fn_spikes).exists() and (not again):
        if verbose: print(f"File {fn} found in routines memory.")
        good_fp_start_end, good_fn_start_end = np.load(dpnm/fn, allow_pickle=True)
        good_spikes_m = np.load(dpnm/fn_spikes)
        
        good_fp_start_end_plot=None if len(good_fp_start_end)==1 else good_fp_start_end
        good_fn_start_end_plot=None if len(good_fn_start_end)==1 else good_fn_start_end
        if plot_debug:
            npyx.plot.plot_fp_fn_rates(unit_train, period_s, unit_amp, good_spikes_m,
                     None, None, None,None,
                     fp_threshold, fn_threshold,
                     good_fp_start_end_plot, good_fn_start_end_plot, title)
        
        return good_spikes_m, good_fp_start_end.tolist(), good_fn_start_end.tolist()
    
    n_spikes = np.count_nonzero((unit_train>period_s[0])&(unit_train<period_s[1]))

    # steps:
        # split into 10 second chunks
        # run 3 concatenated 10 sec chunks through the filter
        # find all the chunks that passed the filter
        # find the time winodws where thes chunks happened
        # extract the spike times again from these consecutive windows
        # append the spike times from these consecutiv windowws
        # calculate features on this array

    recording_span = period_s[1]-period_s[0]
    n_fn_chunks =  int(recording_span / fn_chunk_size)
    n_fp_chunks =  int(recording_span / fp_chunk_size)
    passed_fn = np.zeros((n_fn_chunks,3)).astype('int')
    passed_fp = np.zeros((n_fp_chunks,3)).astype('int')
    fn_chunks = [[period_s[0]+t*fn_chunk_size, period_s[0]+(t+fn_chunk_span)*fn_chunk_size] for t in range(n_fn_chunks)]
    fp_chunks = [[period_s[0]+t*fp_chunk_size, period_s[0]+(t+fp_chunk_span)*fp_chunk_size] for t in range(n_fn_chunks)]


    fp_toplot, chunk_fp_t, fn_toplot, chunk_fn_t = [], [], [], []
    if len(unit_amp) > n_spikes_threshold:
        
        # False negative estimation
        for i, (t1,t2) in enumerate(fn_chunks):
            chunk_mask = (t1 <= unit_train) & (unit_train < t2)
            n_spikes_chunk=np.sum(chunk_mask)

            if n_spikes_chunk > 15:
                amplitudes_chunk = unit_amp[chunk_mask].astype(np.float64)
                chunk_bins = estimate_bins(amplitudes_chunk, rule='Fd')
                if chunk_bins> 3:
                    x_c, p0_c, min_amp_c, n_fit_c, n_fit_no_cut_c, chunk_spikes_missing = gaussian_amp_est(amplitudes_chunk, chunk_bins)
                    fn_toplot.append(chunk_spikes_missing/100)
                    chunk_fn_t.append(t1+(t2-t1)/2)
                    if (~np.isnan(chunk_spikes_missing)) & (chunk_spikes_missing <= fn_threshold*100):
                        passed_fn[i] = [t1, t2, 1]

        # next loop through the chunks made for the ACG extraction
        # get the periods where the ACG filter passed
        
        # Compute denominator of FP on total period
        # acg_tot = npyx.corr.acg(dp, unit, c_bin, c_win, verbose = False,  periods=[period_s])
        # x_block = np.round(np.arange(-c_win/2, c_win/2 + c_bin, c_bin), 8) # round to fix binary imprecisions
        # # rp_mask = (x_block >= -violations_ms) & (x_block <= violations_ms)
        # baseline_mask = (acg_tot*0).astype(bool)
        # baseline_mask[:n_bins_acg_baseline] = True
        # baseline_mask[-n_bins_acg_baseline:] = True
        # # baseline_mean = np.mean(acg_tot[baseline_mask])
        
        # False positive estimation
        for i, (t1,t2) in enumerate(fp_chunks):

            chunk_mask = (t1 <= unit_train) & (unit_train < t2)
            n_spikes_chunk=np.sum(chunk_mask)

            if n_spikes_chunk > 15:

                fp_rate = npyx.metrics.isi_violations(unit_train, min_time=t1, max_time=t2,
                                         isi_threshold=violations_ms/1000, min_isi=0)[0]
                fp_toplot.append(fp_rate)
                chunk_fp_t.append(t1+(t2-t1)/2)
                if (fp_rate <= fp_threshold):
                    passed_fp[i] = [t1, t2, 1]

    # Across all chunks, if at least 1 good chunk for both (else no spike can be called good)
    if (np.sum(passed_fp[:,2])  > 1) & (np.sum(passed_fn[:,2]) > 1):

        # Aggregate FP and FN masks for each elemental chunks
        # e.g. for 30s chunks overlapping at 33% (3 10s chunks),
        # find for each 10s chunk what the final FP/FN rate is
        # based on the rate of the 3 30s chunks which overlap with it (and/or, see use_or_operator)
        # (apart from edges, where only 1 30s chunk overlaps)
        subchunks_fp = np.unique(npa(fp_chunks).flatten())
        subchunks_fp = npa([[subchunks_fp[i], subchunks_fp[i+1]] for i in range(len(subchunks_fp)-1)])
        good_fp_bool = passed_fp[:,2].astype(bool)
        subchunks_fp_bool = np.zeros((subchunks_fp.shape[0], fp_chunk_span))*np.nan
        for i in range(fp_chunk_span):
            subchunks_fp_bool[i:len(subchunks_fp_bool)-fp_chunk_span+i+1,i]=good_fp_bool
        
        subchunks_fn = np.unique(npa(fn_chunks).flatten())
        subchunks_fn = npa([[subchunks_fn[i], subchunks_fn[i+1]] for i in range(len(subchunks_fn)-1)])
        good_fn_bool = passed_fn[:,2].astype(bool)
        subchunks_fn_bool = np.zeros((subchunks_fn.shape[0], fn_chunk_span))*np.nan
        for i in range(fn_chunk_span):
            subchunks_fn_bool[i:len(subchunks_fn_bool)-fn_chunk_span+i+1,i]=good_fn_bool
        
        # Find out if subchunks are good based on overlapping chunks 
        # (NaNs do not count in np/all/any -> decision on edges based on only 1 chunk)
        nanm_fp = np.isnan(subchunks_fp_bool)
        nanm_fn = np.isnan(subchunks_fn_bool)
        if use_or_operator:
            # nans must be 0 to be ignored by any
            subchunks_fn_bool[nanm_fn] = 0
            subchunks_fp_bool[nanm_fp] = 0
            
            subchunks_fn_bool = np.any(subchunks_fn_bool, axis=1)
            subchunks_fp_bool = np.any(subchunks_fp_bool, axis=1)
        else:
            # nans must be 1 to be ignored by all
            subchunks_fn_bool[nanm_fn] = 1
            subchunks_fp_bool[nanm_fp] = 1
            
            subchunks_fn_bool = np.all(subchunks_fn_bool, axis=1)
            subchunks_fp_bool = np.all(subchunks_fp_bool, axis=1)
            
        good_fp_start_end = subchunks_fp[subchunks_fp_bool]
        good_fn_start_end = subchunks_fn[subchunks_fn_bool]
        
        ## Finally, mask spikes meeting the FP AND FN rates
        fp_m = np.zeros(unit_train.shape[0]).astype(bool)
        for subchunk in good_fp_start_end:
            m = (unit_train>subchunk[0])&(unit_train<subchunk[1])
            fp_m = fp_m|m
        fn_m = np.zeros(unit_train.shape[0]).astype(bool)
        for subchunk in good_fn_start_end:
            m = (unit_train>subchunk[0])&(unit_train<subchunk[1])
            fn_m = fn_m|m
        good_spikes_m=fp_m&fn_m

        if save:
            np.save(Path(dpnm,fn), np.array( (np.array(good_fp_start_end), np.array(good_fn_start_end)), dtype = object))
            np.save(dpnm/fn_spikes, good_spikes_m)

        if plot_debug:
            npyx.plot.plot_fp_fn_rates(unit_train, period_s, unit_amp, good_spikes_m,
                     fp_toplot, fn_toplot, chunk_fp_t, chunk_fn_t,
                     fp_threshold, fn_threshold,
                     good_fp_start_end, good_fn_start_end, title)

        return good_spikes_m, good_fp_start_end, good_fn_start_end
    
    else:
        good_spikes_m=(unit_train*0).astype(bool)
        if save:
            np.save(Path(dpnm,fn), np.array(([0], [0]), dtype = object)  )
            np.save(dpnm/fn_spikes, good_spikes_m)
            
        if plot_debug and n_spikes>0:
            npyx.plot.plot_fp_fn_rates(unit_train, period_s, unit_amp, good_spikes_m,
                     fp_toplot, fn_toplot, chunk_fp_t, chunk_fn_t,
                     fp_threshold, fn_threshold,
                     None, None, title)
            
        return good_spikes_m, [0], [0]

@docstring_decorator(train_quality.__doc__)
def trn_filtered(dp, unit, period_m=[0,20],
                  fp_chunk_span=3, fp_chunk_size = 10,
                  fn_chunk_span = 3, fn_chunk_size = 10,
                  use_or_operator = True,
                  violations_ms = 0.8, fp_threshold = 0.05, fn_threshold=0.05,
                  use_consecutive = False, consecutive_n_seconds = 180,
                  again = False, save = True, verbose = False, plot_debug = False,
                  enforced_rp=0):
    """
    Returns spike times (in sample) meeting the false positive and false negative criteria.
    Mainly wrapper of train_quality().
    
    Extra parameters not in train_quality (see below for others):
        - use_consecutive: bool, whether to only return spikes from the longest chunk
                                 (at least consecutive_n_seconds total).
                                 If False, all spikes belonging to good chunks are returned.
        - consecutive_n_seconds: float, minimum tolerated size (in seconds) of recording section
                                 with consecutive good chunks for its spikes to be returned.
                                 If less than consecutive_n_seconds good sections in total, does not pass.
        
    Returns:
        - train_filtered: array of spike times (in samples)
                          belonging to recording chunks where both FP and FN rates are low enough.
                          
    train_quality docstring:
    
    {0}
    """
    dp = Path(dp)
    t = trn(dp,unit, enforced_rp=enforced_rp, again=again)
    t_s=t/30000
    good_spikes_m, good_fp_start_end, good_fn_start_end = train_quality(dp, unit, period_m,
                    fp_chunk_span, fp_chunk_size, fn_chunk_span, fn_chunk_size, use_or_operator,
                    violations_ms, fp_threshold, fn_threshold, again, save, verbose, plot_debug,
                    enforced_rp=enforced_rp)

    # use spike times themselves to define beginning and end of good Sections
    # as the FP and FN sections do not necessarily overlap
    good_sections = good_sections_from_mask(good_spikes_m, t_s)
    
    if len(good_sections)>0:#
        total_good_sections = sum([s[1]-s[0] for s in good_sections])
        if use_consecutive:
            # get the longest consecutive section of time that passes
            # both our criteria
            maxrun = -1
            run_len = {}
            for x in good_sections:
                mrun = run_len[x] = run_len.get(x-1, 0) + 1
                if mrun > maxrun:
                    maxend, maxrun = x, mrun
            consecutive_good_chunk = list(range(maxend-maxrun+1, maxend+1))
            if len(consecutive_good_chunk) > consecutive_n_seconds:
                good_spikes_m = (t_s>consecutive_good_chunk[0])&(t_s<consecutive_good_chunk[-1]+1)
                return t[good_spikes_m], good_spikes_m
        else:
            if total_good_sections > consecutive_n_seconds:
                return t[good_spikes_m], good_spikes_m
        
    if verbose: print('No consecutive section passed the filters')
    return np.array([0]), (t*0).astype(bool)

def good_sections_from_mask(good_times_m, time_series=None):
    """
    Returns a list of good sections [[t1,t2], ...] in units of 'time_series'
    from a boolean mask of time stamps good_times_m.
    """
    edges = np.diff([0]+list(good_times_m.astype(int))+[0])
    good_left = np.nonzero(edges==1)[0]
    good_right = np.nonzero(edges==-1)[0]-1
    if time_series is not None:
        return [[time_series[l], time_series[r]] for l,r in zip(good_left, good_right) if r>l]
    else:
        return [[l,r] for l,r in zip(good_left, good_right) if r>l]

@cache_memory.cache
def get_common_good_sections(good_sections_list):
    """
    good_sections: list of lists of periods
    [
    [[t11, t12], [t13,t14], ...],
    [[t21, t22], [t23,t24], ...],
    ]
    Must be in SAMPLES, INTEGERS.

    Returns:
        - common_good_sections: list of sections common to all the lists of periods.
    """

    periods = [sec[i] for sec in good_sections_list for i in range(len(sec))]
    end = int(np.max(periods))
    bad_sections = []
    for sec in good_sections_list:
        sec = np.array(sec)
        assert sec.shape[1]==2, "All periods must be of size 2!"
        bad_sec = np.concatenate([[0], np.array(sec).ravel(), [end]]).astype(int)
        bad_sec = bad_sec.reshape(-1,2)
        bad_sections.append(bad_sec)
    m = np.full((end,), True)
    for sec in bad_sections:
        for sec_per in sec:
            m[slice(sec_per[0]+1, sec_per[1])] = False

    return good_sections_from_mask(m)

def gaussian_cut(x, a, mu, sigma, x_cut):
    g = a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    g[x < x_cut] = 0
    return g


def curve_fit_(x, num, p1):
    pop_t = curve_fit(gaussian_cut, x, num, p1, maxfev=10000)
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
    if isinstance(mode_seed, np.ndarray):
        mode_seed = mode_seed[0]

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
        n_fit_no_cut = 0
        #n_fit_no_cut = gaussian_cut(x1, a=p0[0], mu=p0[1], sigma=p0[2], x_cut=0)
        percent_missing = round(100 * norm.cdf((min_amp - p0[1]) / p0[2]), 2)

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
        iqr_ = iqr(data, scale=1, nan_policy="omit")
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

# circular import
import npyx.merger
import npyx.metrics
import npyx.plot
