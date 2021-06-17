# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os
import os.path as op; opj=op.join
from pathlib import Path, PosixPath
from ast import literal_eval as ale

from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.optimize import curve_fit
from scipy.stats import norm
from npyx.utils import smooth, thresh_consec, npa, assert_int, assert_float
from npyx.gl import get_units, assert_multi, get_dataset_id, get_ds_table
from npyx.io import read_spikeglx_meta

def ids(dp, unit, sav=True, prnt=False, subset_selection='all', again=False):
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
    - subset_selection = 'all' or [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
    - again: boolean, if True recomputes data from source files without checking routines memory.
    '''

    assert unit in get_units(dp), 'WARNING unit {} not found in dataset {}!'.format(unit, dp)
    # Search if the variable is already saved in dp/routinesMemory
    dprm = Path(dp,'routinesMemory')
    fn='ids{}.npy'.format(unit)
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        indices = np.asarray(np.load(Path(dprm,fn)), dtype='int64')
    # if not, compute it
    else:
        if prnt: print(f"File {fn} not found in routines memory. Will be computed from source files.")
        if not (assert_int(unit)|assert_float(unit)): raise TypeError(f'WARNING unit {unit} type ({type(unit)}) not handled!')
        assert unit in get_units(dp), f'WARNING unit {unit} not found in dataset {dp}!'
        if assert_multi(dp):
            ds_table = get_ds_table(dp)
            if ds_table.shape[0]>1: # If several datasets in prophyler
                spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
                indices = np.nonzero(spike_clusters==unit)[0].ravel()
            else:
                ds_i, unt = get_dataset_id(unit)
                spike_clusters = np.load(Path(ds_table.loc['dp'][ds_i],"spike_clusters.npy"), mmap_mode='r')
                indices = np.nonzero(spike_clusters==unt)[0].ravel()
        else:
            spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
            indices = np.nonzero(spike_clusters==unit)[0].ravel()

        # Save it
        if sav:
            np.save(Path(dprm,fn), indices)

    # Optional selection of a section of the recording.
    # Always computed because cannot reasonably be part of file name.
    if subset_selection!='all': # else, eq to subset_selection=[(0, spike_samples[-1])] # in samples
        try: subset_selection[0][0]
        except: raise TypeError("ERROR subset_selection should be either a string or a list of format [(t1, t2), (t3, t4), ...]!!")
        fs=read_spikeglx_meta(dp)['sRateHz']
        train=trn(dp, unit, again=again)
        sec_bool=np.zeros(len(train), dtype=np.bool)
        for section in subset_selection:
            sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
        indices=indices[sec_bool]

    return indices



def trn(dp, unit, sav=True, prnt=False, subset_selection='all', again=False, enforced_rp=0):
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
    - subset_selection = 'all' or [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
    - again: boolean, if True recomputes data from source files without checking routines memory.
    - enforced_rp: enforced refractory period, in millisecond. 0 by default (only pure duplicates are removed)
    '''

    # Search if the variable is already saved in dp/routinesMemory
    dprm = Path(dp,'routinesMemory')
    fn='trn{}_{}.npy'.format(unit, enforced_rp)
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        train = np.load(Path(dprm,fn))

    # if not, compute it
    else:
        if prnt: print(f"File {fn} not found in routines memory. Will be computed from source files.")
        if not (assert_int(unit)|assert_float(unit)): raise TypeError(f'WARNING unit {unit} type ({type(unit)}) not handled!')
        assert unit in get_units(dp), f'WARNING unit {unit} not found in dataset {dp}!'
        if assert_multi(dp):
            ds_table = get_ds_table(dp)
            if ds_table.shape[0]>1: # If several datasets in prophyler
                spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
                spike_samples = np.load(Path(dp,'spike_times.npy'), mmap_mode='r')
                train = spike_samples[spike_clusters==unit].ravel()
            else:
                ds_i, unt = get_dataset_id(unit)
                spike_clusters = np.load(Path(ds_table['dp'][ds_i],"spike_clusters.npy"), mmap_mode='r')
                spike_samples = np.load(Path(ds_table['dp'][ds_i],'spike_times.npy'), mmap_mode='r')
                train = spike_samples[spike_clusters==unt].ravel()
        else:
            spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
            spike_samples = np.load(Path(dp,'spike_times.npy'), mmap_mode='r')
            train = spike_samples[spike_clusters==unit].ravel()

        # Filter out spike duplicates (spikes following an ISI shorter than enforced_rp)
        fs=read_spikeglx_meta(dp)['sRateHz']
        train=train[np.append(True, np.diff(train)>=enforced_rp*fs/1000)]

        # Save it
        if sav:
            np.save(Path(dprm,fn), train)

    # Optional selection of a section of the recording.
    # Always computed because cannot reasonably be part of file name.
    if subset_selection!='all': # else, eq to subset_selection=[(0, spike_samples[-1])] (in samples)
        try: subset_selection[0][0]
        except: raise TypeError("ERROR subset_selection should be either a string or a list of format [(t1, t2), (t3, t4), ...]!!")
        fs=read_spikeglx_meta(dp)['sRateHz']
        sec_bool=np.zeros(len(train), dtype=np.bool)
        for section in subset_selection:
            sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
        train=train[sec_bool]

    return train

def isi(dp, unit, enforced_rp=0, sav=True, prnt=False, subset_selection='all', again=False):
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
    t=trn(dp, unit, sav, prnt, subset_selection, again, enforced_rp)
    return np.diff(t) if len(t)>1 else None


def mean_firing_rate(t, exclusion_quantile=0.005, fs=30000):
    i = np.diff(t) if len(t)>1 else None
    if i is None: return
    # Remove outlyers
    i=i[(i>=np.quantile(i, exclusion_quantile))&(i<=np.quantile(i, 1-exclusion_quantile))]/fs
    return np.round(1./np.mean(i),2)

def mfr(dp=None, U=None, exclusion_quantile=0.005, enforced_rp=0, subset_selection='all', again=False, train=None, fs=None):
    if train is not None:
        assert fs is not None, 'you need to provide a sampling frequency!'
        train=np.asarray(train)
        assert train.ndim==1
        return mean_firing_rate(train, exclusion_quantile, fs)

    U=npa([U]).flatten()
    MFR=[]
    for u in U:
        t=trn(dp, u, subset_selection=subset_selection, again=again, enforced_rp=enforced_rp)
        fs=read_spikeglx_meta(dp)['sRateHz']
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

def trnb(dp, u, b, subset_selection='all', again=False):
    '''
    ********
    routine from routines_spikes
    computes binarized spike train (1, Nspikes) - int64, in samples
    ********

    - dp (string): DataPath to the Neuropixels dataset.
    - u (int): unit index
    - bin_size: size of binarized spike train bins, in milliseconds.
    '''
    fs=read_spikeglx_meta(dp)['sRateHz']
    assert b>=1000/fs
    t = trn(dp, u, enforced_rp=1, subset_selection=subset_selection, again=again)
    t_end = np.load(Path(dp,'spike_times.npy'))[-1,0]
    return binarize(t, b, fs, t_end)

def get_firing_periods(dp, u, b=1, sd=1000, th=0.02, again=False, train=None, fs=None, t_end=None):
    '''
    Parameters:
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
        dprm = Path(dp,'routinesMemory')
        if not op.isdir(dprm): os.makedirs(dprm)
        fn=f'firing_periods_{u}_{b}_{sd}_{th}.npy'
        if op.exists(Path(dprm,fn)) and not again:
            return np.load(Path(dprm,fn))
        t = trn(dp, u, enforced_rp=1, again=again)
        fs=read_spikeglx_meta(dp)['sRateHz']
        t_end = np.load(Path(dp,'spike_times.npy'))[-1,0]
    else:
        assert fs is not None, "You need to provide the sampling rate of the provided train!"
        assert t_end is not None, "You need to provide the end time 't_end' of recorded train that you provide!"
        t=np.asarray(train)
        assert t.ndim==1

    periods = firing_periods(t, fs, t_end, b=1, sd=1000, th=0.02)

    if sav:
        np.save(Path(dprm,fn), periods)

    return periods

def firing_periods(t, fs, t_end, b=1, sd=1000, th=0.02, again=False, dp=None, u=None):
    '''
    Parameters:
        - t: array of spike times, in samples
        - fs: sampling rate of spike times, in Hz
        - t_end: recording end time, in samples
        - b: float, bin size i.e. temporal resolution of presence periods, in ms | Default 1
        - sd: float, standard deviation of gaussian smoothing window, in ms | Default 1000
        - th: threshold to define presence, in fraction of mean firing rate
    '''
    sav=False
    if u is not None:
        sav=True
        assert dp is not None
        assert len(trn(dp,u,0))==len(t), 'There seems to be a mismatch between the provided spike trains and the unit index.'
        fn=f'firing_periods_{u}_{b}_{sd}_{th}.npy'
        dprm = Path(dp,'routinesMemory')
        if op.exists(Path(dprm,fn)) and not again:return np.load(Path(dprm,fn))

    assert 1<sd<10000
    assert 0<=th<1
    assert t.ndim==1
    t=np.asarray(t)

    assert b>=1000/fs
    tb = binarize(t, b, fs, t_end)
    sd=int(sd/b) # convert from ms to bin units
    b_s=b/1000 # bin seconds
    tbs=smooth(tb, 'gaussian', sd=sd)/b_s # result is inst. firing rate in Hz - speed bottleneck
    fr_th=mean_firing_rate(t, 0.005, fs)*th

    periods = thresh_consec(tbs, fr_th, sgn=1, n_consec=0, exclude_edges=False, only_max=False, ret_values=False)
    if not any(periods): periods=[[0,len(tbs)-1]]
    periods=(np.array(periods)*(b_s*fs)).astype(int) # conversion from bins to samples

    if sav: np.save(Path(dprm,fn), periods)

    return periods


def train_quality(dp, unit, first_n_minutes=20, acg_window_len=3,
        acg_chunk_size = 10, gauss_window_len = 3,
        gauss_chunk_size = 10, use_or_operator = False, violations_ms = 0.8,
        rpv_threshold = 0.05,  missing_spikes_threshold=5, again = False,
        save = True, prnt = False):

    """
    Apply a filter over the spike times in order to find time points with
    low number of 'missed spikes' (false negatives)
    and low number of 'extra spikes' (false positives).

    There are two filters applied:
            - Flase negative filtering: for checking which sections of the recording
                have too many spikes missing, by looking if the section is
                approximately Gaussian. If the time section of the recording
                has too much of the Gaussian distribution cut off (>5%) the section
                has to be discarded
            - False positive filtering: check if there are not too many spikes 
                occuring in the the refractory period of the autocorrelogram.
                If there are too many spikes in the ACG the section of the
                recording will be discarded.
    Once we have both of these filters applied to the recording, we can take
    the intersection of them. Hence we will have the times when both filters
    were passed.
    Returns: times when the false positive, false negative and both filters were passed
    """
    # check that the passed values make sense


    if isinstance(dp, str) == True:
        dp = Path(dp)
    elif isinstance(dp, PosixPath) == False:
        raise TypeError('Provide a string or a pathlib object as the source directory')

    if not isinstance(unit, (int, np.int16, np.int32, np.int64)):
        raise TypeError('Unit provided should be an int')

    if not isinstance(acg_window_len , (int, np.int16, np.int32, np.int64)):
        raise TypeError('acg_window_len provided should be an int')

    if not isinstance(acg_chunk_size , (int, np.int16, np.int32, np.int64)):
        raise TypeError('acg_chunk_size provided should be an int')

    if not isinstance(gauss_window_len , (int, np.int16, np.int32, np.int64)):
        raise TypeError('gauss_window_len provided should be an int')

    if not isinstance(gauss_chunk_size , (int, np.int16, np.int32, np.int64)):
        raise TypeError('gauss_chunk_size provided should be an int')

    assert acg_chunk_size >= 1, "ACG window length needs to be larger than 1 sec"
    assert gauss_chunk_size >= 1, "Gaussian window length needs to be larger than 1 sec"
    assert acg_window_len >= 1, "ACG chunk size needs to be larger than 1 "
    assert gauss_window_len >= 1, "Gaussian chunk size needs to be larger than 1 "

    dprm = Path(dp,'routinesMemory')

    fn=f"trn_qual_{unit}_{str(acg_window_len)}_{str(acg_chunk_size)}_{str(gauss_window_len)}_{str(gauss_chunk_size)}_{str(violations_ms)}_{str(rpv_threshold)}_{str(missing_spikes_threshold)}.npy"

    if not dprm.is_dir(): dprm.mkdir()
    if Path(dprm,fn).is_file() and (not again):
        if prnt: print(f"File {fn} found in routines memory.")
        good_start_end, acg_start_end, gauss_start_end = np.load(Path(dprm,fn), allow_pickle = True)
        return good_start_end.tolist(), acg_start_end.tolist(), gauss_start_end.tolist()

    unit_size_s = first_n_minutes * 60

    no_gauss_chunks =  int(unit_size_s / gauss_chunk_size)
    no_acg_chunks =  int(unit_size_s / acg_chunk_size)
    all_recs = []
    # Parameters
    fs = 30_000
#    exclusion_quantile = 0.02
#    amples_fr = unit_size_s * fs
    c_bin = 0.2
    c_win = 100
#    violations_ms = 0.8
#    rpv_threshold = 0.05
#    missing_spikes_threshold=5
#    taur = 0.0015
    samples_fr = unit_size_s * fs
#    tauc = 0.0005
    spikes_threshold = 300

    routines_mem = dp/'routinesMemory'
    # Create alternative dir for routines
    if routines_mem.is_dir() == False:
        routines_mem.mkdir()

    # Load kilosort aux files
    amplitudes_sample = np.load(dp/'amplitudes.npy')  # shape N_tot_spikes x 1
    spike_times = np.load(dp/'spike_times.npy')  # in samples
    spike_clusters = np.load(dp/'spike_clusters.npy')

    # Parameters
    # Extract good units of current sample
    good_units = get_units(dp, quality='good')
    all_units = get_units(dp)
    n = 0
    x = 0

       # steps:
        # split into 10 second chunks
        # run 3 concatenated 10 sec chunks through the filter
        # find all the chunks that passed the filter
        # find the time winodws where thes chunks happened
        # extract the spike times again from these consecutive windows
        # append the spike times from these consecutiv windowws
        # calculate features on this array

    chunk_acg_qual = np.zeros((no_acg_chunks,3)).astype('int')
    chunk_gauss_qual = np.zeros((no_gauss_chunks,3)).astype('int')

    # Unit spikes during first 20 minutes
    if spike_clusters[spike_clusters == unit].shape[0] > spikes_threshold:
        trn_samples_unit_20 = trn(dp, unit=unit,prnt=False, subset_selection=[(0, unit_size_s)], enforced_rp=0, again=True)
        trn_ms_unit_20 = trn_samples_unit_20 * 1. / (fs * 1. / 1000)
        spikes_unit_20 = len(trn_ms_unit_20)

        # Extract peak channel of current unit (where the deflection is maximum)
        # here we use the peal channel used by KS and not our peak

        if spikes_unit_20 > spikes_threshold:

            # Create a local path to store this unit info

            paths = [routines_mem/'temp_features']

            for pathi in paths:
                if pathi.is_dir() == False:
                    pathi.mkdir()

            # Unit amplitudes
            amplitudes_unit = amplitudes_sample[spike_clusters == unit]
            spike_times_unit = spike_times[spike_clusters == unit]
            unit_mask_20 = (spike_times_unit <= samples_fr)
            spike_times_unit_20 = spike_times_unit[unit_mask_20]
            amplitudes_unit_20 = amplitudes_unit[unit_mask_20]
            # now we have all the spikes in the first 20 min for the unit
            # split the recording into 10 second chunks 
            # find the quality of each chunk
            #
            # first look at the Gaussian filtering and the chunks made for this 

            for chunk_id in range(no_gauss_chunks):
               # find the spikes that happened in this period of time
               # get the ampllitude values for these spikes
               # fit the gaussian model to this
                chunk_start_time = chunk_id * gauss_chunk_size
                chunk_end_time = (chunk_id + 3) * gauss_chunk_size
                chunk_start_samples = chunk_start_time * fs
                chunk_end_samples = chunk_end_time * fs

                chunk_mask = (chunk_start_samples <= spike_times_unit_20) & \
                                 (spike_times_unit_20 < chunk_end_samples)
                chunk_mask = chunk_mask.reshape(len(spike_times_unit_20), )

                # Chunk amplitudes
                amplitudes_chunk = amplitudes_unit_20[chunk_mask]
                amplitudes_chunk = np.asarray(amplitudes_chunk, dtype='float64')

                if amplitudes_chunk.shape[0] > 15:

                    chunk_bins = estimate_bins(amplitudes_chunk, rule='Fd')

                    # % of missing spikes per chunk
                    if chunk_bins> 3:
                        x_c, p0_c, min_amp_c, n_fit_c, n_fit_no_cut_c, chunk_spikes_missing = gaussian_amp_est(amplitudes_chunk, chunk_bins)
                        if (~np.isnan(chunk_spikes_missing)) & (chunk_spikes_missing <= missing_spikes_threshold):
                            chunk_gauss_qual[chunk_id] = [chunk_start_time, chunk_end_time, 1]



            # next loop through the chunks made for the ACG extraction
            # get the periods where the ACG filter passed

            for chunk_id in range(no_acg_chunks):
               # find the spikes that happened in this period of time
               # get the ampllitude values for these spikes
               # fit the gaussian model to this
                chunk_start_time = chunk_id * acg_chunk_size
                chunk_end_time = (chunk_id + 3) * acg_chunk_size
                chunk_start_samples = chunk_start_time * fs
                chunk_end_samples = chunk_end_time * fs

                chunk_mask = (chunk_start_samples <= spike_times_unit_20) & \
                                 (spike_times_unit_20 < chunk_end_samples)
                chunk_mask = chunk_mask.reshape(len(spike_times_unit_20), )

                # Chunk amplitudes
                amplitudes_chunk = amplitudes_unit_20[chunk_mask]
                amplitudes_chunk = np.asarray(amplitudes_chunk, dtype='float64')
                if amplitudes_chunk.shape[0] > 15:
                        block_ACG = acg(dp, unit, c_bin, c_win, prnt = False,  subset_selection=[(chunk_start_time, chunk_end_time)])
                        x_block = np.linspace(-c_win * 1. / 2, c_win * 1. / 2, block_ACG.shape[0])
                        y_block = block_ACG.copy()
#                        y_lim1_unit = 0
#                        yl_unit = max(block_ACG)
#                        y_lim2_unit = int(yl_unit) + 5 - (yl_unit % 5)

                        # Find refractory period violations
                        booleanCond = np.zeros(len(y_block), dtype=np.bool)
                        # find periods where the 
                        booleanCond[(x_block >= -violations_ms) & (x_block <= violations_ms)] = True
                        violations = y_block[booleanCond]
                        violations_mean = np.mean(violations)

                        # Select normal refractory points in the ACG
                        booleanCond2 = np.zeros(len(y_block), dtype=np.bool)
                        booleanCond2[:80] = True
                        booleanCond2[-80:] = True
                        normal_obs = y_block[booleanCond2]
                        normal_obs_mean = np.mean(normal_obs)
                        # Compute ACG ratio to account for refractory violations
                        rpv_ratio_acg = round(violations_mean / normal_obs_mean, 2)
                        if (rpv_ratio_acg <= rpv_threshold):
                            chunk_acg_qual[chunk_id] = [chunk_start_time, chunk_end_time, 1]
    # start at thi col
    # have all the good chunks 
    # find sequences where there are more than 3  
    # run nsliding window over values, when the sum gets to 3
    # drop the middle value

    if (np.sum(chunk_acg_qual[:,2])  > 5) & (np.sum(chunk_gauss_qual[:,2]) > 5):

        # create sum of rolling windows
        # by stacking the vecotrs with overlap and summing across the vectors



        # get where the values of the rolling sums are 0
        # add 1 to the indices of where the rolling sums are 0
        # to get the index of the middle value, which is 'bad'
        # alternatively, check if any sections are bad, not all sections


        # input values to chunk_acg_qual and chunk_gauss_qual
        # this is needed bc I only have the values with positive results recorded
        # I can also remove the following pieces of code, as the 
        # arrays are already initiated at 0, and the size is correct 
        # so the roll_sum summing will work

        chunk_gauss_qual  = np.array(chunk_gauss_qual)
        chunk_acg_qual = np.array(chunk_acg_qual)

        acg_array = chunk_acg_qual[:,2]
        gauss_array = chunk_gauss_qual[:,2]


        # index into the vectors to effectively create sliding windows
        # use fancy indexing, by first making a matrix of vector indices
        # to be used. Then use this matrix to index the vector

        acg_indexer = np.arange(no_acg_chunks - acg_window_len)[None,:] + np.arange(acg_window_len)[:,None]

        roll_sum_acg = acg_array[acg_indexer].sum(axis = 0).reshape(1,-1)

        gauss_indexer = np.arange(no_gauss_chunks - gauss_window_len)[None, :] + np.arange(gauss_window_len)[:,None]

        roll_sum_gauss = gauss_array[gauss_indexer].sum(axis = 0).reshape(1,-1)

        # Use OR or AND operator to find the sections of one filtering
        # process that qualify
        if use_or_operator:
            good_vals_acg = np.where(np.any(roll_sum_acg, axis = 0))[0] +1
            good_vals_gauss = np.where(np.any(roll_sum_gauss, axis =0))[0] +1
        else:
            good_vals_acg = np.where(np.all(roll_sum_acg, axis = 0))[0] +1
            good_vals_gauss = np.where(np.all(roll_sum_gauss, axis =0))[0] +1

        # take the intersection of these two 'good' sets to 
        # get where both filters were passed
        # get the times where each passed individually and look for intersections
        # get the seconds where each passed, look at the intersection of these two lists
        # for both filters want a long list of second start times where 
        # the filter was passed

        # get the good acg seconds
        good_acg_sec = []
        for i in good_vals_acg:
            good_acg_sec.append(list(np.linspace(i*acg_chunk_size,(i+1)*acg_chunk_size-1, acg_chunk_size ).astype('int')))
        good_acg_sec = np.array(good_acg_sec).flatten()

        # if the array is empty add a single 0 to it, so it doesn't break elsewhere
        if len(good_acg_sec) ==0:
            good_acg_sec = np.array([0])
        # get the good gauss seconds
        good_gauss_sec = []
        for i in good_vals_gauss:
            good_gauss_sec.append(list(np.linspace(i*gauss_chunk_size, (i+1)*gauss_chunk_size -1, gauss_chunk_size).astype('int')))
        good_gauss_sec = np.array(good_gauss_sec).flatten()

        # if the array is empty add a single 0 to it, so it doesn't break elsewhere
        if len(good_gauss_sec) == 0:
            good_gauss_sec = np.array([0])
        acg_start_end  = get_consec_sections(good_acg_sec)
        # get the seconds where both the intervals overlap
        good_sec = np.array(list(set(good_gauss_sec) & set(good_acg_sec) ))

        # if the array is empty add a single 0 to it, so it doesn't break elsewhere
        if len(good_sec) == 0:
            good_sec = np.array([0])
        gauss_start_end  = get_consec_sections(good_gauss_sec)

        # get the good overlapping seconds
        good_start_end  = get_consec_sections(good_sec)
        if save:
            np.save(Path(dprm,fn), np.array( (np.array(good_start_end), np.array(acg_start_end), np.array(gauss_start_end)), dtype = object))
        return good_start_end, acg_start_end, gauss_start_end
    else:
        zeros3 = np.zeros((1,3), dtype = np.int8)
        if save:
            np.save(Path(dprm,fn), np.array((zeros3, zeros3, zeros3), dtype = object)  )
        return zeros3, zeros3, zeros3

def get_consec_sections(seconds):
        """
        Given an array with seconds as entries (with 1 sec increments)

        Return: list of consecutive sections start and end times
        """
        sec_all = []
        for k, g in groupby(enumerate(seconds), lambda ix: ix[0]-ix[1]):
            sec_all.append(list( map(itemgetter(1), g)))
        start_end = []

        # get the start and end times of these section
        for good_section in sec_all:
            start_end.append([good_section[0], good_section[-1]])

        return start_end


def trn_filtered(dp, unit, first_n_minutes=20, consecutive_n_seconds = 180, acg_window_len=3, acg_chunk_size = 10, gauss_window_len = 3, gauss_chunk_size = 10, use_or_operator = False, use_consecutive = True, prnt = False, again = False, save = True):

    goodsec, acgsec, gausssec = train_quality(dp, unit, first_n_minutes, acg_window_len, acg_chunk_size, gauss_window_len, gauss_chunk_size, use_or_operator, again=again, save = save, prnt =prnt)

    """
    High level function for getting the spike ids for the spikes that passed
    both filtering criteria in train_quality function.
    Reurns: spike ids passing filters
    """

    good_sec = []
    for i in goodsec:
        good_sec.append(list(range(i[0], i[1]+1)))
    good_sec = np.hstack((good_sec))
    if good_sec.shape[0] > 5:
        # chunks that pass out criteria
        all_spikes = []


        # condition, if the number of consecutive chunks is 
        # higehr than the 3min threshold, extract it
        # Otherwise leave the unit

        # need to extract the spike times again for chunks and concat
        # extract the spike train and append it to one array
        if use_consecutive:

            # get the longest consecutive section of time that passes
            # both our criteria
            maxrun = -1
            run_len = {}
            for x in good_sec:
                mrun = run_len[x] = run_len.get(x-1, 0) + 1
                #print x-run+1, 'to', x
                if mrun > maxrun:
                    maxend, maxrun = x, mrun
            consecutive_good_chunk = list(range(maxend-maxrun+1, maxend+1))
            consecutive_good_chunk_len = len(consecutive_good_chunk)
            if consecutive_good_chunk_len > consecutive_n_seconds:
                all_trn = trn(dp, unit, prnt = False, subset_selection = [(consecutive_good_chunk[0], consecutive_good_chunk[-1]+1)])
                all_spikes_return = all_trn
                # this is the output we need, the spikes that are in
                # the 3 min period that is classified good
                return all_spikes_return#, good_vals_acg, good_acg_sec

            else:
                if prnt: print('No consecutive section passed the filters')
                return np.array([0])
        else:
            # get all the ranges of spikes
            all_consecutive_ranges =[]
            for k,g in groupby(enumerate(good_sec),lambda x:x[0]-x[1]):
                range_group = (map(itemgetter(1),g))
                range_group = list(map(int,range_group))
                all_consecutive_ranges.append((range_group[0],range_group[-1]))
            for train_secs in all_consecutive_ranges:
                curr_spikes = trn(dp, unit, prnt = False, subset_selection = [(train_secs[0], train_secs[1])])
                all_spikes.append(curr_spikes)
            all_spikes_return = np.hstack(all_spikes)

            return all_spikes_return#, good_vals_acg, good_acg_sec
    else:
        return np.array([0])


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



from  npyx.corr import acg
