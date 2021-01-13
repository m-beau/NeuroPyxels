# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os
import os.path as op; opj=op.join
from pathlib import Path
from ast import literal_eval as ale

import numpy as np
import pandas as pd
                    
from npyx.utils import smooth, thresh_consec, npa
from npyx.gl import get_units
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
    fn='ids{}({}).npy'.format(unit, str(subset_selection)[0:10].replace(' ', ''))
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        return np.asarray(np.load(Path(dprm,fn)), dtype='int64')
    # if not, compute it
    if prnt:
        print("File {} not found in routines memory. Will be computed from source files.".format(fn))
    if type(unit) in [str, np.str_]:
        ds_i, unt = unit.split('_'); ds_i, unt = ale(ds_i), ale(unt)
        spike_clusters_samples = np.load(Path(dp, 'merged_clusters_spikes.npy'))
        dataset_mask=(spike_clusters_samples[:, 0]==ds_i); unit_mask=(spike_clusters_samples[:, 1]==unt)
        indices = np.nonzero(dataset_mask&unit_mask)[0]
        indices=np.reshape(indices, (max(indices.shape), ))
    else:
        try:unit=int(unit)
        except:pass
    if type(unit) is int:
        spike_clusters = np.load(Path(dp,"spike_clusters.npy"))
        indices = np.nonzero(spike_clusters==unit)[0]
    if type(unit) not in [str, np.str_, int]:
        print('WARNING unit {} type ({}) not handled!'.format(unit, type(unit)))
        return
    
    # Optional selection of a section of the recording.
    if type(subset_selection) not in [str,np.str_]: # else, eq to subset_selection=[(0, spike_samples[-1])] # in samples
        try: subset_selection[0][0]
        except: raise TypeError("ERROR subset_selection should be either a string or a list of format [(t1, t2), (t3, t4), ...]!!")
        fs=read_spikeglx_meta(dp)['sRateHz']
        train=trn(dp, unit, again=again)
        sec_bool=np.zeros(len(train), dtype=np.bool)
        for section in subset_selection:
            sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
        indices=indices[sec_bool]
    
    # Save it
    if sav:
        np.save(Path(dprm,fn), indices)
    # Either return or draw to global namespace
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
    fn='trn{}({})_{}.npy'.format(unit, str(subset_selection)[0:10].replace(' ', ''), enforced_rp)
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        train = np.load(Path(dprm,fn))

    # if not, compute it
    else:
        if prnt:
            print(f"File {fn} not found in routines memory. Will be computed from source files.")
        
        assert unit in get_units(dp), f'WARNING unit {unit} not found in dataset {dp}!'
        if type(unit) in [str, np.str_]:
            ds_i, unt = unit.split('_'); ds_i, unt = ale(ds_i), ale(unt)
            ds_table=pd.read_csv(Path(dp, 'datasets_table.csv'), index_col='dataset_i')
            if ds_table.shape[0]>1: # If several datasets in prophyler
                spike_clusters_samples = np.load(Path(dp, 'merged_clusters_spikes.npy'))
                dataset_mask=(spike_clusters_samples[:, 0]==ds_i); unit_mask=(spike_clusters_samples[:, 1]==unt)
                train = spike_clusters_samples[dataset_mask&unit_mask, 2].astype(np.int64)
            else:
                spike_clusters = np.load(Path(ds_table['dp'][0],"spike_clusters.npy"))
                spike_samples = np.load(Path(ds_table['dp'][0],'spike_times.npy'))
                train = spike_samples[spike_clusters==unt]
        else:
            try:unit=int(unit)
            except:pass
        if type(unit) is int:
            spike_clusters = np.load(Path(dp,"spike_clusters.npy"), mmap_mode='r')
            spike_samples = np.load(Path(dp,'spike_times.npy'), mmap_mode='r')
            train = spike_samples[spike_clusters==unit].ravel()
        
        if type(unit) not in [str, np.str_, int]:
            raise TypeError(f'WARNING unit {unit} type ({type(unit)}) not handled!')
        
        # Filter out spike duplicates (spikes following an ISI shorter than enforced_rp)
        fs=read_spikeglx_meta(dp)['sRateHz']
        train=train[np.append(True, np.diff(train)>=enforced_rp*fs/1000)]
        
        # Optional selection of a section of the recording.
        if type(subset_selection) not in [str,np.str_]: # else, eq to subset_selection=[(0, spike_samples[-1])] # in samples
            try: subset_selection[0][0]
            except: raise TypeError("ERROR subset_selection should be either a string or a list of format [(t1, t2), (t3, t4), ...]!!")
            sec_bool=np.zeros(len(train), dtype=np.bool)
            for section in subset_selection:
                sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
            train=train[sec_bool]
                
        # Save it
        if sav:
            np.save(Path(dprm,fn), train)
    # Either return or draw to global namespace
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
        train=np.asarray(train)
        assert train.ndim==1
        t=train
        
    assert b>=1000/fs
    tb = binarize(t, b, fs, t_end)
    sd=int(sd/b) # convert from ms to bin units
    b_s=b/1000 # bin seconds
    tbs=smooth(tb, 'gaussian', sd=sd)/b_s # result is inst. firing rate in Hz - speed bottleneck
    fr_th=mfr(dp, u)*th
    
    periods = thresh_consec(tbs, fr_th, sgn=1, n_consec=0, exclude_edges=False, only_max=False, ret_values=False)
    periods=np.array(periods)*(b_s*fs) # conversion from bins to samples
    
    if sav:
        np.save(Path(dprm,fn), periods)
    
    return periods
