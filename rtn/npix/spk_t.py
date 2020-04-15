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
                    
from rtn.npix.gl import get_units
from rtn.npix.io import read_spikeglx_meta

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
    if rec_len==None:
        rec_len=X[-1]
    
    # Binarize spike train
    Xb = np.histogram(X, bins=np.arange(0, rec_len, bin_size))[0]
    
    # Decrease array size as much as possible
    for encode in [32,16,8]:
        if not np.all(Xb==Xb.astype('int{}'.format(encode))):
            break
        Xb=Xb.astype('int{}'.format(encode))
    
    return Xb


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
        indices = np.load(Path(dprm,fn))
        indices=np.asarray(indices, dtype='int64')
    # if not, compute it
    else:
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
            indices=np.reshape(indices, (max(indices.shape), ))
        if type(unit) not in [str, np.str_, int]:
            print('WARNING unit {} type ({}) not handled!'.format(unit, type(unit)))
            return
        
        # Optional selection of a section of the recording.
        if type(subset_selection) not in [str,np.str_]: # else, eq to subset_selection=[(0, spike_samples[-1])] # in samples
            try: subset_selection[0][0]
            except: raise TypeError("ERROR subset_selection should be either a string or a list of format [(t1, t2), (t3, t4), ...]!!")
            fs=read_spikeglx_meta(dp)['sRateHz']
            train=trn(dp, unit, subset_selection=subset_selection)
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
    fn='trn{}({}).npy'.format(unit, str(subset_selection)[0:10].replace(' ', ''))
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        train = np.load(Path(dprm,fn))
        train=np.asarray(train, dtype='int64')
    # if not, compute it
    else:
        if prnt:
            print("File {} not found in routines memory. Will be computed from source files.".format(fn))
        
        assert unit in get_units(dp), 'WARNING unit {} not found in dataset {}!'.format(unit, dp)
        if type(unit) in [str, np.str_]:
            ds_i, unt = unit.split('_'); ds_i, unt = ale(ds_i), ale(unt)
            ds_table=pd.read_csv(Path(dp, 'datasets_table.csv'), index_col='dataset_i')
            if ds_table.shape[0]>1: # If several datasets in prophyler
                spike_clusters_samples = np.load(Path(dp, 'merged_clusters_spikes.npy'))
                dataset_mask=(spike_clusters_samples[:, 0]==ds_i); unit_mask=(spike_clusters_samples[:, 1]==unt)
                train = spike_clusters_samples[dataset_mask&unit_mask, 2]
                train=np.reshape(train, (max(train.shape), )).astype(np.int64)
            else:
                spike_clusters = np.load(Path(ds_table['dp'][0],"spike_clusters.npy"))
                spike_samples = np.load(Path(ds_table['dp'][0],'spike_times.npy'))
                train = spike_samples[spike_clusters==unt]
                train=np.reshape(train, (max(train.shape), )).astype(np.int64)
        else:
            try:unit=int(unit)
            except:pass
        if type(unit) is int:
            spike_clusters = np.load(Path(dp,"spike_clusters.npy"))
            spike_samples = np.load(Path(dp,'spike_times.npy'))
            train = spike_samples[spike_clusters==unit]
            train=np.reshape(train, (max(train.shape), )).astype(np.int64)
        
        if type(unit) not in [str, np.str_, int]:
            print('WARNING unit {} type ({}) not handled!'.format(unit, type(unit)))
            return
        
        # Filter out spike duplicates (spikes following an ISI shorter than enforced_rp)
        fs=read_spikeglx_meta(dp)['sRateHz']
        train=train[np.append(True, np.diff(train)>=enforced_rp*fs/1000)]
        
        # Optional selection of a section of the recording.
        if type(subset_selection) not in [str,np.str_]: # else, eq to subset_selection=[(0, spike_samples[-1])] # in samples
            try: subset_selection[0][0]
            except: raise TypeError("ERROR subset_selection should be either a string or a list of format [(t1, t2), (t3, t4), ...]!!")
            fs=read_spikeglx_meta(dp)['sRateHz']
            sec_bool=np.zeros(len(train), dtype=np.bool)
            for section in subset_selection:
                sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
            train=train[sec_bool]
                
        # Save it
        if sav:
            np.save(Path(dprm,fn), train)
    # Either return or draw to global namespace
    return train

def mfr(dp, unit, exclusion_quantile=0.005, sav=True, prnt=False, subset_selection='all', again=False):
    i = isi(dp, unit, sav, prnt, subset_selection, 30000, again) # output in ms
    # Remove outlyers
    i=i[(i>=np.quantile(i, exclusion_quantile))&(i<=np.quantile(i, 1-exclusion_quantile))]
    
    return 1000./np.mean(i)

def isi(dp, unit, sav=True, prnt=False, subset_selection='all', fs=30000, again=False):
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
    # Search if the variable is already saved in dp/routinesMemory
    dprm = Path(dp,'routinesMemory')
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(dprm+'/isi{}({}).npy'.format(unit, str(subset_selection)[0:10].replace(' ', ''))) and not again:
        if prnt: print("File /isi{}({}).npy found in routines memory.".format(unit, str(subset_selection)[0:10].replace(' ', '')))
        isitvl = np.load(dprm+'/isi{}({}).npy'.format(unit, str(subset_selection)[0:10].replace(' ', '')))
        isitvl=np.asarray(isitvl, dtype='float64')
    # if not, compute it
    else:
        if prnt: print("File isi{}.npy not found in routines memory. Will be computed from source files".format(unit))
        train = trn(dp, unit, sav, prnt, subset_selection, fs, again)
        if train is None:
            print('Train is none!!', unit)
        train = train*1./(fs*1./1000) # Conversion from samples to ms
        isitvl = np.diff(train) # in ms
        isitvl=np.asarray(isitvl, dtype='float64')
        
        # Save it
        if sav:
            np.save(dprm+'/isi{}({}).npy'.format(unit, str(subset_selection)[0:10].replace(' ', '')), isitvl)
    # Either return or draw to global namespace
    return isitvl
        


def trnb(dp, unit, bin_size, sav=True, prnt=False, subset_selection='all', fs=30000, again=False):
    '''
    ********
    routine from routines_spikes
    computes binarized spike train (1, Nspikes) - int64, in samples
    ********
    
    - dp (string): DataPath to the Neuropixels dataset.
    - u (int): unit index
    - bin_size: size of binarized spike train bins, in milliseconds.
    - rec_len: length of the recording, in seconds. If not provided, time of the last spike.
    - ret (bool - default False): if True, train returned by the routine.
      If False, by definition of the routine, drawn to global namespace.
    - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
    '''

    # Search if the variable is already saved in dp/routinesMemory
    dprm = Path(dp,'routinesMemory')
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(dprm+'/trnb{}_{}({}).npy'.format(unit, bin_size, str(subset_selection)[0:10].replace(' ', ''))) and not again:
        if prnt: print("File trnb{}_{}.npy found in routines memory.".format(unit, bin_size))
        train_binarized = np.load(dprm+'/trnb{}_{}({}).npy'.format(unit, bin_size, str(subset_selection)[0:10].replace(' ', '')))
        train_binarized = np.asarray(train_binarized, dtype='int64')
    # if not, compute it
    else:
        if prnt: 
            print('''File trnb{}_{}.npy not found in routines memory.
              Will be computed from source files.'''.format(unit, bin_size))
        train = trn(dp, unit, subset_selection=subset_selection)
        phy_st = np.load(dp+'/spike_times.npy')
        last_st = phy_st[-1,0] # in samples
        del phy_st
        train_binarized = binarize(train, bin_size, fs=fs, rec_len=last_st)
        train_binarized = np.asarray(train_binarized, dtype='int16') #0s and 1s -> int8 to save memory
        # Save it
        if sav:
            np.save(dprm+'/trnb{}_{}({}).npy'.format(unit, bin_size, str(subset_selection)[0:10].replace(' ', '')), train_binarized)
    # Either return or draw to global namespace
    return train_binarized
