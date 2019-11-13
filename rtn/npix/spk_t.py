# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os, sys
import os.path as op
import ast

import numpy as np
                    
from rtn.npix.gl import get_units, assert_multidatasets

def binarize(X, bin_size, fs, rec_len=None, constrainBin=False):
    '''Function to turn a spike train (array of time stamps)
       into a binarized spike train (array of 0 or 1 
                                     of length rec_len with a given bin_size.).
       - X: spike train (array of time stamps, in samples sampled at fs Hertz)
       - bin_size: size of binarized spike train bins, in milliseconds.
       - rec_len: length of the recording, in SAMPLES. If not provided, time of the last spike.
       - fs: sampling frequency, in Hertz.'''
    
    # Process bin_size
    if bin_size>1:
        print('''/!\ Provided binsize>1ms! 
              It is likely that more than one spike will be binned together.''')
        if constrainBin:
            print('->>> Bin size set at 1ms.')
            bin_size=1
            bin_size = np.clip(bin_size, 1000/fs, 1)  # in milliseconds
    bin_size = int(np.ceil(fs * float(bin_size)/1000))  # Conversion ms->samples
    
    # Process rec_len
    if rec_len==None:
        rec_len=X[-1]
    
    # Binarize spike train
    if bin_size==0:
        (X_unq, X_cnt) = np.unique(X, return_counts=True)
        if np.any(X_cnt>=2):
            n_redundant = np.count_nonzero(X_cnt>=2)
            print('''/!\ {} spikes were present more than once in the provided train.'''.format(n_redundant))
        del X_cnt
    else:
        X_unq = X
    Xb = np.histogram(X_unq, bins=np.arange(0, rec_len, bin_size))[0]
    return Xb


def ids(dp, unit, ret=True, sav=True, prnt=False):
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
    '''
    assert unit in get_units(dp)
    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(dprm+'/ids{}.npy'.format(unit)):
        if prnt: print("File ids{}.npy found in routines memory.".format(unit))
        indices = np.load(dprm+'/ids{}.npy'.format(unit))
        indices=np.asarray(indices, dtype='int64')
    # if not, compute it
    else:
        if prnt: 
            print("File ids{}.npy not found in routines memory. Will be computed from source files.".format(unit))
        if type(unit)==str:
            ds_i, unt = unit.split('_'); ds_i, unt = ast.literal_eval(ds_i), ast.literal_eval(unt)
            spike_clusters_samples = np.load(op.join(dp, 'merged_clusters_spikes.npz'))
            spike_clusters_samples=spike_clusters_samples[list(spike_clusters_samples.keys())[0]]
            dataset_mask=(spike_clusters_samples[:, 0]==ds_i); unit_mask=(spike_clusters_samples[:, 1]==unt)
            indices = np.nonzero(dataset_mask&unit_mask)[0]
            indices=np.reshape(indices, (max(indices.shape), ))
        elif type(unit)==int:
            spike_clusters = np.load(dp+"/spike_clusters.npy")
            indices = np.nonzero(spike_clusters==unit)[0]
            indices=np.reshape(indices, (max(indices.shape), ))
                
        # Save it
        if sav:
            np.save(dprm+'/ids{}.npy'.format(unit), indices)
    # Either return or draw to global namespace
    return indices

    

def trn(dp, unit, ret=True, sav=True, prnt=False, rec_section='all', fs=30000):
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
    - rec_section = 'all' or [(t1, t2), ...] with t1, t2 in seconds.
    '''
    assert unit in get_units(dp), "Unit fed to trn function not found in dataset."
    
    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(dprm+'/trn{}({}).npy'.format(unit, str(rec_section)[0:10].replace(' ', ''))):
        if prnt: print("File /trn{}({}).npy found in routines memory.".format(unit, str(rec_section)[0:10].replace(' ', '')))
        train = np.load(dprm+'/trn{}({}).npy'.format(unit, str(rec_section)[0:10].replace(' ', '')))
        train=np.asarray(train, dtype='int64')
    # if not, compute it
    else:
        if prnt:
            print("File trn{}.npy not found in routines memory. Will be computed from source files.".format(unit))
        
        if type(unit)==str:
            ds_i, unt = unit.split('_'); ds_i, unt = ast.literal_eval(ds_i), ast.literal_eval(unt)
            spike_clusters_samples = np.load(op.join(dp, 'merged_clusters_spikes.npz'))
            spike_clusters_samples=spike_clusters_samples[list(spike_clusters_samples.keys())[0]]
            dataset_mask=(spike_clusters_samples[:, 0]==ds_i); unit_mask=(spike_clusters_samples[:, 1]==unt)
            train = spike_clusters_samples[dataset_mask&unit_mask, 2]
            train=np.reshape(train, (max(train.shape), ))
        elif type(unit)==int:
            spike_clusters = np.load(op.join(dp,"spike_clusters.npy"))
            spike_samples = np.load(op.join(dp,'spike_times.npy'))
            train = spike_samples[spike_clusters==unit]
            train=np.reshape(train, (max(train.shape), ))
        
        # Optional selection of a section of the recording.
        if type(rec_section)==str:
            if rec_section=='all':
                pass # eq to rec_section=[(0, spike_samples[-1])] # in samples
        else:
            sec_bool=np.zeros(len(train), dtype=np.bool)
            for section in rec_section:
                sec_bool[(train>=section[0]*fs)&(train<=section[1]*fs)]=True # comparison in samples
            train=train[sec_bool]
                
        train=np.asarray(train, dtype='int64')
        # Save it
        if sav:
            np.save(dprm+'/trn{}({}).npy'.format(unit, str(rec_section)[0:10].replace(' ', '')), train)
    # Either return or draw to global namespace
    return train


def isi(dp, unit, ret=True, sav=True, prnt=False, rec_section='all', fs=30000):
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
    dprm = dp+'/routinesMemory'
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(dprm+'/isi{}({}).npy'.format(unit, str(rec_section)[0:10].replace(' ', ''))):
        if prnt: print("File /isi{}({}).npy found in routines memory.".format(unit, str(rec_section)[0:10].replace(' ', '')))
        isitvl = np.load(dprm+'/isi{}({}).npy'.format(unit, str(rec_section)[0:10].replace(' ', '')))
        isitvl=np.asarray(isitvl, dtype='float64')
    # if not, compute it
    else:
        if prnt: print("File isi{}.npy not found in routines memory. Will be computed from source files".format(unit))
        train = trn(dp, unit, ret=True, sav=sav, prnt=prnt, rec_section=rec_section)
        train = train*1./(fs*1./1000) # Conversion from samples to ms
        isitvl = np.diff(train) # in ms
        isitvl=np.asarray(isitvl, dtype='float64')
        
        # Save it
        if sav:
            np.save(dprm+'/isi{}({}).npy'.format(unit, str(rec_section)[0:10].replace(' ', '')), isitvl)
    # Either return or draw to global namespace
    return isitvl
        


def trnb(dp, unit, bin_size, ret=True, sav=True, prnt=False, constrainBin=False, rec_section='all', fs=30000):
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
    dprm = dp+'/routinesMemory'
    if not op.isdir(dprm): os.makedirs(dprm)
    if op.exists(dprm+'/trnb{}_{}({}).npy'.format(unit, bin_size, str(rec_section)[0:10].replace(' ', ''))):
        if prnt: print("File trnb{}_{}.npy found in routines memory.".format(unit, bin_size))
        train_binarized = np.load(dprm+'/trnb{}_{}({}).npy'.format(unit, bin_size, str(rec_section)[0:10].replace(' ', '')))
        train_binarized = np.asarray(train_binarized, dtype='int64')
    # if not, compute it
    else:
        if prnt: 
            print('''File trnb{}_{}.npy not found in routines memory.
              Will be computed from source files.'''.format(unit, bin_size))
        train = trn(dp, unit, ret=True, sav=False, rec_section=rec_section)
        phy_st = np.load(dp+'/spike_times.npy')
        last_st = phy_st[-1,0] # in samples
        del phy_st
        train_binarized = binarize(train, bin_size, fs=fs, rec_len=last_st, constrainBin=constrainBin)
        train_binarized = np.asarray(train_binarized, dtype='int16') #0s and 1s -> int8 to save memory
        # Save it
        if sav:
            np.save(dprm+'/trnb{}_{}({}).npy'.format(unit, bin_size, str(rec_section)[0:10].replace(' ', '')), train_binarized)
    # Either return or draw to global namespace
    return train_binarized
