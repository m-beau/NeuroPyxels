# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os, sys

import numpy as np

from rtn.utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of
from rtn.npix.gl import get_units

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


def ids(dp, u, ret=True, sav=True, prnt=True):
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
    global indices
    # Preformat
    u, dp = int(u), str(dp)

    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(dprm+'/ids{}.npy'.format(u)):
        if prnt: print("File ids{}.npy found in routines memory.".format(u))
        indices = np.load(dprm+'/ids{}.npy'.format(u))
        indices=np.asarray(indices, dtype='int64')
    # if not, compute it
    else:
        dpme = dp+'/manualExports'
        fn="/spike_ids{}.npy".format(u)
        if os.path.exists(dpme+fn):
            indices = np.load(dpme+fn)
            if prnt: print("File spike_ids{}.npy found in phy manual exports.".format(u))
        else:
            if prnt: 
                print('''File spike_ids{}.npy not found in phy manual exports. Will be computed from source files. 
                  Reminder: perform phy export of selected units ids with :export_ids.'''.format(u))
            spike_clusters = np.load(dp+"/spike_clusters.npy")
            indices = np.nonzero(spike_clusters==u)[0]
            indices=np.reshape(indices, (max(indices.shape), ))
                
        # Save it
        if sav:
            np.save(dprm+'/ids{}.npy'.format(u), indices)
    # Either return or draw to global namespace
    if ret:
        ids=indices.copy()
        del indices
        return ids
    else:
        # if prnt: print("ids{} defined into global namespace.".format(u))
        # exec("ids{} = indices".format(u), globals())
        del indices

    

def trn(dp, u, ret=True, sav=True, prnt=True, rec_section='all'):
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
    global train
    # Preformatpython how to save memory numpy array
    u, dp = int(u), str(dp)
    assert u in get_units(dp)
    fs=30000
    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(dprm+'/trn{}({}).npy'.format(u, str(rec_section)[0:50].replace(' ', ''))):
        if prnt: print("File /trn{}({}).npy found in routines memory.".format(u, str(rec_section)[0:50].replace(' ', '')))
        train = np.load(dprm+'/trn{}({}).npy'.format(u, str(rec_section)[0:50].replace(' ', '')))
        train=np.asarray(train, dtype='int64')
    # if not, compute it
    else:
        if prnt: print("File trn{}.npy not found in routines memory.".format(u))
        dpme = dp+'/manualExports'
        fn="/spike_samples{}.npy".format(u)
        if os.path.exists(dpme+fn):
            train = np.load(dpme+fn)
            if prnt: print("File spike_samples{}.npy found in phy manual exports.".format(u))
        else:
            if prnt: 
                print('''File spike_samples{}.npy not found in phy manual exports. 
                  Will be computed from source files. 
                  Reminder: perform phy export of selected units samples with :export_samples.'''.format(u))
            spike_clusters = np.load(dp+"/spike_clusters.npy")
            spike_samples = np.load(dp+'/spike_times.npy')
            train = spike_samples[spike_clusters==u]
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
            np.save(dprm+'/trn{}({}).npy'.format(u, str(rec_section)[0:50].replace(' ', '')), train)
    # Either return or draw to global namespace
    if ret:
        trn=train.copy()
        del train
        return trn
    else:
        # if prnt: print("trn{} defined into global namespace.".format(u))
        # exec("trn{} = train".format(u), globals())
        del train


def isi(dp, u, ret=True, sav=True, prnt=True, rec_section='all'):
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
    global isitvl
    # Preformatpython how to save memory numpy array
    u, dp = int(u), str(dp)

    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(dprm+'/isi{}({}).npy'.format(u, str(rec_section)[0:50].replace(' ', ''))):
        if prnt: print("File /isi{}({}).npy found in routines memory.".format(u, str(rec_section)[0:50].replace(' ', '')))
        isitvl = np.load(dprm+'/isi{}({}).npy'.format(u, str(rec_section)[0:50].replace(' ', '')))
        isitvl=np.asarray(isitvl, dtype='float64')
    # if not, compute it
    else:
        if prnt: print("File isi{}.npy not found in routines memory. Will be computed from source files".format(u))
        sys.path.append(dp)
        from params import sample_rate as fs
        train = trn(dp, u, ret=True, sav=sav, prnt=prnt, rec_section=rec_section)
        train = train*1./(fs*1./1000) # Conversion from samples to ms
        isitvl = np.diff(train) # in ms
        isitvl=np.asarray(isitvl, dtype='float64')
        
        # Save it
        if sav:
            np.save(dprm+'/isi{}({}).npy'.format(u, str(rec_section)[0:50].replace(' ', '')), isitvl)
    # Either return or draw to global namespace
    if ret:
        isi=isitvl.copy()
        del isitvl
        return isi
    else:
        # if prnt: print("isi{} defined into global namespace.".format(u))
        # exec("isi{} = isitvl".format(u), globals())
        del isitvl
        


def trnb(dp, u, bin_size, ret=True, sav=True, prnt=True, constrainBin=False, rec_section='all'):
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
    global train_binarized
    # Preformat
    u, dp = int(u), str(dp)

    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(dprm+'/trnb{}_{}({}).npy'.format(u, bin_size, str(rec_section)[0:50].replace(' ', ''))):
        if prnt: print("File trnb{}_{}.npy found in routines memory.".format(u, bin_size))
        train_binarized = np.load(dprm+'/trnb{}_{}({}).npy'.format(u, bin_size, str(rec_section)[0:50].replace(' ', '')))
        train_binarized = np.asarray(train_binarized, dtype='int64')
    # if not, compute it
    else:
        if prnt: 
            print('''File trnb{}_{}.npy not found in routines memory.
              Will be computed from source files.'''.format(u, bin_size))
        train = trn(dp, u, ret=True, sav=False, rec_section=rec_section)
        phy_st = np.load(dp+'/spike_times.npy')
        last_st = phy_st[-1,0] # in samples
        del phy_st
        sys.path.append(dp)
        from params import sample_rate as fs
        train_binarized = binarize(train, bin_size, fs=fs, rec_len=last_st, constrainBin=constrainBin)
        train_binarized = np.asarray(train_binarized, dtype='int16') #0s and 1s -> int8 to save memory
        # Save it
        if sav:
            np.save(dprm+'/trnb{}_{}({}).npy'.format(u, bin_size, str(rec_section)[0:50].replace(' ', '')), train_binarized)
    # Either return or draw to global namespace
    if ret:
        trnb=train_binarized.copy()
        del train_binarized
        return trnb
    else:
        # if prnt: print("trnb{}_{} defined into global namespace.".format(u, bin_size))
        # exec("trnb{}_{} = train_binarized".format(u, bin_size), globals())
        del train_binarized
