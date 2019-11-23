# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

import os, ast
import os.path as op

import numpy as np
import pandas as pd

from rtn.utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of
                    
from rtn.npix.gl import get_units
from rtn.npix.spk_t import trn, trnb, isi, binarize
from rtn.npix.spk_wvf import get_depthSort_peakChans

import scipy.signal as sgnl
from statsmodels.nonparametric.smoothers_lowess import lowess
from rtn.stats import pdf_normal, fractile_normal

# cbin=0.2
# cwin=100
# ccg1=ccg('/media/maxime/Npxl_data2/wheel_turning/DK152-153/DK153_190416day1_Probe2_run1', [1820, 1654], cbin, cwin)
# ccg2=ccg('/media/maxime/Npxl_data2/wheel_turning/DK152-153/DK153_190416day1_Probe2_run1', [392, 945], cbin, cwin)


def pearson_corr(M):
    '''
    Calculate the NxN matrix of pairwise Pearson’s correlation coefficients 
    between all combinations of Ncells spike trains.
    # Parameters
    - M: binary trains matrix, Ncells x Nbins
    # Outputs
    - C: pairwise correlations matrix, Ncells x Ncells
    '''
    # Sanity checks
    #assert np.all((M<=1) & (M>=0)) # has to be a binary matrix
    
    # Formula where <bi-mi, bj-mj> is the dot product of mean-substracted tiem series
    # (essentially covariance of i and j)
    # C[i,j] = <bi-mi, bj-mj> / sqrt(<bi-mi, bi-mi>*<bj-mj, bj-mj>)
    # b are trn
    
    # mean substract raws (bi-mi and bj-mj)
    m=np.mean(M, axis=1)[:,np.newaxis]
    Mc = M-np.tile(m, (1,M.shape[1])) # M centered
    
    # Calculate dot products of raws (<bi-mi, bj-mj>)
    Mcov = np.dot(Mc, Mc.T)/Mc.shape[1] # M covariance: Mcov[i,j] = np.cov(M[i,:],M[j,:])
    
    # Calculate C
    MvarVert = np.tile(np.diag(Mcov), (Mcov.shape[1], 1)) # tile diag values (variances) vertically
    MvarHor = np.tile(np.diag(Mcov).reshape((Mcov.shape[1], 1)), (1, Mcov.shape[1]))# tile diag values (variances) horizontally
    MvarProd = np.sqrt(MvarVert*MvarHor) # Variances product: varProd[i,j] = np.sqrt(np.var(M[i,:])*np.var(M[j,:]))
    C = Mcov/MvarProd # corrcoeff pears. is covariance/product of variances

    return C if M.shape[0]>2 else C[0,1] # return corr matrix if more than 2 series, else only the corrcoeff

def pearson_corr_trn(L, b, dp):
    '''
    Calculate the NxN matrix of pairwise Pearson’s correlation coefficients 
    between all combinations of Ncells spike trains.
    # Parameters
    - L: list of Ncells spike time trains (arrays or lists), in samples
    - b: bin size to bin spike trains, in milliseconds
    - dp: data path (to get the recording length)
    # Outputs
    - C: pairwise correlations matrix, Ncells x Ncells
    '''
    def bnr(t, b, rec_len):
        return binarize(t, b, 30000, rec_len, False)
    rec_len=np.load(dp+'/spike_times.npy')[-1]
    tb1= bnr(L[0], b, rec_len)
    M=npa(zeros=(len(L), len(tb1)))
    M[0,:]=tb1
    del tb1
    for i, t in enumerate(L[1:]):
        M[i+1,:] = bnr(t)
    return pearson_corr(M)

def correlation_index(L, dt, dp):
    '''
    Calculate the NxN matrix of pairwise correlation indices from Wong, Meister and Shatz 1993
    reviewed by Cutts and Eglen 2014.
    WARNING firing rate biased!
    # Parameters
    - L: list of Ncells spike time trains (arrays or lists), in samples
    - dt: synchronicity window, in ms
    - dp: datapath, to find recording length
    # Outputs
    - C: pairwise correlations indices, Ncells x Ncells
    '''
    # Sanity checks
    assert type(L)==list
    assert len(L)>1
    rec_len = np.load(dp+'/spike_times.npy')[-1]

    # Formula where <bi-mi, bj-mj> is the dot product of mean-substracted tiem series
    # (essentially covariance of i and j)
    # C[i,j] = (Nab[-dt,dt] * T) / (Na*Nb*2*dt)
    # where Nab[-dt,dt] is the number of spikes A falling within -dt,dt windows around psikes B
    C = npa(zeros=(len(L), len(L)))
    for i1, t1 in enumerate(L):
        Na=len(t1)
        for i2, t2 in enumerate(L):
            Nb=len(t2)
            if i1==i2:
                C[i1,i2]=0
            elif i2<i1:
                pass
            else:
                # all to all differences in one shot
                Nab=0
                if len(t1)<len(t2): # screen the spikes of the shortest spike train to earn time
                    t=t1; tt=t2;
                else:
                    t=t2; tt=t1;
                for spk_a in t:
                    d_spka_allb = np.abs(tt-spk_a)
                    Nab+=np.count_nonzero(d_spka_allb<=dt*30)
                C[i1,i2]=C[i2,i1]=(Nab*rec_len)/(Na*Nb*2*dt)
    
    return C if len(L)>2 else C[0,1] # return corr matrix if more than 2 series, else only the corrcoeff

def spike_time_tiling_coefficient(L, dt, dp):
    """
    Calculates the Spike Time Tiling Coefficient (STTC) as described in
    (Cutts & Eglen, 2014) following Cutts' implementation in C.
    The STTC is a pairwise measure of correlation between spike trains.
    It has been proposed as a replacement for the correlation index as it
    presents several advantages (e.g. it's not confounded by firing rate,
    appropriately distinguishes lack of correlation from anti-correlation,
    periods of silence don't add to the correlation and it's sensitive to
    firing patterns).

    The STTC is calculated as follows:

    .. math::
        STTC = 1/2((PA - TB)/(1 - PA*TB) + (PB - TA)/(1 - PB*TA))

    Where `PA` is the proportion of spikes from train 1 that lie within
    `[-dt, +dt]` of any spike of train 2 divided by the total number of spikes
    in train 1, `PB` is the same proportion for the spikes in train 2;
    `TA` is the proportion of total recording time within `[-dt, +dt]` of any
    spike in train 1, TB is the same proportion for train 2.
    For :math:`TA = PB = 1`and for :math:`TB = PA = 1`
    the resulting :math:`0/0` is replaced with :math:`1`,
    since every spike from the train with :math:`T = 1` is within
    `[-dt, +dt]` of a spike of the other train.

    This is a Python implementation compatible with the elephant library of
    the original code by C. Cutts written in C and avaiable at:
    (https://github.com/CCutts/Detecting_pairwise_correlations_in_spike_trains/blob/master/spike_time_tiling_coefficient.c)

    Parameters
    ----------
    spiketrain_1, spiketrain_2: neo.Spiketrain objects to cross-correlate.
        Must have the same t_start and t_stop.
    dt: Python Quantity.
        The synchronicity window is used for both: the quantification of the
        proportion of total recording time that lies [-dt, +dt] of each spike
        in each train and the proportion of spikes in `spiketrain_1` that lies
        `[-dt, +dt]` of any spike in `spiketrain_2`.
        Default : 0.005 * pq.s

    Returns
    -------
    index:  float
        The spike time tiling coefficient (STTC). Returns np.nan if any spike
        train is empty.

    References
    ----------
    Cutts, C. S., & Eglen, S. J. (2014). Detecting Pairwise Correlations in
    Spike Trains: An Objective Comparison of Methods and Application to the
    Study of Retinal Waves. Journal of Neuroscience, 34(43), 14288–14303.
    """

    def run_P(spiketrain_1, spiketrain_2):
        """
        Check every spike in train 1 to see if there's a spike in train 2
        within dt
        """
        N2 = len(spiketrain_2)

        # Search spikes of spiketrain_1 in spiketrain_2
        # ind will contain index of
        ind = np.searchsorted(spiketrain_2.times, spiketrain_1.times)

        # To prevent IndexErrors
        # If a spike of spiketrain_1 is after the last spike of spiketrain_2,
        # the index is N2, however spiketrain_2[N2] raises an IndexError.
        # By shifting this index, the spike of spiketrain_1 will be compared
        # to the last 2 spikes of spiketrain_2 (negligible overhead).
        # Note: Not necessary for index 0 that will be shifted to -1,
        # because spiketrain_2[-1] is valid (additional negligible comparison)
        ind[ind == N2] = N2 - 1

        # Compare to nearest spike in spiketrain_2 BEFORE spike in spiketrain_1
        close_left = np.abs(
            spiketrain_2.times[ind - 1] - spiketrain_1.times) <= dt
        # Compare to nearest spike in spiketrain_2 AFTER (or simultaneous)
        # spike in spiketrain_2
        close_right = np.abs(
            spiketrain_2.times[ind] - spiketrain_1.times) <= dt

        # spiketrain_2 spikes that are in [-dt, dt] range of spiketrain_1
        # spikes are counted only ONCE (as per original implementation)
        close = close_left + close_right

        # Count how many spikes in spiketrain_1 have a "partner" in
        # spiketrain_2
        return np.count_nonzero(close)

    def run_T(spiketrain):
        """
        Calculate the proportion of the total recording time 'tiled' by spikes.
        """
        N = len(spiketrain)
        time_A = 2 * N * dt  # maximum possible time

        if N == 1:  # for just one spike in train
            if spiketrain[0] - spiketrain.t_start < dt:
                time_A += -dt + spiketrain[0] - spiketrain.t_start
            if spiketrain[0] + dt > spiketrain.t_stop:
                time_A += -dt - spiketrain[0] + spiketrain.t_stop
        else:  # if more than one spike in train
            # Vectorized loop of spike time differences
            diff = np.diff(spiketrain)
            diff_overlap = diff[diff < 2 * dt]
            # Subtract overlap
            time_A += -2 * dt * len(diff_overlap) + np.sum(diff_overlap)

            # check if spikes are within dt of the start and/or end
            # if so subtract overlap of first and/or last spike
            if (spiketrain[0] - spiketrain.t_start) < dt:
                time_A += spiketrain[0] - dt - spiketrain.t_start

            if (spiketrain.t_stop - spiketrain[N - 1]) < dt:
                time_A += -spiketrain[-1] - dt + spiketrain.t_stop

        T = time_A / (spiketrain.t_stop - spiketrain.t_start)
        return T.simplified.item()  # enforce simplification, strip units

    N1 = len(spiketrain_1)
    N2 = len(spiketrain_2)

    if N1 == 0 or N2 == 0:
        index = np.nan
    else:
        TA = run_T(spiketrain_1)
        TB = run_T(spiketrain_2)
        PA = run_P(spiketrain_1, spiketrain_2)
        PA = PA / N1
        PB = run_P(spiketrain_2, spiketrain_1)
        PB = PB / N2
        # check if the P and T values are 1 to avoid division by zero
        # This only happens for TA = PB = 1 and/or TB = PA = 1,
        # which leads to 0/0 in the calculation of the index.
        # In those cases, every spike in the train with P = 1
        # is within dt of a spike in the other train,
        # so we set the respective (partial) index to 1.
        if PA * TB == 1:
            if PB * TA == 1:
                index = 1.
            else:
                index = 0.5 + 0.5 * (PB - TA) / (1 - PB * TA)
        elif PB * TA == 1:
            index = 0.5 + 0.5 * (PA - TB) / (1 - PA * TB)
        else:
            index = 0.5 * (PA - TB) / (1 - PA * TB) + 0.5 * (PB - TA) / (
                    1 - PB * TA)
    return index


def make_phy_like_spikeClustersTimes(dp, U, rec_section='all', prnt=True, trains={}):
    '''If provided, dic must be of the form {unit1:train1InSamples, unit2:...}'''
    if trains=={}:
        for iu, u in enumerate(U):
            # Even lists of strings can be dealt with as integers by being replaced by their indices
            trains[iu]=trn(dp, u, ret=True, sav=False, rec_section=rec_section, prnt=prnt) # trains in samples
    else:
        assert len(trains.items())>1
    spikes = np.empty((2, 0))
    for key, val in trains.items():
        spikes = np.concatenate((spikes, np.vstack((val, np.full(val.shape, key)))), axis=1)
    sortedIdx = np.argsort(spikes[0,:])
    rows = np.array([[0], [1]])
    spikes = spikes[rows, sortedIdx]
    spikes = spikes.astype('int64')
    return spikes[0,:], spikes[1,:] # equivalent of spike_times.npy and spike_clusters.npy


def crosscorrelate_cyrille(dp, bin_size, win_size, U, fs=30000, symmetrize=True, rec_section='all', prnt=False, own_trains={}):
    '''Returns the crosscorrelation function of two spike trains.
       - dp: (string): DataPath to the Neuropixels dataset.
       - win_size (float): window size, in milliseconds
       - bin_size (float): bin size, in milliseconds
       - U (list of integers): list of units indices.
       - fs: sampling rate (Hertz). Default 30000.
       - symmetrize (bool): symmetrize the semi correlograms. Default=True.
       - own_trains: dictionnary of trains, to calculate the CCG of an arbitrary list of trains in SAMPLES for fs=30kHz.'''
       
    #### Troubleshooting
    assert fs > 0.
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
    binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
    assert binsize >= 1 # Cannot be smaller than a sample time

    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    winsize_bins = 2 * int(.5 * win_size *1./ bin_size) + 1 # Both in millisecond
    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1

    #### Get clusters and times
    if type(U)!=list:
        U=list(U)
    
    phy_ss, spike_clusters = make_phy_like_spikeClustersTimes(dp, U, rec_section=rec_section, prnt=prnt, trains=own_trains.copy())
    units = _unique(spike_clusters)#_as_array(U) # Order of the correlogram: order of the inputted list U (replaced by its indices - see make_phy_like_spikeClustersTimes)
    n_units = len(units)

    #### Compute crosscorrelograms
    # Shift between the two copies of the spike trains.
    shift = 1 # in indices of the spike times array... RESOLUTION OF 1 SAMPLE!

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    
    mask = np.ones_like(phy_ss, dtype=np.bool)
    
    correlograms = np.zeros((n_units, n_units, winsize_bins // 2 + 1), dtype=np.int32) # Only computes semi correlograms (//2)
    #print(" - CCG bins: ", winsize_bins)

    # The loop continues as long as there is at least one spike with
    # a matching (neighbouring) spike.
    # Mask is updated at each iteration,
    # shift is incremented at each iteration.
    while mask[:-shift].any():

        # Number of time samples between spike i and spike i+shift.
        phy_ss = _as_array(phy_ss)
        spike_diff = phy_ss[shift:] - phy_ss[:-shift] #phy_ss[:len(phy_ss) - shift]

        # Binarize the delays between spike i and spike i+shift.
        # Spike diff is populated with time differences is samples for an overlap of shift. 
        # max: conversion of spike to spike differences from samples to correlogram bins.
        # "How many bins away are neighbouring spikes"
        spike_diff_b = spike_diff // binsize # binsize is in samples.
        # DELTA_Ts ARE ALWAYS POSITIVE

        # Spikes with no matching spikes are masked.
        # spike_diff_b has the size of phy_ss[:-shift] hence mask[:-shift]
        # max: i.e. spikes which do not have neighbouring spikes are masked 
        # (further than half the correlogram window winsize_bins // 2).
        # -->> THIS IS WHERE OUTLYER DELTA_Ts IN BINS ARE EXCLUDED,
        # THE ONLY REMAINING ONES ARE STRICTLY WITHIN THE CORRELOGRAM WINDOW
        mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False

        # Cache the masked spike delays.
        m = mask[:-shift].copy()
        delta_t_bins_filtered = spike_diff_b[m] # remove the spike diffs calculated from spikes with no neighbouring spike

        # Find the indices in the raveled correlograms array that need
        # to be incremented, taking into account the spike units.
        spike_clusters_i = _index_of(spike_clusters, units)
        end_units_i=spike_clusters_i[:-shift][m]
        start_units_i=spike_clusters_i[+shift:][m]
        # numpy ravel_nulti_index -> PROPERLY INDEX THE DELTA_Ts IN BINS TO THEIR PAIRS OF UNITS
        # first argument: one array for each dimension
        # second argument: size of each dimension (NunitsxNunitsxWindow/binsize)
        indices = np.ravel_multi_index((end_units_i, start_units_i, delta_t_bins_filtered), correlograms.shape)

        # Increment the matching spikes in the correlograms array.
        # arr, indices shapes are NunitsxNunitsxNbins
        # bbins shape is NunitsxNunitsx:len(bbins) where bbins[2] goes from 0 to <=Nbins
        arr = correlograms.ravel() # Alias -> modif of arr will apply to correlograms
        arr = _as_array(arr)
        indices = _as_array(indices)
        bbins = np.bincount(indices) # would turn [0,2,3,5]ms into [1,0,1,1,0,1]
        arr[:len(bbins)] += bbins # increments the NunitsxNunits histograms at the same time

        shift += 1
        
    # Remove ACG peaks (perfectly correlated with themselves)
    correlograms[np.arange(n_units),
                      np.arange(n_units),
                      0] = 0

    if symmetrize==True:
        n_units, _, n_bins = correlograms.shape
        assert n_units == _
        # We symmetrize c[i, j, 0].
        # This is necessary because the algorithm in correlograms()
        # is sensitive to the order of identical spikes.
        correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                          correlograms[..., 0].T)
        sym = correlograms[..., 1:][..., ::-1]
        sym = np.transpose(sym, (1, 0, 2))
        correlograms = np.dstack((sym, correlograms))

#    if normalize:
#        correlograms = np.apply_along_axis(lambda x: x*1./np.sum(x) if np.sum(x)!=0 else x, 2, correlograms)
                
    return correlograms

             
def ccg(dp, U, bin_size, win_size, fs=30000, normalize='Hertz', ret=True, sav=True, prnt=True, rec_section='all', again=False):
    '''
    ********
    routine from routines_spikes
    computes crosscorrelogram (1, window/bin_size) - int64, in Hertz
    ********
    
     - dp (string): DataPath to the Neuropixels dataset.
     - u (list of ints or str): list of units indices. If str, format has to be 'datasetIndex_unitIndex'.
     - win_size: size of binarized spike train bins, in milliseconds.
     - bin_size: size of crosscorrelograms bins, in milliseconds.
     - rec_len: length of the recording, in seconds. If not provided, time of the last spike.
     - fs: sampling frequency, in Hertz. 30000 for standard Neuropixels recordings.
     - Normalize: either 'Counts' (no normalization), 'Hertz' (trigger-units-spikes-aligned inst.FR of target unit) 
      or 'Pearson' (in units of pearson correlation coefficient).
      - ret (bool - default False): if True, train returned by the routine.
      If False, by definition of the routine, drawn to global namespace.
      - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
      
      returns numpy array (Nunits, Nunits, win_size/bin_size)

    '''
    if type(normalize) != str or (normalize not in ['Counts', 'Hertz', 'Pearson', 'zscore']):
        print("WARNING ccg() 'normalize' argument should be a string in ['Counts', 'Hertz', 'Pearson', 'zscore']. Exitting now.")
        return None
    # Preformat
    dp=str(dp)
    U = [U] if type(U)!=list else U
    sortedU=U.copy()
    sortedU.sort()
    
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)
    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    
    fn='/ccg{}_{}_{}_{}({}).npy'.format(str(sortedU).replace(" ", ""), str(bin_size), str(int(win_size)), normalize, str(rec_section)[0:50].replace(' ', ''))
    if os.path.exists(dprm+fn) and not again:
        if prnt: print("File {} found in routines memory. Will be computed from source files.".format(fn))
        crosscorrelograms = np.load(dprm+fn)
        crosscorrelograms = np.asarray(crosscorrelograms, dtype='float64')
    # if not, compute it
    else:
        if prnt: print("File {} not found in routines memory.".format(fn))
        crosscorrelograms = crosscorrelate_cyrille(dp, bin_size, win_size, sortedU, fs, True, rec_section=rec_section, prnt=prnt)#####################
        crosscorrelograms = np.asarray(crosscorrelograms, dtype='float64')
        if normalize in ['Hertz', 'Pearson', 'zscore']:
            for i1,u1 in enumerate(sortedU):
                Nspikes1=len(trn(dp, u1, ret=True, prnt=False))
                #imfr1=np.mean(1000./isi(dp, u1)[isi(dp, u1)>0])
                for i2,u2 in enumerate(sortedU):
                    Nspikes2=len(trn(dp, u2, ret=True, prnt=False))
                    #imfr2=np.mean(1000./isi(dp, u2)[isi(dp, u2)>0])
                    arr=crosscorrelograms[i1,i2,:]
                    if normalize == 'Hertz':
                        crosscorrelograms[i1,i2,:]=arr*1./(Nspikes1*bin_size*1./1000)
                    elif normalize == 'Pearson':
                        crosscorrelograms[i1,i2,:]=arr*1./np.sqrt(Nspikes1*Nspikes2)
                    elif normalize=='zscore':
                        arr=crosscorrelograms[i1,i2,:]
                        mn = np.mean(np.append(arr[:int(len(arr)*2./5)], arr[int(len(arr)*3./5):]))
                        std = np.std(np.append(arr[:int(len(arr)*2./5)], arr[int(len(arr)*3./5):]))
                        crosscorrelograms[i1,i2,:]=(arr-mn)*1./std
                    

        # Save it
        if sav:
            fn='/ccg{}_{}_{}_{}({}).npy'.format(str(sortedU).replace(" ", ""), str(bin_size), str(int(win_size)), normalize, str(rec_section)[0:50].replace(' ', ''))
            np.save(dprm+fn, crosscorrelograms)
    
    # Structure the 3d array to return accordingly to the order of the inputed units U
    if crosscorrelograms.shape[0]>1:
        sortedC = np.zeros(crosscorrelograms.shape)
        sortedU=np.array(sortedU)
        for i1, u1 in enumerate(U):
            for i2, u2 in enumerate(U):
                ii1, ii2 = np.nonzero(sortedU==u1)[0], np.nonzero(sortedU==u2)[0]
                sortedC[i1,i2,:]=crosscorrelograms[ii1, ii2, :]
    else:
        sortedC=crosscorrelograms

    return sortedC

def acg(dp, u, bin_size, win_size, fs=30000, normalize='Hertz', ret=True, sav=True, prnt=True, rec_section='all', again=False):
    '''
    dp, 
    u, 
    bin_size, 
    win_size, 
    fs=30000, 
    symmetrize=True, 
    normalize=False, 
    normalize1=True, 
    ret=True, 
    sav=True, 
    prnt=True'''
    u = u[0] if type(u)==list else u
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)
    '''
    ********
    routine from routines_spikes
    computes autocorrelogram (1, window/bin_size) - int64, in Hertz
    ********
    
     - dp (string): DataPath to the Neuropixels dataset.
     - u (int): unit index
     - win_size: size of binarized spike train bins, in milliseconds.
     - bin_size: size of autocorrelograms bins, in milliseconds.
     - rec_len: length of the recording, in seconds. If not provided, time of the last spike.
     - fs: sampling frequency, in Hertz.
     - ret (bool - default False): if True, train returned by the routine.
      If False, by definition of the routine, drawn to global namespace.
      - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
      
      returns numpy array (win_size/bin_size)
      '''
    # NEVER save as acg..., uses the function ccg() which pulls out the acg from files stored as ccg[...].
    autocorrelogram = ccg(dp, u, bin_size, win_size, fs, normalize, ret, sav, prnt, rec_section, again)
    autocorrelogram = autocorrelogram[0,0,:]

    return autocorrelogram

''''''
#%% Power spectrum of autocorrelograms
    
def PSDxy(dp, U, bin_size, window='hann', nperseg=4096, scaling='spectrum', fs=30000, ret=True, sav=True, prnt=True):
    '''
    ********
    routine from routines_spikes
    computes Power Density Spectrum - float64, in V**2/Hertz
    ********
    
    - dp (string): DataPath to the Neuropixels dataset.
    - u (list of ints): list of units indices
    - bin_size: size of bins of binarized trains, in milliseconds.
    (see scipy.signal.csd for below)
    - window: Desired window to use. 
    - nprerseg: Length of each segment.
    - scaling: 'density' (Cross spectral density: V**2/Hz) or 'spectrum' (Cross spectrum: V**2)
    - ret (bool - default False): if True, train returned by the routine.
      if False, by definition of the routine, drawn to global namespace.
      - sav (bool - default True): if True, by definition of the routine, saves the file in dp/routinesMemory.
    
      returns numpy array (Nunits, Nunits, nperseg/2+1)'''
    # Preformat
    dp=str(dp)
    U = [U] if type(U)!=list else U
    sortedU=list(np.sort(np.array(U)))
    
    # Search if the variable is already saved in dp/routinesMemory
    dprm = dp+'/routinesMemory'
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(dprm+'/PSDxy{}_{}.npy'.format(sortedU, str(bin_size).replace('.','_'))):
        if prnt: print("File PSDxy{}_{}.npy found in routines memory.".format(str(sortedU).replace(" ", ""), str(bin_size).replace('.','_')))
        Pxy = np.load(dprm+'/PSDxy{}_{}.npy'.format(sortedU, str(bin_size).replace('.','_')))
        Pxy = Pxy.astype(np.float64)
    
    # if not, compute it
    else:
        if prnt: print("File ccg_{}_{}.npy not found in routines memory. Will be computed from source files.".format(str(sortedU).replace(" ", ""), str(bin_size).replace('.','_')))
        Pxy = np.empty((len(sortedU), len(sortedU), int(nperseg/2)+1), dtype=np.float64)
        for i, u1 in enumerate(sortedU):
            trnb1 = trnb(dp, u1, bin_size, ret=True)
            for j, u2 in enumerate(sortedU):
                trnb2 = trnb(dp, u2, bin_size, ret=True)
                (f, Pxy[i, j, :]) = sgnl.csd(trnb1, trnb2, fs=fs, window=window, nperseg=nperseg, scaling=scaling)
        Pxy = Pxy.astype(np.float64)
        # Save it
        if sav:
            np.save(dprm+'/PSDxy{}_{}.npy'.format(str(sortedU).replace(" ", ""), str(bin_size).replace('.','_')), Pxy)
    
    # Back to the original order
    sPxy = np.zeros(Pxy.shape)
    sortedU=np.array(sortedU)
    for i1, u1 in enumerate(U):
        for i2, u2 in enumerate(U):
            ii1, ii2 = np.nonzero(sortedU==u1)[0], np.nonzero(sortedU==u2)[0]
            sPxy[i1,i2,:]=Pxy[ii1, ii2, :]
    
    # Either return or draw to global namespace
    if ret:
        PXY=sPxy.copy()
        del sPxy
        f = np.linspace(0, 15000, int(nperseg/2)+1)
        return f, PXY
    else:
        # fn_ = ''
        # for i in range(len(U)):
        #     fn_+='_'+str(U[i])
        # if prnt: print("PSDxy{}_{} and f defined into global namespace.".format(fn_, str(bin_size).replace('.','_')))
        # exec("PSDxy{}_{} = sPxy".format(fn_, str(bin_size).replace('.','_')), globals())
        del sPxy

''''''
#%% Pairwise correlations matrix and population coupling

from elephant.spike_train_generation import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import covariance, corrcoef
from quantities import ms


def make_cm(dp, units, b=5, cbin=1, cwin=100, corrEvaluator='corrcoeff_eleph', vmax=5, vmin=0, rec_section='all'):
    '''Make correlation matrix.
    dp: datapath
    units: units list of the same dataset
    b: bin of spike train if covar, corrcoeff or corrcoeff_MB is used as an evaluator, in milliseconds
    cbin, cwin: CCG bin and win, if CCG is used as correlation evaluator
    corrEvaluator: metric used to evaluate correlation, in ['CCG', 'covar', 'corrcoeff_eleph', 'corrcoeff_MB']
    vmax: upper bound of colormap
    vmin: lower bound of colormap
    rec_section: section of the Neuropixels recording used for evaluation of correlation.'''
    
    # Sanity checks
    allowedCorEvals = ['CCG', 'covar', 'corrcoeff_eleph', 'corrcoeff_MB']
    try:
        assert corrEvaluator in allowedCorEvals
    except:
        print('WARNING: {} should be in {}. Exiting now.'.format(corrEvaluator, allowedCorEvals))
        return
    
    # Initialize empty arrays
    rec_len = np.load(dp+'/spike_times.npy')[-1]*1./30 # in ms
    Nbins_bms = len(trnb(dp, units[0], b, constrainBin=False)) # b in ms
    if corrEvaluator =='corrcoeff_MB':
        trnbM = npa(zeros=(len(units), Nbins_bms))
    elif corrEvaluator in ['covar', 'corrcoeff_eleph']:
        trnLs = []
    elif corrEvaluator == 'CCG':
        cmCCG=npa(empty=(len(units), len(units)))

    # Populate empty arrays
    for i1, u1 in enumerate(units):
        if corrEvaluator =='corrcoeff_MB':
            trnbM[i1,:]=trnb(dp, u1, b, constrainBin=False, rec_section=rec_section) # b in ms
        elif corrEvaluator in ['covar', 'corrcoeff_eleph']:
            t1 = SpikeTrain(trn(dp, u1, rec_section=rec_section)*1./30*ms, t_stop=rec_len)
            trnLs.append(t1)
        elif corrEvaluator == 'CCG':
            for i2, u2 in enumerate(units):
                if u1!=u2:
                    CCG = ccg(dp, [u1, u2], cbin, cwin, normalize='Pearson', rec_section=rec_section)[0,1,:]
                    coeffCCG = CCG[len(CCG)//2+1]
                else:
                    coeffCCG=0
                cmCCG[i1, i2]=coeffCCG
    
    # Set correlation matrix and plotting parameters
    if corrEvaluator == 'CCG':
        cm = cmCCG
        vmax = 5 if vmax == 0 else vmax
    elif corrEvaluator == 'covar':
        cm = covariance(BinnedSpikeTrain(trnLs, binsize=b*ms))
        vmax = 0.05 if vmax == 0 else vmax
    elif corrEvaluator == 'corrcoeff_eleph':
        cm = corrcoef(BinnedSpikeTrain(trnLs, binsize=b*ms))
        vmax = 0.1 if vmax == 0 else vmax
    elif corrEvaluator == 'corrcoeff_MB':
        cm = pearson_corr(trnbM)
        vmax = 0.1 if vmax == 0 else vmax
    
    return cm

        
''''''
#%% Connectivity inferred from correlograms
fractile_normal(p=0.95, m=0, s=1)


def canUse_Nbins(a=0.05, w=100, b=0.2, n_bins=3):
    '''Function to assess the number of expected triplets (3 consecutive bins) in a crosscorrelogram.
    The confidence of the test used cannot exceed the returned value.
    E.g. for 100 bins or 0.1ms and a 
    - a: alpha, confidence level
    - w: correlogram window size, ms
    - b: correlogram bin size, ms
    
    See Kopelowitz, Lev et Cohen, 2014, JNeuro methods,
    Quantification of pairwise neuronal interactions: going beyond the significance lines.'''
    assert 0<a<1
    assert n_bins in [2,3], "Can only handle 2 or 3 bins"
    n=w/b
    if n_bins==3: expected_triplets=a*(a*n-2)*(a*n-4)/(8*(n-1))
    elif n_bins==2: expected_triplets=a*(a*n-2)*(a*n-2)/(4*(n-1))
    if a>expected_triplets:
        print("You can use this test, of confidence {0} since the probability of encountering a triplet by chance is {1:.3f}.".format(a, expected_triplets))
        return True
    else:
        print("You CANNOT use this test, of confidence {0} since the probability of encountering a triplet by chance is {1:.3f}.".format(a, expected_triplets))
        return False

def ccg_significance(CCG, cbin=0.2, cwin=100, law='Normal', alpha=0.05, n_consec_bins=3, multi_comp=False, bin_wise=False):
    '''Function to assess whether a correlogram is significant or not.
    Parameters:
    - a: alpha, confidence level
    - w: correlogram window size, ms
    - b: correlogram bin size, ms
    Returns:
    - True or False, whether the ccg is significantly modulated or not
    See Kopelowitz, Lev et Cohen, 2014, JNeuro methods,
    Quantification of pairwise neuronal interactions: going beyond the significance lines.
    Test1: law='Poisson', alpha=0.05, n_consec_bins=1, bin_wise=True
    Test2: law='Poisson', alpha=0.05, n_consec_bins=1
    Test3: law='Normal',  alpha=0.01, n_consec_bins=1, multi_comp=True
    Test4: law='Normal',  alpha=0.05, n_consec_bins=3                     <<<-- RECOMMENDED BY PAPER
    Test5: law='Normal',  alpha=0.01, n_consec_bins=2
    '''
    
    assert law in ['Poisson', 'Normal']
    assert 0<alpha<1
    out_fraction=0.4;out_fraction/=2
    Ho_ccg = np.concatenate([CCG[:int(out_fraction*cwin/cbin)], CCG[int((1-out_fraction)*cwin/cbin):]]) # It is assumed that before -40/after +40ms, there is no more correlation
    if n_consec_bins in [2,3]:
        assert law=='Normal', "Using more than 1 bin is not handled when assuming a Poisson distribution."
        assert canUse_Nbins(alpha, cwin, cbin, n_consec_bins)
    if not bin_wise: # else: compute one CI for each bin
        if law=='Poisson':
            peaks=[] # not handled yet - how to end up with natural integers as ccg values? Spike counts? Can multiplie by random big number?
        elif law=='Normal':
            m=np.mean(Ho_ccg)
            s=np.std(Ho_ccg)
            fr_a=fractile_normal(1-(alpha/2), 0, 1)
            peaks=extract_hist_modulation_features(CCG, cbin, fr_a, n_consec_bins, m, s)
    else:
        peaks=[] 

    return 1 if np.any(peaks) else 0

def find_transmission_prob(ccg, cbin, holgauss_b, holgauss_w):
    
    return tp, tp_time

def find_threshold_crosses(hist, th):
    '''
    - hist is a 1d array, 
    - th is the threshold in hist unit,
    
    Returns:
        - crosses: a list of 2D arrays[array([[t1, t2, ...tn], [c1, c2, ...cn]]), ...],
        which contains the threshold crosses times and z-scored correlation values.
        
        By definition, all c values are above th or below -th.'''

    ## Peaks
    peaks=[]
    cross_thp, cross_thn = thresh(hist, th, 1), thresh(hist, th, -1)
    if (len(cross_thp)==0 or len(cross_thn)==0) and (len(cross_thp)+len(cross_thn))>0: cross_thp, cross_thn = [], [] # Only one cross at the beginning or the end e.g.
    elif (len(cross_thp)>0 and len(cross_thn)>0):
        flag0, flag1=False,False
        if cross_thp[-1]>cross_thn[-1]: flag1=True # if threshold+ is positively crossed at the end
        if cross_thn[0]<cross_thp[0]: flag0=True # if threshold+ is negatively crossed at the beginning
        if flag1: cross_thp=cross_thp[:-1] # remove aberrant last value
        if flag0: cross_thn=cross_thn[1:] # remove aberrant first value

    if len(cross_thp)==len(cross_thn): # Should be true because starts from and returns to baseline
        for i in range(len(cross_thp)):
            l_ind, r_ind = cross_thp[i], cross_thn[i]
            cross_values=hist[l_ind:r_ind]
            cross_times=np.arange(l_ind, r_ind, 1)
            peaks.append(np.vstack([cross_times, cross_values]))

    else:
        print('ASSERTION ERROR len(cross_thp) ({}) == len(cross_thn) ({}) for hist peaks:'.format(len(cross_thp), len(cross_thn)))
        print(hist)
    
    ## Troughs
    troughs=[]
    cross_th_p, cross_th_n = thresh(hist, -th, 1), thresh(hist, -th, -1)
    if (len(cross_th_p)+len(cross_th_n))==1: cross_th_p, cross_th_n = [], [] # Only one cross at the beginning or the end
    elif len(cross_th_p)>0 and len(cross_th_n)>0:
        flag0, flag1=False,False
        if cross_th_n[-1]>cross_th_p[-1]: flag1=True # if threshold+ is positively crossed at the end
        if cross_th_p[0]<cross_th_n[0]: flag0=True # if threshold+ is negatively crossed at the beginning
        if flag1: cross_th_n=cross_th_n[:-1] # remove aberrant last value
        if flag0: cross_th_p=cross_th_p[1:] # remove aberrant first value
    
    if len(cross_th_p)==len(cross_th_n): # Should be true because starts from and returns to baseline
        for i in range(len(cross_th_p)):
            l_ind, r_ind = cross_th_n[i], cross_th_p[i]
            cross_values=hist[l_ind:r_ind]
            cross_times=np.arange(l_ind, r_ind, 1)
            troughs.append(np.vstack([cross_times, cross_values]))
            
    else:
        print('ASSERTION ERROR len(cross_thp) ({}) ==len(cross_thn) ({}) for hist troughs:'.format(len(cross_th_p), len(cross_th_n)))
        print(hist)
    
    return peaks+troughs # concatenates

def extract_hist_modulation_features(hist, hbin, threshold=2, n_consec_bins=3, ext_mn=None, ext_std=None, pkSgn='all', win_fract_baseline=0.5):
    '''
        - hist is a 1d array, 
        - hbin is in ms, 
        - th is the threshold in hist unit,
        - n_consec_bins is the number of consecutive bins which  have to cross the threshold for a peak to be called a peak,
        - ext_mn is an external mean which can be provided rather than being calculated on the 2 lateral fifth (to measure it on a wider window for instance).
        - ext_std is an external std which can be provided rather than being calculated on the 2 lateral fifth  (measured it on a wider window for instance).
        - pkSgn can be '+', '-' or 'all' -> dictates whether to return peaks, troughs or both.
        - win_fract_baseline: fraction  of the window used to compute the Ho mean and std, is no ext_mn or ext_std are provided. If 0.6, 0:0.3 and 0.7:1 will be used.
        Returns:
        - peaks+troughs: [(l_ms, r_ms, y, t_ms, n_triplets, n_bincrossing, bin_heights, entropy), ...]
        where l_ms, r_ms, y, t_ms stand for left edge, right edge, y value (std), time
        and n_triplets, n_bincrossing, bin_heights, entropy for the 4 correlation strength metrics of Kopelowitz et al.'''
    
    assert pkSgn in ['+', '-', 'all'], "pkSgn has an odd value, should be  '+', '-' or 'all'."
    assert win_fract_baseline<1
    wfl, wfr = win_fract_baseline/2, 1-(win_fract_baseline/2)
    mn = np.mean(np.append(hist[:int(len(hist)*wfl)], hist[int(len(hist)*wfr):])) if ext_mn==None else ext_mn
    std = np.std(np.append(hist[:int(len(hist)*wfl)], hist[int(len(hist)*wfr):])) if ext_std==None else ext_std
    hist=(hist-mn)*1./std # Z-score
    crosses=find_threshold_crosses(hist, threshold)
    
    peaks, troughs = [], []
    
    for cross in crosses:
        # for positive crosses, l_ind is the index of the first bin above threshold;
        # r_ind is the index of the first bin below threshold.
        l_ind, r_ind = cross[0,0], cross[0,-1]+1
        l_ms, r_ms = (l_ind-(len(hist)-1)*1./2)*hbin, (r_ind-(len(hist)-1)*1./2)*hbin
        y=max(np.abs(cross[1,:]))*sign(cross[1,0])
        t_ind=cross[0,:][cross[1,:]==y][0] # If there are 2 equal maxima, the 1st one is picked
        t_ms=(t_ind-(len(hist)-1)*1./2)*hbin
        n_triplets=cross.shape[1]//3
        n_bincrossing=cross.shape[1]
        bin_heights=np.mean(cross[1,:])
        # Assuming that the sides of the correlogram have a normal distribution,
        # the Z-scored crosscorrelogram with the mean and std of the correlogram sides
        # should have a normal distribution if not modulated.
        # Hence the Ho PDF of the Z-scored correlogram is the N(0,1) distribution.
        # So the probability of a bin height is pdf_normal(np.abs(bin_height), m=0, s=1).
        pi=pdf_normal(np.abs(cross[1,:]), m=0, s=1) 
        entropy=-np.mean(np.log(pi))
        if cross.shape[1]>=n_consec_bins:
            if cross[1,0]>0: # peak
                peaks.append((l_ms, r_ms, y, t_ms, n_triplets, n_bincrossing, bin_heights, entropy))
            elif cross[1,0]<0:  # trough
                troughs.append((l_ms, r_ms, y, t_ms, n_triplets, n_bincrossing, bin_heights, entropy))
        
    if pkSgn=='+':
        ret = peaks
    elif pkSgn=='-':
        ret = troughs
    elif pkSgn=='all':
        ret = peaks+troughs # Concatenates (python list). Distinguish peaks and troughs with their peak value sign.
        
    return ret
        
def gen_sfc(dp, cbin=0.2, cwin=100, threshold=2, n_consec_bins=3, rec_section='all', _format='peaks_infos', again=False, againCCG=False):
    '''
    Function generating a NxN functional correlation dataframe SFCDF and matrix SFCM
    from a sorted Kilosort output at 'dp' containing 'N' good units
    with cdf(i,j) is a list of a varying number of (l, r, s) tuples
    where l is the left edge in ms, r is the right edge in ms and s is the size in std
    of significant peaks (either positive or negative).
    
    '''
    assert _format in ['peaks_infos', 'raw_ccgs'],'WARNING {} must be in {}! Exitting now.'.format(_format, str(['peaks_infos', 'raw_sig_ccgs']))
        
    peakChs=get_depthSort_peakChans(dp, quality='good')
    gu, bestChs = peakChs[:,0], peakChs[:,1].astype('int64')
    
    # histo=False
    # if os.path.isfile(dp+'/FeaturesTable/FeaturesTable_histo.csv'):
    #     fth = pd.read_csv(dp+'/FeaturesTable/FeaturesTable_histo.csv', sep=',', index_col=0)
    #     depthIdx = np.argsort(bestChs)[::-1] # From surface (high ch) to DCN (low ch)
    #     histo_str=np.array(fth["Histology_str"])
    #     histo_str=histo_str[depthIdx]
    #     histo=True
    
    dprm = dp+'/routinesMemory'
    if not op.isdir(dprm): os.makedirs(dprm)
    
    if _format=='peaks_infos':
        fn1='/SFCDF{}-{}_{}-{}({})_{}.csv'.format(cbin, cwin, threshold, n_consec_bins, str(rec_section)[0:50].replace(' ', ''), _format)
        fn2='/SFCM'+fn1[6:]
        fn3='/SFCMtime'+fn1[6:-4]+'.npy'
        if op.exists(dprm+fn1) and not again:
            print("File {} found in routines memory.".format(fn1))
            SFCDF = pd.read_csv(dprm+fn1, index_col='Unit')
            SFCDF.replace('0', 0, inplace=True) # Loading a csv turns all the values into strings - now only detected peaks are strings
            SFCDF.replace(0.0, 0, inplace=True)
            SFCM1 = pd.read_csv(dprm+fn2, index_col='Unit')
            SFCMtime = np.load(dprm+fn3)

            return SFCDF, SFCM1, gu, bestChs, SFCMtime
            
    elif _format=='raw_ccgs':
        fn='/SFCDF{}-{}_{}-{}({})_{}.csv'.format(cbin, cwin, threshold, n_consec_bins, str(rec_section)[0:50].replace(' ', ''), _format)
        if op .exists(dprm+fn) and not again:
            print("File {} found in routines memory.".format(fn))
            SFCDF = pd.read_csv(dprm+fn, index_col='Unit')
            return SFCDF
        
    if _format=='peaks_infos':
        gu12=np.array([[u]*12 for u in gu]).flatten()
        SFCDF = pd.DataFrame(index=gu, columns=gu) 
        SFCM = np.zeros((len(gu), len(gu)*12))
        SFCMtime = np.zeros((len(gu), len(gu)*12))
    elif _format=='raw_ccgs':
        #spec, frac, frac2, lms = 'acg', 0.03, 0.005, 2.5
        all_gu_x_all_gu = ['{}->{}'.format(u1,u2) for i1, u1 in enumerate(gu) for i2, u2 in enumerate(gu) if (i1!=i2)]
        if (cwin*1./cbin)%2==0: # even
            bins=np.arange(-cwin*1./2, cwin*1./2+cbin, cbin)
        elif (cwin*1./cbin)%2==1: # odd
            bins=np.arange(-cwin*1./2+cbin*1./2, cwin*1./2+cbin*1./2, cbin)
        SFCDF = pd.DataFrame(index=all_gu_x_all_gu, columns=[str(b) for b in bins])
        #if histo: SFCDF.insert(loc=0, column='regions', value=np.nan)
        SFCDF.insert(loc=0, column='pattern', value=np.nan)


    prct=0
    print('Significant functional correlations (sfc) extraction started!')
    for i1, u1 in enumerate(gu):
        for i2, u2 in enumerate(gu):
            if prct!=int(100*((i1*len(gu)+i2+1)*1./(len(gu)**2))):
                prct=int(100*((i1*len(gu)+i2+1)*1./(len(gu)**2)))
                print('{}%...'.format(prct))
            if i1!=i2:
                print('(Checking ccg:{}->{}...)'.format(u1,u2))
                hist=ccg(dp, [u1, u2], cbin, cwin, normalize='Hertz', prnt=False, rec_section=rec_section, again=againCCG)[0,1,:]
                #hist_wide=ccg(dp, [i,j], 0.5, 50, prnt=False)[0,1,:]
                #mn=np.mean(hist); std=np.std(hist)
                pkSgn='+' if i2>i1 else '-'
                pks = extract_hist_modulation_features(hist, cbin, threshold, n_consec_bins, ext_mn=None, ext_std=None, pkSgn=pkSgn, win_fract_baseline=0.5)
                #pks=[(pk[:4]) for pk in pks]
                if _format=='peaks_infos':
                    SFCDF.loc[u1, u2]=pks if np.array(pks).any() else int(0)
                    if np.array(pks).any():
                        vsplit=len(pks)
                        print('>> Significantly modulated ccg:{}->{}! <<'.format(u1,u2))
                        if 12%vsplit==0: # dividers of 12
                            heatpks=np.array([[p[2]]*int(12*1./vsplit) for p in pks]).flatten()
                            heatpksT=np.array([[p[3]]*int(12*1./vsplit) for p in pks]).flatten()
                        else:
                            if vsplit<=12: # not divider but still smaller
                                heatpks=np.array([p[2] for p in pks]+[0]*(12-vsplit)).flatten()
                                heatpksT=np.array([p[3] for p in pks]+[0]*(12-vsplit)).flatten()
                            else:
                                raise ValueError('WARNING more than 12 peaks found - your threshold is too permissive or CCG f*cked up (units {} and {}), aborting.'.format(u1, u2))
                    else:
                        heatpks=np.array([0]*12)
                        heatpksT=np.array([np.nan]*12)
                    
                    for col in range(12): # 12 columns per unit to make the heatmap
                         SFCM[i1,12*i2:12*i2+12]=heatpks
                         SFCMtime[i1,12*i2:12*i2+12]=heatpksT
                elif _format=='raw_ccgs':
                    if np.array(pks).any():
                        #shist=smooth(hist, frac=frac, it=0, frac2=frac2, spec=spec, cbin=cbin, cwin=cwin, lms=lms)
                        colBin1=SFCDF.columns[1] #SFCDF.columns[2] if histo else SFCDF.columns[1]
                        SFCDF.loc['{}->{}'.format(u1,u2),colBin1:]=hist
                        TY = np.array([[p[3], p[2]] for p in pks])
                        TY = TY[np.argsort(TY[:,0])] # sort by crescent peak time
                        pkSgnReal = ['+' if p[1]>0 else '-' if p[1]<0 else '0' for p in TY]
                        pkSymReal = ['\\' if p[0]<-0.4 else '/' if p[0]>0.4 else '|'  for p in TY]
                        pattern='';
                        for sgn, sym in zip(pkSgnReal, pkSymReal):
                            pattern+= sgn+sym
                        
                        SFCDF.loc['{}->{}'.format(u1,u2),'pattern']=pattern
                        # if histo: SFCDF.loc['{}->{}'.format(u1,u2),'regions']='{}->{}'.format(fth.loc[u1, 'Histology_str'], fth.loc[u2, 'Histology_str'])
                    else:
                        SFCDF.drop(['{}->{}'.format(u1,u2)], inplace=True)
    print('Job done.')
    if _format=='peaks_infos':
        SFCM1 = pd.DataFrame(data=SFCM);
        SFCM1.index=gu; SFCM1.columns=gu12; # 12 columns to split vertically the heatmap pixel of each unit in either 1, 2, 3 or 4 equally sized columns.
        SFCDF.index.name = 'Unit'
        SFCM1.index.name = 'Unit'
        SFCDF.to_csv(dprm+fn1)
        SFCM1.to_csv(dprm+fn2)
        np.save(dprm+fn3, SFCMtime)

        return SFCDF, SFCM1, gu, bestChs, SFCMtime

    elif _format=='raw_ccgs':
        SFCDF.to_csv(dprm+fn)
        return SFCDF
    
def make_acg_df(dp, cbin=0.1, cwin=80, rec_section='all'):
    # Get good units
    gu = get_units(dp, quality='good') # get good units
    
    # Initialize ACG dataframe
    if (cwin*1./cbin)%2==0: # even
        bins=np.arange(-cwin*1./2, cwin*1./2+cbin, cbin)
    elif (cwin*1./cbin)%2==1: # odd
        bins=np.arange(-cwin*1./2+cbin*1./2, cwin*1./2+cbin*1./2, cbin)
    
    ACGDF = pd.DataFrame(index=gu, columns=bins)
    
    # Populate ACG dataframe
    # smooth center less than sides
    spec, frac, frac2, lms = 'acg', 0.06, 0.005, 3
    prct=0
    for i, u in enumerate(gu):
        prct=int(i*100./len(gu))
        print('{}%...'.format(prct), end='\r')
        ACG=acg(dp, u, cbin, cwin, ret=True, sav=True, prnt=False, rec_section=rec_section)
        ACGDF.loc[u, :]=smooth(ACG, frac=frac, it=0, frac2=frac2, spec=spec, cbin=cbin, cwin=cwin, lms=lms)
    print('Job done.')
    
    return ACGDF


def crosscorrelate_maxime(dp, bin_size, win_size, U, trainBin=10, fs=30000, rec_section='all', prnt=True, own_trains={}):
    '''Returns the crosscorrelation function of two spike trains.
    - dp: (string): DataPath to the Neuropixels dataset.
    - win_size (float): window size, in milliseconds
    - bin_size (float): bin size, in milliseconds
    - U (list of integers): list of units indexes. If string, measures it for the whole dataset.
    - trainBin: binsize used to binarize trains before computing corrcoeff, in ms.
    - fs: sampling rate (Hertz). Default 30000.
    - symmetrize (bool): symmetrize the semi correlograms. Default=True.
    - own_trains: dictionnary of trains, to calculate the CCG of an arbitrary list of trains in SAMPLES for fs=30kHz.'''
    
    #### Troubleshooting
    assert fs > 0.
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
    binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
    assert binsize >= 1 # Cannot be smaller than a sample time
 
    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    winsize_bins = 2 * int(.5 * win_size *1./ bin_size) + 1 # Both in millisecond
    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1
 
    #### Get clusters and times
    if own_trains!={}:
        phy_ss, spike_clusters = make_phy_like_spikeClustersTimes(dp, U, rec_section=rec_section, prnt=prnt, dic=own_trains)
        units = _unique(spike_clusters)
        n_units = len(units)
     
    else:
        if type(U)==str:
            # All the CCGs of a Neuropixels dataset
            spike_clusters = np.load(dp+"/spike_clusters.npy")
            units = _unique(spike_clusters)
            n_units = len(units)
            phy_ss = np.load(dp+'/spike_times.npy')
     
        # Between n_units provided units
        else:
            if type(U)!=list:
                U=list(U)
             
            phy_ss, spike_clusters = make_phy_like_spikeClustersTimes(dp, U, rec_section=rec_section, prnt=prnt, dic={})
            units = _unique(spike_clusters)
            n_units = len(units)
 
    #### Compute crosscorrelograms
    rec_len=phy_ss[-1]
    correlograms = np.zeros((n_units, n_units, winsize_bins // 2 + 1), dtype=np.float32) # Only computes semi correlograms (//2)
    for i1, u1 in enumerate(units):
        t1 = phy_ss[spike_clusters==u1] # samples
        t1b=binarize(t1, trainBin, fs, rec_len=rec_len, constrainBin=False)
        for i2, u2 in enumerate(units):
            t2 = phy_ss[spike_clusters==u2] # samples
            for ilag, lag in enumerate(np.arange(0, winsize_bins // 2 + 1)):
                t2_lag = t2+lag*binsize # samples
                t2lb=binarize(t2_lag, trainBin, fs, rec_len=rec_len, constrainBin=False)
                c = pearson_corr(npa([t1b, t2lb]))
                print(u1, u2, lag, c)
                correlograms[i1, i2, ilag]=c
            # set ACG 0 to 0
            if i1==i2:
                correlograms[i1, i2, 0]=0
    del t1, t1b, t2, t2_lag, t2lb
     
    # Symmetrize
    n_units, _, n_bins = correlograms.shape
    assert n_units == _
    correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                      correlograms[..., 0].T)
    sym = correlograms[..., 1:][..., ::-1]
    sym = np.transpose(sym, (1, 0, 2))
    correlograms = np.dstack((sym, correlograms))
     
    return correlograms


def crosscorrelate_maxime1(dp, U, bin_size, win_size, fs=30000, normalize=False, prnt=True):
    '''Returns the crosscorrelation function of two spike trains.
    Second one 'triggered' by the first one.
    - dp: (string): DataPath to the Neuropixels dataset.
    - U (list of integers): list of units indexes.
    - win_size (float): window size, in milliseconds
    - bin_size (float): bin size, in milliseconds
    - fs: sampling rate (Hertz). Default 30000.
    - symmetrize (bool): symmetrize the semi correlograms. Default=True.
    - normalize: normalize the correlograms. Default=False.'''
       
    # Troubleshooting
    assert fs > 0.
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
    binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
    assert binsize >= 1 # Cannot be smaller than a sample time
 
    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    win_size_bins = 2 * int(win_size*0.5/bin_size) + 1 # Both in millisecond
    assert win_size_bins >= 1
    assert win_size_bins % 2 == 1
    
    correlograms=np.zeros((len(U), len(U), win_size_bins))
    if (win_size*1./bin_size)%2==0: # even
        binsedges=np.arange(-win_size*1./2-bin_size*1./2, win_size*1./2+bin_size*3./2, bin_size) # add 1 + two half bins to keep even centered on 0
    elif win_size*1./bin_size%2==1: # odd
        binsedges=np.arange(-win_size*1./2, win_size*1./2+bin_size, bin_size) # add one bin to make even centered on 0
    
    for i1, u1 in enumerate(U):
        t1=trn(dp, u1, ret=True, prnt=False)*1./30 # ms
        for i2, u2 in enumerate(U):
            if i2>=i1:
                t2 = trn(dp, u2, ret=True, prnt=False)*1./30 if u2!=u1 else t1 # ms
                dt=np.array([])
                if prnt: print('CCG {}x{}'.format(u1, u2))
                for si, spk in enumerate(t1):
                    #end = '\r' if si<len(t1)-1 else ''
                    #print('{}%...'.format(int(100*(si+1)*1./len(t1))), end=end)
                    d = t2-spk
                    dt = np.append(dt, d[(d>=-win_size*1./2)&(d<=win_size*1./2)])
            else:
                pass
            
            hist=np.histogram(dt, binsedges)[0]
            if i1==i2:
                hist[int(.5*(len(hist)-1))]=0
            
            correlograms[i1, i2, :]=hist*1./(0.001*bin_size*np.sqrt(len(t1)*len(t2)))
    
    # Symmetrize
    for i1, u1 in enumerate(U):
        for i2, u2 in enumerate(U):
            if i1!=i2:
                correlograms[i2, i1, :]=np.array([hist[-v+1] for v in range(len(hist))])
    
    return correlograms

def crosscorrelate_maxime2(dp, U, bin_size, win_size, trn_binsize=0.1, fs=30000, normalize=False, prnt=True):
    '''
    STILL NOT FUNCTIONAL
    HORIZONTAL SCALING PROBLEM
    VERTICAL SCALING PROBLEM (likely use mfr1 and mfr2 and T)
    SYMMETRIZE NOT OPTIMIZED
    Returns the crosscorrelation function of two spike trains.
    Second one 'triggered' by the first one.
    - dp: (string): DataPath to the Neuropixels dataset.
    - U (list of integers): list of units indexes.
    - win_size (float): window size, in milliseconds
    - bin_size (float): bin size, in milliseconds
    - fs: sampling rate (Hertz). Default 30000.
    - symmetrize (bool): symmetrize the semi correlograms. Default=True.
    - normalize: normalize the correlograms. Default=False.'''
       
    # Troubleshooting
    assert fs > 0.
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
    binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
    assert binsize >= 1 # Cannot be smaller than a sample time
 
    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    win_size_bins = 2 * int(win_size*0.5/bin_size) + 1 # Both in millisecond
    assert win_size_bins >= 1
    assert win_size_bins % 2 == 1
    
    correlograms=np.zeros((len(U), len(U), win_size_bins))
    if (win_size*1./bin_size)%2==0: # even
        #binsedges=np.arange(-win_size*1./2-bin_size*1./2, win_size*1./2+bin_size*3./2, bin_size) # add 1 + two half bins to keep even centered on 0
        bins=np.arange(-win_size*1./2, win_size*1./2+bin_size, bin_size)
    elif win_size*1./bin_size%2==1: # odd
        #binsedges=np.arange(-win_size*1./2, win_size*1./2+bin_size, bin_size) # add one bin to make even centered on 0
        bins=np.arange(-win_size*1./2+bin_size*1./2, win_size*1./2+bin_size*1./2, bin_size)
        
    try:
        assert (bin_size*1e6)%(trn_binsize*1e6)==0
    except:
        if prnt: print('bin_size ({}) is not a multiple of trn_bins ({}), therefore shifts are not integers! Abort.'.format(bin_size, trn_binsize))
        return
    
    for i1, u1 in enumerate(U):
        tb1=trnb(dp, u1, trn_binsize, prnt=False) # binarized at the same sampling rate as the correlogram
        #mfr1=1000./np.mean(isi(dp, u1, prnt=False)) # in s-1
        for i2, u2 in enumerate(U):
            if prnt: print('CCG {}x{}'.foamrt(u1, u2))
            if i2>=i1:
                tb2 = trnb(dp, u2, trn_binsize, prnt=False) if i2!=i1 else tb1 # binarized at the same sampling rate as the correlogram
                #mfr2=1000./np.mean(isi(dp, u2, prnt=False)) # in s-1
                corr=np.zeros((len(bins)))
                shifts=np.asarray(np.round(bins*1./bin_size, 2)*(bin_size*1./trn_binsize), dtype=np.int64)
                #T=len(tb1) # T
                for si, shift in enumerate(shifts): # bins are centered on 0 and include 0
                    #end = '\r' if si<len(shifts)-1 else ''
                    #print('{}%...'.format(int(100*(si+1)*1./len(shifts))), end=end)
                    # C(shift) = (1./(T*bin_size*mfr1*mfr2))*S[t:0->T][(n1(t)-mfr1) * (n2(t+shift)-mfr2)]
                    tb1c = tb1#-mfr1 # n1(t)-mfr1
                    if shift>=0:
                        tb2c_shifted=np.append(tb2[shift:], np.zeros((shift)))#-mfr2 # n2(t+shift)-mfr2 (0 padding on the left)
                    elif shift<0:
                        tb2c_shifted=np.append(np.zeros((-shift)), tb2[:+shift])#-mfr2 # n2(t+shift)-mfr2 (0 padding on the right)
                    
                    C_shift=np.sum(tb1c*tb2c_shifted) # C(shift) (element wise multiplication)
                    corr[si]=C_shift#*1./(T*bin_size*mfr1*mfr2) # Normalize by recording length in ms * mfr1/2 in ms-1
                    
            else:
                pass
            
            if i1==i2:
                corr[int(.5*(len(corr)-1))]=0
            correlograms[i1, i2, :]=corr
    
    # Symmetrize
    for i1, u1 in enumerate(U):
        for i2, u2 in enumerate(U):
            corr=correlograms[i1, i2, :]
            if i1!=i2:
                correlograms[i2, i1, :]=np.array([corr[-v+1] for v in range(len(corr))])*1./(0.001*bin_size*np.sqrt(len(trn(dp, u1, prnt=False))*len(trn(dp, u2, prnt=False))))
    
    return correlograms
