# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

import os
import os.path as op; opj=op.join
from pathlib import Path
import psutil

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

from numba import njit, prange
from numba.typed import List
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
import progressbar as pgb

from npyx.utils import npa, sign, thresh_consec, zscore, split, get_bins, \
                    _as_array, _unique, _index_of, any_n_consec, \
                    assert_int, assert_float, assert_iterable, smooth

from npyx.io import read_spikeglx_meta
from npyx.gl import get_units, get_source_dp_u, get_rec_len, assert_same_dataset, assert_multi
from npyx.spk_t import trn, trnb, binarize, firing_periods,\
                        get_firing_periods, isi, mfr, train_quality

import scipy.signal as sgnl
from npyx.stats import pdf_normal, pdf_poisson, cdf_poisson, fractile_normal

def make_phy_like_spikeClustersTimes(dp, U, subset_selection='all', prnt=True, trains=None):
    '''If provided, dic must be of the form {unit1:train1InSamples, unit2:...}'''
    trains_dic={}
    if trains is None:
        for iu, u in enumerate(U):
            # Even lists of strings can be dealt with as integers by being replaced by their indices
            trains_dic[iu]=trn(dp, u, sav=True, subset_selection=subset_selection, prnt=prnt) # trains in samples
    else:
        assert len(trains)>1
        assert type(trains) in [list, np.ndarray]
        for iu, t in enumerate(trains):
            assert len(t)>0
            trains_dic[iu]=t
    spikes=make_matrix_2xNevents(trains_dic).astype('int64')

    return spikes[0,:], spikes[1,:] # equivalent of spike_times.npy and spike_clusters.npy

def make_matrix_2xNevents(dic):
    '''
    Parameters:
        - dic: dictionnary, keys are timeseries labels (eg. trials, or unit indices) and values timeseries

    Returns:
        - 2 x Nevents numpy array, labels in first row and timestamps of respective timeserie in second row.
          Format equivalent of hstack of spike_times.npy and spike_clusters.npy
    '''
    m = np.empty((2, 0))
    for k, v in dic.items():
        m = np.concatenate((m, np.vstack((v, np.full(v.shape, k)))), axis=1)
    sortedIdx = np.argsort(m[0,:])
    rows = np.array([[0], [1]])
    m = m[rows, sortedIdx]

    return m

def crosscorrelate_cyrille(dp, bin_size, win_size, U, fs=30000, symmetrize=True, subset_selection='all', prnt=False, trains=None):
    '''Returns the crosscorrelation function of two spike trains.
       - dp: (string): DataPath to the Neuropixels dataset.
       - win_size (float): window size, in milliseconds
       - bin_size (float): bin size, in milliseconds
       - U (list of integers): list of units indices.
       - fs: sampling rate (Hertz). Default 30000.
       - symmetrize (bool): symmetrize the semi correlograms. Default=True.
       - trains: dictionnary of trains, to calculate the CCG of an arbitrary list of trains in SAMPLES for fs=30kHz.'''

    #### Get clusters and times
    U=list(U)

    spike_times, spike_clusters = make_phy_like_spikeClustersTimes(dp, U, subset_selection=subset_selection, prnt=prnt, trains=trains)

    return crosscorr_cyrille(spike_times, spike_clusters, win_size, bin_size, fs, symmetrize)

def crosscorr_cyrille(times, clusters, win_size, bin_size, fs=30000, symmetrize=True):
    '''Returns the crosscorrelation function of two spike trains.
       - times: array of concatenated times of all neurons, sorted in time, in samples.
       - clusters: corresponding array of neuron indices
       - win_size (float): window size, in milliseconds
       - bin_size (float): bin size, in milliseconds
       - U (list of integers): list of units indices.
       - fs: sampling rate (Hertz). Default 30000.
       - symmetrize (bool): symmetrize the semi correlograms. Default=True.
       - trains: dictionnary of trains, to calculate the CCG of an arbitrary list of trains in SAMPLES for fs=30kHz.'''
    #### Troubleshooting
    assert fs > 0.
    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
    binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
    assert binsize >= 1 # Cannot be smaller than a sample time

    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    winsize_bins = 2 * int(.5 * win_size *1./ bin_size) + 1 # Both in millisecond
    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1

    phy_ss, spike_clusters = times, clusters
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

        # Spikes with no matching spikes in the window are masked.
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

def ccg(dp, U, bin_size, win_size, fs=30000, normalize='Hertz', ret=True, sav=True, prnt=True, subset_selection='all', again=False, trains=None):
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
      - sav (bool - default True): if True, by definition of the routine, saves the file in dp.

      returns numpy array (Nunits, Nunits, win_size/bin_size)

    '''
    assert normalize in ['Counts', 'Hertz', 'Pearson', 'zscore'], \
        "WARNING ccg() 'normalize' argument should be a string in ['Counts', 'Hertz', 'Pearson', 'zscore']."

    # Preformat
    U=list(U)
    assert len(U)>=2
    if U[0]==U[1]: U=[U[0]] # Handling autocorrelograms
    same_ds=assert_same_dataset(U) if assert_multi(dp) else False
    U_=U.copy()
    for iu,u in enumerate(U_):
        (dp1, U_[iu]) = get_source_dp_u(dp, u) if same_ds else (dp, u)
    dp=dp1;del dp1
    sortedU=U_.copy()
    if trains is not None:
        if len(sortedU)>1:
            trains=[trains[isort] for isort in np.argsort(sortedU)]
    sortedU.sort()

    bin_size = np.clip(bin_size, 1000*1./fs, 1e8)
    # Search if the variable is already saved in dp/routinesMemory
    dprm = Path(dp,'routinesMemory')
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    fn='ccg{}_{}_{}_{}({}).npy'.format(str(sortedU).replace(" ", ""), str(bin_size), str(int(win_size)), normalize, str(subset_selection)[0:50].replace(' ', '').replace('\n',''))
    if os.path.exists(Path(dprm,fn)) and not again and trains is None:
        if prnt: print("File {} found in routines memory.".format(fn))
        crosscorrelograms = np.load(Path(dprm,fn))
        crosscorrelograms = np.asarray(crosscorrelograms, dtype='float64')
    # if not, compute it
    else:
        if prnt: print("File {} not found in routines memory.".format(fn))
        crosscorrelograms = crosscorrelate_cyrille(dp, bin_size, win_size, sortedU, fs, True, subset_selection=subset_selection, prnt=prnt, trains=trains)
        crosscorrelograms = np.asarray(crosscorrelograms, dtype='float64')
        if crosscorrelograms.shape[0]<len(U): # no spikes were found in this period
            # Maybe if not any(crosscorrelograms.ravel()!=0):
            crosscorrelograms=np.zeros((len(U), len(U), crosscorrelograms.shape[2]))
        if normalize in ['Hertz', 'Pearson', 'zscore']:
            for i1,u1 in enumerate(sortedU):
                Nspikes1=len(trn(dp, u1, prnt=False, subset_selection=subset_selection))
                #imfr1=np.mean(1000./isi(dp, u1)[isi(dp, u1)>0])
                for i2,u2 in enumerate(sortedU):
                    Nspikes2=len(trn(dp, u2, prnt=False, subset_selection=subset_selection))
                    #imfr2=np.mean(1000./isi(dp, u2)[isi(dp, u2)>0])
                    arr=crosscorrelograms[i1,i2,:]
                    if normalize == 'Hertz':
                        crosscorrelograms[i1,i2,:]=arr*1./(Nspikes1*bin_size*1./1000)
                    elif normalize == 'Pearson':
                        crosscorrelograms[i1,i2,:]=arr*1./np.sqrt(Nspikes1*Nspikes2)
                    elif normalize=='zscore':
                        crosscorrelograms[i1,i2,:]=zscore(arr, 4./5)


        # Save it only if no custom trains were provided
        if sav and trains is None:
            np.save(Path(dprm,fn), crosscorrelograms)

    # Structure the 3d array to return accordingly to the order of the inputed units U
    if crosscorrelograms.shape[0]>1:
        sortedC = np.zeros(crosscorrelograms.shape)
        sortedU=np.array(sortedU)
        for i1, u1 in enumerate(U_):
            for i2, u2 in enumerate(U_):
                ii1, ii2 = np.nonzero(sortedU==u1)[0], np.nonzero(sortedU==u2)[0]
                sortedC[i1,i2,:]=crosscorrelograms[ii1, ii2, :]
    else:
        sortedC=crosscorrelograms

    return sortedC

def acg(dp, u, bin_size, win_size, fs=30000, normalize='Hertz', ret=True, sav=True, prnt=True, subset_selection='all', again=False):
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
    return ccg(dp, [u,u], bin_size, win_size, fs, normalize, ret, sav, prnt, subset_selection, again)[0,0,:]

def scaled_acg(dp, units, cut_at = 150, bs = 0.5, fs=30000, normalize='Hertz',
            min_sec = 180, again = False, first_n_minutes = 20,
            consecutive_n_seconds = 180, acg_window_len = 3, acg_chunk_size = 10,
            gauss_window_len = 3, gauss_chunk_size = 10, use_or_operator = False,
            violations_ms = 0.8, rpv_threshold = 0.05, missing_spikes_threshold=5):
    """
    - get the spike times passing our quality metric from the first 20 mins
    - get the argmax of the quality ISI
    - find the corresponding acg
    - shift the ACG in x-axis so the peak is aligned at the 100th value of the vector
    - scale the ACG with the the mfr for the quality period
    - return scaled ACG
    - depending whether it is a single unit or a list of units return a matrix
    """

    spike_clusters= np.load(Path(dp, 'spike_clusters.npy'))
    # Ensure units are an iterable of floats/ints
    if assert_int(units) or assert_float(units):
        units = [units]
    elif isinstance(units, str):
        if units.strip() == 'all':
            units = get_units(dp, quality = 'good')
        else:
            raise ValueError("You can only pass 'all' as a string")
    elif assert_iterable(units):
        pass # all good
    else:
        raise TypeError("Only the string 'all', ints, list of ints or ints disguised as floats allowed")

    return_acgs = []
    return_isi_mode = []
    return_isi_hist_counts = []
    return_isi_hist_range_clipped = []
    return_cut_acg_unnormed = []

    for unit in units:
        if len(spike_clusters[spike_clusters == unit]) > 1_000:
            # train quality throws two warnings, ignore these
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # get the spikes that passed our quality metric
                good_times_list = train_quality(dp, unit, first_n_minutes = 20,
                                                acg_window_len=acg_window_len,
                                                acg_chunk_size = acg_chunk_size,
                                                gauss_window_len = gauss_window_len,
                                                gauss_chunk_size = gauss_chunk_size,
                                                use_or_operator = use_or_operator,
                                                violations_ms = violations_ms, rpv_threshold = rpv_threshold,
                                                missing_spikes_threshold=missing_spikes_threshold)

            if len(good_times_list) >1 :

                # make a tuple to be passed as the good_sections parameter
                good_sections = [tuple(x) for x in good_times_list[0]]

                # arbitrary filter, need at least 180 good seconds to continue
                all_time = np.sum(np.ptp(good_sections, axis = 1))
                if all_time >min_sec:

                    unit_isi= isi(dp, unit, subset_selection = good_sections, again = again)/30
                    # get the mfr of the section that pass our criteria
                    mean_fr = mfr(dp, unit, subset_selection = good_sections)
                    # pass the outputs of the unit ISI (in ms) to get a histogram with given binsize
                    isi_hist_counts, isi_hist_range = np.histogram(unit_isi, bins = np.arange(0,100,bs)) # ms
                    #get the mode of the ISI values that are larges than 3ms
                    # first smooth the ISI, convolving it with a gaussian
                    isi_hist_counts = smooth(isi_hist_counts, sd=1)
                    # next the ISI and the ACG need to be made of the same shape, ISI is longer by one
                    isi_hist_range_clipped = isi_hist_range[:-1]
                    isi_mode = isi_hist_range_clipped[np.argmax(isi_hist_counts)]
                    # get the ACG for the unit
                    unit_acg = acg(dp, unit, bin_size= bs, win_size = isi_mode * 20, fs = fs, normalize = normalize,  subset_selection = good_sections, prnt = False, again = again)

                    # rewrite ISI mode so it is divided by bin size
                    isi_mode_bin = isi_mode / bs
                    half_len = unit_acg.shape[0]//2

                    # divide by MFR to normalise shape of ACG
                    cut_acg_unnormed = unit_acg[half_len:]
                    cut_acg = cut_acg_unnormed/mean_fr

                    # upsample the ACG so that the peak is at 100
                    interp_fn = interp1d(np.linspace(0, half_len, cut_acg.shape[0]),cut_acg,axis=-1)
                    new_wave = interp_fn(np.linspace(0, half_len, int(cut_acg.shape[0]*100/isi_mode_bin )))

                    normed_new = new_wave/ np.mean(new_wave[:-50])
                    return_acgs.append(normed_new[:cut_at])
                    return_isi_mode.append(isi_mode)
                    return_isi_hist_counts.append(isi_hist_counts)
                    return_isi_hist_range_clipped.append(isi_hist_range_clipped)
                    return_cut_acg_unnormed.append(cut_acg_unnormed)

                else:
                    normed_new = np.zeros(cut_at)
                    short_zeros = np.zeros(10)
                    return_acgs.append(normed_new)
                    return_isi_mode.append(short_zeros)
                    return_isi_hist_counts.append(short_zeros)
                    return_isi_hist_range_clipped.append(short_zeros)
                    return_cut_acg_unnormed.append(short_zeros)

            else:
                normed_new = np.zeros(cut_at)
                return_acgs.append(normed_new)
                short_zeros = np.zeros(10)
                return_isi_mode.append(short_zeros)
                return_isi_hist_counts.append(short_zeros)
                return_isi_hist_range_clipped.append(short_zeros)
                return_cut_acg_unnormed.append(short_zeros)

        else:
            normed_new = np.zeros(cut_at)
            return_acgs.append(normed_new)
            short_zeros = np.zeros(10)
            return_isi_mode.append(short_zeros)
            return_isi_hist_counts.append(short_zeros)
            return_isi_hist_range_clipped.append(short_zeros)
            return_cut_acg_unnormed.append(short_zeros)

    # I want to return numpy arrays, and inorder to do this the lists going into
    # the arrays need to be the same length. Hence we can find the list with the maximal length,
    # and for all lists that are not this long add np.nan values to it

    # get the maximal length of each sublist
    len_isi_hist_counts = np.max([ i.shape[0] for i in return_isi_hist_counts])
    len_isi_hist_range_clipped = np.max([i.shape[0] for i in return_isi_hist_range_clipped])
    len_cut_acg_unnormed = np.max([i.shape[0] for i in return_cut_acg_unnormed])

    # append the missing number of np.nan values to each list that is shorter than max value
    np_isi_hist_counts = np.array([list(xi) + [np.nan] * (len_isi_hist_counts - len(xi)) for xi in return_isi_hist_counts])
    np_isi_hist_range_clipped = np.array([list(xi) + [np.nan] * (len_isi_hist_range_clipped - len(xi)) for xi in return_isi_hist_range_clipped])
    np_cut_acg_unnormed = np.array([list(xi) + [np.nan] * (len_cut_acg_unnormed - len(xi)) for xi in return_cut_acg_unnormed])


    return np.vstack(return_acgs), np.array(return_isi_mode), np_isi_hist_counts, np_isi_hist_range_clipped, np_cut_acg_unnormed


def ccg_stack(dp, U_src=[], U_trg=[], cbin=0.2, cwin=80, normalize='Counts', all_to_all=False, name=None, sav=True, again=False, subset_selection='all'):
    '''
    Routine generating a stack of correlograms for faster subsequent analysis,
    between all U_src and U_trg units.
    In order to save a stack and retrieve it later, it is mandatory to provide a 'name' argument.
    It is possible to retrieve a stack with a given name without providing the corresponding units.
    Parameters:
        - dp:        string, datapath
        - U_src:     list/array, source units of correlograms
        - U_trg:     list/array, target units of correlograms
          If no U_src or U_trg are provided but a name is provided, 2 empty arrays will be returned.
        - cbin:      float, correlograms bin size (ms)
        - cwin:      float, correlogram window size (ms)
        - normalize: string, normalization of correlograms | Default: Counts
        - all_to_all: bool, if True returns a U_src x U_trg x bins array (all U_src to U_trg ccgs are computed)
                            else returns a U_src=U_trg x bins array (only the list of pairs U_src[i], U_trg[i] ccgs are computed)
        - name:      string, name of ccg stack (e.g. corticonuclear).
                     HAS to be provided so that the stack can be saved!
        - sav:       bool, whether to save the stack in routines memory.
                     Will only be saved if a name is provided!
    Returns:
        - sigstack: np array, ccg stack containing the ccgs, of shape U_src=U_trg x cwin//cbin+1 if all_to_all=False or U_src x U_trg x cwin//cbin+1 else
        - sigustack: np array, matching unit pairs for each ccg, of shape U_src=U_trg if all_to_all=False or U_src x U_trg else
    '''
    dprm = Path(dp,'routinesMemory')
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    Nu=len(U_src)+len(U_trg)
    if name is not None:
        norm={'Counts':'c', 'zscore':'z', 'Hertz':'h', 'Pearson':'p'}[normalize]
        fn='ccgstack_{}_{}_{}_{}_{}.npy'.format(name, norm, cbin, cwin, str(subset_selection)[0:50].replace(' ', '').replace('\n',''))
        fnu='ccgstack_{}_{}_{}_{}_{}_U.npy'.format(name, norm, cbin, cwin, str(subset_selection)[0:50].replace(' ', '').replace('\n',''))

        if op.exists(dprm/fn) and not again:
            stack=np.load(dprm/fn)
            ustack=np.load(dprm/fnu)
            if all_to_all:
                assert stack.ndim==3
            else:
                assert stack.ndim==2
            return stack, ustack
        else:
            if Nu==0: return npa([]),npa([])

    assert len(U_src)>0 and len(U_trg)>0, 'You need to provide at least one source and one target unit!'
    #print('Computing ccg stack...\n'.format())
    bins=get_bins(cwin, cbin)
    if all_to_all:
        pgbar=pgb.ProgressBar(maxval=len(U_src)*len(U_trg)).start()
        ustack=npa(zeros=(len(U_src), len(U_trg), 2)).astype(npa(U_src).dtype)
        stack=npa(zeros=(len(U_src), len(U_trg), len(bins))).astype(float)
        # Case where every CCG would be computed twice - gotta save time if you can
        if np.all(U_src==U_trg):
            for i1, u1 in enumerate(U_src):
                for i2, u2 in enumerate(U_trg):
                    #pgbar.update(i1*len(U_trg)+i2+1)
                    ustack[i1, i2, :]=[u1,u2]
                    if i1==i2:
                        stack[i1, i2, :]=ccg(dp, [u1, u2], cbin, cwin, normalize=normalize, prnt=False, again=again, subset_selection=subset_selection).squeeze()
                    elif i2>i1:
                        stack[i1, i2, :]=ccg(dp, [u1, u2], cbin, cwin, normalize=normalize, prnt=False, again=again, subset_selection=subset_selection)[0,1,:]
                        stack[i2, i1, :]=stack[i1, i2, ::-1]
        else:
            for i1, u1 in enumerate(U_src):
                for i2, u2 in enumerate(U_trg):
                    pgbar.update(i1*len(U_trg)+i2+1)
                    ustack[i1, i2, :]=[u1,u2]
                    stack[i1, i2, :]=ccg(dp, [u1, u2], cbin, cwin, normalize=normalize, prnt=False, again=again, subset_selection=subset_selection)[0,1,:]
    else:
        assert len(U_src)==len(U_trg)
        assert not np.any(U_src==U_trg), 'Looks like you requested to compute a CCG between a unit and itself - check U_src and U_trg.'
        pgbar=pgb.ProgressBar(maxval=len(U_src)).start()
        ustack=npa(zeros=(len(U_src), 2))
        stack=npa(zeros=(len(U_src), len(bins)))
        for i, (u1, u2) in enumerate(zip(U_src, U_trg)):
            pgbar.update(i+1)
            ustack[i, :]=[u1,u2]
            stack[i, :]=ccg(dp, [u1, u2], cbin, cwin, normalize=normalize, prnt=False, again=again, subset_selection=subset_selection)[0,1,:]

    if sav and name is not None:
        np.save(dprm/fn, stack)
        np.save(dprm/fnu, ustack)

    return stack, ustack

def get_ustack_i(U, ustack):
    '''
    Finds indices of units inside a ccg stack.
    Parameters:
        - U: 2d array or unit pairs
        - ustack: 2d or 3d array, units matching a ccg stack (returned by ccg_stack())
    '''
    U=npa(U)
    if U.ndim==1:
        assert U.shape[0]==2
        U=U.reshape((1,2))
    assert U.shape[1]==2, 'U must be 2d array or unit pairs!'
    ii=np.zeros(U.shape) if ustack.ndim==3 else np.zeros(U.shape[0])
    for i, u in enumerate(U):
        mask=(ustack==np.tile([u[0], u[1]], ustack.shape[:-1]+(1,)))
        ii[i]=npa(np.nonzero(np.all(mask, axis=ustack.ndim-1))).flatten()
    return ii.astype(np.int64)

#%% Cross spike intervals distribution (is to ISI what CCG is to ACG)

@njit(cache=True, parallel=False)
def cisi_numba_para(spk1, spk2, available_memory):
    spk1=np.sort(spk1).astype(np.float64)
    spk2=np.sort(spk1).astype(np.float64)
    memory_el=(0.01*available_memory)//spk1.itemsize
    if memory_el<((len(spk1)*len(spk2))*0.1):
        s=int(len(spk1)//((len(spk1)*len(spk2))//memory_el))
        chunks=split(List(spk1), sample_size=s, return_last=True, overlap=0)
    else:
        s=len(spk1)
        chunks=np.expand_dims(spk1, 0)
    # the trick to make things fast is to only consider
    # the relevant slice of spk2, which is possible assuming that spk1 and spk2 are sorted
    isi_1to2=np.zeros(len(spk1))
    n=chunks.shape[0]
    for i in range(n):
        chunk=chunks[i]
        if i==n-1:chunk=chunk[~np.isnan(chunk)]

        m1=np.array(list(spk2>=chunk[0])[1:]+[True]) # shift left to add spike right before
        m2=np.array([True]+list(spk2<=chunk[-1])[:-1]) # shift right to add spike right after
        chunk2=spk2[m1&m2]

        a1=(chunk*np.ones((chunk2.shape[0], chunk.shape[0]))).T
        a2=chunk2*np.ones((chunk.shape[0], chunk2.shape[0]))
        d=np.abs(a1-a2)
        for di in range(d.shape[0]):
            isi_1to2[i*s+di]=np.min(d[di])

    return isi_1to2


@njit
def cisi_numba(spk1, spk2, available_memory):
    spk1=np.sort(spk1).astype(np.float64)
    spk2=np.sort(spk1).astype(np.float64)
    memory_el=(0.01*available_memory)//spk1.itemsize
    if memory_el<((len(spk1)*len(spk2))*0.1):
        s=int(len(spk1)//((len(spk1)*len(spk2))//memory_el))
        chunks=split(List(spk1), sample_size=s, return_last=True, overlap=0)
    else:
        s=len(spk1)
        chunks=np.expand_dims(spk1, 0)
    # the trick to make things fast is to only consider
    # the relevant slice of spk2, which is possible assuming that spk1 and spk2 are sorted
    isi_1to2=np.zeros(len(spk1))
    n=chunks.shape[0]
    for i in prange(n):
        chunk=chunks[i]
        if i==n-1:chunk=chunk[~np.isnan(chunk)]

        m1=np.array(list(spk2>=chunk[0])[1:]+[True]) # shift left to add spike right before
        m2=np.array([True]+list(spk2<=chunk[-1])[:-1]) # shift right to add spike right after
        chunk2=spk2[m1&m2]

        d=np.abs(np.expand_dims(chunk,1)-chunk2)
        for di in prange(d.shape[0]):
            isi_1to2[i*s+di]=np.min(d[di])

    return isi_1to2

# @njit
# def next_cisi(spk1, spk2, direction=1):
#     t_12=np.array(list(spk1)+list(spk2))
#     i_12=np.array([False]*len(spk1)+[True]*len(spk2)) # numba compatible .astype(bool)
#     i_12=i_12[np.argsort(t_12)]
#     t_12.sort()
#     i_init=np.arange(len(t_12))
#     nxt_12=np.zeros(len(i_12))
#     m=np.array([True])
#     while np.any(m):
#         m=list((~i_12[:-1])&i_12[1:])+[False] # 0s followed by 1s
#         mshift=np.array([False]+m[:-1]) # m is the mask for spk1, mshift for spk2
#         m=np.array(m)
#         nxt_12[i_init[m]]=t_12[mshift]-t_12[m] # t of index 1 - t of index 0 = cross interspike interval
#         i_12=i_12[~m]
#         t_12=t_12[~m]
#         i_init=i_init[~m]
#     return t_12, nxt_12

# @njit
# def prev_cisi(spk1, spk2, direction=1):
#     t_12=np.array(list(spk1)+list(spk2))
#     i_12=np.array([False]*len(spk1)+[True]*len(spk2)) # numba compatible .astype(bool)
#     i_12=i_12[np.argsort(t_12)]
#     t_12.sort()
#     i_init=np.arange(len(t_12))
#     nxt_12=np.zeros(len(i_12))
#     m=np.array([True])
#     count=0
#     while np.any(m):
#         count+=1
#         m=list(i_12[:-1]&(~i_12[1:]))+[False] # 1s followed by 0s
#         mshift=np.array([False]+m[:-1]) # m is the mask for spk2, mshift for spk1
#         m=np.array(m)
#         nxt_12[i_init[mshift]]=t_12[mshift]-t_12[m]
#         i_12=i_12[~mshift]
#         t_12=t_12[~mshift]
#         i_init=i_init[~mshift]
#     return t_12, nxt_12, count

def get_cisi1(spk1, spk2, direction=0, prnt=True):
    '''
    Computes cross spike intervals i.e time differences between
    every spike of spk1 and the following/preceeding spike of spk2.
    Parameters:
        - spk1: list/array of INTEGERS, time series
        - spk2: list/array of INTEGERS, time series
        - direction: 1, -1 or 0, whether to return following or preceeding interval
                    or for 0, the smallest interval of either
                    (in this case not only consecutive 1,2 or 2,1 ISIs are considered but all spikes of 1)
    Returns:
        - cisi: cross interspike intervals corresponding to spk1 spikes
    '''
    assert direction in [1, 0, -1]

    # Concatenate and sort spike times of spk1 and 2
    # (Ensure that there is at least one spk2 spike smaller than/bigger than any spk1 spike)
    t_12=np.append(spk1, spk2)
    i_12=np.array([False]*len(spk1)+[True]*len(spk2), dtype=np.bool)
    i_12=i_12[np.argsort(t_12)]
    t_12.sort()

    # Get spikes 1 and 2 relative indices in the concatenated sorted train
    i1_12,=np.nonzero(~i_12)
    i2_12,=np.nonzero(i_12)

    # get previous and next spk2 for every spk1
    # trick: argmax returns the first max value if several (i.e. 1 in binary array)
    # need to ensure th at the mask resulting from the comparison has at least one 1
    # i.e. the very first and very last spikes should be from spk2 - cf.startpad and endpad
    memory_el=(0.01*psutil.virtual_memory().available)//spk1.itemsize
    if memory_el<((len(spk1)*len(spk2))*0.1):
        s=int(len(spk1)//((len(spk1)*len(spk2))//memory_el))
        chunks=split(List(i1_12), sample_size=s, return_last=True, overlap=0)
    else:
        s=len(spk1)
        chunks=i1_12[np.newaxis,:]

    nxt2_i=np.zeros(len(i1_12), dtype=int)
    nanmasknxt=np.zeros(len(i1_12), dtype=bool)
    prv2_i=np.zeros(len(i1_12), dtype=int)
    nanmaskprv=np.zeros(len(i1_12), dtype=bool)
    n=chunks.shape[0]
    for i in range(n):
        chunk=chunks[i]
        if i==n-1:chunk=chunk[~np.isnan(chunk)]
        m1=np.append((i2_12>=chunk[0])[1:], [True]) # shift left to add spike right before
        m2=np.append([True], (i2_12<=chunk[-1])[:-1]) # shift right to add spike right after
        chunk2=i2_12[m1&m2]
        if direction in [0,1]:
            m=chunk2>=chunk[:,np.newaxis]
            nxt2_i[i*s:i*s+chunk.shape[0]]=chunk2[np.argmax(m, axis=1)]
            nanmasknxt[i*s:i*s+chunk.shape[0]]=np.all(~m, axis=1)
        if direction in [0,-1]:
            m=chunk2[::-1]<=chunk[:,np.newaxis]
            prv2_i[i*s:i*s+chunk.shape[0]]=chunk2[::-1][np.argmax(m, axis=1)]
            nanmaskprv[i*s:i*s+chunk.shape[0]]=np.all(~m, axis=1)
        if prnt: print(f'Chunk {i+1}/{n} processed...')
    del m
    nxt2_t=t_12[nxt2_i].astype(float)
    nxt2_t[nanmasknxt]=np.nan
    prv2_t=t_12[prv2_i].astype(float)
    prv2_t[nanmaskprv]=np.nan

    # among the next and/or previous spk2 spikes, keep the closest one in time
    # (just keep the time diff, not the spk2 spike time)
    if direction==0:
        cisi=np.nanmin(np.vstack([(nxt2_t-spk1), (spk1-prv2_t)]), axis=0)
    elif direction==1:
        cisi=nxt2_t-spk1
    elif direction==-1:
        cisi=spk1-prv2_t

    return cisi

def get_cisi(spk1, spk2, direction=0, prnt=True):
    '''
    Computes cross spike intervals i.e time differences between
    every spike of spk1 and the following/preceeding spike of spk2.
    Parameters:
        - spk1: list/array of INTEGERS, time series
        - spk2: list/array of INTEGERS, time series
        - direction: 1, -1 or 0, whether to return following or preceeding interval
                    or for 0, the smallest interval of either
                    (in this case not only consecutive 1,2 or 2,1 ISIs are considered but all spikes of 1)
    Returns:
        - spk_1to2: spikes of spk1 directly followed/preceeded by a spike of spk2
        - isi_1to2: corresponding interspike intervals
    '''
    assert direction in [1, 0, -1]
    spk1=np.sort(spk1).astype(np.float64)
    spk2=np.sort(spk2).astype(np.float64)
    isi_1to2=np.zeros(len(spk1))
    # Chunks of 50% of available memory.
    # Chunk size is overestimated because chunks.shape[1] is
    # len(spk2[start_spk2:end_spk2[1]]) not len(spk2)
    memory_el=(0.01*psutil.virtual_memory().available)//spk1.itemsize
    if memory_el<((len(spk1)*len(spk2))*0.1):
        s=int(len(spk1)//((len(spk1)*len(spk2))//memory_el))
        chunks=split(List(spk1), sample_size=s, return_last=True, overlap=0)
    else:
        s=len(spk1)
        chunks=spk1[np.newaxis,:]
    # the trick to make things fast is to only consider
    # the relevant slice of spk2, which is possible assuming that spk1 and spk2 are sorted
    n=chunks.shape[0]
    for i in range(n):
        chunk=chunks[i]
        if i==n-1:chunk=chunk[~np.isnan(chunk)]
        # start_spk2=np.nonzero(spk2<chunk[0])[0][-1:]
        # if not np.any(start_spk2): start_spk2=np.array([0]) #  case when first spk2 is later than all chunk spikes
        # end_spk2=np.nonzero(spk2>chunk[-1])[0][:1]+1
        # if not np.any(end_spk2): chunk2=spk2[start_spk2[0]:] #  case when last spk2 is earlier than all chunk spikes
        # else: chunk2=spk2[start_spk2[0]:end_spk2[0]]
        m1=np.append((spk2>=chunk[0])[1:], [True]) # shift left to add spike right before
        m2=np.append([True], (spk2<=chunk[-1])[:-1]) # shift right to add spike right after
        chunk2=spk2[m1&m2]
        d=(chunk[:, np.newaxis]-chunk2)
        if direction==1:
            d*=-1
            d[d<0]=np.nan
        elif direction==-1:
            d[d<0]=np.nan
        elif direction==0:
            d=np.abs(d)
        isi_1to2[i*s:i*s+d.shape[0]]=np.nanmin(d, axis=1)
        if prnt: print(f'Chunk {i+1}/{n} processed...')

    return isi_1to2

def par_process(i, chunk, spk2, n, direction):
    # the trick to make things fast is to only consider
    # the relevant slice of spk2, which is possible assuming that spk1 and spk2 are sorted
    if i==n-1:chunk=chunk[~np.isnan(chunk)]
    # start_spk2=np.nonzero(spk2<chunk[0])[0][-1:]
    # if not np.any(start_spk2): start_spk2=np.array([0]) #  case when first spk2 is later than all chunk spikes
    # end_spk2=np.nonzero(spk2>chunk[-1])[0][:1]+1
    # if not np.any(end_spk2): chunk2=spk2[start_spk2[0]:] #  case when last spk2 is earlier than all chunk spikes
    # else: chunk2=spk2[start_spk2[0]:end_spk2[0]]
    m1=np.append((spk2>=chunk[0])[1:], [True]) # shift left to add spike right before
    m2=np.append([True], (spk2<=chunk[-1])[:-1]) # shift right to add spike right after
    chunk2=spk2[m1&m2]
    d=(chunk[:, np.newaxis]-chunk2)
    if direction==1:
        d*=-1
        d[d<0]=np.nan
    elif direction==-1:
        d[d<0]=np.nan
    elif direction==0:
        d=np.abs(d)
    return np.nanmin(d, axis=1)

def get_cisi_parprocess(spk1, spk2, direction=0, prnt=True):
    '''
    Computes cross spike intervals i.e time differences between
    every spike of spk1 and the following/preceeding spike of spk2.
    Parameters:
        - spk1: list/array of INTEGERS, time series
        - spk2: list/array of INTEGERS, time series
        - direction: 1, -1 or 0, whether to return following or preceeding interval
                    or for 0, the smallest interval of either
                    (in this case not only consecutive 1,2 or 2,1 ISIs are considered but all spikes of 1)
    Returns:
        - spk_1to2: spikes of spk1 directly followed/preceeded by a spike of spk2
        - isi_1to2: corresponding interspike intervals
    '''
    assert direction in [1, 0, -1]
    spk1=np.sort(spk1).astype(np.float64)
    spk2=np.sort(spk2).astype(np.float64)
    # Chunks of 50% of available memory.
    # Chunk size is overestimated because chunks.shape[1] is
    # len(spk2[start_spk2:end_spk2[1]]) not len(spk2)
    memory_el=(0.01*psutil.virtual_memory().available)//spk1.itemsize
    if memory_el<((len(spk1)*len(spk2))*0.1):
        s=int(len(spk1)//((len(spk1)*len(spk2))//memory_el))
        chunks=split(List(spk1), sample_size=s, return_last=True, overlap=0)
    else:
        s=len(spk1)
        chunks=spk1[np.newaxis,:]

    n=chunks.shape[0]
    inputs=[(i, chunk) for i, chunk in enumerate(chunks)]
    results=Parallel(n_jobs=num_cores, backend="threading")(delayed(par_process)(inp[0], inp[1], spk2, n, direction) for inp in inputs)

    return np.concatenate(results).ravel()

#%% Pairwise correlations, synchrony, population coupling

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
        M[i+1,:] = bnr(t, b, rec_len)
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

def synchrony_regehr(CCG, cbin, sync_win=1, fract_baseline=2./5):
    '''
    - CCG: crosscorrelogram array, units does not matter. Should be long enough.
    - cbin"correlogram binsize in millisecond
    - sync_win: synchrony full window in milliseconds
    - baseline_fract: CCG fraction to use to compute baseline
    '''
    nbins=int(sync_win/cbin)+1
    sync_CCG=CCG[len(CCG)//2-nbins//2:len(CCG)//2+nbins//2+1]
    
    start_index = int(len(CCG)*fract_baseline/2)
    end_index = int(len(CCG)*(1-fract_baseline)/2)
    baseline_CCG=np.append(CCG[:start_index],CCG[end_index:])

    sync=np.mean(sync_CCG)/np.mean(baseline_CCG)

    return sync

def synchrony(CCG, cbin, sync_win=1, fract_baseline=4./5):
    '''
    - CCG: crosscorrelogram array, units does not matter. Should be long enough.
    - cbin: correlogram binsize in millisecond
    - sync_win: synchrony full window in milliseconds
    - baseline_fract: CCG fraction to use to compute baseline
    '''
    assert CCG.ndim==1
    CCG=zscore(CCG, fract_baseline)
    nbins=int(sync_win/cbin)+1
    left = len(CCG)//2-nbins//2
    right = len(CCG)//2+nbins//2+1
    sync_CCG=CCG[left:right]

    sync=np.mean(sync_CCG)

    return sync

def cofiring_tags(t1, t2, fs, t_end, b=1, sd=1000, th=0.02, again=False, dp=None, u2=None):
    '''
    Returns a boolean array of len of train of t.
    Each t timestamp is tagged True if t1 timestamp(s) occur in the same period,
    False if it is not because the cell was lost due to drift.

    By 'firing' we mean firing above th*100% of its mean firing rate (see get_firing_periods()).

    Serves to compute the denominator of %cells firing around a given spike.
    '''
    periods = firing_periods(t2, fs, t_end, b, sd, th, again, dp, u2)

    tags=(t1*0).astype(bool)
    for p in periods:
        tags=tags|((t1>=p[0])&(t1<=p[1]))

    return tags

def frac_pop_sync(t1, trains, fs, t_end, sync_win=2, b=1, sd=1000, th=0.02, again=False, dp=None, U=None):
    '''
    Returns an array of size len(t1),
    consisting of the fraction of timeseries in trains
    having a timestamp occurring within 'sync_win' ms of each t1 time stamp.
        Denominator - running total N of cells firing, handles drift
        Numerator - N of cells firing withing the predefined synchorny window

    Parameters:
    - t1: np array, time series in SAMPLES - MUST BE INTEGERS
    - trains: list of np arrays in, in SAMPLES - MUST BE INTEGERS
    - fs: float in Hz, t1 and trains sampling frequency
    - t_end: int in samples, end of recording of t1 and trains, in samples
    - sync_win: float in ms, synchrony window to define synchrony
    - b: int in ms, binsize defining the binning of timestamps to define 'broad firing periods' (see npyx.spk_t.firing_periods)
    - sd: int in ms, gaussian window sd to convolve the binned timestamps defining 'broad firing periods' (see npyx.spk_t.firing_periods)
    - th: float [0-1], threshold defining the fraction of mean firing rate reached in the 'broad firing periods' (see npyx.spk_t.firing_periods)
    - again: bool, whether to recompute the firing periods of units in U (trains)
    - dp: string, datapath to dataset with units corresponding to trains - optional, to ensure fast loading of firing_periods
    - U: list, units matching trains (NOT t1)!

    Returns:
    - frac_pop_sync: np array of shape (len(t1),),
      fraction of trains firing within sync_win ms of t1 time stamps [0-1].
    '''
    if U is None:
        U=[None]*len(trains)
    else:
        assert len(U)==len(trains), 'u1 should not be included in U!'
        assert dp is not None, 'Need to provide datapath along with unit indices.'
        t_end = np.load(Path(dp,'spike_times.npy')).ravel()[-1]
    if t_end is None: t_end=np.max(np.concatenate(trains))

    sync_win=sync_win*fs/1000 # conversion to samples
    # Trick: simply threshold the crossinterspike interval!
    # Each spike gets a 1 or 0 tag for whether the cell#2 fired within this window or not.
    N_pop_firing=(t1*0).astype(float)
    pop_sync=t1*0
    for it2, t2 in enumerate(trains):
        N_cell_firing=cofiring_tags(t1, t2, fs, t_end, b, sd, th, again, dp, U[it2]) # denominator
        N_pop_firing=N_pop_firing+N_cell_firing.astype(int)
        cell_sync=(get_cisi(t1, t2, direction=0, prnt=False)<=sync_win/2) # UNDERCOVER BUG, sync_win not originally converted in samples!!
        pop_sync=pop_sync+(cell_sync&N_cell_firing).astype(int) # cell_sync only counts when cell is considered to fire (single spikes ignored)

    # Last spike will be 0 if t1 last spike happens is the last to happen of the bunch
    # so just give it the value of the spike before
    N_pop_firing[-1]=N_pop_firing[-2]
    # no division by 0 allowed (reflects that cases where no one fired do not count)
    N_pop_firing[N_pop_firing==0]=np.nan
    return pop_sync/N_pop_firing

def fraction_pop_sync(dp, u1, U, sync_win=2, b=1, sd=1000, th=0.02, again=False,
                      t1=None, trains=None, fs=None, t_end=None):
    f'''Wrapper for frac_pop_sync:
        {frac_pop_sync.__doc__}'''
    if t1 is None:
        t1=trn(dp, u1, enforced_rp=0)
        trains=[trn(dp, u2, enforced_rp=0) for u2 in U]
        fs=read_spikeglx_meta(dp)['sRateHz']
        t_end = np.load(Path(dp,'spike_times.npy')).ravel()[-1]

        return frac_pop_sync(t1, trains, fs, t_end, sync_win=2, b=1, sd=1000, th=0.02, again=again, dp=dp, U=U)
    else:
        assert trains is not None
        assert fs is not None
        assert t_end is not None

        return frac_pop_sync(t1, trains, fs, t_end, sync_win=2, b=1, sd=1000, th=0.02)


def get_cm(dp, units, cbin=0.2, cwin=100, b=5, corrEvaluator='CCG', subset_selection='all'):
    '''Make correlation matrix.
    dp: datapath
    units: units list of the same dataset
    b: bin of spike train if covar, corrcoeff or corrcoeff_MB is used as an evaluator, in milliseconds
    cbin, cwin: CCG bin and win, if CCG is used as correlation evaluator
    corrEvaluator: metric used to evaluate correlation, in ['CCG', 'covar', 'corrcoeff_eleph', 'corrcoeff_MB']
    subset_selection: section of the Neuropixels recording used for evaluation of correlation.'''

    # Sanity checks
    allowedCorEvals = ['CCG', 'corrcoeff_MB']
    try:
        assert corrEvaluator in allowedCorEvals
    except:
        print('WARNING: {} should be in {}. Exiting now.'.format(corrEvaluator, allowedCorEvals))
        return

    # Initialize empty arrays

    if corrEvaluator =='corrcoeff_MB':
        Nbins_bms = len(trnb(dp, units[0], b)) # b in ms
        trnbM = npa(zeros=(len(units), Nbins_bms))
    elif corrEvaluator == 'CCG':
        cmCCG=npa(empty=(len(units), len(units)))

    # Populate empty arrays
    for i1, u1 in enumerate(units):
        if corrEvaluator =='corrcoeff_MB':
            trnbM[i1,:]=trnb(dp, u1, b, subset_selection=subset_selection) # b in ms
        elif corrEvaluator == 'CCG':
            for i2, u2 in enumerate(units):
                if u1==u2:
                    cmCCG[i1, i2]=0
                if i1<i2:
                    CCG = ccg(dp, [u1, u2], cbin, cwin, normalize='Counts', subset_selection=subset_selection,prnt=False)[0,1,:]
                    cmCCG[i1, i2] = cmCCG[i2, i1] = synchrony(CCG, cbin, sync_win=1, fract_baseline=2./5)

    # Set correlation matrix and plotting parameters
    if corrEvaluator == 'CCG':
        cm = cmCCG
    elif corrEvaluator == 'corrcoeff_MB':
        cm = pearson_corr(trnbM)

    return cm

#%% Assessment of significance of correlogram modulation

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
        #print("You can use this test, of confidence {0} since the probability of encountering a triplet by chance is {1:.3f}.".format(a, expected_triplets))
        return True
    else:
        print("You CANNOT use this test, of confidence {0} since the probability of encountering a triplet by chance is {1:.3f}.".format(a, expected_triplets))
        return False

def KopelowitzCohen2014_ccg_significance(CCG, cbin=0.2, cwin=100, p_th=0.01, n_consec_bins=3, sgn=-1, fract_baseline=4./5,
                                         law='Normal', multi_comp=False, bin_wise=False, ret_values=True, only_max=True, plot=False):
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
    assert 0<p_th<1, "p_th should be between 0 and 1!"
    assert sgn in [-1,1]
    assert n_consec_bins>=1 and round(n_consec_bins)==n_consec_bins

    if n_consec_bins in [2,3]:
        assert law=='Normal', "Using more than 1 bin is not handled when assuming a Poisson distribution."
        assert canUse_Nbins(p_th, cwin, cbin, n_consec_bins)
    if not bin_wise: # else: compute one CI for each bin
        if law=='Poisson':
            crosses=[] # Poisson not handled yet - how to end up with natural integers as ccg values? Spike counts? Can multiplie by random big number?
        elif law=='Normal':
            threshold=fractile_normal(1-p_th/2)*sgn
            CCG=zscore(CCG, frac=fract_baseline) # Z-score
            crosses=thresh_consec(CCG, threshold, sgn=sgn, n_consec=n_consec_bins, only_max=only_max)
    else:
        crosses=[] # bin_wise not handled yet

    if plot:
        fig=plot_pval_borders(CCG, p_th, dist='normal', gauss_baseline_fract=fract_baseline, x=np.arange(-(len(CCG)//2*cbin), len(CCG)//2*cbin+cbin, cbin),
                          xlabel='Time (ms)', ylabel='crosscorrelation (z-score)', title='Test: Kopelowitz et al. 2014'.format(p_th))
        return fig

    if ret_values:
        return crosses
    return np.any(crosses)

def StarkAbeles2009_ccg_sig(CCG, W, WINTYPE='gauss', HF=None, CALCP=True, sgn=-1):
    '''
    Predictor and p-values for CCG using convolution.

    Parameters:
        - CCG: 1D numpy array (a single CCG) or 2D (CCGs in columns). Has to be non-negative integers (counts)
        - W: int, convolution window standard deviation (in samples). Has to be smaller than the CCG length.
        - WINTYPE: string, window type -> 'gaussian' - with SD of W/2; has optimal statistical properties
                                  'rect' - of W samples; equivalent to jittering one spike train by a rectangular window of width W
                                  'triang' - of ~2W samples; equivalent to jittering both trains by a rectangular window of width W
        - CALP: bool, if true compute p value based on a poisson ditribution with a continuity correction
        - sgn: -1 or 1, whether to return small p-values for troughs or peaks
     Returns:
        - PVALS       p-values (bin-wise) for higher (if sgn is '+') or lower than (if '-') chance.
                      If p-val<0.001 at a given point,
        - PRED        predictor(expected values)

    ADVICE                    for minimal run-time, collect multiple CCHs in
                                  the columns of CCH and call this routine once

    revisions
    11-jan-12 added qvals for deficient counts. to get global significance
              (including correction for multiple comparisons), check crossing of alpha
              divided by the number of bins tested
    17-aug-19 cleaned up
    '''
    ## Local functions
    def local_firfilt(x, win):
        '''
        Zero-phase lag low-pass filtering of x's columns with the FIR W.
        '''
        C = len(win)
        D = int(np.ceil( C / 2 ) - 1)
        Y = sgnl.lfilter(win.astype(float), 1,
                         np.vstack([np.flipud(x[:C, :]).astype(float), x.astype(float), np.flipud(x[-1-C+1:,:])]).astype(float),
                         axis=0) # pad with reversed CCG edges at the beginning and the end to prevnt edge effects...
        Y = Y[C+D:-C+D,:] #... then remove them
        return Y

    def local_gausskernel(sigmaX, N):
        '''
        1D Gaussian kernel K with N samples and SD sigmaX.
        '''
        x = np.arange(-(N-1)/ 2, ( N - 1 ) / 2 + 1)
        K = 1/(2*np.pi*sigmaX )*np.exp(-(x**2/2/sigmaX**2));
        return K

    ## Preprocess arguments

    assert sgn in [1,-1]

    assert sum(CCG<0) <= 0, 'CCG seems to contain negative integers!'

    if CCG.ndim==1: CCG=CCG.reshape((1, CCG.shape[0]))
    m, n = CCG.shape
    if m == 1:
        CCG             = CCG.T
        nsamps          = n
    else:
        nsamps          = m

    winlist=['gauss', 'rect', 'triang']
    assert W == round(W) and W >= 1, 'W must be non-negative integer!'
    W=int(W)
    assert WINTYPE in winlist, "WINTYPE should be either 'gauss', 'rect' or 'triang', not {}.".format(WINTYPE)
    if HF is None: HF={'gauss':0.6, 'rect':0.42, 'triang':0.63}[WINTYPE]
    assert 0<HF<1, 'HF should be between 0 and 1.'

    ## Compute the convolution window
    conv_wins = {
            winlist[0]: {0: (local_gausskernel( W/2, 6 * W/2+1), W/2*3), # gaussian even W
                         1: (local_gausskernel( W/2, 6 * W/2+2), W/2*3+0.5)}, # gaussian odd W
            winlist[1]: {0: (np.ones((1, W+1)), W/2), # rect even W
                         1: (np.ones((1, W)), np.ceil(W/2)-1)}, # rect odd W
            winlist[2]:{0: (sgnl.triang(2*W+1), W), # triang even W
                        1: (sgnl.triang(2*W-1), W-1)} # triang odd W
            }
    win, cidx = conv_wins[WINTYPE][W%2]

    win[int(cidx)] = win[int(cidx)]*(1-HF)
    win = win/sum(win)

    assert nsamps >= ( 1.5 * len( win ) ),'CCG-W mismatch (W too large for CCG length: reduce W or elongate CCG)'

    ## Compute a predictor by convolving the CCG with the window
    pred = local_firfilt(CCG, win); # pred is convolved CCG

    ## Compute p-value based on a Poisson ditribution with a continuity correction
    if CALCP:
        pvals = npa(zeros=CCG.shape)
        for i, (c, p) in enumerate(zip(CCG.flatten(),pred.flatten())):
            pvals[i] = 1 - cdf_poisson(c-1, p) - pdf_poisson(c, p)*0.5; # excess, deterministic
    else:
        pvals = np.nan

    if sgn==-1: pvals = 1-pvals # deficient

    return pred, pvals

def StarkAbeles2009_ccg_significance(CCG, cbin, p_th, n_consec, sgn, W_sd, ret_values=True, plot=False, only_max=True):
    '''
    Parameters:
        - CCG: numpy array, crosscorrelogram in Counts
        - cbin: float, CCG bins value, in milliseconds. Used to convert W_sd (ms) in samples.
        - pval_thresh: float [0-1], threshold of modulation, in pvalue (based on Poisson distribution with continuity correction)
        - n_consec: int, number of consecutive
        - sgn: 1 or -1, direction of threshold crossing, either positive (1) or negative (-1)
        - W_sd: float, sd of convolution window, in millisecond
          (this is the standard deviation of the gaussian used to compute the predictor = convolved CCG).
          E.g. if looking for monosynaptic event, use 5 millisecond.
        - ret_values: bool, returns values of CCG corresponding to threshold crosses if True, in Poisson standard deviations
    '''

    assert np.all(CCG==np.round(CCG)), 'CCG should be in counts -> integers!'
    assert 0<p_th<1, "p_th should be between 0 and 1!"
    assert n_consec>=1 and round(n_consec)==n_consec

    W_sd=int(W_sd/cbin)
    pred, pvals = StarkAbeles2009_ccg_sig(CCG, W=2*W_sd, WINTYPE='gauss', HF=None, CALCP=True, sgn=sgn)
    pred, pvals = pred.flatten(), pvals.flatten()

    if plot:
        fig=plot_pval_borders(CCG, p_th, dist='poisson', Y_pred=pred, x=np.arange(-(len(CCG)//2*cbin), len(CCG)//2*cbin+cbin, cbin),
                          xlabel='Time (ms)', ylabel='crosscorrelation (Counts)', title='Test: Stark et al. 2009'.format(p_th))
        return fig

    if ret_values:
        sig_pvals=thresh_consec(pvals, p_th/2, sgn=-1, n_consec=n_consec, only_max=only_max)
        poisson_zscore=(CCG-pred)/np.sqrt(pred)
        for sp in sig_pvals: sp[1,:]=poisson_zscore[sp[0,:].astype(np.int)]
        return sig_pvals

    comp = (pvals<=p_th/2)
    return any_n_consec(comp, n_consec, where=False)


def get_cross_features(cross, cbin, cwin):
    '''Returns features of a correlogram modulation as defined in Kopelowitz et al. 2014.
    Parameters:
        - cross: 2d array, with cross[0] being the indices and cross[1] the values of a CCG significant modulation.
        - cbin: ccg bin size (ms)
        - cwin: ccg window size (ms)
    Returns:
        - a tuple of 8 features below:
        l_ms:  the left edge in ms,
        r_ms:  the right edge in ms,
        amp_z: the amplitude in z-score units (standard deviations from predictor depending of the distribution used),
        t_ms:  the time of the highest peak/deepest trough,
        and the ones below as defined in KopelowitzCohen2014:
            n_triplets:    the number of triplets of consecutive bins beyond the significance threshold (depends on bin size),
            n_bincrossing: the number of consecutive bins beyond the significance threshold (depends on bin size),
            bin_heights:   the mean amp_z across all significant bins,
            entropy:       see KopelowitzCohen2014
    '''
    nbins=cwin/cbin+1
    # for positive crosses, l_ind is the index of the first bin above threshold;
    # r_ind is the index of the first bin below threshold.
    l_ind, r_ind = cross[0,0], cross[0,-1]+1
    l_ms, r_ms = (l_ind-(nbins-1)*1./2)*cbin, (r_ind-(nbins-1)*1./2)*cbin
    amp_z=max(np.abs(cross[1,:]))*sign(cross[1,0])
    t_ind=cross[0,:][cross[1,:]==amp_z][0] # If there are 2 equal maxima, the 1st one is picked
    t_ms=(t_ind-(nbins-1)*1./2)*cbin
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
    if np.inf in [entropy]:entropy=0

    return (l_ms, r_ms, amp_z, t_ms, n_triplets, n_bincrossing, bin_heights, entropy)

def get_ccg_sig(CCG, cbin, cwin, p_th=0.02, n_consec_bins=3, sgn=0, fract_baseline=4./5, W_sd=10, test='Poisson_Stark', ret_features=True, only_max=True):
    '''
    Parameters:
        - CCG: 1d array,
        - cbin: float, CCG bin size (ms)
        - cwin: float, CCG window size (ms)
        - p_th: float, significance threshold in pvalues
        - n_consec_bins: int, number of bins beyond significance threshold,
        - sgn: -1, 0 or 1, sign of the modulation (negative, either or positive).
               WARNING if sgn=0 and only_max=True, only the largest modulation will be returned in absolute value
               so there can be different results for sgn=-1 (or 1) and 0!
        - fract_baseline: float, fraction of the CCG used to compute the Ho mean and std if test='Normal_Kopelowitz'. | Default 4./5
        - W_sd: float, size of the hollow gaussian window used to compute the correlogram predictor if test='Poisson_Stark' in ms | Default 10
        - test: 'Normal_Kopelowitz' or 'Poisson_Stark', test to use to assess significance | Default Poisson_Stark
        - ret_features: bool, whether to return or not the features tuples instead of the crosses indices and values.
        - only_max: bool, whether to return only the largest significant modulation of the correlogram (there can be several!)

        Returns:
            if ret_features==False:
                - crosses: list of 2xNbins 2d arrays [indices, values] where 'indices' are the ccg indices of the significant modulation and 'values' the respective ccg values,
                           in units of standard deviations from predictor (based on normal or poisson distribution)
            else:
                - features: list of tuples (1 per modulation) containing the features (see get_cross_features() doc)
        '''

    assert test in ['Normal_Kopelowitz', 'Poisson_Stark']
    assert sgn in [0,1,-1], "sgn should be either 0, 1 or -1!"

    crosses=[]
    if test=='Normal_Kopelowitz':
        if sgn==0 or sgn==1:
            crosses+=KopelowitzCohen2014_ccg_significance(CCG, cbin, cwin, p_th, n_consec_bins, 1, fract_baseline, ret_values=True, only_max=False)
        if sgn==0 or sgn==-1:
            crosses+=KopelowitzCohen2014_ccg_significance(CCG, cbin, cwin, p_th, n_consec_bins, -1, fract_baseline, ret_values=True, only_max=False)
    elif test=='Poisson_Stark':
        if sgn==0 or sgn==1:
            crosses+=StarkAbeles2009_ccg_significance(CCG, cbin, p_th, n_consec_bins, 1, W_sd, ret_values=True, only_max=False)
        if sgn==0 or sgn==-1:
            crosses+=StarkAbeles2009_ccg_significance(CCG, cbin, p_th, n_consec_bins, -1, W_sd, ret_values=True, only_max=False)

    if only_max and len(crosses)>0:
        cross=crosses[0]
        for c in crosses[1:]:
            if max(abs(c[1,:]))>max(abs(cross[1,:])): cross = c
        crosses=[cross]
        assert len(crosses)==1

    if not ret_features: return crosses

    # Compute and return crosses features
    return [get_cross_features(cross, cbin, cwin) for cross in crosses]


def ccg_sig_stack(dp, U_src, U_trg, cbin=0.5, cwin=100, name=None,
                  p_th=0.01, n_consec_bins=3, sgn=-1, fract_baseline=4./5, W_sd=10, test='Poisson_Stark',
                  again=False, againCCG=False, ret_features=False, only_max=True, subset_selection='all'):
    '''
    Parameters:
        - dp: string, datapath to manually curated kilosort output
        - U_src: list/array of source units of the correlograms to consider for significance assessment
        - U_trg: list/array of target units of the correlograms to consider for significance assessment
        - cbin: float, CCG bin size (ms)
        - cwin: float, CCG window size (ms)
        - name: name of the ccg stack to consider for significance assessment. HAS to be provided to be able to save and reload the ccg stack in the future.
        - p_th: float, significance threshold in pvalues
        - n_consec_bins: int, number of bins beyond significance threshold,
        - sgn: -1, 0 or 1, sign of the modulation (negative, either or positive).
               WARNING if sgn=0 and only_max=True, only the largest modulation will be returned in absolute value
               so there can be different results for sgn=-1 (or 1) and 0!
        - fract_baseline: float, fraction of the CCG used to compute the Ho mean and std if test='Normal_Kopelowitz'. | Default 4./5
        - W_sd: float, size of the hollow gaussian window used to compute the correlogram predictor if test='Poisson_Stark' in ms | Default 10
        - test: 'Normal_Kopelowitz' or 'Poisson_Stark', test to use to assess significance | Default Poisson_Stark
        - again: bool, whether to reassess significance of ccg stack rather than loading from memory if already computed in the past.
        - againCCG: bool, whether to recompute ccg stack rather than loading from memory if already computed in the past.
        - ret_features: bool, whether to return or not the features dataframe instead of the crosses indices and values.

        Returns:
            if ret_features==False:
                - sigstack: np array, ccg stack containing the significant ccgs, of shape NsignificantUnits x cwin//cbin+1
                - sigustack: np array, matching unit pairs for each significant ccg, of shape NsignificantUnits
            else:
                - sigstack: see above.
                - sigustack: see above.
                - features: dataframe of NsignificantUnits x
                            ['uSrc'', uTrg', 'l_ms', 'r_ms', 'amp_z', 't_ms', 'n_triplets', 'n_bincrossing', 'bin_heights', 'entropy']
                            (see __doc__ of get_ccg_sig() for features description)
        '''
    assert test in ['Normal_Kopelowitz', 'Poisson_Stark']
    assert sgn in [0,1,-1], "sgn should be either 0, 1 or -1!"

    feat_columns=['uSrc', 'uTrg', 'l_ms', 'r_ms', 'amp_z', 't_ms',
                  'n_triplets', 'n_bincrossing', 'bin_heights', 'entropy']

    # Directly load sig stack if was already computed
    if name is not None:
        # in signame, only parameters not fed to ccg_stack
        # (as others will already be added to the saved file name by ccg_stack)
        signame=name+'-{}-{}-{}-{}-{}'.format(test, p_th, n_consec_bins, fract_baseline, W_sd)
        dprm = Path(dp,'routinesMemory')
        if not op.isdir(dprm): os.makedirs(dprm)
        feat_path=Path(dp,dprm,'ccgstack_{}_{}_{}_{}_{}_{}_{}_features.csv'.format(\
                       signame, 'Counts', cbin, cwin, str(subset_selection)[0:50].replace(' ', '').replace('\n',''), sgn, only_max))

        sigstack, sigustack = ccg_stack(dp, [], [], cbin, cwin, normalize='Counts', all_to_all=False, name=signame, again=again,
                                        subset_selection=subset_selection)
        if np.any(sigstack): # will be empty if the array exists but again=True
            if not ret_features:
                return sigstack, sigustack
            if op.exists(feat_path):
                features=pd.read_csv(feat_path)
                return sigstack, sigustack, features
            features=pd.DataFrame(columns=feat_columns)
            for i,c in enumerate(sigstack):
                pks=get_ccg_sig(c, cbin, cwin, p_th, n_consec_bins, sgn,
                                fract_baseline, W_sd, test, ret_features=ret_features, only_max=only_max)
                for p in pks:
                    features=features.append(dict(zip(features.columns,np.append(sigustack[i, :], p))), ignore_index=True)
            features.to_csv(feat_path, index=False)
            return sigstack, sigustack, features

    assert any(U_src)&any(U_trg)
    ptdic={1:'peak', -1:'trough'}
    sigustack=[]
    if ret_features: features=pd.DataFrame(columns=feat_columns)

    stack, ustack = ccg_stack(dp, U_src, U_trg, cbin, cwin, normalize='Counts', all_to_all=True, name=name, again=againCCG,
                              subset_selection=subset_selection)
    same_src_trg=np.all(U_src==U_trg) if len(U_src)==len(U_trg) else False
    inco=False
    if same_src_trg:
        if len(np.unique(ustack))!=len(np.unique(U_src)): inco=True
        else:
            if not np.all(np.unique(ustack)==np.unique(U_src)): inco=True
    if inco:
        print(f'Incoherence detected between loaded ccg_stack ({len(np.unique(ustack))} units) \
              and expected ccg_stack ({len(U_src)} units) - recomputing as if againCCG were True...')
        stack, ustack = ccg_stack(dp, U_src, U_trg, cbin, cwin, normalize='Counts', all_to_all=True, name=name, again=True,
                                  subset_selection=subset_selection)

    for i in range(stack.shape[0]):
        for j in range(stack.shape[1]):
            if same_src_trg and i<=j: continue
            CCG=stack[i, j, :]
            pks=get_ccg_sig(CCG, cbin, cwin, p_th, n_consec_bins, sgn, fract_baseline, W_sd, test, ret_features=ret_features, only_max=only_max)
            if np.any(pks):
                sg=sign(pks[0][2]) if ret_features else sign(pks[0][1][0])
                print('Significant {}: {}->{}'.format(ptdic[sg], *ustack[i, j, :]))
                sigustack.append(ustack[i, j, :])
                if ret_features:
                    for p in pks:
                        features=features.append(dict(zip(features.columns,np.append(ustack[i, j, :], p))), ignore_index=True)

    sigustack=npa(sigustack)
    if np.any(sigustack):
        sigstack, sigustack = ccg_stack(dp, sigustack[:,0], sigustack[:,1], cbin, cwin, normalize='Counts', all_to_all=False, name=signame, again=True,
                                        subset_selection=subset_selection)
    else:
        bins=get_bins(cwin, cbin)
        sigstack, sigustack = npa(zeros=(0, len(bins))), sigustack

    if ret_features:
        if name is not None:
            features.to_csv(feat_path, index=False)
        return sigstack, sigustack, features
    return sigstack, sigustack

def gen_sfc(dp, corr_type='connections', metric='amp_z', cbin=0.5, cwin=100,
            p_th=0.02, n_consec_bins=3, fract_baseline=4./5, W_sd=10, test='Poisson_Stark',
            again=False, againCCG=False, drop_seq=['sign', 'time', 'max_amplitude'],
            pre_chanrange=None, post_chanrange=None, units=None, name=None, use_template_for_peakchan=False,
            subset_selection='all'):
    '''
    Function generating a functional correlation dataframe sfc (Nsig x 2+8 features) and matrix sfcm (Nunits x Nunits)
    from a sorted Kilosort output at 'dp' containing 'N' good units
    with cdf(i,j) a list of ('l_ms', 'r_ms', 'amp_z', 't_ms', 'n_triplets', 'n_bincrossing', 'bin_heights', 'entropy') tuples

    Parameters:
        - dp: string, datapath to manually curated kilosort output
        - corr_type: string in ['main', 'synchrony', 'excitation', 'inhibition']
            - main: among all modulations, take the biggest one.
                    Positive mods will be plotted in the bottomleft corner, negatives ones in the other.
            - synchrony: among all positive modulations, take the one between -1 and 1ms
                         Mods will be plotted symmetrically in both corners.
            - excitations: among all positive modulations, take the one between 1 and 2.5ms
                          Mods a->b will be plotted in the upper right corner, b->a in the other, if chan(a)>chan(b)
            - inhibitions: among all negative modulations, take the one between 1 and 2.5ms
                          Mods a->b will be plotted in the upper right corner, b->a in the other, if chan(a)>chan(b)
            - connections: among all modulations, take the one between 1 and 2.5ms
                          Inhibitions will be plotted in the upper right corner, excitations in the other.
            - cs_pause: complex spike driven pauses in simple spike
                        = inhibitions between 0.5 and 15 ms, lasting at least 10ms, centered after 4ms
        - metric: string, feature used to fill the sfc matrix | Default: amp_z
        - again: bool, whether to reassess significance of ccg stack rather than loading from memory if already computed in the past.
        - cbin: float, correlogram bin size | Default 0.5
        - cwin: float, correlogram window size | Default 100
        - p_th: float, significance threshold in p value | Default 0.02
        - n_consec_bins: int, number of bins beyons significance threshold required | Default 3
        - fract_baseline: float, fraction of the CCG used to compute the Ho mean and std if test='Normal_Kopelowitz'. | Default 4./5
        - W_sd: float, size of the hollow gaussian window used to compute the correlogram predictor if test='Poisson_Stark' in ms | Default 10
        - test: 'Normal_Kopelowitz' or 'Poisson_Stark', test to use to assess significance | Default Poisson_Stark
        - againCCG: bool, whether to recompute ccg stack rather than loading from memory if already computed in the past.
        - drop_seq: list of str, sequence in which to filter connections
        - pre_range: range of channels to which presynaptic units must belong
        - post_range: range of channels to which postsynaptic units must belong
        - units: list/array, units to consider to test correlations | Default: None (i.e. use all the good units)
        - name: string, name of the all-to-all ccg_stack corresponding to the above-provided units
                MANDATORY if you provide a list of units. | Default: None
        - cross_cont_proof: bool, ignore CCGs which look like a best guess of cross-contamination | Default: False
                            (i.e. a big trough centered around 0 or 2 big symmetrical troughs)

    Returns:
        - sfc: Pandas dataframe of NsignificantUnits x
               ['uSrc'', uTrg', 'l_ms', 'r_ms', 'amp_z', 't_ms', 'n_triplets', 'n_bincrossing', 'bin_heights', 'entropy']
               (see __doc__ of get_ccg_sig() for features description)
        - sfcm: np array, Nunits x Nunit with 0 if no significant correlation and metric if significant correlation.
    '''
    assert corr_type in ['all', 'main', 'synchrony', 'excitations', 'inhibitions', 'connections', 'cs_pause']
    # filter for main modulation irrespectively of sign
    sgn=0
    only_max=False
    if corr_type=='all':
        tfilt=[]
        sfilt=[]
    elif corr_type=='main':
        tfilt=[]
        sfilt=[]
        only_max=True
    elif corr_type=='synchrony':
        tfilt=[[-1,1]]
        sfilt=1
    elif corr_type=='excitations':
        tfilt=[[-2.5,-1],[1,2.5]]
        sfilt=1
    elif corr_type=='inhibitions':
        tfilt=[[-2.5,-1],[1,2.5]]
        sfilt=-1
    elif corr_type=='connections':
        tfilt=[[-2.5,-1],[1,2.5]]
        sfilt=[]
    elif corr_type=='cs_pause':
        tfilt=[[-15,-4],[4,15]] # not time of extremum but time of pause center
        sfilt=-1
        n_consec_bins=int(5//cbin)
    # Get depth-sorted units and sig ccg stack
    if units is not None:
        assert np.all(np.isin(units, get_units(dp))), 'Some of the provided units are not found in this dataset.'
        assert name is not None, 'You MUST provide a custom name for the provided list of units to ensure that your results can be saved.'
        peakChs = get_depthSort_peakChans(dp, units=units, use_template=use_template_for_peakchan)
        gu = peakChs[:,0]
    if name is not None:
        if units is None:
            print('You provided a name without any units - this will only work\
            if this name has been used in the past to generate a ccg_stack, with units provided.')
            gu=[]
    else:
        name='good-all_to_all'
        peakChs = get_depthSort_peakChans(dp, quality='good', use_template=use_template_for_peakchan)
        gu = peakChs[:,0]

    sigstack, sigustack, sfc = ccg_sig_stack(dp, gu, gu, cbin, cwin, name,
                  p_th, n_consec_bins, sgn, fract_baseline, W_sd, test, again, againCCG, ret_features=True, only_max=only_max,
                  subset_selection=subset_selection)

    # If filtering of connections wishes to be done at a later stage, simply return
    if corr_type=='all': return sfc, np.zeros((len(gu),len(gu))), peakChs

    # Else, proceed to filtering of connection types

    # Get rid of false positive connections due to cross-contamination

    if corr_type!='main': # then only_max is always False
        # Filter out based on sign
        def drop_sign(sfc, sfilt, corr_type):
            s=sfc['amp_z'].values
            s_mask=np.zeros((sfc.shape[0])).astype('bool')
            if np.any(sfilt):
                s_mask=(sign(s)!=sfilt)
            sfc.drop(index=sfc.index[np.isin(sfc.index, sfc.index[s_mask])], inplace=True)
            sfc.reset_index(inplace=True, drop=True)
            return sfc

        # Filter out based on time
        def drop_time(sfc, tfilt, corr_type):
            # for complex spike pauses: use time of trough center, not minimum
            t=(sfc.l_ms+(sfc.r_ms-sfc.l_ms)/2).values if corr_type=='cs_pause' else sfc['t_ms'].values
            t_mask=np.zeros((sfc.shape[0])).astype('bool')
            if np.any(tfilt):
                for tm in tfilt:
                    t_mask=t_mask|(t>=tm[0])&(t<=tm[1])
                t_mask=~t_mask
            sfc.drop(index=sfc.index[np.isin(sfc.index, sfc.index[t_mask])], inplace=True)
            sfc.reset_index(inplace=True, drop=True)
            return sfc

        # Filter out based on max amplitude
        def drop_amp(sfc, afilt, corr_type):
            z_mask=np.zeros((sfc.shape[0])).astype('bool')
            dgp=sfc.groupby(['uSrc','uTrg'])
            duplicates=npa(list(dgp.indices.values()))[(dgp.size()>1).values]
            for d in duplicates:
                zz=sfc.loc[d, 'amp_z'].abs()
                largest=zz.max()
                z_mask=z_mask|np.isin(sfc.index,d[zz!=largest])
            sfc.drop(index=sfc.index[np.isin(sfc.index, sfc.index[z_mask])], inplace=True)
            sfc.reset_index(inplace=True, drop=True)
            return sfc

        # Drop stuff
        drop_dic={'sign':drop_sign,'time':drop_time,'max_amplitude':drop_amp,
                  'signfilt':sfilt, 'timefilt':tfilt, 'max_amplitudefilt':None}
        for drop in drop_seq:
            sfc=drop_dic[drop](sfc, drop_dic[drop+'filt'], corr_type)

    if (pre_chanrange is not None)|(post_chanrange is not None):
        peakChs = get_depthSort_peakChans(dp, use_template=use_template_for_peakchan)
        if pre_chanrange is not None:
            pre_units=sfc.uSrc[sfc.t_ms>=0].append(sfc.uTrg[sfc.t_ms<0]).sort_index().values
            peak_m=(peakChs[:,1]>pre_chanrange[0])&(peakChs[:,1]<pre_chanrange[1])
            range_m=np.isin(pre_units,peakChs[peak_m,0])
            sfc.drop(index=sfc.index[~range_m], inplace=True)
            sfc.reset_index(inplace=True, drop=True)
        if post_chanrange is not None:
            post_units=sfc.uSrc[sfc.t_ms<0].append(sfc.uTrg[sfc.t_ms>=0]).sort_index().values
            peak_m=(peakChs[:,1]>post_chanrange[0])&(peakChs[:,1]<post_chanrange[1])
            range_m=np.isin(post_units,peakChs[peak_m,0])
            sfc.drop(index=sfc.index[~range_m], inplace=True)
            sfc.reset_index(inplace=True, drop=True)

    sfcm = np.zeros((len(gu),len(gu)))
    for i in sfc.index:
        u1,u2=sfc.loc[i,'uSrc':'uTrg']
        ui1,ui2=np.nonzero(gu==u1)[0][0], np.nonzero(gu==u2)[0][0] # ORDER OF gu MATTERS
        v=sfc.loc[i, metric]
        # If showing all main modulations or all connections,
        # plotting inhibitions top right corner and excitations bottom left corner
        if corr_type in ['main', 'connections']:
            i1,i2 = (ui1,ui2) if ui1<ui2 else (ui2,ui1)
            if sign(v)==-1:
                sfcm[i1,i2]=v
            else:
                sfcm[i2,i1]=v
        # If plotting synchrony, which is symmetrical,
        # plotting symmetrically
        elif corr_type=='synchrony':
            sfcm[ui1,ui2]=sfcm[ui2,ui1]=v

        # If plotting inhibitions xor excitations,
        # plotting u1-> u2 in top right corner
        # u2 -> u1 in bottom left corner
        elif corr_type in ['excitations', 'inhibitions']:
            t=sfc.loc[i, 't_ms']
            tmask1=(t>=tfilt[1][0])&(t<=tfilt[1][1])
            tmask2=(t>=tfilt[0][0])&(t<=tfilt[0][1])
            if np.any(tmask1):
                sfcm[ui1,ui2]=v[tmask1]
            elif np.any(tmask2):
                sfcm[ui2,ui1]=v[tmask2]

    if not np.any(sigstack):
        return sfc, sfcm, peakChs, sigstack, sigustack

    filter_m = np.isin(sigustack[:,0], sfc.loc[:,'uSrc'].values)&np.isin(sigustack[:,1], sfc.loc[:,'uTrg'].values)
    sigstack=sigstack[filter_m]
    sigustack=sigustack[filter_m]

    return sfc, sfcm, peakChs, sigstack, sigustack


#%% Work in progress

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
    dprm = Path(dp,'routinesMemory')
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(Path(dprm,'PSDxy{}_{}.npy'.format(sortedU, str(bin_size).replace('.','_')))):
        if prnt: print("File PSDxy{}_{}.npy found in routines memory.".format(str(sortedU).replace(" ", ""), str(bin_size).replace('.','_')))
        Pxy = np.load(Path(dprm,'PSDxy{}_{}.npy'.format(sortedU, str(bin_size).replace('.','_'))))
        Pxy = Pxy.astype(np.float64)

    # if not, compute it
    else:
        if prnt: print("File ccg_{}_{}.npy not found in routines memory.".format(str(sortedU).replace(" ", ""), str(bin_size).replace('.','_')))
        Pxy = np.empty((len(sortedU), len(sortedU), int(nperseg/2)+1), dtype=np.float64)
        for i, u1 in enumerate(sortedU):
            trnb1 = trnb(dp, u1, bin_size)
            for j, u2 in enumerate(sortedU):
                trnb2 = trnb(dp, u2, bin_size)
                (f, Pxy[i, j, :]) = sgnl.csd(trnb1, trnb2, fs=fs, window=window, nperseg=nperseg, scaling=scaling)
        Pxy = Pxy.astype(np.float64)
        # Save it
        if sav:
            np.save(Path(dprm,'PSDxy{}_{}.npy'.format(str(sortedU).replace(" ", ""), str(bin_size).replace('.','_'))), Pxy)

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


#%% Circular imports
from npyx.plot import plot_pval_borders

#%% Archived


# def crosscorrelate_maxime(dp, bin_size, win_size, U, trainBin=10, fs=30000, subset_selection='all', prnt=True, own_trains={}):
#     '''Returns the crosscorrelation function of two spike trains.
#     - dp: (string): DataPath to the Neuropixels dataset.
#     - win_size (float): window size, in milliseconds
#     - bin_size (float): bin size, in milliseconds
#     - U (list of integers): list of units indexes. If string, measures it for the whole dataset.
#     - trainBin: binsize used to binarize trains before computing corrcoeff, in ms.
#     - fs: sampling rate (Hertz). Default 30000.
#     - symmetrize (bool): symmetrize the semi correlograms. Default=True.
#     - own_trains: dictionnary of trains, to calculate the CCG of an arbitrary list of trains in SAMPLES for fs=30kHz.'''

#     #### Troubleshooting
#     assert fs > 0.
#     bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
#     binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
#     assert binsize >= 1 # Cannot be smaller than a sample time

#     win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
#     winsize_bins = 2 * int(.5 * win_size *1./ bin_size) + 1 # Both in millisecond
#     assert winsize_bins >= 1
#     assert winsize_bins % 2 == 1

#     #### Get clusters and times
#     if own_trains!={}:
#         phy_ss, spike_clusters = make_phy_like_spikeClustersTimes(dp, U, subset_selection=subset_selection, prnt=prnt, dic=own_trains)
#         units = _unique(spike_clusters)
#         n_units = len(units)

#     else:
#         if type(U)==str:
#             # All the CCGs of a Neuropixels dataset
#             spike_clusters = np.load(dp+"/spike_clusters.npy")
#             units = _unique(spike_clusters)
#             n_units = len(units)
#             phy_ss = np.load(dp+'/spike_times.npy')

#         # Between n_units provided units
#         else:
#             if type(U)!=list:
#                 U=list(U)

#             phy_ss, spike_clusters = make_phy_like_spikeClustersTimes(dp, U, subset_selection=subset_selection, prnt=prnt, dic={})
#             units = _unique(spike_clusters)
#             n_units = len(units)

#     #### Compute crosscorrelograms
#     rec_len=phy_ss[-1]
#     correlograms = np.zeros((n_units, n_units, winsize_bins // 2 + 1), dtype=np.float32) # Only computes semi correlograms (//2)
#     for i1, u1 in enumerate(units):
#         t1 = phy_ss[spike_clusters==u1] # samples
#         t1b=binarize(t1, trainBin, fs, rec_len=rec_len, constrainBin=False)
#         for i2, u2 in enumerate(units):
#             t2 = phy_ss[spike_clusters==u2] # samples
#             for ilag, lag in enumerate(np.arange(0, winsize_bins // 2 + 1)):
#                 t2_lag = t2+lag*binsize # samples
#                 t2lb=binarize(t2_lag, trainBin, fs, rec_len=rec_len, constrainBin=False)
#                 c = pearson_corr(npa([t1b, t2lb]))
#                 print(u1, u2, lag, c)
#                 correlograms[i1, i2, ilag]=c
#             # set ACG 0 to 0
#             if i1==i2:
#                 correlograms[i1, i2, 0]=0
#     del t1, t1b, t2, t2_lag, t2lb

#     # Symmetrize
#     n_units, _, n_bins = correlograms.shape
#     assert n_units == _
#     correlograms[..., 0] = np.maximum(correlograms[..., 0],
#                                       correlograms[..., 0].T)
#     sym = correlograms[..., 1:][..., ::-1]
#     sym = np.transpose(sym, (1, 0, 2))
#     correlograms = np.dstack((sym, correlograms))

#     return correlograms


# def crosscorrelate_maxime1(dp, U, bin_size, win_size, fs=30000, normalize=False, prnt=True):
#     '''Returns the crosscorrelation function of two spike trains.
#     Second one 'triggered' by the first one.
#     - dp: (string): DataPath to the Neuropixels dataset.
#     - U (list of integers): list of units indexes.
#     - win_size (float): window size, in milliseconds
#     - bin_size (float): bin size, in milliseconds
#     - fs: sampling rate (Hertz). Default 30000.
#     - symmetrize (bool): symmetrize the semi correlograms. Default=True.
#     - normalize: normalize the correlograms. Default=False.'''

#     # Troubleshooting
#     assert fs > 0.
#     bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
#     binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
#     assert binsize >= 1 # Cannot be smaller than a sample time

#     win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
#     win_size_bins = 2 * int(win_size*0.5/bin_size) + 1 # Both in millisecond
#     assert win_size_bins >= 1
#     assert win_size_bins % 2 == 1

#     correlograms=np.zeros((len(U), len(U), win_size_bins))
#     if (win_size*1./bin_size)%2==0: # even
#         binsedges=np.arange(-win_size*1./2-bin_size*1./2, win_size*1./2+bin_size*3./2, bin_size) # add 1 + two half bins to keep even centered on 0
#     elif win_size*1./bin_size%2==1: # odd
#         binsedges=np.arange(-win_size*1./2, win_size*1./2+bin_size, bin_size) # add one bin to make even centered on 0

#     for i1, u1 in enumerate(U):
#         t1=trn(dp, u1, prnt=False)*1./30 # ms
#         for i2, u2 in enumerate(U):
#             if i2>=i1:
#                 t2 = trn(dp, u2, prnt=False)*1./30 if u2!=u1 else t1 # ms
#                 dt=np.array([])
#                 if prnt: print('CCG {}x{}'.format(u1, u2))
#                 for si, spk in enumerate(t1):
#                     #end = '\r' if si<len(t1)-1 else ''
#                     #print('{}%...'.format(int(100*(si+1)*1./len(t1))), end=end)
#                     d = t2-spk
#                     dt = np.append(dt, d[(d>=-win_size*1./2)&(d<=win_size*1./2)])
#             else:
#                 pass

#             hist=np.histogram(dt, binsedges)[0]
#             if i1==i2:
#                 hist[int(.5*(len(hist)-1))]=0

#             correlograms[i1, i2, :]=hist*1./(0.001*bin_size*np.sqrt(len(t1)*len(t2)))

#     # Symmetrize
#     for i1, u1 in enumerate(U):
#         for i2, u2 in enumerate(U):
#             if i1!=i2:
#                 correlograms[i2, i1, :]=np.array([hist[-v+1] for v in range(len(hist))])

#     return correlograms

# def crosscorrelate_maxime2(dp, U, bin_size, win_size, trn_binsize=0.1, fs=30000, normalize=False, prnt=True):
#     '''
#     STILL NOT FUNCTIONAL
#     HORIZONTAL SCALING PROBLEM
#     VERTICAL SCALING PROBLEM (likely use mfr1 and mfr2 and T)
#     SYMMETRIZE NOT OPTIMIZED
#     Returns the crosscorrelation function of two spike trains.
#     Second one 'triggered' by the first one.
#     - dp: (string): DataPath to the Neuropixels dataset.
#     - U (list of integers): list of units indexes.
#     - win_size (float): window size, in milliseconds
#     - bin_size (float): bin size, in milliseconds
#     - fs: sampling rate (Hertz). Default 30000.
#     - symmetrize (bool): symmetrize the semi correlograms. Default=True.
#     - normalize: normalize the correlograms. Default=False.'''

#     # Troubleshooting
#     assert fs > 0.
#     bin_size = np.clip(bin_size, 1000*1./fs, 1e8)  # in milliseconds
#     binsize = int(np.ceil(fs * bin_size*1./1000))  # in samples
#     assert binsize >= 1 # Cannot be smaller than a sample time

#     win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
#     win_size_bins = 2 * int(win_size*0.5/bin_size) + 1 # Both in millisecond
#     assert win_size_bins >= 1
#     assert win_size_bins % 2 == 1

#     correlograms=np.zeros((len(U), len(U), win_size_bins))
#     if (win_size*1./bin_size)%2==0: # even
#         #binsedges=np.arange(-win_size*1./2-bin_size*1./2, win_size*1./2+bin_size*3./2, bin_size) # add 1 + two half bins to keep even centered on 0
#         bins=np.arange(-win_size*1./2, win_size*1./2+bin_size, bin_size)
#     elif win_size*1./bin_size%2==1: # odd
#         #binsedges=np.arange(-win_size*1./2, win_size*1./2+bin_size, bin_size) # add one bin to make even centered on 0
#         bins=np.arange(-win_size*1./2+bin_size*1./2, win_size*1./2+bin_size*1./2, bin_size)

#     try:
#         assert (bin_size*1e6)%(trn_binsize*1e6)==0
#     except:
#         if prnt: print('bin_size ({}) is not a multiple of trn_bins ({}), therefore shifts are not integers! Abort.'.format(bin_size, trn_binsize))
#         return

#     for i1, u1 in enumerate(U):
#         tb1=trnb(dp, u1, trn_binsize, prnt=False) # binarized at the same sampling rate as the correlogram
#         #mfr1=1000./np.mean(isi(dp, u1, prnt=False)) # in s-1
#         for i2, u2 in enumerate(U):
#             if prnt: print('CCG {}x{}'.foamrt(u1, u2))
#             if i2>=i1:
#                 tb2 = trnb(dp, u2, trn_binsize, prnt=False) if i2!=i1 else tb1 # binarized at the same sampling rate as the correlogram
#                 #mfr2=1000./np.mean(isi(dp, u2, prnt=False)) # in s-1
#                 corr=np.zeros((len(bins)))
#                 shifts=np.asarray(np.round(bins*1./bin_size, 2)*(bin_size*1./trn_binsize), dtype=np.int64)
#                 #T=len(tb1) # T
#                 for si, shift in enumerate(shifts): # bins are centered on 0 and include 0
#                     #end = '\r' if si<len(shifts)-1 else ''
#                     #print('{}%...'.format(int(100*(si+1)*1./len(shifts))), end=end)
#                     # C(shift) = (1./(T*bin_size*mfr1*mfr2))*S[t:0->T][(n1(t)-mfr1) * (n2(t+shift)-mfr2)]
#                     tb1c = tb1#-mfr1 # n1(t)-mfr1
#                     if shift>=0:
#                         tb2c_shifted=np.append(tb2[shift:], np.zeros((shift)))#-mfr2 # n2(t+shift)-mfr2 (0 padding on the left)
#                     elif shift<0:
#                         tb2c_shifted=np.append(np.zeros((-shift)), tb2[:+shift])#-mfr2 # n2(t+shift)-mfr2 (0 padding on the right)

#                     C_shift=np.sum(tb1c*tb2c_shifted) # C(shift) (element wise multiplication)
#                     corr[si]=C_shift#*1./(T*bin_size*mfr1*mfr2) # Normalize by recording length in ms * mfr1/2 in ms-1

#             else:
#                 pass

#             if i1==i2:
#                 corr[int(.5*(len(corr)-1))]=0
#             correlograms[i1, i2, :]=corr

#     # Symmetrize
#     for i1, u1 in enumerate(U):
#         for i2, u2 in enumerate(U):
#             corr=correlograms[i1, i2, :]
#             if i1!=i2:
#                 correlograms[i2, i1, :]=np.array([corr[-v+1] for v in range(len(corr))])*1./(0.001*bin_size*np.sqrt(len(trn(dp, u1, prnt=False))*len(trn(dp, u2, prnt=False))))

#     return correlograms
from npyx.spk_wvf import get_depthSort_peakChans
