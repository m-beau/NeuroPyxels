# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

import os
import os.path as op; opj=op.join
import psutil
from pathlib import Path
from tqdm.notebook import tqdm

from collections.abc import Iterable

import multiprocessing
num_cores = multiprocessing.cpu_count()

import numpy as np
from math import ceil

import matplotlib.pyplot as plt

from npyx.utils import npa, split, xcorr_1d_loop
from npyx.inout import read_metadata, get_binary_file_path, chan_map, whitening, bandpass_filter, apply_filter, med_substract
from npyx.gl import get_units, get_npyx_memory

def wvf(dp, u=None, n_waveforms=100, t_waveforms=82, selection='regular', periods='all',
        spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
        save=True, verbose=False, again=False,
        whiten=False, med_sub=False, hpfilt=False, hpfiltf=300,
        nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=True):
    '''
    ********
    routine from rtn.npyx.spk_wvf
    Extracts a sample of waveforms from the raw data file.
    ********

    Parameters:
        - dp:                 str or PosixPath, path to kilosorted dataset.
        - u:                  int, unit index.
        - n_waveforms:        int, number of waveform to return, selected according to the periods parameter | Default 100
        - t_waveforms:        int, temporal span of waveforms | Default 82 (about 3ms)
        - selection:          str, way to select subset of n_waveforms spikes to return the waveforms of.
                              Either 'regular' (homogeneous selection or in batches) or 'random'.
        - periods:            recording periods to sample waveforms from. Either 'all' (default)
                              or [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
        - spike_ids:          list/array, relative indices of spikes in the whole recording.
                                          Takes precedence over every other parameter: if provided, u, n_waveforms and periods will be ignored.
        - wvf_batch_size:     int, if >1 and 'regular' selection, selects ids as batches of spikes. | Default 10
        - save:               bool, whether to save to routine memory. | Default True
        - verbose:            bool, whether to print informaiton. | Default False
        - again:              bool, whether to recompute waveforms even if ofund in routines memory. | Default False
        - ignore_nwvf:        bool, whether to ignore n_waveforms parameter when a list of times is provided as periods,
                                    to return all the spikes in the window instead. | Default True
        - whiten:             bool, whether to whiten across channels.
                                    Globally by default, using the nRangeWhiten closest channels if nRangeWhiten is provided. | Default False
        - med_sub:            bool, whether to median-subtract across channels.
                                    Globally by default, using the nRangeMedSub closest channels if nRangeWhiten is provided. | Default False
        - hpfilt:             bool, whether to high-pass filter with a butterworth filter (order 3) of cutoff frequency hpfiltf. | Default False
        - hpfiltf:            int, high-pass filter cutoff frequency | Default 300
        - nRangeWhiten        int, number of channels to use to compute the local median. | Default None
        - nRangeMedSub:       int, number of channels to use to compute the local median. | Default None
        - ignore_ks_chanfilt: bool, whether to ignore kilosort channel filtering
                                    (if False, output shape will always be n_waveforms x t_waveforms x 384) | Default False
    Returns:
        waveforms:            numpy array of shape (n_waveforms x t_waveforms x n_channels)
                                    where n_channels is defined by the channel map if ignore_ks_chanfilt is False.

    '''
    dp = Path(dp)

    if spike_ids is not None:
        if u is not None and verbose: print('WARNING you provided both u and spike_ids! u is ignored.')
        if n_waveforms !=100 and verbose: print('WARNING you provided both n_waveforms and spike_ids! n_waveforms is ignored.')
        if not isinstance(periods,str) and verbose: print('WARNING you provided both periods and spike_ids! periods is ignored.')
        u=np.unique(np.load(Path(dp)/'spike_clusters.npy')[spike_ids])
        assert len(u)==1, 'WARNING the spike ids that you provided seem to belong to different units!! Double check!'
        u=u[0]
    dp, u = get_source_dp_u(dp, u)

    dpnm = get_npyx_memory(dp)

    if isinstance(periods, str): assert periods=='all', "WARNING periods should either be 'all' or [[t1,t2],[t3,t4]...]."
    per_str = str(periods)[0:50].replace(' ', '').replace('\n','')
    fn=f"wvf{u}_{n_waveforms}-{t_waveforms}_{per_str}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}-{ignore_ks_chanfilt}.npy"
    if os.path.exists(Path(dpnm,fn)) and (not again) and (spike_ids is None):
        if verbose: print("File {} found in routines memory.".format(fn))
        return np.load(Path(dpnm,fn))

    waveforms = get_waveforms(dp, u, n_waveforms, t_waveforms, selection, periods, spike_ids, wvf_batch_size, ignore_nwvf,
                 whiten, med_sub, hpfilt, hpfiltf, nRangeWhiten, nRangeMedSub, ignore_ks_chanfilt, verbose)
    # Save it
    if (save and (spike_ids is None)):
        np.save(Path(dpnm,fn), waveforms)

    return waveforms

# def get_w(traces, slc, _n_samples_extract):
#     # Get slice
#     extract = traces[slc].astype(np.float32)
#     # Center channels individually
#     extract = extract-np.median(extract, axis=0)
#     # Pad the extracted chunk if at recording limit.
#     if slc.start <= 0: extract = _pad(extract, _n_samples_extract, 'left')
#     elif slc.stop >= traces.shape[0] - 1: extract = _pad(extract, _n_samples_extract, 'right')
#     # Add this waveform, all good!
#     return extract.T

def get_waveforms(dp, u, n_waveforms=100, t_waveforms=82, selection='regular', periods='all',
                  spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
                  whiten=0, med_sub=0, hpfilt=0, hpfiltf=300,
                  nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=0, verbose=False):
    f"{wvf.__doc__}"

    # Extract and process metadata
    dp = Path(dp)
    meta = read_metadata(dp)
    dat_path = get_binary_file_path(dp, 'ap')

    dp_source = get_source_dp_u(dp, u)[0]
    meta=read_metadata(dp_source)
    dtype=np.dtype(meta['highpass']['datatype'])
    n_channels_dat=meta['highpass']['n_channels_binaryfile']
    n_channels_rec = n_channels_dat-1 if meta['acquisition_software']=='SpikeGLX' else n_channels_dat
    sample_rate=meta['highpass']['sampling_rate']
    item_size = dtype.itemsize
    fileSizeBytes=meta['highpass']['binary_byte_size']
    if meta['acquisition_software']=='SpikeGLX':
        if meta['highpass']['fileSizeBytes'] != fileSizeBytes:
            print((f"\033[91;1mMismatch between ap.meta and ap.bin file size"
            "(assumed encoding is {str(dtype)} and Nchannels is {n_channels_dat})!! "
            f"Probably wrong meta file - just edit fileSizeBytes in the .ap.meta file at {dp} "
            f"(replace {int(meta['highpass']['fileSizeBytes'])} with {fileSizeBytes}) "
            "and be aware that something went wrong in your data management...\033[0m"))

    # Select subset of spikes
    spike_samples = np.load(Path(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    if spike_ids is None:
        spike_ids_subset=get_ids_subset(dp, u, n_waveforms, wvf_batch_size, selection, periods, ignore_nwvf, verbose)
    else:
        assert isinstance(spike_ids, Iterable), "WARNING spike_ids must be a list/array of ids!"
        spike_ids_subset=np.array(spike_ids)
    n_spikes = len(spike_ids_subset)

    # Get waveforms times in bytes
    # and check that, for this waveform width,
    # they no not go beyond file limits
    waveforms_t = spike_samples[spike_ids_subset].astype(np.int64)
    waveforms_t1 = (waveforms_t-t_waveforms//2)*n_channels_dat*item_size
    waveforms_t2 = (waveforms_t+t_waveforms//2)*n_channels_dat*item_size
    wcheck_m=(0<=waveforms_t1)&(waveforms_t2<fileSizeBytes)
    if not np.all(wcheck_m):
        print(f"Invalid times: {waveforms_t[~wcheck_m]}")
        waveforms_t1 = waveforms_t1[wcheck_m]
        waveforms_t2 = waveforms_t2[wcheck_m]

    # Iterate over waveforms
    waveforms = np.zeros((n_spikes, t_waveforms, n_channels_rec), dtype=np.float32)
    if verbose: print(f'Loading waveforms of unit {u} ({n_spikes})...')
    with open(dat_path, "rb") as f:
        for i,t1 in enumerate(waveforms_t1):
            if n_spikes>10:
                if i%(n_spikes//10)==0 and verbose: print(f'{round((i/n_spikes)*100)}%...', end=' ')
            f.seek(t1, 0) # 0 for absolute file positioning
            wave=f.read(n_channels_dat*t_waveforms*item_size)
            wave=np.frombuffer(wave, dtype=dtype).reshape((t_waveforms,n_channels_dat))
            wave = wave-np.median(wave, axis = 0)[np.newaxis,:] # center the waveforms on 0
            # get rid of sync channel
            waveforms[i,:,:] = wave[:,:-1] if meta['acquisition_software']=='SpikeGLX' else wave
    if verbose: print('\n')

    # Preprocess waveforms
    if hpfilt|med_sub|whiten:
        waveforms=waveforms.reshape((n_spikes*t_waveforms, n_channels_rec))
        if hpfilt:
            waveforms=apply_filter(waveforms, bandpass_filter(rate=sample_rate, low=None, high=hpfiltf, order=3), axis=0)
        if med_sub:
            waveforms=med_substract(waveforms, axis=1, nRange=nRangeMedSub)
        if whiten:
            waveforms=whitening(waveforms.T, nRange=nRangeWhiten).T # whitens across channels so gotta transpose
        waveforms=waveforms.reshape((n_spikes,t_waveforms, n_channels_rec))

    # Filter channels ignored by kilosort if necesssary
    if not ignore_ks_chanfilt:
        channel_ids_ks = np.load(Path(dp, 'channel_map.npy'), mmap_mode='r').squeeze()
        channel_ids_ks=channel_ids_ks[channel_ids_ks!=384]
        waveforms=waveforms[:,:,channel_ids_ks] # only AFTER processing, filter out channels

    # Correct voltage scaling
    waveforms*=meta['bit_uV_conv_factor']

    return  waveforms.astype(np.float32)

def wvf_dsmatch(dp, u, n_waveforms=100, t_waveforms=82, periods='all',
                wvf_batch_size=10, ignore_nwvf=True, med_sub = False, spike_ids = None,
                save=True, verbose=False, again=False,
                whiten=False,  hpfilt=False, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None,
                n_waves_used_for_matching = 50000, peakchan_allowed_range=10,
                use_average_peakchan = False, max_allowed_amplitude = 1800, max_allowed_shift=3,
                n_waves_to_average=5000, plot_debug=False, do_shift_match=True, n_waveforms_per_batch=10):
    """
    ********
    Extract the drift and shift matched mean waveforms of the specified unit.
    Drift and shift matching consists of two steps:

    First: load all waveforms, average them 10 by 10 = 'spike batches'
    (else, they would be too noisy to work with. Assumption: 10 consecutive waveforms have the same 'drift state')

    Drift matching:
        - Z-drift-matching: sub-select spike batches peaking on same peak channel (modal channel)
        - XY-drift-marching: sub-select n_waves_to_average/10 spikes batches
          with the highest amplitude on this peak channel
          (up to 99th percentile, not highest amplitude)
        - Exclude batches with amplitude higher than max_allowed_amplitude uV (gets rid of potential artefacts)

    Shift matching:
        - Define a template from the 5 drift-matched batches with the highest amplitude
        - Compute crosscorrelation between each batch and template
        - Re-align each batch to the template accordingly to peaking crosscorrelation
        - Exclude batches which were required to shift by more than +/-max_allowed_shift samples
          (naturally gets rid of
          - completely off batches - there is probably something wrong with them
          - noisy batches - which do not match the template well enough for a match to be found around 0

    To diagnose issues: set **plot_debug=True** (and again=true of course), to plot the distributions of peak channel, amplitudes and shifts

    Currently only supports passing a single unit as input, hence
    prints error message if 'spike_ids = single_slice' if passed.
    ********


     Parameters:
        - dp:                 str or PosixPath, path to kilosorted dataset.
        - u:                  int, unit index.
        - n_waveforms:        int, number of waveform to return, selected according to the periods parameter | Default 100
        - t_waveforms:        int, temporal span of waveforms | Default 82 (about 3ms)
        - periods:   str/list of tuples, either 'regular' (homogeneous selection or in batches), 'random',
                                                  or a list of time chunks [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
        - spike_ids:          list/array, relative indices of spikes in the whole recording.
                                          If provided, u, n_waveforms and periods will be ignored.
        - wvf_batch_size:     int, if >1 and 'regular' selection, selects ids as batches of spikes. | Default 10
        - save: bool,         whether to save to routine memory. | Default True
        - verbose: bool,         whether to print information. | Default False
        - again: bool,        whether to recompute waveforms even if found in routines memory. | Default False
        - ignore_nwvf:        bool, whether to ignore n_waveforms parameter when a list of times is provided as periods,
                                    to return all the spikes in the window instead. | Default True
        - whiten:             bool, whether to whiten across channels.
                                    Globally by default, using the nRangeWhiten closest channels if nRangeWhiten is provided. | Default False
        - med_sub:            bool, whether to median-subtract across channels.
                                    Globally by default, using the nRangeMedSub closest channels if nRangeWhiten is provided. | Default False
        - hpfilt:             bool, whether to high-pass filter with a butterworth filter (order 3) of cutoff frequency hpfiltf. | Default False
        - hpfiltf:            int, high-pass filter cutoff frequency | Default 300
        - nRangeWhiten        int, number of channels to use to compute the local median. | Default None
        - nRangeMedSub:       int, number of channels to use to compute the local median. | Default None
        - ignore_ks_chanfilt: bool, whether to ignore kilosort channel filtering
                                    (if False, output shape will always be n_waveforms x t_waveforms x 384) | Default False
        - n_waves_used_for_matching:   int, how many spikes to subsample to perform matching (default 50000 waveforms)
        - peakchan_allowed_range: int (channel id), maximum allowed distance between original pek channel and ds-matched drift channel
        - use_average_peakchan: bool, if True simply use the channel with highest average amplitude across spikes as peak channel
                                       instead of using the channel where the most spikes peak on
        - max_allowed_amplitude: float, maximum amplitude in uV (peak to trough) that a spike average can have to be considered (above, must be artefactual)
        - max_allowed_shift: int (samples), maximum allowed temporal shift during shift-matching (see Shift-matching explanation above)
        - n_waves_to_average: int, maximum number of waveforms averaged together (5000 waveforms = 500 batches)
        - plot_debug: bool, whether to plot informative histograms displaying the distribution of peak channels (Z drift matching),
                      amplitudes on this peak channel (XY drift matching) and shifts (shift matching)
        - do_shift_match: bool, whether to perform shift matching
        - n_waveforms_per_batch: int, number of waveforms to use per batch for drift matching
                                 (in an ideal world 1, but too noisy - we assume that
                                  n_waveforms_per_batch consecutive waveforms have the same drift state)

    Returns:
        - peak_dsmatched_waveform: (n_samples,) array (t_waveforms samples) storing the peak channel waveform
        - dsmatched_waveform: (n_samples, n_channels) array storing the drift-shift-matched waveform across channels (384 for Neuropixels 1.0)
        - spike_ids: (n_spikes,) array of absolute ids (w/r all spikes in recording)
                     of spikes subset selected to compute the final drift-shift-matched waveform
        - peak_channel: (1,) array storing the channel used to select the subset of waveforms during drift matching (de facto, peak channel)
    """

    dp = Path(dp)

    if spike_ids is not None:
        raise ValueError('No support yet for passing multiple spike indices. Exiting.')


    dpnm = get_npyx_memory(dp)

    per_str = str(periods)[0:50].replace(' ', '').replace('\n','')
    fn=f"dsm_{u}_{n_waveforms}-{t_waveforms}_{per_str}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"
    fn_all=f"dsm_{u}_all_waves_{n_waveforms}-{t_waveforms}_{per_str}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"
    fn_spike_id=f"dsm_{u}_spike_id_{n_waveforms}-{t_waveforms}_{per_str}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"
    fn_peakchan=f"dsm_{u}_peakchan_{n_waveforms}-{t_waveforms}_{per_str}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"

    if Path(dpnm,fn).is_file() and (not again) and (spike_ids is None):
        if verbose: print(f"File {fn} found in routines memory.")
        drift_shift_matched_mean = np.load(Path(dpnm,fn_all))
        if plot_debug:
            w=wvf(dp, u, n_waveforms=n_waveforms, t_waveforms=t_waveforms)
            fig = quickplot_n_waves(np.mean(w, 0))
            fig = quickplot_n_waves(drift_shift_matched_mean, f'blue: 100 random waveforms\norange: dsmatched_waveforms (unit {u})', fig=fig)
        return np.load(Path(dpnm,fn)),drift_shift_matched_mean,np.load(Path(dpnm,fn_spike_id)), np.load(Path(dpnm,fn_peakchan))

    ## Extract spike ids so we can extract consecutive waveforms
    spike_ids_all = ids(dp, u, periods=periods)
    spike_ids_split_all = split(spike_ids_all, n_waveforms_per_batch, return_last = False).astype(np.int64)
    
    ## Subsample waveforms based on available RAM
    vmem=dict(psutil.virtual_memory()._asdict())
    available_RAM = vmem['total']-vmem['used']
    single_w_size = wvf(dp, None, t_waveforms=t_waveforms, spike_ids=[0]).nbytes
    max_n_waveforms = available_RAM//single_w_size-100 # -100 to be safe
    n_waves_used_for_matching = min(n_waves_used_for_matching, max_n_waveforms)
    
    # now subsample ids based on the RAM-safe n_waves_used_for_matching
    if n_waves_used_for_matching<len(spike_ids_all):
        n_batches_used_for_matching=n_waves_used_for_matching//n_waveforms_per_batch # floor division
        spike_ids_subsample=np.round(np.linspace(0,spike_ids_split_all.shape[0]-1,n_batches_used_for_matching)).astype(int)
        spike_ids_split = spike_ids_split_all[spike_ids_subsample]
    else:
        spike_ids_split=spike_ids_split_all
    spike_ids_split_indices = np.arange(0,spike_ids_split.shape[0],1)

    ## Extract the waveforms using the wvf function in blocks of 10 (n_waveforms_per_batch).
    # After waves have been extracted, put the index of the channel with the
    # max amplitude as well as the max amplitude into the peak_chan_split array
    spike_ids_split = spike_ids_split.flatten()
    raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = t_waveforms,
                    selection='regular', periods=periods, spike_ids=spike_ids_split,
                    wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
                    save=save , verbose = verbose,  again=True, whiten = whiten,
                    hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
                    nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True)
    spike_ids_split = spike_ids_split.reshape(-1,n_waveforms_per_batch)
    raw_waves = raw_waves.reshape(spike_ids_split.shape[0], n_waveforms_per_batch, t_waveforms, -1)
    mean_waves = np.mean(raw_waves, axis = 1)
    ## Find peak channel (and store amplitude) of every batch
    # only consider amplitudes on channels around original peak channel
    original_peak_chan = get_peak_chan(dp, u)
    c_left, c_right = max(0, original_peak_chan-peakchan_allowed_range), min(original_peak_chan+peakchan_allowed_range, mean_waves.shape[2])
    # calculate amplitudes ("peak-to-peak"), but ONLY using 2ms (-30,30) in the middle
    t1, t2 = max(0,mean_waves.shape[1]//2-30), min(mean_waves.shape[1]//2+30, mean_waves.shape[1])
    amplitudes = np.ptp(mean_waves[:,t1:t2,c_left:c_right], axis=1)
    batch_peak_channels = np.zeros(shape=(spike_ids_split_indices.shape[0], 3))
    batch_peak_channels[:,0] = spike_ids_split_indices # store batch indices (batch = averaged 10 spikes)
    batch_peak_channels[:,1] = c_left+np.argmax(amplitudes, axis = 1) # store peak channel of each batch
    batch_peak_channels[:,2] = np.max(amplitudes, axis = 1) # store peak channel amplitude

    # Filter out batches with too large amplitude (probably artefactual)
    batch_peak_channels = batch_peak_channels[batch_peak_channels[:,2] < max_allowed_amplitude]

    #### Z-drift matching ####
    # subselect batches with same peak channel
    if use_average_peakchan:
        peak_channel = int(original_peak_chan)
    else:
        # use mode of peak channel distribution across spikes
        chans, count = np.unique(batch_peak_channels[:,1], return_counts = True)
        peak_channel = int(chans[np.argmax(count)])

    if plot_debug:
        fig = hist_MB(batch_peak_channels[:,1], a=peak_channel-20, b=peak_channel+20, s=1,
        title=f'Z drift matching:\ndistribution of peak channel across spike batches\n({n_waveforms_per_batch} spikes/batch - mode: chan {peak_channel})')
        ylim = fig.get_axes()[0].get_ylim()
        fig.get_axes()[0].plot([peak_channel,peak_channel], ylim, color='red', ls='--')
        fig.get_axes()[0].set_ylim(ylim)

    batch_peak_channels = batch_peak_channels[batch_peak_channels[:,1] == peak_channel]

    #### X-Y-drift matching ####
    # subselect batches with similar amplitude (i.e. similar distance to probe)
    # and in particular, close to largest amplitude (close to probe, but not max to avoid artefacts)
    # aim for 500 spikes (50 batches)
    # should average enough, but still use a small subset of drift-matched spikes!
    n_driftmatched_subset = n_waves_to_average//n_waveforms_per_batch
    batch_peak_channels = batch_peak_channels[np.argsort(batch_peak_channels[:,2])] # sort by amplitude

    if plot_debug:
        max_amp_hist = np.max(batch_peak_channels[:,2])
        max_amp_hist += 10-max_amp_hist%10
        nbatches_hist = batch_peak_channels.shape[0]
        fig = hist_MB(batch_peak_channels[:,2], a=10, b=max_amp_hist, s=5, color='grey', alpha=0.7)

    # if less than n_driftmatched_subset batches below 95th percentile,
    # use all up to n_driftmatched_subset batches
    ## TODO only perform the following if distribution is unimodal
    # (Hartigan Dip-test of Unimodality)
    prct_95_i = int(batch_peak_channels.shape[0]*0.95)
    if prct_95_i<n_driftmatched_subset: 
        batch_peak_channels = batch_peak_channels[0:n_driftmatched_subset]
    else:
        i_left = max(prct_95_i - n_driftmatched_subset, 0) # should never be negative given if statement, but precaution
        batch_peak_channels = batch_peak_channels[i_left:prct_95_i]
    drift_matched_spike_ids = np.sort(batch_peak_channels[:,0])

    if plot_debug:
        fig = hist_MB(batch_peak_channels[:,2], a=10, b=max_amp_hist, s=5, ax=fig.get_axes()[0], color='orange', alpha=0.7,
        title=(f'XY drift matching:\ndistribution of amplitude on peak channel across spike batches\n'
               f'({n_waveforms_per_batch} spikes/batch - {batch_peak_channels.shape[0]}/{nbatches_hist} batches)'))


    #### shift matching ####
    # extract drift-matched raw waveforms
    dsmatch_batch_ids = batch_peak_channels[:,0].astype(np.int64)
    drift_matched_waves = raw_waves[dsmatch_batch_ids]#.reshape(-1, t_waveforms, raw_waves.shape[-1])
    drift_matched_batches = np.mean(drift_matched_waves, axis=1)

    # shift waves using simple negative peak matching
    recenter_spikes = False
    if do_shift_match:
        drift_shift_matched_batches = shift_match(drift_matched_batches, peak_channel, max_allowed_shift, recenter_spikes, plot_debug)
    else:
        drift_shift_matched_batches = drift_matched_batches
    # Get the median of the drift and shift matched waves (not sensitive to outliers)
    drift_shift_matched_mean = np.median(drift_shift_matched_batches, axis=0)
    drift_shift_matched_mean_peak = drift_shift_matched_mean[:,peak_channel]
    # recenter spike absolute maximum
    shift = (np.argmax(np.abs(drift_shift_matched_mean_peak)) - drift_shift_matched_mean_peak.shape[0]//2)%drift_shift_matched_mean_peak.shape[0]
    drift_shift_matched_mean = np.concatenate([drift_shift_matched_mean[shift:], drift_shift_matched_mean[:shift]], axis=0)
    drift_shift_matched_mean_peak = np.concatenate([drift_shift_matched_mean_peak[shift:], drift_shift_matched_mean_peak[:shift]], axis=0)

    if save:
        np.save(Path(dpnm,fn), drift_shift_matched_mean_peak)
        np.save(Path(dpnm,fn_all), drift_shift_matched_mean)
        np.save(Path(dpnm,fn_spike_id), drift_matched_spike_ids)
        np.save(Path(dpnm, fn_peakchan), peak_channel)

    if plot_debug:
        if verbose: print(f'Total averaged waveform batches ({n_waveforms_per_batch}/batch) after drift-shift matching: {batch_peak_channels.shape[0]}')
        fig = quickplot_n_waves(np.mean(mean_waves[np.random.randint(0, mean_waves.shape[0], batch_peak_channels.shape[0]),:,:], axis=0), '', peak_channel)
        fig = quickplot_n_waves(np.mean(drift_matched_batches, axis=0), '', peak_channel, fig=fig)
        fig = quickplot_n_waves(drift_shift_matched_mean, 'raw:blue\ndrift-matched:orange\ndrift-shift-matched:green', peak_channel, fig=fig)
        #breakpoint()

    return drift_shift_matched_mean_peak, drift_shift_matched_mean, drift_matched_spike_ids, peak_channel

def shift_match(waves, alignment_channel,
                chan_range=2, recenter_spikes=False,
                plot_debug=False, dynamic_template=False,
                max_shift_allowed = 5):
    """
    Iterate through waveforms to align them to each other
    by maximizing their convolution.

    In order to have a time complexity O(n),
    starts by aligning waves[1] to waves[0],
    then waves[2] to mean(waves[[0], aligned waves[1]])...
    So every wave will be aligned to the first wave.

    When shifting a wave, fills the gap
    with the bit clipped from the other end.

    Parameters:
        - waves: (n_waves, n_samples, n_channels,) array
        - alignment_channel: channel to use to compute the convolution
        - chan_range: int, range of channels around alignment channel used for template matching
                      (3 corresponds to 6 channels, 5 to 10 etc)
        - recenter_spikes: bool, whether to align the maximum of template to 0
        - dynamic_template: bool, whether to update the template by averaging it with the aligned spike
        - max_shift_allowed: int, maximum shift allowed (half window) - other waveforms are discarded (if need to shift more, they must be way too noisy)

    Returns:
        - shifted_waves: (n_waves, n_samples, n_channels,) array
    """
    

    # sort waveforms by amplitude
    amplitudes = np.ptp(waves[:,:,alignment_channel], axis=1)
    amplitudes_i = np.argsort(amplitudes, axis=0)
    waves_sort = waves[amplitudes_i[::-1],:,:]
    # use median of 50 waves of highest amplitude as template
    # most arbitrary decision 0 but seems reasonable and empirically works
    n_waveforms_template=50
    template = np.median(waves_sort[:n_waveforms_template,:,:], axis=0)
    if recenter_spikes:
        shift = (np.argmax(np.abs(template[:,alignment_channel])) - template.shape[0]//2)%template.shape[0]
        if plot_debug:
            plt.figure()
            plt.plot(template[:,alignment_channel])
        template = np.concatenate([template[shift:], template[:shift]], axis=0)# shift template to center maximum
        if plot_debug: plt.plot(template[:,alignment_channel])

    # initialize array
    aligned_waves = np.zeros(waves_sort.shape)
    chan_min, chan_max = max(0,alignment_channel-chan_range), min(alignment_channel+chan_range, template.shape[1])
    template = template[:,chan_min:chan_max] # defined across 10 closest channels
    shifts = []
    for i, w in enumerate(waves_sort):
        w_closestchannels = w[:,chan_min:chan_max]
        xcorr_w_template = xcorr_1d_loop(template, w_closestchannels)
        # average xcorr across channels to find the optimal alignment
        # using information from all channels around peak!
        xcorr_max = np.argmax(np.mean(xcorr_w_template, axis=1))
        shift = (xcorr_max-waves_sort.shape[1]//2)%waves_sort.shape[1]
        relative_shift = (shift+waves_sort.shape[1]//2)%waves_sort.shape[1]-waves_sort.shape[1]//2
        shifts.append(relative_shift)
        realigned_w = np.concatenate([w[-shift:,:], w[:-shift,:]], axis=0)
        if abs(relative_shift)>max_shift_allowed:
            realigned_w=realigned_w*np.nan
        if plot_debug and i==0:
            fig=imshow_cbar(template)
            fig=imshow_cbar(w_closestchannels)
            fig=imshow_cbar(xcorr_w_template)
        # store realigned_wave in array
        aligned_waves[i,:,:] = realigned_w

        # optionnally update template by averaging with realigned waveform
        # didn't work very well - better to keep a relatively 'focused' template
        if dynamic_template:
            template = np.mean(np.stack(
                            [template,
                            realigned_w[:,chan_min:chan_max]],
                            axis=2), axis=2)

    # discard nans (beyond max_shift_allowed) and re-sort waves properly
    aligned_waves = aligned_waves[np.arange(aligned_waves.shape[0])[amplitudes_i[::-1]],:,:]
    nan_m = np.isnan(aligned_waves[:,0,0])
    aligned_waves = aligned_waves[~nan_m,:,:]

    if plot_debug:
        #fig = imshow_cbar(template.T)
        a = np.max(np.abs(shifts))
        a += 10-a%10
        fig = hist_MB(shifts, s=1, a=-a, b=a, color='darkgreen', alpha=1,
        title=(f'Shift matching:\ndistribution of shifts w/r template across spike batches'))
        ylim = fig.get_axes()[0].get_ylim()
        fig.get_axes()[0].plot([max_shift_allowed,max_shift_allowed], ylim, color='red', ls='--')
        fig.get_axes()[0].plot([-max_shift_allowed,-max_shift_allowed], ylim, color='red', ls='--')
        fig.get_axes()[0].set_ylim(ylim)

    return aligned_waves

def get_pc(waveforms):
    wvf_m = np.mean(waveforms, axis=0)
    max_min_wvf=np.max(wvf_m,0)-np.min(wvf_m,0)
    peak_chan = np.argmax(max_min_wvf)
    return peak_chan

def get_peak_chan(dp, unit, use_template=True, again=False, ignore_ks_chanfilt=True, periods='all'):
    '''
    Returns index of peak channel, either according to the full probe channel map (0 through 383)
                                   or according to the kilosort channel map (0 through N with N<=383)

    Parameters:
        - datapath, string
        - unit, integer or float (if merged dataset)
        - use_template: bool, whether to use templates instead of raw waveform to find peak channel.
        - again: whether to recompute the waveforms/templates
        - ignore_ks_chanfilt: bool, whether to return the channel rank on the full probe
                    rather than the channel rank on the kilosort cahnnel map (jumping some channels).
                    They will be the same if all channels are used for spike sorting.
                    E.g. if kilosort only used 380 channels (out of 384),
                    the last channel, 383, has the relative index 379.
    Returns:
        - best_channel, integer indexing the channel
          where the unit averaged raw waveform (n=100 spanning the whole recording)
          has the largest peak to trough amplitude.

          WARNING: this value is ABSOLUTE ON THE PROBE SHANK BY DEFAULT. If you wish the relative channel index
          taking into account channels ignored by kilosort, set ignore_ks_chanfilt to False.
    '''
    dp = Path(dp)
    dp, unit = get_source_dp_u(dp, unit)

    strdic={True:'templates', False:'raw-waveforms'}
    f_all=f'peak_channels_{strdic[use_template]}_all.npy'
    f_good=f'peak_channels_{strdic[use_template]}_good.npy'
    for f in [f_all, f_good]:
        if op.exists(Path(dp, f)):
            peak_chans=np.load(Path(dp, f))
            if unit in peak_chans[:,0]:
                return int(peak_chans[peak_chans[:,0]==unit, 1])

    cm=chan_map(dp, probe_version='local')
    if use_template:
        waveforms=templates(dp, unit)
        ks_peak_chan = get_pc(waveforms)
        peak_chan = cm[:,0][ks_peak_chan]
    else:
        waveforms=wvf(dp, u=unit, n_waveforms=200, t_waveforms=82,
                      selection='regular', periods=periods, spike_ids=None, again=again,
                      ignore_ks_chanfilt=True)
        probe_peak_chan = get_pc(waveforms)
        if ignore_ks_chanfilt: # absolute == relative channel index
            peak_chan = probe_peak_chan
        else: #
            ks_peak_chan = np.nonzero(cm[:,0]==probe_peak_chan)
            peak_chan = ks_peak_chan

    return int(peak_chan)


def get_depthSort_peakChans(dp, units=[], quality='all', use_template=True, again=False, verbose = False):
    '''
    Usage:
        Either feed in a list of units - the function will return their indices/channels sorted by depth in a n_units x 2 array,
        or simply indicate units 'quality' as 'all', 'mua' or good - will behave as if you had fed the list of units of this given quality.
    Parameters:
        - datapath, string
        - units, list of integers or strings
        - quality: string, 'all', 'mua' or 'good'
    Returns:
        - best_channels, numpy array of shape (n_units, 2).
          Column 1: unit indices, column 2: respective peak channel indices.
    '''

    dp = Path(dp)
    save=False # can only turn True if no (i.e. all) units are fed
    strdic={True:'templates', False:'raw-waveforms'}

    if len(units)==0:
        # If no units, load them all from dataset
        # and prepare to save the FULL array of peak channels at the end
        units=get_units(dp, quality=quality, again=again)
        assert np.any(units), f'No units of quality {quality} found in this dataset.'
        pc_fname=f'peak_channels_{strdic[use_template]}_{quality}.npy'
        if op.exists(Path(dp, pc_fname)) and not again:
            peak_chans=np.load(Path(dp, pc_fname))
            if np.all(np.isin(units, peak_chans[:,0])):
                return peak_chans
            else:
                save=True
        else:
            save=True
    else:
        # If units are fed, try to load the peak channels
        # from the saved FULL array of peak channels
        units=npa(units).flatten()
        f_all=f'peak_channels_{strdic[use_template]}_all.npy'
        f_good=f'peak_channels_{strdic[use_template]}_good.npy'
        for f in [f_all, f_good]:
            if op.exists(Path(dp, f)):
                peak_chans=np.load(Path(dp, f))
                if np.all(np.isin(units, peak_chans[:,0])):
                    units_mask=np.isin(peak_chans[:,0], units)
                    return peak_chans[units_mask]

    dt=np.float64 if assert_multi(dp) else np.int64
    peak_chans=npa(zeros=(len(units),2),dtype=dt)
    for iu, u in enumerate(units):
        if verbose: print("Getting peak channel of unit {}...".format(u))
        peak_chans[iu,0] = u
        peak_chans[iu,1] = np.array([get_peak_chan(dp, u, use_template)]).astype(dt)
    if assert_multi(dp):
        depth_ids = np.lexsort((-peak_chans[:,1], get_ds_ids(peak_chans[:,0])))
    else:
        depth_ids = np.argsort(peak_chans[:,1])[::-1] # From surface (high ch) to DCN (low ch)
    peak_chans=peak_chans[depth_ids,:]

    if save:
        np.save(Path(dp, pc_fname), peak_chans)

    return peak_chans # units, channels

def get_peak_pos(dp, unit, use_template=False, periods='all'):
    "Returns [x,y] relative position on the probe in um (y=0 at probe tip)."
    dp, unit = get_source_dp_u(dp, unit)
    peak_chan=get_peak_chan(dp, unit, use_template, periods=periods)
    pos = np.load(Path(dp,'channel_positions.npy'))
    cm=chan_map(dp, probe_version='local')
    return pos[cm[:,0]==peak_chan].ravel()

def get_chDis(dp, ch1, ch2):
    '''dp: datapath to dataset
    ch1, ch2: channel indices (1 to 384)
    returns distance in um.'''
    assert 1<=ch1<=384
    assert 1<=ch2<=384
    ch_pos = np.load(Path(dp,'channel_positions.npy')) # positions in um
    ch_pos1=ch_pos[ch1-1] # convert from ch number to ch relative index
    ch_pos2=ch_pos[ch2-1] # convert from ch number to ch relative index
    chDis=np.sqrt((ch_pos1[0]-ch_pos2[0])**2+(ch_pos1[1]-ch_pos2[1])**2)
    return chDis

def templates(dp, u, ignore_ks_chanfilt=False):
    '''
    ********
    routine from rtn.npyx.spk_wvf
    Extracts the template used by kilosort to cluster this unit.
    ********

    Parameters:
        - dp: string, datapath to kilosort output
        - unit: int, unit index
        - ignore_ks_chanfilt: bool, whether to ignore kilosort channel map (skipping unactive channels).
                              If true, n_channels is lower than 384.

    Returns:
        temaplate: numpy array of shape (n_templates, 82, n_channels) where n_channels is defined by the channel map and n_templates by how many merges were done.

    '''
    dp, u = get_source_dp_u(dp, u)

    IDs=ids(dp,u)
    #mean_amp=np.mean(np.load(Path(dp,'amplitudes.npy'))[IDs])
    template_ids=np.unique(np.load(Path(dp,'spike_templates.npy'))[IDs])
    templates = np.load(Path(dp, 'templates.npy'))[template_ids]#*mean_amp

    if ignore_ks_chanfilt:
        cm=chan_map(dp, y_orig='surface', probe_version='local')[:,0]
        cm_all=chan_map(dp, y_orig='surface', probe_version=None)[:,0]
        jumped_chans=np.sort(cm_all[~np.isin(cm_all, cm)])
        for ch in jumped_chans: # need to use a for loop because indices are right only absolutely, not relatively
            templates=np.insert(templates, ch, np.zeros(templates.shape[1]), 2)

    return templates

#%% wvf utilities

def get_ids_subset(dp, unit, n_waveforms, batch_size_waveforms, selection, periods, ignore_nwvf, verbose=False):
    
    # if periods were provided
    if not isinstance(periods, str):
        ids_subset = ids(dp, unit, periods=periods)
        if not ignore_nwvf:
            n_waveforms1=min(n_waveforms, len(ids_subset))
            ids_subset = np.unique(np.random.choice(ids_subset, n_waveforms1, replace=False))
        if verbose: print(f'In subset {periods}, {len(ids_subset)} waveforms were found (n_waveforms={n_waveforms}).')
    # if no periods were provided
    else:
        if n_waveforms in (None, 0):
            ids_subset = ids(dp, unit)
        else:
            assert n_waveforms > 0
            spike_ids = ids(dp, unit)
            assert any(spike_ids)
            assert selection in ['regular', 'random']
            if selection == 'regular':
                ids_subset = select_waveforms_in_batch(spike_ids, n_waveforms, batch_size_waveforms)
            elif selection == 'random' and len(spike_ids) > n_waveforms:
                ids_subset = np.unique(np.random.choice(spike_ids, n_waveforms, replace=False))

    return ids_subset

def select_waveforms_in_batch(spike_ids, n_waveforms, batch_size_waveforms):
    "Batch selection of spikes."
    if batch_size_waveforms is None or len(spike_ids) <= max(batch_size_waveforms, n_waveforms):
        step = ceil(np.clip(1. / n_waveforms * len(spike_ids),
                        1, len(spike_ids)))
        ids_subset = spike_ids[0::step][:n_waveforms]
    else:
        n_excerpts=n_waveforms // batch_size_waveforms
        excerpt_size=batch_size_waveforms
        ids_subset = np.concatenate([data_chunk(spike_ids, chunk)
                    for chunk in excerpts(len(spike_ids),
                                        n_excerpts=n_excerpts,
                                        excerpt_size=excerpt_size)])
        
    return ids_subset

def data_chunk(data, chunk, with_overlap=False):
    """Get a data chunk."""
    assert isinstance(chunk, tuple)
    if len(chunk) == 2:
        i, j = chunk
    elif len(chunk) == 4:
        if with_overlap:
            i, j = chunk[:2]
        else:
            i, j = chunk[2:]
    else:
        raise ValueError("'chunk' should have 2 or 4 elements, "
                         "not {0:d}".format(len(chunk)))
    return data[i:j, ...]

def excerpts(n_samples, n_excerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    assert n_excerpts >= 2
    step = _excerpt_step(n_samples,
                         n_excerpts=n_excerpts,
                         excerpt_size=excerpt_size)
    for i in range(n_excerpts):
        start = i * step
        if start >= n_samples:
            break
        end = min(start + excerpt_size, n_samples)
        yield start, end

def _excerpt_step(n_samples, n_excerpts=None, excerpt_size=None):
    """Compute the step of an excerpt set as a function of the number
    of excerpts or their sizes."""
    assert n_excerpts >= 2
    step = max((n_samples - excerpt_size) // (n_excerpts - 1),
               excerpt_size)
    return step

# Recurrent imports
from npyx.merger import assert_multi, get_ds_ids, get_source_dp_u
from npyx.spk_t import ids
from npyx.plot import hist_MB, imshow_cbar, quickplot_n_waves
