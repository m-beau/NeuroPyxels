# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

import os
import os.path as op; opj=op.join
from pathlib import Path
import struct

import tempfile
from joblib import Parallel, delayed, load, dump
import multiprocessing
num_cores = multiprocessing.cpu_count()

import imp
from ast import literal_eval as ale
import numpy as np
from math import ceil

from npyx.io import ConcatenatedArrays, _pad, _range_from_slice, read_spikeglx_meta, chan_map, whitening, bandpass_filter, apply_filter, med_substract
from npyx.utils import _as_array, npa, split, n_largest_samples
import time

def wvf(dp, u=None, n_waveforms=100, t_waveforms=82, subset_selection='regular', spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
        save=True, prnt=False, again=False,
        whiten=False, med_sub=False, hpfilt=False, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=True,
        use_old=False, loop=True, parallel=False, memorysafe=False, ask_peak_channel=False):
    '''
    ********
    routine from rtn.npyx.spk_wvf
    Extracts a sample of waveforms from the raw data file.
    ********

    Parameters:
        - dp:                 str or PosixPath, path to kilosorted dataset.
        - u:                  int, unit index.
        - n_waveforms:        int, number of waveform to return, selected according to the subset_selection parameter | Default 100
        - t_waveforms:        int, temporal span of waveforms | Default 82 (about 3ms)
        - subset_selection:   str/list of tuples, either 'regular' (homogeneous selection or in batches), 'random',
                                                  or a list of time chunks [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
        - spike_ids:          list/array, relative indices of spikes in the whole recording.
                                          If provided, u, n_waveforms and subset_selection will be ignored.
        - wvf_batch_size:     int, if >1 and 'regular' selection, selects ids as batches of spikes. | Default 10
        - save: bool,         whether to save to routine memory. | Default True
        - prnt: bool,         whether to print informaiton. | Default False
        - again: bool,        whether to recompute waveforms even if ofund in routines memory. | Default False
        - ignore_nwvf:        bool, whether to ignore n_waveforms parameter when a list of times is provided as subset_selection,
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
        - use_old:            bool, whether to use phy 1 implementation of waveform loading. | Default False
        - loop:               bool, whether to use a loop to iterate over waveforms
                                    instead of masking of the whole memory-mapped binary file to eaxtract waveforms. | Default True
                                    Looping is faster, especially if parallel is True.
        - parallel:           bool, if loop is True, whether to use parallel processing to go faster
                                    (depends on number of CPU cores). | Default False

    Returns:
        waveforms:            numpy array of shape (n_waveforms x t_waveforms x n_channels)
                                    where n_channels is defined by the channel map if ignore_ks_chanfilt is False.

    '''

    if spike_ids is not None:
        if u is not None and prnt: print('WARNING you provided both u and spike_ids! u is ignored.')
        if n_waveforms !=100 and prnt: print('WARNING you provided both n_waveforms and spike_ids! n_waveforms is ignored.')
        if not isinstance(subset_selection,str) and prnt: print('WARNING you provided both subset_selection and spike_ids! subset_selection is ignored.')
        u=np.unique(np.load(Path(dp)/'spike_clusters.npy')[spike_ids])
        assert len(u)==1, 'WARNING the spike ids that you provided seem to belong to different units!! Double check!'
        u=u[0]
    dp, u = get_source_dp_u(dp, u)

    dprm = Path(dp,'routinesMemory')
    fn=f"wvf{u}_{n_waveforms}-{t_waveforms}_{str(subset_selection)[0:10].replace(' ', '')}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}-{ignore_ks_chanfilt}.npy"
    if not os.path.isdir(dprm): os.makedirs(dprm)
    if os.path.exists(Path(dprm,fn)) and (not again) and (spike_ids is None):
        if prnt: print("File {} found in routines memory.".format(fn))
        return np.load(Path(dprm,fn))

    #if use_old:
        #waveforms = get_waveform_old(dp, u, n_waveforms, t_waveforms, subset_selection, wvf_batch_size, ignore_nwvf)
    if use_old:
        waveforms = get_waveforms_old2(dp, u, n_waveforms, t_waveforms, subset_selection, spike_ids, wvf_batch_size, ignore_nwvf,
                                 whiten, med_sub, hpfilt, hpfiltf, nRangeWhiten, nRangeMedSub, ignore_ks_chanfilt, prnt,
                                 loop, parallel, memorysafe)
    else:
        waveforms = get_waveforms(dp, u, n_waveforms, t_waveforms, subset_selection, spike_ids, wvf_batch_size, ignore_nwvf,
                 whiten, med_sub, hpfilt, hpfiltf, nRangeWhiten, nRangeMedSub, ignore_ks_chanfilt, prnt)
    # Save it
    if (save and (spike_ids is None)):
        np.save(Path(dprm,fn), waveforms)

    #if ask_peak_channel:print(f'Peak channel:{} ({}th out of {}).')

    return waveforms

def get_w(traces, slc, _n_samples_extract):
    # Get slice
    extract = traces[slc].astype(np.float32)
    # Center channels individually
    extract = extract-np.median(extract, axis=0)
    # Pad the extracted chunk if at recording limit.
    if slc.start <= 0: extract = _pad(extract, _n_samples_extract, 'left')
    elif slc.stop >= traces.shape[0] - 1: extract = _pad(extract, _n_samples_extract, 'right')
    # Add this waveform, all good!
    return extract.T

def get_waveforms(dp, u, n_waveforms=100, t_waveforms=82, subset_selection='regular', spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
                 whiten=0, med_sub=0, hpfilt=0, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=0,
                 prnt=False):
    f"{wvf.__doc__}"

    # Extract and process metadata
    dat_path=None
    for f in os.listdir(dp):
        if f.endswith(".ap.bin"):
            dat_path=Path(dp, f)
            if prnt: print(f'Loading waveforms from {dat_path}.')
    assert dat_path is not None, f'No binary file (*.ap.bin) found in folder {dp}!!'

    meta=read_spikeglx_meta(dp, subtype='ap')
    dtype=np.dtype('int16')
    n_channels_dat=int(meta['nSavedChans'])
    n_channels_rec = n_channels_dat-1
    sample_rate=meta['sRateHz']
    item_size = dtype.itemsize
    fileSizeBytes=op.getsize(dat_path)
    assert meta['fileSizeBytes'] == fileSizeBytes,\
        f'''Mismatch between ap.meta and ap.bin file size (assumed encoding is {dtype} and Nchannels is {n_channels_dat})!!
        Prob wrong meta file - just edit fileSizeBytes and be aware that something went wrong in your data management...'''

    # Select subset of spikes
    spike_samples = np.load(Path(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    if spike_ids is None:
        spike_ids_subset=get_ids_subset(dp, u, n_waveforms, wvf_batch_size, subset_selection, ignore_nwvf, prnt)
    else: spike_ids_subset=spike_ids
    n_spikes = len(spike_ids_subset)

    # Get waveforms times in bytes
    # and check that, for this waveform width,
    # they no not go beyond file limits
    waveforms_t = spike_samples[spike_ids_subset].astype(int)
    waveforms_t1 = (waveforms_t-t_waveforms//2)*n_channels_dat*item_size
    waveforms_t2 = (waveforms_t+t_waveforms//2)*n_channels_dat*item_size
    wcheck_m=(0<=waveforms_t1)&(waveforms_t2<fileSizeBytes)
    if not np.all(wcheck_m):
        print(f"Invalid times: {waveforms_t[~wcheck_m]}")
        waveforms_t1 = waveforms_t1[wcheck_m]
        waveforms_t2 = waveforms_t2[wcheck_m]

    # Iterate over waveforms
    waveforms = np.zeros((n_spikes, t_waveforms, n_channels_rec), dtype=np.float32)
    with open(dat_path, "rb") as f:
        for i,t1 in enumerate(waveforms_t1):
            f.seek(t1, 0) # 0 for absolute file positioning
            wave=f.read(n_channels_dat*t_waveforms*item_size)
            wave=np.frombuffer(wave, dtype=dtype).reshape((t_waveforms,n_channels_dat))
            wave = wave-np.median(wave, axis = 0)[np.newaxis,:]
            waveforms[i,:,:] = wave[:,:-1]# get rid of sync channel

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
    waveforms*=read_spikeglx_meta(dp, 'ap')['scale_factor']

    return  waveforms.astype(np.float32)

def wvf_dsmatch(dp, u, n_waveforms=100,
                  t_waveforms=82, subset_selection='regular',
                  wvf_batch_size=10, ignore_nwvf=True,med_sub = False, spike_ids = None,
                  save=True, prnt=False, again=False,
                  whiten=False,  hpfilt=False, hpfiltf=300,
                  nRangeWhiten=None, nRangeMedSub=None,
                  use_old=False, parallel=False,
                  memorysafe=False, sample_spikes = 2 ):
    """
    ********
    Extract the drift and shift matched mean waveforms of the specified unit.
    Drift and shift matching consists of two steps:

    Drift matching:
        The algorithm first selects all the waveforms that are registered
        on the channel with the highest median amplitude. From the waveforms on
        this channel, the waves that have the highest amplitude (diff between neg
        and positive peaks) is selected.

    Shift matching:
        These waves with the highest amplitudes are then aligned in time to
        match the negative peaks.


    Currently only supports passing a single unit as input, hence
    prints error message if 'spike_ids = single_slice' if passed.
    ********


     Parameters:
        - dp:                 str or PosixPath, path to kilosorted dataset.
        - u:                  int, unit index.
        - n_waveforms:        int, number of waveform to return, selected according to the subset_selection parameter | Default 100
        - t_waveforms:        int, temporal span of waveforms | Default 82 (about 3ms)
        - subset_selection:   str/list of tuples, either 'regular' (homogeneous selection or in batches), 'random',
                                                  or a list of time chunks [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
        - spike_ids:          list/array, relative indices of spikes in the whole recording.
                                          If provided, u, n_waveforms and subset_selection will be ignored.
        - wvf_batch_size:     int, if >1 and 'regular' selection, selects ids as batches of spikes. | Default 10
        - save: bool,         whether to save to routine memory. | Default True
        - prnt: bool,         whether to print informaiton. | Default False
        - again: bool,        whether to recompute waveforms even if ofund in routines memory. | Default False
        - ignore_nwvf:        bool, whether to ignore n_waveforms parameter when a list of times is provided as subset_selection,
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
        - use_old:            bool, whether to use phy 1 implementation of waveform loading. | Default False
        - loop:               bool, whether to use a loop to iterate over waveforms
                                    instead of masking of the whole memory-mapped binary file to eaxtract waveforms. | Default True
                                    Looping is faster, especially if parallel is True.
        - parallel:           bool, if loop is True, whether to use parallel processing to go faster
                                    (depends on number of CPU cores). | Default False

    Returns:
        waveform:            numpy array of shape (t_waveforms)

    """

    if spike_ids is not None:
        raise ValueError('No support yet for passing multiple spike indices. Exiting.')


    # dp, u = get_prophyler_source(dp, u)
    dprm = Path(dp,'routinesMemory')

    fn=f"dsm_{u}_{n_waveforms}-{t_waveforms}_{str(subset_selection)[0:10].replace(' ', '')}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"
    fn_all=f"dsm_{u}_all_waves_{n_waveforms}-{t_waveforms}_{str(subset_selection)[0:10].replace(' ', '')}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"
    fn_spike_id=f"dsm_{u}_spike_id_{n_waveforms}-{t_waveforms}_{str(subset_selection)[0:10].replace(' ', '')}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"
    fn_peakchan=f"dsm_{u}_peakchan_{n_waveforms}-{t_waveforms}_{str(subset_selection)[0:10].replace(' ', '')}_{hpfilt}{hpfiltf}-{whiten}{nRangeWhiten}-{med_sub}{nRangeMedSub}.npy"

    if not dprm.is_dir(): dprm.mkdir()
    # if not os.path.isdir(dprm): os.makedirs(dprm)
    if Path(dprm,fn).is_file() and (not again) and (spike_ids is None):
        if prnt: print(f"File {fn} found in routines memory.")
        return np.load(Path(dprm,fn)),np.load(Path(dprm,fn_all)),np.load(Path(dprm,fn_spike_id)), np.load(Path(dprm,fn_peakchan))

    # Load the spike clusters file
    spike_clusters= np.load(Path(dp, 'spike_clusters.npy')).flatten()

    # Get the spikes ids for the current unit
    spike_cluster_unit = np.nonzero(spike_clusters == u)[0]

    # Reshape the spike ids so we can extract consecutive waves
    #spike_ids_split = reshape_ids_to(spike_clusters, u, size = 10)
    spike_ids_split_all = split(spike_cluster_unit, 10, return_last = False).astype(int)

    spike_ids_split = spike_ids_split_all[::sample_spikes]
    chans_counts = int(100/sample_spikes)
    largest_n = chans_counts

    peak_chan_split = np.zeros((spike_ids_split.shape[0], 2),dtype='float32')

    # Insert an extra column to the existing matrix
    # so we can track which spike it belongs to
    peak_chan_split_indices = np.insert(peak_chan_split,0,
                                    np.arange(0,peak_chan_split.shape[0],1),
                                    axis = 1)

    # Extract the waveforms using the wvf function in blocks of 10.
    # After waves have been extracted, put the index of the channel with the
    # max amplitude as well as the max amplitude into the peak_chan_split array

    spike_ids_split = spike_ids_split.flatten()
    raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = t_waveforms,
                            subset_selection =None ,  spike_ids =spike_ids_split,
                            wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
                            save=save , prnt = prnt,  again=True, whiten = whiten,
                            hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
                            nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True,
                            use_old=use_old, loop=False, parallel=parallel,
                            memorysafe=memorysafe)
    raw_waves = raw_waves.reshape(peak_chan_split.shape[0], 10, 82, -1)
    mean_times = np.mean(raw_waves, axis = 1)

    for no_per_ten in np.arange(mean_times.shape[0]):
        peak_chan_split_indices[no_per_ten,1:] = max_amp_consecutive_peaks(mean_times[no_per_ten])
    spike_ids_split = spike_ids_split.reshape(-1,10)
    no_chans = raw_waves.shape[-1]

#    if fast:
#        spike_ids_split = spike_ids_split.flatten()
#        raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = t_waveforms,
#                                subset_selection =None ,  spike_ids =spike_ids_split,
#                                wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
#                                save=save , prnt = prnt,  again=True, whiten = whiten,
#                                hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
#                                nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True,
#                                use_old=use_old, loop=False, parallel=parallel,
#                                memorysafe=memorysafe)
#        raw_waves = raw_waves.reshape(peak_chan_split.shape[0], 10, 82, -1)
#        mean_times = np.mean(raw_waves, axis = 1)
#
#        for no_per_ten in np.arange(mean_times.shape[0]):
#            peak_chan_split_indices[no_per_ten,1:] = max_amp_consecutive_peaks(mean_times[no_per_ten])
#        spike_ids_split = spike_ids_split.reshape(-1,10)
#        no_chans = raw_waves.shape[-1]
#    else:
#        for idx, idx_spike in enumerate(spike_ids_split):
#
#            raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = t_waveforms,
#                                subset_selection =None ,  spike_ids =idx_spike,
#                                wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
#                                save=save , prnt = prnt,  again=True, whiten = whiten,med_sub=med_sub,
#                                hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
#                                nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True,
#                                use_old=use_old, loop=True, parallel=parallel,
#                                memorysafe=memorysafe)
#            mean_times = np.mean(raw_waves, axis = 0)
#            peak_chan_split_indices[idx,1:] = max_amp_consecutive_peaks(mean_times)

#        no_chans = raw_waves.shape[-1]
    # find the 10long vecotrs where teh peak channel is the most common one
    # sum up the values of these 10 loong blocks

#    breakpoint() 

    # get the slices of tens where the peak channel was the overall most likley peak channel
    # take these waves and extract them again so they can be averaged together

    # count the frequncy of each channel to get which channels were the most active overall

    # chans,count  = np.unique(peak_chan_split[:,0], return_counts = True) 
    all_chan_peaks = peak_chan_split_indices[:,1].astype(np.int32)
#    chans,count  = np.unique(peak_chan_split_indices[:,1].astype(np.int32), return_counts = True) 
    chans,count  = np.unique(all_chan_peaks, return_counts = True)

    more_than_100_chans = chans[count>chans_counts]

    # Find the median amplitude channel
    median_chans= []
    for current_chan in more_than_100_chans:
        current_chan_distr = peak_chan_split_indices[peak_chan_split_indices[:,1] == current_chan][:,2]
        median_chans.append(np.median(current_chan_distr))
#        median_chans.append(np.max(current_chan_distr))

    # Find the channel with the highest median from all channels
    median_common_chan =int(more_than_100_chans[np.argmax(median_chans)])

#    median_common_chan = 123 
#    print('peak chan set at 123!')
    # So we have the channel that has the highest median of amplitudes.
    # Now we pick out the blocks of 10 spikes that were on this channel
    median_chan_splits = peak_chan_split_indices[peak_chan_split_indices[:,1] == median_common_chan]

    # in some cases there are Kilosort artefacts, where a very large spike
    # might be included in the here
    # so we filter out the batches, where the spike is larger than 1000

    median_chan_splits = median_chan_splits[median_chan_splits[:,2] <1800]

    # check if the median_chan_splits matrix is large enough for our needs
    if median_chan_splits.shape[0] < largest_n:
        largest_n = median_chan_splits.shape[0]

    # find the 10 rows where the amplitudue is maxed
    median_chan_max_amp_indices = n_largest_samples(median_chan_splits[:,2],
                                                    largest_n=largest_n)

    # find the slices with the largest amplitude of the waveform
    # this will tell us where the cell was closest to the specific peak channel
    # to get the spike_ids of the section with the highest amplitude difference
    median_max_spike_ids = spike_ids_split[median_chan_splits[median_chan_max_amp_indices][:,0].astype('int16')]


    # now we have the max amplitude spikes on the best single channel
    # extract the waves from this channel

    # initialise array to store data
    closest_waves_median_max= np.zeros((len(median_max_spike_ids),82))
    all_closest_waves = np.zeros((len(median_max_spike_ids),82, no_chans))

    extract_spk_id = median_max_spike_ids.flatten()
    raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = 82,
                    subset_selection =None,  spike_ids =extract_spk_id,
                    wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
                    save=save, prnt = prnt,  again=True, whiten = whiten,
                    hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
                    nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True,
                    use_old=use_old, loop=False, parallel=parallel,
                    memorysafe=memorysafe)
    raw_waves = raw_waves.reshape(median_max_spike_ids.shape[0],10, 82, no_chans )
    closest_waves_median_max = np.mean(raw_waves, axis = 1)[:,:,median_common_chan]
    all_closest_waves= np.mean(raw_waves, axis=1)
#    if fast:
#        extract_spk_id = median_max_spike_ids.flatten()
#        raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = 82,
#                        subset_selection =None,  spike_ids =extract_spk_id,
#                        wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
#                        save=save, prnt = prnt,  again=True, whiten = whiten,
#                        hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
#                        nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True,
#                        use_old=use_old, loop=False, parallel=parallel,
#                        memorysafe=memorysafe)
#        raw_waves = raw_waves.reshape(median_max_spike_ids.shape[0],10, 82, no_chans )
#        closest_waves_median_max = np.mean(raw_waves, axis = 1)[:,:,median_common_chan]
#        all_closest_waves= np.mean(raw_waves, axis=1)
#    else:
#        for slice_id, single_slice in enumerate(median_max_spike_ids):
#            raw_waves = wvf(dp, u = None, n_waveforms= 100, t_waveforms = 82,
#                        subset_selection =None,  spike_ids =single_slice,
#                        wvf_batch_size =wvf_batch_size , ignore_nwvf=ignore_nwvf,
#                        save=save, prnt = prnt,  again=True, whiten = whiten,med_sub = med_sub,
#                        hpfilt = hpfilt, hpfiltf = hpfiltf, nRangeWhiten=nRangeWhiten,
#                        nRangeMedSub=nRangeMedSub, ignore_ks_chanfilt=True,
#                        use_old=use_old, loop=True, parallel=parallel,
#                        memorysafe=memorysafe)
#
#            closest_waves_median_max[slice_id] = np.mean(raw_waves, axis = 0)[:,median_common_chan]
#            all_closest_waves[slice_id] = np.mean(raw_waves, axis=0)
    # shift waves using simple negative peak matching
    shifted_waves_median_max = shift_many_waves(closest_waves_median_max)
    # Get the mean of the drift and shift matched waves
    mean_shifted_waves = np.mean(shifted_waves_median_max, axis = 0)

    # shift waves for all channels
    shifted_all_waves = np.zeros((len(median_max_spike_ids),82,no_chans ))
    # need to make sure that for each wave the shifts along channels are same
    # so find the shift needed on peak chan, apply to all waves

    need_shift = shift_neg_peak(all_closest_waves[:,:,median_common_chan])
    for chani in range(all_closest_waves.shape[2]):
        shifted_all_waves[:,:,chani] = shift_many_waves(all_closest_waves[:,:,chani], need_shift)
#    breakpoint()
    #for chani in range(all_closest_waves.shape[2]):
       # shifted_all_waves[:,:,chani] = shift_many_waves(all_closest_waves[:,:,chani])

    mean_shift_all = np.mean(shifted_all_waves, axis = 0)
    # save the drift and shift mean eave

    if save:
        np.save(Path(dprm,fn), mean_shifted_waves)
        np.save(Path(dprm,fn_all), mean_shift_all)
        np.save(Path(dprm,fn_spike_id), median_max_spike_ids)
        np.save(Path(dprm, fn_peakchan),median_common_chan )


    return mean_shifted_waves, mean_shift_all, median_max_spike_ids, median_common_chan
    

def max_amp_consecutive_peaks_break(mean_waves: np.array) -> tuple:

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
    """

    truncated_waves = np.zeros_like(mean_waves.T)
    loc_min_val = np.argmin(mean_waves,axis = 0)

    # truncate the waves so we can look at the amplitudes of the neg and next peak
    for idx, row in enumerate(mean_waves.T):
        truncated_waves[idx, loc_min_val[idx]:] = row[loc_min_val[idx]:]
    truncated_waves = truncated_waves.T
    breakpoint()
    return [np.argmax(np.ptp(truncated_waves,axis=0)), np.max(np.ptp(truncated_waves,axis=0))]


def max_amp_consecutive_peaks(mean_waves: np.array) -> tuple:

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
    """

    truncated_waves = np.zeros_like(mean_waves.T)

    loc_min_val = np.argmin(mean_waves, axis = 0)
#    # truncate the waves so we can look at the amplitudes of the neg and next peak
#    for idx, row in enumerate(mean_waves.T):
#        truncated_waves[idx, loc_min_val[idx]:] = row[loc_min_val[idx]:]
#    truncated_waves = truncated_waves.T
#    return [np.argmax(np.ptp(truncated_waves,axis=0)), np.max(np.ptp(truncated_waves,axis=0))]

    # implement the option for also looking at the peaks before the most negative peak
    # find the most positive peak before the negative peak
    # for cases, where there is a triple peak this should improve the extraction
    # where there is no first peak, this will just find some value near 0, hence will also work here

    # first find the most negative peak
    # look for the positive peak after this
    # look for the positive peak before this

    triple_peaks = np.min(mean_waves, axis=0)
    for idx, row in enumerate(mean_waves.T):
        # peak before the min value
        before_peak = np.max(row[:loc_min_val[idx]], initial = 0)
        after_peak = np.max(row[loc_min_val[idx]:], initial = 0)
        triple_peaks[idx]  = np.abs(triple_peaks[idx]) + before_peak + after_peak
#    breakpoint()
#    return [np.argmax(triple_peaks), np.max(triple_peaks)]
    return [np.argmax(np.ptp(mean_waves,axis=0)), np.max(np.ptp(mean_waves,axis=0))]


def shift_many_waves(waves:np.array, *args) -> np.array:

    """

    Shift each wave in a matrix of n waves, so that the negative peak is aligned
    for each single wave

    Input:
        waves -- Matrix of waves

    Return:
        np.array -- matrix with the same dimensions, but with each wave shifted

    """

    shifts_needed = shift_neg_peak(waves)
    shifted_waves  = np.zeros_like((waves))

    if len(args) !=0:
        shifts_needed = args[0]

    for i in range(waves.shape[0]):
        current_shift = shifts_needed[i]
        shifted_waves[i] = np.concatenate([waves[i,current_shift:], waves[i,:current_shift]])

    return shifted_waves


def shift_neg_peak(first_wvf : np.array) -> int:

    """
    Takes as an argument either a single wvf or a matrix of waves
    returns an array with all the shifts needed to put each wave at 40

    Input:
        first_wave -- matrix or vector, containing many waves or a single wave

    Return:
        int/vector -- returns a single int or a vector with each element
                        specifying the shift needed for the current wave

    """

    if len(first_wvf.shape) ==1:
        first_wvf = first_wvf.reshape(1,-1)
    first_neg_peak = np.argmin(first_wvf, axis = 1)

    fourties =  np.repeat(40, repeats =first_neg_peak.shape[0] )
    return first_neg_peak -fourties

def get_pc(waveforms):
    wvf_m = np.mean(waveforms, axis=0)
    max_min_wvf=np.max(wvf_m,0)-np.min(wvf_m,0)
    peak_chan = np.argmax(max_min_wvf)
    return peak_chan

def get_peak_chan(dp, unit, use_template=True, again=False, ignore_ks_chanfilt=True):
    '''
    Parameters:
        - datapath, string
        - unit, integer or string
        - use_template: bool, whether to use templates instead of raw waveform to find peak channel.
        - again: whether to recompute the waveforms/templates
        - ignore_ks_chanfilt: whether to return the absolute channel name
                    rather than the relative channel index.
                    They will be the same if all channels are used for spike sorting.
                    E.g. if kilosort only used 380 channels (out of 384),
                    the last channel absolutely named 383 has the relative index 379.
    Returns:
        - best_channel, integer indexing the channel
          where the unit averaged raw waveform (n=100 spanning the whole recording)
          has the largest peak to trough amplitude.

          WARNING: this value is ABSOLUTE ON THE PROBE SHANK BY DEFAULT. If you wish the relative channel index
          taking into account channels ignored by kilosort, set ignore_ks_chanfilt to False.
    '''
    dp, unit = get_source_dp_u(dp, unit)

    strdic={True:'templates', False:'raw-waveforms'}
    f_all=f'peak_channels_{strdic[use_template]}_all.npy'
    f_good=f'peak_channels_{strdic[use_template]}_good.npy'
    for f in [f_all, f_good]:
        if op.exists(Path(dp, f)):
            peak_chans=np.load(Path(dp, f))
            if unit in peak_chans[:,0]:
                return peak_chans[peak_chans[:,0]==unit, 1]

    cm=chan_map(dp, probe_version='local')
    if use_template:
        waveforms=templates(dp, unit)
    else:
        waveforms=wvf(dp, u=unit, n_waveforms=200, t_waveforms=82, subset_selection='regular', spike_ids=None, again=again,
                      use_old=False, loop=True, parallel=False, memorysafe=False, ignore_ks_chanfilt=True)

    peak_chan = get_pc(waveforms)

    if use_template: # will always be in kilosort relative channel index
        peak_chan=cm[:,0][peak_chan]

    if not ignore_ks_chanfilt:
        peak_chan=np.nonzero(cm[:,0]==peak_chan)

    return peak_chan


def get_depthSort_peakChans(dp, units=[], quality='all', use_template=True, again=False, prnt = False):
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
    save=False # can only turn True if no (i.e. all) units are fed
    strdic={True:'templates', False:'raw-waveforms'}

    if not np.any(units):
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
        if prnt: print("Getting peak channel of unit {}...".format(u))
        peak_chans[iu,0] = u
        peak_chans[iu,1] = get_peak_chan(dp, u, use_template).astype(dt)
    if assert_multi(dp):
        depth_ids = np.lexsort((-peak_chans[:,1], get_ds_ids(peak_chans[:,0])))
    else:
        depth_ids = np.argsort(peak_chans[:,1])[::-1] # From surface (high ch) to DCN (low ch)
    peak_chans=peak_chans[depth_ids,:]

    if save:
        np.save(Path(dp, pc_fname), peak_chans)

    return peak_chans # units, channels

def get_peak_pos(dp, unit, use_template=False):
    dp, unit = get_source_dp_u(dp, unit)
    peak_chan=get_peak_chan(dp, unit, use_template)
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

def get_ids_subset(dp, unit, n_waveforms, batch_size_waveforms, subset_selection, ignore_nwvf, prnt=False):
    if type(subset_selection) not in [str, np.str_]:
        ids_subset = ids(dp, unit, subset_selection=subset_selection)
        if not ignore_nwvf:
            n_waveforms1=min(n_waveforms, len(ids_subset))
            ids_subset = np.unique(np.random.choice(ids_subset, n_waveforms1, replace=False))
            if prnt: print('In subset {}, {} waveforms were found (n_waveforms={}).'.format(subset_selection, len(ids_subset), n_waveforms))
        else:
            if prnt: print('In subset {}, {} waveforms were found (parameter n_waveforms={} ignored).'.format(subset_selection, len(ids_subset), n_waveforms))
    else:
        assert subset_selection in ['regular', 'random', 'all']
        if n_waveforms in (None, 0) or subset_selection=='all':
            ids_subset = ids(dp, unit)
        else:
            assert n_waveforms > 0
            spike_ids = ids(dp, unit)
            assert any(spike_ids)
            if subset_selection == 'regular':
                # Regular subselection.
                if batch_size_waveforms is None or len(spike_ids) <= max(batch_size_waveforms, n_waveforms):
                    step = ceil(np.clip(1. / n_waveforms * len(spike_ids),
                                 1, len(spike_ids)))
                    ids_subset = spike_ids[0::step][:n_waveforms] #regular_subset(spike_ids, n_spikes_max=n_samples_waveforms)
                else:
                    # Batch selections of spikes.
                    n_excerpts=n_waveforms // batch_size_waveforms
                    excerpt_size=batch_size_waveforms
                    ids_subset = np.concatenate([data_chunk(spike_ids, chunk)
                              for chunk in excerpts(len(spike_ids),
                                                    n_excerpts=n_excerpts,
                                                    excerpt_size=excerpt_size)]) #get_excerpts(spike_ids, n_samples_waveforms // batch_size_waveforms, batch_size_waveforms)
            elif subset_selection == 'random' and len(spike_ids) > n_waveforms:
                # Random subselection.
                ids_subset = np.unique(np.random.choice(spike_ids, n_waveforms, replace=False))

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

#%% Waveform loader from phy


def wvf_old(dp, u, n_waveforms=100, t_waveforms=82, subset_selection='regular', wvf_batch_size=10, sav=True, prnt=False, again=False, ignore_nwvf=True, bpfilter=False, whiten=False, med_sub=False):
    '''
    ********
    routine from rtn.npyx.spk_wvf
    Extracts a sample of waveforms from the raw data file.
    ********

    Parameters:
        - dp:
        - unit:
        - n_waveforms:
        - t_waveforms:
        - wvf_subset_selection: either 'regular' (homogeneous selection or in batches), 'random',
                                                  or a list of time chunks [(t1, t2), (t3, t4), ...] with t1, t2 in seconds.
        - wvf_batch_size: if >1 and 'regular' selection, selects ids as batches of spikes.

    Returns:
        waveforms: numpy array of shape (n_waveforms, t_waveforms, n_channels) where n_channels is defined by the channel map.

    '''
    dp, u = get_source_dp_u(dp, u)

    dprm = Path(dp,'routinesMemory')
    fn="wvf{}_{}-{}_{}_{}-{}-{}.npy".format(u, n_waveforms, t_waveforms, str(subset_selection)[0:10].replace(' ', ''), bpfilter, whiten, med_sub)
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        waveforms = np.load(Path(dprm,fn))

    # if not, compute it
    else:
        waveforms = get_waveform_old(dp, u, n_waveforms, t_waveforms, subset_selection, wvf_batch_size, ignore_nwvf, bpfilter, whiten, med_sub)
        # Save it
        if sav:
            np.save(Path(dprm,fn), waveforms)

    return waveforms


def get_waveforms_old2(dp, u, n_waveforms=100, t_waveforms=82, subset_selection='regular', spike_ids=None, wvf_batch_size=10, ignore_nwvf=True,
                 whiten=0, med_sub=0, hpfilt=0, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None, ignore_ks_chanfilt=0,
                 prnt=False, loop=True, parallel=False, memorysafe=False):
    f"{wvf.__doc__}"

    # Extract and process metadata
    meta=read_spikeglx_meta(dp, subtype='ap')
    dtype='int16'
    offset=0
    n_channels_dat=int(meta['nSavedChans'])
    sample_rate=meta['sRateHz']
    dat_path=None
    for f in os.listdir(dp):
        if f.endswith(".ap.bin"):
            dat_path=Path(dp, f)
            if prnt: print(f'Loading waveforms from {dat_path}.')
    assert dat_path is not None, f'No binary file (*.ap.bin) found in folder {dp}!!'

    # Load traces from binary file
    n_channels_rec = n_channels_dat-1
    item_size = np.dtype(dtype).itemsize
    bin_file_size=op.getsize(dat_path)
    n_samples_dat = (bin_file_size - offset) // (item_size * n_channels_dat)
    traces = np.memmap(dat_path, dtype=dtype, mode='r',
                       offset=offset, shape=(n_samples_dat, n_channels_dat))[:,:-1] # ignore the sync channel at the end
    assert meta['fileSizeBytes'] == bin_file_size,\
        f'Mismatch between ap.meta and ap.bin file size (assumed encoding is {dtype} and Nchannels is {n_channels_dat})!!'

    # Select subset of waveforms
    spike_samples = np.load(Path(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    if spike_ids is None:
        spike_ids_subset=get_ids_subset(dp, u, n_waveforms, wvf_batch_size, subset_selection, ignore_nwvf, prnt)
    else: spike_ids_subset=spike_ids
    n_spikes = len(spike_ids_subset)

    # Define waveform temporal indexing
    pad=_before_after(100) # 1000 is fine - should be at least 3*filter_order, which is 3!
    n_samples_before_after=_before_after(t_waveforms)
    _n_samples_extract=t_waveforms + sum(pad)

    # Iterate over waveforms
    if memorysafe: loop=True
    if loop:
        waveforms_t = spike_samples[spike_ids_subset].astype(int)
        wcheck_m=(0<=waveforms_t)&(waveforms_t<traces.shape[0])
        assert np.all(wcheck_m), f"Invalid times: {waveforms_t[~wcheck_m]}"
        slices=[slice(max(t-n_samples_before_after[0]-pad[0],0), t+n_samples_before_after[1]+pad[1]) for t in waveforms_t]
        if parallel:
            print("Parallelism doesn't work because of communication overhead  - do not try to use it! n_jobs set to 1.")
            all_waves=Parallel(n_jobs=1, prefer='threads')(delayed(get_w)(traces, slc, _n_samples_extract) for slc in slices)
            waveforms = np.array(all_waves)
        else:
            waveforms = np.zeros((n_spikes, n_channels_rec, _n_samples_extract), dtype=np.float32)
            for i, slc in enumerate(slices):
                # Extract from memorymapped traces
                extract = traces[slc].astype(np.float32)
                # Center channels individually
                extract = extract-np.median(extract, axis=0)
                # Pad the extracted chunk if at recording limit.
                if slc.start <= 0: extract = _pad(extract, _n_samples_extract, 'left')
                elif slc.stop >= traces.shape[0] - 1: extract = _pad(extract, _n_samples_extract, 'right')
                # Preprocess the waveforms.
                extract=extract.T
                if hpfilt|med_sub|whiten and memorysafe:
                    if hpfilt:
                        extract=apply_filter(extract, bandpass_filter(rate=sample_rate, low=None, high=hpfiltf, order=3), axis=1)
                    if med_sub:
                        extract=med_substract(extract, axis=0, nRange=nRangeMedSub)
                    if whiten:
                        extract=whitening(extract, nRange=nRangeWhiten)
                # Add this waveform, all good!
                waveforms[i,:,:] = extract
        # Preprocess the waveforms.
        if hpfilt|med_sub|whiten and not memorysafe:
            waveforms=np.transpose(waveforms, (1,0,2)).reshape((n_channels_rec, n_spikes*_n_samples_extract))
            if hpfilt:
                waveforms=apply_filter(waveforms, bandpass_filter(rate=sample_rate, low=None, high=hpfiltf, order=3), axis=1)
            if med_sub:
                waveforms=med_substract(waveforms, axis=0, nRange=nRangeMedSub)
            if whiten:
                waveforms=whitening(waveforms, nRange=nRangeWhiten)
            waveforms=np.transpose(waveforms.reshape((n_channels_rec, n_spikes, _n_samples_extract)), (1,2,0)) # now axis 0 is waveform index, 1 is time, 2 is channels
        else:
            waveforms=np.transpose(waveforms, (0, 2, 1))
    else:
        waveforms_t = spike_samples[spike_ids_subset].astype(int)
        assert np.all(0 <= waveforms_t)&np.all(waveforms_t < traces.shape[0]), "Invalid time!"
        waveforms_tspans=(waveforms_t+np.arange(-n_samples_before_after[0]-pad[0], n_samples_before_after[1]+pad[1])[:,np.newaxis]).T.ravel()
        stitched_waveforms=traces[waveforms_tspans,:].astype(np.float32).T
        # Center median subtract in time before preprocessing to center waveforms on channels
        waveforms=stitched_waveforms.reshape((n_channels_rec, n_spikes, _n_samples_extract))
        waveforms=waveforms-np.median(waveforms, axis=2)[:,:,np.newaxis] # median has shape n_channels_rec x n_spikes, 1 per channel per waveform!
        # Preprocess waveforms before reshaping - much faster!
        if hpfilt|med_sub|whiten:
            waveforms=waveforms.reshape((n_channels_rec, n_spikes*_n_samples_extract))
            if hpfilt:
                waveforms=apply_filter(waveforms, bandpass_filter(rate=sample_rate, low=None, high=hpfiltf, order=3), axis=1)
            if med_sub:
                waveforms=med_substract(waveforms, axis=0, nRange=nRangeMedSub)
            if whiten:
                waveforms=whitening(waveforms, nRange=nRangeWhiten)
            # Reshape waveforms properly
            waveforms=waveforms.reshape((n_channels_rec, n_spikes, _n_samples_extract))
        waveforms=np.transpose(waveforms, (1,2,0)) # now axis 0 is waveform index, 1 is time, 2 is channels

    # Remove the padding after processing, then filter channels.
    assert waveforms.shape[1] == _n_samples_extract
    waveforms = waveforms[:, pad[0]:-pad[1], :]
    assert waveforms.shape[2]==n_channels_rec
    if not ignore_ks_chanfilt:
        channel_ids_ks = np.load(Path(dp, 'channel_map.npy'), mmap_mode='r').squeeze()
        channel_ids_ks=channel_ids_ks[channel_ids_ks!=384]
        waveforms=waveforms[:,:,channel_ids_ks] # only AFTER processing, filter out channels

    # Correct voltage scaling
    waveforms*=read_spikeglx_meta(dp, 'ap')['scale_factor']

    return  waveforms

def get_waveform_old(dp, unit, n_waveforms=100, t_waveforms=82, subset_selection='regular', wvf_batch_size=10, ignore_nwvf=True, bpfilter=False, whiten=False, med_sub=False):
    '''Function to extract a subset of waveforms from a given unit.
    Parameters:
        - dp:
        - unit:
        - n_waveforms:
        - t_waveforms:
        - wvf_subset_selection: either 'regular' (homogeneous selection or in batches) or 'random'
        - wvf_batch_size: if >1 and 'regular' selection, selects ids as batches of spikes.

    Returns:
        waveforms: numpy array of shape (n_waveforms, t_waveforms, n_channels) where n_channels is defined by the channel map.

    '''
    # Extract metadata
    channel_ids_abs = np.load(Path(dp, 'channel_map.npy'), mmap_mode='r').squeeze()
    channel_ids_rel = np.arange(channel_ids_abs.shape[0])
    params={}; par=imp.load_source('params', str(Path(dp,'params.py')))
    for p in dir(par):
        exec("if '__'not in '{}': params['{}']=par.{}".format(p, p, p))
    params['filter_order'] = 3 if bpfilter else None #if params['hp_filtered'] is False else 3
    dat_path=Path(dp, params['dat_path'])

    # Compute traces from binary file
    item_size = np.dtype(params['dtype']).itemsize
    n_samples = (op.getsize(dat_path) - params['offset']) // (item_size * params['n_channels_dat'])
    trc = np.memmap(dat_path, dtype=params['dtype'], shape=(n_samples, params['n_channels_dat']),
                         offset=params['offset'])
    traces = ConcatenatedArrays([trc], channel_ids_abs, scaling=None) # Here, the ABSOLUTE channel indices must be used to extract the correct channels

    # Get spike times subset
    spike_samples = np.load(Path(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    spike_ids_subset=get_ids_subset(dp, unit, n_waveforms, wvf_batch_size, subset_selection, ignore_nwvf)

    # Extract waveforms i.e. bits of traces at spike times subset
    waveform_loader=WaveformLoader(traces=traces,
                                      spike_samples=spike_samples,
                                      n_samples_waveforms=t_waveforms,
                                      filter_order=params['filter_order'],
                                      sample_rate=params['sample_rate'])

    waveforms = waveform_loader.get(spike_ids_subset, channel_ids_rel, whiten, med_sub) # Here, the relative indices must be used since only n_channel traces were extracted.


    # Correct voltage scaling
    meta=read_spikeglx_meta(dp, 'ap')
    waveforms*=meta['scale_factor']

    # Common average referencing: substract median for each channel, then median for each time point
    # medians_chan=np.median(traces[:1000000, :], 0).reshape((1,1)+(waveforms.shape[2],)) # across time for each channel
    # medians_chan=np.repeat(medians_chan, waveforms.shape[0], axis=0)
    # medians_chan=np.repeat(medians_chan, waveforms.shape[1], axis=1)

    wvf_baselines=np.append(waveforms[:,:int(waveforms.shape[1]*2./5),:], waveforms[:,int(waveforms.shape[1]*3./5):,:], axis=1)
    medians_chan = np.median(wvf_baselines, axis=1).reshape((waveforms.shape[0], 1, waveforms.shape[2]))
    medians_chan = np.repeat(medians_chan, waveforms.shape[1], axis=1)
    waveforms-=medians_chan

    medians_t = np.median(waveforms, axis=2).reshape(waveforms.shape[:2]+(1,)) # across channels for each time point
    medians_t = np.repeat(medians_t, waveforms.shape[2], axis=2)
    waveforms-=medians_t

    # Re-align waveforms in time?... not yet

    return  waveforms

class WaveformLoader(object):
    """Load waveforms from filtered or unfiltered traces."""

    def __init__(self,
                 traces=None,
                 sample_rate=None,
                 spike_samples=None,
                 filter_order=None,
                 n_samples_waveforms=None,
                 ):

        # Traces.
        if traces is not None:
            self.traces = traces
            self.n_samples_trace, self.n_channels = traces.shape
        else:
            self._traces = None
            self.n_samples_trace = self.n_channels = 0

        assert spike_samples is not None
        self._spike_samples = spike_samples
        self.n_spikes = len(spike_samples)

        # Define filter.
        if filter_order is not None:
            filter_margin = filter_order * 3
            b_filter = bandpass_filter(rate=sample_rate,
                                       low=500.,
                                       high=sample_rate * .475,
                                       order=filter_order,
                                       )
            self._filter = lambda x, axis=0: apply_filter(x, b_filter,
                                                          axis=axis)
        else:
            filter_margin = 0
            self._filter = lambda x, axis=0: x

        # Number of samples to return, can be an int or a
        # tuple (before, after).
        assert n_samples_waveforms is not None
        self.n_samples_before_after = _before_after(n_samples_waveforms)
        self.n_samples_waveforms = sum(self.n_samples_before_after)
        # Number of additional samples to use for filtering.
        self._filter_margin = _before_after(filter_margin)
        # Number of samples in the extracted raw data chunk.
        self._n_samples_extract = (self.n_samples_waveforms +
                                   sum(self._filter_margin))

        self.dtype = np.float32
        self.shape = (self.n_spikes, self._n_samples_extract, self.n_channels)
        self.ndim = 3

    @property
    def traces(self):
        """Raw traces."""
        return self._traces

    @traces.setter
    def traces(self, value):
        self.n_samples_trace, self.n_channels = value.shape
        self._traces = value

    @property
    def spike_samples(self):
        return self._spike_samples

    def _load_at(self, time, channels=None, whiten=False, med_sub=False):
        """Load a waveform at a given time."""
        if channels is None:
            channels = slice(None, None, None)
        time = int(time)
        time_o = time
        ns = self.n_samples_trace
        if not (0 <= time_o < ns):
            raise ValueError("Invalid time {0:d}/{1:d}.".format(time_o, ns))

        pad=1000 if whiten else 100
        nsba1=(self.n_samples_before_after[0]+pad, self.n_samples_before_after[1]+pad)
        slice_extract = _slice(time_o,
                               nsba1,
                               self._filter_margin)
        extract = self._traces[slice_extract].astype(np.float32)

        # Center channels individually
        # offsets = np.median(extract, axis=0)
        # extract-=offsets

        # whiten the waveform
        if whiten:
            extract=whitening(extract.T).T

        extract=extract[pad:-pad, :]

        # median-substract
        if med_sub:
            extract=med_substract(extract.T).T

        # remove masked channels after all processings
        extract=extract[:, channels]

        # Pad the extracted chunk if needed.
        if slice_extract.start <= 0:
            extract = _pad(extract, self._n_samples_extract, 'left')
        elif slice_extract.stop >= ns - 1:
            extract = _pad(extract, self._n_samples_extract, 'right')

        assert extract.shape[0] == self._n_samples_extract
        return extract

    def get(self, spike_ids, channels=None, whiten=False, med_sub=False):
        """Load the waveforms of the specified spikes."""

        if isinstance(spike_ids, slice):
            spike_ids = _range_from_slice(spike_ids,
                                          start=0,
                                          stop=self.n_spikes,
                                          )
        if not hasattr(spike_ids, '__len__'):
            spike_ids = [spike_ids]
        if channels is None:
            channels = slice(None, None, None)
            nc = self.n_channels
        else:
            channels = np.asarray(channels, dtype=np.int32)
            assert np.all(channels < self.n_channels)
            nc = len(channels)

        # Ensure a list of time samples are being requested.
        spike_ids = _as_array(spike_ids)
        n_spikes = len(spike_ids)

        # Initialize the array.
        # NOTE: last dimension is time to simplify things.
        shape = (n_spikes, nc, self._n_samples_extract)
        waveforms = np.zeros(shape, dtype=np.float32)

        # No traces: return null arrays.
        if self.n_samples_trace == 0:
            return np.transpose(waveforms, (0, 2, 1))

        # Load all spikes.
        for i, spike_id in enumerate(spike_ids):
            assert 0 <= spike_id < self.n_spikes
            time = self._spike_samples[spike_id]

            # Extract the waveforms on the unmasked channels.
            try:
                w = self._load_at(time, channels, whiten, med_sub)
            except ValueError as e:  # pragma: no cover
                print("Error while loading waveform: %s", str(e))
                continue

            assert w.shape == (self._n_samples_extract, nc)

            waveforms[i, :, :] = w.T

        # Filter the waveforms.
        waveforms_f = waveforms.reshape((-1, self._n_samples_extract))
        # Only filter the non-zero waveforms.
        unmasked = waveforms_f.max(axis=1) != 0
        waveforms_f[unmasked] = self._filter(waveforms_f[unmasked], axis=1)
        waveforms_f = waveforms_f.reshape((n_spikes, nc,
                                           self._n_samples_extract))

        # Remove the margin.
        margin_before, margin_after = self._filter_margin
        if margin_after > 0:
            assert margin_before >= 0
            waveforms_f = waveforms_f[:, :, margin_before:-margin_after]

        assert waveforms_f.shape == (n_spikes,
                                     nc,
                                     self.n_samples_waveforms,
                                     )

        # NOTE: we transpose before returning the array.
        return np.transpose(waveforms_f, (0, 2, 1))

    def __getitem__(self, spike_ids):
        return self.get(spike_ids)

def _before_after(n_samples):
    """Get the number of samples before and after."""
    if not isinstance(n_samples, (tuple, list)):
        before = n_samples // 2
        after = n_samples - before
    else:
        assert len(n_samples) == 2
        before, after = n_samples
        n_samples = before + after
    assert before >= 0
    assert after >= 0
    assert before + after == n_samples
    return before, after


def _slice(index, n_samples, margin=None):
    """Return a waveform slice."""
    if margin is None:
        margin = (0, 0)
    assert isinstance(n_samples, (tuple, list))
    assert len(n_samples) == 2
    before, after = n_samples
    assert isinstance(margin, (tuple, list))
    assert len(margin) == 2
    margin_before, margin_after = margin
    before += margin_before
    after += margin_after
    index = int(index)
    before = int(before)
    after = int(after)
    return slice(max(0, index - before), index + after, None)
from npyx.gl import get_units, assert_multi, get_ds_ids, get_source_dp_u
from npyx.spk_t import ids
