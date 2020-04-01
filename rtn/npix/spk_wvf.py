# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

import os
import os.path as op; opj=op.join
from pathlib import Path

import imp
from ast import literal_eval as ale
import numpy as np
from math import ceil
from scipy import signal

from rtn.npix.spk_t import ids
from rtn.npix.gl import get_units, get_prophyler_source
from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice, read_spikeglx_meta, chan_map
from rtn.utils import _as_array, npa

#%% Concise home made function

def wvf(dp, u, n_waveforms=100, t_waveforms=82, subset_selection='regular', wvf_batch_size=10, sav=True, prnt=False, again=False):
    '''
    ********
    routine from rtn.npix.spk_wvf
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
    dp, u = get_prophyler_source(dp, u)

    dprm = Path(dp,'routinesMemory')
    fn="wvf{}_{}-{}_{}.npy".format(u, n_waveforms, t_waveforms, str(subset_selection)[0:10].replace(' ', ''))
    if not os.path.isdir(dprm):
        os.makedirs(dprm)
    if os.path.exists(Path(dprm,fn)) and not again:
        if prnt: print("File {} found in routines memory.".format(fn))
        waveforms = np.load(Path(dprm,fn))

    # if not, compute it
    else:
        waveforms = get_waveform(dp, u, n_waveforms, t_waveforms, subset_selection, wvf_batch_size)
        # Save it
        if sav:
            np.save(Path(dprm,fn), waveforms)

    return waveforms


def get_waveform(dp, unit, n_waveforms=100, t_waveforms=82, subset_selection='regular', wvf_batch_size=10):
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
    params={}; par=imp.load_source('params', opj(dp,'params.py'))
    for p in dir(par):
        exec("if '__'not in '{}': params['{}']=par.{}".format(p, p, p))
    params['filter_order'] = None if params['hp_filtered'] is False else 3
    dat_path=Path(dp, params['dat_path'])
    
    # Compute traces from binary file
    item_size = np.dtype(params['dtype']).itemsize
    n_samples = (op.getsize(dat_path) - params['offset']) // (item_size * params['n_channels_dat'])
    trc = np.memmap(dat_path, dtype=params['dtype'], shape=(n_samples, params['n_channels_dat']),
                         offset=params['offset'])
    traces = ConcatenatedArrays([trc], channel_ids_abs, scaling=None) # Here, the ABSOLUTE channel indices must be used to extract the correct channels

    # Get spike times subset
    spike_samples = np.load(Path(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    spike_ids_subset=get_ids_subset(dp, unit, n_waveforms, wvf_batch_size, subset_selection)
    
    # Extract waveforms i.e. bits of traces at spike times subset
    waveform_loader=WaveformLoader(traces=traces,
                                      spike_samples=spike_samples,
                                      n_samples_waveforms=t_waveforms,
                                      filter_order=params['filter_order'],
                                      sample_rate=params['sample_rate'])
    
    waveforms = waveform_loader.get(spike_ids_subset, channel_ids_rel) # Here, the relative indices must be used since only n_channel traces were extracted.
    
    
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


def get_peak_chan(dp, unit):
    '''
    Parameters:
        - datapath, string
        - unit, integer or string
    Returns:
        - best_channel, integer indexing the channel
          where the unit averaged raw waveform (n=100 spanning the whole recording)
          has the largest peak to trough amplitude.
    '''
    dp, unit = get_prophyler_source(dp, unit)
    cm=chan_map(dp, probe_version='local')
    waveforms=wvf(dp, unit, 200)
    wvf_m = np.mean(waveforms, axis=0)
    max_min_wvf=np.max(wvf_m,0)-np.min(wvf_m,0)
    peak_chan = int(np.nonzero(np.max(max_min_wvf)==max_min_wvf)[0])
    return cm[:,0][peak_chan]

def get_depthSort_peakChans(dp, units=[], quality='all'):
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
    
    if not any(units):
        # If no units, load them all from dataset
        # and prepare to save the FULL array of peak channels at the end
        units=get_units(dp, quality=quality)
        pc_fname='peak_channels_{}.npy'.format(quality)
        if op.exists(Path(dp, pc_fname)):
            peak_chans=np.load(Path(dp, pc_fname))
            if np.all(np.isin(units, peak_chans[:,0])):
                return peak_chans
            else:
                save=True
        else:
            save=True
    else:
        # If units are fed, try to load the peak channels from the saved FULL array of peak channels
        # else, just calculate the peak channels for the relevant units
        units=npa(units).flatten()
        if op.exists(Path(dp, 'peak_channels_all.npy')):
            peak_chans=np.load(Path(dp, 'peak_channels_all.npy'))
            if np.all(np.isin(units, peak_chans[:,0])):
                units_mask=np.isin(peak_chans[:,0], units)
                return peak_chans[units_mask]
    
    if type(units[0]) in [str, np.str_]:
        datasets={}
        for iu, u in enumerate(units):
            ds_i, u = u.split('_'); ds_i, u = ale(ds_i), ale(u)
            if ds_i not in datasets.keys(): datasets[ds_i]=1
            else: datasets[ds_i]+=1
        peak_chans_dic={}
        for ds_i, Nu in datasets.items():
            peak_chans_dic[ds_i]=npa(zeros=(Nu,2),dtype='<U6')
        for iu, u in enumerate(units):
            print("Getting peak channel of unit {}...".format(u))
            ds_i = ale(u.split('_')[0])
            if ds_i>=1: iu=iu%datasets[ds_i-1]
            peak_chans_dic[ds_i][iu,0] = u
            peak_chans_dic[ds_i][iu,1] = int(get_peak_chan(dp, u))
        peak_chans=npa(zeros=(0,2),dtype='<U6')
        for ds_i in sorted(datasets.keys()):
            depthIdx = np.argsort(peak_chans_dic[ds_i].astype('int64')[:,1])[::-1]
            peak_chans_dic[ds_i]=peak_chans_dic[ds_i][depthIdx]
            peak_chans=np.vstack([peak_chans, peak_chans_dic[ds_i]])

    else:
        peak_chans=npa(zeros=(len(units),2),dtype='int64')
        for iu, u in enumerate(units):
            print("Getting peak channel of unit {}...".format(u))
            peak_chans[iu,0] = u
            peak_chans[iu,1] = int(get_peak_chan(dp, u))
        depthIdx = np.argsort(peak_chans[:,1])[::-1] # From surface (high ch) to DCN (low ch)
        peak_chans=peak_chans[depthIdx]
    
    if save:
        np.save(Path(dp, pc_fname), peak_chans)
    
    return peak_chans # units, channels


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

def templates(dp, u):
    '''
    ********
    routine from rtn.npix.spk_wvf
    Extracts the template used by kilosort to cluster this unit.
    ********
    
    Parameters:
        - dp: string, datapath to kilosort output
        - unit: int, unit index
    
    Returns:
        temaplate: numpy array of shape (n_templates, 82, n_channels) where n_channels is defined by the channel map and n_templates by how many merges were done.
    
    '''
    dp, u = get_prophyler_source(dp, u)

    IDs=ids(dp,u)
    #mean_amp=np.mean(np.load(Path(dp,'amplitudes.npy'))[IDs])
    template_ids=np.unique(np.load(Path(dp,'spike_templates.npy'))[IDs])
    templates = np.load(Path(dp, 'templates.npy'))[template_ids]#*mean_amp
                
    return templates

#%% get_wvf utilities
    
def get_ids_subset(dp, unit, n_waveforms, batch_size_waveforms, subset_selection):
    if type(subset_selection) not in [str, np.str_]:
        ids_subset = ids(dp, unit, subset_selection=subset_selection)
        print('In subset {}, {} waveforms were found (parameter n_waveforms={} ignored).'.format(subset_selection, len(ids_subset), n_waveforms))
    else:
        assert subset_selection in ['regular', 'random', 'all']
        if n_waveforms in (None, 0) or subset_selection=='all':
            ids_subset = ids(dp, unit)
        else:
            assert n_waveforms > 0
            spike_ids = ids(dp, unit)
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
        if filter_order:
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

    def _load_at(self, time, channels=None):
        """Load a waveform at a given time."""
        if channels is None:
            channels = slice(None, None, None)
        time = int(time)
        time_o = time
        ns = self.n_samples_trace
        if not (0 <= time_o < ns):
            raise ValueError("Invalid time {0:d}/{1:d}.".format(time_o, ns))
        slice_extract = _slice(time_o,
                               self.n_samples_before_after,
                               self._filter_margin)
        extract = self._traces[slice_extract][:, channels].astype(np.float32)

        # Pad the extracted chunk if needed.
        if slice_extract.start <= 0:
            extract = _pad(extract, self._n_samples_extract, 'left')
        elif slice_extract.stop >= ns - 1:
            extract = _pad(extract, self._n_samples_extract, 'right')

        assert extract.shape[0] == self._n_samples_extract
        return extract

    def get(self, spike_ids, channels=None):
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
                w = self._load_at(time, channels)
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

def bandpass_filter(rate=None, low=None, high=None, order=None):
    """Butterworth bandpass filter."""
    assert low < high
    assert order >= 1
    return signal.butter(order,
                         (low / (rate / 2.), high / (rate / 2.)),
                         'pass')

def apply_filter(x, filter=None, axis=0):
    """Apply a filter to an array."""
    x = _as_array(x)
    if x.shape[axis] == 0:
        return x
    b, a = filter
    return signal.filtfilt(b, a, x, axis=axis)
