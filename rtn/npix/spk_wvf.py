# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

import os.path as op
import imp

import numpy as np
from math import ceil
from scipy import signal

from rtn.npix.spk_t import ids
from rtn.npix.gl import get_units
from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice
from rtn.utils import _as_array, npa

dp='/media/maxime/Npxl_data2/wheel_turning/DK152-153/DK153_190416day1_Probe2_run1'

#%% Concise home made function

def get_wvf(dp, unit, n_waveforms=200, t_waveforms=82, wvf_subset_selection='regular', wvf_batch_size=10, ampFactor=500, probe_type='3A'):
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
    
    WARNING: waveforms are neither whitened nor common average referenced.
    '''
    # Extract metadata
    channel_ids_abs = np.load(op.join(dp, 'channel_map.npy'), mmap_mode='r').squeeze()
    channel_ids_rel = np.arange(channel_ids_abs.shape[0])
    params={}; par=imp.load_source('params', op.join(dp,'params.py'))
    for p in dir(par):
        exec("if '__'not in '{}': params['{}']=par.{}".format(p, p, p))
    params['filter_order'] = None if params['hp_filtered'] is False else 3
    dat_path=op.join(dp, params['dat_path'])
    
    # Compute traces from binary file
    item_size = np.dtype(params['dtype']).itemsize
    n_samples = (op.getsize(dat_path) - params['offset']) // (item_size * params['n_channels_dat'])
    traces = np.memmap(dat_path, dtype=params['dtype'], shape=(n_samples, params['n_channels_dat']),
                         offset=params['offset'])
    traces = ConcatenatedArrays([traces], channel_ids_abs, scaling=None) # Here, the ABSOLUTE channel indices must be used to extract the correct channels
    
    # Get spike times subset
    spike_samples = np.load(op.join(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    spike_ids_subset=get_ids_subset(dp, unit, n_waveforms, wvf_batch_size, wvf_subset_selection)
    
    # Extract waveforms i.e. bits of traces at spike times subset
    waveform_loader=WaveformLoader(traces=traces,
                                      spike_samples=spike_samples,
                                      n_samples_waveforms=t_waveforms,
                                      filter_order=params['filter_order'],
                                      sample_rate=params['sample_rate'])
    
    waveforms = waveform_loader.get(spike_ids_subset, channel_ids_rel) # Here, the relative indices must be used since only n_channel traces were extracted.
    
    # Correct voltage values
    assert probe_type in ['3A', '3B', '1.0', '2.0', '2.0singleshank']
    if probe_type in ['3A', '3B', '1.0']:
        Vrange=1.2e6
        bits_encoding=10
    elif probe_type in ['2.0', '2.0singleshank']:
        Vrange=1.2e6
        bits_encoding=14
    waveforms*=(Vrange/2**bits_encoding/ampFactor) # (voltageRange/2^10bitsEncoding/amplification_gain, typically 500)
    
    return  waveforms

def get_best_chan(dp, unit):
    waveforms=get_wvf(dp, unit, 1000)
    wvf_m = np.rollaxis(waveforms.mean(0),1)
    best_chan = np.argwhere(np.max(abs(wvf_m),1) == np.max(np.max(abs(wvf_m),1)))
    return best_chan

def get_depthSorted_channels(dp, units=None):
    if ~np.any(units):
        units=get_units(dp)
    
    bestChs=[]
    for u in units:
        bestChs.append(get_best_chan(dp, u))
    depthIdx = np.argsort(npa(bestChs))[::-1] # From surface (high ch) to DCN (low ch)
    
    units = units[depthIdx]
    channels = bestChs[depthIdx]

    return units, channels


def get_chDis(dp, ch1, ch2):
    '''dp: datapath to dataset
    ch1, ch2: channel indices (1 to 384)
    returns distance in um.'''
    assert 1<=ch1<=384
    assert 1<=ch2<=384
    ch_pos = np.load(op.join(dp,'channel_positions.npy')) # positions in um	
    ch_pos1=ch_pos[ch1-1] # convert from ch number to ch relative index	
    ch_pos2=ch_pos[ch2-1] # convert from ch number to ch relative index	
    chDis=np.sqrt((ch_pos1[0]-ch_pos2[0])**2+(ch_pos1[1]-ch_pos2[1])**2)
    return chDis



#%% get_wvf utilities
    
def get_ids_subset(dp, unit, n_waveforms, batch_size_waveforms, subset):
    assert subset in ['regular', 'random']
    if n_waveforms in (None, 0):
        ids_subset = ids(dp, unit)
    else:
        assert n_waveforms > 0
        spike_ids = ids(dp, unit)
        if subset == 'regular':
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
        elif subset == 'random' and len(spike_ids) > n_waveforms:
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