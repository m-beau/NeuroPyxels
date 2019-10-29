# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os.path as op
import imp

import numpy as np
from scipy import signal

from rtn.npix import spk_t
from rtn.npix.io import ConcatenatedArrays, _pad, _range_from_slice, Selector
from rtn.utils import _as_array

dp='/media/maxime/Npxl_data2/wheel_turning/DK152-153/DK153_190416day1_Probe2_run1'

#%% Concise home made function
def get_wvf(dp, unit, n_samples_waveforms=100, n_channels_around=384, batch_size_waveforms=10):
    
    # Extract metadata
    channel_ids = np.load(op.join(dp, 'channel_map.npy'), mmap_mode='r').squeeze()
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
    traces = ConcatenatedArrays([traces], channel_ids, scaling=None)
    
    # Get spike times
    spike_samples = np.load(op.join(dp, 'spike_times.npy'), mmap_mode='r').squeeze()
    
    # Get sample of spike ids and relevant channel bank of unit
    # def _set_supervisor(self):
    #     # Load the new cluster id.
    #     new_cluster_id = self.context.load('new_cluster_id'). \
    #         get('new_cluster_id', None)
    #     cluster_groups = self.model.get_metadata('group')
    #     supervisor = Supervisor(self.model.spike_clusters,
    #                             similarity=self.similarity,
    #                             cluster_groups=cluster_groups,
    #                             new_cluster_id=new_cluster_id,
    #                             context=self.context,
    #                             )
    # spikes_per_cluster=supervisor.clustering.spikes_per_cluster[cluster_id]
    
    spikes_per_cluster=spk_t.ids(dp, unit)
    spike_ids = Selector(spikes_per_cluster).select_spikes([unit], n_samples_waveforms,batch_size_waveforms)
        
    # Extract waveforms i.e. bits of traces at the right spike times
    waveform_loader=WaveformLoader(traces=traces,
                                      spike_samples=spike_samples,
                                      n_samples_waveforms=n_samples_waveforms,
                                      filter_order=params['filter_order'],
                                      sample_rate=params['sample_rate'],
                                      )
    
    waveforms = waveform_loader.get(spike_ids, channel_ids)
    
    return  waveforms

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