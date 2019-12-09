# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:30:50 2019

@author: Maxime Beau, Hausser lab, University College London

Input/output utilitaries to deal with Neuropixels files.
"""

import psutil
import os
from ast import literal_eval as ale
import os.path as op; opj=op.join

import numpy as np
from math import floor

from rtn.utils import npa

#%% Extract metadata and sync channel

def read_spikeglx_meta(dp, subtype='ap'):
    '''
    Read spikeGLX metadata file.
    '''
    metafile=''
    for file in os.listdir(dp):
        if file.endswith(".{}.meta".format(subtype)):
            metafile=opj(dp, file)
            break
    if metafile=='':
        raise FileNotFoundError('*.{}.meta not found in directory. Aborting.'.format(subtype))
            
    with open(metafile, 'r') as f:
        meta = {}
        for ln in f.readlines():
            tmp = ln.split('=')
            k, val = tmp
            k = k.strip()
            val = val.strip('\r\n')
            if '~' in k:
                meta[k] = val.strip('(').strip(')').split(')(')
            else:
                try:  # is it numeric?
                    meta[k] = float(val)
                except:
                    try:
                        meta[k] = float(val)
                    except:
                        meta[k] = val
    # Set the sample rate depending on the recording mode

    meta['sRateHz'] = meta[meta['typeThis'][:2] + 'SampRate']
    if meta['typeThis'] == 'imec':
        meta['sRateHz'] = meta['imSampRate']
    
    probe_versions = {'imProbeOpt':{3.0:'3A'},
               'imDatPrb_type':{0:'1.0_staggered',
                                21:'2.0_singleshank',
                                24:'2.0_fourshanked'}}
    meta['probe_version']=probe_versions['imProbeOpt'][meta['imProbeOpt']] if 'imProbeOpt' in meta.keys() else probe_versions['imDatPrb_type'][meta['imDatPrb_type']] if 'imDatPrb_type' in meta.keys() else 'N/A'
    assert meta['probe_version'] in ['3A', '1.0_staggered', '1.0_aligned', '2.0_singleshank', '2.0_fourshanked']
    
    Vrange=(meta['imAiRangeMax']-meta['imAiRangeMin'])*1e6
    if meta['probe_version'] in ['3A', '1.0_staggered', '1.0_aligned']:
        assert Vrange==1.2e6
        bits_encoding=10
        ampFactor=ale(meta['~imroTbl'][1].split(' ')[3])
        assert ampFactor==500
    elif meta['probe_version'] in ['2.0_singleshank', '2.0_fourshanked']:
        assert Vrange==1e6
        bits_encoding=14
        ampFactor=80
    meta['scale_factor']=(Vrange/2**bits_encoding/ampFactor)
    
    return meta

def chan_map(dp=None, y_orig='surface', probe_version=None):
    
    assert y_orig in ['surface', 'tip']
    if probe_version is None: probe_version=read_spikeglx_meta(dp)['probe_version']
    
    if probe_version in probe_version in ['3A', '1.0_staggered']:
        Nchan=384
        cm_el = npa([[  27,   0],
                           [  59,   0],
                           [  11,   20],
                           [  43,   20]])
        vert=npa([[  0,   40],
                  [  0,   40],
                  [  0,   40],
                  [  0,   40]])
        
        cm=cm_el.copy()
        for i in range(int(Nchan/cm_el.shape[0])-1):
            cm = np.vstack((cm, cm_el+vert*(i+1)))
        cm=np.hstack([np.arange(Nchan).reshape(Nchan,1), cm])
    
    elif probe_version=='1.0_aligned':
        Nchan=384
        cm_el = npa([[  11,   0],
                           [  43,   0]])
        vert=npa([[  0,   20],
                  [  0,   20]])
        
        cm=cm_el.copy()
        for i in range(int(Nchan/cm_el.shape[0])-1):
            cm = np.vstack((cm, cm_el+vert*(i+1)))
        cm=np.hstack([np.arange(Nchan).reshape(Nchan,1), cm])
        
    elif probe_version=='2.0_singleshank':
        Nchan=384
        cm_el = npa([[  0,   0],
                           [  32,   0]])
        vert=npa([[  0,   15],
                  [  0,   15]])
        
        cm=cm_el.copy()
        for i in range(int(Nchan/cm_el.shape[0])-1):
            cm = np.vstack((cm, cm_el+vert*(i+1)))
        cm=np.hstack([np.arange(Nchan).reshape(Nchan,1), cm])
    
    elif probe_version=='local':
        if dp is None:
            raise ValueError("dp argument is not provided - when channel map is \
                             atypical and probe_version is hence called 'local', \
                             the datapath needs to be provided to load the channel map.")
        c_ind=np.load(op.join(dp, 'channel_map.npy'));cp=np.load(op.join(dp, 'channel_positions.npy'));
        cm=npa(np.hstack([c_ind, cp]), dtype=np.int32)
        
    if y_orig=='surface':
        cm[:,1:]=cm[:,1:][::-1]
        
    return cm

#%% Binary file I/O

def unpackbits(x,num_bits = 16):
    '''
    unpacks numbers in bits.
    '''
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

def get_npix_sync(dp, output_binary = False, sourcefile='lf'):
    '''Unpacks neuropixels phase external input data
    events = unpack_npix3a_sync(trigger_data_channel)
        Inputs:
            syncdat               : trigger data channel to unpack (pass the last channel of the memory mapped file)
            srate (1)             : sampling rate of the data; to convert to time - meta['imSampRate']
            output_binary (False) : outputs the unpacked signal
        Outputs
            events        : dictionary of events. the keys are the channel number, the items the sample times of the events.
    
        Usage:
    Load and get trigger times in seconds:
        dp='path/to/binary'
        onsets,offsets = unpack_npix_sync(dp);
    Plot events:
        plt.figure(figsize = [10,4])
        for ichan,times in onsets.items():
            plt.vlines(times,ichan,ichan+.8,linewidth = 0.5)
        plt.ylabel('Sync channel number'); plt.xlabel('time (s)')
    '''
    assert sourcefile in ['ap', 'lf']
    fname=''
    onsets={}
    offsets={}
    sync_dp=opj(dp, 'sync_chan')
    
    # Tries to directly generate and output onsets and offsets
    if op.exists(sync_dp) and not output_binary:
        print('sync channel extraction directory found: {}'.format(sync_dp))
        for file in os.listdir(sync_dp):
            if file.endswith("on.npy"):
                fname=file[:-13]
                for file in os.listdir(sync_dp):
                    if file.endswith("on.npy"):
                        file_i = ale(file[-7])
                        onsets[file_i]=np.load(opj(sync_dp,file))
                    elif file.endswith("of.npy"):
                        file_i = ale(file[-7])
                        offsets[file_i]=np.load(opj(sync_dp,file))
                    
                return onsets, offsets

    # Generates binary and eventually outputs it
    if op.exists(sync_dp):
        print("No file ending in 'on.npy' found in sync_chan directory: extracting sync channel from binary.".format(sync_dp))
        for file in os.listdir(sync_dp):
            if file.endswith("_sync.npz"):
                fname=file[:-9]
                sync_fname=fname+'_sync'
                binary=np.load(opj(sync_dp, sync_fname+'.npz'))
                binary=binary[dir(binary.f)[0]].astype(np.int8)
                meta=read_spikeglx_meta(dp, fname[-2:])
                break
    else: os.mkdir(sync_dp)
    
    if fname=='':
        for file in os.listdir(dp):
            if file.endswith(".lf.bin"):
                if sourcefile=='ap': break
                fname=file[:-4]
                break
        if fname=='':
            for file in os.listdir(dp):
                if file.endswith(".ap.bin"):
                    fname=file[:-4]
                    print('{}.lf.bin not found in directory - .ap.bin used instead: extracting sync channel will be slow.'.format(fname))
                    break
        if fname=='':
            raise FileNotFoundError('No binary file found in {}!! Aborting.'.format(dp))
        
        sync_fname=fname+'_sync'
        meta=read_spikeglx_meta(dp, fname[-2:])
        nchan=int(meta['nSavedChans'])
        #all_channels = np.array([ale(ch.split(':')[-1]) for ch in meta['~snsChanMap'][1:-1]], dtype=np.int16);
        #syncChan=nchan-ale(meta['acqApLfSy'].split(',')[-1])
    
        dt=np.dtype(np.int16)
        nsamples = os.path.getsize(opj(dp, fname+'.bin')) / (nchan * dt.itemsize)
        syncdat=np.memmap(opj(dp, fname+'.bin'),
                        mode='r',
                        dtype=dt,
                        shape=(int(nsamples), int(nchan)))[:,-1]
        
        
        print('Unpacking {}...'.format(fname+'.bin'))
        binary = unpackbits(syncdat.flatten(),16).astype(np.int8)
        np.savez_compressed(opj(sync_dp, sync_fname+'.npz'), binary)

    if output_binary:
        return binary
    
    # Generates onsets and offsets from binary
    mult = 1
    srate = meta['sRateHz']
    sync_idx_onset = np.where(mult*np.diff(binary, axis = 0)>0)
    sync_idx_offset = np.where(mult*np.diff(binary, axis = 0)<0)
    for ichan in np.unique(sync_idx_onset[1]):
        ons = sync_idx_onset[0][
              sync_idx_onset[1] == ichan]/srate
        onsets[ichan] = ons
        np.save(opj(sync_dp, sync_fname+'{}on.npy'.format(ichan)), ons)
    for ichan in np.unique(sync_idx_offset[1]):
        ofs = sync_idx_offset[0][
              sync_idx_offset[1] == ichan]/srate
        offsets[ichan] = ofs
        np.save(opj(sync_dp, sync_fname+'{}of.npy'.format(ichan)), ofs)
    
    return onsets,offsets


def extract_rawChunk(dp, times, channels=np.arange(384), subtype='ap', save=0, ret=1, whiten=0):
    '''Function to extract a chunk of raw data on a given range of channels on a given time window.
    ## PARAMETERS
    - dp: datapath to folder with binary path (files must ends in .bin, typically ap.bin)
    - times: list of boundaries of the time window, in seconds [t1, t2].
    - channels (default: np.arange(384)): list of channels of interest, in 0 indexed integers [c1, c2, c3...]
    - fs (default 30000): sampling rate
    - ampFactor (default 500): gain factor of recording (can be different for LFP and AP, check SpikeGLX/OpenEphys)
    - Nchans (default 385): total number of channels on the probe, including sync channel (3A: 385)
    - syncChan: sync channel, 0 indexed(3A: 384)
    - save (default 0): save the raw chunk in the bdp directory as '{bdp}_t1-t2_c1-c2.npy'
    
    ## RETURNS
    rawChunk: numpy array of shape ((c2-c1), (t2-t1)*fs).
    rawChunk[0,:] is channel 0; rawChunk[1,:] is channel 1, etc.
    '''
    
    # Find binary file
    assert len(times)==2
    fname=''
    for file in os.listdir(dp):
        if file.endswith(".{}.bin".format(subtype)):
            fname=opj(dp, file)
            break
    if fname=='':
        raise FileNotFoundError('*.{}.bin not found in directory. Aborting.'.format(subtype))
    
    # Extract and format meta data
    meta=read_spikeglx_meta(dp, subtype)
    fs = int(meta['sRateHz'])
    Nchans=int(meta['nSavedChans'])
    bytes_per_sample=2
    
    # Format inputs
    cm=chan_map(dp, probe_version='local'); assert cm.shape[0]<=Nchans-1
    channels=assert_chan_in_dataset(dp, channels) # index out of 384, should remain the same because rc initial shape is 384!
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))
    bn = op.basename(fname) # binary name
    rcn = '{}_t{}-{}_ch{}-{}.npy'.format(bn, times[0], times[1], channels[0], channels[-1]) # raw chunk name
    rcp = opj(dp, rcn)
    
    if os.path.isfile(rcp):
        return np.load(rcp)
    
    # Check that available memory is high enough to load the raw chunk
    vmem=dict(psutil.virtual_memory()._asdict())
    chunkSize = int(fs*Nchans*bytes_per_sample*(times[1]-times[0]))
    print('Used RAM: {0:.1f}% ({1:.2f}GB total).'.format(vmem['used']*100/vmem['total'], vmem['total']/1024/1024/1024))
    print('Chunk size:{0:.3f}MB. Available RAM: {1:.3f}MB.'.format(chunkSize/1024/1024, vmem['available']/1024/1024))
    if chunkSize>0.9*vmem['available']:
        print('WARNING you are trying to load {0:.3f}MB into RAM but have only {1:.3f}MB available.\
              Pick less channels or a smaller time chunk.'.format(chunkSize/1024/1024, vmem['available']/1024/1024))
        return
    
    # Get chunk from binary file
    with open(fname, 'rb') as f_src:
        # each sample for each channel is encoded on 16 bits = 2 bytes: samples*Nchannels*2.
        byte1 = int(t1*Nchans*bytes_per_sample)
        byte2 = int(t2*Nchans*bytes_per_sample)
        bytesRange = byte2-byte1
        
        f_src.seek(byte1)
        
        bData = f_src.read(bytesRange)
    
    # Decode binary data
    assert len(bData)%2==0
    rc = np.frombuffer(bData, dtype=np.int16) # 16bits decoding
    rc = rc.reshape((int(t2-t1), Nchans)).T

    # Whiten data
    if whiten:
        # with open(fname, 'rb') as f_src:
        #     # each sample for each channel is encoded on 16 bits = 2 bytes: samples*Nchannels*2.
        #     byte1 = int(0*Nchans*bytes_per_sample)
        #     byte2 = int(300000*Nchans*bytes_per_sample)
        #     bytesRange = byte2-byte1
            
        #     f_src.seek(byte1)
            
        #     bDataW = f_src.read(bytesRange)
    
        # assert len(bDataW)%2==0
        # rcW = np.frombuffer(bDataW, dtype=np.int16) # 16bits decoding
        # rcW = rcW.reshape((300000, Nchans))
        # rcW = rcW[:, channels] # get the right channels range
        # w = whitening_matrix(rcW)
        # rc_scales=(np.max(rc, 1)-np.min(rc, 1))
        # rcW=np.dot(rc.T,w)
        # rcW=rcW.T
        # rcW_scales=(np.max(rcW, 1)-np.min(rcW, 1))
        # rc=rcW*np.repeat((rc_scales/rcW_scales).reshape(rc.shape[0], 1), rc.shape[1], axis=1)
        
        w = whitening_matrix(rc.T)
        rc_scales=(np.max(rc, 1)-np.min(rc, 1))
        rc=np.dot(rc.T,w)
        rc=rc.T
        rcW_scales=(np.max(rc, 1)-np.min(rc, 1))
        rc=rc*np.repeat((rc_scales/rcW_scales).reshape(rc.shape[0], 1), rc.shape[1], axis=1)
    
    # get the right channels range, AFTER WHITENING
    rc = rc[channels, :] 
    # Scale data
    rc = rc*meta['scale_factor'] # convert into uV

    # Center channels individually
    offsets = np.median(rc, axis=1)
    print('Channels are offset by {}uV on average!'.format(np.mean(offsets)))
    offsets = np.tile(offsets[:,np.newaxis], (1, rc.shape[1]))
    rc-=offsets
    
    if save: # sync chan saved in extract_syncChan
        np.save(rcp, rc)
    
    if ret:
        return rc
    else:
        return

def assert_chan_in_dataset(dp, channels):
    cm=chan_map(dp, probe_version='local')
    if not np.all(np.isin(channels, cm[:,0])):
        print("WARNING Kilosort excluded some channels that you provided for analysis \
    because they did not display enough threshold crossings! Jumping channels:{}\
    ".format(channels[~np.isin(channels, cm[:,0])]))
    channels=channels[np.isin(channels, cm[:,0])]
    return channels
#%% Compute stuff to preprocess data
        
def whitening_matrix(x, fudge=1e-18):
    """
    wmat = whitening_matrix(dat, fudge=1e-18)
    Compute the whitening matrix.
        - dat is a matrix nsamples x nchannels
    Apply using np.dot(dat,wmat)
    Adapted from phy
    """
    assert x.ndim == 2
    ns, nc = x.shape
    x_cov = np.cov(x, rowvar=0)
    assert x_cov.shape == (nc, nc)
    d, v = np.linalg.eigh(x_cov)
    d = np.diag(1. / np.sqrt(d + fudge))
    w = np.dot(np.dot(v, d), v.T)
    return w

#%% I/O array functions from phy
    
def _start_stop(item):
    """Find the start and stop indices of a __getitem__ item.

    This is used only by ConcatenatedArrays.

    Only two cases are supported currently:

    * Single integer.
    * Contiguous slice in the first dimension only.

    """
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, slice):
        # Slice.
        if item.step not in (None, 1):
            raise NotImplementedError()
        return item.start, item.stop
    elif isinstance(item, (list, np.ndarray)):
        # List or array of indices.
        return np.min(item), np.max(item)
    else:
        # Integer.
        return item, item + 1

def _fill_index(arr, item):
    if isinstance(item, tuple):
        item = (slice(None, None, None),) + item[1:]
        return arr[item]
    else:
        return arr

class ConcatenatedArrays(object):
    """This object represents a concatenation of several memory-mapped
    arrays. Coming from phy.io.array.py"""
    def __init__(self, arrs, cols=None, scaling=None):
        assert isinstance(arrs, list)
        self.arrs = arrs
        # Reordering of the columns.
        self.cols = cols
        self.offsets = np.concatenate([[0], np.cumsum([arr.shape[0]
                                                       for arr in arrs])],
                                      axis=0)
        self.dtype = arrs[0].dtype if arrs else None
        self.scaling = scaling

    @property
    def shape(self):
        if self.arrs[0].ndim == 1:
            return (self.offsets[-1],)
        ncols = (len(self.cols) if self.cols is not None
                 else self.arrs[0].shape[1])
        return (self.offsets[-1], ncols)

    def _get_recording(self, index):
        """Return the recording that contains a given index."""
        assert index >= 0
        recs = np.nonzero((index - self.offsets[:-1]) >= 0)[0]
        if len(recs) == 0:  # pragma: no cover
            # If the index is greater than the total size,
            # return the last recording.
            return len(self.arrs) - 1
        # Return the last recording such that the index is greater than
        # its offset.
        return recs[-1]

    def _get(self, item):
        cols = self.cols if self.cols is not None else slice(None, None, None)
        # Get the start and stop indices of the requested item.
        start, stop = _start_stop(item)
        # Return the concatenation of all arrays.
        if start is None and stop is None:
            return np.concatenate(self.arrs, axis=0)[..., cols]
        if start is None:
            start = 0
        if stop is None:
            stop = self.offsets[-1]
        if stop < 0:
            stop = self.offsets[-1] + stop
        # Get the recording indices of the first and last item.
        rec_start = self._get_recording(start)
        rec_stop = self._get_recording(stop)
        assert 0 <= rec_start <= rec_stop < len(self.arrs)
        # Find the start and stop relative to the arrays.
        start_rel = start - self.offsets[rec_start]
        stop_rel = stop - self.offsets[rec_stop]
        # Single array case.
        if rec_start == rec_stop:
            # Apply the rest of the index.
            out = _fill_index(self.arrs[rec_start][start_rel:stop_rel], item)
            out = out[..., cols]
            return out
        chunk_start = self.arrs[rec_start][start_rel:]
        chunk_stop = self.arrs[rec_stop][:stop_rel]
        # Concatenate all chunks.
        l = [chunk_start]
        if rec_stop - rec_start >= 2:
            print("Loading a full virtual array: this might be slow "
                        "and something might be wrong.")
            l += [self.arrs[r][...] for r in range(rec_start + 1,
                                                   rec_stop)]
        l += [chunk_stop]
        # Apply the rest of the index.
        return _fill_index(np.concatenate(l, axis=0), item)[..., cols]

    def __getitem__(self, item):
        out = self._get(item)
        assert out is not None
        if self.scaling is not None and self.scaling != 1:
            out = out * self.scaling
        return out

    def __len__(self):
        return self.shape[0]

def _pad(arr, n, dir='right'):
    """Pad an array with zeros along the first axis.

    Parameters
    ----------

    n : int
        Size of the returned array in the first axis.
    dir : str
        Direction of the padding. Must be one 'left' or 'right'.

    """
    assert dir in ('left', 'right')
    if n < 0:
        raise ValueError("'n' must be positive: {0}.".format(n))
    elif n == 0:
        return np.zeros((0,) + arr.shape[1:], dtype=arr.dtype)
    n_arr = arr.shape[0]
    shape = (n,) + arr.shape[1:]
    if n_arr == n:
        assert arr.shape == shape
        return arr
    elif n_arr < n:
        out = np.zeros(shape, dtype=arr.dtype)
        if dir == 'left':
            out[-n_arr:, ...] = arr
        elif dir == 'right':
            out[:n_arr, ...] = arr
        assert out.shape == shape
        return out
    else:
        if dir == 'left':
            out = arr[-n:, ...]
        elif dir == 'right':
            out = arr[:n, ...]
        assert out.shape == shape
        return out
    
def _range_from_slice(myslice, start=None, stop=None, step=None, length=None):
    """Convert a slice to an array of integers."""
    assert isinstance(myslice, slice)
    # Find 'step'.
    step = myslice.step if myslice.step is not None else step
    if step is None:
        step = 1
    # Find 'start'.
    start = myslice.start if myslice.start is not None else start
    if start is None:
        start = 0
    # Find 'stop' as a function of length if 'stop' is unspecified.
    stop = myslice.stop if myslice.stop is not None else stop
    if length is not None:
        stop_inferred = floor(start + step * length)
        if stop is not None and stop < stop_inferred:
            raise ValueError("'stop' ({stop}) and ".format(stop=stop) +
                             "'length' ({length}) ".format(length=length) +
                             "are not compatible.")
        stop = stop_inferred
    if stop is None and length is None:
        raise ValueError("'stop' and 'length' cannot be both unspecified.")
    myrange = np.arange(start, stop, step)
    # Check the length if it was specified.
    if length is not None:
        assert len(myrange) == length
    return myrange