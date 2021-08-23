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
from pathlib import Path

import numpy as np
from math import floor
from scipy import signal

from npyx.utils import npa


#%% IO utilities

def list_files(directory, extension, full_path=False):
    directory=str(directory)
    files = [f for f in os.listdir(directory) if f.endswith('.' + extension)]
    files.sort()
    if full_path:
        return ['/'.join([directory,f]) for f in files]
    return files

#%% Extract metadata and sync channel

def read_spikeglx_meta(dp, subtype='ap'):
    '''
    Read spikeGLX metadata file.
    '''
    if assert_multi(dp):
        dp=get_ds_table(dp)['dp'][0]
        #print(f'Multidataset detected - spikeGLX metadata taken from 1st dataset ({dp}).')
    assert subtype in ['ap', 'lf']
    metafile=''
    for file in os.listdir(dp):
        if file.endswith(".{}.meta".format(subtype)):
            metafile=Path(dp, file)
            break
    if metafile=='':
        raise FileNotFoundError('*.{}.meta not found in directory. Aborting.'.format(subtype))

    with open(metafile, 'r') as f:
        meta = {}
        for ln in f.readlines():
            tmp = ln.split('=')
            k, val = tmp[0], ''.join(tmp[1:])
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

    if probe_version in ['3A', '1.0_staggered']:
        Nchan=384
        cm_el = npa([[  43,   0],
                           [  11,   0],
                           [  59,   20],
                           [  27,   20]])
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
        cm_el = npa([[  43,   0],
                     [  11,   0]])
        vert=npa([[  0,   20],
                  [  0,   20]])

        cm=cm_el.copy()
        for i in range(int(Nchan/cm_el.shape[0])-1):
            cm = np.vstack((cm, cm_el+vert*(i+1)))
        cm=np.hstack([np.arange(Nchan).reshape(Nchan,1), cm])

    elif probe_version=='2.0_singleshank':
        Nchan=384
        cm_el = npa([[  32,   0],
                     [  0,   0]])
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
        c_ind=np.load(op.join(dp, 'channel_map.npy'));cp=np.load(op.join(dp, 'channel_positions.npy'))
        cm=npa(np.hstack([c_ind.reshape(max(c_ind.shape),1), cp]), dtype=np.int32)

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

def get_npix_sync(dp, output_binary = False, sourcefile='ap', unit='seconds'):
    '''Unpacks neuropixels phase external input data
    events = unpack_npix3a_sync(trigger_data_channel)
        Inputs:
            dp               : trigger data channel to unpack (pass the last channel of the memory mapped file)
            output_binary (False) : outputs the unpacked signal
            sourcefile            : whether to use .ap or .lf file (neuropixxels 1.0)
            unit     ['seconds','samples'] : returns ons and ofs in either seconds or samples
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

    if assert_multi(dp):
        ds_table = get_ds_table(dp)
        dp=get_ds_table(dp)['dp'][0]
        print(f'Loading npix sync channel from a merged dataset - assuming temporal reference frame of dataset 0:\n{dp}')

    assert sourcefile in ['ap', 'lf']
    assert unit in ['seconds', 'samples']
    fname=''
    onsets={}
    offsets={}
    sync_dp=Path(dp, 'sync_chan')

    # Tries to load pre-saved onsets and offsets
    if op.exists(sync_dp) and not output_binary:
        print(f'sync channel extraction directory found: {sync_dp}')
        for file in os.listdir(sync_dp):
            if file.endswith("on_samples.npy"):
                sourcefile_loaded=file.split('.')[-2][:2]
                if sourcefile_loaded==sourcefile: # if samples are at the instructed sampling rate i.e. lf (2500) or ap (30000)!
                    print(f'sync channel onsets extracted from {sourcefile_loaded} file found and loaded.')
                    srate=read_spikeglx_meta(dp, sourcefile_loaded)['sRateHz'] if unit=='seconds' else 1
                    file_i = ale(file[-15])
                    onsets[file_i]=np.load(Path(sync_dp,file))/srate
                    offsets[file_i]=np.load(Path(sync_dp,file[:-13]+'f'+file[-12:]))/srate

        return onsets, offsets

    # Tries to load pre-saved compressed binary
    if op.exists(sync_dp):
        print(f"No file ending in 'on_samples.npy' with the right sampling rate ({sourcefile}) found in sync_chan directory: extracting sync channel from binary.")
        for file in os.listdir(sync_dp):
            if file.endswith(f"{sourcefile}_sync.npz"):
                fname=file[:-9]
                sync_fname=fname+'_sync'
                binary=np.load(Path(sync_dp, sync_fname+'.npz'))
                binary=binary[dir(binary.f)[0]].astype(np.int8)
                meta=read_spikeglx_meta(dp, fname[-2:])
                break
    else: os.mkdir(sync_dp)

    # If still no file name, memorymaps binary directly
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
                    if sourcefile=='lf': print('{}.lf.bin not found in directory - .ap.bin used instead: extracting sync channel will be slow.'.format(fname))
                    break
        if fname=='':
            raise FileNotFoundError('No binary file found in {}!! Aborting.'.format(dp))

        sync_fname=fname+'_sync'
        meta=read_spikeglx_meta(dp, fname[-2:])
        nchan=int(meta['nSavedChans'])
        #all_channels = np.array([ale(ch.split(':')[-1]) for ch in meta['~snsChanMap'][1:-1]], dtype=np.int16);
        #syncChan=nchan-ale(meta['acqApLfSy'].split(',')[-1])

        dt=np.dtype(np.int16)
        nsamples = os.path.getsize(Path(dp, fname+'.bin')) / (nchan * dt.itemsize)
        syncdat=np.memmap(Path(dp, fname+'.bin'),
                        mode='r',
                        dtype=dt,
                        shape=(int(nsamples), int(nchan)))[:,-1]


        print('Unpacking {}...'.format(fname+'.bin'))
        binary = unpackbits(syncdat.flatten(),16).astype(np.int8)
        np.savez_compressed(Path(sync_dp, sync_fname+'.npz'), binary)

    if output_binary:
        return binary

    # Generates onsets and offsets from binary
    mult = 1
    sync_idx_onset = np.where(mult*np.diff(binary, axis = 0)>0)
    sync_idx_offset = np.where(mult*np.diff(binary, axis = 0)<0)
    for ichan in np.unique(sync_idx_onset[1]):
        ons = sync_idx_onset[0][
              sync_idx_onset[1] == ichan]
        onsets[ichan] = ons
        np.save(Path(sync_dp, sync_fname+'{}on_samples.npy'.format(ichan)), ons)
    for ichan in np.unique(sync_idx_offset[1]):
        ofs = sync_idx_offset[0][
              sync_idx_offset[1] == ichan]
        offsets[ichan] = ofs
        np.save(Path(sync_dp, sync_fname+'{}of_samples.npy'.format(ichan)), ofs)

    srate = meta['sRateHz'] if unit=='seconds' else 1
    onsets={ok:ov/srate for ok, ov in onsets.items()}
    offsets={ok:ov/srate for ok, ov in offsets.items()}

    return onsets,offsets


def extract_rawChunk(dp, times, channels=np.arange(384), subtype='ap', save=0,
                     whiten=0, med_sub=0, hpfilt=0, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None,
                     ignore_ks_chanfilt=0):
    '''Function to extract a chunk of raw data on a given range of channels on a given time window.
    ## PARAMETERS
    - dp: datapath to folder with binary path (files must ends in .bin, typically ap.bin)
    - times: list of boundaries of the time window, in seconds [t1, t2].
    - channels (default: np.arange(384)): list of channels of interest, in 0 indexed integers [c1, c2, c3...]
    - subtype: 'ap' or 'lf', whether to exxtract from the high-pass or low-pass filtered binary file
    - save (default 0): save the raw chunk in the bdp directory as '{bdp}_t1-t2_c1-c2.npy'
    - whiten: whether to whiten the data across channels. If nRangeWhiten is not None, whitening matrix is computed with the nRangeWhiten closest channels.
    - med_sub: whether to median-subtract the data across channels. If nRangeMedSub is not none, median of each channel is computed using the nRangeMedSub closest channels.
    - hpfilt: whether to high-pass filter the data, using a 3 nodes butterworth filter of cutoff frequency hpfiltf.
    - hpfiltf: see hpfilt
    - nRangeWhiten: int, see whiten.
    - nRangeMedSub: int, see med_sub.
    - ignore_ks_chanfilt: whether to ignore the filtering made by kilosort, which only uses channels with average events rate > ops.minfr to spike sort. | Default False

    ## RETURNS
    rawChunk: numpy array of shape ((c2-c1), (t2-t1)*fs).
    rawChunk[0,:] is channel 0; rawChunk[1,:] is channel 1, etc.
    '''
    # Find binary file
    assert len(times)==2
    assert times[0]>0
    assert times[1]<get_rec_len(dp, unit='seconds')
    fname=''
    for file in os.listdir(dp):
        if file.endswith(".{}.bin".format(subtype)):
            fname=Path(dp, file)
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
    if not ignore_ks_chanfilt: channels=assert_chan_in_dataset(dp, channels) # index out of 384, should remain the same because rc initial shape is 384!
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))
    if whiten:
        whitenpad=200
        t1, t2 = t1-whitenpad, t2+whitenpad
    bn = op.basename(fname) # binary name
    rcn = f'{bn}_t{times[0]}-{times[1]}_ch{channels[0]}-{channels[-1]}_{whiten}_{med_sub}.npy' # raw chunk name
    rcp = Path(dp, 'routinesMemory', rcn)

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
    # channels on axis 0, time on axis 1
    assert len(bData)%2==0
    rc = np.frombuffer(bData, dtype=np.int16) # 16bits decoding
    rc = rc.reshape((int(t2-t1), Nchans)).T

    # Align signal on each channel
    rc=rc-np.median(rc[:,:10],axis=1)[:,np.newaxis]

    # Median subtraction = CAR
    if med_sub:
        rc=med_substract(rc, 0, nRange=nRangeMedSub)

    # Highpass filter with a 3rd order butterworth filter, like in kilosort2
    if hpfilt:
        rc=apply_filter(rc, bandpass_filter(rate=fs, low=None, high=hpfiltf, order=3), axis=1)

    # Whiten data
    if whiten:
        rc=whitening(rc, nRange=nRangeWhiten)
        rc=rc[:, whitenpad:-whitenpad]

    # get the right channels range, AFTER WHITENING
    rc = rc[channels, :]
    # Scale data
    rc = rc*meta['scale_factor'] # convert into uV

    if save: # sync chan saved in extract_syncChan
        np.save(rcp, rc)

    return rc


def assert_chan_in_dataset(dp, channels):
    cm=chan_map(dp, probe_version='local')
    if not np.all(np.isin(channels, cm[:,0])):
        print("WARNING Kilosort excluded some channels that you provided for analysis \
    because they did not display enough threshold crossings! Jumping channels:{}\
    ".format(channels[~np.isin(channels, cm[:,0])]))
    channels=channels[np.isin(channels, cm[:,0])]
    return channels

#%% Compute stuff to preprocess data

def whitening_matrix_old(x, epsilon=1e-18):
    """
    wmat = whitening_matrix(dat, fudge=1e-18)
    Compute the whitening matrix.
        - dat is a matrix nsamples x nchannels
    Apply using np.dot(dat,wmat)
    Adapted from phy
    Parameters:
        - x: 2D array, rows represents a variable (time), axis 1 an observation (e.g. channel).
              Multiplying by the whitening matrix whitens across obervations.
        - fudge: small value added to diagonal to regularize D
        - nRange: if integer, number of channels to locally compute whitening filter (more robust to noise) | Default None
    """
    assert x.ndim == 2
    x=x.T
    ncols = x.shape[1]
    x_cov = np.cov(x, rowvar=0) # get covariance matrix
    assert x_cov.shape == (ncols, ncols)
    d, v = np.linalg.eigh(x_cov) # covariance eigendecomposition (same as svd for positive-definite matrix)
    d[d<0]=0 # handles calculation innacurracies leading to very tiny negative values instead of tiny positive values
    d = np.diag(1. / np.sqrt(d + epsilon))
    w = np.dot(np.dot(v, d), v.T) # V * D * V': ZCA transform
    return w.T

def whitening_matrix(x, epsilon=1e-18, nRange=None):
    """
    wmat = whitening_matrix(dat, fudge=1e-18)
    Compute the whitening matrix.
        - dat is a matrix nsamples x nchannels
    Apply using np.dot(dat,wmat)
    Adapted from phy
    Parameters:
        - x: 2D array, axis 1 spans time, axis 0 observations (e.g. channel).
             Multiplying by the whitening matrix whitens across obervations.
        - epsilon: small value added to diagonal to regularize D
        - nRange: if integer, number of channels to locally compute whitening filter (more robust to noise) | Default None
    """
    assert x.ndim == 2
    nrows, ncols = x.shape
    x_cov = np.cov(x, rowvar=1) # get covariance matrix across rows (each row is an observation)
    assert x_cov.shape == (nrows, nrows)
    if nRange is None:
        d, v = np.linalg.eigh(x_cov) # covariance eigendecomposition (same as svd for positive-definite matrix)
        d[d<0]=0 # handles calculation innacurracies leading to very tiny negative values instead of tiny positive values
        d = np.diag(1. / np.sqrt(d + epsilon))
        w = np.dot(np.dot(v, d), v.T) # V * D * V': ZCA transform
        return w
    ##TODO make that fast with numba
    rows=np.arange(nrows)
    w=np.zeros((nrows,nrows))
    for i in range(x_cov.shape[0]):
        closest=np.sort(rows[np.argsort(np.abs(rows-i))[:nRange+1]])
        span=slice(closest[0],closest[-1]+1)
        x_cov_local=x_cov[span,span]
        d, v = np.linalg.eigh(x_cov_local) # covariance eigendecomposition (same as svd for positive-definite matrix)
        d[d<0]=0 # handles calculation innacurracies leading to very tiny negative values instead of tiny positive values
        d = np.diag(1. / np.sqrt(d + epsilon))
        w[i,span] = np.dot(np.dot(v, d), v.T)[:,0] # V * D * V': ZCA transform
    return w

def whitening(x, nRange=None):
    '''
    Whitens along axis 0.
    For instance, time should be axis 1 and channels axis 0 to whiten across channels.
    Axis 1 must be larger than axis 0 (need enough samples to properly estimate variance, covariance).
        - x: 2D array, axis 1 spans time, axis 0 observations (e.g. channel).
    Parameters:
        - x: 2D array, axis 1 spans time, axis 0 observations (e.g. channel).
             Multiplying by the whitening matrix whitens across obervations.
        - nRange: if integer, number of channels to locally compute whitening filter (more robust to noise) | Default None
    '''
    assert x.shape[1]>=x.shape[0]
    # Compute whitening matrix
    w=whitening_matrix(x, epsilon=1e-18, nRange=nRange)
    # Whiten
    scales=(np.max(x, 1)-np.min(x, 1))
    x=np.dot(x.T,w).T
    W_scales=(np.max(x, 1)-np.min(x, 1))
    x=x*np.repeat((scales/W_scales).reshape(x.shape[0], 1), x.shape[1], axis=1)

    return x

def bandpass_filter(rate=None, low=None, high=None, order=1):
    """Butterworth bandpass filter."""
    assert low is not None or high is not None
    if low is not None and high is not None: assert low < high
    assert order >= 1
    if high is not None and low is not None:
        return signal.butter(order, (low,high), 'bandpass', fs=rate)
    elif low is not None:
        return signal.butter(order, low, 'lowpass', fs=rate)
    elif high is not None:
        return signal.butter(order, high, 'highpass', fs=rate)

def apply_filter(x, filt, axis=0):
    """Apply a filter to an array."""
    x = np.asarray(x)
    if x.shape[axis] == 0:
        return x
    b, a = filt
    return signal.filtfilt(b, a, x, axis=axis)

def med_substract(x, axis=0, nRange=None):
    '''Median substract along axis 0
    (for instance, channels should be axis 0 and time axis 1 to median substract across channels)'''
    assert axis in [0,1]
    if nRange is None:
        return x-np.median(x, axis=axis) if axis==0 else x-np.median(x, axis=axis)[:,np.newaxis]
    n_points=x.shape[axis]
    x_local_med=np.zeros(x.shape)
    points=np.arange(n_points)
    for xi in range(n_points):
        closest=np.sort(points[np.argsort(np.abs(points-xi))[:nRange+1]])
        if axis==0: x_local_med[xi,:]=np.median(x[closest,:], axis=axis)
        elif axis==1: x_local_med[:,xi]=np.median(x[:,closest], axis=axis)
    return x-x_local_med

def whiten_data(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

#%% paqIO file loading utilities

def paq_read(file_path):
    """
    Read PAQ file (from PackIO) into python
    Lloyd Russell 2015
    See https://github.com/llerussell/paq2py
    Parameters
    ==========
    file_path : str, optional
        full path to file to read in. if none is supplied a load file dialog
        is opened, buggy on mac osx - Tk/matplotlib. Default: None.
    Returns
    =======
    data : ndarray
        the data as a m-by-n array where m is the number of channels and n is
        the number of datapoints
    chan_names : list of str
        the names of the channels provided in PackIO
    hw_chans : list of str
        the hardware lines corresponding to each channel
    units : list of str
        the units of measurement for each channel
    rate : int
        the acquisition sample rate, in Hz
    """

    # file load gui
    # if file_path is None:
    #     root = Tkinter.Tk()
    #     root.withdraw()
    #     file_path = tkFileDialog.askopenfilename()
    #     root.destroy()

    # open file
    fid = open(file_path, 'rb')

    # get sample rate
    rate = int(np.fromfile(fid, dtype='>f', count=1))

    # get number of channels
    num_chans = int(np.fromfile(fid, dtype='>f', count=1))

    # get channel names
    chan_names = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        chan_name = ''
        for j in range(num_chars):
            chan_name = chan_name + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        chan_names.append(chan_name)

    # get channel hardware lines
    hw_chans = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        hw_chan = ''
        for j in range(num_chars):
            hw_chan = hw_chan + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        hw_chans.append(hw_chan)

    # get acquisition units
    units = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        unit = ''
        for j in range(num_chars):
            unit = unit + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        units.append(unit)

    # get data
    temp_data = np.fromfile(fid, dtype='>f', count=-1)
    num_datapoints = int(len(temp_data)/num_chans)
    data = np.reshape(temp_data, [num_datapoints, num_chans]).transpose()

    # close file
    fid.close()

    return {"data": data,
            "chan_names": chan_names,
            "hw_chans": hw_chans,
            "units": units,
            "rate": rate}

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


from npyx.gl import get_rec_len, assert_multi, get_ds_table
