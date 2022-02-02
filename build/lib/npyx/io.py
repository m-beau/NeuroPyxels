# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:30:50 2019

@author: Maxime Beau, Hausser lab, University College London

Input/output utilitaries to deal with Neuropixels files.
"""

import psutil
import os
from ast import literal_eval as ale
from pathlib import Path

import numpy as np
from scipy import signal

from npyx.utils import npa, read_pyfile, list_files

import json

#%% Extract metadata and sync channel

def read_metadata(dp):
    f'''
    {metadata.__doc__}

    If ran on a merged dataset, an additional layer of keys is added: the probes used as keys of
    dp_dict when npyx.merger.merge_datasets was ran (typically 'probe1' and 'probe2'):
    the structure of meta is then 'probe1':meta_data_dataset1,
                                  'probe2':meta_data_dataset_2, ...

    '''

    if assert_multi(dp):
        meta = {}
        for dpx, probe in get_ds_table(dp).loc[:,'dp':'probe'].values:
            meta[probe] = metadata(dpx)
    else:
        meta = metadata(dp)

    return meta


def metadata(dp):
    '''
    Read spikeGLX (.ap/lf.meta) or openEphys (.oebin) metadata files
    and returns their contents as dictionnaries.

    The 'highpass' or 'lowpass' nested dicts correspond to Neuropixels 1.0 high or low pass filtered metadata.
    2.0 recordings only have a 'highpass' key, as they are acquired as a single file matched with a .ap.meta file.
        for spikeGLX, corresponds to metadata of .ap.meta and .lf.meta files.
        for OpenEphys, .oebin metadata relating to the first and second dictionnaries in 'continuous' of the .oebin file
                       which match the /continuous/Neuropix-PXI-100.0 or .1 folders respectively.

    Parameters:
        - dp: str, datapath to spike sorted dataset

    Returns:
        - meta: dictionnary containing contents of meta file.
        the structure of meta is as follow:
        {
        'probe_version':'3A', '1.0_staggered', '2.0_1shank', '2.0_4shanks',
        'highpass':
            {
            'binary_relative_path':relative path to binary file from dp,
            'sampling_rate':int, # sampling rate
            'n_channels_binaryfile':int, # n channels saved on file, typically 385 for .bin and 384 for .dat
            'n_channels_analysed':int, # n channels used for spikesorting. Will set the shape of temp_wh.daat for kilosort.
            'datatype':str, # datatype of binary encoding, typically int16
            'binary_relative_path':relative path to binary file from dp,
            'key1...': all other keys present in meta file, that you must be familiar with!
                       e.g. 'fileSizeBytes' for spikeGLX or 'channels' for OpenEphys...
            },
        'lowpass': {...}, # same as high for low pass filtered data (not existing in 2.0 recordings)
        'events': {...}, # only for openephys recordings, contents of oebin file
        'spikes': {...} # only for openephys recordings, contents of oebin file
        }
    '''
    dp = Path(dp)

    probe_versions = {
        'glx':{3.0:'3A', # option 3
               0.0:'1.0',
               21:'2.0_singleshank',
               24:'2.0_fourshanks'},
        'oe':{"Neuropix-3a":'3A', # source_processor_name keys
                "Neuropix-PXI":'1.0',
                '?1':'2.0_singleshank', # do not know yet
                '?2':'2.0_fourshanks'} # do not know yet
        }

    # import params.py data
    params=read_pyfile(dp/'params.py')

    # find meta file
    glx_ap_files = list_files(dp, "ap.meta", True)
    glx_lf_files = list_files(dp, "lf.meta", True)
    oe_files = list_files(dp, "oebin", True)
    glx_found = np.any(glx_ap_files) or np.any(glx_lf_files)
    oe_found = np.any(oe_files)
    assert glx_found or oe_found, \
        f'WARNING no .ap/lf.meta (spikeGLX) or .oebin (OpenEphys) file found at {dp}.'
    assert not (glx_found and oe_found),\
        'WARNING dataset seems to contain both an open ephys and spikeGLX metafile - fix this!'
    assert len(glx_ap_files)==1 or len(glx_lf_files)==1 or len(oe_files)==1,\
        'WARNING more than 1 .ap.meta or 1 .oebin files found!'

    # Formatting of openephys meta file
    meta = {}
    if oe_found:
        meta['acquisition_software']='OpenEphys'
        # Load OpenEphys metadata
        metafile=Path(oe_files[0])
        with open(metafile) as f:
            meta_oe = json.load(f)

        # find probe version
        oe_probe_version = meta_oe["continuous"][0]["source_processor_name"]
        assert oe_probe_version in probe_versions['oe'].keys(),\
            f'WARNING only probe version {oe_probe_version} not handled with openEphys - post an issue at www.github.com/m-beau/NeuroPyxels'
        meta['probe_version']=probe_versions['oe'][oe_probe_version]

        # Find conversion factor
        # should be 0.19499999284744262695
        meta['bit_uV_conv_factor']=meta_oe["continuous"][0]["channels"][0]["bit_volts"]

        # find everything else
        for filt_key in ['highpass','lowpass']:
            meta[filt_key]={}
            filt_key_i={'highpass':0, 'lowpass':1}[filt_key]
            meta[filt_key]['sampling_rate']=int(meta_oe["continuous"][filt_key_i]['sample_rate'])
            meta[filt_key]['n_channels_binaryfile']=int(meta_oe["continuous"][filt_key_i]['num_channels'])
            meta[filt_key]['n_channels_analysed']=params['n_channels_dat']
            meta[filt_key]['datatype']=params['dtype']
            binary_folder = './continuous/'+meta_oe["continuous"][filt_key_i]['folder_name']
            binary_file = list_files(dp/binary_folder, "dat", False)
            if any(binary_file):
                binary_rel_path = binary_folder+binary_file[0]
                meta[filt_key]['binary_relative_path']=binary_rel_path
                meta[filt_key]['binary_byte_size']=os.path.getsize(dp/binary_rel_path)
                if filt_key=='highpass' and params['dat_path']!=binary_rel_path:
                    print((f'\033[34;1mWARNING edit dat_path in params.py '
                    f'so that it matches relative location of high pass filtered binary file: {binary_rel_path}'))
            else:
                meta[filt_key]['binary_relative_path']='not_found'
                meta[filt_key]['binary_byte_size']='unknown'
            meta[filt_key]={**meta[filt_key], **meta_oe["continuous"][filt_key_i]}
        meta["events"]=meta_oe["events"]
        meta["spikes"]=meta_oe["spikes"]


    # Formatting of SpikeGLX meta file
    elif glx_found:
        meta['acquisition_software']='SpikeGLX'
        # Load SpikeGLX metadata
        meta_glx = {}
        for metafile in glx_ap_files+glx_lf_files:
            if metafile in glx_ap_files: filtkey='highpass'
            elif metafile in glx_lf_files: filtkey='lowpass'
            metafile=Path(metafile)
            meta_glx[filtkey]={}
            with open(metafile, 'r') as f:
                for ln in f.readlines():
                    tmp = ln.split('=')
                    k, val = tmp[0], ''.join(tmp[1:])
                    k = k.strip()
                    val = val.strip('\r\n')
                    if '~' in k:
                        meta_glx[filtkey][k] = val.strip('(').strip(')').split(')(')
                    else:
                        try:  # is it numeric?
                            meta_glx[filtkey][k] = float(val)
                        except:
                            meta_glx[filtkey][k] = val

        # find probe version
        if 'imProbeOpt' in meta_glx["highpass"].keys(): # 3A
            glx_probe_version = meta_glx["highpass"]["imProbeOpt"]
        elif 'imDatPrb_type' in meta_glx["highpass"].keys(): # 1.0 and beyond
            glx_probe_version = meta_glx["highpass"]["imDatPrb_type"]
        else:
             glx_probe_version = 'N/A'

        assert glx_probe_version in probe_versions['glx'].keys(),\
            f'WARNING probe version {glx_probe_version} not handled - post an issue at www.github.com/m-beau/NeuroPyxels'
        meta['probe_version']=probe_versions['glx'][glx_probe_version]

        # Based on probe version,
        # Find the voltage range, gain, encoding
        # and deduce the conversion from units/bit to uV
        Vrange=(meta_glx["highpass"]['imAiRangeMax']-meta_glx["highpass"]['imAiRangeMin'])*1e6
        if meta['probe_version'] in ['3A', '1.0']:
            if Vrange!=1.2e6: print(f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (1.2*10^6). Might be normal!')
            bits_encoding=10
            ampFactor=ale(meta_glx["highpass"]['~imroTbl'][1].split(' ')[3]) # typically 500
            #if ampFactor!=500: print(f'\u001b[31mHeads-up, the voltage amplification factor seems to be {ampFactor}, which is not the default (500). Might be normal!')
        elif meta['probe_version'] in ['2.0_singleshank', '2.0_fourshanks']:
            if Vrange!=1e6: print(f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (10^6). Might be normal!')
            bits_encoding=14
            ampFactor=80 # hardcoded
        meta['bit_uV_conv_factor']=(Vrange/2**bits_encoding/ampFactor)


        # find everything else
        for filt_key in ['highpass','lowpass']:
            if filt_key not in meta_glx.keys(): continue
            meta[filt_key]={}

            # binary file
            filt_suffix={'highpass':'ap','lowpass':'lf'}[filt_key]
            binary_file = list_files(dp, f"{filt_suffix}.bin", False)
            if any(binary_file):
                binary_rel_path = './'+binary_file[0]
                meta[filt_key]['binary_relative_path']=binary_rel_path
                meta[filt_key]['binary_byte_size']=os.path.getsize(dp/binary_rel_path)
            else:
                meta[filt_key]['binary_relative_path']='not_found'
                meta[filt_key]['binary_byte_size']='unknown'

            # sampling rate
            if meta_glx[filt_key]['typeThis'] == 'imec':
                meta[filt_key]['sampling_rate']=int(meta_glx[filt_key]['imSampRate'])
            else:
                meta[filt_key]['sampling_rate']=int(meta_glx[meta_glx['typeThis'][:2]+'SampRate'])

            meta[filt_key]['n_channels_binaryfile']=int(meta_glx[filt_key]['nSavedChans'])
            meta[filt_key]['n_channels_analysed']=params['n_channels_dat']
            meta[filt_key]['datatype']=params['dtype']
            meta[filt_key]={**meta[filt_key], **meta_glx[filt_key]}

    # Calculate length of recording
    high_fs = meta['highpass']['sampling_rate']
    if meta['highpass']['binary_byte_size']=='unknown':
        t_end=np.load(dp/'spike_times.npy').ravel()[-1]
        meta['recording_length_seconds']=t_end/high_fs
    else:
        file_size = meta['highpass']['binary_byte_size']
        item_size = np.dtype(meta['highpass']['datatype']).itemsize
        nChans = meta['highpass']['n_channels_binaryfile']
        meta['recording_length_seconds'] = file_size/item_size/nChans/high_fs


    return meta

def chan_map(dp=None, y_orig='surface', probe_version=None):
    '''
    Returns probe channel map.
    Parameters:
        - dp: str, datapath
        - y_orig: 'surface' or 'tip', where to position channel 0.
                If surface (default), channel map is flipped vertically (0 is now at the surface).
        - probe_version: None, 'local', '3A', '1.0' or '2.0_singleshank' (other types not handled yet, reach out to give your own!).
                        If 'local', will load channelmap from dp (only contains analyzed channels, not all channels)
                        If explicitely given, will return complete channelmap of electrode.
                        If None, will guess probe version from metadata and return complete channelmap.
    Returns:
        - chan_map: array of shape (N_electrodes, 3).
                    1st column is channel indices, 2nd x position, 3rd y position
    '''

    dp = Path(dp)
    assert y_orig in ['surface', 'tip']
    if probe_version is None: probe_version=read_metadata(dp)['probe_version']

    if probe_version in ['3A', '1.0']:
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

    elif probe_version=='1.0':
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

    else:
        probe_version='local'

    if probe_version=='local':
        if dp is None:
            raise ValueError("dp argument is not provided - when channel map is \
                             atypical and probe_version is hence called 'local', \
                             the datapath needs to be provided to load the channel map.")
        c_ind=np.load(dp/'channel_map.npy');cp=np.load(dp/'channel_positions.npy')
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

def get_npix_sync(dp, output_binary = False, filt_key='highpass', unit='seconds'):
    '''Unpacks neuropixels external input data, to align spikes to events.
    Parameters:
        - dp: str, datapath
        - output_binary: bool, whether to output binary sync channel as 0/1s
        - filt_key: str, 'highpass' or 'lowpass' (SpikeGLX: ap/lf, OIpenEphys: Neuropix-PXI-100.0/.1)
        - unit: str, 'seconds' or 'samples', units of returned onsets/offset times

    Returns:
        Dictionnaries of length n_channels = number of channels where threshold crossings were found, [0-16]
        - onsets: dict, {channel_i:np.array(onset1, onset2, ...), ...} in 'unit'
        - offsets: dict, {channel_i:np.array(offset1, offset2, ...), ...} in 'unit'

    '''
    dp = Path(dp)
    if assert_multi(dp):
        dp=get_ds_table(dp)['dp'][0]
        print(f'Loading npix sync channel from a merged dataset - assuming temporal reference frame of dataset 0:\n{dp}')

    assert filt_key in ['highpass', 'lowpass']
    filt_suffix = {'highpass':'ap', 'lowpass':'lf'}[filt_key]
    assert unit in ['seconds', 'samples']
    fname=''
    onsets={}
    offsets={}
    sync_dp=dp/'sync_chan'
    meta = read_metadata(dp)
    srate = meta[filt_key]['sampling_rate'] if unit=='seconds' else 1

    if meta['acquisition_software']=='OpenEphys':
        raise('OpenEphys sync channel loading not implemented yet.')
        filt_id = 0 if filt_key=='highpass' else 1
        timestamps = np.load(dp/f'events/Neuropix-PXI-100.{filt_id}/TTL_1/timestamps.npy')

        onsets={ok:ov/srate for ok, ov in onsets.items()}
        offsets={ok:ov/srate for ok, ov in offsets.items()}

        return onsets,offsets

    elif meta['acquisition_software']=='SpikeGLX':

        # Tries to load pre-saved onsets and offsets
        if sync_dp.exists() and not output_binary:
            print(f'Sync channel extraction directory found: {sync_dp}\n')
            for file in os.listdir(sync_dp):
                if file.endswith("on_samples.npy"):
                    filt_suffix_loaded=file.split('.')[-2][:2]
                    if filt_suffix_loaded==filt_suffix: # if samples are at the instructed sampling rate i.e. lf (2500) or ap (30000)!
                        print(f'Sync channel onsets extracted from {filt_key} ({filt_suffix_loaded}) file found and loaded.')
                        file_i = ale(file[-15])
                        onsets[file_i]=np.load(sync_dp/file)/srate
                        offsets[file_i]=np.load(sync_dp/(file[:-13]+'f'+file[-12:]))/srate
            if any(onsets):
                # else, might be that sync_dp is empty
                return onsets, offsets

        # Tries to load pre-saved compressed binary
        if sync_dp.exists():
            print(f"No file ending in 'on_samples.npy' with the right sampling rate ({filt_suffix}) found in sync_chan directory: extracting sync channel from binary.\n")
            npz_files = list_files(sync_dp, 'npz')
            if any(npz_files):
                print('Compressed binary found - extracting from there...')
                fname=npz_files[0][:-9]
                sync_fname = npz_files[0][:-4]
                binary=np.load(sync_dp/(sync_fname+'.npz'))
                binary=binary[dir(binary.f)[0]].astype(np.int8)

        else: os.mkdir(sync_dp)

        # If still no file name, memorymaps binary directly
        if fname=='':
            ap_files = list_files(dp, 'ap.bin')
            lf_files = list_files(dp, 'lf.bin')

            if filt_suffix=='ap':
                assert any(ap_files), f'No .ap.bin file found at {dp}!! Aborting.'
                fname=ap_files[0]
            elif filt_suffix=='lf':
                assert any(lf_files), f'No .lf.bin file found at {dp}!! Aborting.'
                fname=lf_files[0]

            nchan=meta[filt_key]['n_channels_binaryfile']
            dt=np.dtype(meta[filt_key]['datatype'])
            nsamples = os.path.getsize(dp/fname) / (nchan * dt.itemsize)
            syncdat=np.memmap(dp/fname,
                            mode='r',
                            dtype=dt,
                            shape=(int(nsamples), int(nchan)))[:,-1]


            print('Unpacking {}...'.format(fname))
            binary = unpackbits(syncdat.flatten(),16).astype(np.int8)
            sync_fname = fname[:-4]+'_sync'
            np.savez_compressed(sync_dp/(sync_fname+'.npz'), binary)

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

        onsets={ok:ov/srate for ok, ov in onsets.items()}
        offsets={ok:ov/srate for ok, ov in offsets.items()}

        assert any(onsets), ("WARNING no sync channel found in dataset - "
            "make sure you are running this function on a dataset with a synchronization TTL!")

        return onsets,offsets


def extract_rawChunk(dp, times, channels=np.arange(384), filt_key='highpass', save=0,
                     whiten=0, med_sub=0, hpfilt=0, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None,
                     ignore_ks_chanfilt=0):
    '''Function to extract a chunk of raw data on a given range of channels on a given time window.
    ## PARAMETERS
    - dp: datapath to folder with binary path (files must ends in .bin, typically ap.bin)
    - times: list of boundaries of the time window, in seconds [t1, t2].
    - channels (default: np.arange(384)): list of channels of interest, in 0 indexed integers [c1, c2, c3...]
    - filt_key: 'ap' or 'lf', whether to exxtract from the high-pass or low-pass filtered binary file
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
    dp = Path(dp)
    meta = read_metadata(dp)
    assert meta[filt_key]['binary_relative_path']!='not_found',\
        f'No binary file (./*.ap.bin or ./continuous/Neuropix-PXI-100.0/*.dat) found in folder {dp}!!'
    fname = dp/meta[filt_key]['binary_relative_path']

    assert len(times)==2
    assert times[0]>0
    assert times[1]<meta['recording_length']

    fs = meta[filt_key]['sampling_rate']
    Nchans=meta[filt_key]['n_channels_binaryfile']
    bytes_per_sample=2

    # Format inputs
    cm=chan_map(dp, probe_version='local'); assert cm.shape[0]<=Nchans-1
    if not ignore_ks_chanfilt: channels=assert_chan_in_dataset(dp, channels) # index out of 384, should remain the same because rc initial shape is 384!
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))
    if whiten:
        whitenpad=200
        t1, t2 = t1-whitenpad, t2+whitenpad
    bn = os.path.basename(fname) # binary name
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
    rc = rc*meta['bit_uV_conv_factor'] # convert into uV

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
        full path to file to read in.

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

    # open file
    fid = open(file_path, 'rb')

    # get sample rate
    rate = int(np.fromfile(fid, dtype='>f', count=1))
    assert rate!=0, 'WARNING something went wrong with the paq file, redownload it.'

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

# def _range_from_slice(myslice, start=None, stop=None, step=None, length=None):
#     """Convert a slice to an array of integers."""
#     assert isinstance(myslice, slice)
#     # Find 'step'.
#     step = myslice.step if myslice.step is not None else step
#     if step is None:
#         step = 1
#     # Find 'start'.
#     start = myslice.start if myslice.start is not None else start
#     if start is None:
#         start = 0
#     # Find 'stop' as a function of length if 'stop' is unspecified.
#     stop = myslice.stop if myslice.stop is not None else stop
#     if length is not None:
#         stop_inferred = floor(start + step * length)
#         if stop is not None and stop < stop_inferred:
#             raise ValueError("'stop' ({stop}) and ".format(stop=stop) +
#                              "'length' ({length}) ".format(length=length) +
#                              "are not compatible.")
#         stop = stop_inferred
#     if stop is None and length is None:
#         raise ValueError("'stop' and 'length' cannot be both unspecified.")
#     myrange = np.arange(start, stop, step)
#     # Check the length if it was specified.
#     if length is not None:
#         assert len(myrange) == length
#     return myrange



from npyx.gl import assert_multi, get_ds_table
#
