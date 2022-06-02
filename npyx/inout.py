# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:30:50 2019

@author: Maxime Beau, Hausser lab, University College London

Input/output utilitaries to deal with Neuropixels files.
"""

import psutil
import shutil
import os
from ast import literal_eval as ale
from pathlib import Path
from tqdm.auto import tqdm

from math import ceil
import numpy as np
try:
    import cupy as cp
except ImportError:
    print(("cupy could not be imported - "
    "some functions dealing with the binary file (filtering, whitening...) will not work."))

from npyx.utils import npa, read_pyfile, list_files
from npyx.preprocess import apply_filter, bandpass_filter, whitening, approximated_whitening_matrix, med_substract,\
                            gpufilter, cu_median, adc_realign, kfilt

import json

#%% Load metadata and channel map

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
    assert dp.exists(), "Provided path does not exist!"
    assert dp.is_dir(), f"Provided path {dp} is a filename!"

    probe_versions = {
        'glx':{3.0:'3A', # option 3
               0.0:'1.0',
               21:'2.0_singleshank',
               24:'2.0_fourshanks'},
        'oe':{"Neuropix-3a":'3A', # source_processor_name keys
                "Neuropix-PXI":'1.0',
                '?1':'2.0_singleshank', # do not know yet
                '?2':'2.0_fourshanks'}, # do not know yet
        'int':{'3A':1, '1.0':1,
               '2.0_singleshank':2, '2.0_fourshanks':2}
        }

    # import params.py data
    params_f = dp/'params.py'
    if params_f.exists():
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
    meta['path'] = os.path.realpath(dp)
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
        meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]

        # Find conversion factor
        # should be 0.19499999284744262695
        meta['bit_uV_conv_factor']=meta_oe["continuous"][0]["channels"][0]["bit_volts"]

        # find everything else
        for filt_key in ['highpass','lowpass']:
            meta[filt_key]={}
            filt_key_i={'highpass':0, 'lowpass':1}[filt_key]
            meta[filt_key]['sampling_rate']=int(meta_oe["continuous"][filt_key_i]['sample_rate'])
            meta[filt_key]['n_channels_binaryfile']=int(meta_oe["continuous"][filt_key_i]['num_channels'])
            if params_f.exists():
                meta[filt_key]['n_channels_analysed']=params['n_channels_dat']
                meta[filt_key]['datatype']=params['dtype']
            else:
                meta[filt_key]['n_channels_analysed']=meta[filt_key]['n_channels_binaryfile']
                meta[filt_key]['datatype']='int16'
            binary_folder = './continuous/'+meta_oe["continuous"][filt_key_i]['folder_name']
            binary_file = list_files(dp/binary_folder, "dat", False)
            if any(binary_file):
                binary_rel_path = binary_folder+binary_file[0]
                meta[filt_key]['binary_relative_path']=binary_rel_path
                meta[filt_key]['binary_byte_size']=os.path.getsize(dp/binary_rel_path)
                if filt_key=='highpass' and params_f.exists():
                    if params['dat_path']!=binary_rel_path:
                        print((f'\033[34;1mWARNING edit dat_path in params.py '
                        f'so that it matches relative location of high pass filtered binary file: {binary_rel_path}'))
            else:
                meta[filt_key]['binary_relative_path']='not_found'
                meta[filt_key]['binary_byte_size']='unknown'
                print(f"\033[91;1mWARNING {filt_key} binary file not found at {dp}\033[0m")
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
        meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]

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
            binary_rel_path = get_binary_file_path(dp, filt_suffix, False)
            if binary_rel_path!='not_found':
                meta[filt_key]['binary_byte_size']=os.path.getsize(dp/binary_rel_path)
                meta[filt_key]['binary_relative_path']='./'+binary_rel_path
            else:
                meta[filt_key]['binary_byte_size']='unknown'
                meta[filt_key]['binary_relative_path']=binary_rel_path
                #print(f"\033[91;1mWARNING binary file .{filt_suffix}.bin not found at {dp}\033[0m")

            # sampling rate
            if meta_glx[filt_key]['typeThis'] == 'imec':
                meta[filt_key]['sampling_rate']=int(meta_glx[filt_key]['imSampRate'])
            else:
                meta[filt_key]['sampling_rate']=int(meta_glx[meta_glx['typeThis'][:2]+'SampRate'])

            meta[filt_key]['n_channels_binaryfile']=int(meta_glx[filt_key]['nSavedChans'])
            if params_f.exists():
                meta[filt_key]['n_channels_analysed']=params['n_channels_dat']
                meta[filt_key]['datatype']=params['dtype']
            else:
                meta[filt_key]['n_channels_analysed']=meta[filt_key]['n_channels_binaryfile']
                meta[filt_key]['datatype']='int16'
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

#%% Binary file I/O, including sync channel

def get_binary_file_path(dp, filt_suffix='ap', absolute_path=True):
    f'''Return the path of the binary file (.bin) from a directory.

    Wrapper of get_glx_file_path:
    {get_glx_file_path.__doc__}
    '''
    
    return get_glx_file_path(dp, 'bin', filt_suffix, absolute_path)

def get_meta_file_path(dp, filt_suffix='ap', absolute_path=True):
    f'''Return the path of the meta file (.meta) from a directory.

    Wrapper of get_glx_file_path:
    {get_glx_file_path.__doc__}
    '''
    
    return get_glx_file_path(dp, 'meta', filt_suffix, absolute_path)

def get_glx_file_path(dp, suffix, filt_suffix='ap', absolute_path=True):
    '''Return the path of a spikeGLX file (.bin or .meta) from a directory.

    Parameters:
    - dp: str, directory
    - filt_suffix: 'ap' or 'lf', seek ap (highpass) or lfp (lowpass) binary file,
                   for 1.0 recordings. Always 'ap' for 2.0.
    - absolute_path: bool, whether to return path from root
                     (if False, returns relative path from dp)

    Returns:
        The absolute path to the spikeGLX file with the associated filt key, 'ap' or 'lf'.
    '''

    dp = Path(dp)
    assert suffix in ['bin', 'meta']
    assert filt_suffix in ['ap','lf']
    glx_files = list_files(dp, f"{filt_suffix}.{suffix}", absolute_path)
    assert len(glx_files) <= 1, (f"More than one {filt_suffix}.{suffix} files found at {dp}! ",
        "If you keep several versions, store other files in a subdirectory (e.g. original_data).")

    if len(glx_files)==0:
        glx_files = ['not_found']
    
    return glx_files[0]

def unpackbits(x,num_bits = 16):
    '''
    unpacks numbers in bits.
    '''
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(np.int64).reshape(xshape + [num_bits])

def get_npix_sync(dp, output_binary = False, filt_key='highpass', unit='seconds',
                  verbose=False, again=False):
    '''Unpacks neuropixels external input data, to align spikes to events.
    Parameters:
        - dp: str, datapath
        - output_binary: bool, whether to output binary sync channel as 0/1s
        - filt_key: str, 'highpass' or 'lowpass' (SpikeGLX: ap/lf, OIpenEphys: Neuropix-PXI-100.0/.1)
        - unit: str, 'seconds' or 'samples', units of returned onsets/offset times
        - verbose: bool, whether to print rich information
        - again: bool, whether to reload sync channel from binary file.

    Returns:
        Dictionnaries of length n_channels = number of channels where threshold crossings were found, [0-16]
        - onsets: dict, {channel_i:np.array(onset1, onset2, ...), ...} in 'unit'
        - offsets: dict, {channel_i:np.array(offset1, offset2, ...), ...} in 'unit'

    '''
    dp = Path(dp)
    if assert_multi(dp):
        dp=Path(get_ds_table(dp)['dp'][0])
        if verbose: print(f'Loading npix sync channel from a merged dataset - assuming temporal reference frame of dataset 0:\n{dp}')

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
        raise('OpenEphys sync channel loading not implemented yet - manually load ./Neuropix-PXI-100.0/TTL_1/timestamps.npy.')
        filt_id = 0 if filt_key=='highpass' else 1
        timestamps = np.load(dp/f'events/Neuropix-PXI-100.{filt_id}/TTL_1/timestamps.npy')

        onsets={ok:ov/srate for ok, ov in onsets.items()}
        offsets={ok:ov/srate for ok, ov in offsets.items()}

        return onsets,offsets

    elif meta['acquisition_software']=='SpikeGLX':

        # Tries to load pre-saved onsets and offsets
        if sync_dp.exists() and not output_binary:
            if verbose: print(f'Sync channel extraction directory found: {sync_dp}\n')
            for file in os.listdir(sync_dp):
                if file.endswith("on_samples.npy"):
                    filt_suffix_loaded=file.split('.')[-2][:2]
                    if filt_suffix_loaded==filt_suffix: # if samples are at the instructed sampling rate i.e. lf (2500) or ap (30000)!
                        if verbose: print(f'Sync channel onsets ({file}) file found and loaded.')
                        file_i = ale(file[-15])
                        onsets[file_i]=np.load(sync_dp/file)/srate
                elif file.endswith("of_samples.npy"):
                    filt_suffix_loaded=file.split('.')[-2][:2]
                    if filt_suffix_loaded==filt_suffix: # if samples are at the instructed sampling rate i.e. lf (2500) or ap (30000)!
                        if verbose: print(f'Sync channel offsets ({file}) file found and loaded.')
                        file_i = ale(file[-15])
                        offsets[file_i]=np.load(sync_dp/file)/srate
            if any(onsets):
                # else, might be that sync_dp is empty
                return onsets, offsets

        # Tries to load pre-saved compressed binary
        if sync_dp.exists() and not again:
            if verbose: print(f"No file ending in 'on_samples.npy' with the right sampling rate ({filt_suffix}) found in sync_chan directory: extracting sync channel from binary.\n")
            npz_files = list_files(sync_dp, 'npz')
            if any(npz_files):
                if verbose: print('Compressed binary found - extracting from there...')
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
                     ignore_ks_chanfilt=0, center_chans_on_0=False, verbose=False, scale=True, again=False):
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
    - scale: A boolean variable specifying whether we should convert the resulting raw
             A2D samples to uV. Defaults to True
    ## RETURNS
    rawChunk: numpy array of shape ((c2-c1), (t2-t1)*fs).
    rawChunk[0,:] is channel 0; rawChunk[1,:] is channel 1, etc.
    '''
    # Find binary file
    dp = Path(dp)
    meta = read_metadata(dp)
    fname = get_binary_file_path(dp, filt_suffix='ap', absolute_path=True)

    fs = meta[filt_key]['sampling_rate']
    Nchans=meta[filt_key]['n_channels_binaryfile']
    bytes_per_sample=2
    whitenpad=200

    assert len(times)==2
    assert times[0]>=0
    assert times[1]<meta['recording_length_seconds']

    # Format inputs
    cm=chan_map(dp, probe_version='local'); assert cm.shape[0]<=Nchans-1
    if not ignore_ks_chanfilt: channels=assert_chan_in_dataset(dp, channels) # index out of 384, should remain the same because rc initial shape is 384!
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))
    if whiten:
        if t1<whitenpad:
            print(f"times[0] set to {round(whitenpad/30000, 5)}s because whitening requires a pad.")
            t1 = whitenpad
            times[0] = t1/30000
        t1, t2 = t1-whitenpad, t2+whitenpad
    bn = os.path.basename(fname) # binary name
    rcn = f'{bn}_t{times[0]}-{times[1]}_ch{channels[0]}-{channels[-1]}_{whiten}_{med_sub}_{scale}.npy' # raw chunk name
    rcp = get_npyx_memory(dp) / rcn

    if os.path.isfile(rcp) and not again:
        return np.load(rcp)

    # Check that available memory is high enough to load the raw chunk
    vmem=dict(psutil.virtual_memory()._asdict())
    chunkSize = int(fs*Nchans*bytes_per_sample*(times[1]-times[0]))
    if verbose:
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
    if center_chans_on_0:
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
    if scale:
        rc = rc*meta['bit_uV_conv_factor'] # convert into uV

    # convert from cupy to numpy array
    if 'cp' in globals():
        rc = cp.asnumpy(rc)

    if save: # sync chan saved in extract_syncChan
        np.save(rcp, rc)

    return rc

def assert_chan_in_dataset(dp, channels):
    channels = np.array(channels)
    cm=chan_map(dp, probe_version='local')
    if not np.all(np.isin(channels, cm[:,0])):
        print(("WARNING Kilosort excluded some channels that you provided for analysis "
               "because they did not display enough threshold crossings! Jumping channels:"
               f"{channels[~np.isin(channels, cm[:,0])]}"))
    channels=channels[np.isin(channels, cm[:,0])]
    return channels

#%% Binary file filtering wrappers

def preprocess_binary_file(dp=None, filt_key='ap', fname=None, target_dp=None, move_orig_data=True,
                       ADC_realign = False, median_subtract=False, f_low=None, f_high=300, order=3,
                       spatial_filt=False, whiten = False, whiten_range=32,
                       again_Wrot=False, verbose=False):
    """Creates a preprocessed copy of binary file at dp/fname_filtered.bin,
    and moves the original binary file to dp/original_data.fname.bin.

    One must precise either dp (path to directory or ) or fname (absolute path to binary file).

    Preprocessing steps:
    - optional - realigning data according to ADCs, like global demux from CatGT does
    - optional - CAR, common average referencing (must be BEFORE filtering as filtering re-offsets each channel)
    - necessary - high pass filtering the data, to remove DC offsets
    - removing correlated noise across channels, using either/all of these 2 options:
        - spatial filtering with butterworth filter
        - whitening
        using both is not recommended, has not been well tested.

    By default, the data is ADC realigned -> CAR -> high pass filtered at 300Hz 
    with a 3nodes butterworth filter (bidirectional to prevent phase shifting).


    Parameters:
    - dp: optional str, path to binary file directory. dp/*.bin file will be found and used for filtering.
    - filt_key: str, 'ap' or 'lf' (if filtering ap.bin or lf.bin file)
    - fname: optional str, absolute path of binary file to filter (if provided, *.bin will not be guessed)
    - target_dp: str or Path, directory to save preprocessed binary file (by default, dp)
    - move_orig_data: bool, if true a directory is created at dp/original_data, and the original binary file is moved there.
    - ADC_realign: bool, whether to realign data based on Neuropixels ADC shifts (slow because requires FFT)
    - CAR: bool, whether to perform common average subtraction
    - f_low: optional float, lowpass filter frequency
    - f_high: float, highpass filter frequency (necessary)
    - order: int, butterworth filter order (default 3)
    - spatial_filt: bool, whether to high pass filter across channels at 0.1 Hz
    - whiten: bool, whether to whiten across channels
    - verbose: bool, whether to print extra information
    """

    # Parameters check
    assert dp is not None or fname is not None,\
        "You must either provide a path to the binary file directory (dp)\
            or the absolute path to the binary file (fname)."
    assert f_low is not None or f_high is not None,\
        "You must either provide a lowpass (low) or highpass (high) pass filter frequency."
    assert filt_key in ['ap', 'lf']

    # samples of symmetrical buffer for whitening and spike detection
    # Must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).
    ntb = 64 # in kilosort: called ntbuff
    nSkipCov = 25

    # Fetch binary file name, define target file name
    if fname is None:
        fname = get_binary_file_path(dp, filt_key, True)
    fname=Path(fname)
    dp = fname.parent
    if target_dp is None:
        target_dp = dp
    else:
        target_dp = Path(target_dp)
    print(f"Preprocessing {fname}...")

    filter_suffix = ""
    message = ""
    if ADC_realign:
        filter_suffix+=f"_adcshift{ADC_realign}"
        message+="    - shifting ADCs,\n"
    if median_subtract:
        filter_suffix+="_medsub"
        message+="    - median subtraction (aka common average referencing CAR),\n"
    filter_suffix+=f"_tempfilt{f_low}{f_high}"
    low_s = 0 if f_low is None else f_low
    message+=f"    - filtering in time (between {low_s} and {f_high} Hz),\n"
    if whiten:
        filter_suffix+=f"_whit{whiten}{whiten_range}"
        message+=f"    - whitening (using {whiten_range} closest channels),\n"
    if spatial_filt:
        filter_suffix+=f"_spatfilt{spatial_filt}"
        message+=f"    - filtering in space ({spatial_filt} 'Hz'),\n"
    filtered_fname = str(fname.name)[:-7]+filter_suffix+".ap.bin"
    message = message[:-2]+"."
    print(message)

    # fetch metadata
    fk = {'ap':'highpass', 'lf':'lowpass'}[filt_key]
    meta = read_metadata(dp)
    fs = meta[fk]['sampling_rate']
    binary_byte_size = meta[fk]['binary_byte_size']

    # Check that there is enough free memory on disk
    free_memory = shutil.disk_usage(target_dp)[2]
    assert free_memory > binary_byte_size + 2**10,\
        f"Not enough free space on disk at {target_dp} (need {(binary_byte_size+2**10)//2**20} MB)"

    # memory map binary file
    n_channels = meta[fk]['n_channels_binaryfile']
    channels_to_process = np.arange(n_channels-1) # would allow in the future to process specific channels
    chans_mask = np.isin(np.arange(n_channels), channels_to_process)
    dtype = meta[fk]['datatype']
    offset = 0
    item_size = np.dtype(dtype).itemsize
    n_samples = (binary_byte_size - offset) // (item_size * n_channels)
    memmap_f = np.memmap(fname, dtype=dtype, offset=offset, shape=(n_samples, n_channels), mode='r+')


    # fetch whitening matrix (estimate covariance over a few batches)
    if whiten:
        Wrot_path = dp / 'whitening_matrix.npy'
        Wrot = approximated_whitening_matrix(memmap_f, Wrot_path, whiten_range,
            NT, Nbatch, NTbuff, ntb, nSkipCov, n_channels, channels_to_process,
            f_high, fs, again=again_Wrot, verbose=verbose)

    # Preprocess iteratively, batch by batch
    NT = 64 * 1024 + ntb
    Nbatch = ceil(n_samples / NT)
    NTbuff = NT + 3 * ntb
    w_edge = cp.linspace(0,1,ntb).reshape(-1, 1) # weights to combine data batches at the edge
    buff_prev = cp.zeros((ntb, n_channels-1), dtype=np.int32)
    last_batch=False
    with open(target_dp / filtered_fname, 'wb') as fw:  # open for writing processed data
        for ibatch in tqdm(range(Nbatch), desc="Preprocessing"):
            # we'll create a binary file of batches of NT samples, which overlap consecutively
            # on params.ntbuff samples
            # in addition to that, we'll read another params.ntbuff samples from before and after,
            # to have as buffers for filtering

            # Collect data batch
            i = max(0, NT * ibatch - ntb)
            rawData = memmap_f[i:i + NTbuff]
            if rawData.size == 0:
                print("Loaded buffer has an empty size!")
                break  # this shouldn't really happen, unless we counted data batches wrong
            nsampcurr = rawData.shape[0]  # how many time samples the current batch has
            if nsampcurr < NTbuff:
                # when reaching end of file, mirror end by adding missing samples to fit in GPU
                last_batch = True
                n_extra_samples = NTbuff - nsampcurr
                rawData = np.concatenate(
                    (rawData, np.tile(rawData[nsampcurr - 1], (n_extra_samples, 1))), axis=0)
            if i == 0:
                bpad = np.tile(rawData[0], (ntb, 1))
                rawData = np.concatenate((bpad, rawData[:NTbuff - ntb]), axis=0)
            rawData = cp.asarray(rawData, dtype=np.float32)

            # Extract channels to use for processing
            # at minima, removes sync channel
            batch = rawData[:,chans_mask]

            # Re-alignment based on ADCs shifts (like CatGT)
            # should be the first preprocessing step,
            # it simply consists in properly realigning the data!
            if ADC_realign:
                batch = cp.asnumpy(batch)
                batch = adc_realign(batch, version=meta['probe_version_int'])
                batch = cp.asarray(batch, dtype=np.float32)

            # CAR (optional) -> temporal filtering -> weight to combine edges -> unpadding
            batch = gpufilter(batch, fs=fs, fshigh=f_high, fslow=f_low, order=order,
                              car=median_subtract, bidirectional=False)
            assert batch.flags.c_contiguous # check that ordering is still C, not F
            batch[ntb:2*ntb] = w_edge * batch[ntb:2*ntb] + (1 - w_edge) * buff_prev
            buff_prev = batch[NT + ntb: NT + 2*ntb]
            batch = batch[ntb:ntb + NT, :]  # remove timepoints used as buffers

            # Spatial filtering (replaces whitening)
            if spatial_filt:
                batch = kfilt(batch.T, butter_kwargs = {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}).T

            # whiten the data and scale by 200 for int16 range
            if whiten:
                print("Whitening not implemented yet.")
                batch = cp.dot(batch, Wrot)
            
            assert batch.flags.c_contiguous  # check that ordering is still C, not F
            if batch.shape[0] != NT:
                raise ValueError(f'Batch {ibatch} processed incorrectly')

            # add unprocessed channels back to batch
            # (at minima including last 16 bits for sync signal)
            rebuilt_batch = rawData[ntb:ntb + NT, :] # remove timepoints used as buffers; includes unprocessed channels
            rebuilt_batch[:,chans_mask] = batch
            if last_batch:
                # remove mirrored data at the end
                n_extra_samples = n_extra_samples - 2*ntb # account for buffers
                rebuilt_batch = rebuilt_batch[:-n_extra_samples]

            # convert to int16, and gather on the CPU side
            datcpu = cp.asnumpy(rebuilt_batch.astype(np.dtype(dtype)))

            # write this batch to binary file
            if verbose and (ibatch%(Nbatch//50)==0 or last_batch):
                print(f"{(target_dp/filtered_fname).stat().st_size} total, {rebuilt_batch.size * 2} bytes written to file {datcpu.shape} array size")
            datcpu.tofile(fw)
        if verbose: print(f"{(target_dp/filtered_fname).stat().st_size} total")

    # Finally, if everything ran smoothly,
    # move original binary to new directory
    if move_orig_data:
        orig_dp = dp/'original_data'
        orig_dp.mkdir(exist_ok=True)
        if not (orig_dp/fname.name).exists(): fname.replace(orig_dp/fname.name)
        meta_f = get_meta_file_path(dp, filt_key, False)
        if not (orig_dp/meta_f).exists():
            if (dp/meta_f).exists():
                shutil.copy(dp/meta_f, orig_dp/meta_f)
            if (dp/'channel_map.npy').exists():
                shutil.copy(dp/'channel_map.npy', orig_dp/'channel_map.npy')
            if (dp/'channel_positions.npy').exists():
                shutil.copy(dp/'channel_positions.npy', orig_dp/'channel_positions.npy')

    return target_dp/filtered_fname

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

# def _start_stop(item):
#     """Find the start and stop indices of a __getitem__ item.

#     This is used only by ConcatenatedArrays.

#     Only two cases are supported currently:

#     * Single integer.
#     * Contiguous slice in the first dimension only.

#     """
#     if isinstance(item, tuple):
#         item = item[0]
#     if isinstance(item, slice):
#         # Slice.
#         if item.step not in (None, 1):
#             raise NotImplementedError()
#         return item.start, item.stop
#     elif isinstance(item, (list, np.ndarray)):
#         # List or array of indices.
#         return np.min(item), np.max(item)
#     else:
#         # Integer.
#         return item, item + 1

# def _fill_index(arr, item):
#     if isinstance(item, tuple):
#         item = (slice(None, None, None),) + item[1:]
#         return arr[item]
#     else:
#         return arr

# class ConcatenatedArrays(object):
#     """This object represents a concatenation of several memory-mapped
#     arrays. Coming from phy.io.array.py"""
#     def __init__(self, arrs, cols=None, scaling=None):
#         assert isinstance(arrs, list)
#         self.arrs = arrs
#         # Reordering of the columns.
#         self.cols = cols
#         self.offsets = np.concatenate([[0], np.cumsum([arr.shape[0]
#                                                        for arr in arrs])],
#                                       axis=0)
#         self.dtype = arrs[0].dtype if arrs else None
#         self.scaling = scaling

#     @property
#     def shape(self):
#         if self.arrs[0].ndim == 1:
#             return (self.offsets[-1],)
#         ncols = (len(self.cols) if self.cols is not None
#                  else self.arrs[0].shape[1])
#         return (self.offsets[-1], ncols)

#     def _get_recording(self, index):
#         """Return the recording that contains a given index."""
#         assert index >= 0
#         recs = np.nonzero((index - self.offsets[:-1]) >= 0)[0]
#         if len(recs) == 0:  # pragma: no cover
#             # If the index is greater than the total size,
#             # return the last recording.
#             return len(self.arrs) - 1
#         # Return the last recording such that the index is greater than
#         # its offset.
#         return recs[-1]

#     def _get(self, item):
#         cols = self.cols if self.cols is not None else slice(None, None, None)
#         # Get the start and stop indices of the requested item.
#         start, stop = _start_stop(item)
#         # Return the concatenation of all arrays.
#         if start is None and stop is None:
#             return np.concatenate(self.arrs, axis=0)[..., cols]
#         if start is None:
#             start = 0
#         if stop is None:
#             stop = self.offsets[-1]
#         if stop < 0:
#             stop = self.offsets[-1] + stop
#         # Get the recording indices of the first and last item.
#         rec_start = self._get_recording(start)
#         rec_stop = self._get_recording(stop)
#         assert 0 <= rec_start <= rec_stop < len(self.arrs)
#         # Find the start and stop relative to the arrays.
#         start_rel = start - self.offsets[rec_start]
#         stop_rel = stop - self.offsets[rec_stop]
#         # Single array case.
#         if rec_start == rec_stop:
#             # Apply the rest of the index.
#             out = _fill_index(self.arrs[rec_start][start_rel:stop_rel], item)
#             out = out[..., cols]
#             return out
#         chunk_start = self.arrs[rec_start][start_rel:]
#         chunk_stop = self.arrs[rec_stop][:stop_rel]
#         # Concatenate all chunks.
#         l = [chunk_start]
#         if rec_stop - rec_start >= 2:
#             print("Loading a full virtual array: this might be slow "
#                         "and something might be wrong.")
#             l += [self.arrs[r][...] for r in range(rec_start + 1,
#                                                    rec_stop)]
#         l += [chunk_stop]
#         # Apply the rest of the index.
#         return _fill_index(np.concatenate(l, axis=0), item)[..., cols]

#     def __getitem__(self, item):
#         out = self._get(item)
#         assert out is not None
#         if self.scaling is not None and self.scaling != 1:
#             out = out * self.scaling
#         return out

#     def __len__(self):
#         return self.shape[0]

# def _pad(arr, n, dir='right'):
#     """Pad an array with zeros along the first axis.

#     Parameters
#     ----------

#     n : int
#         Size of the returned array in the first axis.
#     dir : str
#         Direction of the padding. Must be one 'left' or 'right'.

#     """
#     assert dir in ('left', 'right')
#     if n < 0:
#         raise ValueError("'n' must be positive: {0}.".format(n))
#     elif n == 0:
#         return np.zeros((0,) + arr.shape[1:], dtype=arr.dtype)
#     n_arr = arr.shape[0]
#     shape = (n,) + arr.shape[1:]
#     if n_arr == n:
#         assert arr.shape == shape
#         return arr
#     elif n_arr < n:
#         out = np.zeros(shape, dtype=arr.dtype)
#         if dir == 'left':
#             out[-n_arr:, ...] = arr
#         elif dir == 'right':
#             out[:n_arr, ...] = arr
#         assert out.shape == shape
#         return out
#     else:
#         if dir == 'left':
#             out = arr[-n:, ...]
#         elif dir == 'right':
#             out = arr[:n, ...]
#         assert out.shape == shape
#         return out

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



from npyx.gl import assert_multi, get_ds_table, get_npyx_memory
#
