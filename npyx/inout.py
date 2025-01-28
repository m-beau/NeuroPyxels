# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:30:50 2019

@author: Maxime Beau, Hausser lab, University College London

Input/output utilitaries to deal with Neuropixels files.
"""

import os
import shutil
from ast import literal_eval as ale
from math import ceil
from pathlib import Path

import multiprocessing
import numpy as np
import psutil
from tqdm.auto import tqdm

try:
    import cupy as cp
except ImportError:
    if multiprocessing.current_process().name == 'MainProcess':
        print(("cupy could not be imported - "
        "some functions dealing with the binary file (filtering, whitening...) will not work."))

import json

from npyx.utils import list_files, npa, read_pyfile, npyx_cacher, is_writable

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

    Arguments:
        - dp: str, datapath to spike sorted dataset

    Returns:
        - meta: dictionnary containing contents of meta file.
        the structure of meta is as follow:
        {
        'probe_version': either of '3A', '1.0_staggered', '2.0_1shank', '2.0_4shanks', 'ultra_high_density';
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
        'glx':{3.0:  '3A', # option 3
               0.0:  '1.0',
               1.0:  '1.0', # precise type unknown

               1020:  '1.0', # precise type unknown
               1100:  '1.0', # precise type unknown
               1200:  '1.0', # precise type unknown
               1300:  '1.0', # precise type unknown

               1110:  '1.0', # precise type unknown
               1120:  '1.0', # precise type unknown
               1121:  '1.0', # precise type unknown
               1122:  '1.0', # precise type unknown
               1123:  'ultra_high_density',

               1030: 'NHP_1.0',

               21:   '2.0_singleshank',
               2003: '2.0_singleshank',
               2004: '2.0_singleshank',

               24:   '2.0_fourshanks',
               2013: '2.0_fourshanks',
               2014: '2.0_fourshanks', # assumed type
               2020: '2.0_fourshanks', # assuned type
               },
        'oe':{"Neuropix-3a":'3A', # source_processor_name keys
                "Neuropix-PXI":'1.0',
                '?1':'2.0_singleshank', # do not know yet
                '?2':'2.0_fourshanks'}, # do not know yet
        'int':{'3A':1,
               '1.0':1,
               'NHP_1.0':1,
               '2.0_singleshank':2,
               '2.0_fourshanks':2,
               'ultra_high_density':3}
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
        for i,processor in enumerate(meta_oe['continuous']):
            if 'Neuropix-PXI' in processor["source_processor_name"]:
                probe_index = i
                break
        oe_probe_version = meta_oe["continuous"][probe_index]["source_processor_name"]
        assert oe_probe_version in probe_versions['oe'].keys(),\
            f'WARNING only probe version {oe_probe_version} not handled with openEphys - post an issue at www.github.com/m-beau/NeuroPyxels'
        meta['probe_version']=probe_versions['oe'][oe_probe_version]
        meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]

        # Find conversion factor
        # should be 0.19499999284744262695
        meta['bit_uV_conv_factor']=meta_oe["continuous"][probe_index]["channels"][0]["bit_volts"]

        # index for highpass and lowpass
        filt_index = {'highpass': [], 'lowpass': []}
        for i,processor in enumerate(meta_oe['continuous']):
            if 'AP' in processor['folder_name']:
                filt_index['highpass'] = i
            if 'LFP' in processor['folder_name']:
                filt_index['lowpass'] = i


        # find everything else
        for filt_key in ['highpass','lowpass']:
            meta[filt_key]={}
            filt_key_i=filt_index[filt_key]
            meta[filt_key]['sampling_rate']=float(meta_oe["continuous"][filt_key_i]['sample_rate'])
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
                if filt_key=='highpass' and params_f.exists() and params['dat_path']!=binary_rel_path:
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
        if 'imProbeOpt' in meta_glx["highpass"]: # 3A
            glx_probe_version = meta_glx["highpass"]["imProbeOpt"]
        elif 'imDatPrb_type' in meta_glx["highpass"]: # 1.0 and beyond
            glx_probe_version = meta_glx["highpass"]["imDatPrb_type"]
        else:
             glx_probe_version = 'N/A'

        if glx_probe_version in probe_versions['glx']:
            meta['probe_version'] = probe_versions['glx'][glx_probe_version]
            meta['probe_version_int'] = probe_versions['int'][meta['probe_version']]
        else:
            print(f'WARNING probe version {glx_probe_version} not handled - post an issue at www.github.com/m-beau/NeuroPyxels and provide your .ap.meta file.')
            meta['probe_version'] = glx_probe_version
            meta['probe_version_int'] = 0
            

        # Based on probe version,
        # Find the voltage range, gain, encoding
        # and deduce the conversion from units/bit to uV
        Vrange=(meta_glx["highpass"]['imAiRangeMax']-meta_glx["highpass"]['imAiRangeMin'])*1e6
        if meta['probe_version'] in ['3A', '1.0', 'ultra_high_density', 'NHP_1.0']:
            if Vrange!=1.2e6: print(f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (1.2*10^6). Might be normal!')
            bits_encoding=10
            ampFactor=ale(meta_glx["highpass"]['~imroTbl'][1].split(' ')[3]) # typically 500
            #if ampFactor!=500: print(f'\u001b[31mHeads-up, the voltage amplification factor seems to be {ampFactor}, which is not the default (500). Might be normal!')
        elif meta['probe_version'] in ['2.0_singleshank', '2.0_fourshanks']:
            if Vrange!=1e6:
                print(f'\u001b[31mHeads-up, the voltage range seems to be {Vrange}, which is not the default (10^6). Might be normal!')
            bits_encoding=14
            ampFactor=80 # hardcoded
        else:
            raise ValueError(f"Probe version unhandled - bits_encoding unknown.")
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
                meta[filt_key]['sampling_rate']=float(meta_glx[filt_key]['imSampRate'])
            else:
                meta[filt_key]['sampling_rate']=float(meta_glx[meta_glx['typeThis'][:2]+'SampRate'])

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
        if (dp/'spike_times.npy').exists():
            t_end=np.load(dp/'spike_times.npy').ravel()[-1]
            meta['recording_length_seconds']=t_end/high_fs
        else:
            meta['recording_length_seconds'] = 'unkown'
    else:
        file_size = meta['highpass']['binary_byte_size']
        item_size = np.dtype(meta['highpass']['datatype']).itemsize
        nChans = meta['highpass']['n_channels_binaryfile']
        meta['recording_length_seconds'] = file_size/item_size/nChans/high_fs

    return meta


def chan_map(dp=None, y_orig='surface', probe_version=None):
    '''
    Returns probe channel map.
    Arguments:
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

    assert y_orig in ['surface', 'tip']
    if probe_version is None:
        assert dp is not None, "You need to provide either a path or a probe version!"
        probe_version=read_metadata(dp)['probe_version']

    if probe_version in ['3A', '1.0', '2.0_singleshank']:
        cm = predefined_chanmap(probe_version)
    else:
        probe_version='local'

    if probe_version=='local':
        if dp is None:
            raise ValueError("dp argument is not provided - when channel map is \
                             atypical and probe_version thus called 'local', \
                             the datapath needs to be provided to load the channel map.")
        dp = Path(dp)
        c_ind=np.load(dp/'channel_map.npy');cp=np.load(dp/'channel_positions.npy')
        cm=npa(np.hstack([c_ind.reshape(max(c_ind.shape),1), cp]), dtype=np.int32)

    if y_orig=='surface':
        cm[:,1:]=cm[:,1:][::-1]

    return cm

def predefined_chanmap(probe_version='1.0'):
    '''
    Returns predefined channel map.
    Arguments:
        - probe_version: None, 'local', '3A', '1.0' or '2.0_singleshank' (other types not handled yet, reach out to give your own!).
    Returns:
        - chan_map: array of shape (N_electrodes, 3).
                    1st column is channel indices, 2nd x position, 3rd y position
    '''

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

    return cm


#%% Binary file I/O, including sync channel

def get_binary_file_path(dp, filt_suffix='ap', absolute_path=True):
    f'''Return the path of the binary file (.bin) from a directory.

    Wrapper of get_glx_file_path:
    {get_glx_file_path.__doc__}
    '''

    if 'continuous' in os.listdir(dp):
        meta = read_metadata(dp)
        if 'ap' in filt_suffix:
            return f'{dp}/continuous/{meta["highpass"]["folder_name"][:-1]}/continuous.dat'
        if 'lf' in filt_suffix:
            return f'{dp}/continuous/{meta["lowpass"]["folder_name"][:-1]}/continuous.dat'
    else:
        return get_glx_file_path(dp, 'bin', filt_suffix, absolute_path)
    
def get_meta_file_path(dp, filt_suffix='ap', absolute_path=True):
    f'''Return the path of the meta file (.meta) from a directory.

    Wrapper of get_glx_file_path:
    {get_glx_file_path.__doc__}
    '''
    
    return get_glx_file_path(dp, 'meta', filt_suffix, absolute_path)

def get_glx_file_path(dp, suffix, filt_suffix='ap', absolute_path=True):
    '''Return the path of a spikeGLX file (.bin or .meta) from a directory.

    Arguments:
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
                  verbose=False, again=False, sample_span=int(1e6)):
    '''Unpacks neuropixels external input data, to align spikes to events.
    Arguments:
        - dp: str, datapath
        - output_binary: bool, whether to output binary sync channel as 0/1s
        - filt_key: str, 'highpass' or 'lowpass' (SpikeGLX: ap/lf, OIpenEphys: Neuropix-PXI-100.0/.1)
        - unit: str, 'seconds' or 'samples', units of returned onsets/offset times
        - verbose: bool, whether to print rich information
        - again: bool, whether to reload sync channel from binary file.
        - sample_span: int, number of samples to load at once (prevents memory errors). If -1, all file loaded at once.

    Returns:
        Dictionnaries of length n_channels = number of channels where threshold crossings were found, [0-16]
        - onsets: dict, {channel_i:np.arquitray(onset1, onset2, ...), ...} in 'unit'
        - offsets: dict, {channel_i:np.array(offset1, offset2, ...), ...} in 'unit'

    '''

    # process arguments
    dp = Path(dp)
    if assert_multi(dp):
        dp = Path(get_ds_table(dp)['dp'][0])
        if verbose: print(( 'Loading npix sync channel from a merged dataset - '
                           f'assuming temporal reference frame of dataset 0:\n{dp}'))

    assert filt_key in ['highpass', 'lowpass']
    filt_suffix = {'highpass':'ap', 'lowpass':'lf'}[filt_key]
    assert unit in ['seconds', 'samples']

    sync_dp = dp / 'sync_chan'
    meta    = read_metadata(dp)
    srate   = meta[filt_key]['sampling_rate'] if unit=='seconds' else 1

    # initialize variables
    fname   = ''
    onsets  = {}
    offsets = {}

    # proceed
    if meta['acquisition_software'] == 'OpenEphys':

        events_dirs = [p for p in (dp/'events').iterdir() if 'PXI' in str(p)]
        high_pass_dir = [p for p in events_dirs if ("AP" in str(p))|("100.0" in str(p))][0]
        low_pass_dir = [p for p in events_dirs if ("LF" in str(p))|("100.1" in str(p))][0]


        events_dir = high_pass_dir if filt_key=='highpass' else low_pass_dir
        for i, ttl_dir in enumerate(events_dir.iterdir()):
            timestamps = np.load(ttl_dir / "timestamps.npy")
            ttl_i = ttl_dir.name

            onsets  = {**onsets, **{ttl_i:timestamps}}
            offsets = {**offsets, **{ttl_i:"openephys dataset: only onsets available (see onsets dictionnary)"}}

        return onsets, offsets

    elif meta['acquisition_software'] == 'SpikeGLX':

        # Tries to load pre-saved onsets and offsets
        if sync_dp.exists() and not output_binary:
            if verbose: print(f'Sync channel extraction directory found: {sync_dp}\n')
            for file in os.listdir(sync_dp):

                if file.endswith("on_samples.npy"):
                    filt_suffix_loaded = file.split('.')[-2][:2]
                    # if samples are at the instructed sampling rate i.e. lf (2500) or ap (30000)!
                    if filt_suffix_loaded == filt_suffix:
                        if verbose: print(f'Sync channel onsets ({file}) file found and loaded.')
                        file_i = ale(file[-15])
                        onsets[file_i] = np.load(sync_dp/file)/srate

                elif file.endswith("of_samples.npy"):
                    filt_suffix_loaded = file.split('.')[-2][:2]
                    # if samples are at the instructed sampling rate i.e. lf (2500) or ap (30000)!
                    if filt_suffix_loaded == filt_suffix:
                        if verbose: print(f'Sync channel offsets ({file}) file found and loaded.')
                        file_i = ale(file[-15])
                        offsets[file_i] = np.load(sync_dp/file)/srate
            
            if any(onsets):
                # else, might be that sync_dp is empty
                return onsets, offsets

        # Tries to load pre-saved compressed binary
        if sync_dp.exists() and not again:
            if verbose: print((f"No file ending in 'on_samples.npy' with the right sampling rate ({filt_suffix}) "
                                "found in sync_chan directory: extracting sync channel from binary.\n"))
            npz_files = list_files(sync_dp, 'npz')
            if any(npz_files):
                for npz_file in npz_files:
                    filt_suffix_npz = npz_file.split('.')[-2][:2]
                    if filt_suffix_npz == filt_suffix:
                        if verbose: print(f'Compressed binary found {npz_file} - extracting from there...')
                        fname      = npz_file[:-9]
                        sync_fname = npz_file[:-4]
                        binary     = np.load(sync_dp/(sync_fname+'.npz'))
                        binary     = binary[dir(binary.f)[0]].astype(np.int8)

        else: os.mkdir(sync_dp)

        # If still no file name, memorymaps binary directly
        if fname=='':
            # find binary files
            ap_files = list_files(dp, 'ap.bin')
            lf_files = list_files(dp, 'lf.bin')

            if filt_suffix == 'ap':
                assert any(ap_files), f'No .ap.bin file found at {dp}!! Aborting.'
                fname=ap_files[0]
            elif filt_suffix == 'lf':
                assert any(lf_files), f'No .lf.bin file found at {dp}!! Aborting.'
                fname=lf_files[0]

            # preprocess binary file properties
            nchan    = int(meta[filt_key]['n_channels_binaryfile'])
            dt       = np.dtype(meta[filt_key]['datatype'])
            nsamples = os.path.getsize(dp/fname) / (nchan * dt.itemsize)
            assert nsamples == int(nsamples),\
                f'Non-integer number of samples in binary file given nchannels {nchan} and encoding {dt}!'
            nsamples = int(nsamples)

            # Loads binary in chunks of sample_span samples
            print(f'Loading {dt} data at {fname}...')
            if sample_span == -1:
                sample_span = nsamples
            else:
                assert isinstance(sample_span, int) and sample_span>0,\
                    'sample_span must be a strictly positive integer!'
            sample_slices = [[int(s_on), int(min(s_on+sample_span, nsamples))] \
                              for s_on in  np.arange(0, nsamples, sample_span)]
            syncdat       = np.zeros(nsamples, dtype=dt)
            for sample_slice in sample_slices:
                syncdat_slice = np.memmap(dp/fname,
                                            mode='r',
                                            dtype=dt,
                                            shape=(nsamples, nchan))[slice(*sample_slice),-1]
                syncdat[slice(*sample_slice)] = syncdat_slice.flatten()

            # unpack loaded int16 data into 
            print(f'Unpacking bits from {dt} data ...')
            binary     = unpackbits(syncdat, 8*dt.itemsize).astype(np.int8)
            sync_fname = fname[:-4]+'_sync'
            if is_writable(sync_dp):
                np.savez_compressed(sync_dp/(sync_fname+'.npz'), binary)

        if output_binary:
            return binary

        # Generates onsets and offsets from binary
        mult = 1
        sync_idx_onset  = np.where(mult*np.diff(binary, axis = 0)>0)
        sync_idx_offset = np.where(mult*np.diff(binary, axis = 0)<0)
        for ichan in np.unique(sync_idx_onset[1]):
            ons = sync_idx_onset[0][
                sync_idx_onset[1] == ichan]
            onsets[ichan] = ons
            if is_writable(sync_dp):
                np.save(Path(sync_dp, sync_fname+'{}on_samples.npy'.format(ichan)), ons)
        for ichan in np.unique(sync_idx_offset[1]):
            ofs = sync_idx_offset[0][
                sync_idx_offset[1] == ichan]
            offsets[ichan] = ofs
            if is_writable(sync_dp):
                np.save(Path(sync_dp, sync_fname+'{}of_samples.npy'.format(ichan)), ofs)

        onsets  = {ok:ov/srate for ok, ov in onsets.items()}
        offsets = {ok:ov/srate for ok, ov in offsets.items()}

        assert any(onsets), ("WARNING no sync channel found in dataset - "
            "make sure you are running this function on a dataset with a synchronization TTL!")

        return onsets,offsets

@npyx_cacher
def extract_rawChunk(dp, times, channels=np.arange(384),
                     filt_key='highpass', save=0,
                     whiten=0, med_sub=0, hpfilt=0, hpfiltf=300, filter_forward=True, filter_backward=True,
                     nRangeWhiten=None, nRangeMedSub=None, use_ks_w_matrix=True,
                     ignore_ks_chanfilt=True, center_chans_on_0=False, verbose=False, scale=True,
                     again=False, cache_results=False, cache_path=None):
    '''Function to extract a chunk of raw data on a given range of channels on a given time window.
    Arguments:
    - dp: datapath to folder with binary path (files must ends in .bin, typically ap.bin)
    - times: list of boundaries of the time window, in seconds [t1, t2].
    - channels (default: np.arange(384)): list of channels of interest, in 0 indexed integers [c1, c2, c3...]
    - filt_key: 'highpass' or 'lowpass', whether to extract from the high-pass or low-pass filtered binary file
    - save (default 0): save the raw chunk in the bdp directory as '{bdp}_t1-t2_c1-c2.npy'
    - whiten: whether to whiten the data across channels. If nRangeWhiten is not None,
              whitening matrix is computed with the nRangeWhiten closest channels.
    - med_sub: whether to median-subtract the data across channels. If nRangeMedSub is not none,
               median of each channel is computed using the nRangeMedSub closest channels.
    - hpfilt: whether to high-pass filter the data, using a 3 nodes butterworth filter of cutoff frequency hpfiltf, bidirectionally.
    - hpfiltf: see hpfilt
    - filter_forward: bool, filter the data forward (also set filter_backward to True for bidirectional filtering)
    - filter_backward: bool, filter the data backward (also set filter_forward to True for bidirectional filtering)
    - nRangeWhiten: int, see whiten.
    - nRangeMedSub: int, see med_sub.
    - use_ks_w_matrix: bool, whether to use kilosort's original whitening matrix to perform the whitening
                     (rather than recomputing it from the data at hand)
    - ignore_ks_chanfilt: whether to ignore the filtering made by kilosort,
                          which only uses channels with average events rate > ops.minfr to spike sort.
    - scale: A boolean variable specifying whether we should convert the resulting raw
             A2D samples to uV. Defaults to True
    - again: bool, whether to recompute results rather than loading them from cache.
    - cache_results: bool, whether to cache results at local_cache_memory.
    - cache_path: None|str, where to cache results.
                    If None, dp/.NeuroPyxels will be used.
                    
    Returns:
    - rawChunk: numpy array of shape ((c2-c1), (t2-t1)*fs).
                rawChunk[0,:] is channel 0; rawChunk[1,:] is channel 1, etc.
                dtype: int16 if scale=False, float64 if scale=True.
    '''
    # Find binary file
    dp = Path(dp)
    meta = read_metadata(dp)
    fname = get_binary_file_path(dp, filt_suffix='ap' if filt_key == 'highpass' else 'lf', absolute_path=True)

    fs = meta[filt_key]['sampling_rate']
    Nchans=meta[filt_key]['n_channels_binaryfile']
    bytes_per_sample=2
    whitenpad=200

    assert len(times)==2
    assert times[0]>=0
    assert times[1]<meta['recording_length_seconds']

    # Format inputs
    channels=assert_chan_in_dataset(dp, channels, ignore_ks_chanfilt)
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))
    if whiten:
        if t1<whitenpad:
            print(f"times[0] set to {round(whitenpad/30000, 5)}s because whitening requires a pad.")
            t1 = whitenpad
            times[0] = t1/30000
        t1, t2 = t1-whitenpad, t2+whitenpad
    bn = os.path.basename(fname) # binary name
    rcn = (f"{bn}_t{times[0]}-{times[1]}_ch{channels[0]}-{channels[-1]}"
          f"_{whiten}{nRangeWhiten}_{med_sub}{nRangeMedSub}_{hpfilt}{hpfiltf}"
          f"_{ignore_ks_chanfilt}_{center_chans_on_0}_{scale}.npy") # raw chunk name
    
    # DEPRECATED - now caching with cachecache
    # rcp = get_npyx_memory(dp) / rcn

    # if os.path.isfile(rcp) and not again:
    #     return np.load(rcp)

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
    rc = rc[:-1,:] # remove sync channel

    # Median subtraction = CAR
    if med_sub:
        rc=med_substract(rc, 0, nRange=nRangeMedSub)

    # get the right channels range,
    # after median subtraction
    if (whiten and use_ks_w_matrix) or not whiten:
        # saves computation time to preselect channels
        rc = rc[channels, :]

    # Highpass filter with a 3rd order butterworth filter
    if hpfilt:
        rc_t = np.ascontiguousarray(rc.T)
        rc_t = cp.asarray(rc_t)
        rc = gpufilter(rc_t, fs=fs, fslow=None, fshigh=hpfiltf, order=3,
             car=False, forward=filter_forward, backward=filter_backward, ret_numpy=True).T

    # Whiten data
    if whiten:
        channels_mask = np.isin(np.arange(Nchans-1), channels)
        rc=whitening(rc, nRangeWhiten, use_ks_w_matrix, dp, channels_mask)
        rc=rc[:, whitenpad:-whitenpad]

    # get the right channels range, AFTER WHITENING
    if whiten and not use_ks_w_matrix:
        rc = rc[channels, :]

    # Scale data
    if scale:
        rc = rc * meta['bit_uV_conv_factor'] # convert into uV

    # Align signal on each channel, option convenient for plotting
    if center_chans_on_0:
        rc=rc-np.median(rc[:,:10],axis=1)[:,np.newaxis]

    # eventually convert from cupy to numpy array
    # (necessary if whitened on GPU)
    if 'cp' in globals():
        rc = cp.asnumpy(rc)

    # DEPRECATED - now caching with cachecache
    # if save: # sync chan saved in extract_syncChan
    #     np.save(rcp, rc)

    return rc

def extract_binary_channel_subset(directory_with_binary,
                           chanmin,
                           chanmax,
                           batch_size_s=500,
                           filt_suffix='ap'):
    """Extract subset of channels from binary file into another binary file.

    Saves the extracted subset of channels to another binary file in the same directory,
    with the same name + the channel range appended to it.

    Arguments:
        - directory_with_binary: str, path to dataset (will find binary file in there)
        - chanmin: minimum channel
        - chanmax: end of channel range to extract, included
        - batch_size_s: size of data batches loaded and saved, in seconds
                        If memory error: decrease batch_size_s
        - filt_suffix: str, 'ap' or 'lf', whether to extract channels from the ap or lfp binary.
    """

    ## Compute channels to export and define file names
    binary_fn = get_binary_file_path(directory_with_binary, filt_suffix)
    assert binary_fn != "not_found",\
        f"Binary not found at {directory_with_binary}!"
    target_fname = f'_chan{chanmin}-{chanmax}'.join([binary_fn.name.split(f'.{filt_suffix}.bin')[0], f'.{filt_suffix}.bin'])
    target_fn = binary_fn.parent / target_fname
    
    ## Load binary metadata
    meta = read_metadata(directory_with_binary)
    filt_suffix_long = {'ap': 'highpass', 'lf': 'lowpass'}[filt_suffix]
    fs = meta[filt_suffix_long]['sampling_rate']
    dtype = np.dtype(meta[filt_suffix_long]['datatype'])
    item_size = dtype.itemsize
    Nchans = meta[filt_suffix_long]['n_channels_binaryfile']
    assert (chanmin < Nchans - 1) & (chanmax < Nchans - 1)
    
    ## precompute binary batch time windows
    filesize_bytes = os.path.getsize(binary_fn)
    filesize_samples = filesize_bytes / Nchans / item_size
    assert filesize_samples == int(filesize_samples),\
        f"It doesn't seem like the binary file {binary_fn} holds a multiple of {Nchans} channels encoded as {item_size} bytes items...!"
    filesize_samples = int(filesize_samples)
    
    batch_size_samples = int(batch_size_s * fs)
    Nbatch = int(np.ceil(filesize_samples / batch_size_samples))
    batch_windows = [[i * batch_size_samples,
                      i * batch_size_samples + batch_size_samples]
                     for i in range(Nbatch)]
    batch_windows[-1][-1] = filesize_samples - 1

    ## Run data extraction loop

    # Memory-map binary data
    memmap_f = np.memmap(binary_fn,
                         dtype=dtype,
                         offset=0,
                         shape=(filesize_samples, Nchans),
                         mode='r')
    
    with open(target_fn, 'wb') as fw:
        for (t1, t2) in tqdm(batch_windows, desc="Extracting channels"):
        
            # Read binary data
            rawData = memmap_f[t1:t2, chanmin:chanmax + 1]
        
            # Write binary data
            rawData.tofile(fw)

def read_custom_binary(fn, Nchans, dtype='int16'):
    """
    Returns a memory map of a neuropixels binary file with a custom number of channels.
    
    Arguments:
    - fn: str, path to binary file
    - Nchans: int, number of channels of binary file
    - dtype: str, datatype of saved binary data (original neuropixels 1.0 data: int16)

    Returns:
    - memmap_f: memory mapped binary file of shape (n_samples, Nchans).
                Index it as follow to extract data: memmap_f[time1:time2, channel1:channel2].
                Sampling rate is typically 30_000 Hz, check with source binary file.
    """
    dtype = np.dtype(dtype)
    item_size = dtype.itemsize
    filesize_bytes = os.path.getsize(fn)
    filesize_samples = int(filesize_bytes / Nchans / item_size)

    memmap_f = np.memmap(fn,
                     dtype=dtype,
                     offset=0,
                     shape=(filesize_samples, Nchans),
                     mode='r')
    
    return memmap_f

def assert_chan_in_dataset(dp, channels, ignore_ks_chanfilt=False):
    channels = np.array(channels)
    if ignore_ks_chanfilt:
        probe_version = read_metadata(dp)['probe_version']
        cm=chan_map(dp, probe_version=probe_version)
    else:
        cm=chan_map(dp, probe_version='local')
    if not np.all(np.isin(channels, cm[:,0])):
        print(("WARNING Kilosort excluded some channels that you provided for analysis "
               "because they did not display enough threshold crossings! Jumping channels:"
               f"{channels[~np.isin(channels, cm[:,0])]}"))
    channels=channels[np.isin(channels, cm[:,0])]
    return channels

#%% Binary file filtering wrappers

def detect_hardware_filter(dp):
    "Detects if the Neuropixels hardware filter was on or off during recording."
    imro_table = metadata(dp)['highpass']['~imroTbl']
    hpfiltered_int = int(imro_table[1][-1]) # 0 or 1
    return bool(hpfiltered_int)

def preprocess_binary_file(dp=None, filt_key='ap', fname=None, target_dp=None, move_orig_data=True,
                       ADC_realign = False, median_subtract=True, f_low=None, f_high=300, order=3,
                       filter_forward=True, filter_backward=False,
                       spatial_filt=False, whiten = False, whiten_range=32,
                       again_Wrot=False, verbose=False, again_if_preprocessed_filename=False,
                       delete_original_data=False, data_deletion_double_check=False):
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


    Arguments:
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
    - filter_forward: bool, filter the data forward (also set filter_backward to True for bidirectional filtering)
    - filter_backward: bool, filter the data backward (also set filter_forward to True for bidirectional filtering)
    - spatial_filt: bool, whether to high pass filter across channels at 0.1 Hz
    - whiten: bool, whether to whiten across channels
    - verbose: bool, whether to print extra information
    - again_if_preprocessed_filename: bool, whether to re-filter if the name of the binary file has hallmarks of preprocessing.
    - delete_original_data: bool, whether to delete the original binary file after filtering
    - data_deletion_double_check: bool, must ALSO be true to alow deletion of the original binary file
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

    if not again_if_preprocessed_filename and detected_preprocessed_fname(fname):
        print(f"\nBinary file name {fname} seems to have already been preprocessed. If you wish to re-process it, set 'again_if_preprocessed_filename' to True.\n")
        return fname

    fname=Path(fname)
    dp = fname.parent
    if target_dp is None:
        target_dp = dp
    else:
        target_dp = Path(target_dp)
    print(f"Preprocessing {fname}...")

    filtered_fname, message = \
        make_preprocessing_fname(fname, ADC_realign, median_subtract,
                            f_low, f_high, filter_forward, filter_backward,
                            whiten, whiten_range, spatial_filt)    
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
    memmap_f = np.memmap(fname, dtype=dtype, offset=offset, shape=(n_samples, n_channels), mode='r')


    # fetch whitening matrix (estimate covariance over a few batches)
    if whiten:
        raise ImplementationError("Whitening here is still experimental - cannot be used for now.")
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
    assert not (target_dp / filtered_fname).exists(),\
        f"WARNING file {target_dp / filtered_fname} exists already - to process again, delete or move it."
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
                              car=median_subtract, forward=filter_forward, backward=filter_backward)
            assert batch.flags.c_contiguous # check that ordering is still C, not F
            batch[ntb:2*ntb] = w_edge * batch[ntb:2*ntb] + (1 - w_edge) * buff_prev
            buff_prev = batch[NT + ntb: NT + 2*ntb]
            batch = batch[ntb:ntb + NT, :]  # remove timepoints used as buffers

            # Spatial filtering (replaces whitening)
            if spatial_filt:
                batch = kfilt(batch.T, butter_kwargs = {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}).T

            # whiten the data and scale by 200 for int16 range
            ## TODO implement whitening here
            # if whiten:
            #     batch = cp.dot(batch, Wrot)
            
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
    
    # Apparently Windows is unhappy if memory mapped files
    # aren't explicitely closed if trying to rename/move them
    # so... close the memory mapped file o_o
    memmap_f._mmap.close()

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
            if (dp/'whitening_mat.npy').exists():
                shutil.copy(dp/'whitening_mat.npy', orig_dp/'whitening_mat.npy')

    if delete_original_data:
        assert data_deletion_double_check,\
        "WARNING you are attempting to delete the original binary file - if you wish to proceed, you must also set 'data_deletion_double_check' to True."
        if move_orig_data:
            shutil.rmtree(dp/'original_data')
        else:
            os.remove(fname)
    else:
        if data_deletion_double_check:
            print("WARNING you attempted to delete the original binary file - 'delete_original_data' was not set to True, so the deletion was cancelled.")

    return target_dp/filtered_fname

def make_preprocessing_fname(fname, ADC_realign, median_subtract,
                            f_low, f_high, filter_forward, filter_backward,
                            whiten, whiten_range, spatial_filt):
        
    filter_suffix = ""
    message = ""
    if ADC_realign:
        filter_suffix+=f"_adcshift{ADC_realign}"
        message+="    - shifting ADCs,\n"
    if median_subtract:
        filter_suffix+="_medsub"
        message+="    - median subtraction (aka common average referencing CAR),\n"
    filter_suffix+=f"_tempfilt{f_low}{f_high}{filter_forward}{filter_backward}"
    low_s = 0 if f_low is None else f_low
    message+=(f"    - filtering in time (between {low_s} and {f_high} Hz)"
              f" forward:{filter_forward}, backward:{filter_backward},\n")
    if whiten:
        filter_suffix+=f"_whit{whiten}{whiten_range}"
        message+=f"    - whitening (using {whiten_range} closest channels),\n"
    if spatial_filt:
        filter_suffix+=f"_spatfilt{spatial_filt}"
        message+=f"    - filtering in space ({spatial_filt} 'Hz'),\n"
    filtered_fname = str(fname.name)[:-7]+filter_suffix+".ap.bin"
    message = message[:-2]+"."

    return filtered_fname, message

def detected_preprocessed_fname(fname):

    fname_str = str(fname)
    patterns = ["adcshift", "medsub", "tempfilt", "whit"]
    for pat in patterns:
        if pat in fname_str:
            return True
    return False

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

class ImplementationError(Exception):
    pass

from npyx.gl import assert_multi, get_ds_table, get_npyx_memory
from npyx.preprocess import (
    adc_realign,
    approximated_whitening_matrix,
    gpufilter,
    kfilt,
    med_substract,
    whitening,
)
