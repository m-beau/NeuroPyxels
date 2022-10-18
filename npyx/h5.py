from pathlib import Path
import re
import sys
import warnings

import numpy as np 

import json
import h5py

from npyx.utils import assert_int, assert_float
from npyx.inout import get_binary_file_path, get_npix_sync, chan_map, extract_rawChunk, read_metadata, preprocess_binary_file
from npyx.spk_t import ids, trn, trn_filtered
from npyx.spk_wvf import wvf_dsmatch
from npyx.gl import get_units, check_periods



# High level C4 functions
def label_optotagged_unit_h5(h5_path, dataset, unit, label, prnt=False):
    """
    Add optotagged label to neuron.

    - h5_path: full path to h5 file
    - dataset: str, neuron dataset (yy-mm-dd_mouse_probex)
    - unit: neuron unit index
    - label: label to add
    """
    authorized_labels = ["PkC_ss", "PkC_cs", "MLI", "MFB", "GoC", "GrC"]
    assert label in authorized_labels, f"{label} must match either of the following: {authorized_labels}"
    add_data_to_unit_h5(h5_path, dataset, unit, label, 'optotagged_label') 
    if prnt: print(f"Labelled unit {unit} as {label}.")

def reset_optotagged_labels(h5_path):
    """Resets all optotagged labels to 0"""
    with h5py.File(h5_path, "a") as h5_f:
        for neuron in h5_f.keys():
            if 'hausser_neuron' not in neuron: continue
            data_path = f"{neuron}/optotagged_label"
            write_to_dataset(h5_f, data_path, 0, overwrite=True)

def add_units_to_h5(h5_path, dp, units=None, **kwargs):
    f"""
    Add all or specified units at the respective data path to an HDF5 file.

    This is a high-level function designed to add many units at the
    specified datapath to an HDF5 file. All additional key-value 
    arguments are passed to `add_unit_h5`

    Example:
      add_units_to_h5('my_lab_data.h5', '/path/to/dataset_id', lab_id='pi_last_name')
    Will add all sorted units in the 'kilosort_results' directory 
    to the HDF5 file called 'my_lab_data.h5' (in the current directory).

    Other arguments from add_unit_h5:
    {add_unit_h5.__doc__}
    """

    if units is None:
        units = get_units(dp)
    
    for u in units:
        add_unit_h5(h5_path, dp, u, **kwargs)

def add_unit_h5(h5_path, dp, unit, lab_id, periods='all',
                unit_abolute_id=None, sync_chan_id=None,
                again=False, again_wvf=False, plot_debug=False, verbose=False,
                dataset=None,
                raw_window=[0.1, 30.1], center_raw_window_on_spikes=True,
                include_raw_snippets=True, include_whitened_snippets=True,
                raw_snippet_halfrange=2, mean_wvf_half_range=11,
                sane_spikes=None, sane_before_opto=False, include_fp_fn_mask=True,
                optostims=None, optostims_from_sync=False, optostims_threshold=None,
                n_waveforms_for_matching=5000,
                **kwargs):
    """
    Add a spike-sorted unit to an HDF5 file (on a phy compatible dataformat).

    Adds a spike-sorted unit to a new or existing HDF5 five file using the
    file format specified by the C4 collaboration.
    
     ---->>> data format details here www.tinyurl.com/c4database <<<---

    Each unit can be accessed from 2 paths which point to the same data:
    - an absolute path, {unit_abolute_id}/
                        which allows for a flat hierarchy and makes it easier to compute features,
                        easier to work on data from other labs
    - a relative path, datasets/{dataset}/{unit}/
                       which allows to index units per dataset,
                       easier to work on your own data

    Example:
        add_unit_h5('my_lab_data.h5', '/path/to/kilosort_results', 1, lab_id='pi_last_name')
    Adds the unit with id 1 to the HDF5 file in the current directory called 'my_lab_data.h5'.
    Additional units can be added by calling the same function with identical arguments,
    but incrementing the unit id field.

    Required parameters:
    - h5_path: Path to the h5 file to create/append
    - dp: Path the Kilosort data directory
    - unit: The unit id/neuron unit index
    - lab_id: The lab/PI id to use to label the units
    - periods: 'all' or [[t1,t2],[t3,t4],...] in seconds

    Key-value parameters:
    - unit_absolute_id: unit absolute id. Will increment from the last unit added to h5 file.
    - sync_chan_id: The channel id used to denote opto stimulation. Defaults to None.
    - again: Whether to use cached results for storage in the HDF5 file (defaults to False)
    - again_wvf: Whether to recompute drift-shift matched waveform in particular (very computationally intensive, defaults to False)
    - verbose: Additional verbosity/progress
    - dataset: A unique ID for this dataset. By default this value is None, in which case
      the dataset id is assumed to the dirname of the data directory passed as the dp argument
      
    - raw_window: A two item list containing the start and stop times (in SECONDS) of the snippet of raw data used for
                  1) computing noise RMS (-> SNR)
                  2) extracting representative examples of raw data (raw, and whitened).
    - center_raw_window_on_spikes: bool, whether to roughly center the raw voltage snippets window (raw_window) on neuron's first spike
    - raw_snippet_halfrange: int, range of channels around peak channel to consider for the snippet of raw data (max 10)
    - include_raw_snippets: bool, whether to include (memory heavy) raw data snippets (normally median subtracted and forward high pass filtered).
    - include_whitened_snippets: bool, whether to include (memory heavy) preprocessed (whitened, filtered backward) data snippets (raw).
    
    - sane_spikes: optional bool array, custom definition of 'sane spikes' for whatever reason.
                   By default, will be all spikes within 'periods'.
    - sane_before_opto: bool, whether to consider all spikes before the 1st optostim
                        to constitute the sane_spikes boolean mask.
                        Will only work if optostims are provided (or alternatively optostims_from_sync is True.)
    - include_fp_fn_mask: bool, whether to compute the false positive and false negative rates for recording periods
                          and subsequently compute the 'fn_fp_filtered_spikes', which is True for spikes in periods
                          passing the 5% fp and 5% fn quality requirement and False for other spikes.
                   
    - optostims: an optional 2D array (n_stims, 2) containing the optostimuli times in seconds
                 (1st column: onsets, 2nd column: offsets).
                 By default None, will be read from sync channel (at sync_chan_id) if optostims_from_sync is True.
    - optostims_from_sync: bool, whether to pick up optostims from sync channel if None are provided.
    - optostims_threshold: float, time before which optostims will be ignored
                           (handles sync signals without light at beginning of recording)
                           
    - n_waveforms_for_matching: int, number of waveforms to subsample for drift-shift matching

    Additional key-value parameteters:
    - *any_key* = *any_value*
    All additional key-value parameters passed to this function are stored in the HDF5 file.
    Therefore, custom keys can be stored in the HDF5 file should additional
    information be required for an individual neuron. E.g., calling
        add_unit_h5('my_lab_data.h5', '/path_to_kilosort_results', 1, my_note="Cool info")
    will result in a key of 'my_note' and a value of "Cool info" being stored in the HDF5 file
    for this unit.
    """
    
    dp=Path(dp)
    meta = read_metadata(dp) 
    samp_rate = meta['highpass']['sampling_rate']

    # hard-coded parameters
    waveform_samples = 6  # ms
    waveform_samples = int(waveform_samples*samp_rate/1000)

    # open file in append mode
    h5_path = Path(h5_path)
    assert_h5_file(h5_path)
    with h5py.File(h5_path, "a") as h5_file:

        # check whether neuron already exists in dataset
        if dataset is None:
            dataset = dp.name
        check_dataset_format(dataset)
        neuron_path = f"datasets/{dataset}/{unit}"
        if neuron_path in h5_file:
            neuron_absolute_path=h5_file[f'{neuron_path}/neuron_absolute_id'][()].decode()
            if again:
                del h5_file[neuron_path]
                del h5_file[neuron_absolute_path]
            else:
                print(f"Neuron found in h5 file: {neuron_path} ({neuron_absolute_path})")
                neuron_group = h5_file[neuron_absolute_path]
                #return neuron_path
            
        if neuron_path not in h5_file or again:
            # figure out where we're at
            if unit_abolute_id is None:
                if f"{lab_id}_neuron_0" not in h5_file:
                    unit_abolute_id = 0
                else:
                    root_groups = list(h5_file.keys())
                    neuron_ids = [int(x.split('_')[-1]) for x in root_groups if f"{lab_id}_neuron" in x]
                    unit_abolute_id = np.sort(neuron_ids)[-1] + 1

            # create group for new neuron
            neuron_absolute_path=f'{lab_id}_neuron_{unit_abolute_id}'
            neuron_group = h5_file.create_group(neuron_absolute_path)
            h5_file[neuron_path] = neuron_group
            print(f"Adding data at {neuron_path} ({neuron_absolute_path})...")

        # metadata
        write_to_group(neuron_group, 'lab_id', lab_id, again)
        write_to_group(neuron_group, 'dataset_id', dataset, again)
        write_to_group(neuron_group, 'neuron_id', unit, again)
        write_to_group(neuron_group, 'neuron_absolute_id', neuron_group.name, again)
        write_to_group(neuron_group, 'sampling_rate', samp_rate, again)
        write_to_group(neuron_group, 'periods', samp_rate, again)
        
        # add any additional keys passed to this function
        for key, value in kwargs.items():
            write_to_group(neuron_group, key, value, again)

        # spike_times
        periods = check_periods(periods)
        if 'spike_indices' not in neuron_group or again:
            t = trn(dp, unit, periods=periods, again=again)
            write_to_group(neuron_group, 'spike_indices', t, again)
            
            if optostims is None and optostims_from_sync:
                ons, offs = get_npix_sync(dp, verbose=False)
                if sync_chan_id is None:
                    sync_chan_id = get_stim_chan(ons)
                ons, offs = ons[sync_chan_id], offs[sync_chan_id]
                if ons[0] > offs[0]:
                    ons, offs = offs, ons 
                if len(offs) == len(ons) - 1:
                    offs = np.append(offs, meta['recording_length_seconds'])
                optostims = np.hstack([ons[:, None], offs[:, None]])
            if optostims is not None and optostims_threshold is not None:
                opto_m = optostims[:,0] > optostims_threshold
                optostims = optostims[opto_m,:]
                
            if sane_spikes is None and\
                optostims is not None and\
                sane_before_opto:
                # Only consider spikes 10s before first opto onset
                sane_spikes = (t < (optostims[0,0]-10)*samp_rate)
            else:
                sane_spikes = (t*0+1).astype(bool)
                
        if optostims is None:
            optostims = np.array([])
        write_to_group(neuron_group, 'optostims', optostims, again)
        write_to_group(neuron_group, 'sane_spikes', sane_spikes, again)

        # waveforms
        k = ['mean_waveform_preprocessed', 'amplitudes', 'voltage_sample', 'peakchan_SNR']
        if not all_keys_in_group(k, neuron_group) or again:
            # must recompute chan_bottom and chan_top - suboptimal, can be rewritten
            dsm_tuple = wvf_dsmatch(dp, unit, t_waveforms=waveform_samples, periods=periods,
                                    again=again_wvf, plot_debug=plot_debug, verbose=verbose,
                                    n_waves_used_for_matching=n_waveforms_for_matching)
            dsm_waveform, peak_chan = dsm_tuple[1], dsm_tuple[3]
            write_to_group(neuron_group, 'primary_channel', peak_chan)
            chan_range = np.arange(peak_chan-mean_wvf_half_range, peak_chan+mean_wvf_half_range)
            chan_range_m = (chan_range>=0)&(chan_range<=383)
            chan_bottom, chan_top = chan_range[chan_range_m][0], chan_range[chan_range_m][-1]
            peak_chan_rel = np.nonzero(peak_chan == chan_range[chan_range_m])[0]
            dsm_waveform_chunk = dsm_waveform[:, chan_bottom:chan_top]
            write_to_group(neuron_group, 'mean_waveform_preprocessed',
                           dsm_waveform_chunk.T, again)
            write_to_group(neuron_group, 'consensus_waveform',
                           dsm_waveform_chunk.T*np.nan, again)
            cm = chan_map(dp)
            write_to_group(neuron_group, 'channel_ids',
                           np.arange(chan_bottom, chan_top, dtype=np.dtype('uint16')),
                           again)
            write_to_group(neuron_group, 'channelmap', cm[chan_bottom:chan_top, 1:2], again)

        # Extract voltage snippets
        k = ['amplitudes', 'channel_noise_std', 'peakchan_SNR', 'voltage_sample']
        if not all_keys_in_group(k, neuron_group)\
            or again:
            if center_raw_window_on_spikes:
                t = h5_file[neuron_path+'/spike_indices'][...]/samp_rate
                if raw_window[1]>t[0]: # spike starting after end of original window
                    raw_window = np.array(raw_window)+t[0]
                    raw_window[1]=min(raw_window[1], t[-1])
            chunk = extract_rawChunk(dp, raw_window, channels=np.arange(chan_bottom, chan_top), 
                                        scale=False, med_sub=False, whiten=False, center_chans_on_0=False,
                                        hpfilt=False, verbose=False)

        # quality metrics
        k = ['amplitudes', 'channel_noise_std', 'peakchan_SNR']
        if not all_keys_in_group(k, neuron_group) or again:
                
            amps = np.load(dp/'amplitudes.npy').squeeze()[ids(dp, unit, periods=periods)]
            
            mad = np.median(np.abs(chunk) - np.median(chunk, axis=1)[:, None], axis=1) 
            std_estimate = (mad / 0.6745) # Convert to std
            peakchan_S = np.ptp(dsm_waveform[:,peak_chan])
            peakchan_N = std_estimate[peak_chan_rel]
            peakchan_SNR = peakchan_S / peakchan_N
            
            write_to_group(neuron_group, 'amplitudes', amps, again)
            write_to_group(neuron_group, 'channel_noise_std', std_estimate, again)
            write_to_group(neuron_group, 'peakchan_SNR', peakchan_SNR, again)
    
        if ('fn_fp_filtered_spikes' not in neuron_group or again) and include_fp_fn_mask:
            # get good spikes mask for all spikes
            # because trn_filtered can only work on a contiguous chunk
            if periods is 'all':
                periods_m_range = [0, meta['recording_length_seconds']/60]
            else:
                periods_m_range = [periods.min()/60, periods.max()/60]
            fp_fn_good_spikes = trn_filtered(dp, unit, plot_debug=plot_debug,
                                             again=again, period_m=periods_m_range)[1]


            # if periods is not all, trim down the mask to spikes in periods
            if periods is not 'all':
                t = trn(dp, unit, periods=periods) # if again, as recomputed just above anyway, so don't pass the argument
                t_all = trn(dp, unit) # grab all spikes
                periods_mask = np.isin(t_all, t)
                fp_fn_good_spikes = fp_fn_good_spikes[periods_mask]

            write_to_group(neuron_group, 'fn_fp_filtered_spikes', fp_fn_good_spikes, again)
            
        # voltage snippets
        if ('voltage_sample' not in neuron_group or again)\
            and include_raw_snippets:
            # Only store the voltage sample for the primary channel
            peak_chan = neuron_group['primary_channel']
            raw_snippet_halfrange = np.clip(raw_snippet_halfrange, 0, 10)
            c1 = max(0,int(chunk.shape[0]/2-raw_snippet_halfrange))
            c2 = min(chunk.shape[0]-1, int(chunk.shape[0]/2+raw_snippet_halfrange+1))
            raw_snippet = chunk[c1:c2,:]
            write_to_group(neuron_group, 'voltage_sample', raw_snippet) # still centered on peak channel, but half the size
            write_to_group(neuron_group, 'voltage_sample_start_index', int(raw_window[0] * samp_rate))
            write_to_group(neuron_group, 'scaling_factor', meta['bit_uV_conv_factor'])
            
        if ('whitened_voltage_sample' not in neuron_group or again)\
            and include_whitened_snippets:
            if center_raw_window_on_spikes:
                t = h5_file[neuron_path+'/spike_indices'][...]/samp_rate
                if raw_window[1]>t[0]: # spike starting after end of original window
                    raw_window = np.array(raw_window)+t[0]
                    raw_window[1]=min(raw_window[1], t[-1])
            # check that file filtered properly
            bin_f = get_binary_file_path(dp)
            if 'medsub' not in bin_f.name:
                warnings.warn((f"WARNING file {bin_f.name} is expected to have been median subtracted,"
                                " but its file name does not contain medsub..."))
            if 'tempfiltNone300TrueFalse' not in bin_f.name:
                warnings.warn((f"WARNING file {bin_f.name} is expected to have been highpass filtered forward only at 300Hz,"
                                " but its file name does not contain tempfiltNone300TrueFalse..."))
            # reprocess it
            white_chunk = extract_rawChunk(dp, raw_window, channels=np.arange(chan_bottom, chan_top), 
                                    scale=True, med_sub=False, hpfilt=True, filter_forward=False, filter_backward=True,
                                    whiten=True, use_ks_w_matrix=True,
                                    verbose=False)
            raw_snippet_halfrange = np.clip(raw_snippet_halfrange, 0, 10)
            c1 = max(0,int(white_chunk.shape[0]/2-raw_snippet_halfrange))
            c2 = min(white_chunk.shape[0]-1, int(white_chunk.shape[0]/2+raw_snippet_halfrange+1))
            raw_snippet = white_chunk[c1:c2,:].astype(np.float32)
            write_to_group(neuron_group, 'whitened_voltage_sample', raw_snippet)

        # layer
        write_to_group(neuron_group, 'phyllum_layer', 0, again)
        write_to_group(neuron_group, 'human_layer', 0, again)

        # ground truth labels
        write_to_group(neuron_group, 'expert_label', 0, again)
        write_to_group(neuron_group, 'optotagged_label', 0, again)

        # predicted labels
        write_to_group(neuron_group, 'lisberger_label', 0, again)
        write_to_group(neuron_group, 'hausser_label', 0, again)
        write_to_group(neuron_group, 'medina_label', 0, again)

    return neuron_path

def load_json_datasets(json_path, include_missing_datasets=False):
    with open(json_path) as f:
        json_f = json.load(f)

    print(f"\nLoading data from file {json_path}...\n")

    DSs = {}
    for ds in json_f.values():

        for key in ['dp', 'ct', 'units', 'ss', 'cs']:
            assert key in ds, f"{key} not in json file for dataset #{ds}!"
        
        dp=Path(ds['dp'])
        
        if not dp.exists():
            print(f"Dataset {dp} not found on system!\n")
            if include_missing_datasets: DSs[dp.name] = ds
            continue
        DSs[dp.name] = ds
        units = list(ds['units'])
        ss = list(ds['ss'])
        cs = list(ds['cs'])
        print(f"Dataset {dp} found with opto responsive units {units}, simple spikes {ss}, complex spikes {cs}.\n")

        units_m=np.isin(units+ss+cs, get_units(dp))
        assert all(units_m), f"Units {np.array(units)[~units_m]} not found in {dp}!"

    return DSs


def add_json_datasets_to_h5(json_path, h5_path, preprocess_binary = True, lab_id="hausser",
                            delete_original_data=False, data_deletion_double_check=False):

    DSs = load_json_datasets(json_path, include_missing_datasets=False)

    for ds_name, ds in DSs.items():
        dp=Path(ds['dp'])

        if preprocess_binary:
            preprocess_binary_file(dp,
                delete_original_data=delete_original_data,
                data_deletion_double_check=data_deletion_double_check)

        optolabel=ds['ct']
        if optolabel=="PkC": optolabel="PkC_ss"
        units=ds['units']
        ss=ds['ss']
        cs=ds['cs']
        for u in units+ss+cs:
            add_unit_h5(h5_path, dp, u, lab_id=lab_id,
                    unit_abolute_id=None, sync_chan_id=None,
                    again=False, plot_debug=False, verbose=True)
            if u in units:
                label = optolabel
            elif u in ss:
                label="PkC_ss"
            elif u in cs:
                label="PkC_cs"
            label_optotagged_unit_h5(h5_path, ds_name, u, label)


def add_data_to_unit_h5(h5_path, dataset, unit, data, field):
    """
    Add data to neuron already in h5 file.

    - h5_path: full path to h5 file
    - dataset: str, neuron dataset (yy-mm-dd_mouse_probex)
    - unit: unit index
    - data: data to add to unit
    - field: name of dataset to add data (id exists already, will overwrite)
    """
    check_dataset_format(dataset)
    neuron_path = f"datasets/{dataset}/{unit}"
    with h5py.File(h5_path, "a") as h5_file:
        assert neuron_path in h5_file, f"WARNING unit {neuron_path} does not seem to be present in the file. To add it, use add_unit_h5()."
        write_to_group(h5_file[neuron_path], field, data, True)

# h5 vizualisattion functions
def print_h5_contents(h5_path, txt_output=False):
    """
    h5_path: str, path to .h5 file
    txt_output: bool, if True prints contents to file
    (same name as h5 name_content.txt)
    """
    h5_path = Path(h5_path)
    if txt_output:
        txt_output_path = h5_path.parent / f"{h5_path.name[:-3]}_content.txt"
    with h5py.File(h5_path, "a") as hdf:
        if txt_output:
            with open(txt_output_path, "w") as txt:
                original_stdout = sys.stdout
                sys.stdout = txt
                visititems(hdf, visitor_func)
                sys.stdout = original_stdout
        else:
            visititems(hdf, visitor_func)

def visititems(group, func):
    with h5py._hl.base.phil:
        def proxy(name):
            """ Call the function with the text name, not bytes """
            name = group._d(name)
            return func(name, group[name])
        return group.id.links.visit(proxy)
       
def visitor_func(name, node):
    """
    prints name followed by a meangingful description of an hdf5 node.
    Node is either an h5 dataset (array, string...) or a group.
    """
    if isinstance(node, h5py.Dataset):
        n=node[()]
        if isinstance(n, bytes):
            s=n.decode()
        elif isinstance(n, np.ndarray):
            s=f"ndarray {n.shape}"
        elif assert_int(n) or assert_float(n):
            s=n
        else:
            s=type(n)
        string = f"{name}: {s}"
    else:
        string = name
    print(string)

def check_dataset_format(dataset):
    """
    Checks whether dataset name is formatted properly
    i.e. aa-mm-dd_iiXXX_probeX (mouse can only be names )
    """
    warning = "WARNING last folder of path should match format: aa-mm-dd_ii[0-1000]_probe[0-9] (ii = 2 [a-z] initials)"
    pattern = "[0-9]{2}-[0-9]{2}-[0-9]{2}_[a-z]{2}[0-9]{3}_probe[0-9]"
    if re.match(pattern, dataset, re.IGNORECASE) is None:
        warnings.warn(warning)

def assert_h5_file(h5_path):
    assert check_h5_file(h5_path), f"WARNING file at {h5_path} is not a .h5 file."

def check_h5_file(h5_path):
    """
    Check whether h5_path indeed points to h5
    returns True or False
    """
    return h5_path.name[-3:] == '.h5'

# h5 writing functions
def write_to_h5(h5_path, data_path, data,
                overwrite=False, must_exist=False):
    """
    Writes data at data_path to .h5 file at h5_path
    (creates non existing groups and datasets).
    """
    assert_h5_file(h5_path)
    with h5py.File(h5_path, "a") as h5_file:
        write_to_group(h5_file, data_path, data,
                       overwrite, must_exist)

def write_to_group(group, dataset, data,
                         overwrite=True, must_exist=False):
    """Write data to hdf5 group
    i.e. create a dataset +/- groups on the path to dataset
    and write data to this dataset.
    Parameters:
    - group: h5py group
    - dataset: str, name of dataset
    - data: data to add to dataset
    - overwrite: bool, whether to overwrite pre-existing dataset
    - must_exist: bool, whether to raise an error if dataset does not pre-exist
    """
    if dataset in group:
        if overwrite:
            del group[dataset]
        else:
            return
    elif must_exist:
        raise KeyError(f"Dataset {dataset} does not exist in group {group.name}!")
    group[dataset] = data

def write_to_dataset(group, dataset, data, overwrite=True):
    """write_to_groupwrite_to_group
    Write data to pre-existing dataset in group.
    Will crash if dataset does not exist in group.
    """
    write_to_group(group, dataset, data,
                         overwrite, True)

# h5 reading functions
def read_h5(h5_path, datapath):
    """
    Returns data at datapath from h5 file at h5_path
    """

    h5_path = Path(h5_path)
    
    assert_h5_file(h5_path)
    with h5py.File(h5_path) as h5_file:
        assert datapath in h5_file, f"WARNING {datapath} not found in {h5_path}"
        data = h5_file[datapath][()]
        if isinstance(data, bytes):
            data = data.decode() # for strings
    return data
    
def h5_group_keys(group):
    """
    Returns list of keys of h5 file group
    to allow easy overview of group content.
    """
    return list(group.keys())

def all_keys_in_group(keys, group):
    assert isinstance(keys, list)
    b = True
    for k in keys:
        b = b & (k in group)
    return b

# C4 utilities
def get_stim_chan(ons, min_th=20):
    chan = -1
    for k, v in ons.items():
        if len(v) > min_th:
            chan = k
    assert chan != -1
    return chan