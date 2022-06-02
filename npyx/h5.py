import h5py
from pathlib import Path
import re
import sys
import warnings

import numpy as np 

from npyx.utils import assert_int, assert_float
from npyx.inout import get_npix_sync, chan_map, extract_rawChunk, read_metadata
from npyx.spk_t import ids, trn, trn_filtered
from npyx.spk_wvf import wvf_dsmatch
from npyx.gl import get_units, check_periods

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

def label_optotagged_unit_h5(h5_path, dataset, unit, label):
    """
    Add optotagged label to neuron.

    - h5_path: full path to h5 file
    - dataset: str, neuron dataset (yy-mm-dd_mouse_probex)
    - unit: neuron unit index
    - label: label to add
    """
    authorized_labels = ["PkC_ss", "PkC_cs", "MLI", "MFB", "GoC", "GrC"]
    assert label in authorized_labels
    add_data_to_unit_h5(h5_path, dataset, unit, label, 'optotagged_label') 

def add_unit_h5(h5_path, dp, unit, lab_id, periods=[[0,20*60]],
                unit_abolute_id=None, sync_chan_id=None,
                again=False, again_wvf=False, plot_debug=False, verbose=False,
                dataset=None, snr_window=[0.1, 30.1], raw_snippet_halfrange=2,
                optostims=None, optostims_threshold=None,
                **kwargs):
    """
    Add a Kilosort sorted unit to an HDF5 file.

    Adds a Kilosort unit to a new or existing HDF5 five file using the
    file format specified by the C4 collaboration.

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
    - snr_window: A two item list containing the start and stop times (in seconds) for computation of the
      snr/voltage clip/waveform results
    - raw_snippet_halfrange: int, range of channels around peak channel to consider for the snippet of raw data (max 10)
    - optostims: an optional 2D array (n_stims, 2) containing the optostimuli times in seconds
                 (1st column: onsets, 2nd column: offsets). By default None, will be read from sync channel (at sync_chan_id)
    - optostims_threshold: float, time before which optostims will be ignored
                           (handles sync signals without light at beginning of recording)

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
    h5_file = h5py.File(h5_path, "a")

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
    add_dataset_to_group(neuron_group, 'lab_id', lab_id, again)
    add_dataset_to_group(neuron_group, 'dataset_id', dataset, again)
    add_dataset_to_group(neuron_group, 'neuron_id', unit, again)
    add_dataset_to_group(neuron_group, 'neuron_absolute_id', neuron_group.name, again)
    add_dataset_to_group(neuron_group, 'sampling_rate', samp_rate, again)
    
    # add any additional keys passed to this function
    for key, value in kwargs.items():
        add_dataset_to_group(neuron_group, key, value, again)

    # spike_times
    periods = check_periods(periods)
    if 'spike_indices' not in neuron_group or again:
        t = trn(dp, unit, periods=periods, again=again)
        add_dataset_to_group(neuron_group, 'spike_indices', t, again)
        if optostims is None:
            ons, offs = get_npix_sync(dp, verbose=False)
            if sync_chan_id is None:
                sync_chan_id = get_stim_chan(ons)
            ons, offs = ons[sync_chan_id], offs[sync_chan_id]
            if ons[0] > offs[0]:
                ons, offs = offs, ons 
            if len(offs) == len(ons) - 1:
                offs = np.append(offs, meta['recording_length_seconds'])
            optostims = np.hstack([ons[:, None], offs[:, None]])
        if optostims_threshold is not None:
            opto_m = optostims[:,0] > optostims_threshold
            optostims = optostims[opto_m,:]
        add_dataset_to_group(neuron_group, 'optostims', optostims, again)
        # Only consider spikes 10s before first opto onset
        sane_spikes = (t < (optostims[0,0]-10)*samp_rate)
        add_dataset_to_group(neuron_group, 'sane_spikes', sane_spikes, again)
        
    if 'fn_fp_filtered_spikes' not in neuron_group or again:
        # get good spikes mask for all spikes
        # because trn_filtered can only work on a contiguous chunk
        if periods is 'all':
            periods_m_range = [0, meta['recording_length_seconds']/60]
        else:
            periods_m_range = [periods.min()/60, periods.max()/60]
        fp_fn_good_spikes = trn_filtered(dp, unit, plot_debug=plot_debug, again=again, period_m=periods_m_range)[1]


        # if periods is not all, trim down the mask to spikes in periods
        if periods is not 'all':
            t = trn(dp, unit, periods=periods) # if again, as recomputed just above anyway, so don't pass the argument
            t_all = trn(dp, unit) # grab all spikes
            periods_mask = np.isin(t_all, t)
            fp_fn_good_spikes = fp_fn_good_spikes[periods_mask]

        add_dataset_to_group(neuron_group, 'fn_fp_filtered_spikes', fp_fn_good_spikes, again)

    # waveforms
    if 'mean_waveform_preprocessed' not in neuron_group\
        or ('amplitudes' not in neuron_group)\
        or ('voltage_sample' not in neuron_group)\
        or again: # must recompute chan_bottom and chan_top - suboptimal, can be rewritten
        dsm_tuple = wvf_dsmatch(dp, unit, t_waveforms=waveform_samples, periods=periods,
                                again=again_wvf, plot_debug=plot_debug, verbose=verbose, n_waves_used_for_matching=500)
        dsm_waveform, peak_chan = dsm_tuple[1], dsm_tuple[3]
        add_dataset_to_group(neuron_group, 'primary_channel', peak_chan)
        chan_bottom = max(0, peak_chan-11)
        chan_top = min(383, peak_chan+11)
        dsm_waveform_chunk = dsm_waveform[:, chan_bottom:chan_top]
        add_dataset_to_group(neuron_group, 'mean_waveform_preprocessed', dsm_waveform_chunk.T, again)
        add_dataset_to_group(neuron_group, 'consensus_waveform', dsm_waveform_chunk.T*np.nan, again)
        cm = chan_map(dp)
        add_dataset_to_group(neuron_group, 'channel_ids', np.arange(chan_bottom, chan_top, dtype=np.dtype('uint16')), again)
        add_dataset_to_group(neuron_group, 'channelmap', cm[chan_bottom:chan_top, 1:2], again)

    chunk = None
    if ('amplitudes' not in neuron_group) or ('voltage_sample' not in neuron_group) or again:
        chunk = extract_rawChunk(dp, snr_window, channels=np.arange(chan_bottom, chan_top), 
                                 scale=False, whiten=False, hpfilt=False, verbose=False)

    # quality metrics
    if 'amplitudes' not in neuron_group or again:
        add_dataset_to_group(neuron_group, 'amplitudes', np.load(dp/'amplitudes.npy').squeeze()[ids(dp, unit, periods=periods)], again)
        mad = np.median(np.abs(chunk) - np.median(chunk, axis=1)[:, None], axis=1) 
        std_estimate = (mad / 0.6745) # Convert to std
        add_dataset_to_group(neuron_group, 'channel_noise_std', std_estimate, again)
    
    # voltage snippets
    if 'voltage_sample' not in neuron_group or again:
        # Only store the voltage sample for the primary channel
        peak_chan = neuron_group['primary_channel']
        raw_snippet_halfrange = np.clip(raw_snippet_halfrange, 0, 10)
        c1, c2 = max(0,int(chunk.shape[0]/2-raw_snippet_halfrange)), min(chunk.shape[0]-1, int(chunk.shape[0]/2+raw_snippet_halfrange+1))
        raw_snippet = chunk[c1:c2,:]
        add_dataset_to_group(neuron_group, 'voltage_sample', raw_snippet) # still centered on peak channel, but half the size
        add_dataset_to_group(neuron_group, 'voltage_sample_start_index', int(snr_window[0] * samp_rate))
        add_dataset_to_group(neuron_group, 'scaling_factor', meta['bit_uV_conv_factor']) 

    # layer
    add_dataset_to_group(neuron_group, 'phyllum_layer', 0, again)
    add_dataset_to_group(neuron_group, 'human_layer', 0, again)

    # ground truth labels
    add_dataset_to_group(neuron_group, 'expert_label', 0, again)
    add_dataset_to_group(neuron_group, 'optotagged_label', 0, again)

    # predicted labels
    add_dataset_to_group(neuron_group, 'lisberger_label', 0, again)
    add_dataset_to_group(neuron_group, 'hausser_label', 0, again)
    add_dataset_to_group(neuron_group, 'medina_label', 0, again)

    # close file
    h5_file.close()

    return neuron_path

def add_data_to_unit_h5(h5_path, dataset, unit, data, field):
    """
    Add data to neuron already in h5 file.

    - h5_path: full path to h5 file
    - dataset: str, neuron dataset (yy-mm-dd_mouse_probex)
    - unit: unit index
    - data: data to add to unit
    - field: name of dataset to add data (id exists already, will overwrite)
    """
    with h5py.File(h5_path, "a") as h5_file:
        check_dataset_format(dataset)
        neuron_path = f"datasets/{dataset}/{unit}"
        assert neuron_path in h5_file, f"WARNING unit {neuron_path} does not seem to be present in the file. To add it, use add_unit_h5()."
        add_dataset_to_group(h5_file[neuron_path], field, data, True)

def add_dataset_to_group(group, dataset, data, again=False):
    if dataset in group:
        if again:
            del group[dataset]
        else:
            return
    group[dataset] = data
    return
    
def add_units_to_h5(h5_path, dp, **kwargs):
    """
    Add all units at the respective data path to an HDF5 file.

    This is a high-level function designed to add all units at the
    specified datapath to an HDF5 file. All additional key-value 
    arguments are passed to `add_unit_h5`

    Example:
      add_units_to_h5('my_lab_data.h5', '/path/to/kilosort_results', lab_id='pi_last_name')
    Will add all sorted units in the 'kilosort_results' directory 
    to the HDF5 file called 'my_lab_data.h5' (in the current directory).
    """
    for unit_id in get_units(dp):
        add_unit_h5(h5_path, dp, unit_id, **kwargs)

def get_stim_chan(ons, min_th=20):
    chan = -1
    for k, v in ons.items():
        if len(v) > min_th:
            chan = k
    assert chan != -1
    return chan

def visititems(group, func):
    with h5py._hl.base.phil:
        def proxy(name):
            """ Call the function with the text name, not bytes """
            name = group._d(name)
            return func(name, group[name])
        return group.id.links.visit(proxy)

def visitor_func(name, node):
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