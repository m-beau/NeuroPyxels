import h5py
from pathlib import Path
import re
import sys

import numpy as np

from npyx.utils import assert_int, assert_float
from npyx.inout import get_npix_sync, chan_map, extract_rawChunk
from npyx.spk_t import ids, trn, trn_filtered
from npyx.spk_wvf import wvf_dsmatch


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

def check_dataset_format(dataset):
    """
    Checks whether dataset name is formatted properly
    i.e. aa-mm-dd_iiXXX_probeX (mouse can only be names )
    """
    warning = "WARNING last folder of path must match format: aa-mm-dd_ii[0-1000]_probe[0-9] (ii = 2 [a-z] initials)"
    pattern = "[0-9]{2}-[0-9]{2}-[0-9]{2}_[a-z]{2}[0-9]{3}_probe[0-9]"
    assert bool(re.match(pattern, dataset, re.IGNORECASE)), warning


def label_unit_h5(h5_path, dataset, unit, label):
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
        add_dataset_to_group(h5_file[neuron_path], field, data)

def add_dataset_to_group(group, dataset, data, again=0):
    if dataset in group:
        if again:
            del group[dataset]
        else:
            return
    group[dataset] = data
    return
    

def add_unit_h5(h5_path, dp, unit,
                unit_abolute_id=None, sync_chan_id=None,
                again=False, again_wvf=False, plot_debug=False, verbose=False):
    """
    Assumes that dataset id is last folder of directory (yy-mm-dd_iiXXX_probeX).
    """

    # hard-coded parameters
    samp_rate = 30000
    waveform_samples = 6  # ms
    waveform_samples = int(waveform_samples*samp_rate/1000)
    lab_id = "hausser"
    snr_window = [0.1, 30.1]

    # open file in append mode
    h5_file = h5py.File(h5_path, "a")

    # check whether neuron already exists in dataset
    dp=Path(dp)
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
    add_dataset_to_group(neuron_group, 'labneuron_id_id', unit, again)
    add_dataset_to_group(neuron_group, 'neuron_absolute_id', neuron_group.name, again)
    add_dataset_to_group(neuron_group, 'sampling_rate', samp_rate, again)

    # spike_times
    if 'spike_indices' not in neuron_group or again:
        t = trn(dp, unit)
        add_dataset_to_group(neuron_group, 'spike_indices', t, again)
        ons, ofs = get_npix_sync(dp, verbose=False)
        if sync_chan_id is None:
            sync_chan_id = get_stim_chan(ons)
        ons, ofs = ons[sync_chan_id], ofs[sync_chan_id]
        optostims = np.hstack([ons[:, None], ofs[:, None], (ofs-ons)[:, None]])
        add_dataset_to_group(neuron_group, 'optostims', optostims, again)
        # Only consider spikes 10s before opto onset
        sane_spikes = (t < ons[0]-10*samp_rate)
        add_dataset_to_group(neuron_group, 'sane_spikes', sane_spikes, again)
        
    if 'fn_fp_filtered_spikes' not in neuron_group or again: 
        fp_fn_good_spikes = trn_filtered(dp, unit, plot_debug=plot_debug)[1]
        add_dataset_to_group(neuron_group, 'fn_fp_filtered_spikes', fp_fn_good_spikes, again)

    # waveforms
    if 'mean_waveform_preprocessed' not in neuron_group or again: 
        dsm_tuple = wvf_dsmatch(dp, unit, t_waveforms=waveform_samples,
                                again=again_wvf, plot_debug=plot_debug, verbose=verbose)
        dsm_waveform, peak_chan = dsm_tuple[1], dsm_tuple[3]
        chan_bottom = max(0, peak_chan-11)
        chan_top = min(383, peak_chan+11)
        dsm_waveform_chunk = dsm_waveform[:, chan_bottom:chan_top]
        add_dataset_to_group(neuron_group, 'mean_waveform_preprocessed', dsm_waveform_chunk.T, again)
        add_dataset_to_group(neuron_group, 'consensus_waveform', dsm_waveform_chunk.T*np.nan, again)
        cm = chan_map(dp)
        add_dataset_to_group(neuron_group, 'channelmap', cm[chan_bottom:chan_top, :], again)

    # quality metrics
    if 'amplitudes' not in neuron_group or again:
        add_dataset_to_group(neuron_group, 'amplitudes', np.load(dp/'amplitudes.npy').squeeze()[ids(dp, unit)], again)
        chunk = extract_rawChunk(dp, snr_window, channels=np.arange(chan_bottom, chan_top))
        mad = np.median(np.abs(chunk) - np.median(chunk, axis=1)[:, None], axis=1) 
        std_estimate = (mad / 0.6745)
        add_dataset_to_group(neuron_group, 'channel_noise_std', std_estimate, again)

    # layer
    add_dataset_to_group(neuron_group, 'phyllum_layer', 0, again)
    add_dataset_to_group(neuron_group, 'human_layer', 0, again)

    # ground truth labels
    add_dataset_to_group(neuron_group, 'human_label', 0, again)
    add_dataset_to_group(neuron_group, 'optotagged_label', 0, again)

    # predicted labels
    add_dataset_to_group(neuron_group, 'lisberger_label', 0, again)
    add_dataset_to_group(neuron_group, 'hausser_label', 0, again)
    add_dataset_to_group(neuron_group, 'medina_label', 0, again)

    # close file
    h5_file.close()

    return neuron_path