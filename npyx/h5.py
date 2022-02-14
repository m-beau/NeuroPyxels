import h5py
from pathlib import Path
import re
import sys

import numpy as np

from npyx.io import get_npix_sync, chan_map, extract_rawChunk
from npyx.spk_t import ids, trn
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
        string = f"{name}: {type(node[()])}"
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


def label_unit_h5(h5_path, dataset, unit, label, field='optotagged_label'):
    """
    Add optotagged label to neuron.

    - h5_path: full path to h5 file
    - dataset: str, neuron dataset (yy-mm-dd_mouse_probex)
    - unit: neuron unit index
    """
    h5_file = h5py.File(h5_path, "r")

    check_dataset_format(dataset)
    neuron_path = f"datasets/{dataset}/{unit}"

    authorized_labels = ["PkC_ss", "PkC_cs", "MLI", "MFB", "GoC", "GrC"]
    assert label in authorized_labels
    h5_file[neuron_path][field] = label

    h5_file.close()


def add_unit_h5(h5_path, dp, unit,
                unit_abolute_id=None, opto_id=None,
                again=False, plot_debug=False):
    """
    Assumes that dataset id is last folder of directory (yy-mm-dd_iiXXX_probeX).
    """

    # hard-coded parameters
    samp_rate = 30000
    waveform_samples = 6  # ms
    waveform_samples = int(waveform_samples*30000/1000)
    lab_id = "hausser"

    # open file in append mode
    h5_file = h5py.File(h5_path, "a")

    # check whether neuron already exists in dataset
    dataset = Path(dp).name
    check_dataset_format(dataset)
    neuron_path = f"datasets/{dataset}/{unit}"
    if neuron_path in h5_file:
        print(f"Neuron already in h5 file: {neuron_path}")
        return neuron_path

    # figure out where we're at
    if unit_abolute_id is None:
        if f"{lab_id}_neuron_0" not in h5_file:
            unit_abolute_id = 0
        else:
            root_groups = list(h5_file.keys())
            neuron_ids = [int(x.split('_')[-1]) for x in root_groups if f"{lab_id}_neuron" in x]
            unit_abolute_id = np.sort(neuron_ids)[-1] + 1

    # create group for new neuron
    neuron_group = h5_file.create_group(f'{lab_id}_neuron_{unit_abolute_id}')
    h5_file[neuron_path] = neuron_group

    # metadata
    neuron_group['lab_id'] = lab_id
    neuron_group['dataset_id'] = dataset
    neuron_group['neuron_id'] = unit
    neuron_group['neuron_absolute_id'] = neuron_group.name
    neuron_group['sampling_rate'] = samp_rate

    # spike_times
    t = trn(dp, unit)
    neuron_group['spike_indices'] = t
    ons, ofs = get_npix_sync(dp, verbose=False)
    if opto_id is None:
        opto_id = get_stim_chan(ons)
    ons, ofs = ons[opto_id], ofs[opto_id]
    optostims = np.hstack([ons[:, None], ofs[:, None], (ofs-ons)[:, None]])
    neuron_group['optostims'] = optostims
    # Only consider spikes 10s before opto onset
    neuron_group['sane_spikes'] = (t < ons[0]-10*samp_rate)

    # waveforms
    dsm_tuple = wvf_dsmatch(dp, unit, t_waveforms=waveform_samples,
                            again=again, plot_debug=plot_debug)
    dsm_waveform, peak_chan = dsm_tuple[1], dsm_tuple[3]
    chan_bottom = max(0, peak_chan-11)
    chan_top = min(383, peak_chan+11)
    dsm_waveform_chunk = dsm_waveform[:, chan_bottom:chan_top]
    neuron_group['mean_waveform_preprocessed'] = dsm_waveform_chunk.T
    neuron_group['consensus_waveform'] = dsm_waveform_chunk*np.nan
    cm = chan_map(dp)
    neuron_group['channelmap'] = cm[chan_bottom:chan_top, :]

    # quality metrics
    neuron_group['amplitudes'] = np.load(dp/'amplitudes.npy').squeeze()[ids(dp, unit)]
    chunk = extract_rawChunk(dp, [0.1, 30.1], channels=np.arange(chan_bottom, chan_top))
    mad = np.median(np.abs(chunk) - np.median(chunk, axis=1)[:, None], axis=1) 
    std_estimate = (mad / 0.6745)
    neuron_group['channel_noise_std'] = std_estimate

    # layer
    neuron_group['phyllum_layer'] = 0
    neuron_group['human_layer'] = 0

    # ground truth labels
    neuron_group['human_label'] = 0
    neuron_group['optotagged_label'] = 0

    # predicted labels
    neuron_group['lisberger_label'] = 0
    neuron_group['hausser_label'] = 0
    neuron_group['medina_label'] = 0

    # close file
    h5_file.close()

    return neuron_path