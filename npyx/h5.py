import gc
import json
import re
import sys
import time
import warnings
from pathlib import Path
from io import StringIO

import warnings
warnings.filterwarnings("ignore")

import h5py
import numpy as np
from tqdm import tqdm

from npyx.gl import check_periods, get_units
from npyx.inout import (
    chan_map,
    detect_hardware_filter,
    extract_rawChunk,
    get_binary_file_path,
    get_npix_sync,
    preprocess_binary_file,
    read_metadata,
)
from npyx.spk_t import (
    duplicates_mask,
    find_stable_recording_period,
    ids,
    trn,
    trn_filtered,
)
from npyx.spk_wvf import across_channels_SNR, get_waveforms, wvf_dsmatch
from npyx.utils import assert_float, assert_int, docstring_decorator

## Ground truth labelling functions


def label_optotagged_unit_h5(h5_path, dataset, unit, label, source=None, prnt=False):
    """
    Add optotagged label to neuron.

    - h5_path: full path to h5 file
    - dataset: str, neuron dataset (yy-mm-dd_mouse_probex)
    - unit: neuron unit index
    - label: label to add
    """
    authorized_labels = ["PkC_ss", "PkC_cs", "MLI", "MFB", "GoC", "GrC",
                         "unlabelled", ""]
    assert (
        label in authorized_labels
    ), f"{label} must match either of the following: {authorized_labels}"
    add_data_to_unit_h5(h5_path, dataset, unit, label, "ground_truth_label")
    if source is not None:
        add_data_to_unit_h5(h5_path, dataset, unit, source, "ground_truth_source")
    if prnt:
        print(f"Labelled unit {unit} as {label}.")


def reset_optotagged_labels(h5_path):
    """Resets all optotagged labels to 0"""
    with h5py.File(h5_path, "a") as h5_f:
        for neuron in h5_f.keys():
            if "hausser_neuron" not in neuron:
                continue
            data_path = f"{neuron}/ground_truth_label"
            write_to_dataset(h5_f, data_path, 0, overwrite=True)


### Major C4 database generation function


def add_unit_h5(
    h5_path,
    dp,
    unit_id,
    lab_id,
    
    genetic_line             = "",
    dataset                  = None,
    sane_periods             = None,
    sane_spikes              = None,
    sane_before_opto         = False,
    
    again_npyx               = False,
    again_npyx_wvf           = False,
    
    overwrite_h5             = False,
    selective_overwrite      = None,
    
    include_fp_fn_mask       = True,
    include_raw_snippets     = True,
    
    raw_window               = None,
    raw_snippet_halfrange    = 3,
    mean_wvf_half_range      = 11,
    
    opto_sync_chan_id        = None,
    optostims                = None,
    optostims_from_sync      = False,
    optostims_threshold      = None,
    
    n_waveforms_for_matching = 5000,
    n_raw_waveforms          = 1000,
    plot_debug               = False,
    **kwargs,
):
    """
    Adds a spike-sorted unit to a new or existing HDF5 five file using the
    file format specified by the C4 collaboration.

    Data format details here ---->>> www.tinyurl.com/c4database <<<---

    Each unit can be accessed from 2 paths which point to the same data:
    - an absolute path, {unit_absolute_id}/
                        which allows for a flat hierarchy and makes it easier to compute features,
                        easier to work on data from other labs
    - a relative path, datasets/{dataset}/{unit}/
                       which allows to index units per dataset,
                       easier to work on your own data

    Example:
        add_unit_h5('my_lab_data.h5', '/path/to/kilosort_results', 1, lab_id='pi_last_name')
    Adds the unit with id 1 to the HDF5 file in the current directory called 'my_lab_data.h5'.
    Additional units can be added by calling the same function with identical
    arguments, but incrementing the unit_id field.

    Required Arguments:
    - h5_path: Path to the h5 file to create/append
    - dp: Path the Kilosort data directory
    - unit_id: The unit id/neuron unit index
    - lab_id: The lab/PI id to use to label the units


    Optional Arguments:

    - genetic_line: The genetic line of the animal used to record this data. None by default.
    - dataset: A unique ID for this dataset. By default this value is None, in which case
                the dataset id is assumed to the dirname of the data directory passed as the dp argument.

    - sane_periods: optional list of [start, stop] periods, in seconds, to define the 'sane spikes' boolean mask.
                    These will be the spikes used to compute the mean waveform and included in any futher analysis.
                    Will prevail over "sane_before_opto" if both are defined.
    - sane_spikes: optional bool array, custom definition of 'sane spikes'.
                   By default, will be all spikes within 'sane_periods' and will prevail over them if both are defined.
    - sane_before_opto: bool, whether to consider all spikes before the 1st optostim
                        to constitute the sane_spikes boolean mask.
                        Will only work if optostims are provided (or alternatively optostims_from_sync is True.)

    - again_npyx: Whether to recompute data from scratch rather than load NeuroPyxels cached results for storage in the HDF5 file (defaults to False)
    - again_npyx_wvf: Whether to recompute drift-shift matched waveform in particular (very computationally intensive, defaults to False)
    - overwrite_h5: Whether to fully delete any pre-existing unit data and overwrite it (defaults to False).
    - selective_overwrite: list of strings, which fields to specifically recompute even if again is False (much faster if only one field to overwrite).

    - include_fp_fn_mask: bool, whether to compute the false positive and false negative rates for recording periods
                          and subsequently compute the 'fn_fp_filtered_spikes', which is True for spikes in periods
                          passing the 5% fp and 5% fn quality requirement and False for other spikes.
    - include_raw_snippets: bool, whether to include 30sec raw data snippets
                            alongside a 3d matrix of 1000 randomly sampled waveforms.

    - raw_window: A two item list containing the start and stop times (in SECONDS) of the snippet of raw data used for
                  1) computing single channel noise RMS (-> SNR)
                  2) extracting representative examples of raw data (raw, and whitened).
                  If None, will use a period of 30 seconds within sane_periods with the most spikes.
    - raw_snippet_halfrange: int, range of channels around peak channel to consider for the snippet
                             of raw data (max 10, default 3 resulting in 7 channels total)
    - mean_wvf_half_range: int, range of channels around peak channel to consider for the mean waveform. Default 11 (23 channels total).

    - opto_sync_chan_id: The index of the npix binary channel where the optostim triggers are written. Defaults to None.
    - optostims: an optional 2D array (n_stims, 2) containing the optostimuli times in seconds
                 (1st column: onsets, 2nd column: offsets).
                 By default None, will be read from sync channel (at opto_sync_chan_id) if optostims_from_sync is True.
    - optostims_from_sync: bool, whether to pick up optostims from sync channel if None are provided.
    - optostims_threshold: float, time before which optostims will be ignored (same units as optostims, seconds if optostims_from_sync=True).
                           (handles sync signals without light at beginning of recording)

    - n_waveforms_for_matching: int, number of waveforms to subsample for drift-shift matching
    - n_raw_waveforms: int, number of raw waveforms to store in the HDF5 file at "raw_waveforms"
    - plot_debug: bool, whether to save the debugging plots showing the the false positive/negative rate estimations

    Additional key-value parameters:
    - *any_key* = *any_value*
        All additional key-value parameters passed to this function are stored in the HDF5 file.
        Therefore, custom keys can be stored in the HDF5 file should additional
        information be required for an individual neuron. E.g., calling
            add_unit_h5('my_lab_data.h5', '/path_to_kilosort_results', 1, my_note="Cool info")
        will result in a key of 'my_note' and a value of "Cool info" being stored in the HDF5 file
        for this unit.
    """

    if selective_overwrite is not None:
        assert isinstance(
            selective_overwrite, list
        ), "Selective_overwrite must be None or a list of strings."
    else:
        selective_overwrite = []

    dp        = Path(dp)
    meta      = read_metadata(dp)
    samp_rate = meta["highpass"]["sampling_rate"]
    # hard-coded parameters
    waveform_samples = 6  # ms
    waveform_samples = int(waveform_samples * samp_rate / 1000)

    # open file in append mode
    h5_path = Path(h5_path)
    assert_h5_file(h5_path)
    with h5py.File(h5_path, "a") as h5_file:
        
        # --------------- h5 preformatting ---------------#

        # format dataset name
        if dataset is None:
            dataset = dp.name
        check_dataset_format(dataset)

        # Define or fetch unit h5 paths
        relative_unit_path, absolute_unit_path = get_unit_paths_h5(
            h5_file, dataset, unit_id, lab_id
        )
        if relative_unit_path in h5_file:
            if overwrite_h5:
                del h5_file[relative_unit_path]
                del h5_file[absolute_unit_path]
            else:
                print(f"Neuron found in h5 file: '{relative_unit_path}' ({absolute_unit_path})")
                neuron_group = h5_file[absolute_unit_path]
        # redefine unit paths in case the unit got deleted
        # relative_unit_path, absolute_unit_path = get_unit_paths_h5(h5_file, dataset, unit, lab_id)

        # create group for new neuron if necessary
        if relative_unit_path not in h5_file:
            neuron_group = h5_file.create_group(absolute_unit_path)
            h5_file[relative_unit_path] = neuron_group
            print(f"Adding neuron at '{relative_unit_path}' ({absolute_unit_path})...")
        pbar = tqdm(
            total    = 7,
            desc     = f"Adding unit at '{relative_unit_path}' ({absolute_unit_path})...",
            position = 0,
            leave    = True,
            disable  = False,
        )


        # --------------- metadata ---------------#

        pbar.set_description(f"Adding metadata for unit '{relative_unit_path}'...")
        pbar.update(1)
        write_to_group(neuron_group, "lab_id",             lab_id,            overwrite_h5)
        write_to_group(neuron_group, "line",               genetic_line,      overwrite_h5)
        write_to_group(neuron_group, "dataset_id",         dataset,           overwrite_h5)
        write_to_group(neuron_group, "neuron_relative_id", unit_id,           overwrite_h5)
        write_to_group(neuron_group, "neuron_id",          neuron_group.name, overwrite_h5)
        write_to_group(neuron_group, "sampling_rate",      samp_rate,         overwrite_h5)

        # add any additional keys passed to this function
        for key, value in kwargs.items():
            write_to_group(neuron_group, key, value, overwrite_h5)


        # --------------- spike time-related ---------------#

        pbar.set_description(f"Adding spike times for unit '{relative_unit_path}'...")
        pbar.update(1)

        # sane periods
        if sane_periods is None:
            sane_periods = "all"
        sane_periods = check_periods(sane_periods)
        write_to_group(neuron_group, "sane_periods", sane_periods,
                       assert_recompute("sane_periods", neuron_group, overwrite_h5, selective_overwrite)
                       )

        # spike_times
        key = "spike_indices"
        if assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite):
            t = trn(dp, unit_id, again=again_npyx).astype(np.uint32)
            write_to_group(neuron_group, key, t)

        # optostims
        pbar.set_description(f"Adding optostims for unit '{relative_unit_path}'...")
        pbar.update(1)
        keys = ["optostims", "sane_spikes"]
        if assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite):
            if optostims is None and optostims_from_sync:
                ons, offs = get_npix_sync(dp)
                if opto_sync_chan_id is None:
                    opto_sync_chan_id = get_stim_chan(ons)
                ons, offs = ons[opto_sync_chan_id], offs[opto_sync_chan_id]
                if ons[0] > offs[0]:
                    ons, offs = offs, ons
                if len(offs) == len(ons) - 1:
                    offs = np.append(offs, meta["recording_length_seconds"])
                optostims = np.hstack([ons[:, None], offs[:, None]])
            if optostims is not None and optostims_threshold is not None:
                opto_m = optostims[:, 0] > optostims_threshold
                optostims = optostims[opto_m, :]

            if optostims is not None:
                write_to_group(neuron_group, "optostims", optostims)

        # sane spikes
        key = "sane_spikes"
        if assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite):
            t = neuron_group["spike_indices"][()]
            if sane_spikes is None:
                if sane_before_opto:
                    if "optostims" in neuron_group:
                        optostims = neuron_group["optostims"][()]
                    # Only consider spikes 10s before first opto onset
                    end_spont_period = (optostims[0, 0] - 10) * samp_rate
                    sane_spikes = t < end_spont_period
                elif sane_periods is not None and sane_periods != "all":
                    sane_spikes = (t * 0).astype(bool)
                    for sane_period in sane_periods:
                        sane_spikes = sane_spikes | (t >= sane_period[0] * samp_rate) & (t <= sane_period[1] * samp_rate)
                else:
                    # if sane_periods is None or 'all', keep them all
                    sane_spikes = (t * 0 + 1).astype(bool)
            write_to_group(neuron_group, key, sane_spikes)


        # --------------- quality metrics 1/2 ---------------#
        # required to define the snippet raw_window and the single waveforms to pick

        pbar.set_description(f"Computing false positive/negative rate estimates for unit '{relative_unit_path}'...")
        pbar.update(1)
        # false positive and negative estimation
        key = "fn_fp_filtered_spikes"
        if assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite) and include_fp_fn_mask:
            # get good spikes mask for all spikes
            # because trn_filtered can only work on a contiguous chunk
            periods_m_range = [0, meta["recording_length_seconds"] / 60]
            fp_fn_good_spikes = trn_filtered(dp, unit_id,
                                             plot_debug=plot_debug, again=again_npyx, 
                                             period_m=periods_m_range)[1]
            write_to_group(neuron_group, key, fp_fn_good_spikes)


        # --------------- waveform-related ---------------#

        pbar.set_description(f"Extracting waveform-related data for unit '{relative_unit_path}'...")
        pbar.update(1)

        # define fields which are needed to compute anything related
        # to the extraction of raw data
        # or the selection of periods from which raw data should be extracted
        raw_data_keys = [
            "mean_waveform_preprocessed",
            "peakchan_SNR",
            "raw_voltage_snippet",
            "raw_waveforms",
            "fn_fp_filtered_spikes",
            "sane_spikes"]

        # extract waveforms (and also load peak channel found by dsmatching)
        keys = [
            "channel_ids",
            "channelmap",
            "amplitudes"] + raw_data_keys
        if assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite):
            # must recompute chan_bottom and chan_top - suboptimal, can be rewritten
            dsm_tuple = wvf_dsmatch(dp, unit_id,
                t_waveforms=waveform_samples, periods=sane_periods,
                again=again_npyx_wvf, plot_debug=plot_debug,
                n_waves_used_for_matching=n_waveforms_for_matching,
                med_sub=True, nRangeMedSub=None)
            dsm_waveform, peak_chan = dsm_tuple[1], dsm_tuple[3]
            peak_chan          = int(peak_chan)
            chan_bottom        = np.max([0, peak_chan - mean_wvf_half_range])
            chan_top           = np.min([383, peak_chan + mean_wvf_half_range])
            chan_range         = np.arange(chan_bottom, chan_top)
            peak_chan_rel      = np.nonzero(peak_chan == chan_range)[0][0]
            dsm_waveform_chunk = dsm_waveform[:, chan_bottom:chan_top]
            cm                 = chan_map(dp)

            write_to_group(neuron_group, "mean_waveform_preprocessed", dsm_waveform_chunk.T)
            write_to_group(neuron_group, "channel_ids", np.arange(chan_bottom, chan_top, dtype=np.dtype("uint16")))
            write_to_group(neuron_group, "channelmap", cm[chan_bottom:chan_top, 1:3])

        # scaling factor
        write_to_group(neuron_group, "scaling_factor", meta["bit_uV_conv_factor"], True)

        # Extraction of voltage snippet
        keys = ["channel_noise_std", "peakchan_SNR", "raw_voltage_snippet", "fn_fp_filtered_spikes", "sane_spikes"]
        if assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite):
            # find optimal window for raw snippet
            if raw_window is None:
                good_spikes_m = neuron_group["fn_fp_filtered_spikes"][()] & neuron_group["sane_spikes"][()]
                good_t = neuron_group["spike_indices"][()][good_spikes_m]
                # handle cases where there is no good fp/fn section
                if not np.any(good_t):
                    good_t = neuron_group["spike_indices"][()]
                raw_window = find_stable_recording_period(
                    good_t,
                    samp_rate,
                    meta["recording_length_seconds"] * samp_rate,
                    target_period = 30,
                    b             = 1000,
                    sd            = 10000,
                    minimum_fr    = 0.4)
                raw_window = np.array(raw_window) / samp_rate  # converge to seconds
                
            chunk = extract_rawChunk(dp, raw_window,
                channels          = np.arange(chan_bottom, chan_top),
                scale             = False,
                med_sub           = False,
                whiten            = False,
                center_chans_on_0 = False,
                hpfilt            = False) # already int16

        # inclusion of voltage snippet
        keys = ["raw_voltage_snippet", "voltage_snippet_start_index", "fn_fp_filtered_spikes", "sane_spikes"]
        if assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite) and include_raw_snippets:
            raw_snippet_halfrange = np.clip(raw_snippet_halfrange, 0, 10)
            c1                    = max(0, peak_chan_rel - raw_snippet_halfrange)
            c2                    = min(chunk.shape[0] - 1, peak_chan_rel + raw_snippet_halfrange + 1)
            raw_snippet           = chunk[c1:c2, :]
            write_to_group(neuron_group, "raw_voltage_snippet", raw_snippet)  # still centered on peak channel, but half the size
            write_to_group(neuron_group, "voltage_snippet_start_index", int(raw_window[0] * samp_rate))

        # extraction of raw waveforms 3d matrix
        keys = ["raw_waveforms", "fn_fp_filtered_spikes", "sane_spikes"]
        if assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite) and include_raw_snippets:
            # relect spike ids
            spike_ids  = ids(dp, unit_id, enforced_rp=0)
            spike_mask = neuron_group["fn_fp_filtered_spikes"][()] & neuron_group["sane_spikes"][()]
            if np.any(spike_mask):
                spike_ids = spike_ids[spike_mask]

            assert n_raw_waveforms > 0, "n_raw_waveforms must be > 0"
            if len(spike_ids) > n_raw_waveforms:
                random_ids = np.random.randint(0, spike_ids.shape[0]-1, n_raw_waveforms)
                spike_ids  = spike_ids[random_ids]
                
            # load waveforms
            raw_waveforms = get_waveforms(dp, unit_id, t_waveforms=180,
                                            spike_ids=spike_ids, med_sub_in_time=False,  ignore_ks_chanfilt=True)
            raw_waveforms = raw_waveforms[:, :, chan_bottom:chan_top].transpose(0,2,1)

            # recast to int16
            raw_waveforms = raw_waveforms / meta["bit_uV_conv_factor"]
            raw_waveforms = raw_waveforms.astype(np.int16)

            write_to_group(neuron_group, "raw_waveforms", raw_waveforms)


        # --------------- quality metrics 2/2 ---------------#

        pbar.set_description(f"Computing SNR for unit '{relative_unit_path}'...")
        pbar.update(1)

        # SNR
        keys = ["channel_noise_std", "peakchan_SNR", "raw_waveforms", "fn_fp_filtered_spikes", "sane_spikes"]
        if assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite):
            # both chunk and raw waveforms are in bits (int16)
            mad           = np.median(np.abs(chunk - np.median(chunk, axis=1)[:, None]), axis=1)
            std_estimate  = mad / 0.6745  # Convert to std
            # peakchan_S  = np.ptp(dsm_waveform[:, peak_chan]) #overestimate
            w             = neuron_group["raw_waveforms"][()].mean(0)
            emp_peak_chan = np.argmax(np.ptp(w, axis=1))
            peakchan_S    = np.ptp(w, axis=1)[emp_peak_chan]
            peakchan_N    = std_estimate[emp_peak_chan]
            peakchan_SNR  = peakchan_S / peakchan_N

            write_to_group(neuron_group, "channel_noise_std", std_estimate)
            write_to_group(neuron_group, "peakchan_SNR", peakchan_SNR)

        # across channels SNR
        key = "acrosschan_SNR"
        if assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite):
            acrosschan_SNR = across_channels_SNR(dp, unit_id)
            write_to_group(neuron_group, key, acrosschan_SNR)

        # amplitudes
        key = "amplitudes"
        if assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite):
            amps = np.load(dp / "amplitudes.npy").squeeze()[
                ids(dp, unit_id, enforced_rp=0)
            ]
            write_to_group(neuron_group, key, amps)


        # --------------- unit labels ---------------#

        pbar.set_description(f"Adding labels for unit '{relative_unit_path}'...")
        pbar.update(1)
        
        # layer and ground truth labels
        for key in ["phyllum_layer", "human_layer",
                    "expert_label", "ground_truth_label", "ground_truth_source", "mli_cluster"]:
            value = kwargs[key] if key in kwargs else ""
            if assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite):
                print(key, value)
                write_to_group(neuron_group, key, value)

        pbar.set_description(f"Done with '{relative_unit_path}'.")
        pbar.refresh()
        time.sleep(0.01)
        pbar.close()

    return relative_unit_path


### json wrapper functions


def add_json_datasets_to_h5(json_path, h5_path, lab_id,
                            preprocess_if_raw             = False,
                            delete_original_data          = False,
                            data_deletion_double_check    = False,
                            include_all_good              = True,
                            selective_overwrite           = None,
                            overwrite_h5                  = False,
                            **kwargs): 
    """
    Wrapper function to loop over all datasets in a json file
    and add them to an HDF5 file according to the C4 data format specification.

    Data format details here ---->>> www.tinyurl.com/c4database <<<---

    Arguments:
        - json_path: str, path to the json file containing the datasets info
            according to the following structure ("sane_periods" is optional):
            "0": {
                "ct": "celltype", # must be either of the following: ["PkC_ss", "PkC_cs", "MLI", "MFB", "GoC", "GrC", "unlabelled", ""]
                "line": "mouseline", # can be anything
                "dp": "/path/to/dataset", # path to neuropixels dataset
                "units": [u1, u2, u3, u4], # optotagged/unlabelled units to add to h5
                "ss": [u5, u6], # simple spikes to add to h5
                "cs": [u7, u8], # complex spikes to add to h5
                "sane_periods":{u1:[[t1,t2], [t3,t4]], u2:[], u3:[],
                                u4:[], u5:[], u6:[], u7:[], u8:[]} # windows of time to use to compute features etc for any particular neuron, in seconds.
                "global_sane_periods": [[t1,t2], [t3,t4]], # windows of time to use to compute features etc for all neurons, in seconds
                "phyllum_layers": "layer", # can be ["ML", "PCL", "GCL", "unknown", "not_cortex", ""]
                }
        - h5_path: path/to/database_file.h5
        - lab_id: str, lab id. see format at www.tinyurl.com/c4database
        - preprocess_if_raw: bool, whether to high-pass filter the raw data
                             if it is found to be so (for Hull lab)
        - delete_original_data: bool, whether to delete the original raw file if it is preprocessed
                                according to preprocess_if_raw
        - data_deletion_double_check: bool, double check which must also be set to True
                                      to allow deletion of the original data.
        - include_all_good: bool, whether to include all good units in the dataset.
                            Useful for unlabelled data, to train variational autoencoders.
        - selective_overwrite: list of fields (e.g. spike_indices...) to recompute
                                and overwrite if they already exist in the h5 file.
        - overwrite_h5: bool, whether to recompute everything from scratch and overwrite the data
                         (only for the reworked units, the other units will be left untouched in the file)
    """

    DSs = load_json_datasets(json_path, include_missing_datasets=False)

    for ds_name_ct, ds in DSs.items():
        dp = Path(ds["dp"])

        if preprocess_if_raw:
            assert not detect_hardware_filter(dp),\
                ("ERROR preprocess_if_raw set to True but hardware filter on binary file detected "
                  "(last column of .ap.meta ~imro field is set to 1)! Check your .ap.meta file or set preprocess_if_raw to False.")
            "\033[34;1mUnfiltered file detected - filtering with 1st order butterworth highpass at 300Hz...\033[0m"
            preprocess_binary_file(
                dp,
                delete_original_data=delete_original_data,
                data_deletion_double_check=data_deletion_double_check,
                median_subtract=False,
                filter_forward=True,
                filter_backward=False,
                order=1,
            )

        # extract metadata
        ds_name, optolabel = ds_name_ct.split("&")
        assert optolabel   == ds["ct"]
        if optolabel       == "PkC":
            optolabel = "PkC_ss"
        genetic_line        = ds["line"]

        # extract units
        units               = ds["units"]
        ss                  = ds["ss"]
        cs                  = ds["cs"]
        good_units          = list(get_units(dp, "good", again=True))

        if include_all_good:
            units_for_h5 = np.unique(units + ss + cs + good_units)
        else:
            units_for_h5 = units + ss + cs

        # extract periods
        sane_periods_dic    = ds["sane_periods"] if "sane_periods" in ds else {}
        sane_periods_dic    = {int(k):v for k,v in sane_periods_dic.items()}
        if len(sane_periods_dic)>0:
            assert np.all(np.isin(list(sane_periods_dic.keys()), units_for_h5)),\
                f"sane_periods is {sane_periods_dic} but units must be in {units_for_h5}!"
        global_sane_periods = ds["global_sane_periods"] if "global_sane_periods" in ds else []

        # extract layers
        allowed_layers = ["ML", "PCL", "GCL", "unknown", "not_cortex", ""]
        if "phyllum_layers" in ds:
            phyllum_layers  = ds["phyllum_layers"]
            phyllum_layers  = {int(k):v for k,v in phyllum_layers.items()}
            assert np.all(np.isin(list(phyllum_layers.values()), allowed_layers)),\
                f"phyllum_layers is {phyllum_layers} but layers must be in {allowed_layers}!"
            assert np.all(np.isin(list(phyllum_layers.keys()), units_for_h5)),\
                f"phyllum_layers is {phyllum_layers} but units must be in {units_for_h5}!"

        for u in units_for_h5:
            sane_periods = sane_periods_dic[u] if u in sane_periods_dic else None
            if sane_periods is None and np.any(global_sane_periods):
                sane_periods = global_sane_periods

            if "phyllum_layers" in ds:
                phyllum_layer = phyllum_layers[u] if u in phyllum_layers else ""
                kwargs["phyllum_layer"] = phyllum_layer

            add_unit_h5(
                h5_path,
                dp,
                u,
                lab_id,
                genetic_line=genetic_line,
                dataset=None,  # end of dp by default
                sane_periods=sane_periods,
                selective_overwrite=selective_overwrite,
                overwrite_h5=overwrite_h5,
                **kwargs,
            )
            if u in units:
                label = optolabel
                source = "optogagged"
            elif u in ss:
                label = "PkC_ss"
                source = "CSxSS"
            elif u in cs:
                label = "PkC_cs"
                source = "CSxSS"
            else:
                continue
            label_optotagged_unit_h5(h5_path, ds_name, u, label, source)
            gc.collect()


def add_json_datasets_to_h5_hausser(
    json_path,
    h5_path,
    include_all_good=True,
    selective_overwrite=None,
    overwrite_h5=False,
    **kwargs,
):
    # parameters ensuring that only the
    # spontaneous period before opto is used
    sane_before_opto = True
    optostims_from_sync = True
    optostims_threshold = 20 * 60

    add_json_datasets_to_h5(
        json_path,
        h5_path,
        "hausser",
        include_all_good    = include_all_good,
        optostims_from_sync = optostims_from_sync,
        optostims_threshold = optostims_threshold,
        sane_before_opto    = sane_before_opto,
        selective_overwrite = selective_overwrite,
        overwrite_h5        = overwrite_h5,
        **kwargs,
    )


def load_json_datasets(json_path, include_missing_datasets=False):
    with open(json_path) as f:
        json_f = json.load(f)

    print(f"\nLoading data from file {json_path}...\n")

    DSs = {}
    for ds in json_f.values():
        for key in ["dp", "ct", "line", "units", "ss", "cs"]:
            assert key in ds, f"{key} not in json file for dataset #{ds}!"

        dp = Path(ds["dp"])

        if not dp.exists():
            print(f"Dataset {dp} NOT found on system!\n")
            if include_missing_datasets:
                DSs[dp.name] = ds
            continue
        DSs[dp.name + "&" + ds["ct"]] = ds

        units = list(ds["units"])
        ss = list(ds["ss"])
        cs = list(ds["cs"])
        print(
            f"Dataset {dp} found with opto responsive units {units}, simple spikes {ss}, complex spikes {cs}.\n"
        )
        all_units = units + ss + cs
        units_m = np.isin(all_units, get_units(dp))
        assert all(units_m), f"Units {np.array(all_units)[~units_m]} not found in {dp}!"

    return DSs


### h5 unit management wrapper functions


@docstring_decorator(add_unit_h5.__doc__)
def add_units_to_h5(h5_path, dp, units=None, **kwargs):
    """
    Add all or specified units at the respective data path to an HDF5 file.

    This is a high-level function designed to add many units at the
    specified datapath to an HDF5 file. All additional key-value
    arguments are passed to `add_unit_h5`

    Example:
      add_units_to_h5('my_lab_data.h5', '/path/to/dataset_id', lab_id='pi_last_name')
    Will add all sorted units in the 'kilosort_results' directory
    to the HDF5 file called 'my_lab_data.h5' (in the current directory).

    Other arguments from add_unit_h5:
    {0}
    """

    if units is None:
        units = get_units(dp)

    for u in units:
        add_unit_h5(h5_path, dp, u, **kwargs)


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
    unit_path = relative_unit_path_h5(dataset, unit)
    with h5py.File(h5_path, "a") as h5_file:
        assert (
            unit_path in h5_file
        ), f"WARNING unit {unit_path} does not seem to be present in the file. To add it, use add_unit_h5()."
        write_to_group(h5_file[unit_path], field, data, True)


def get_unit_paths_h5(h5_file, dataset, unit, lab_id="hausser", unit_absolute_id=None):
    relative_unit_path = relative_unit_path_h5(dataset, unit)
    if relative_unit_path in h5_file:
        absolute_unit_path = h5_file[f"{relative_unit_path}/neuron_id"][()].decode()
    else:
        if unit_absolute_id is None:
            if f"{lab_id}_neuron_0" not in h5_file:
                unit_absolute_id = 0
            else:
                root_groups = list(h5_file.keys())
                neuron_ids = [
                    int(x.split("_")[-1])
                    for x in root_groups
                    if f"{lab_id}_neuron" in x
                ]
                unit_absolute_id = np.sort(neuron_ids)[-1] + 1
        absolute_unit_path = f"{lab_id}_neuron_{unit_absolute_id}"

    return relative_unit_path, absolute_unit_path


def remove_unit_h5(h5_path, dp, unit, lab_id="hausser", dataset=None):
    if dataset is None:
        dataset = Path(dp).name
    with h5py.File(h5_path, "a") as h5_file:
        relative_unit_path, absolute_unit_path = get_unit_paths_h5(
            h5_file, dataset, unit, lab_id
        )
        del h5_file[relative_unit_path]
        del h5_file[absolute_unit_path]
        dataset_path = str(Path(relative_unit_path).parent)
        if len(h5_file[dataset_path].keys()) == 0:
            del h5_file[dataset_path]


### h5 vizualisation functions

def get_absolute_neuron_ids(h5_path, again=False):
    h5_contents = print_h5_contents(h5_path, again=again)
    return [p.split('/')[0] for p in h5_contents if ('_neuron_' in p.split('/')[0] and len(p.split('/'))==1)]

def get_neuron_id_dict(h5_path):
    
    absolute_neuron_ids = get_absolute_neuron_ids(h5_path)

    neuron_id_dict = {}
    with h5py.File(h5_path, "r") as hdf:
        for neuron_id in absolute_neuron_ids:

            neuron_relative_id = int(hdf[neuron_id]['neuron_relative_id'][()])
            dataset = hdf[neuron_id]['dataset_id'][()].decode()

            relative_id_path = f"datasets/{dataset}/{neuron_relative_id}"
            neuron_id_dict[neuron_id] = relative_id_path

            if dataset not in neuron_id_dict: neuron_id_dict[dataset] = {}
            neuron_id_dict[dataset][neuron_relative_id] = neuron_id

    return neuron_id_dict

def print_h5_contents(h5_path, display = False, txt_output=True, again=False):
    """
    Arguments:
        - h5_path: str, path to .h5 file
        - display: bool, if True prints contents to console
        - txt_output: bool, if True prints contents to file
                            (same name as h5 name_content.txt).
                            Recommended to leave 'True' to reload the h5 contents faster later.
        - again: bool, if False reloads data from txt file when found
                        rather than recomputing from h5 file.

    Returns:
        - list of paths in h5 file
    """
    h5_path = Path(h5_path)
    txt_output_path = h5_path.parent / f"{h5_path.name[:-3]}_content.txt"
    if txt_output_path.exists() and not again:
        with open(txt_output_path, "r") as f:
            print_string = f.read()
    
    else:
        with h5py.File(h5_path, "a") as hdf:
            # save to variable
            print_output    = StringIO()
            original_stdout = sys.stdout
            sys.stdout      = print_output
            visititems(hdf, visitor_func)
            print_string   = print_output.getvalue()
            sys.stdout      = original_stdout

    # save to txt file
    if txt_output and (again or not txt_output_path.exists()):
        with open(txt_output_path, "w") as f:
            f.write(print_string)

    # print to console
    if display:
        print(print_string)

    return print_string.split('\n')
            


def visititems(group, func):
    with h5py._hl.base.phil:

        def proxy(name):
            """Call the function with the text name, not bytes"""
            name = group._d(name)
            return func(name, group[name])

        return group.id.links.visit(proxy)


def visitor_func(name, node):
    """
    prints name followed by a meangingful description of an hdf5 node.
    Node is either an h5 dataset (array, string...) or a group.
    """
    if isinstance(node, h5py.Dataset):
        n = node[()]
        if isinstance(n, bytes):
            s = n.decode()
        elif isinstance(n, np.ndarray):
            s = f"ndarray {n.shape}"
        elif assert_int(n) or assert_float(n):
            s = n
        else:
            s = type(n)
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
    return h5_path.name[-3:] == ".h5"


### h5 writing functions
def write_to_h5(h5_path, data_path, data, overwrite=False, must_exist=False):
    """
    Writes data at data_path to .h5 file at h5_path
    (creates non existing groups and datasets).
    """
    assert_h5_file(h5_path)
    with h5py.File(h5_path, "a") as h5_file:
        write_to_group(h5_file, data_path, data, overwrite, must_exist)


def write_to_group(group, dataset, data, overwrite=True, must_exist=False):
    """Write data to hdf5 group
    i.e. create a dataset +/- groups on the path to dataset
    and write data to this dataset.
    Arguments:
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
    write_to_group(group, dataset, data, overwrite, True)


### h5 reading functions
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
            data = data.decode()  # for strings
    return data


### C4 utilities
def get_stim_chan(ons, min_th=20):
    chan = -1
    for k, v in ons.items():
        if len(v) > min_th:
            chan = k
    assert chan != -1
    return chan


def assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite):
    return (key not in neuron_group) or overwrite_h5 or (key in selective_overwrite)


def assert_recompute_any(keys, neuron_group, overwrite_h5, selective_overwrite):
    return np.any(
        [
            assert_recompute(key, neuron_group, overwrite_h5, selective_overwrite)
            for key in keys
        ]
    )


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


def relative_unit_path_h5(dataset, unit):
    return f"datasets/{dataset}/{unit}"
