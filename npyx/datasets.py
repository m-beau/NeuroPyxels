# -*- coding: utf-8 -*-
"""
2022-12
Authors: @fededagos

This module contains the functions to load the data from the hdf5 files used
in the C4 collaboration. It also contains the functions to preprocess the data.
"""
import copy
import pickle
from typing import Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample
from tqdm.auto import tqdm

import npyx

CENTRAL_RANGE = 60

N_CHANNELS = 10

LABELLING = {
    "PkC_cs": 5,
    "PkC_ss": 4,
    "MFB": 3,
    "MLI": 2,
    "GoC": 1,
    "GrC": 0,
    "unlabelled": -1,
}
CORRESPONDENCE = {value: key for key, value in LABELLING.items()}

LABELLING_NO_GRC = {
    "PkC_cs": 4,
    "PkC_ss": 3,
    "MFB": 2,
    "MLI": 1,
    "GoC": 0,
    "unlabelled": -1,
}

CORRESPONDENCE_NO_GRC = {value: key for key, value in LABELLING_NO_GRC.items()}

LABELLING_MLI_CLUSTER = {
    "PkC_cs": 6,
    "PkC_ss": 5,
    "MFB": 4,
    "MLI_B": 3,
    "MLI_A": 2,
    "GoC": 1,
    "GrC": 0,
    "unlabelled": -1,
}


CORRESPONDENCE_MLI_CLUSTER = {
    value: key for key, value in LABELLING_MLI_CLUSTER.items()
}

LABELLING_MLI_CLUSTER_NO_GRC = {
    "PkC_cs": 5,
    "PkC_ss": 4,
    "MFB": 3,
    "MLI_B": 2,
    "MLI_A": 1,
    "GoC": 0,
}

CORRESPONDENCE_MLI_CLUSTER_NO_GRC = {
    value: key for key, value in LABELLING_MLI_CLUSTER_NO_GRC.items()
}


LAYERS = {0: "unknown", 1: "GCL", 2: "PCL", 3: "ML"}
LAYERS_CORRESPONDENCE = {
    "unknown": 0,
    "GCL": 1,
    "PCL": 2,
    "": 0,
    "ML": 3,
    "GrC_layer": 1,
    "PkC_layer": 2,
    "ML_layer": 3,
}
# pylint: disable=no-member


def save(file_name, obj):
    with open(file_name, "wb") as fobj:
        pickle.dump(obj, fobj)


def load(file_name):
    with open(file_name, "rb") as fobj:
        return pickle.load(fobj)


def get_neuron_attr(hdf5_file_path, id=None, file=None):
    """
    Prompts the user to select a given neuron's file to load.
    Otherwise, can specify which neuron's id and which file we want to load directly
    """
    neuron_ids = []
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        for name in hdf5_file:
            if "neuron" not in name:
                continue
            pi = name.split("_")[0]
            neuron_id = name.split("_")[-1]
            neuron_ids.append(neuron_id)
        if id is None:
            return get_neuron_attr_generic(neuron_ids, pi, hdf5_file)
        return_path = f"{str(pi)}_neuron_{str(id)}/{str(file)}"
        return hdf5_file[return_path][(...)]


def get_neuron_attr_generic(neuron_ids, pi, hdf5_file):
    neuron_ids = [int(neuron_id) for neuron_id in neuron_ids]
    first_input = input(f"Select a neuron id from: {neuron_ids}")
    if first_input == "":
        print("No neuron id selected, exiting")
        return None
    first_path = f"{str(pi)}_neuron_{str(first_input)}"

    second_input = input(f"Select a file to load from: {ls(hdf5_file[first_path])}")
    if second_input == "":
        print("No attribute selected, exiting")
        return None
    second_path = first_path + "/" + str(second_input)

    return hdf5_file[second_path][(...)]


def ls(hdf5_file_path):
    """
    Given an hdf5 file path or an open hdf5 file python object, returns the child directories.
    """
    if type(hdf5_file_path) is not str:
        return list(hdf5_file_path.keys())
    with h5py.File(hdf5_file_path, "r") as hdf5_file:
        return list(hdf5_file.keys())


def normalise_wf(wf):
    """
    Custom normalisation so that the through of the waveform is set to -1
    or the peak is set to +1 if the waveform is dendritic
    """
    baseline = wf[:, :20].mean(axis=1, keepdims=True)
    wf = wf - baseline
    through = wf.min()
    peak = wf.max()
    return wf / np.abs(through) if np.abs(through) > np.abs(peak) else wf / np.abs(peak)


def crop_original_wave(waveform, central_range=60, n_channels=10):
    """
    It takes a waveform of shape (n_channels, central_range) and returns a copy of
    the waveform with the central 60 samples in the horizontal direction and the central 10
    channels in the vertical direction.

    The function first finds the peak channels by sorting the channels by their maximum amplitude.
    If the waveform has less than or equal to n_channels channels, the function returns the
    waveform cropped to the central range and the middle channel. Otherwise, the function finds
    the peak channel by iterating through the sorted channels by amplitude and selecting the
    channel with the highest amplitude that is not too close to the edge. The function then
    returns the waveform cropped to the central range and exactly n_channels channels around the
    peak channel. If n_channels is odd, the peak channel will be at the center (unless it is too close to the edge).

    Args:
      waveform: the waveform to be preprocessed
      central_range: the number of samples to take from the center of the waveform. Defaults to 60
      n_channels: The number of channels to use around the peak. Defaults to 10

    Returns:
      The waveform cropped to the central range and the number of channels specified.
    """
    # First argsort to find the peak channels
    # Then if the absolute max amplitude channel is "too close to the edge", find the second max and so on.
    # If the peak channel is in the middle, then just take the central channels
    centre = waveform.shape[1] // 2
    if waveform.shape[0] <= n_channels:
        return (
            waveform[:, (centre - central_range // 2) : (centre + central_range // 2)],
            waveform.shape[0] // 2,
        )

    channels_by_amplitude = np.argsort(np.ptp(waveform, axis=1))

    cropped_wvf = np.array([0])
    i = 1
    while cropped_wvf.shape[0] < n_channels and i < waveform.shape[0]:
        peak_channel = channels_by_amplitude[-i]
        if n_channels % 2 == 0:
            start_channel = max(0, peak_channel - n_channels // 2)
        else:
            start_channel = max(0, peak_channel - (n_channels - 1) // 2)
        end_channel = min(waveform.shape[0], start_channel + n_channels)
        cropped_wvf = waveform[
            start_channel:end_channel,
            (centre - central_range // 2) : (centre + central_range // 2),
        ].copy()
        i += 1

    return cropped_wvf, peak_channel


def crop_chanmap(chanmap, peak_channel_idx, n_channels=10):
    return np.array(
        chanmap[
            (peak_channel_idx - n_channels // 2) : (peak_channel_idx + n_channels // 2),
            :,
        ]
    )


def resample_acg(acg, window_size=20, keep_same_size=True):
    """
    Given an ACG, add artificial points to it.
    If keep_same_size is True, the ACG will be of the same size: this is achieved
    by undersapling points at the end of the ACG.
    """
    y = np.array(acg).copy()
    X = np.linspace(0, len(y), len(y))

    interpolated_window = y[:window_size]
    # Create interpolating points
    avg_arr = (interpolated_window + np.roll(interpolated_window, -1)) / 2.0
    avg_enhanced = np.vstack([interpolated_window, avg_arr]).flatten("F")[:-1]

    # Create new_y enhanced with interpolating points
    new_y = np.concatenate((avg_enhanced.ravel(), y[window_size:].ravel()), axis=0)

    if keep_same_size is False:
        return new_y

    # Select final points to remove
    idxes = np.ones_like(new_y).astype(bool)
    idxes[-2 * window_size :: 2] = False

    return new_y[idxes]


def get_h5_absolute_ids(h5_path):
    neuron_ids = []
    lab = None
    with h5py.File(h5_path, "r") as hdf5_file:
        for name in hdf5_file:
            if "neuron" in name:
                neuron_id = name.split("_")[-1]
                neuron_ids.append(int(neuron_id))
                if lab is None:
                    lab = name.split("_")[0]
    return neuron_ids, lab


def decode_string(value):
    """
    The function decodes a given value to a string if it is of type bytes or numpy bytes, and returns
    the original value otherwise.

    Args:
      value: The input value that needs to be decoded.

    Returns:
      The decoded string value of the input `value`.
    """
    if type(value) in (bytes, np.bytes_):
        return str(value.decode("utf-8"))
    elif type(value) == np.ndarray:
        return str(value.item().decode("utf-8"))
    return value


def process_label(label):
    if len(label) == 0 or label == "unlabeled":
        return 0
    return label


class NeuronsDataset:
    """
    Custom class for the cerebellum dataset, containing all information about the labelled and unlabelled neurons.
    """

    def __init__(
        self,
        dataset,
        quality_check=True,
        normalise_wvf=False,
        normalise_acg=False,
        resample_acgs=False,
        cut_acg=True,
        central_range=CENTRAL_RANGE,
        n_channels=N_CHANNELS,
        flip_waveforms=True,
        reshape_fortran_to_c=False,
        _label="ground_truth_label",
        _labelling=LABELLING,
        _use_amplitudes=False,
        _bin_size=1,
        _win_size=200,
        _debug=False,
        _lisberger=False,
        _labels_only=False,
        _id_type="neuron_relative_id",
        _extract_mli_clusters=False,
        _extract_layer=False,
        _keep_singchan=True,
    ):
        # Store useful metadata about how the dataset was extracted
        self.dataset = dataset
        self._n_channels = n_channels
        self._central_range = central_range
        self.flip_waveforms = flip_waveforms
        self._sampling_rate = get_neuron_attr(dataset, 0, "sampling_rate").item()
        self.mli_clustering = _extract_mli_clusters
        self._keep_singchan = _keep_singchan

        # Initialise empty lists to extract data
        self.wf_list = []
        self.conformed_waveforms = []
        self.acg_list = []
        self.spikes_list = []
        self.labels_list = []
        self.info = []
        self.chanmap_list = []
        self.genetic_line_list = []
        self.h5_ids = []

        if not self._keep_singchan:
            self.singchan_mask = []
        if _use_amplitudes:
            self.amplitudes_list = []

        if _extract_layer:
            self.layer_list = []

        if _extract_mli_clusters:
            _labelling = LABELLING_MLI_CLUSTER

        neuron_ids, lab = get_h5_absolute_ids(dataset)

        if not quality_check:
            self.quality_checks_mask = []
            self.fn_fp_list = []
            self.sane_spikes_list = []

        discarded_df = pd.DataFrame(columns=["neuron_id", "label", "dataset", "reason"])
        for i, wf_n in tqdm(
            enumerate(np.sort(neuron_ids)),
            total=len(neuron_ids),
            desc="Reading dataset",
            leave=False,
        ):
            try:
                # Get the label for this wvf
                label = get_neuron_attr(dataset, wf_n, _label).ravel()[0]
                label = decode_string(label)
                label = process_label(label)

                # If the neuron is labelled we extract it anyways
                if label != 0 and not isinstance(label, (np.ndarray, np.int64)):
                    if _extract_mli_clusters and label == "MLI":
                        mli_cluster = get_neuron_attr(dataset, wf_n, "mli_cluster")
                        mli_cluster = decode_string(mli_cluster)
                        mli_cluster = mli_cluster.replace("1", "A").replace("2", "B")
                        label = mli_cluster
                    self.labels_list.append(label)

                else:
                    if _labels_only:
                        continue
                    self.labels_list.append("unlabelled")

                spikes = get_neuron_attr(dataset, wf_n, "spike_indices")

                if not _lisberger:
                    sane_spikes = get_neuron_attr(dataset, wf_n, "sane_spikes")
                    fn_fp_spikes = get_neuron_attr(
                        dataset, wf_n, "fn_fp_filtered_spikes"
                    )
                else:
                    sane_spikes = np.ones_like(spikes, dtype=bool)
                    fn_fp_spikes = np.ones_like(spikes, dtype=bool)

                quality_mask = fn_fp_spikes & sane_spikes

                # if spikes is void after quality checks, skip this neuron (if quality checks are enabled)
                if len(spikes[quality_mask].copy()) == 0 and quality_check:
                    dataset_name = (
                        get_neuron_attr(dataset, wf_n, "dataset_id")
                        .ravel()[0]
                        .decode("utf-8")
                    )
                    discarded_df = pd.concat(
                        (
                            discarded_df,
                            pd.DataFrame(
                                {
                                    "neuron_id": [
                                        get_neuron_attr(
                                            dataset,
                                            wf_n,
                                            _id_type,
                                        ).ravel()[0]
                                    ],
                                    "label": [label],
                                    "dataset": [dataset_name],
                                    "reason": ["quality checks"],
                                }
                            ),
                        ),
                        ignore_index=True,
                    )
                    del self.labels_list[-1]
                    continue

                # Even without quality checks, we want to save only the spikes in the spontaneous period
                if quality_check:
                    self.spikes_list.append(spikes[quality_mask].astype(int))
                else:
                    self.spikes_list.append(spikes[sane_spikes].astype(int))
                    self.fn_fp_list.append(fn_fp_spikes)
                    self.sane_spikes_list.append(sane_spikes)

                    if len(spikes[quality_mask].copy()) == 0:
                        self.quality_checks_mask.append(False)
                    else:
                        self.quality_checks_mask.append(True)

                # Extract amplitudes if requested
                if _use_amplitudes:
                    amplitudes = get_neuron_attr(dataset, wf_n, "amplitudes")
                    try:
                        self.amplitudes_list.append(
                            amplitudes[sane_spikes]
                            if not quality_check
                            else amplitudes[quality_mask]
                        )
                    except IndexError:
                        # print(
                        #     f"Shape mismatch between amplitudes and spikes for neuron {wf_n}. {len(amplitudes)} vs {len(spikes)}."
                        # )
                        # print("Enforcing them to be of equal size.")
                        if quality_check:
                            amplitudes, quality_mask = force_amplitudes_length(
                                amplitudes, quality_mask
                            )
                            self.amplitudes_list.append(amplitudes[quality_mask])
                        else:
                            amplitudes, sane_spikes = force_amplitudes_length(
                                amplitudes, sane_spikes
                            )
                            self.amplitudes_list.append(amplitudes[sane_spikes])

                discard_wave = False
                # Extract waveform using provided parameters
                wf = get_neuron_attr(dataset, wf_n, "mean_waveform_preprocessed")

                if reshape_fortran_to_c:
                    wf = wf.reshape(list(wf.shape)[::-1])

                # Make sure if we need to transpose the waveform or not
                if wf.shape[0] > wf.shape[1]:
                    wf = wf.T

                # Also, if the waveform is 1D (i.e. only one channel), we need to tile it to make it 2D.
                if wf.squeeze().ndim == 1:
                    if _keep_singchan:
                        wf = np.tile(wf, (n_channels, 1))
                    else:
                        discard_wave = True

                if not self._keep_singchan:
                    self.singchan_mask.append(discard_wave)

                # Alternatively, if it is not spread on enough channels, we want to tile the remaining
                if wf.shape[0] < n_channels:
                    wf = pad_matrix_with_decay(wf, n_channels)

                # Extract the waveform conformed to the common preprocessing strategy in C4
                peak_chan = np.argmax(np.ptp(wf, axis=1))
                conformed_wave = preprocess_template(
                    wf[peak_chan, :], self._sampling_rate,
                    peak_sign = "negative" if self.flip_waveforms else None,
                )
                self.conformed_waveforms.append(conformed_wave)

                if normalise_wvf:
                    cropped_wave, peak_idx = crop_original_wave(
                        normalise_wf(wf), central_range, n_channels
                    )
                    self.wf_list.append(cropped_wave.ravel().astype(float))
                else:
                    cropped_wave, peak_idx = crop_original_wave(
                        wf, central_range, n_channels
                    )
                    self.wf_list.append(cropped_wave.ravel().astype(float))
                if (
                    self.wf_list[-1].shape[0]
                    != n_channels * central_range
                    # or discard_wave
                ):
                    dataset_name = (
                        get_neuron_attr(dataset, wf_n, "dataset_id")
                        .ravel()[0]
                        .decode("utf-8")
                    )
                    discarded_df = pd.concat(
                        (
                            discarded_df,
                            pd.DataFrame(
                                {
                                    "neuron_id": [
                                        get_neuron_attr(
                                            dataset,
                                            wf_n,
                                            _id_type,
                                        ).ravel()[0]
                                    ],
                                    "label": [label],
                                    "dataset": [dataset_name],
                                    "reason": ["waveform shape"],
                                }
                            ),
                        ),
                        ignore_index=True,
                    )
                    del self.labels_list[-1]
                    del self.wf_list[-1]
                    del self.spikes_list[-1]
                    del self.conformed_waveforms[-1]

                    if not quality_check:
                        del self.fn_fp_list[-1]
                        del self.sane_spikes_list[-1]
                        del self.quality_checks_mask[-1]
                    if hasattr(self, "amplitudes_list"):
                        del self.amplitudes_list[-1]
                    continue

                # Extract ACG. Even if we don't apply quality checks, we still want to use spikes from the spontaneous period

                acg_spikes = (
                    spikes[quality_mask] if quality_check else spikes[sane_spikes]
                )

                if len(acg_spikes) == 0:
                    self.acg_list.append(
                        np.zeros(int(_win_size / _bin_size + 1)).astype(float)
                    )

                else:
                    if normalise_acg:
                        acg = npyx.corr.acg(
                            ".npyx_placeholder",
                            4,
                            _bin_size,
                            _win_size,
                            fs=self._sampling_rate,
                            train=acg_spikes,
                        )
                        normal_acg = np.clip(acg / np.max(acg), 0, 10)
                        # For some bin and window sizes, the ACG is all zeros. In this case, we want to set it to a constant value
                        normal_acg = np.nan_to_num(normal_acg, nan=0)
                        self.acg_list.append(normal_acg.astype(float))
                    else:
                        acg = npyx.corr.acg(
                            ".npyx_placeholder",
                            4,
                            _bin_size,
                            _win_size,
                            fs=self._sampling_rate,
                            train=acg_spikes,
                        )
                        self.acg_list.append(acg.astype(float))

                # Extract useful metadata
                dataset_name = (
                    get_neuron_attr(dataset, wf_n, "dataset_id")
                    .ravel()[0]
                    .decode("utf-8")
                )
                neuron_id = get_neuron_attr(
                    dataset,
                    wf_n,
                    _id_type,
                ).ravel()[0]
                if not isinstance(neuron_id, (np.ndarray, np.int64, np.int32, int)):
                    neuron_id = neuron_id.decode("utf-8")
                neuron_metadata = dataset_name + "/" + str(neuron_id)
                self.info.append(str(neuron_metadata))

                self.h5_ids.append(f"{lab}_neuron_{wf_n}")

                chanmap = get_neuron_attr(dataset, wf_n, "channelmap")
                chanmap = crop_chanmap(np.array(chanmap), peak_idx, n_channels)
                self.chanmap_list.append(chanmap)

                try:
                    genetic_line = get_neuron_attr(dataset, wf_n, "line")
                    self.genetic_line_list.append(genetic_line.item().decode("utf-8"))
                except KeyError:
                    self.genetic_line_list.append("unknown")

                if _extract_layer:
                    if _lisberger:
                        layer = get_neuron_attr(dataset, wf_n, "human_layer")
                    else:
                        layer = get_neuron_attr(dataset, wf_n, "phyllum_layer")
                    layer = decode_string(layer)
                    self.layer_list.append(layer)

            except KeyError:
                if _debug:
                    raise
                dataset_name = (
                    get_neuron_attr(dataset, wf_n, "dataset_id")
                    .ravel()[0]
                    .decode("utf-8")
                )
                discarded_df = pd.concat(
                    (
                        discarded_df,
                        pd.DataFrame(
                            {
                                "neuron_id": [
                                    get_neuron_attr(
                                        dataset,
                                        wf_n,
                                        _id_type,
                                    ).ravel()[0]
                                ],
                                "label": [label],
                                "dataset": [dataset_name],
                                "reason": ["KeyError"],
                            }
                        ),
                    ),
                    ignore_index=True,
                )
                continue

        self.discarded_df = discarded_df
        if cut_acg:
            acg_list_cut = [x[len(x) // 2 :] for x in self.acg_list]
        else:
            acg_list_cut = self.acg_list
        if resample_acgs:
            acg_list_resampled = list(map(resample_acg, acg_list_cut))
        else:
            acg_list_resampled = acg_list_cut

        self.targets = np.array(
            (pd.Series(self.labels_list).replace(_labelling).values)
        )
        if len(self.wf_list) == 0:
            raise NotImplementedError(
                "No neurons could be extracted from the dataset with the provided parameters."
            )
        self.wf = np.stack(self.wf_list, axis=0)
        self.acg = np.stack(acg_list_resampled, axis=0)

        if hasattr(self, "quality_checks_mask"):
            self.quality_checks_mask = np.array(self.quality_checks_mask)

        print(
            f"{sum(self.targets == -1)} unlabelled and {sum(self.targets != -1)} labelled neurons loaded. \n"
            f"{len(discarded_df)} neurons discarded, of which labelled: {len(discarded_df[discarded_df.label != 0])}. More details at the 'discarded_df' attribute. \n"
        )

        # Compute conformed_waveforms
        # self.conformed_waveforms = []
        # for wf in self.wf.reshape(-1, self._n_channels, self._central_range):
        #     peak_chan = np.argmax(np.max(np.abs(wf), axis=1))
        #     conformed_wave = preprocess_template(wf[peak_chan, :], self._sampling_rate)
        #     self.conformed_waveforms.append(conformed_wave)
        self.conformed_waveforms = np.stack(self.conformed_waveforms, axis=0)

        self.h5_ids = np.array(self.h5_ids)

    def make_labels_only(self):
        """
        It removes all the data points that have no labels
        """
        mask = self.targets != -1
        self._apply_mask(mask)

    def make_unlabelled_only(self):
        """
        Removes all datapoints that have labels
        """
        mask = self.targets == -1
        self._apply_mask(mask)

    def _apply_mask(self, mask):
        self.wf = self.wf[mask]
        self.conformed_waveforms = self.conformed_waveforms[mask]
        self.acg = self.acg[mask]
        self.targets = self.targets[mask]
        self.info = np.array(self.info)[mask].tolist()
        self.spikes_list = np.array(self.spikes_list, dtype=object)[mask].tolist()
        self.labels_list = np.array(self.labels_list)[mask].tolist()
        self.acg_list = np.array(self.acg_list)[mask].tolist()
        self.h5_ids = self.h5_ids[mask]
        try:
            self.chanmap_list = np.array(self.chanmap_list, dtype=object)[mask].tolist()
        # Numpy has still a bug in treating arrays as objects
        except ValueError:
            self.chanmap_list = [self.chanmap_list[i] for i in np.where(mask)[0]]

        self.genetic_line_list = np.array(self.genetic_line_list, dtype=object)[
            mask
        ].tolist()

        if hasattr(self, "amplitudes_list"):
            self.amplitudes_list = np.array(self.amplitudes_list, dtype=object)[
                mask
            ].tolist()
        if hasattr(self, "quality_checks_mask"):
            self.quality_checks_mask = self.quality_checks_mask[mask]
            self.fn_fp_list = np.array(self.fn_fp_list, dtype=object)[mask].tolist()
            self.sane_spikes_list = np.array(self.sane_spikes_list, dtype=object)[
                mask
            ].tolist()
        if hasattr(self, "full_dataset"):
            self.full_dataset = self.full_dataset[mask]
        if hasattr(self, "layer_list"):
            self.layer_list = np.array(self.layer_list, dtype=object)[mask].tolist()
        if hasattr(self, "singchan_mask"):
            self.singchan_mask = np.array(self.singchan_mask)[mask].tolist()

    def make_full_dataset(self, wf_only=False, acg_only=False):
        """
        This function takes the waveform and ACG data and concatenates them into a single array

        Args:
            wf_only: If True, only the waveform data will be used. Defaults to False
            acg_only: If True, only the ACG data will be used. Defaults to False
        """
        if wf_only:
            self.full_dataset = self.wf
        elif acg_only:
            self.full_dataset = self.acg
        else:
            self.full_dataset = np.concatenate((self.wf, self.acg), axis=1)

    def min_max_scale(self, mean=False, acg_only=True):
        """
        `min_max_scale` takes the waveform and ACG and scales them to the range [-1, 1] by dividing by the
        maximum absolute value of the waveform and ACG

        Args:
            mean: If True, the mean of the first 100 largest waveforms will be used as the scaling value.
            If False, the maximum value of the waveforms will be used. Defaults to False.
        """
        if mean:
            self._scale_value_wf = (np.sort(self.wf.ravel())[:100]).mean()
            self._scale_value_acg = (np.sort(self.acg.ravel())[-100:]).mean()
        else:
            self._scale_value_wf = np.max(np.abs(self.wf))
            self._scale_value_acg = np.max(np.abs(self.acg))

        if not acg_only:
            self.wf = self.wf / self._scale_value_wf
        self.acg = self.acg / self._scale_value_acg

    def filter_out_granule_cells(self, return_mask=False):
        """
        Filters out granule cells from the dataset and returns new LABELLING and CORRESPONDENCE dictionaries for plotting.
        """

        granule_cell_mask = self.targets == LABELLING["GrC"]

        self._apply_mask(~granule_cell_mask)
        self.targets = (self.targets - 1).astype(int)
        self.targets[self.targets < 0] = -1  # Reset the label of unlabeled cells

        # To convert text labels to numbers
        new_labelling = (
            LABELLING_NO_GRC
            if not self.mli_clustering
            else LABELLING_MLI_CLUSTER_NO_GRC
        )
        new_correspondence = (
            CORRESPONDENCE_NO_GRC
            if not self.mli_clustering
            else CORRESPONDENCE_MLI_CLUSTER_NO_GRC
        )
        if return_mask:
            return new_labelling, new_correspondence, granule_cell_mask

        return new_labelling, new_correspondence

    def wvf_from_info(self, dp, unit):
        info_path = dp + "/" + str(unit)
        assert info_path in self.info, "No neuron for the dp and unit provided"

        idx = self.info.index(info_path)

        return self.wf[idx].reshape(self._n_channels, self._central_range)

    def train_from_info(self, dp, unit):
        info_path = dp + "/" + str(unit)
        assert info_path in self.info, "No neuron for the dp and unit provided"

        idx = self.info.index(info_path)

        return self.spikes_list[idx]

    def plot_from_info(self, dp, unit):
        info_path = dp + "/" + str(unit)
        assert info_path in self.info, "No neuron for the dp and unit provided"

        wvf = self.wvf_from_info(dp, unit)
        train = self.train_from_info(dp, unit)

        npyx.plot.plt_wvf(wvf.T)
        plt.show()
        npyx.plot.plot_acg(".npyx_placeholder", 0, train=train)
        plt.show()

    def apply_quality_checks(self):
        """
        It takes a dataset, checks that it has a quality_checks_mask attribute, and then applies that
        mask to the dataset

        Returns:
          A new dataset with the quality checks applied.
        """
        assert hasattr(
            self, "quality_checks_mask"
        ), "No quality checks mask found, perhaps you have applied them already?"
        checked_dataset = copy.deepcopy(self)
        checked_dataset.spikes_list = [
            train[fn_fp_mask[sane_mask]]
            for train, fn_fp_mask, sane_mask in zip(
                self.spikes_list, self.fn_fp_list, self.sane_spikes_list
            )
        ]
        checked_dataset._apply_mask(checked_dataset.quality_checks_mask)
        del checked_dataset.quality_checks_mask
        del checked_dataset.fn_fp_list
        del checked_dataset.sane_spikes_list

        return checked_dataset

    def save(self, path):
        """
        Saves the dataset to a given path

        Args:
            path: Path to save the dataset to
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __len__(self):
        return len(self.wf)


def merge_h5_datasets(*args: NeuronsDataset) -> NeuronsDataset:
    """Merges multiple NeuronsDatasets instances into one"""

    def merge_attributes(attr_name, merge_func, dtype=None):
        if hasattr(new_dataset, attr_name):
            if hasattr(dataset, attr_name):
                attr_value = getattr(new_dataset, attr_name)
                other_attr_value = getattr(dataset, attr_name)
                if dtype:
                    attr_value = np.array(attr_value, dtype=dtype)
                    other_attr_value = np.array(other_attr_value, dtype=dtype)
                if merge_func in (np.hstack, np.vstack):
                    setattr(
                        new_dataset,
                        attr_name,
                        merge_func((attr_value, other_attr_value)),
                    )
                else:
                    setattr(
                        new_dataset, attr_name, merge_func(attr_value, other_attr_value)
                    )
            else:
                raise NotImplementedError(
                    "Attempted to merge datasets with different attributes"
                )

    new_dataset = copy.deepcopy(args[0])
    for dataset in args[1:]:
        assert isinstance(dataset, NeuronsDataset)
        new_dataset.wf = np.vstack((new_dataset.wf, dataset.wf))
        new_dataset.acg = np.vstack((new_dataset.acg, dataset.acg))
        new_dataset.targets = np.hstack((new_dataset.targets, dataset.targets))
        new_dataset.chanmap_list += dataset.chanmap_list
        new_dataset.conformed_waveforms = np.vstack(
            (new_dataset.conformed_waveforms, dataset.conformed_waveforms)
        )
        new_dataset.genetic_line_list += dataset.genetic_line_list
        new_dataset.info = np.hstack(
            (np.array(new_dataset.info), np.array(dataset.info))
        ).tolist()
        new_dataset.acg_list = np.vstack(
            (np.array(new_dataset.acg_list), np.array(dataset.acg_list))
        ).tolist()
        new_dataset.h5_ids = np.hstack((new_dataset.h5_ids, dataset.h5_ids))

        merge_attributes("spikes_list", np.hstack, dtype=object)
        new_dataset.discarded_df = pd.concat(
            (new_dataset.discarded_df, dataset.discarded_df), axis=0
        )
        new_dataset.labels_list += dataset.labels_list

        merge_attributes("amplitudes_list", lambda x, y: x + y)
        merge_attributes("quality_checks_mask", np.hstack)
        merge_attributes("fn_fp_list", lambda x, y: x + y)
        merge_attributes("sane_spikes_list", lambda x, y: x + y)
        merge_attributes("layer_list", lambda x, y: x + y)

    new_dataset.dataset = "merged"

    return new_dataset


def resample_waveforms(
    dataset: NeuronsDataset, new_sampling_rate: int = 30_000
) -> NeuronsDataset:
    """
    It takes a dataset, resizes the waveforms to a new sampling rate, and returns a new dataset with the
    resized waveforms

    Args:
      dataset (NeuronsDataset): the dataset to be resampled
      new_sampling_rate (int): the sampling rate of the new waveforms. Defaults to 30_000

    Returns:
      A new dataset with the same properties as the original dataset, but with the waveforms resampled.
    """

    import torch
    from torchvision import transforms

    original_wf = dataset.wf.reshape(-1, 1, dataset._n_channels, dataset._central_range)

    new_range = int(dataset._central_range * new_sampling_rate / dataset._sampling_rate)

    resize = transforms.Resize((dataset._n_channels, new_range))

    resized_wf = resize(torch.tensor(original_wf)).squeeze().numpy()

    resized_wf = resized_wf.reshape(-1, dataset._n_channels * new_range)

    resampled_dataset = copy.deepcopy(dataset)
    resampled_dataset.wf = resized_wf
    resampled_dataset._central_range = new_range
    resampled_dataset.wf_list = list(resized_wf)

    return resampled_dataset


def force_amplitudes_length(amplitudes, times):
    if len(times) > len(amplitudes):
        times = times[: len(amplitudes)]
    if len(amplitudes) > len(times):
        amplitudes = amplitudes[: len(times)]
    return amplitudes, times


def preprocess_template(
    waveform: np.ndarray,
    original_sampling_rate: float = 30000,
    output_sampling_rate: float = 30000,
    clip_size: Tuple[float, float] = (1e-3, 2e-3),
    peak_sign: Union[None, str] = "negative",
    normalize: bool = True,
) -> np.ndarray:
    """
    This function preprocesses a given template by resampling it, aligning it to a peak, flipping it if
    necessary, and normalizing it.

    Args:
      template (np.ndarray): A numpy array representing a waveform template.
      original_sampling_rate (float): The original sampling rate of the input template waveform in Hz
    (Hertz). Defaults to 30000
      output_sampling_rate (float): The desired sampling rate of the output template. The function will
    resample the input template to match this sampling rate if the original sampling rate is different.
    Defaults to 30000
      clip_size (Tuple[float, float]): The clip_size parameter is a tuple of two floats representing the
    start and end times (in seconds) of the desired clip from the original template waveform. The
    preprocess_template function uses this parameter to construct an output template of a specific
    length based on the desired sampling rate.
      peak_sign (Union[None, str]): The parameter "peak_sign" is used to specify whether the peak in the
    template should be positive or negative. It can take on the values "positive", "negative", or None.
    If it is set to "positive", the template will be flipped if the peak is negative, and if it is.
    Defaults to negative
      normalize (bool): A boolean parameter that determines whether or not to normalize the output
    template. If set to True, the output template will be normalized by dividing it by the absolute
    value of the peak amplitude. Defaults to True

    Returns:
      a preprocessed template as a numpy array.

    Authors:
       Original Julia Implementation by David J. Herzfeld <herzfeldd@gmail.com>
       Adapted to Python and multi-channel waveforms by @fededagos
    """
    assert original_sampling_rate >= output_sampling_rate

    # Check if provided waveform is 2D
    multi_chan = False
    if len(waveform.shape) == 2:
        peak_channel = np.argmax(np.max(np.abs(waveform), axis=1))
        template = waveform[peak_channel, :]
        multi_chan = True
    else:
        template = waveform

    if original_sampling_rate != output_sampling_rate:
        template = resample(
            template, int(output_sampling_rate / original_sampling_rate * len(template))
        )

    alignment_idx = int(round(abs(clip_size[0]) * output_sampling_rate))

    # Search through our template to find our desired alignment point
    # We only align to peaks, so our goal is to find a set of local peaks
    # first and then choose the optimal one

    peaks, _ = npyx.feat.detect_peaks(template, margin=0.5, onset=0.2)
    # If we don't find any peaks, we will search for them in a more brute force way
    if len(peaks) == 0:
        peaks = []
        for i in range(1, len(template) - 1):
            if (
                (template[i] > template[i - 1])
                and (template[i] >= template[i + 1])
                and (template[i] > 0)
            ):
                peaks.append(i)  # Positive peak
            elif (
                (template[i] < template[i - 1])
                and (template[i] <= template[i + 1])
                and (template[i] < 0)
            ):
                peaks.append(i)  # Negative peak

    # Given our list of peaks, our goal is to find the optimal peak,
    # typically this will be the maximum value, but we align to the first
    # peak that is at least 75% of the maximum value

    peak_values = np.abs(template[peaks])
    extremum = np.max(peak_values)
    reference_peak_idx = peaks[np.where(peak_values > 0.75 * extremum)[0][0]]

    peak_val = np.abs(template[reference_peak_idx])

    # Determine if we need to flip our template based on the value of the peak
    # ensuring that the peak is negative
    if (
        peak_sign is not None
        and peak_sign == "negative"
        and template[reference_peak_idx] > 0
    ):
        template = template * -1
        if multi_chan:
            waveform = waveform * -1
    elif (
        peak_sign is not None
        and peak_sign == "positive"
        and template[reference_peak_idx] < 0
    ):
        template = template * -1
        if multi_chan:
            waveform = waveform * -1

    # Construct our output template based on our desired clip_size
    num_indices = int(
        round((abs(clip_size[0]) + abs(clip_size[1])) * output_sampling_rate)
    )
    if reference_peak_idx < alignment_idx:
        if multi_chan:
            padding = np.tile(waveform[:, 0], (alignment_idx - reference_peak_idx, 1)).T
            waveform = np.concatenate((padding, waveform), axis=1)
        else:
            template = np.concatenate(
                (np.ones(alignment_idx - reference_peak_idx) * template[0], template)
            )
    elif reference_peak_idx > alignment_idx:
        shift = reference_peak_idx - alignment_idx
        if multi_chan:
            padding = np.tile(
                waveform[:, -1], (reference_peak_idx - alignment_idx, 1)
            ).T
            waveform = np.concatenate((waveform[:, shift:], padding), axis=1)
        else:
            template = np.concatenate((template[shift:], np.full(shift, template[-1])))

    if multi_chan:
        assert np.abs(waveform[peak_channel, alignment_idx]) == peak_val
    else:
        assert (
            np.abs(template[alignment_idx]) == peak_val
        ), f"Peak value is {peak_val}, but template value there is {template[alignment_idx]}"

    if len(template) > num_indices:
        template = template[:num_indices]
        if multi_chan:
            waveform = waveform[:, :num_indices]
    elif len(template) < num_indices:
        template = np.pad(
            template,
            (0, num_indices - len(template)),
            mode="constant",
            constant_values=template[-1],
        )
        if multi_chan:
            padding = np.tile(waveform[:, -1], (num_indices - len(waveform), 1)).T
            waveform = np.concatenate((waveform, padding), axis=1)

    assert len(template) == num_indices
    if multi_chan:
        assert (
            waveform.shape[1] == num_indices
        ), f"Expected waveform shape to be {num_indices} after processing but got {waveform.shape[1]}"

    # Remove any (noisy) offset
    num_indices = int(round(abs(clip_size[0]) * output_sampling_rate))
    template = template - np.median(template[:num_indices])
    if multi_chan:
        waveform = waveform - np.median(
            waveform[:, :num_indices], axis=1, keepdims=True
        )

    if normalize:
        # Normalize the result
        template = template / np.abs(template[alignment_idx])
        if multi_chan:
            waveform = waveform / np.abs(waveform[peak_channel, alignment_idx])

    return template if not multi_chan else waveform


def pad_matrix_with_decay(matrix, target_channels=10):
    n_channels, _ = matrix.shape
    padding_needed = target_channels - n_channels

    if padding_needed <= 0:
        return matrix

    # Calculate the maximum absolute amplitude as the reference amplitude
    reference_amplitude = np.max(np.abs(matrix))

    # Find the peak signal value and its position in the matrix
    peak_row = np.argmax(np.ptp(matrix, axis=1))

    # Find the closest non-peak signal value to the peak position
    distances_to_peak = np.abs(np.arange(n_channels) - peak_row)
    closest_non_peak_row = np.argmin(np.ma.masked_equal(distances_to_peak, 0))
    closest_non_peak_value = np.max(np.abs(matrix[closest_non_peak_row, :]))

    # Generate a decay pattern for padding based on the reference amplitude
    decay_factor = closest_non_peak_value / reference_amplitude
    decay_pattern = decay_factor ** np.arange(1, padding_needed + 1)

    # Calculate the required padding for both top and bottom separately
    top_padding = int(np.ceil(padding_needed / 2))
    bottom_padding = padding_needed - top_padding

    # Create separate top and bottom padding rows with the decay pattern
    top_padding_rows = (matrix[0] - closest_non_peak_value) * decay_pattern[
        :top_padding, np.newaxis
    ][::-1]
    bottom_padding_rows = (matrix[-1] - closest_non_peak_value) * decay_pattern[
        :bottom_padding, np.newaxis
    ]

    # Stack the padding and the original matrix vertically
    padded_matrix = np.vstack((top_padding_rows, matrix, bottom_padding_rows))
    return padded_matrix
