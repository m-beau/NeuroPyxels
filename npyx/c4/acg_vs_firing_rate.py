import argparse
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.optimize import OptimizeWarning
from scipy.signal import fftconvolve
from tqdm.auto import tqdm

import npyx.corr as corr
from npyx.spk_t import duplicates_mask

from .dataset_init import (
    extract_and_check,
    extract_and_merge_datasets,
    get_paths_from_dir,
)
from .monkey_dataset_init import MONKEY_CENTRAL_RANGE, get_lisberger_dataset
from .predict_cell_types import get_n_cores, redirect_stdout_fd

BIN_SIZE = 1
WIN_SIZE = 200


class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def fast_acg3d(
    spike_times,
    win_size,
    bin_size,
    fs=30000,
    num_firing_rate_bins=10,
    smooth=250,
    cut=None,
):
    if cut is not None:
        cut = min(cut, len(spike_times))
        spike_times = spike_times[:cut]

    assert fs > 0.0
    bin_size = np.clip(bin_size, 1000 * 1.0 / fs, 1e8)  # in milliseconds
    win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
    winsize_bins = 2 * int(0.5 * win_size * 1.0 / bin_size) + 1  # Both in millisecond
    assert winsize_bins >= 1
    assert winsize_bins % 2 == 1
    time_axis = np.linspace(-win_size / 2, win_size / 2, num=winsize_bins)
    spike_counts = np.zeros(
        (num_firing_rate_bins, len(time_axis))
    )  # Counts number of occurences of spikes in a given bin in time axis
    times = np.zeros(num_firing_rate_bins, dtype=np.int64)  # total occurence

    # Samples per bin
    samples_per_bin = int(np.ceil(fs / (1000 / bin_size)))

    # Convert times_1 and times_2 (which are in units of fs to units of bin_size)
    spike_times = np.floor(spike_times / samples_per_bin).astype(np.int64)

    # Convert times_1 into a binary spike train
    max_indices = int(np.ceil(max(spike_times[-1], spike_times[-1]) + 1))
    spiketrain = np.zeros(max_indices, dtype=bool)
    spiketrain[spike_times] = True

    bin_size_seconds = bin_size / 1000.0
    intervals = np.searchsorted(spike_times, np.arange(max_indices))
    firing_rate = 1 / (
        (bin_size_seconds) * (spike_times[intervals] - spike_times[intervals - 1])
    )
    firing_rate = np.nan_to_num(firing_rate)

    # Smooth the firing rate with a convolution if requested
    if type(smooth) in [int, float] and smooth > 0:
        kernel_size = int(np.ceil(smooth / bin_size))
        half_kernel_size = kernel_size // 2
        kernel = np.ones(kernel_size) / kernel_size

        # Pad the input firing rate array
        padded_firing_rate = np.pad(firing_rate, pad_width=kernel_size, mode="edge")

        # Convolve the padded firing rate with the kernel, use cupy if available as this is an expensive operation
        smoothed_firing_rate = fftconvolve(padded_firing_rate, kernel, mode="valid")

        # Trim the edges of the smoothed firing rate array to remove the padded values
        trimmed_smoothed_firing_rate = smoothed_firing_rate[
            half_kernel_size:-half_kernel_size
        ]

        firing_rate = trimmed_smoothed_firing_rate

    # Get firing rate quantiles
    quantile_bins = np.linspace(0, 1, num_firing_rate_bins + 3)[1:-1]
    firing_rate_bins = np.quantile(firing_rate[spike_times], quantile_bins)

    # Find the bin number for each spike based on its firing rate
    current_firing_rate = firing_rate[spike_times]
    current_firing_rate_bin_number = np.searchsorted(
        firing_rate_bins, current_firing_rate
    )
    current_firing_rate_bin_number[current_firing_rate_bin_number == 0] = 1
    current_firing_rate_bin_number[
        current_firing_rate_bin_number == len(firing_rate_bins)
    ] = (len(firing_rate_bins) - 1)

    # Calculate spike counts for each firing rate bin
    bin_indices = np.arange(num_firing_rate_bins)
    spike_counts = np.zeros((num_firing_rate_bins, len(time_axis)))
    for bin_number in bin_indices:
        bin_spikes = spike_times[current_firing_rate_bin_number == bin_number + 1]
        start = bin_spikes + np.ceil(time_axis[0] / bin_size)
        stop = start + len(time_axis)
        mask = (
            (start >= 0)
            & (stop < len(spiketrain))
            & (bin_spikes >= spike_times[0])
            & (bin_spikes < spike_times[-1])
        )
        masked_start = start[mask].astype(int)
        masked_stop = stop[mask].astype(int)

        spike_counts[bin_number, :] = np.sum(
            [
                spiketrain[masked_start[i] : masked_stop[i]]
                for i in range(len(masked_start))
            ],
            axis=0,
        )

        times[bin_number] += np.sum(mask)

    acg_3d = spike_counts / (np.ones((len(time_axis), num_firing_rate_bins)) * times).T
    # Divison by zero cases will return nans, so we fix this
    acg_3d = np.nan_to_num(acg_3d)
    # remove bin 0, which will always be 1
    acg_3d[:, acg_3d.shape[1] // 2] = 0

    return firing_rate_bins, acg_3d


def delete_spikes(spikes, deletion_prob=0.1):
    mask = np.random.rand(spikes.shape[0]) > deletion_prob
    return spikes[mask]


def add_spikes(spikes, max_addition=0.1):
    random_addition = np.random.randint(
        low=spikes[0],
        high=spikes[-1],
        size=int(spikes.shape[0] * max_addition),
    )
    return np.unique(np.concatenate((spikes, random_addition)))


def random_jitter(spikes, max_shift=60):
    random_moving = np.random.randint(-max_shift, max_shift, size=spikes.shape[0])
    return (spikes + random_moving).astype(int)


def augment_spikes(spikes_list, *transforms):
    new_spikes_list = spikes_list.copy()
    for spikes in tqdm(spikes_list, desc="Augmenting spikes"):
        if len(spikes) < 100:
            continue
        for transform in transforms:
            new_train = transform(spikes)
            duplicates_m = duplicates_mask(new_train, enforced_rp=1)
            new_train = new_train[~duplicates_m]
            new_spikes_list.append(new_train)
    return new_spikes_list


def aux_compute_acgs(spikes, win_size, bin_size, sampling_rate, fast, i):
    try:
        if fast:
            (_, acg_3d) = fast_acg3d(spikes, win_size, bin_size, fs=sampling_rate)
        else:
            (_, acg_3d) = corr.crosscorr_vs_firing_rate(
                spikes, spikes, win_size, bin_size, fs=sampling_rate
            )
    except IndexError:
        # Handles the occasional error of the fast function.
        try:
            (_, acg_3d) = corr.crosscorr_vs_firing_rate(
                spikes, spikes, win_size, bin_size, fs=sampling_rate
            )
        except IndexError:
            print(
                f"Error with neuron {i}. Both fast and slow functions failed. Skipping."
            )
            return None
    return acg_3d.ravel()


def main(
    data_path=".",
    dataset="feature_spaces",
    name="acgs_vs_firing_rate",
    labelled=True,
    augment=False,
    fast=False,
    monkey=False,
    log=True,
):
    # Parse arguments into class
    args = ArgsNamespace(
        data_path=data_path,
        dataset=dataset,
        name=name,
        labelled=labelled,
        augment=augment,
        fast=fast,
        monkey=monkey,
        log=log,
    )

    if args.monkey:
        datasets_abs = get_lisberger_dataset(args.data_path)

        # Extract and check the datasets, saving a dataframe with the results
        _, dataset_class = extract_and_check(
            datasets_abs,
            save_folder=args.data_path,
            lisberger=True,
            _label="expert_label",
            n_channels=10,
            central_range=MONKEY_CENTRAL_RANGE,
            _use_amplitudes=False,
            _lisberger=True,
            _id_type="neuron_id",
            _labels_only=args.labelled,
        )
        prefix = "monkey_"

    else:
        datasets_abs = get_paths_from_dir(
            args.data_path,
            include_medina=False,
            include_hull_unlab=not args.labelled,
        )

        dataset_class = extract_and_merge_datasets(
            *datasets_abs,
            quality_check=False,
            normalise_acg=False,
            labelled=args.labelled,
            _labels_only=args.labelled,
        )

        if args.labelled:
            prefix = ""
            used_mask = pd.read_csv(
                os.path.join(args.data_path, args.dataset, "dataset_info.csv")
            )["included"].values.astype(bool)

            # Use only the neurons that pass quality checks and are used by other models (e.g. RF)
            dataset_class._apply_mask(used_mask)

            dataset_class = dataset_class.apply_quality_checks()

        if not args.labelled:
            prefix = "unlabelled_"
            dataset_class.make_unlabelled_only()
            dataset_class = dataset_class.apply_quality_checks()

    suffix = ""
    acgs_3d = []
    spikes_list = dataset_class.spikes_list

    if args.augment:
        spikes_list = augment_spikes(
            spikes_list,
            delete_spikes,
            add_spikes,
            random_jitter,
            lambda spikes: delete_spikes(add_spikes(spikes)),
            lambda spikes: random_jitter(add_spikes(delete_spikes(spikes))),
        )
        suffix = "_augmented"

    if args.log:
        suffix += "_logscale"
        WIN_SIZE = 2000
    else:
        WIN_SIZE = 200

    # for i, spikes in tqdm(
    #     enumerate(spikes_list),
    #     total=len(spikes_list),
    #     desc="Computing 3D ACGs",
    #     position=0,
    #     leave=False,
    # ):
    #     try:
    #         if args.fast:
    #             (
    #                 _,
    #                 acg_3d,
    #             ) = fast_acg3d(
    #                 spikes, WIN_SIZE, BIN_SIZE, fs=dataset_class._sampling_rate
    #             )
    #         else:
    #             (
    #                 _,
    #                 acg_3d,
    #             ) = corr.crosscorr_vs_firing_rate(
    #                 spikes, spikes, WIN_SIZE, BIN_SIZE, fs=dataset_class._sampling_rate
    #             )
    #     except IndexError:
    #         # Handles the occasional error of the fast function.
    #         try:
    #             (
    #                 _,
    #                 acg_3d,
    #             ) = corr.crosscorr_vs_firing_rate(
    #                 spikes, spikes, WIN_SIZE, BIN_SIZE, fs=dataset_class._sampling_rate
    #             )
    #         except IndexError:
    #             print(
    #                 f"Error with neuron {i}. Both fast and slow functions failed. Skipping."
    #             )
    #             continue
    #     acgs_3d.append(acg_3d.ravel())

    # acgs_3d = np.stack(acgs_3d, axis=0)

    num_cores = get_n_cores(len(spikes_list))
    with redirect_stdout_fd(open(os.devnull, "w")):
        acgs_3d = Parallel(n_jobs=num_cores)(
            delayed(aux_compute_acgs)(
                spikes, WIN_SIZE, BIN_SIZE, dataset_class._sampling_rate, args.fast, i
            )
            for i, spikes in tqdm(
                enumerate(spikes_list),
                total=len(spikes_list),
                desc="Computing 3D ACGs",
                position=0,
                leave=False,
            )
        )

    acgs_3d = np.stack([acg for acg in acgs_3d if acg is not None], axis=0)

    if not os.path.exists(os.path.join(args.data_path, args.name)):
        os.mkdir(os.path.join(args.data_path, args.name))

    with open(
        os.path.join(args.data_path, args.name, f"{prefix}acgs_3d{suffix}.npy"), "wb"
    ) as f:
        np.save(f, acgs_3d)

    log_acgs = []
    if args.log:
        for acg in acgs_3d:
            acg = acg.reshape(10, -1)
            acg, _ = corr.convert_acg_log(acg, BIN_SIZE, WIN_SIZE)
            log_acgs.append(acg.ravel())
        log_acgs = np.stack(log_acgs, axis=0)
        with open(
            os.path.join(args.data_path, args.name, f"{prefix}acgs_3d{suffix}.npy"),
            "wb",
        ) as f:
            np.save(f, log_acgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a random forest model on the given features."
    )

    parser.add_argument(
        "-dp",
        "--data-path",
        type=str,
        default=".",
        help="Path to the folder containing the .h5 files.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="feature_spaces",
        help="Name of the dataset used to filter labelled units, in case we are using labels.",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="acgs_vs_firing_rate",
        help="Name assigned to the folder containing the 3D_acgs.",
    )

    parser.add_argument("--labelled", action="store_true")
    parser.add_argument("--unlabelled", dest="labelled", action="store_false")
    parser.set_defaults(labelled=True)

    parser.add_argument("--augment", action="store_true")
    parser.set_defaults(augment=False)

    parser.add_argument("--fast", action="store_true")
    parser.set_defaults(fast=False)

    parser.add_argument("--monkey", action="store_true")
    parser.set_defaults(monkey=False)

    parser.add_argument("--log", action="store_true")
    parser.set_defaults(log=False)

    args = parser.parse_args()

    main(**vars(args))
