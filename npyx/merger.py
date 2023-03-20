# -*- coding: utf-8 -*-
"""
2019-06-13
@author: Maxime Beau, Neural Computations Lab, University College London

Dataset merger class: represents a collection of Neuropixels Datasets (no theoretical upper limit on number of probes)
as a networkx network
with nodes being characterized as cell types
and edges as putative connections.

The way to use it is as follow:

# initiation
dp_dic = {'probe1': 'path/to/dataset1', 'probe2':'path/to/dataset2', ...}
merged = Merger(dp_dic)

# Connect the graph with putative monosynaptic connections
merged.connect_graph()

# Plot the graph
merged.plot_graph()

# Get N of putative connections of a given node spotted on the graph
merged.get_node_edges(node)

# Only keep a set of relevant nodes or edges and plot it again
merged.keep_edges(edges_list)
merged.keep_nodes_list(nodes_list)
merged.plot_graph()

# every graph operation of dataset merger can be performed on external networkx graphs
# provided with the argument 'src_graph'
g=merged.get_graph_copy(mergerGraph='undigraph')
merged.keep_nodes_list(nodes_list, src_graph=g) # g itself will be modified, not need to do g=...
merged.plot_graph(graph_src=g)


"""


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import os.path as op

opj = op.join
from pathlib import Path

import numpy as np
import pandas as pd

from npyx.utils import npa, align_timeseries_interpol, assert_float

from npyx.inout import get_npix_sync
from npyx.gl import (
    get_units,
    load_merged_units_qualities,
    detect_new_spikesorting,
    save_qualities,
)


def merge_datasets(datapaths, again=False):
    """
    Merges datasets together and aligns data accross probes, modelling drift as a affine function.

    Any npyx function will run on a merged dataset just as on a regular dataset directory.

    If datasets have already been merged, will simply return the right datapath and dataset table.

    Arguments:
    - datapaths: dict of structure
        {'name_probe_1':'path/to/kilosort/output1',
         'name_probe_2':'path/to/kilosort/output2', ...}
    - again: bool, whether to force re-merge the datasets (even if a previous merge is found).

    Returns:
        - dp_merged: str, path to merged dataset
        - ds_table: pandas dataframe, containing datasets names, paths, probe names
    """

    # Handle datapaths format
    val_e = ValueError(
        """
    Datapath should be a dict of kilosort paths which exist:
    {'name_probe_1':'path/to/kilosort/output1', ...}"""
    )
    if type(datapaths) is dict:
        for i, v in datapaths.items():
            datapaths[i] = Path(v)
            if not datapaths[i].exists():
                raise val_e
    else:
        raise val_e

    # Make an alphabetically indexed table of all fed datasets (if several)
    ds_table = pd.DataFrame(columns=["dataset_name", "dp", "probe"])
    for i, (prb, dp) in enumerate(datapaths.items()):
        ds_table.loc[i, "dataset_name"] = op.basename(dp)
        ds_table.loc[i, "dp"] = str(dp)
        ds_table.loc[i, "probe"] = prb
    ds_table.insert(
        0, "dataset_i", ds_table["dataset_name"].argsort()
    )  # sort table by dataset_name alphabetically
    ds_table.sort_values("dataset_i", axis=0, inplace=True)
    ds_table.set_index("dataset_i", inplace=True)

    n_datasets = len(ds_table.index)
    mess_prefix = "\033[34;1m--- "
    mess_suffix = "\033[0m"
    if n_datasets == 1:
        print(
            f"{mess_prefix}Only one dataset provided - simply returning its path.{mess_suffix}."
        )
        return ds_table.loc[0, "dp"], ds_table

    # Instanciate datasets and define the merger path, where the 'merged' data will be saved,
    # together with the dataset table, holding information about the datasets (indices, names, source paths etc)
    ds_names = ""
    predirname = op.dirname(ds_table["dp"][0])
    for ds_i in ds_table.index:
        dp, prb = ds_table.loc[ds_i, "dp"], ds_table.loc[ds_i, "probe"]
        ds_names += "_" + ds_table.loc[ds_i, "dataset_name"]
        if predirname != op.dirname(dp):
            print(
                "WARNING: all your datasets are not stored in the same pre-directory - {} is picked anyway.".format(
                    predirname
                )
            )
    dp_merged = Path(predirname) / (
        "merged" + ds_names
    )  # Path(predirname, 'merger'+ds_names)
    old_prophyler_path = Path(predirname) / ("prophyler" + ds_names)
    if old_prophyler_path.exists():
        print(
            "Backward compatibility - prophyler renamed according to merger convention."
        )
        os.rename(str(old_prophyler_path), str(dp_merged))
    if not op.isdir(dp_merged):
        print(
            f"\n{mess_prefix}Merged data (from {n_datasets} datasets) will be saved here: {dp_merged}.{mess_suffix}"
        )
        os.mkdir(dp_merged)
    else:
        print(f"\n{mess_prefix}Merged dataset found at {dp_merged}.{mess_suffix}")
        ds_table_old = pd.read_csv(
            Path(dp_merged, "datasets_table.csv"), index_col="dataset_i"
        )
        old_dataset_names = list(ds_table_old.loc[:, "dataset_name"].values)
        for ds_i in ds_table.index:
            assert (
                ds_table.loc[ds_i, "dataset_name"] in old_dataset_names
            ), f"""WARNING you ran dataset merger on these {n_datasets} datasets in the past
                but used the dataset names {old_dataset_names}!!"""

    ds_table.to_csv(Path(dp_merged, "datasets_table.csv"))

    # Load and save units qualities
    # + check whether datasets have been re-spike sorted if not first instanciation
    qualities = load_merged_units_qualities(dp_merged, ds_table)
    re_spikesorted = detect_new_spikesorting(dp_merged, True, qualities)
    save_qualities(dp_merged, qualities)

    # Merge spike times (or copy them if only 1 dataset)
    # Only if merged_clusters_times does not exist already or does but re-spikesorting has been detected
    merge_fname_times = "spike_times"
    merge_fname_clusters = "spike_clusters"
    if (
        (not op.exists(Path(dp_merged, merge_fname_times + ".npy")))
        or re_spikesorted
        or again
    ):
        print(
            f"\n{mess_prefix}Loading spike trains of {n_datasets} datasets...{mess_suffix}"
        )
        # precompute all sync channels without prompting the user
        onsets = [
            get_npix_sync(dp, output_binary=False, filt_key="highpass", unit="samples")[
                0
            ]
            for dp in ds_table["dp"]
        ]
        spike_times, spike_clusters, sync_signals = [], [], []
        for ds_i, dp in enumerate(ds_table["dp"]):
            spike_times.append(np.load(Path(dp, "spike_times.npy")).flatten())
            spike_clusters.append(np.load(Path(dp, "spike_clusters.npy")).flatten())
            ons = onsets[ds_i]
            syncchan = ask_syncchan(ons)
            if syncchan == "q":
                print("Aborted. Returning Nones.")
                return None, None
            sync_signals.append(ons[syncchan])
        assert all(
            len(x) == len(sync_signals[0]) for x in sync_signals
        ), "WARNING different number of events on sync channels of both probes! Try again."
        NspikesTotal = 0
        for i in range(len(spike_times)):
            NspikesTotal += len(spike_times[i])

        # If several datasets are fed to the merger, align their spike times.
        # Merged dataset is saved as two arrays:
        # spike_times and spike_clusters, as in a regular dataset,
        # but where spike_clusters is a larger int (1 is unit 1 from dataset 0, 1,000,000,001 is unit 1 from dataset 1)
        print(
            f"\n{mess_prefix}Aligning spike trains of {n_datasets} datasets (w/r to 1st dataset)...{mess_suffix}"
        )
        spike_times = align_timeseries_interpol(spike_times, sync_signals, 30000)
        # Now save the new array
        merged_spike_clusters = npa(zeros=(NspikesTotal), dtype=np.float64)
        merged_spike_times = npa(zeros=(NspikesTotal), dtype=np.uint64)
        cum_Nspikes = 0
        for ds_i in ds_table.index:
            Nspikes = len(spike_times[ds_i])
            merged_spike_clusters[cum_Nspikes : cum_Nspikes + Nspikes] = (
                spike_clusters[ds_i] + 1e-1 * ds_i
            )
            merged_spike_times[cum_Nspikes : cum_Nspikes + Nspikes] = spike_times[ds_i]
            cum_Nspikes += Nspikes
        merged_spike_clusters = merged_spike_clusters[np.argsort(merged_spike_times)]
        merged_spike_times = np.sort(merged_spike_times)
        np.save(dp_merged / (merge_fname_clusters + ".npy"), merged_spike_clusters)
        np.save(dp_merged / (merge_fname_times + ".npy"), merged_spike_times)
        sync_dir = dp_merged / "sync_chan"
        sync_dir.mkdir(exist_ok=True)
        sync_file = (
            sync_dir
            / f"merged_ref_{ds_table.loc[0, 'dataset_name']}.ap_sync_on_samples.npy"
        )
        np.save(sync_file, sync_signals[0])
        print(
            f"\n{mess_prefix}Merged {merge_fname_times} and {merge_fname_clusters} saved at {dp_merged}.{mess_suffix}"
        )

        success_message = "\n--> Merge successful! Use a float u.x in any npyx function to call unit u from dataset x:"
        for ds_i in ds_table.index:
            success_message += (
                f"\n- u.{ds_i} for dataset {ds_table.loc[ds_i,'dataset_name']},"
            )
        success_message = "\033[92;1m" + success_message[:-1] + ".\033[0m"
        print(success_message)

    return dp_merged, ds_table


def ask_syncchan(ons):
    chan_len = "".join([f"chan {k} ({len(v)} events)." for k, v in ons.items()])
    print(f"Data found on sync channels:\n{chan_len}")
    if len(ons) == 1:
        print("Only one sync channel with data -> using it to synchronize probes.\n")
        return list(ons.keys())[0]
    syncchan = None
    while syncchan is None:
        syncchan = input(
            "Which channel shall be used to synchronize probes? (q to restart and pick other channel on previous probe) >>> "
        )
        try:
            syncchan = int(syncchan)
            if syncchan not in ons.keys():
                print(f"!! You need to feed an integer amongst {list(ons.keys())}!")
                syncchan = None
        except:
            if syncchan == "q":
                print("Aborting...")
                return syncchan
            print("!! You need to feed an integer!")
            syncchan = None
        if syncchan is not None:
            if not any(ons[syncchan]):
                print(
                    "!! This sync channel does not have any events! Pick another one!"
                )
                syncchan = None
    return syncchan


def get_ds_table(dp):
    assert assert_multi(dp), "Cannot be ran on a non merged dataset."
    dp = Path(dp)
    ds_table = pd.read_csv(dp / "datasets_table.csv", index_col="dataset_i")
    for dp in ds_table["dp"]:
        assert op.exists(
            dp
        ), f"WARNING you have instanciated this merged dataset from paths of which at least one doesn't exist anymore:{dp}!"
    return ds_table


def get_dataset_id(u):
    """
    Arguments:
        - u: float, of format u.ds_i
    Returns:
        - u: int, unit index
        - ds_i: int, dataset index
    """
    assert assert_float(
        u
    ), "Seems like the unit passed isn't a float - \
        calling this function on a merged dataset is meaningless \
        (cannot tell which dataset the unit belongs to!)."
    return int(round(u % 1 * 10)), int(u)


def assert_same_dataset(U):
    """Asserts if all provided units belong to the same dataset."""
    return all(get_dataset_id(U[0])[0] == get_dataset_id(u)[0] for u in U[1:])


def assert_multi(dp):
    return "merged" in op.basename(dp)


def get_ds_ids(U):
    return (U % 1 * 10).round(0).astype(np.int64)


def get_dataset_ids(dp_pro):
    """
    Arguments:
        - dp_pro: str, path to merged dataset
    Returns:
        - dataset_ids: np array of shape (N_spikes,), indices of dataset of origin for all spikes
    """
    assert assert_multi(dp_pro)
    return get_ds_ids(get_units(dp_pro))


def get_source_dp_u(dp, u):
    """If dp is a merged datapath, returns datapath from source dataset and unit as integer.
    Else, returns dp and u as they are.
    """
    if assert_multi(dp):
        ds_i, u = get_dataset_id(u)
        ds_table = get_ds_table(dp)
        dp = ds_table["dp"][ds_i]

    dp = Path(dp)
    return dp, u


def merge_units_across_ss():
    """
    Merge units across spike sortings.

    Only works across spike sortings of the same dataset.

    Requires to manually provide pairs of units from the 1st and second spike sorting.
    """
    return