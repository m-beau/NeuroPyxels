# -*- coding: utf-8 -*-
"""
2018-07-20

@author: Maxime Beau, Neural Computations Lab, University College London

Dataset: Neuropixels dataset -> dp is phy directory (kilosort or spyking circus output)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from npyx.utils import npa


def get_npyx_memory(dp):
    if dp is None:
        return ""
    dpnm = Path(dp) / "npyxMemory"
    old_dpnm = Path(dp) / "routinesMemory"
    if old_dpnm.exists() and not dpnm.exists():
        try:  # because of parallel proccessing, might have been renamed in the process!
            os.rename(str(old_dpnm), str(dpnm))
            print("Backward compatibility - renaming routinesMemory as npyxMemory.")
        except:
            assert dpnm.exists()
    os.makedirs(dpnm, exist_ok=True)

    return dpnm


def get_datasets(ds_master, ds_paths_master, ds_behav_master=None, warnings=True):
    """
    Function to load dictionnary of dataset paths and relevant units.

    Below is a high level description of the 3 json file types used to generate your datasets dictionnary,
    you can find a more detailed structure description in the Parameters section a bit further.

    ------------------------------------------------------------------------------

    **ds_master** contains information about the probe structure of your recording
    as well as anything you wish to have readily accessible in your scripts
    as key:value pairs.

    You NEED to add 'probeX':{key:value, key:value...} to each dataset
    in order to generate the right path blah/datasetY_probeX.

    A typical example is
    'datasetY':{'probeX':{'interesting_unit_type':[unit1, unit2]}},
    but it could also be
    'datasetY':{'probeX':{'comment_on_behaviour':'great for 100 trials then got tired'}}.

    In **ds_paths_master**, the paths must be following this structure:
    pathToDataset1Root/datasetName/datasetName_probe1.
    Every dataset in ds_master must also be in **ds_paths_master**
    and vice versa (every dataset must have a path!).

    **ds_behav_master** contains dictionnaries of anything, allowing you to flexibly
    add any data you would like to have access to while doing your analysis.
    For instance, {'running_wheel_diameter':20, 'ntrials':'143', 'mouse nickname':'mickey'}.
    Every dataset in ds_behav_master must also be in ds_master ;
    the inverse is not true (not every dataset has behavioural parameters).

    ------------------------------------------------------------------------------

    Arguments:

        - ds_master: str, path to json file with following structure:
            {
            'dataset1_name':{
                'probe1':{
                    'key1':[unit1, unit2, unit3],
                    'key2':'some_string',
                    'key3':{some dict},
                    ...
                    }
                'probe2': ... # eventually
                }
            'dataset2_name': ... # eventually
            }

        - ds_paths_master str, path to json file with following structure:
            {
            'dataset1_name':'path_to_dataset1_root',
            'dataset2_name':'path_to_dataset2_root',
            ...
            }

        - ds_behav_master str, path to json file with following structure:
            {
            'dataset1_name':{behaviour parameters, any keys/values},
            'dataset2_name':{behaviour parameters, any keys/values},
            ...
            }

    ------------------------------------------------------------------------------

    Returns:

        - DSs: dictionnary of datasets with the following structure:
            {
            'dataset1_name':{}
                'dp':'path_to_dataset1_root_from_ds_paths_master/dataset1_name',

                'behav_params':{behaviour parameters from ds_behav_master},

                'probe1':{
                    'dp':'path/to/probe/subdirectory' # contains recording and kilosort files of probe1
                    'key1':[unit1, unit2, unit3],
                    'key2':'some_string',
                    'key3':{some dict},...
                    }
                'probe2': ... # eventually
                }
            'dataset2_name': ... # eventually
            }
    """
    with open(ds_master) as f:
        print(f"\033[34;1m--- ds_master\033[0m read from {ds_master}.")
        DSs = json.load(f)

    with open(ds_paths_master) as f:
        print(f"\033[34;1m--- ds_paths_master\033[0m read from {ds_paths_master}.")
        dp_dic = json.load(f)
        for ds_name, dp in dp_dic.items():
            dp = Path(dp) / f"{ds_name}"
            if not dp.exists():
                if warnings:
                    print(
                        f"\n\033[31;1mWARNING path {dp} does not seem to exist. Edit path of {ds_name}:path in \033[34;1mds_paths_master\033[31;1m."
                    )
                continue
            assert ds_name in DSs.keys(), print(
                f"\n\033[31;1mWARNING dataset {ds_name} from \033[34;1mds_paths_master\033[31;1m isn't referenced in \033[34;1mds_master\033[31;1m."
            )
            DSs[ds_name]["dp"] = dp
            for prb in DSs[ds_name].keys():
                if "probe" in prb:
                    dp_prb = dp / f"{ds_name}_{prb}"
                    if not dp_prb.exists():
                        if warnings:
                            print(
                                (
                                    f"\n\033[31;1mWARNING path {dp_prb} does not seem to exist! "
                                    f"Edit path of {ds_name}:path in \033[34;1mds_paths_master\033[31;1m "
                                    "and check that all probes are in subdirectories."
                                )
                            )
                        continue
                    DSs[ds_name][prb]["dp"] = dp_prb

    for ds_name, ds in DSs.items():
        if "dp" not in ds.keys():
            if warnings:
                print(
                    f"\n\033[31;1mWARNING dataset {ds_name} does not have a path! Add it as a key:value pair in \033[34;1mds_paths_master\033[31;1m."
                )

    # Add behavioural parameters to the dict for relevant datasets,
    # if any
    if ds_behav_master is not None:
        with open(ds_behav_master) as f:
            behav_params_dic = json.load(f)
            for ds_name, params in behav_params_dic.items():
                DSs[ds_name]["behav_params"] = params

    return DSs


def json_connected_pairs_df(ds_master, ds_paths_master, ds_behav_master):

    DSs = get_datasets(ds_master, ds_paths_master, ds_behav_master, warnings=False)
    
    pair_dicts = []
    for dsname, ds in DSs.items():
        for dsk, dsv in ds.items():
            if "probe" in dsk:
                probei = 0  # if merged dataset: int(dsk[-1])-1
                if "ss_cnc_real" in dsv.keys():
                    if np.any(dsv["ss_cnc_real"]):
                        dp_dic = {probe: str(ds[probe]["dp"]) for probe in ds.keys() if "probe" in probe}
                        dp, ds_table = merge_datasets(dp_dic)

                        putative_pairs = [str(pair) for pair in dsv["ss_cnc_put"]]
                        assert len(putative_pairs) > 0, f"No connected pairs found in dataset {dsname} at {ds_master}!"
                        real_pairs = [str(pair) for pair in dsv["ss_cnc_real"]]
                        if not np.all(np.isin(real_pairs, putative_pairs)):
                            print(
                                (
                                    "Some real connected pairs are not in putative connected pairs: "
                                    f"{np.array(real_pairs)[~np.isin(real_pairs, putative_pairs)]}.\n"
                                    f"Edit json file at {ds_master}!"
                                )
                            )

                        for ss, cnc in dsv["ss_cnc_put"]:
                            cs = np.nan
                            if np.any(dsv["ss_cs"]):
                                ss_cs = npa(dsv["ss_cs"])
                                if ss in ss_cs[:, 0]:
                                    cs = ss_cs[np.isin(ss_cs[:, 0], [ss]), 1][0]

                            if str([ss, cnc]) in real_pairs:
                                holds_no_sync = True
                            else:
                                holds_no_sync = False

                            holds_no_behav = True  ## TODO
                            pair_dicts.append({
                                        "ds": dsname,
                                        "dp": dp,
                                        "ss": ss + 0.1 * probei,
                                        "cs": cs + 0.1 * probei,
                                        "nc": cnc + 0.1 * probei,
                                        "holds_no_behav": holds_no_behav,
                                        "holds_no_sync": holds_no_sync,
                                    })

    connected_pairs_df = pd.DataFrame(pair_dicts)
    return connected_pairs_df


def make_connected_pairs_df(
    ds_master,
    ds_paths_master,
    ds_behav_master,
    upsample_sync=True,  # sync spikes are the synchronous spikes + random fraction of the asynchronous spikes to match Ns
    pval_th=0.05,
    sync_win=0.5,  # ms
    cisi_upper_threshold=0.1,  # s
    n_pcs_firing_fraction_threshold=0.5,  # [0-1]
    cbin_inh=0.5,
    cwin_inh=100,
    min_win=[0.5, 3],
    min_win_nbins=3,
    enforced_rp=0.3,  # ms
    W_sd=10,
):

    df = json_connected_pairs_df(ds_master, ds_paths_master, ds_behav_master)

    connected_pairs_df_fn = (
        f"sync_df"
        f"{pval_th}_{sync_win}_{cisi_upper_threshold}_{n_pcs_firing_fraction_threshold}"
        f"_{cbin_inh}_{cwin_inh}_{min_win}_{min_win_nbins}_{enforced_rp}_{W_sd}_{upsample_sync}"
    )

    DPs = np.unique(df["dp"])
    connected_pairs_df = None
    for dp in DPs:

        saveDir = Path(dp).parent / "popsync_analysis"
        assert saveDir.exists(), f"WARNING : {saveDir} does not exist. Run monosynapticity_exploration notebook first."
        df_ = pd.read_pickle(saveDir / f"{connected_pairs_df_fn}.pkl")

        df_["pval_thresh"] = df_["pcnc_inh_pval"] < pval_th
        df_ = df_.pivot(index=["pc", "nc"], columns="cisi_popsync")
        df_.columns = ["_".join(col).strip() for col in df_.columns.values]
        df_.reset_index(inplace=True)
        df_["connected"] = [
            {"TrueTrue": True, "TrueFalse": False, "FalseTrue": False, "FalseFalse": False}[f"{p1}{p2}"]
            for p1, p2 in zip(df_["pval_thresh_>0"], df_["pval_thresh_0"])
        ]
        df_["color"] = [
            {"TrueTrue": "red", "TrueFalse": "grey", "FalseTrue": "grey", "FalseFalse": "black"}[f"{p1}{p2}"]
            for p1, p2 in zip(df_["pval_thresh_>0"], df_["pval_thresh_0"])
        ]

        # add dp
        df_.insert(0, "dp", dp)

        if connected_pairs_df is None:
            connected_pairs_df = df_
        else:
            connected_pairs_df = pd.concat((connected_pairs_df, df_), axis=0)
    connected_pairs_df.reset_index(inplace=True, drop=True)
    connected_pairs_df.insert(2, "cs", np.nan)
    connected_pairs_df.rename(columns={"pc": "ss"}, inplace=True)

    for i in range(len(connected_pairs_df)):
        dp, ss, nc = connected_pairs_df.loc[i, ["dp", "ss", "nc"]]
        m = (df["dp"] == dp) & (df["ss"] == ss) & (df["nc"] == nc)
        if np.sum(m) != 1:
            continue
        cs = df.loc[m, "cs"].values[0]
        connected_pairs_df.loc[i, "cs"] = cs

    connected_pairs_df.to_pickle(Path(ds_master).parent / "connected_pairs_df.pkl")

    return connected_pairs_df


def get_rec_len(dp, unit="seconds"):
    "returns recording length in seconds or samples"
    assert unit in ["samples", "seconds"]
    meta = read_metadata(dp)
    if "recording_length_seconds" in meta.keys():
        rec_length = meta["recording_length_seconds"]
    else:
        v = list(meta.values())[0]
        rec_length = v["recording_length_seconds"]
    if unit == "samples":
        rec_length = int(rec_length * meta["highpass"]["sampling_rate"])
    return rec_length


def detect_new_spikesorting(dp, print_message=True, qualities=None):
    """
    Detects whether a dataset has been respikesorted
    based on spike_clusters.npy and cluster_group.tsv time stamps.
    Arguments:
        - dp: str, path to original sorted dataset (not a merged dataset)
        - print_message: whether to print a warning when new spikesorting is detected.
        - qualities: option to feed new qualities dataframe to compare to old one
                     (if different units are found, new spikesorting is detected)
    Returns:
        - spikesorted: bool, True or False if new spike sorting detected or not.
    """
    dp = Path(dp)
    spikesorted = False
    if not (dp / "cluster_group.tsv").exists():
        # if first time this is ran on a merged dataset
        return True

    if qualities is None:
        assert "merged" not in str(
            dp
        ), "this function should be ran on an original sorted dataset, not on a merged dataset."
        last_spikesort = os.path.getmtime(dp / "spike_clusters.npy")
        last_tsv_update = os.path.getmtime(dp / "cluster_group.tsv")
        # spikesorted = last_tsv_update == last_spikesort
        spikesorted = abs(last_tsv_update - last_spikesort) < 1.0 # do not edit out!!

    else:
        qualities_old = pd.read_csv(dp / "cluster_group.tsv", delim_whitespace=True)
        old_clusters = qualities_old.loc[:, "cluster_id"]
        new_clusters = qualities.loc[:, "cluster_id"]
        if not np.all(np.isin(old_clusters, new_clusters)):
            spikesorted = True

    if spikesorted and print_message:
        print("\n\033[34;1m--- New spike-sorting detected.\033[0m")

    return spikesorted


def save_qualities(dp, qualities):
    dp = Path(dp)
    qualities.to_csv(dp / "cluster_group.tsv", sep="\t", index=False)


def generate_units_qualities(dp):
    """
    Creates an empty table of units qualities ("groups" as in good, mua,...).
    """
    units = np.unique(np.load(Path(dp, "spike_clusters.npy")))
    qualities = pd.DataFrame({"cluster_id": units, "group": ["unsorted"] * len(units)})
    return qualities


def load_units_qualities(dp, again=False):
    """
    Load unit qualities (groups tsv table) from dataset.

    Arguments:
        - dp_merged: str, datapath to merged dataset
        - again: bool, whether to recompute from spike_clusters.npy (long)

    Returns:
        - qualities: panda dataframe, dataset units qualities
    """
    f = "cluster_group.tsv"
    dp = Path(dp)
    if (dp / f).exists():
        qualities = pd.read_csv(dp / f, delim_whitespace=True)
        re_spikesorted = detect_new_spikesorting(dp)
        regenerate = True if (again or re_spikesorted) else False
        assert (
            "cluster_id" in qualities.columns
        ), f"WARNING the tsv file {str(dp/f)} should have a column called 'cluster_id'!"
        if "group" not in qualities.columns:
            print(
                "WARNING there does not seem to be any group column in cluster_group.tsv - kilosort >2 weirdness. Making a fresh file."
            )
            qualities = generate_units_qualities(dp)
        else:
            if "unsorted" not in qualities["group"].values and re_spikesorted:
                # the file can only be 'not re_spikesorted' if it has been edited by npyx (or manually)
                # so in the rare case where all neurons are called 'mua' or 'good' or 'noise' with none left unsorted,
                # but npyx already regenerated the tsv file, we should NOT regenerate the file
                regenerate = True
        if regenerate:
            qualities_new = generate_units_qualities(dp)

            sorted_clusters = qualities.loc[qualities["group"] != "unsorted", :]
            unsorted_clusters_m = ~np.isin(qualities_new["cluster_id"], sorted_clusters["cluster_id"])
            unsorted_clusters = qualities_new.loc[unsorted_clusters_m, :]

            qualities = pd.concat((unsorted_clusters, sorted_clusters), axis=0)
            qualities = qualities.sort_values("cluster_id")
            save_qualities(dp, qualities)
    else:
        print("cluster groups table not found in provided data path. Generated from spike_clusters.npy.")
        qualities = generate_units_qualities(dp)
        save_qualities(dp, qualities)

    return qualities


def load_merged_units_qualities(dp_merged, ds_table=None):
    """
    Load unit qualities from merged dataset.
    Different from load_units_qualities, as it always recomputes and saves
    the units qualities tsv table from the source datasets
    to ensure handling of new spikesorting.

    Arguments:
        - dp_merged: str, datapath to merged dataset
        - ds_table: optional panda dataframe, datasets table
                    (if None, assumed to be dp/datasets_table.csv)

    Returns:
        - qualities: panda dataframe, merged dataset units qualities
    """

    dp_merged = Path(dp_merged)

    if ds_table is None:
        ds_table = get_ds_table(dp_merged)

    qualities = pd.DataFrame(columns=["cluster_id", "group"])
    for ds_i in ds_table.index:
        cl_grp = load_units_qualities(ds_table.loc[ds_i, "dp"])
        cl_grp["cluster_id"] = cl_grp["cluster_id"] + 1e-1 * ds_i
        qualities = qualities.append(cl_grp, ignore_index=True)

    return qualities


def get_units(dp, quality="all", chan_range=None, again=False):
    assert quality in ["all", "good", "mua", "noise", "unsorted"]

    if assert_multi(dp):
        qualities = load_merged_units_qualities(dp)
        try:
            re_spikesorted = detect_new_spikesorting(dp, qualities=qualities)
        except:
            re_spikesorted = False  # weird inability to load clutsr_group.tsv with pandas while using joblib

        assert not re_spikesorted, (
            f"It seems that a source dataset of {dp} has been re-spikesorted!! "
            "you need to run merge_datasets(dp_dic) again before being able to call get_units()."
        )
        save_qualities(dp, qualities)
    else:
        qualities = load_units_qualities(dp, again=again)

    if quality == "all":
        units = qualities["cluster_id"].values
    else:
        quality_m = qualities["group"] == quality
        units = qualities.loc[quality_m, "cluster_id"].values

    if chan_range is None:
        return units

    assert len(chan_range) == 2, "chan_range should be a list or array with 2 elements!"

    # For regular datasets
    peak_channels = get_depthSort_peakChans(dp, units=units, quality=quality)
    chan_mask = (peak_channels[:, 1] >= chan_range[0]) & (peak_channels[:, 1] <= chan_range[1])
    units = peak_channels[chan_mask, 0].flatten()

    return units


def get_good_units(dp):
    return get_units(dp, quality="good")


def check_periods(periods):
    err_mess = "periods can only be 'all' or a list of lists/tuples [[t1.1,t1.2], [t2.1,t2.2]...] in seconds!"
    if isinstance(periods, str):
        assert periods == "all", err_mess
        return periods
    periods = npa(periods)
    if periods.ndim == 1:
        periods = periods.reshape(1, -1)
    assert periods.ndim == 2, "When feeding a single period [t1,t2], do not forget the outer brackets [[t1,t2]]!"
    assert periods.shape[1] == 2, err_mess
    assert np.all(
        np.diff(periods, axis=1) > 0
    ), "all pairs of periods must be in ascendent order (t1.1<t1.2 etc in [[t1.1,t1.2],...])!"
    return periods


# circular imports
from npyx.inout import read_metadata
from npyx.merger import assert_multi, get_ds_table, merge_datasets
from npyx.spk_wvf import get_depthSort_peakChans
