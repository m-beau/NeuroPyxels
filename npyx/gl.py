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


def get_npyx_memory(dp):
    dprm = Path(dp,'npyxMemory')
    old_dprm =Path(dp,'routinesMemory')
    if old_dprm.exists():
        print("Backward compatibility - renaming routinesMemory as npyxMemory.")
        try: # because of parallel proccessing, might have been renamed in the process!
            os.rename(str(old_dprm), str(dprm))
        except:
            assert dprm.exists()
    if not os.path.isdir(dprm): os.makedirs(dprm)

    return dprm

def get_datasets(ds_master, ds_paths_master, ds_behav_master=None):
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

    Parameters:

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
            dp=Path(dp)/f'{ds_name}'
            if not dp.exists():
                print(\
                    f"\n\033[31;1mWARNING path {dp} does not seem to exist. Edit path of {ds_name}:path in \033[34;1mds_paths_master\033[31;1m.")
                continue
            assert ds_name in DSs.keys(),\
                print(f"\n\033[31;1mWARNING dataset {ds_name} from \033[34;1mds_paths_master\033[31;1m isn't referenced in \033[34;1mds_master\033[31;1m.")
            DSs[ds_name]["dp"]=str(dp)
            for prb in DSs[ds_name].keys():
                if 'probe' in prb:
                    dp_prb=dp/f'{ds_name}_{prb}'
                    if not dp_prb.exists():
                        print((f"\n\033[31;1mWARNING path {dp_prb} does not seem to exist! "
                        f"Edit path of {ds_name}:path in \033[34;1mds_paths_master\033[31;1m "
                        "and check that all probes are in subdirectories."))
                        continue
                    DSs[ds_name][prb]["dp"]=str(dp_prb)

    for ds_name, ds in DSs.items():
        if "dp" not in ds.keys():
            print(f"\n\033[31;1mWARNING dataset {ds_name} does not have a path! Add it as a key:value pair in \033[34;1mds_paths_master\033[31;1m.")

    # Add behavioural parameters to the dict for relevant datasets,
    # if any
    if ds_behav_master is not None:
        with open(ds_behav_master) as f:
            behav_params_dic = json.load(f)
            for ds_name, params in behav_params_dic.items():
                DSs[ds_name]["behav_params"]=params

    return DSs

def get_rec_len(dp, unit='seconds'):
    ' returns recording length in seconds or samples'
    assert unit in ['samples', 'seconds']
    meta = read_metadata(dp)
    rec_length=meta['recording_length_seconds']
    if unit=='samples':
        rec_length=int(rec_length*meta['sampling_rate'])
    return rec_length

def detect_new_spikesorting(dp):
    '''
    Detects whether a dataset has been respikesorted
    based on spike_clusters.npy and cluster_group.tsv time stamps.
    Parameters:
        - dp: str, path to original sorted dataset (not a merged dataset)
    Returns:
        - spikesorted: bool, True or False if new spike sorting detected or not.
    '''
    dp=Path(dp)
    assert 'merged' not in str(dp), 'this function should be ran on an original sorted dataset, not on a merged dataset.'
    last_spikesort = os.path.getmtime(dp/'spike_clusters.npy')
    last_tsv_update = os.path.getmtime(dp/'cluster_group.tsv')
    spikesorted = (last_tsv_update==last_spikesort)
    if spikesorted: print('\n\033[34;1m--- New spike-sorting detected.')
    return spikesorted

def generate_cluster_groups(dp):
    units=np.unique(np.load(Path(dp,"spike_clusters.npy")))
    qualities=pd.DataFrame({'cluster_id':units, 'group':['unsorted']*len(units)})
    return qualities

def load_units_qualities(dp, again=False):
    f='cluster_group.tsv'
    quality_dp = Path(dp, f)
    if quality_dp.exists():
        qualities = pd.read_csv(quality_dp,delimiter='\t')
        re_spikesorted = detect_new_spikesorting(dp)
        regenerate=True if (again or re_spikesorted) else False
        assert 'cluster_id' in qualities.columns,\
            f"WARNING the tsv file {quality_dp} should have a column called 'cluster_id'!"
        if 'group' not in qualities.columns:
            print('WARNING there does not seem to be any group column in cluster_group.tsv - kilosort >2 weirdness. Making a fresh file.')
            qualities=generate_cluster_groups(dp)
        else:
            if 'unsorted' not in qualities['group'].values:
                regenerate=True
        if regenerate:
            qualities_new=generate_cluster_groups(dp)

            sorted_clusters     = qualities.loc[qualities['group']!='unsorted', :]
            unsorted_clusters_m = ~np.isin(qualities_new['cluster_id'],sorted_clusters['cluster_id'])
            unsorted_clusters   = qualities_new.loc[unsorted_clusters_m,:]

            qualities=unsorted_clusters.append(sorted_clusters).sort_values('cluster_id')
            qualities.to_csv(quality_dp, sep='\t', index=False)
    else:
        print('cluster groups table not found in provided data path. Generated from spike_clusters.npy.')
        qualities=generate_cluster_groups(dp)
        qualities.to_csv(quality_dp, sep='\t', index=False)

    return qualities

def get_units(dp, quality='all', chan_range=None, again=False):

    assert quality in ['all', 'good', 'mua', 'noise']

    if assert_multi(dp):
        ds_table = get_ds_table(dp)
        qualities=pd.DataFrame(columns=['cluster_id', 'group'])
        for ds_i in ds_table.index:
            dpp=ds_table.loc[ds_i, 'dp']
            re_spikesorted = detect_new_spikesorting(dpp)
            qual = load_units_qualities(dpp, again=again)
            qual['cluster_id']=qual['cluster_id']+1e-1*ds_i
            qualities=qualities.append(qual, ignore_index=True)
        qualities_dp=Path(dp, 'cluster_group.tsv')
        qualities.to_csv(qualities_dp, sep='	', index=False)
        assert not re_spikesorted, 'WARNING new spike sorting in one of the source datasets of this merged dataset - you need to run merger again!'
    else:
        qualities = load_units_qualities(dp, again=again)

    units=qualities.loc[qualities['group']==quality,'cluster_id'].values if quality!='all' else qualities['cluster_id'].values

    if chan_range is None:
        return units

    assert len(chan_range)==2, 'chan_range should be a list or array with 2 elements!'

    # For regular datasets
    peak_channels=get_depthSort_peakChans(dp, units=units, quality=quality)
    chan_mask=(peak_channels[:,1]>=chan_range[0])&(peak_channels[:,1]<=chan_range[1])
    units=peak_channels[chan_mask,0].flatten()

    return units

def get_good_units(dp):
    return get_units(dp, quality='good')

# circular imports
from npyx.io import read_metadata
from npyx.spk_wvf import get_depthSort_peakChans
from npyx.merger import assert_multi, get_ds_table
