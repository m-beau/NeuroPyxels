# -*- coding: utf-8 -*-
"""
2018-07-20

@author: Maxime Beau, Neural Computations Lab, University College London

Dataset: Neuropixels dataset -> dp is phy directory (kilosort or spyking circus output)
"""
import json
import os
import os.path as op
from pathlib import Path

import numpy as np
import pandas as pd

def get_npyx_memory(dp):
    dprm = Path(dp,'npyxMemory')
    old_dprm =Path(dp,'routinesMemory')
    if old_dprm.exists():
        print("Backward compatibility - renaming routinesMemory as npyxMemory.")
        os.rename(str(old_dprm), str(dprm))
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
    A typical example is 'interesting_unit_type':[unit1, unit2],
    but it could also be 'comment_on_behaviour':'great for 100 trials then got tired'.

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
        DSs = json.load(f)

    with open(ds_paths_master) as f:
        dp_dic = json.load(f)
        for ds_name, dp in dp_dic.items():
            dp=Path(dp)/f'{ds_name}'
            DSs[ds_name]["dp"]=str(dp)
            assert dp.exists(),\
                f"""WARNING path {dp} does not seem to exist!
                Edit path of {ds_name}:path in {ds_paths_master}."""
            for prb in DSs[ds_name].keys():
                if 'probe' in prb:
                    dp_prb=dp/f'{ds_name}_{prb}'
                    assert dp.exists(),\
                        f"""WARNING path {dp_prb} does not seem to exist!
                        Edit path of {ds_name}:path in {ds_paths_master}
                        and check that all probes are in subdirectories."""
                    DSs[ds_name][prb]["dp"]=str(dp_prb)

    for ds_name, ds in DSs.items():
        assert "dp" in ds.keys(), \
            f"""WARNING dataset {ds_name} does not have a path!
            Add it as a key:value pair in {ds_paths_master}."""

    # Add behavioural parameters to the dict for relevant datasets,
    # if any
    if ds_behav_master is not None:
        with open(ds_behav_master) as f:
            behav_params_dic = json.load(f)
            for ds_name, params in behav_params_dic.items():
                DSs[ds_name]["behav_params"]=params

    return DSs

def get_rec_len(dp, unit='seconds'):
    assert unit in ['samples', 'seconds', 'milliseconds']
    fs=read_spikeglx_meta(dp)['sRateHz']
    t_end=np.load(Path(dp,'spike_times.npy')).ravel()[-1]
    if unit in ['seconds', 'milliseconds']:
        t_end/=fs
        if unit=='milliseconds':t_end*=1e3
    return t_end

def regenerate_cluster_groups(dp):
    units=np.unique(np.load(Path(dp,"spike_clusters.npy")))
    qualities=pd.DataFrame({'cluster_id':units, 'group':['unsorted']*len(units)})
    return qualities

def load_units_qualities(dp, again=False):
    f='cluster_group.tsv'
    regenerate=False if not again else True
    if Path(dp, f).exists():
        qualities = pd.read_csv(Path(dp, f),delimiter='\t')
        if 'group' not in qualities.columns:
            print('WARNING there does not seem to be any group column in cluster_group.tsv - kilosort >2 weirdness. Making a fresh file.')
            regenerate=True
        else:
            if 'unsorted' not in qualities['group'].values:
                regenerate=True
        if regenerate:
            qualities_new=regenerate_cluster_groups(dp)
            missing_clusters_m=(~np.isin(qualities_new['cluster_id'], qualities['cluster_id']))
            qualities_missing=qualities_new.loc[missing_clusters_m,:]
            qualities=qualities.append(qualities_missing).sort_values('cluster_id')
            qualities.to_csv(Path(dp, 'cluster_group.tsv'), sep='\t', index=False)

    else:
        print('cluster groups table not found in provided data path. Generated from spike_clusters.npy.')
        qualities=regenerate_cluster_groups(dp)
        qualities.to_csv(Path(dp, 'cluster_group.tsv'), sep='\t', index=False)

    return qualities

def get_units(dp, quality='all', chan_range=None, again=False):
    assert quality in ['all', 'good', 'mua', 'noise']

    cl_grp = load_units_qualities(dp, again=again)

    units=[]
    if assert_multi(dp): get_ds_table(dp)
    units=cl_grp.loc[cl_grp['group']==quality,'cluster_id'].values if quality!='all' else cl_grp['cluster_id'].values

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
from npyx.io import read_spikeglx_meta
from npyx.spk_wvf import get_depthSort_peakChans
from npyx.merger import assert_multi, get_ds_table
