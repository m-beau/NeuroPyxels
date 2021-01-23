# -*- coding: utf-8 -*-
"""
2018-07-20

@author: Maxime Beau, Neural Computations Lab, University College London

Dataset: Neuropixels dataset -> dp is phy directory (kilosort or spyking circus output)
"""
import os
import os.path as op
from pathlib import Path

from npyx.io import read_spikeglx_meta

import numpy as np
import pandas as pd

def get_rec_len(dp, unit='seconds'):
    assert unit in ['samples', 'seconds', 'milliseconds']
    fs=read_spikeglx_meta(dp)['sRateHz']
    t_end=np.load(Path(dp,'spike_times.npy'))[-1,0]
    if unit in ['seconds', 'milliseconds']:
        t_end/=fs
        if unit=='milliseconds':t_end*=1e3
    return t_end

def load_units_qualities(dp, again=False):
    f='cluster_group.tsv'
    if os.path.isfile(Path(dp, f)):
        qualities = pd.read_csv(Path(dp, f),delimiter='	')
    else:
        print('cluster groups table not found in provided data path. Generated from spike_clusters.npy.')
        units=np.unique(np.load(Path(dp,"spike_clusters.npy")))
        qualities=pd.DataFrame({'cluster_id':units, 'group':['unsorted']*len(units)})
        qualities.to_csv(Path(dp, 'cluster_group.tsv'), sep='	', index=False)
        return qualities
        
    if again: # file was found if this line is reached
        units=np.unique(np.load(Path(dp,"spike_clusters.npy")))
        new_unsorted_units=units[~np.isin(units, qualities['cluster_id'])]
        qualities=qualities.append(pd.DataFrame({'cluster_id':new_unsorted_units, 'group':['unsorted']*len(new_unsorted_units)}), ignore_index=True)
        qualities=qualities.sort_values('cluster_id')
        qualities.to_csv(Path(dp, 'cluster_group.tsv'), sep='	', index=False)
        
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

### Below, utilities for circuit prophyler
### (in particular used to merge simultaneously recorded datasets)

def assert_multi(dp):
    cl_grp = load_units_qualities(dp)
    U=cl_grp['cluster_id'].values
    return not np.all(U==U.astype(int))

def get_ds_table(dp_pro):
    ds_table = pd.read_csv(Path(dp_pro, 'datasets_table.csv'), index_col='dataset_i')
    for dp in ds_table['dp']:
        assert op.exists(dp), f"WARNING you have instanciated this prophyler merged dataset from paths of which one doesn't exist anymore:{dp}!"
    return ds_table

def get_dataset_id(u):
    '''
    Parameters:
        - u: float, of format u.ds_i
    Returns:
        - u: int, unit index
        - ds_i: int, dataset index
    '''
    return int(round(u%1*10)), int(u)

def get_dataset_ids(dp_pro):
    '''
    Parameters:
        - dp_pro: str, path to prophyler dataset
    Returns:
        - dataset_ids: np array of shape (N_spikes,), indices of dataset of origin for all spikes
    '''
    assert assert_multi(dp_pro)
    return (get_units(dp_pro)%1*10).round(0).astype(int)

def get_prophyler_source(dp_pro, u):
    '''If dp is a prophyler datapath, returns datapath from source dataset and unit as integer.
       Else, returns dp and u as they are.
    '''
    if op.basename(dp_pro)[:9]=='prophyler':
        ds_i, u = get_dataset_id(u)
        ds_table=get_ds_table(dp_pro)
        dp=ds_table['dp'][ds_i]
    return dp, u

from npyx.spk_wvf import get_depthSort_peakChans
