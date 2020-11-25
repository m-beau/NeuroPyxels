# -*- coding: utf-8 -*-
"""
2018-07-20

@author: Maxime Beau, Neural Computations Lab, University College London

Dataset: Neuropixels dataset -> dp is phy directory (kilosort or spyking circus output)
"""
import os
import os.path as op
from pathlib import Path

from npyx.utils import npa
from npyx.io import read_spikeglx_meta

from ast import literal_eval as ale

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

def assert_multidatasets(dp):
    'Returns unpacked merged_clusters_spikes.npz if it exists in dp, None otherwise.'
    if op.exists(Path(dp, 'merged_clusters_spikes.npz')):
        mcs=np.load(Path(dp, 'merged_clusters_spikes.npz'))
        return mcs[list(mcs.keys())[0]]

def load_units_qualities(dp, again=False):
    f='cluster_group.tsv'
    if os.path.isfile(Path(dp, f)):
        qualities = pd.read_csv(Path(dp, f),delimiter='	')
    elif os.path.isfile(Path(dp, 'merged_'+f)):
        qualities = pd.read_csv(Path(dp, 'merged_'+f), delimiter='	', index_col='dataset_i')
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
    if cl_grp.index.name=='dataset_i':
        if quality=='all':
            for ds_i in cl_grp.index.unique():
                ds_table=pd.read_csv(Path(dp, 'datasets_table.csv'), index_col='dataset_i')
                ds_dp=ds_table['dp'][ds_i]
                assert op.exists(ds_dp), """WARNING you have instanciated this prophyler merged dataset from paths of which one doesn't exist anymore:{}!n\
                Please add the new path of dataset {} in the csv file {}.""".format(ds_dp, ds_table['dataset_name'][ds_i], Path(dp, 'datasets_table.csv'))
                ds_units=np.unique(np.load(Path(ds_dp, 'spike_clusters.npy')))
                units += ['{}_{}'.format(ds_i, u) for u in ds_units]
        else:
            for ds_i in cl_grp.index.unique():
                # np.all(cl_grp.loc[ds_i, 'group'][cl_grp.loc[ds_i, 'cluster_id']==u]==quality)
                units += ['{}_{}'.format(ds_i, u) for u in cl_grp.loc[(cl_grp['group']==quality)&(cl_grp.index==ds_i), 'cluster_id']]
        
    else:
        try:
            np.all(np.isnan(cl_grp['group'])) # Units have not been given a class yet
            units=[]
        except:
            if quality=='all':
                units = cl_grp.loc[:, 'cluster_id'].values.astype(np.int64)
                if 'unsorted' not in cl_grp['group'].unique():
                    units1 = cl_grp.loc[:, 'cluster_id'].astype(np.int64)
                    units=np.unique(np.load(Path(dp,"spike_clusters.npy")))
                    unsort_u=units[~np.isin(units, units1)]
                    unsort_df=pd.DataFrame({'cluster_id':unsort_u, 'group':['unsorted']*len(unsort_u)}) 
                    cl_grp=cl_grp.append(unsort_df, ignore_index=True)
                    cl_grp.to_csv(Path(dp, 'cluster_group.tsv'), sep='	', index=False)
            else:
                raise ValueError(f'you cannot try to load {quality} units before manually curating a dataset - run phy once and try again.')
        
    if chan_range is None:
        return units
    
    assert len(chan_range)==2, 'chan_range should be a list or array with 2 elements!'
    
    peak_channels=get_depthSort_peakChans(dp, units=[], quality=quality)
    chan_mask=(peak_channels[:,1]>=chan_range[0])&(peak_channels[:,1]<=chan_range[1])
    units=peak_channels[chan_mask,0].flatten()
    
    return units

def get_good_units(dp):
    return get_units(dp, quality='good')

def get_prophyler_source(dp_pro, u):
    '''If dp is a prophyler datapath, returns datapath from source dataset and unit as integer.
       Else, returns dp and u as they are.
    '''
    if op.basename(dp_pro)[:9]=='prophyler':
        ds_i, u = u.split('_'); ds_i, u = ale(ds_i), ale(u)
        ds_table=pd.read_csv(Path(dp_pro, 'datasets_table.csv'), index_col='dataset_i')
        ds_dp=ds_table['dp'][ds_i]
        assert op.exists(ds_dp), """WARNING you have instanciated this prophyler merged dataset from paths of which one doesn't exist anymore:{}!n\
        Please add the new path of dataset {} in the csv file {}.""".format(ds_dp, ds_table['dataset_name'][ds_i], Path(dp_pro, 'datasets_table.csv'))
        dp_pro=ds_dp
    return dp_pro, u

from npyx.spk_wvf import get_depthSort_peakChans
