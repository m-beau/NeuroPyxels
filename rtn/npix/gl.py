# -*- coding: utf-8 -*-
"""
2018-07-20

@author: Maxime Beau, Neural Computations Lab, University College London

Dataset: Neuropixels dataset -> dp is phy directory (kilosort or spyking circus output)
"""
import os
import os.path as op

import numpy as np
import pandas as pd

from rtn.utils import npa

def chan_map(probe_version='3A', dp=None):
    assert probe_version in ['3A', '3B', '1.0', '2.0_singleshank', 'local']
    
    if probe_version in probe_version in ['3A', '3B', '1.0']:
        Nchan=384
        cm_el = npa([[  27,   0],
                           [  59,   0],
                           [  11,   20],
                           [  43,   20]])
        vert=npa([[  0,   40],
                  [  0,   40],
                  [  0,   40],
                  [  0,   40]])
        
        cm=cm_el.copy()
        for i in range(int(Nchan/cm_el.shape[0])-1):
            cm = np.vstack((cm, cm_el+vert*(i+1)))
        cm=np.hstack([np.arange(Nchan).reshape(Nchan,1), cm])
        
    elif probe_version=='2.0_singleshank':
        Nchan=384
        cm_el = npa([[  0,   0],
                           [  32,   0]])
        vert=npa([[  0,   15],
                  [  0,   15]])
        
        cm=cm.copy()
        for i in range(int(Nchan/cm_el.shape[0])-1):
            cm = np.vstack((cm, cm+vert*(i+1)))
        cm=np.hstack([np.arange(Nchan).reshape(Nchan,1), cm])
    
    elif probe_version=='local':
        if dp is None:
            raise ValueError("dp argument is not provided - when channel map is \
                             atypical and probe_version is hence called 'local', \
                             the datapath needs to be provided to load the channel map.")
        c_ind=np.load(op.join(dp, 'channel_map.npy'));cp=np.load(op.join(dp, 'channel_positions.npy'));
        cm=npa(np.hstack([c_ind, cp]), dtype=np.int32)
        
    return cm

def get_units(dp):
    f1=dp+'/cluster_group.tsv'
    f2=dp+'/cluster_groups.csv'
    if os.path.isfile(f1):
        cl_grp = pd.read_csv(f1,delimiter='	')
    elif os.path.isfile(f2):
        cl_grp = pd.read_csv(f2)
    else:
        print('cluster groups table not found in provided data path. Exiting.')
        return
    try:
        if np.all(np.isnan(cl_grp['group'])): # Units have not been given a class yet
            units=[]
        else:
            units = cl_grp.loc[:, 'cluster_id']
    except:
        units = cl_grp.loc[:, 'cluster_id']
    return np.array(units, dtype=np.int64)

def get_good_units(dp):
    f1=dp+'/cluster_group.tsv'
    f2=dp+'/cluster_groups.csv'
    if os.path.isfile(f1):
        cl_grp = pd.read_csv(f1,delimiter='	')
    elif os.path.isfile(f2):
        cl_grp = pd.read_csv(f2)
    else:
        print('cluster groups table not found in provided data path. Exiting.')
        return
    try:
        if np.all(np.isnan(cl_grp['group'])): # Units have not been given a class yet
            goodUnits=[]
        else:
            goodUnits = cl_grp.loc[np.nonzero(cl_grp['group']=='good')[0], 'cluster_id']
    except:
        goodUnits = cl_grp.loc[np.nonzero(cl_grp['group']=='good')[0], 'cluster_id']
    return np.array(goodUnits, dtype=np.int64)
