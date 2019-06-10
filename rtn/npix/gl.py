# -*- coding: utf-8 -*-
"""
2018-07-20

@author: Maxime Beau, Neural Computations Lab, University College London

Dataset: Neuropixels dataset -> dp is phy directory (kilosort or spyking circus output)
"""
import os

import numpy as np
import pandas as pd

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
