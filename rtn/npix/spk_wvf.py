# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os, sys

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of

def get_channels(dp, units=None):
    if ~np.any(units):
        units=get_units()
    if os.path.isfile(dp+'/FeaturesTable/FeaturesTable_good.csv'):
        ft = pd.read_csv(dp+'/FeaturesTable/FeaturesTable_good.csv', sep=',', index_col=0)
        bestChs=npa(ft["WVF-MainChannel"])
        depthIdx = np.argsort(bestChs)[::-1] # From surface (high ch) to DCN (low ch)
        table_units=npa(ft.index, dtype=np.int64)[depthIdx]
        table_channels = bestChs[depthIdx]
    else:
        print('You need to export the features tables using phy first!!')
        return
    units = table_units[np.isin(table_units, units)]
    channels = table_channels[np.isin(table_units, units)]
    
    return units, channels

def get_chDis(dp, ch1, ch2):
    '''dp: datapath to dataset
    ch1, ch2: channel indices (1 to 384)
    
    returns distance in um.'''
    assert 1<=ch1<=384
    assert 1<=ch2<=384
    ch_pos = np.load(dp+'/channel_positions.npy') # positions in um
    ch_pos1=ch_pos[ch1-1] # convert from ch number to ch relative index
    ch_pos2=ch_pos[ch2-1] # convert from ch number to ch relative index
    chDis=np.sqrt((ch_pos1[0]-ch_pos2[0])**2+(ch_pos1[1]-ch_pos2[1])**2)
    return chDis
    
    
