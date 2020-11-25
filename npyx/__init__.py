# -*- coding: utf-8 -*-

from . import utils, io, gl, spk_t, spk_wvf, corr, plot, ml, behav, circuitProphyler, stats

import os, sys

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mpl_colors, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, zscore, \
                    _as_array, _unique, _index_of

__doc__="""

npyx submodules:
    npyx.utils

    npyx.io
    
    npyx.gl
    
    npyx.spk_t
    
    npyx.spk_wvf
    
    npyx.corr
    
    npyx.plot
    
    npyx.ml
    
    npyx.behav

    npyx.stats
"""