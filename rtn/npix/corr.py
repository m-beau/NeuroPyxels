# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""

from ..utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of
from .gl import get_good_units
from .spk_t import trn, trnb, isi, binarize

import scipy.signal as sgnl
from statsmodels.nonparametric.smoothers_lowess import lowess
