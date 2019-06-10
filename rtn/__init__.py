# -*- coding: utf-8 -*-

import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from six import integer_types


#%% Colors dictionnaries

phyColorsDic = {
    0:(53./255, 127./255, 255./255),
    1:(255./255, 0./255, 0./255),
    2:(255./255,215./255,0./255),
    3:(238./255, 53./255, 255./255),
    4:(84./255, 255./255, 28./255),
    5:(255./255,165./255,0./255),
    -1:(0., 0., 0.),
    }

seabornColorsDic = {
    0:sns.color_palette()[0],
    1:sns.color_palette()[1],
    2:sns.color_palette()[2],
    3:sns.color_palette()[3],
    4:sns.color_palette()[4],
    5:sns.color_palette()[5],
    6:sns.color_palette()[6],
    7:sns.color_palette()[7],
    8:sns.color_palette()[8],
    9:sns.color_palette()[9]
    }

DistinctColors20 = [(127,127,127),(0,0,143),(182,0,0),(0,140,0),(195,79,255),(1,165,202),(236,157,0),(118,255,0),(255,127,0),
    (255,117,152),(148,0,115),(0,243,204),(72,83,255),(0,127,255),(0,67,1),(237,183,255),(138,104,0),(97,0,163),(92,0,17),(255,245,133)]
DistinctColors20 = [(c[0]/255, c[1]/255, c[2]/255) for c in DistinctColors20]
DistinctColors15 = [(127,127,127),(255,255,0),(0,0,143),(255,0,0),(50,255,255),(255,0,255),(94,0,33),(0,67,0),
    (255,218,248),(0,178,0),(124,72,255),(211,145,0),(5,171,253),(126,73,0),(147,0,153)]
DistinctColors15 = [(c[0]/255, c[1]/255, c[2]/255) for c in DistinctColors15]

mark_dict = {
".":"point",
",":"pixel",
"o":"circle",
"v":"triangle_down",
"^":"triangle_up",
"<":"triangle_left",
">":"triangle_right",
"1":"tri_down",
"2":"tri_up",
"3":"tri_left",
"4":"tri_right",
"8":"octagon",
"s":"square",
"p":"pentagon",
"*":"star",
"h":"hexagon1",
"H":"hexagon2",
"+":"plus",
"D":"diamond",
"d":"thin_diamond",
"|":"vline",
"_":"hline"
}
                          

#%% Utils

_ACCEPTED_ARRAY_DTYPES = (np.float, np.float32, np.float64,
                          np.int, np.int8, np.int16, np.uint8, np.uint16,
                          np.int32, np.int64, np.uint32, np.uint64,
                          np.bool)

def npa(arr=[0], **kwargs):
    '''Returns np.array(param).
    Optional aprams:
        - zeros: tuple. If provided, returns np.zeros(zeros)
        - ones: tuple. If provided, returns np.ones(ones)
        - empty: tuple. If provided, returns np.empty(empty)'''
    if 'zeros' in kwargs.keys():
        return np.zeros(kwargs['zeros'])
    elif 'ones' in kwargs.keys():
        return np.empty(kwargs['ones'])
    elif 'empty' in kwargs.keys():
        return np.empty(kwargs['empty'])
    else:
        if 'dtype' in kwargs.keys():
            return np.array(arr, dtype=kwargs['dtype'])
        else:
            return np.array(arr)

def sign(x):
    "Returns the sign of the input number (1 or -1). 1 for 0 or -0."
    return int(x*1./abs(x)) if x!=0 else 1

def minus_is_1(x):
    return abs(1-1*x)*1./2


def thresh(arr, th, sgn=1):
    '''Returns indices of the data points just following a directed threshold crossing.
    - data: numpy array.
    - th: threshold value.
    - sgn: direction of the threshold crossing, either positive or negative.'''
    arr=np.asarray(arr)
    assert arr.ndim==1
    sgn=sign(sgn) # Turns into eiter 1 or -1
    arr= (arr-th)*sgn+th # Flips the array around threshold if sgn==-1
    i=np.nonzero(arr>=th)[0]
    # If no value is above the threshold or all of them are already above the threshold
    if len(i)==0 or len(i)==len(arr): 
        return np.array([])
    return (i-1)[arr[np.clip(i-1, 0, len(arr)-1)]<th]+1

def _as_array(arr, dtype=None):
    """Convert an object to a numerical NumPy array.

    Avoid a copy if possible.

    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, integer_types + (float,)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out

def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    if x is None or len(x) == 0:
        return np.array([], dtype=np.int64)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]

def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.

    Implicitely assume that:

    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.

    This is not checked for performance reasons.

    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]
