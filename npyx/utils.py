# -*- coding: utf-8 -*-

import os
from pathlib import Path
from tracemalloc import start

from numba import njit
from numba.typed import List
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("default", category=NumbaDeprecationWarning) #'ignore'
warnings.simplefilter('default', category=NumbaPendingDeprecationWarning)#'ignore'

from ast import literal_eval as ale
import numpy as np
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt

from six import integer_types
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy.stats as stt
import scipy.signal as sgnl

import logging
from math import pi, log
from scipy import fft, ifft
from scipy.optimize import curve_fit
from scipy.signal import cspline1d_eval, cspline1d

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

mpl_colors=plt.rcParams['axes.prop_cycle'].by_key()['color']

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

def assert_float(x):
    return isinstance(x, (float, np.float,
                          np.float16, np.float32, np.float64))

def assert_int(x):
    return isinstance(x, (int, np.int, np.int8, np.int16,
                          np.uint8, np.uint16, np.int32,
                          np.int64, np.uint32, np.uint64))

def assert_iterable(x):
    return hasattr(x, '__iter__')

def npa(arr=[], **kwargs):
    '''Returns np.array of some kind.
    Optional params:
        - zeros: tuple. If provided, returns np.zeros(zeros)
        - ones: tuple. If provided, returns np.ones(ones)
        - empty: tuple. If provided, returns np.empty(empty)
        - dtype: numpy datatype. If provided, returns np.array(arr, dtype=dtype) .'''

    dtype=kwargs['dtype'] if 'dtype' in kwargs.keys() else None
    if 'zeros' in kwargs.keys():
        arr = np.zeros(kwargs['zeros'], dtype=dtype)
    elif 'ones' in kwargs.keys():
        arr = np.ones(kwargs['ones'], dtype=dtype)
    elif 'empty' in kwargs.keys():
        arr = np.empty(kwargs['empty'], dtype=dtype)
    else:
        arr=np.array(arr, dtype=dtype)
    return arr

def isnumeric(x):
    x=str(x).replace('−','-')
    try:
        ale(x)
        return True
    except:
        return False

def sign(x):
    "Returns the sign of the input number (1 or -1). 1 for 0 or -0."
    x=npa(x)
    x[x==0]=1
    return (x/np.abs(x)).astype(np.int64)

def minus_is_1(x):
    return abs(1-1*x)*1./2

def read_pyfile(filepath, ignored_chars=[" ", "'", "\"", "\n", "\r"]):
    '''
    Reads .py file and returns contents as dictionnary.

    Assumes that file only has "variable=value" pairs, no fucntions etc

    - filepath: str, path to file
    - ignored_chars: list of characters to remove (only trailing and leading)
    '''
    filepath = Path(filepath)
    assert filepath.exists(), f'{filepath} not found!'

    params={}
    with open(filepath) as f:
        for ln in f.readlines():
            assert '=' in ln, 'WARNING read_pyfile only works for list of variable=value lines!'
            tmp = ln.split('=')
            for i, string in enumerate(tmp):
                string=string.strip("".join(ignored_chars))
                tmp[i]=string
            k, val = tmp[0], tmp[1]
            try: val = ale(val)
            except: pass
            params[k]=val

    return params

def list_files(directory, extension, full_path=False):
    directory=str(directory)
    files = [f for f in os.listdir(directory) if f.endswith('.' + extension)]
    files.sort()
    if full_path:
        return [Path('/'.join([directory,f])) for f in files]
    return files

def any_n_consec(X, n_consec, where=False):
    '''
    The trick below finds whether there are n_consec consecutive ones in the array comp
    by adding each element with its neighbour.
    At least 2 consec ones: add them 1 time and check if there is a 2=2**1 somewhere.
    At least 2 consec ones: add them 3 times and check if there is a 4=2**2 somewhere.
    ...
    Parameters:
        - X: array of booleans or binary
        - n_consec: int, number of consecutive True/False values to find
        - where: bool, returns array of indices if True
    '''
    X=npa(X).astype(np.int)
    assert np.all((X==0)|(X==1)), 'X array should be only 0s and 1s!'
    for i in range(n_consec-1):
        X = X[:-1]+X[1:]
    b=np.any(X==2**(n_consec-1))
    if not where:
        return b
    if b:
        i=np.nonzero(X==2**(n_consec-1))[0]
        i_edges1=i[np.append(0,np.diff(i))!=1]
        i_edges2=(i+n_consec-1)[np.append(np.diff(i), 0)!=1]
        return b, [np.arange(i_edges1[e],i_edges2[e]+1) for e in range(len(i_edges1))]
    return b, npa([])

def thresh_consec0(arr, th, n_consec, sgn=1, exclude_edges=True):
    '''
    SLOWER THAN thresh_consec AND FORCED TO PROVIDE N_CONSEC -> SHITTY
    Returns indices and values of threshold crosses lasting >=n_consec consecutive samples in arr.
    Parameters:
        - arr: numpy array to threshold
        - th: float, threshold
        - n_consec: int, minimum number of consecutive elements beyond threshold
        - sgn: int, positive (for cases above threshold) or negative (below threshold)
        - exclude_edges: bool, if true edges of arr are not considered as threshold crosses
                         in cases where arr starts or ends beyond threshold
    '''
    arr=npa(arr)
    if not arr.ndim==1:
        assert 1 in arr.shape
        arr=arr.flatten()

    arr= (arr-th)*sign(sgn)+th # Flips the array around threshold if sgn==-1
    comp = (arr>=th)

    crosses=any_n_consec(comp, n_consec, where=True)[1]

    if exclude_edges:
        if crosses[0][0]==0: # starts beyond threshold and lasts >=n_consec_bins
            crosses=crosses[1:]
        if crosses[-1][-1]==len(arr)-1: # starts beyond threshold and lasts >=n_consec_bins
            crosses=crosses[:-1]

    return [np.vstack([cross, arr[cross]]) for cross in crosses]

@njit(cache=True)
def thresh_numba(arr, th, sgn=1, pos=1):
    '''Returns indices of the data points closest to a directed crossing of th.
    - data: numpy array.
    - th: threshold value.
    - sgn: 1 or -1, direction of the threshold crossing, either positive (1) or negative (-1).
    - pos: 1 or -1, position of closest value, just following or just preceeding.'''
    assert pos in [-1,1]
    assert sgn in [-1,1]
    arr=np.asarray(arr)
    tp=arr.dtype
    assert arr.ndim==1
    arr= (arr-th)*sgn+th # Flips the array around threshold if sgn==-1

    i=np.nonzero(arr>=th)[0] if pos==1 else np.nonzero(arr<=th)[0]
    # If no value is above the threshold or all of them are already above the threshold
    if len(i)==0 or len(i)==len(arr):
        return np.zeros(0).astype(tp)
    if pos==1:
        clip=np.maximum(0, np.minimum(i-1, len(arr)-1))
        return (i[arr[clip]<th]).astype(tp)
    else:
        clip=np.maximum(0, np.minimum(i+1, len(arr)-1))
        return (i[arr[clip]>th]).astype(tp)

def thresh(arr, th, sgn=1, pos=1):
    '''Returns indices of the data points closest to a directed crossing of th.
    - data: numpy array.
    - th: threshold value.
    - sgn: 1 or -1, direction of the threshold crossing, either positive (1) or negative (-1).
    - pos: 1 or -1, position of closest value, just following or just preceeding.'''
    assert pos in [-1,1]
    assert sgn in [-1,1]
    arr=np.asarray(arr).copy()
    assert arr.ndim==1
    arr = (arr-th)*sgn+th # Flips the array around threshold if sgn==-1

    i=np.nonzero(arr>=th)[0] if pos==1 else np.nonzero(arr<=th)[0]
    # If no value is above the threshold or all of them are already above the threshold
    if len(i)==0 or len(i)==len(arr):
        return np.array([])
    return  i[arr[np.clip(i-1, 0, len(arr)-1)]<th] if pos==1 else i[arr[np.clip(i+1, 0, len(arr)-1)]>th]

def thresh_fast(arr, th, sgn=1, pos=1):
    '''Returns indices of the data points closest to a directed crossing of th.
    - data: numpy array.
    - th: threshold value.
    - sgn: 1 or -1, direction of the threshold crossing, either positive (1) or negative (-1).
    - pos: 1 or -1, position of closest value, just following or just preceeding.'''
    assert pos in [-1,1]
    assert sgn in [-1,1]
    assert arr.ndim==1
    m=(arr>th).astype(np.int8)
    ths=np.nonzero(np.diff(m)==sgn)[0]
    if pos: ths+=1
    return ths

def thresh_consec(arr, th, sgn=1, n_consec=0, exclude_edges=True, only_max=False, ret_values=True):
    '''
    Returns indices and values of threshold crosses lasting >=n_consec consecutive samples in arr.
    Parameters:
        - arr: numpy array to threshold
        - th: float, threshold
        - sgn: 1 or -1, positive (for cases above threshold) or negative (below threshold)
        - n_consec: optional int, minimum number of consecutive elements beyond threshold | Defult 0 (any cross)
        - exclude_edges: bool, if true edges of arr are not considered as threshold crosses
                         in cases where arr starts or ends beyond threshold
        - only_max: bool, if True returns only the most prominent threshold cross.
        - ret_values: bool, whether to return crosses values (list of 2d np arrays [indices, array values] of len Ncrosses)
                            rather than mere crosses indices (2d np array of shape (Ncrosses, 2))
    Returns:
        - crosses values, list of 2d np arrays [indices, array values] if ret_values is True
               or indices, 2d np array of shape (Ncrosses, 2) if ret_values is False
    '''
    def thresh_cons(arr, th, sgn=1, n_consec=0, exclude_edges=True, ret_values=True):
        arr=npa(arr)
        if not arr.ndim==1:
            assert 1 in arr.shape
            arr=arr.flatten()

        assert sgn in [-1,1]
        arr= (arr-th)*sgn+th # Flips the array around threshold if sgn==-1

        cross_thp, cross_thn = thresh(arr, th, 1, 1), thresh(arr, th, -1, -1)
        if exclude_edges:
            if len(cross_thp)+len(cross_thn)<=1: cross_thp, cross_thn = [], [] # Only one cross at the beginning or the end e.g.
            else:
                flag0,flag1=False,False
                if cross_thp[-1]>cross_thn[-1]: flag1=True # if + cross at the end
                if cross_thn[0]<cross_thp[0]: flag0=True # if - cross at the beginning
                if flag1: cross_thp=cross_thp[:-1] # remove last + cross
                if flag0: cross_thn=cross_thn[1:] # remove first - cross
        else:
            if len(cross_thp)+len(cross_thn)<=1: cross_thp, cross_thn = [], [] # Only one cross at the beginning or the end e.g.
            else:
                flag0,flag1=False,False
                if cross_thp[-1]>cross_thn[-1]: flag1=True # if + cross at the end
                if cross_thn[0]<cross_thp[0]: flag0=True # if - cross at the beginning
                if flag1: cross_thn=np.append(cross_thn, [len(arr)-1]) # add fake - cross at the end
                if flag0: cross_thp=np.append([0],cross_thp) # add fake + cross at the beginning

        assert len(cross_thp)==len(cross_thn)

        if ret_values or only_max:
            crosses=[np.vstack([np.arange(cross_thp[i], cross_thn[i]+1, 1), ((arr-th)*sgn+th)[cross_thp[i]:cross_thn[i]+1]]) for i in range(len(cross_thp)) if cross_thn[i]+1-cross_thp[i]>=n_consec]
        else:
            crosses=[[cross_thp[i], cross_thn[i]] for i in range(len(cross_thp)) if cross_thn[i]+1-cross_thp[i]>=n_consec]
        return crosses

    sgn=[-1,1] if sgn==0 else [sgn]
    crosses=[]
    for s in sgn:
        crosses+=thresh_cons(arr, th, s, n_consec, exclude_edges, ret_values)

    if only_max and len(crosses)>0:
        cross=crosses[0]
        for c in crosses[1:]:
            if max(abs(c[1,:]))>max(abs(cross[1,:])): cross = c
        if ret_values:
            crosses=[cross]
        else:
            crosses=[[cross[0,0], cross[0,-1]]]

    return crosses

def zscore(arr, frac=4./5, mn_ext=None, sd_ext=None):
    '''
    Returns z-scored (centered, reduced) array using outer edges of array to compute mean and std.
    Parameters:
        - arr: 1D np array
        - frac: fraction of array used to compute mean and standard deviation
        - mn_ext: optional, provide mean computed outside of function
        - sd_ext: optional, provide standard deviation computed outside of function
    '''
    assert 0<frac<=1, 'Z-score fraction should be between 0 and 1!'
    mn = np.mean(np.append(arr[:int(len(arr)*frac/2)], arr[int(len(arr)*(1-frac/2)):])) if mn_ext is None else mn_ext
    sd = np.std(np.append(arr[:int(len(arr)*frac/2)], arr[int(len(arr)*(1-frac/2)):])) if sd_ext is None else sd_ext
    if sd==0: sd=1
    return (arr-mn)*1./sd

def smooth(arr, method='gaussian_causal', sd=5, axis=1, gamma_a=5):
    '''
    Smoothes a 1D array or a 2D array along specified axis.
    Parameters:
        - arr: ndarray/list, array to smooth
        - method: string, see methods implemented below | Default 'gaussian'
        - sd: int, gaussian window sd (in unit of array samples - e.g. use 10 for a 1ms std if bin size is 0.1ms) | Default 5
        - axis: int axis along which smoothing is performed.
        - a_gamma: sqrt of Gamma function rate (essentially std) | Default 5

    methods implemented:
        - gaussian
        - gaussian_causal
        - gamma (is causal)
    '''
    assert arr.ndim<=2,\
        "WARNING this function runs on 3D arrays but seems to shift data leftwards - not functional yet."
    if arr.ndim==1: axis=0
    
    ## Checks and formatting
    assert method in ['gaussian', 'gaussian_causal', 'gamma']
    assert type(sd) in [int, np.int]


    ## pad array at beginning and end to prevent edge artefacts
    C = arr.shape[axis]//2
    pad_width = [[C,C] if i==axis else [0,0] for i in range(arr.ndim)]
    padarr=np.pad(arr, pad_width, 'symmetric')


    ## Compute the kernel
    if method in ['gaussian', 'gaussian_causal']:
        X=np.arange(-4*sd, 4*sd+1)
        kernel=stt.norm.pdf(X, 0, sd)
        if method=='gaussian_causal':
            kernel[:len(kernel)//2]=0
    elif method=='gamma':
        # a = shape, b = scale = 1/rate. std=sqrt(a)/b = sqrt(a) for b=1
        X=np.arange(gamma_a**2//2, max((gamma_a**2)*3//2+1, 10))
        kernel=stt.gamma.pdf(X, gamma_a**2)
    
    # center the maximum to prevent data shift in time
    # This is achieved by padding the left/right of the kernel with zeros.
    mx=np.argmax(kernel)
    if mx<len(kernel)/2:
        kernel=np.append(np.zeros(len(kernel)-2*mx), kernel)
    elif mx>len(kernel)/2:
        kernel=np.append(kernel, np.zeros(mx-(len(kernel)-mx)))
    assert len(kernel)<padarr.shape[axis],\
        'The kernel is longer than the array to convolved, you must decrease sd.'

    # normalize kernel to prevent vertical scaling
    kernel=kernel/sum(kernel)


    ## Convolve array with kernel
    sarr = np.apply_along_axis(lambda m:np.convolve(m, kernel, mode='same'), axis=axis, arr=padarr)


    ## Remove padding
    sarr = sarr[slice_along_axis(C,-C,axis=axis)]
    assert np.all(sarr.shape==arr.shape)


    return sarr


def slice_along_axis(a,b,s=1,axis=0):
    """
    Returns properly formatted slice to slice array/list along specified axis.
    - a: start
    - b: end
    - s: step
    """
    slc = slice(a,b,s)
    return (slice(None),) * axis + (slc,)


def xcorr_axis(x, y, axis=0):
    """
    Cross-correlation between two Nd arrays
    only along the specified axis
    (eg. for 2D array along axis 1, equivalent of applying np.correlate() to every pair of 1D array).

    Written because np.correlate does not handle axis argument, whereas fft does.

    Equivalent to (for axis 0, is x is 2D):
    c = np.zeros(x.shape)
    for i, (x_, y_) in enumerate(zip(normalize(x,0).T, normalize(y,0).T)):
        c[:,i] = np.correlate(x_, y_, mode='same')
    (actually this for loop is faster for small array because everything is done in C by numpy)
    """
    assert x.shape[axis]>3, f'array too short along axis {axis} to compute crosscorrelation.'
    y_rev = np.flip(y, axis=axis)
    c = irfft(rfft(x, axis=axis)*rfft(y_rev, axis=axis), axis=axis)

    # now need to simply reindex along the specified axis
    c_first = c.take(indices=range(0, c.shape[axis]//2-1), axis=axis)
    c_last = c.take(indices=range(c.shape[axis]//2-1, c.shape[axis]), axis=axis)

    return np.concatenate([c_last, c_first], axis=axis)

def xcorr_1d_fft(w1, w2, axis = 0):
    """
    Cross-correlation along specific axis between two Nd arrays.
    Typically for template matching when you know that you only need
    to 'slide' the template along a single axis (e.g. a column of 100x3 pixels in a 100x100 image,
    or a waveform across channels (10x82 samples) and a waveform template (10x82 as well)).

    w1, w2: Nd arrays
    axis: int, axis along which to perform crosscorrelation
    """
    c = xcorr_axis(normalize(w1,axis), normalize(w2,axis), axis=axis)
    # reindex along axis - fucked up at the moment
    return c

def xcorr_1d_loop(w1, w2):
    """
    Cross-correlation along axis 0 between two 2D arrays.
    Typically for template matching when you know that you only need
    to 'slide' the template along a single axis (e.g. a column of 100x3 pixels in a 100x100 image,
    or a waveform across channels (10x82 samples) and a waveform template (10x82 as well)).

    Faster than xcorr_1d_fft for small 2D arrays.

    w1, w2: 2d arrays
    """
    c = np.zeros(w1.shape)
    for i, (wave1, wave2) in enumerate(zip(normalize(w1,0).T, normalize(w2,0).T)):
        c[:,i] = np.correlate(wave1, wave2, mode='same')
    c[np.isnan(c)]=0 # replace nans with 0 correlation
    return c

def xcorr_2d(w1, w2):
    """
    Cross-correlation along ALL axis between two Nd arrays.
    Typically for template matching when you want to 'slide' the template along all axis
    (e.g. a 10x10 element pixels in a 100x100 image,
    or a 10x10x10 voxel in a 100x100x100 volume).
    """
    c = sgnl.correlate(normalize(w1, 0), normalize(w2, 0),  mode='same')
    return c

def normalize(x, axis=0):
    """
    Vanilla normalization (center, reduce) along specified axis
    """
    s = np.std(x, axis=axis)
    m = np.mean(x, axis=axis)
    return (x-m)/s

def get_bins(cwin, cbin):
    mod2=(cwin/cbin)%2
    if mod2==0: # even
        return np.arange(-cwin/2, cwin/2+cbin, cbin)
    else: # odd
        return np.arange(-cwin/2+cbin/2, cwin/2+cbin/2, cbin)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def mask_2d(x, m):
    '''
    Mask a 2D array and preserve the
    dimension on the resulting array
    ----------
    x: np.array
       2D array on which to apply a mask
    m: np.array
        2D boolean mask
    Returns
    -------
    List of arrays. Each array contains the
    elements from the rows in x once masked.
    If no elements in a row are selected the
    corresponding array will be empty
    '''
    take = m.sum(axis=1)
    return np.split(x[m], np.cumsum(take)[:-1])

@njit(cache=True)
def make_2D_array(arr_lis, accept_heterogeneous=False):
    """Function to get 2D array from a list of lists
    """
    lis=[np.asarray(l) for l in arr_lis]
    n = len(lis)
    lengths = [len(l) for l in lis]
    if accept_heterogeneous:
        max_len = max(lengths)
        arr = np.zeros((n, max_len))*np.nan
        for i,l in enumerate(lis):
            arr[i, :lengths[i]] = l
    else:
        assert max(lengths)==min(lengths)
        arr = np.zeros((n, len(lis[0])))*np.nan
        for i,l in enumerate(lis):
            arr[i, :] = l
    return arr

@njit(cache=True)
def split(arr, sample_size=0, n_samples=0, overlap=0, return_last=True, verbose=True):
    '''
    Parameters:
        - arr: array to split into EITHER n_samples OR samples of size sample_size.
               samples_size has priority over n_samples.
        - s_samples: int, size of of samples to split the array into.
        - n_samples: int, number of samples to split the array into.
        - overlap: 0<=float<1, fraction of overlapping between consecutive samples | Default 0
        - return_last: bool, whether to return the last sample (whose size usually does not match the ones before).
    Returns:
        samples: array or list of samples
    '''

    arr=np.asarray(arr)
    assert n_samples!=0 or sample_size!=0, 'You need to specify either a sample number or size!'
    assert 0<=overlap<1, 'overlap needs to be between 0 and 1, 1 excluded (samples would all be the same)!'

    if n_samples!=0 and sample_size!=0:
        print('''WARNING you provided n_samples AND sample_size!
              By convention, sample_size has priority over n_samples. n_samples ignored.''')
        n_samples=0

    if n_samples==0: n_samples=len(arr)//sample_size+1
    if sample_size==0: sample_size=len(arr)//(n_samples)

    assert n_samples<=len(arr)

    # step=1 for maximum overlap (every sample would be the first sample for step = 0);
    # step=s for 0 overlap; overlap is
    s=sample_size
    if len(arr)<s: # no split necessary
        return make_2D_array([list(arr)])
    step = s-round(s*overlap)
    real_o=round((s-step)/s, 2)
    if overlap!=real_o and verbose: print('Real overlap: ', round(real_o, 2))
    samples = List([arr[i : i + s] for i in range(0, len(arr), step)])

    # always return last sample if len matches
    if len(samples[-1])==s:
        return make_2D_array(samples)

    if return_last:
        return make_2D_array(samples, accept_heterogeneous=True)

    lasti=-1
    sp=samples[lasti]
    while len(sp)!=s:
        lasti-=1
        sp=samples[lasti]
    return make_2D_array(samples[:lasti+1])

# def n_largest_samples(to_sort: np.array, largest_n: int) -> np.array:

#     """
#     Returns the n largest sorted samples from an array
#     """

#     sorted_n = np.argpartition(to_sort, -largest_n, axis=0 )[-largest_n:].flatten()
#     return sorted_n

def align_timeseries(timeseries, sync_signals, fs, offset_policy='original'):
    '''
    Usage 1: align >=2 time series in the same temporal reference frame with the same sampling frequency fs
        aligned_ts1, aligned_ts2, ... = align_timeseries([ts1,ts2,...], [sync1,sync2,...], fs)
    Usage 2:  align 1 time serie to another temporal reference frame
        aligned_ts = align_timeseries([ts], [sync_ts, sync_other], [fs_ts, fs_other])

    Re-aligns in time series based on provided sync signals.
    - timeseries: list of numpy arrays of time stamps (e.g. spikes), in SAMPLES
      If Usage 1: THEY MUST BE IN THE SAME TIME REFERENCE FRAME
    - sync_signals: list of numpy arrays of synchronization time stamps,
      ordered with respect to timeseries
      If Usage 1: THEY MUST ALSO BE IN THE SAME TIME REFERENCE FRAME
      - fs: int (usage 1) or list of 2 ints (usage 2), sampling frequencies of timeseries and respective sync_signals.
      - offset_policy: 'original' or 'zero', only for usage 1: whether to set timeseries[0] as 0 or as its original value after alignement.
      The FIRST sync_signal is used as a reference.

    Returns:
     timeseries, aligned accordingly to sync signals.

    '''
    assert type(timeseries) is type(sync_signals) is list, "You must provide timeseries and sync_signals as lists of arrays!"
    for tsi, ts in enumerate(timeseries):
        assert np.all(ts.astype(np.int64)==ts), 'Timeseries need to be integers, in samples acquired at fs sampling rate!'
        timeseries[tsi]=ts.astype(np.int64)
    for tsi, ts in enumerate(sync_signals):
        assert np.all(ts.astype(np.int64)==ts), 'Sync signals need to be integers, in samples acquired at fs sampling rate!'
        sync_signals[tsi]=ts.astype(np.int64)


    for ss in sync_signals:
        assert len(sync_signals[0])==len(ss), "WARNING all sync signals do not have the same size, the acquisition must have been faulty!"
    assert len(sync_signals[0])>=1, "Only one synchronization signal has been provided - this is dangerous practice as this does not account for cumulative time drift."
    if len(sync_signals[0])>50:
        print('More than 50 sync signals found - for performance reasons, sub-sampling to 50 homogenoeously spaced sync signals to align data.')
        subselect_ids=np.random.choice(np.arange(1,len(sync_signals[0])-1), 48, replace=False)
        subselect_ids=np.unique(np.append(subselect_ids,[0,len(sync_signals[0])-1])) # enforce first and last stim
        for synci,sync in enumerate(sync_signals):
            sync_signals[synci]=sync[subselect_ids]

    if len(timeseries)==1: # Usage 2
        usage=2
        offset=sync_signals[1][0]
        assert len(sync_signals)==2 and len(npa([fs]).flatten())==2, '''When providing a single time series ts, you need to provide 2 sync signals,
                                        [sync_ts, sync_other] to align ts to sync_other reference frame!'''
        fs_master=fs[1]
        fsconv=fs[1]/fs[0]
        timeseries=[sync_signals[1], timeseries[0]*fsconv]
        sync_signals=[sync_signals[1], sync_signals[0]*fsconv]
    elif len(timeseries)>=2: # Usage 1
        usage=1
        assert offset_policy in ['zero', 'original'], "offset_policy must be in ['zero', 'original']"
        offset = 0 if offset_policy=='original' else -timeseries[0][0]
        assert len(timeseries)==len(sync_signals)>=2, "There must be as many time series as sync signals, at least 2 of each!"
        assert len(npa([fs]).flatten())==1, 'You must provide a single sampling frequency fs when aligning >=2 time series (Usage 1)!'
        fs_master=npa([fs]).flatten()[0]


    Nevents, totDft, avDft, stdDft = len(sync_signals[0]), (sync_signals[1]-sync_signals[0])[-1], np.mean(np.diff(sync_signals[1]-sync_signals[0])), np.std(np.diff(sync_signals[1]-sync_signals[0]))
    totDft, avDft, stdDft = totDft*1000/fs_master, avDft*1000/fs_master, stdDft*1000/fs_master
    print("{} sync events used for alignement - start-end drift of {}ms".format(Nevents, round(totDft,3)))

    for dataset_i in range(len(timeseries)):
        if dataset_i==0: continue #first dataset is reference so left untouched
        array0, syncs = timeseries[dataset_i], sync_signals[dataset_i]
        array=array0-syncs[0] # initially center on first sync
        for i, sync in enumerate(syncs):
            if i>0: # first sync already subtracted
                array1=array0-sync # refer to own sync stamp, sync
                # re-time to where own sync stamp should be with respect to reference dataset (the 1st one)
                first_sync_ref0=sync_signals[0][i]-sync_signals[0][0]
                array=np.append(array[array1<0], array1[array1>=0]+first_sync_ref0)
        if usage==1: array+=syncs[0] # restore original offset
        timeseries[dataset_i]=array+offset

    return timeseries if usage==1 else timeseries[1]


def  align_timeseries_interpol(timeseries, sync_signals, fs=None):
    '''
    Align a list of N timeseries in the temporal reference frame of the first timeserie.

    Assumes that the drift is going to be linear, it is interpolated for times far from sync signals
    (sync_signal1 = a * sync_signal0 + b).

    Parameters:
    - timeseries: list[array[int]], list of np arrays (len N), timeseries to align. In SAMPLES to ensure accurate alignment
    - sync_signals: list[array[int]], list of np arrays (len N), sync signals respective to timeseries. In SAMPLES to ensure accurate alignment
    fs: float or list of floats (len N), sampling frequencies of time series.
        fs is optional (only to print drift and offset in seconds).

    returns:
        - timeseries: list of np arrays (len N, in samples), timeries aligned in temporal reference frame of timeseries[0]
    '''

    # Parameters formatting and checks
    assert len(timeseries)>=2
    assert len(sync_signals)==len(timeseries)
    if fs is not None:
        if assert_iterable(fs):assert len(timeseries)==len(fs)
        else:fs=[fs]*len(timeseries)
    for tsi, ts in enumerate(timeseries):
        assert np.all(ts.astype(np.int64)==ts), 'Timeseries need to be integers, in samples acquired at fs sampling rate!'
        timeseries[tsi]=ts.astype(np.int64)
    for tsi, ts in enumerate(sync_signals):
        assert np.all(ts.astype(np.int64)==ts), 'Sync signals need to be integers, in samples acquired at fs sampling rate!'
        sync_signals[tsi]=ts.astype(np.int64)

    # Align
    ref_sync=sync_signals[0]

    for i, (ts, ss) in enumerate(zip(timeseries[1:], sync_signals[1:])):
        (a, b) = np.polyfit(ss, ref_sync, 1)
        if fs is not None:
            drift=round(abs(a*fs[i]/fs[0]-1)*3600*1000,2)
            offset=round(b/fs[0],2)
            print(f'Drift (assumed linear) of {drift}ms/h, \noffset of {offset}s between time series 1 and {i+2}.\n')
        timeseries[i+1]=np.round(a*ts+b, 0).astype(np.int64)

    return timeseries

#%% Stolen from phy
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

#%% Peakdetect functions

__all__ = [
        "peakdetect",
        "peakdetect_fft",
        "peakdetect_parabola",
        "peakdetect_sine",
        "peakdetect_sine_locked",
        "peakdetect_spline",
        "peakdetect_zero_crossing",
        "zero_crossings",
        "zero_crossings_sine_fit"
        ]



def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        raise ValueError(
                "Input vectors y_axis and x_axis must have same length")

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def _pad(fft_data, pad_len):
    """
    Pads fft data to interpolate in time domain

    keyword arguments:
    fft_data -- the fft
    pad_len --  By how many times the time resolution should be increased by

    return: padded list
    """
    l = len(fft_data)
    n = _n(l * pad_len)
    fft_data = list(fft_data)

    return fft_data[:l // 2] + [0] * (2**n-l) + fft_data[l // 2:]

def _n(x):
    """
    Find the smallest value for n, which fulfils 2**n >= x

    keyword arguments:
    x -- the value, which 2**n must surpass

    return: the integer n
    """
    return int(log(x)/log(2)) + 1


def _peakdetect_parabola_fitter(raw_peaks, x_axis, y_axis, points):
    """
    Performs the actual parabola fitting for the peakdetect_parabola function.

    keyword arguments:
    raw_peaks -- A list of either the maxima or the minima peaks, as given
        by the peakdetect functions, with index used as x-axis

    x_axis -- A numpy array of all the x values

    y_axis -- A numpy array of all the y values

    points -- How many points around the peak should be used during curve
        fitting, must be odd.


    return: A list giving all the peaks and the fitted waveform, format:
        [[x, y, [fitted_x, fitted_y]]]

    """
    func = lambda x, a, tau, c: a * ((x - tau) ** 2) + c
    fitted_peaks = []
    distance = abs(x_axis[raw_peaks[1][0]] - x_axis[raw_peaks[0][0]]) / 4
    for peak in raw_peaks:
        index = peak[0]
        x_data = x_axis[index - points // 2: index + points // 2 + 1]
        y_data = y_axis[index - points // 2: index + points // 2 + 1]
        # get a first approximation of tau (peak position in time)
        tau = x_axis[index]
        # get a first approximation of peak amplitude
        c = peak[1]
        a = np.sign(c) * (-1) * (np.sqrt(abs(c))/distance)**2
        """Derived from ABC formula to result in a solution where A=(rot(c)/t)**2"""

        # build list of approximations

        p0 = (a, tau, c)
        popt, pcov = curve_fit(func, x_data, y_data, p0)
        # retrieve tau and c i.e x and y value of peak
        x, y = popt[1:3]

        # create a high resolution data set for the fitted waveform
        x2 = np.linspace(x_data[0], x_data[-1], points * 10)
        y2 = func(x2, *popt)

        fitted_peaks.append([x, y, [x2, y2]])

    return fitted_peaks


def peakdetect_parabole(*args, **kwargs):
    """
    Misspelling of peakdetect_parabola
    function is deprecated please use peakdetect_parabola
    """
    logging.warn("peakdetect_parabole is deprecated due to misspelling use: peakdetect_parabola")

    return peakdetect_parabola(*args, **kwargs)


def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)

    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200)
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value

    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def peakdetect_fft(y_axis, x_axis, pad_len = 20):
    """
    Performs a FFT calculation on the data and zero-pads the results to
    increase the time domain resolution after performing the inverse fft and
    send the data to the 'peakdetect' function for peak
    detection.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as the index 50.234 or similar.

    Will find at least 1 less peak then the 'peakdetect_zero_crossing'
    function, but should result in a more precise value of the peak as
    resolution has been increased. Some peaks are lost in an attempt to
    minimize spectral leakage by calculating the fft between two zero
    crossings for n amount of signal periods.

    The biggest time eater in this function is the ifft and thereafter it's
    the 'peakdetect' function which takes only half the time of the ifft.
    Speed improvements could include to check if 2**n points could be used for
    fft and ifft or change the 'peakdetect' to the 'peakdetect_zero_crossing',
    which is maybe 10 times faster than 'peakdetct'. The pro of 'peakdetect'
    is that it results in one less lost peak. It should also be noted that the
    time used by the ifft function can change greatly depending on the input.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.

    pad_len -- By how many times the time resolution should be
        increased by, e.g. 1 doubles the resolution. The amount is rounded up
        to the nearest 2**n amount
        (default: 20)


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    zero_indices = zero_crossings(y_axis, window_len = 11)
    #select a n amount of periods
    last_indice = - 1 - (1 - len(zero_indices) & 1)
    ###
    # Calculate the fft between the first and last zero crossing
    # this method could be ignored if the beginning and the end of the signal
    # are unnecessary as any errors induced from not using whole periods
    # should mainly manifest in the beginning and the end of the signal, but
    # not in the rest of the signal
    # this is also unnecessary if the given data is an amount of whole periods
    ###
    fft_data = fft(y_axis[zero_indices[0]:zero_indices[last_indice]])
    padd = lambda x, c: x[:len(x) // 2] + [0] * c + x[len(x) // 2:]
    n = lambda x: int(log(x)/log(2)) + 1
    # pads to 2**n amount of samples
    fft_padded = padd(list(fft_data), 2 **
                n(len(fft_data) * pad_len) - len(fft_data))

    # There is amplitude decrease directly proportional to the sample increase
    sf = len(fft_padded) / float(len(fft_data))
    # There might be a leakage giving the result an imaginary component
    # Return only the real component
    y_axis_ifft = ifft(fft_padded).real * sf #(pad_len + 1)
    x_axis_ifft = np.linspace(
                x_axis[zero_indices[0]], x_axis[zero_indices[last_indice]],
                len(y_axis_ifft))
    # get the peaks to the interpolated waveform
    max_peaks, min_peaks = peakdetect(y_axis_ifft, x_axis_ifft, 500,
                                    delta = abs(np.diff(y_axis).max() * 2))
    #max_peaks, min_peaks = peakdetect_zero_crossing(y_axis_ifft, x_axis_ifft)

    # store one 20th of a period as waveform data
    data_len = int(np.diff(zero_indices).mean()) / 10
    data_len += 1 - data_len & 1


    return [max_peaks, min_peaks]


def peakdetect_parabola(y_axis, x_axis, points = 31):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by fitting the model function: y = k (x - tau) ** 2 + m
    to the peaks. The amount of points used in the fitting is set by the
    points argument.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly, if it was returned as index 50.234 or similar.

    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.

    points -- How many points around the peak should be used during curve
        fitting (default: 31)


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    #points += 1 - int(points) & 1 slower when int conversion needed

    # get raw peaks
    max_raw, min_raw = peakdetect_zero_crossing(y_axis)

    # define output variable
    max_peaks = []
    min_peaks = []

    max_ = _peakdetect_parabola_fitter(max_raw, x_axis, y_axis, points)
    min_ = _peakdetect_parabola_fitter(min_raw, x_axis, y_axis, points)

    max_peaks = map(lambda x: [x[0], x[1]], max_)
    max_fitted = map(lambda x: x[-1], max_)
    min_peaks = map(lambda x: [x[0], x[1]], min_)
    min_fitted = map(lambda x: x[-1], min_)

    return [max_peaks, min_peaks]


def peakdetect_sine(y_axis, x_axis, points = 31, lock_frequency = False):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by fitting the model function:
    y = A * sin(2 * pi * f * (x - tau)) to the peaks. The amount of points used
    in the fitting is set by the points argument.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as index 50.234 or similar.

    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.

    The function might have some problems if the sine wave has a
    non-negligible total angle i.e. a k*x component, as this messes with the
    internal offset calculation of the peaks, might be fixed by fitting a
    y = k * x + m function to the peaks for offset calculation.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.

    points -- How many points around the peak should be used during curve
        fitting (default: 31)

    lock_frequency -- Specifies if the frequency argument of the model
        function should be locked to the value calculated from the raw peaks
        or if optimization process may tinker with it.
        (default: False)


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # make the points argument odd
    points += 1 - points % 2
    #points += 1 - int(points) & 1 slower when int conversion needed

    # get raw peaks
    max_raw, min_raw = peakdetect_zero_crossing(y_axis)

    # define output variable
    max_peaks = []
    min_peaks = []

    # get global offset
    offset = np.mean([np.mean(max_raw, 0)[1], np.mean(min_raw, 0)[1]])
    # fitting a k * x + m function to the peaks might be better
    #offset_func = lambda x, k, m: k * x + m

    # calculate an approximate frequency of the signal
    Hz_h_peak = np.diff(zip(*max_raw)[0]).mean()
    Hz_l_peak = np.diff(zip(*min_raw)[0]).mean()
    Hz = 1 / np.mean([Hz_h_peak, Hz_l_peak])



    # model function
    # if cosine is used then tau could equal the x position of the peak
    # if sine were to be used then tau would be the first zero crossing
    if lock_frequency:
        func = lambda x_ax, A, tau: A * np.sin(
            2 * pi * Hz * (x_ax - tau) + pi / 2)
    else:
        func = lambda x_ax, A, Hz, tau: A * np.sin(
            2 * pi * Hz * (x_ax - tau) + pi / 2)
    #func = lambda x_ax, A, Hz, tau: A * np.cos(2 * pi * Hz * (x_ax - tau))


    #get peaks
    fitted_peaks = []
    for raw_peaks in [max_raw, min_raw]:
        peak_data = []
        for peak in raw_peaks:
            index = peak[0]
            x_data = x_axis[index - points // 2: index + points // 2 + 1]
            y_data = y_axis[index - points // 2: index + points // 2 + 1]
            # get a first approximation of tau (peak position in time)
            tau = x_axis[index]
            # get a first approximation of peak amplitude
            A = peak[1]

            # build list of approximations
            if lock_frequency:
                p0 = (A, tau)
            else:
                p0 = (A, Hz, tau)

            # subtract offset from wave-shape
            y_data -= offset
            popt, pcov = curve_fit(func, x_data, y_data, p0)
            # retrieve tau and A i.e x and y value of peak
            x = popt[-1]
            y = popt[0]

            # create a high resolution data set for the fitted waveform
            x2 = np.linspace(x_data[0], x_data[-1], points * 10)
            y2 = func(x2, *popt)

            # add the offset to the results
            y += offset
            y2 += offset
            y_data += offset

            peak_data.append([x, y, [x2, y2]])

        fitted_peaks.append(peak_data)

    # structure date for output
    max_peaks = map(lambda x: [x[0], x[1]], fitted_peaks[0])
    max_fitted = map(lambda x: x[-1], fitted_peaks[0])
    min_peaks = map(lambda x: [x[0], x[1]], fitted_peaks[1])
    min_fitted = map(lambda x: x[-1], fitted_peaks[1])


    return [max_peaks, min_peaks]


def peakdetect_sine_locked(y_axis, x_axis, points = 31):
    """
    Convenience function for calling the 'peakdetect_sine' function with
    the lock_frequency argument as True.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
    points -- How many points around the peak should be used during curve
        fitting (default: 31)

    return: see the function 'peakdetect_sine'
    """
    return peakdetect_sine(y_axis, x_axis, points, True)


def peakdetect_spline(y_axis, x_axis, pad_len=20):
    """
    Performs a b-spline interpolation on the data to increase resolution and
    send the data to the 'peakdetect_zero_crossing' function for peak
    detection.

    Omitting the x_axis is forbidden as it would make the resulting x_axis
    value silly if it was returned as the index 50.234 or similar.

    will find the same amount of peaks as the 'peakdetect_zero_crossing'
    function, but might result in a more precise value of the peak.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks.
        x-axis must be equally spaced.

    pad_len -- By how many times the time resolution should be increased by,
        e.g. 1 doubles the resolution.
        (default: 20)


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # could perform a check if x_axis is equally spaced
    #if np.std(np.diff(x_axis)) > 1e-15: raise ValueError
    # perform spline interpolations
    dx = x_axis[1] - x_axis[0]
    x_interpolated = np.linspace(x_axis.min(), x_axis.max(), len(x_axis) * (pad_len + 1))
    cj = cspline1d(y_axis)
    y_interpolated = cspline1d_eval(cj, x_interpolated, dx=dx,x0=x_axis[0])
    # get peaks
    max_peaks, min_peaks = peakdetect_zero_crossing(y_interpolated, x_interpolated)

    return [max_peaks, min_peaks]

def peakdetect_zero_crossing(y_axis, x_axis = None, window = 11):
    """
    Function for detecting local maxima and minima in a signal.
    Discovers peaks by dividing the signal into bins and retrieving the
    maximum and minimum value of each the even and odd bins respectively.
    Division into bins is performed by smoothing the curve and finding the
    zero crossings.

    Suitable for repeatable signals, where some noise is tolerated. Executes
    faster than 'peakdetect', although this function will break if the offset
    of the signal is too large. It should also be noted that the first and
    last peak will probably not be found, as this function only can find peaks
    between the first and last zero crossing.

    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used.
        (default: None)

    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)


    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*max_peaks)
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)

    zero_indices = zero_crossings(y_axis, window_len = window)
    period_lengths = np.diff(zero_indices)

    bins_y = [y_axis[index:index + diff] for index, diff in
        zip(zero_indices, period_lengths)]
    bins_x = [x_axis[index:index + diff] for index, diff in
        zip(zero_indices, period_lengths)]

    even_bins_y = bins_y[::2]
    odd_bins_y = bins_y[1::2]
    even_bins_x = bins_x[::2]
    odd_bins_x = bins_x[1::2]
    hi_peaks_x = []
    lo_peaks_x = []

    #check if even bin contains maxima
    if abs(even_bins_y[0].max()) > abs(even_bins_y[0].min()):
        hi_peaks = [bin.max() for bin in even_bins_y]
        lo_peaks = [bin.min() for bin in odd_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
    else:
        hi_peaks = [bin.max() for bin in odd_bins_y]
        lo_peaks = [bin.min() for bin in even_bins_y]
        # get x values for peak
        for bin_x, bin_y, peak in zip(odd_bins_x, odd_bins_y, hi_peaks):
            hi_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])
        for bin_x, bin_y, peak in zip(even_bins_x, even_bins_y, lo_peaks):
            lo_peaks_x.append(bin_x[np.where(bin_y==peak)[0][0]])

    max_peaks = [[x, y] for x,y in zip(hi_peaks_x, hi_peaks)]
    min_peaks = [[x, y] for x,y in zip(lo_peaks_x, lo_peaks)]

    return [max_peaks, min_peaks]


def _smooth(x, window_len=11, window="hanning"):
    """
    smooth the data using a window of the requested size.

    This method is based on the convolution of a scaled window on the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    keyword arguments:
    x -- the input signal

    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)

    window -- the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman', where flat is a moving average
        (default: 'hanning')

    return: the smoothed signal

    example:
    t = linspace(-2,2,0.1)
    x = sin(t)+randn(len(t))*0.1
    y = _smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve, scipy.signal.lfilter
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x
    #declare valid windows in a dictionary
    window_funcs = {
        "flat": lambda _len: np.ones(_len, "d"),
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman
        }

    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    try:
        w = window_funcs[window](window_len)
    except KeyError:
        raise ValueError(
            "Window is not one of '{0}', '{1}', '{2}', '{3}', '{4}'".format(
            *window_funcs.keys()))

    y = np.convolve(w / w.sum(), s, mode = "valid")

    return y


def zero_crossings(y_axis, window_len = 11,
    window_f="hanning", offset_corrected=False):
    """
    Algorithm to find zero crossings. Smooths the curve and finds the
    zero-crossings by looking for a sign change.


    keyword arguments:
    y_axis -- A list containing the signal over which to find zero-crossings

    window_len -- the dimension of the smoothing window; should be an odd
        integer (default: 11)

    window_f -- the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman' (default: 'hanning')

    offset_corrected -- Used for recursive calling to remove offset when needed


    return: the index for each zero-crossing
    """
    # smooth the curve
    length = len(y_axis)

    # discard tail of smoothed signal
    y_axis = _smooth(y_axis, window_len, window_f)[:length]
    indices = np.where(np.diff(np.sign(y_axis)))[0]

    # check if zero-crossings are valid
    diff = np.diff(indices)
    if diff.std() / diff.mean() > 0.1:
        #Possibly bad zero crossing, see if it's offsets
        if ((diff[::2].std() / diff[::2].mean()) < 0.1 and
        (diff[1::2].std() / diff[1::2].mean()) < 0.1 and
        not offset_corrected):
            #offset present attempt to correct by subtracting the average
            offset = np.mean([y_axis.max(), y_axis.min()])
            return zero_crossings(y_axis-offset, window_len, window_f, True)
        #Invalid zero crossings and the offset has been removed
        print(diff.std() / diff.mean())
        print(np.diff(indices))
        raise ValueError(
            "False zero-crossings found, indicates problem {0!s} or {1!s}".format(
            "with smoothing window", "unhandled problem with offset"))
    # check if any zero crossings were found
    if len(indices) < 1:
        raise ValueError("No zero crossings found")
    #remove offset from indices due to filter function when returning
    return indices - (window_len // 2 - 1)
    # used this to test the fft function's sensitivity to spectral leakage
    #return indices + np.asarray(30 * np.random.randn(len(indices)), int)

############################Frequency calculation#############################
#    diff = np.diff(indices)
#    time_p_period = diff.mean()
#
#    if diff.std() / time_p_period > 0.1:
#        raise ValueError(
#            "smoothing window too small, false zero-crossing found")
#
#    #return frequency
#    return 1.0 / time_p_period
##############################################################################


def zero_crossings_sine_fit(y_axis, x_axis, fit_window = None, smooth_window = 11):
    """
    Detects the zero crossings of a signal by fitting a sine model function
    around the zero crossings:
    y = A * sin(2 * pi * Hz * (x - tau)) + k * x + m
    Only tau (the zero crossing) is varied during fitting.

    Offset and a linear drift of offset is accounted for by fitting a linear
    function the negative respective positive raw peaks of the wave-shape and
    the amplitude is calculated using data from the offset calculation i.e.
    the 'm' constant from the negative peaks is subtracted from the positive
    one to obtain amplitude.

    Frequency is calculated using the mean time between raw peaks.

    Algorithm seems to be sensitive to first guess e.g. a large smooth_window
    will give an error in the results.


    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks

    x_axis -- A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the position of the peaks. If
        omitted an index of the y_axis is used. (default: None)

    fit_window -- Number of points around the approximate zero crossing that
        should be used when fitting the sine wave. Must be small enough that
        no other zero crossing will be seen. If set to none then the mean
        distance between zero crossings will be used (default: None)

    smooth_window -- the dimension of the smoothing window; should be an odd
        integer (default: 11)


    return: A list containing the positions of all the zero crossings.
    """
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    #get first guess
    zero_indices = zero_crossings(y_axis, window_len = smooth_window)
    #modify fit_window to show distance per direction
    if fit_window == None:
        fit_window = np.diff(zero_indices).mean() // 3
    else:
        fit_window = fit_window // 2

    #x_axis is a np array, use the indices to get a subset with zero crossings
    approx_crossings = x_axis[zero_indices]



    #get raw peaks for calculation of offsets and frequency
    raw_peaks = peakdetect_zero_crossing(y_axis, x_axis)
    #Use mean time between peaks for frequency
    ext = lambda x: list(zip(*x)[0])
    _diff = map(np.diff, map(ext, raw_peaks))


    Hz = 1 / np.mean(map(np.mean, _diff))
    #Hz = 1 / np.diff(approx_crossings).mean() #probably bad precision


    #offset model function
    offset_func = lambda x, k, m: k * x + m
    k = []
    m = []
    amplitude = []

    for peaks in raw_peaks:
        #get peak data as nparray
        x_data, y_data = map(np.asarray, zip(*peaks))
        #x_data = np.asarray(x_data)
        #y_data = np.asarray(y_data)
        #calc first guess
        A = np.mean(y_data)
        p0 = (0, A)
        popt, pcov = curve_fit(offset_func, x_data, y_data, p0)
        #append results
        k.append(popt[0])
        m.append(popt[1])
        amplitude.append(abs(A))

    #store offset constants
    p_offset = (np.mean(k), np.mean(m))
    A = m[0] - m[1]
    #define model function to fit to zero crossing
    #y = A * sin(2*pi * Hz * (x - tau)) + k * x + m
    func = lambda x, tau: A * np.sin(2 * pi * Hz * (x - tau)) + offset_func(x, *p_offset)


    #get true crossings
    true_crossings = []
    for indice, crossing in zip(zero_indices, approx_crossings):
        p0 = (crossing, )
        subset_start = max(indice - fit_window, 0.0)
        subset_end = min(indice + fit_window + 1, len(x_axis) - 1.0)
        x_subset = np.asarray(x_axis[subset_start:subset_end])
        y_subset = np.asarray(y_axis[subset_start:subset_end])
        #fit
        popt, pcov = curve_fit(func, x_subset, y_subset, p0)

        true_crossings.append(popt[0])


    return true_crossings
