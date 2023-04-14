from pathlib import Path

from math import ceil
import ctypes
from functools import wraps

import numpy as np
import scipy as sp
from scipy import signal as sgnl
from scipy.signal import butter

try:
    import cupy as cp
except ImportError:
    pass
try:
    import cupy._core as cp_core
except ImportError:
    try:
        import cupy.core as cp_core
    except ImportError:
        pass
    
from textwrap import dedent
from tqdm.auto import tqdm


#%% Whitening

def whitening(x, nRange=None, use_ks_matrix=True,
             dp=None, channels_mask=None):
    '''
    Whitens along axis 0.
    For instance, time should be axis 1 and channels axis 0 to whiten across channels.
    Axis 1 must be larger than axis 0 (need enough samples to properly estimate variance, covariance).

    Arguments:
        - x: 2D array, axis 1 spans time, axis 0 observations (e.g. channel).
             Multiplying by the whitening matrix whitens across obervations.
        - nRange: if integer, number of channels to locally compute whitening filter (more robust to noise) | Default None
        - use_ks_matrix: bool, whether to use kilosort's original whitening matrix to perform the whitening
                     (rather than recomputing it from the data at hand)
        - dp: str/Path, kilosort path with whitening matrix whiten_mat.npy
        - channels_mask: boolean array, matches channels to whiten (important to save memory on large arrays)
    '''
    assert x.shape[1]>=x.shape[0]

    # Compute whitening matrix
    if use_ks_matrix:
        w, unprocessed_channels = whitening_matrix(x, nRange, use_ks_matrix, dp)
        
        if channels_mask is None:
            unprocessed_traces = x[unprocessed_channels,:]
            x = x[~unprocessed_channels,:]
        else:
            # only subselect some channels for whitening if instructed
            # can only work when loading precomputed whitening matrix
            assert np.sum(channels_mask) == x.shape[0],\
                "WARNING inconsistency between provided channel_mask and partial array to whiten!"
            unprocessed_channels_masked = unprocessed_channels[channels_mask]
            channels_mask = channels_mask[~unprocessed_channels]
            channels_2D_mask = channels_mask[:,None] & channels_mask[None,:]
            w = w[np.nonzero(channels_2D_mask)]
            w = w.reshape((channels_mask.sum(), channels_mask.sum()))

            unprocessed_traces = x[unprocessed_channels_masked,:]
            x = x[~unprocessed_channels_masked, :]
    else:
        w = whitening_matrix(x, nRange, use_ks_matrix, dp)

    
    # Whiten and re-scale to match original microvolts
    x = cp.array(x)
    scales=(np.max(x, 1)-np.min(x, 1))
    x = np.dot(x.T,w).T
    W_scales=(np.max(x, 1)-np.min(x, 1))
    x=x*np.repeat((scales/W_scales).reshape(x.shape[0], 1), x.shape[1], axis=1)

    # re-plug in unprocessed channels
    if use_ks_matrix:
        if channels_mask is not None:
            x_full = cp.zeros((unprocessed_channels_masked.shape[0], x.shape[1]))
            x_full[~unprocessed_channels_masked] = x
            x_full[unprocessed_channels_masked] = unprocessed_traces
        else:
            x_full = cp.zeros((unprocessed_channels.shape[0], x.shape[1]))
            x_full[~unprocessed_channels] = x
            x_full[unprocessed_channels] = unprocessed_traces
        
        x = x_full
    
    if 'cp' in globals():
        x = cp.asnumpy(x)

    return x

def whitening_matrix(x, nRange=None, use_ks_matrix=True, dp=None):
    """
    Compute the whitening matrix using ZCA.

    Arguments:
        - x: 2D array, axis 1 spans time, axis 0 observations (e.g. channel).
             Multiplying by the whitening matrix whitens across obervations.
        - epsilon: small value added to diagonal to regularize D
        - nRange: if integer, number of channels to locally compute whitening filter (more robust to noise) | Default None
        - use_ks_matrix: bool, whether to return kilosort's original whitening matrix
                     (rather than recomputing it from the data at hand)
        - dp: str/Path, kilosort path with whitening matrix whiten_mat.npy
    """
    assert x.ndim == 2
    nrows, ncols = x.shape

    if use_ks_matrix:
        if nRange is not None:
            print(("WARNING you instructed to use kilosort's original whitening matrix,"
                  " so nRange is not taken into account (kilosort uses nRange=32 by default)"))
        assert dp is not None, "You must provide a datapath when instructing to use kilosort's whitening matrix."
        w, unprocessed_channels = load_ks_whitening_matrix(dp)
        return cp.array(w), unprocessed_channels

    # get covariance matrix across rows (each row is an observation)
    x_cov = cp.cov(x, rowvar=1) 
    return cov_to_whitening_matrix(x_cov, nRange)

def load_ks_whitening_matrix(dp, return_full=False):
    """
    Return kilosort whitening matrix
    if return_full: also with missing channels (not preprocessed) replaced with 0s.

    Arguments:
    - dp: str/Path, datapath
    - return_full: bool, whether to return full whitening matrix
                   (adding arrays of 0s off-diagonal and 1 on-diagonal for channels skipped by kilosort)

    Returns:
    - kilosort whitening matrix (nchans, nchans) with nchans <= 384
    - channels_mask: bool array, channels not processed by kilosort
    """

    probe_version = read_metadata(dp)['probe_version']
    local_cm = chan_map(dp, probe_version='local')[:,0]
    full_cm = chan_map(dp, probe_version=probe_version)[:,0]

    channels_mask = np.isin(full_cm, local_cm)
    channels_2D_mask = channels_mask[:,None] & channels_mask[None,:]

    local_Wrot = np.load(Path(dp) / 'whitening_mat.npy')

    if not return_full:
        return local_Wrot, ~channels_mask

    full_Wrot = np.zeros((full_cm.shape[0], full_cm.shape[0])) # * np.nan
    full_Wrot[np.nonzero(channels_2D_mask)] = local_Wrot.ravel()
    full_Wrot[np.nonzero(~channels_mask)[0], np.nonzero(~channels_mask)[0]] = 1

    return full_Wrot, ~channels_mask



def approximated_whitening_matrix(memmap_f, Wrot_path, whiten_range,
        NT, Nbatch, NTbuff, ntb, nSkipCov, n_channels, channels_to_process,
        f_high, fs, again=False, verbose=False):
    """
    Rather than computing the true whitening matrix from a signal x,
    approximate whitening matrix from the approximated covariance between the channels
    (median of cov_all, where cov_all stores the covariance for a subset of data batches).

    Arguments:
    - memmap_f: memory mapped file, n_samples x n_channels (whitening across channels)
    - Wrot_path: path to save whitening matrix to
    - whiten_range: int, range of channels to consider to compute local covaraince/whitening matrix
    - NT, Nbatch, NTbuff, ntb: ints, size of data batches etc (see npyx.inout.preprocess_binary_file)
    - nSkipCov: int, use every nSkipCov batches to approximate covariance -> whitening matrix
    - channels_to_process: arrya of channels to use
    - f_high: float, high pass filtered frequency
    - fs: int, sampling frequency
    - again: bool, whether to recompute whitening matrix
    - verbose: bool, whether to print extra info
    """
    if Wrot_path.exists() and not again:
        return cp.load(Wrot_path)
    else:
        # cov = np.zeros((n_channels,n_channels))
        # for ibatch in tqdm(range(0, Nbatch, nSkipCov), desc="Computing whitening matrix..."):
        #     i = max(0, NT * ibatch - ntb)
        #     # raw_data is nsamples x NchanTOT
        #     buff = memmap_f[i:i + NTbuff]
        #     if nsampcurr < NTbuff:
        #             buff = np.concatenate(
        #                 (buff, np.tile(buff[nsampcurr - 1], (NTbuff - nsampcurr, 1))), axis=0)
        #     buff = cp.asarray(buff, dtype=np.float32)
        #     datr = gpufilter(buff, fs=fs, fshigh=f_high, chanMap=channels_to_process)
        #     cov = cov + cp.cov(datr.T) #cp.dot(datr.T, datr) / NT  # sample covariance
        # cov = cov / max(ceil((Nbatch - 1) / nSkipCov), 1) # mean covariance

        # Nchan is obtained after the bad channels have been removed
        nbatches_cov = np.arange(0, Nbatch, nSkipCov).size
        cov_all = cp.zeros((nbatches_cov, n_channels, n_channels))

        for icc, ibatch in enumerate(tqdm(range(0, Nbatch, nSkipCov), desc="Computing the whitening matrix")):
            i = max(0, NT * ibatch - ntb)
            # WARNING: we no longer use Fortran order, so raw_data is nsamples x NchanTOT
            buff = memmap_f[i:i + NTbuff]
            assert buff.shape[0] > buff.shape[1]
            assert buff.flags.c_contiguous
            nsampcurr = buff.shape[0]
            if nsampcurr < NTbuff:
                buff = np.concatenate(
                    (buff, np.tile(buff[nsampcurr - 1], (NTbuff - nsampcurr, 1))), axis=0)
            buff_g = cp.asarray(buff, dtype=np.float32)
            # high pass filter
            datr = gpufilter(buff_g, fs=fs, fshigh=f_high, chanMap=channels_to_process)

            # remove buffers on either side of the data batch
            datr = datr[ntb: NT + ntb]
            assert datr.flags.c_contiguous
            datr_centered = datr - datr.mean(1)[:,None]
            cov_all[icc, :, :] = cp.dot(datr_centered.T, datr_centered) / datr_centered.shape[0]
        cov = cp.median(cov_all, axis=0) # ensures outlier batches do not alter the final covariance
        Wrot = cov_to_whitening_matrix(cov, nRange=whiten_range)
        cp.save(Wrot_path, Wrot)

    condition_number = np.linalg.cond(cp.asnumpy(Wrot))
    if verbose: print(f"Computed the whitening matrix cond = {condition_number}.")
    if condition_number > 50:
        print("high conditioning of the whitening matrix can result in noisy and poor results")

    return cp.asarray(Wrot) # ensure runs on gpu (cupy)

def cov_to_whitening_matrix(cov, nRange):
    if nRange is None:
        return zca_whitening(cov)
    return zca_whitening_local(cov, nRange)

def zca_whitening(cov):
    # function Wrot = whiteningFromCovariance(cov)
    # takes as input the covariance matrix cov of channel pairwise correlations
    # outputs a symmetric rotation matrix (also Nchan by Nchan) that rotates
    # the data onto uncorrelated, unit-norm axes

    # good reference https://theclevermachine.wordpress.com/2013/03/30/the-statistical-whitening-transform/

    # covariance eigendecomposition (same as svd for positive-definite matrix)
    E, D, _ = cp.linalg.svd(cov)
    D[D<0]=0
    eps = 1e-6
    Di = cp.diag(1. / (D + eps) ** .5)
    W = cp.dot(cp.dot(E, Di), E.T)  # this is the symmetric whitening matrix (ZCA transform)
    return W


def zca_whitening_local(cov, nRange):
    # function to perform local whitening of channels
    # cov is a matrix of Nchan by Nchan correlations
    # nRange is the number of nearest channels to consider
    # assuming that channels (as in the covariance matrix) are already ordered by distance

    nchans = cov.shape[0]
    chans=np.arange(nchans)

    W = cp.zeros((nchans, nchans))
    for i in range(nchans):
        
        # take the closest channels to the primary channel.
        # First channel in this list will always be the primary channel.
        distances = np.abs(chans-i)
        closest = np.argsort(distances)[:nRange+1]

        Wlocal = cp.asnumpy(zca_whitening(cov[np.ix_(closest, closest)]))
        # the first column of wrot0 is the whitening filter for the primary channel
        W[closest, i] = Wlocal[:, 0]

    return W

def whitening_matrix_cpu(x, epsilon=1e-18, nRange=None):
    """
    wmat = whitening_matrix(dat, fudge=1e-18)
    Compute the whitening matrix using ZCA.
        - dat is a matrix nsamples x nchannels
    Apply using np.dot(dat,wmat)
    Adapted from phy
    Arguments:
        - x: 2D array, axis 1 spans time, axis 0 observations (e.g. channel).
             Multiplying by the whitening matrix whitens across obervations.
        - epsilon: small value added to diagonal to regularize D
        - nRange: if integer, number of channels to locally compute whitening filter (more robust to noise) | Default None
    """
    assert x.ndim == 2
    nrows, ncols = x.shape
    x_cov = np.cov(x, rowvar=1) # get covariance matrix across rows (each row is an observation)
    assert x_cov.shape == (nrows, nrows)
    if nRange is None:
        d, v = np.linalg.eigh(x_cov) # covariance eigendecomposition (same as svd for positive-definite matrix)
        d[d<0]=0 # handles calculation innacurracies leading to very tiny negative values instead of tiny positive values
        d = np.diag(1. / np.sqrt(d + epsilon))
        w = np.dot(np.dot(v, d), v.T) # V * D * V': ZCA transform
        return w
    ##TODO make that fast with numba
    rows=np.arange(nrows)
    w=np.zeros((nrows,nrows))
    for i in range(x_cov.shape[0]):
        closest=np.sort(rows[np.argsort(np.abs(rows-i))[:nRange+1]])
        span=slice(closest[0],closest[-1]+1)
        x_cov_local=x_cov[span,span]
        d, v = np.linalg.eigh(x_cov_local) # covariance eigendecomposition (same as svd for positive-definite matrix)
        d[d<0]=0 # handles calculation innacurracies leading to very tiny negative values instead of tiny positive values
        d = np.diag(1. / np.sqrt(d + epsilon))
        w[i,span] = np.dot(np.dot(v, d), v.T)[:,0] # V * D * V': ZCA transform
    return w

def whiten_multimethod(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

#%% Filtering
# Most code borrowed from pykilosort (mouseland)

def med_substract(x, axis=0, nRange=None):
    '''Median substract along axis 0
    (for instance, channels should be axis 0 and time axis 1 to median substract across channels)'''
    assert axis in [0,1]
    if nRange is None:
        return x-np.median(x, axis=axis) if axis==0 else x-np.median(x, axis=axis)[:,np.newaxis]
    n_points=x.shape[axis]
    x_local_med=np.zeros(x.shape)
    points=np.arange(n_points)
    for xi in range(n_points):
        closest=np.sort(points[np.argsort(np.abs(points-xi))[:nRange+1]])
        if axis==0: x_local_med[xi,:]=np.median(x[closest,:], axis=axis)
        elif axis==1: x_local_med[:,xi]=np.median(x[:,closest], axis=axis)
    return x-x_local_med

def bandpass_filter(rate=None, low=None, high=None, order=1):
    """Butterworth bandpass filter."""
    assert low is not None or high is not None
    if low is not None and high is not None: assert low < high
    assert order >= 1
    if high is not None and low is not None:
        return sgnl.butter(order, (low,high), 'bandpass', fs=rate)
    elif low is not None:
        return sgnl.butter(order, low, 'lowpass', fs=rate)
    elif high is not None:
        return sgnl.butter(order, high, 'highpass', fs=rate)

def apply_filter(x, filt, axis=0, forward=True, backward=True):
    """Apply a filter to an array, bidirectionally."""
    x = np.asarray(x)
    if x.shape[axis] == 0:
        return x
    b, a = filt

    if forward and backward:
        # probably faster to use filtfilt than lfilter twice
        return sgnl.filtfilt(b, a, x, axis=axis)

    elif forward:
        return sgnl.lfilter(b, a, x, axis=axis)

    else:
        assert backward and not forward # precaution
        x = np.flip(x, axis)
        x = sgnl.lfilter(b, a, x, axis=axis)
        return np.flip(x, axis)

def gpufilter(buff, fs=None, fslow=None, fshigh=None, order=3,
             car=False, forward=True, backward=True, ret_numpy=False):
    # filter this batch of data after common average referencing with the
    # median
    # buff is timepoints by channels
    # chanMap are indices of the channels to be kep
    # params.fs and params.fshigh are sampling and high-pass frequencies respectively
    # if params.fslow is present, it is used as low-pass frequency (discouraged)

    dataRAW = buff  # .T  # NOTE: we no longer use Fortran order upstream
    assert dataRAW.flags.c_contiguous
    assert dataRAW.ndim == 2
    assert dataRAW.shape[0] > dataRAW.shape[1]
    assert dataRAW.ndim == 2
    assert forward or backward, "You should either filter forward or backward."

    # subtract the mean from each channel
    # Maxime: I would use the median, but kilosort uses the mean
    dataRAW = dataRAW - cp.mean(dataRAW, axis=0)
    assert dataRAW.ndim == 2

    # CAR, common average referencing by median
    if car:
        # subtract median across channels
        dataRAW = dataRAW - cu_median(dataRAW, axis=1)[:, np.newaxis]

    # set up the parameters of the filter
    filter_params = get_filter_params(fs, fshigh=fshigh, fslow=fslow, order=order)

    # next four lines should be equivalent to filtfilt (which cannot be
    # used because it requires float64)
    if forward:
        dataRAW = lfilter(*filter_params, dataRAW, axis=0)  # causal forward filter
    if backward:
        # Maxime note: I do not understand why pykilosort folks
        # did not do the same as MATLAB kilosort (i.e. simply revesing dataRAW)
        dataRAW = lfilter(*filter_params, dataRAW, axis=0, reverse=True)  # backward
    
    if ret_numpy:
        dataRAW = cp.asnumpy(dataRAW)

    return dataRAW

def get_filter_params(fs, fshigh=None, fslow=None, order=3):
    # Wn should be the cutoff frequency in fraction of the Nyquist frequency:
    # fc / (fs / 2) = fc / fs * 2
    if fslow and fslow < fs / 2:
        return butter(order, (2 * fshigh / fs, 2 * fslow / fs), 'bandpass')
    else:
        return butter(order, fshigh / fs * 2, 'high')

def make_kernel(kernel, name, **const_arrs):
    """Compile a kernel and pass optional constant ararys."""
    mod = cp_core.core.compile_with_cache(kernel, prepend_cupy_headers=False)
    b = cp_core.core.memory_module.BaseMemory() 
    # Pass constant arrays.
    for n, arr in const_arrs.items():
        b.ptr = mod.get_global_var(n)
        p = cp_core.core.memory_module.MemoryPointer(b, 0)
        p.copy_from_host(arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes)
    return mod.get_function(name)


def get_lfilter_kernel(N, isfortran, reverse=False):
    order = 'f' if isfortran else 'c'
    return dedent("""
    const int N = %d;
    __constant__ float a[N + 1];
    __constant__ float b[N + 1];
    __device__ int get_idx_f(int n, int col, int n_samples, int n_channels) {
        return n_samples * col + n;  // Fortran order.
    }
    __device__ int get_idx_c(int n, int col, int n_samples, int n_channels) {
        return n * n_channels + col;  // C order.
    }
    // LTI IIR filter implemented using a difference equation.
    // see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
    extern "C" __global__ void lfilter(
            const float* x, float* y, const int n_samples, const int n_channels){
        // Initialize the state variables.
        float d[N + 1];
        for (int k = 0; k <= N; k++) {
            d[k] = 0.0;
        }
        float xn = 0.0;
        float yn = 0.0;
        int idx = 0;
        // Column index.
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        // IMPORTANT: avoid out of bounds memory accesses, which cause no errors but weird bugs.
        if (col >= n_channels) return;
        for (int n = 0; n < n_samples; n++) {
            idx = get_idx_%s(%s, col, n_samples, n_channels);
            // Load the input element.
            xn = x[idx];
            // Compute the output element.
            yn = (b[0] * xn + d[0]) / a[0];
            // Update the state variables.
            for (int k = 0; k < N; k++) {
                d[k] = b[k + 1] * xn - a[k + 1] * yn + d[k + 1];
            }
            // Update the output array.
            y[idx] = yn;
        }
    }
    """ % (N, order, 'n' if not reverse else 'n_samples - 1 - n'))


def _get_lfilter_fun(b, a, is_fortran=True, axis=0, reverse=False):
    assert axis == 0, "Only filtering along the first axis is currently supported."

    b = np.atleast_1d(b).astype(np.float32)
    a = np.atleast_1d(a).astype(np.float32)
    N = max(len(b), len(a))
    if len(b) < N:
        b = np.pad(b, (0, (N - len(b))), mode='constant')
    if len(a) < N:
        a = np.pad(a, (0, (N - len(a))), mode='constant')
    assert len(a) == len(b)
    kernel = get_lfilter_kernel(N - 1, is_fortran, reverse=reverse)

    lfilter = make_kernel(kernel, 'lfilter', b=b, a=a)

    return lfilter


def _apply_lfilter(lfilter_fun, arr):
    assert isinstance(arr, cp.ndarray)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    n_samples, n_channels = arr.shape

    block = (min(128, n_channels),)
    grid = (int(ceil(n_channels / float(block[0]))),)

    arr = cp.asarray(arr, dtype=np.float32)
    y = cp.zeros_like(arr, order='F' if arr.flags.f_contiguous else 'C', dtype=arr.dtype)

    assert arr.dtype == np.float32
    assert y.dtype == np.float32
    assert arr.shape == y.shape

    lfilter_fun(grid, block, (arr, y, int(y.shape[0]), int(y.shape[1])))
    return y


def lfilter(b, a, arr, axis=0, reverse=False):
    """Perform a linear filter along the first axis on a GPU array."""
    lfilter_fun = _get_lfilter_fun(
        b, a, is_fortran=arr.flags.f_contiguous, axis=axis, reverse=reverse)
    return _apply_lfilter(lfilter_fun, arr)


# GPU FFT-based convolution
# -------------------------

def _clip(x, a, b):
    return max(a, min(b, x))


def pad(fcn_convolve):
    @wraps(fcn_convolve)
    def function_wrapper(x, b, axis=0, **kwargs):
        # add the padding to the array
        xsize = x.shape[axis]
        if 'pad' in kwargs and kwargs['pad']:
            npad = b.shape[axis] // 2
            padd = cp.take(x, cp.arange(npad), axis=axis) * 0
            if kwargs['pad'] == 'zeros':
                x = cp.concatenate((padd, x, padd), axis=axis)
            if kwargs['pad'] == 'constant':
                x = cp.concatenate((padd * 0 + cp.mean(x[:npad]), x, padd + cp.mean(x[-npad:])),
                                   axis=axis)
            if kwargs['pad'] == 'flip':
                pad_in = cp.flip(cp.take(x, cp.arange(1, npad + 1), axis=axis), axis=axis)
                pad_out = cp.flip(cp.take(x, cp.arange(xsize - npad - 1, xsize - 1),
                                          axis=axis), axis=axis)
                x = cp.concatenate((pad_in, x, pad_out), axis=axis)
        # run the convolution
        y = fcn_convolve(x, b, **kwargs)
        # remove padding from both arrays (necessary for x ?)
        if 'pad' in kwargs and kwargs['pad']:
            # remove the padding
            y = cp.take(y, cp.arange(npad, x.shape[axis] - npad), axis=axis)
            x = cp.take(x, cp.arange(npad, x.shape[axis] - npad), axis=axis)
            assert xsize == x.shape[axis]
            assert xsize == y.shape[axis]
        return y
    return function_wrapper


def convolve_cpu(x, b):
    """CPU convolution based on scipy.signal."""
    x = np.asarray(x)
    b = np.asarray(b)
    if b.ndim == 1:
        b = b[:, np.newaxis]
    assert b.ndim == 2
    y = sgnl.convolve(x, b, mode='same')
    return y


@pad
def convolve_gpu_direct(x, b, **kwargs):
    """Straight GPU FFT-based convolution that fits in memory."""
    if not isinstance(x, cp.ndarray):
        x = np.asarray(x)
    if not isinstance(x, cp.ndarray):
        b = np.asarray(b)
    assert b.ndim == 1
    n = x.shape[0]
    xf = cp.fft.rfft(x, axis=0, n=n)
    if xf.shape[0] > b.shape[0]:
        bp = cp.pad(b, (0, n - b.shape[0]), mode='constant')
        bp = cp.roll(bp, - b.size // 2 + 1)
    else:
        bp = b
    bf = cp.fft.rfft(bp, n=n)[:, np.newaxis]
    y = cp.fft.irfft(xf * bf, axis=0, n=n)
    return y


DEFAULT_CONV_CHUNK = 10_000


@pad
def convolve_gpu_chunked(x, b, pad='flip', nwin=DEFAULT_CONV_CHUNK, ntap=500, overlap=2000):
    """Chunked GPU FFT-based convolution for large arrays.
    This memory-controlled version splits the signal into chunks of n samples.
    Each chunk is tapered in and out, the overlap is designed to get clear of the taper
    splicing of overlaping chunks is done in a cosine way.
    param: pad None, 'zeros', 'constant', 'flip'
    """
    x = cp.asarray(x)
    b = cp.asarray(b)
    assert b.ndim == 1
    n = x.shape[0]
    assert overlap >= 2 * ntap
    # create variables, the gain is to control the splicing
    y = cp.zeros_like(x)
    gain = cp.zeros(n)
    # compute tapers/constants outside of the loop
    taper_in = (-cp.cos(cp.linspace(0, 1, ntap) * cp.pi) / 2 + 0.5)[:, cp.newaxis]
    taper_out = cp.flipud(taper_in)
    assert b.shape[0] < nwin < n
    # this is the convolution wavelet that we shift to be 0 lag
    bp = cp.pad(b, (0, nwin - b.shape[0]), mode='constant')
    bp = cp.roll(bp, - b.size // 2 + 1)
    bp = cp.fft.rfft(bp, n=nwin)[:, cp.newaxis]
    # this is used to splice windows together: cosine taper. The reversed taper is complementary
    scale = cp.minimum(cp.maximum(0, cp.linspace(-0.5, 1.5, overlap - 2 * ntap)), 1)
    splice = (-cp.cos(scale * cp.pi) / 2 + 0.5)[:, cp.newaxis]
    # loop over the signal by chunks and apply convolution in frequency domain
    first = 0
    while True:
        first = min(n - nwin, first)
        last = min(first + nwin, n)
        # the convolution
        x_ = cp.copy(x[first:last, :])
        x_[:ntap] *= taper_in
        x_[-ntap:] *= taper_out
        x_ = cp.fft.irfft(cp.fft.rfft(x_, axis=0, n=nwin) * bp, axis=0, n=nwin)
        # this is to check the gain of summing the windows
        tt = cp.ones(nwin)
        tt[:ntap] *= taper_in[:, 0]
        tt[-ntap:] *= taper_out[:, 0]
        # the full overlap is outside of the tapers: we apply a cosine splicing to this part only
        if first > 0:
            full_overlap_first = first + ntap
            full_overlap_last = first + overlap - ntap
            gain[full_overlap_first:full_overlap_last] *= (1. - splice[:, 0])
            gain[full_overlap_first:full_overlap_last] += tt[ntap:overlap - ntap] * splice[:, 0]
            gain[full_overlap_last:last] = tt[overlap - ntap:]
            y[full_overlap_first:full_overlap_last] *= (1. - splice)
            y[full_overlap_first:full_overlap_last] += x_[ntap:overlap - ntap] * splice
            y[full_overlap_last:last] = x_[overlap - ntap:]
        else:
            y[first:last, :] = x_
            gain[first:last] = tt
        if last == n:
            break
        first += nwin - overlap
    return y


def convolve_gpu(x, b, **kwargs):
    n = x.shape[0]
    # Default chunk size : N samples along the first axis, the one to be chunked and over
    # which to compute the convolution.
    nwin = kwargs.get('nwin', DEFAULT_CONV_CHUNK)
    assert nwin >= 0
    if n <= nwin or nwin == 0:
        return convolve_gpu_direct(x, b)
    else:
        nwin = max(nwin, b.shape[0] + 1)
        return convolve_gpu_chunked(x, b, **kwargs)


def svdecon(X, nPC0=None):
    """
    Input:
    X : m x n matrix
    Output:
    X = U*S*V'
    Description:
    Does equivalent to svd(X,'econ') but faster
        Vipin Vijayan (2014)
    """

    m, n = X.shape

    nPC = nPC0 or min(m, n)

    if m <= n:
        C = cp.dot(X, X.T)
        D, U = cp.linalg.eigh(C, 'U')

        ix = cp.argsort(np.abs(D))[::-1]
        d = D[ix]
        U = U[:, ix]
        d = d[:nPC]
        U = U[:, :nPC]

        V = cp.dot(X.T, U)
        s = cp.sqrt(d)
        V = V / s.T
        S = cp.diag(s)
    else:
        C = cp.dot(X.T, X)
        D, V = cp.linalg.eigh(C)

        ix = cp.argsort(cp.abs(D))[::-1]
        d = D[ix]
        V = V[:, ix]

        # convert evecs from X'*X to X*X'. the evals are the same.
        U = cp.dot(X, V)
        s = cp.sqrt(d)
        U = U / s.T
        S = cp.diag(s)

    return U, S, V


def svdecon_cpu(X):
    U, S, V = np.linalg.svd(cp.asnumpy(X))
    return U, np.diag(S), V


def free_gpu_memory():
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


# Work around CuPy bugs and limitations
# -----------------------------------------------------------------------------

def cu_mean(x, axis=0):
    if x.ndim == 1:
        return cp.mean(x) if x.size else cp.nan
    else:
        s = list(x.shape)
        del s[axis]
        return (
            cp.mean(x, axis=axis) if x.shape[axis] > 0
            else cp.zeros(s, dtype=x.dtype, order='F'))


def cu_median(a, axis=0):
    """Compute the median of a CuPy array on the GPU."""
    a = cp.asarray(a)

    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    part = cp.partition(a, kth, axis=axis)

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)
    
    indexer = tuple(indexer) # compatibility with cupy 11+ (MB)

    return cp.mean(part[indexer], axis=axis)


def cu_var(x):
    return cp.var(x, ddof=1) if x.size > 0 else cp.nan


def cu_ones(shape, dtype=None, order=None):
    # HACK: cp.ones() has no order kwarg at the moment !
    x = cp.zeros(shape, dtype=dtype, order=order)
    x.fill(1)
    return x


def cu_zscore(a, axis=0):
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=0)
    return (a - mns) / sstd

#%% Borrowed from IBL code base (Olivier Winter)

def adc_realign(data, version=1):
    f"""
    Realign Neuropixels data according to the small sub-sampling frequency shifts due to serial digitalization.

    Arguments:
    - data: n_samples x n_channels
    - version: 1 or 2, Neuropixels version

    {fshift.__doc__}

    {adc_shifts.__doc__}
    """

    assert data.ndim == 2
    assert data.shape[1] == 384, "Only implemented for 384 channels atm."
    assert version in [1,1.,2,2.]

    sample_shift, adc = adc_shifts(version)

    shifted_data = fshift(data.T, sample_shift, 1, None).T

    return shifted_data

def fshift(w, s, axis=-1, ns=None):
    """
    Shifts a 1D or 2D signal in frequency domain, to allow for accurate non-integer shifts
    :param w: input signal (n_channels, n_samples) (if complex, need to provide ns too)
    :param s: shift in samples, positive shifts forward
    :param axis: axis along which to shift (last axis by default)
    :param axis: axis along which to shift (last axis by default)
    :param ns: if a rfft frequency domain array is provided, give a number of samples as there
     is an ambiguity
    :return: w
    """

    # create a vector that contains a 1 sample shift on the axis
    ns = ns or w.shape[axis]
    shape = np.array(w.shape) * 0 + 1
    shape[axis] = ns
    dephas = np.zeros(shape)
    np.put(dephas, 1, 1)
    dephas = sp.fft.rfft(dephas, axis=axis)
    # fft the data along the axis and the dephas
    do_fft = np.invert(np.iscomplexobj(w))
    if do_fft:
        W = sp.fft.rfft(w, axis=axis)
    else:
        W = w
    # if multiple shifts, broadcast along the other dimensions, otherwise keep a single vector
    if not np.isscalar(s):
        s_shape = np.array(w.shape)
        s_shape[axis] = 1
        s = s.reshape(s_shape)
    # apply the shift (s) to the fft angle to get the phase shift and broadcast
    W *= np.exp(1j * np.angle(dephas) * s)
    if do_fft:
        W = np.real(sp.fft.irfft(W, ns, axis=axis))
        W = W.astype(w.dtype)
    return W

def adc_shifts(version=1):
    """
    The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
    The ADC to channel mapping is done per odd and even channels:
    ADC1: ch1, ch3, ch5, ch7...
    ADC2: ch2, ch4, ch6....
    ADC3: ch33, ch35, ch37...
    ADC4: ch34, ch36, ch38...
    Therefore, channels 1, 2, 33, 34 get sample at the same time. I hope this is more or
    less clear. In 1.0, it is similar, but there we have 32 ADC that sample each 12 channels."
    - Nick on Slack after talking to Carolina - ;-)

    Arguments:
    - version: 1 or 2, Neuropixels 1.0 or 2.0
    """
    n_channels=384

    if version == 1:
        adc_channels = 12
        # version 1 uses 32 ADC that sample 12 channels each
    elif np.floor(version) == 2:
        # version 2 uses 24 ADC that sample 16 channels each
        adc_channels = 16
    adc = np.floor(np.arange(n_channels) / (adc_channels * 2)) * 2 + np.mod(np.arange(n_channels), 2)
    sample_shift = np.zeros_like(adc)
    for a in adc:
        sample_shift[adc == a] = np.arange(adc_channels) / adc_channels
    return sample_shift, adc

def kfilt(x, ntr_pad=0, ntr_tap=None, lagc=300, butter_kwargs=None):
    """
    Applies a butterworth filter on the 0-axis with tapering / padding
    :param x: the input array to be filtered. dimension, the filtering is considering
     n_channels x n_samples
    :param collection:
    :param ntr_pad: traces added to each side (mirrored)
    :param ntr_tap: n traces for apodization on each side
    :param lagc: window size for time domain automatic gain control (no agc otherwise)
    :param butter_kwargs: filtering Arguments: defaults: {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    :return:
    """
    if butter_kwargs is None:
        butter_kwargs = {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}

    nx, nt = x.shape

    # lateral padding left and right
    ntr_pad = int(ntr_pad)
    ntr_tap = ntr_pad if ntr_tap is None else ntr_tap
    nxp = nx + ntr_pad * 2

    # apply agc and keep the gain in handy
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=1.0)
    if ntr_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        xf = np.r_[np.flipud(xf[:ntr_pad]), xf, np.flipud(xf[-ntr_pad:])]
    if ntr_tap > 0:
        taper = fcn_cosine([0, ntr_tap])(np.arange(nxp))  # taper up
        taper *= 1 - fcn_cosine([nxp - ntr_tap, nxp])(np.arange(nxp))   # taper down
        xf = xf * taper[:, np.newaxis]
    sos = sp.signal.butter(**butter_kwargs, output='sos')
    xf = sp.signal.sosfiltfilt(sos, xf, axis=0)

    if ntr_pad > 0:
        xf = xf[ntr_pad:-ntr_pad, :]
    return xf / gain

def agc(x, wl=.5, si=.002, epsilon=1e-8):
    """
    Automatic gain control
    w_agc, gain = agc(w, wl=.5, si=.002, epsilon=1e-8)
    such as w_agc / gain = w
    :param x: seismic array (sample last dimension)
    :param wl: window length (secs)
    :param si: sampling interval (secs)
    :param epsilon: whitening (useful mainly for synthetic data)
    :return: AGC data array, gain applied to data
    """
    ns_win = np.round(wl / si / 2) * 2 + 1
    w = np.hanning(ns_win)
    w /= np.sum(w)
    gain = ibl_convolve(np.abs(x), w, mode='same')
    gain += (np.sum(gain, axis=1) * epsilon / x.shape[-1])[:, np.newaxis]
    gain = 1 / gain
    return x * gain, gain

def fcn_cosine(bounds):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :return: lambda function
    """
    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2
    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func

def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y

def ibl_convolve(x, w, mode='full'):
    """
    Frequency domain convolution along the last dimension (2d arrays)
    Will broadcast if a matrix is convolved with a vector
    :param x:
    :param w:
    :return: convolution
    """
    nsx = x.shape[-1]
    nsw = w.shape[-1]
    ns = ns_optim_fft(nsx + nsw)
    x_ = np.concatenate((x, np.zeros([*x.shape[:-1], ns - nsx], dtype=x.dtype)), axis=-1)
    w_ = np.concatenate((w, np.zeros([*w.shape[:-1], ns - nsw], dtype=w.dtype)), axis=-1)
    xw = np.real(np.fft.irfft(np.fft.rfft(x_, axis=-1) * np.fft.rfft(w_, axis=-1), axis=-1))
    xw = xw[..., :(nsx + nsw)]  # remove 0 padding
    if mode == 'full':
        return xw
    elif mode == 'same':
        first = int(np.floor(nsw / 2)) - ((nsw + 1) % 2)
        last = int(np.ceil(nsw / 2)) + ((nsw + 1) % 2)
        return xw[..., first:-last]

def ns_optim_fft(ns):
    """
    Gets the next higher combination of factors of 2 and 3 than ns to compute efficient ffts
    :param ns:
    :return: nsoptim
    """
    p2, p3 = np.meshgrid(2 ** np.arange(25), 3 ** np.arange(15))
    sz = np.unique((p2 * p3).flatten())
    return sz[np.searchsorted(sz, ns)]


from npyx.inout import chan_map, read_metadata