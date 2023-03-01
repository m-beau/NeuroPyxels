import traceback

import numpy as np

import npyx
from npyx.inout import get_npix_sync
from npyx.gl import get_units, read_metadata
from npyx.spk_t import ids, trn, trn_filtered
from npyx.spk_wvf import wvf, wvf_dsmatch, get_peak_chan, templates
from npyx.corr import ccg
from npyx.plot import plot_acg, plot_ccg, plot_wvf, plot_raw

prefix = "\033[34;1m--- "
red_prefix = "\033[91;1m--- "
suffix = "\033[0m" 

def test_npyx(dp, raise_error=False):
    """
    Function for unit testing of npyx core functions.

    Arguments:
    - dp: path to Neuropixels data directory
          (must be compatible with phy, e.g. kilosort output)
    - raise_error: bool, whether to pause and raise error when test fails
                   (allows to enter pydebug interactive mode,
                   given that %pdb was set in notebook/ipython session)
    """
    print(f"{prefix}npyx version {npyx.__version__} unit testing initiated, on directory {dp}...{suffix}\n")

    test_function(read_metadata, raise_error, dp=dp)

    test_function(get_npix_sync, raise_error, dp=dp)

    units = test_function(get_units, raise_error, True, dp=dp)
    if units is None:
        raise ValueError("Something went really wrong, get_units did not run. Fix code and try again before testing next function.")
    u, u1 = units[np.random.randint(0, len(units)-1, 2)]

    test_function(ids, raise_error, dp=dp, unit=u, again=1)

    test_function(trn, raise_error, dp=dp, unit=u, again=1)

    test_function(trn_filtered, raise_error, dp=dp, unit=u, plot_debug=True, again=1)

    test_function(wvf, raise_error, dp=dp, u=u, again=1)

    test_function(wvf_dsmatch, raise_error, dp=dp, u=u, plot_debug=True, again=1, verbose=True)

    test_function(get_peak_chan, raise_error, dp=dp, unit=u, again=1)

    test_function(templates, raise_error, dp=dp, u=u)
    
    test_function(ccg, raise_error, dp=dp, U=[u,u1], bin_size=0.2, win_size=40, again=1)

    test_function(plot_wvf, raise_error, dp=dp, u=u, again=1, use_dsmatch=True)

    test_function(plot_acg, raise_error, dp=dp, unit=u, again=1)
    
    test_function(plot_ccg, raise_error, dp=dp, units=[u,u1], as_grid=True, again=1)

    test_function(plot_raw, raise_error, dp=dp, times=[0.1,0.15], channels=list(range(50)), again=1)


def test_function(fun, raise_error=False, ret=False, **kwargs):
    """
    Function to test a function with rich printed information.

    Arguments:
    - fun: function to test
    - ret: bool, whether to return output of fun
    - raise_error: bool, whether to pause and raise error when test fails
                   (allows to enter pydebug interactive mode,
                   given that %pdb was set in notebook/ipython session)
    - kwargs: parameters to fun
    """
    try:
        r = fun(**kwargs)
        print(f"{prefix}Successfully ran '{fun.__name__}' from {fun.__module__}.{suffix}")
        if ret: return r
    except Exception as err:
        print(f"{red_prefix}Failed to run '{fun.__name__}' from {fun.__module__} with the following error:{suffix}")
        e = traceback.format_exc()
        print(e)
        if raise_error:
            print(("\033[34;1mIf you wish to enter interactive debugging, "
            "make sure to have run the magic command %pdb in your notebook/ipython session.\033[0m"))
            raise FailedNpyxTest().with_traceback(err.__traceback__) from err

class FailedNpyxTest(Exception):
    pass