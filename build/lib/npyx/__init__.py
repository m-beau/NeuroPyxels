# -*- coding: utf-8 -*-

from . import utils, io, gl, spk_t, spk_wvf, corr, plot, behav, merger, stats

from .utils import *
from .io import *
from .gl import *
from .spk_t import *
from .spk_wvf import *
from .corr import *
from .stats import *
from .plot import *
from .behav import *
from .merger import *
from .circuitProphyler import *

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

__version__ = '2.1.0'

print(f'npyx version {__version__} imported.')