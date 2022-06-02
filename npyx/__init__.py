# -*- coding: utf-8 -*-

from . import inout, utils, gl, spk_t, spk_wvf, corr, plot, behav, merger, stats

from .utils import *
from .inout import *
from .gl import *
from .spk_t import *
from .spk_wvf import *
from .corr import *
from .stats import *
from .plot import *
from .behav import *
from .merger import *
from .circuitProphyler import *
from .feat import *
from .info import *
from .h5 import *
from .testing import *

__doc__="""

npyx submodules:
 .utils
 .io
 .gl
 .spk_t
 .spk_wvf
 .corr
 .stats
 .plot
 .behav
 .merger
 .circuitProphyler
 .feat
 .h5
"""

__version__ = '2.4.7'

print(f'npyx version {__version__} imported.')