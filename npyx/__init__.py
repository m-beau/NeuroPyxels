# -*- coding: utf-8 -*-

from . import utils, inout, gl, spk_t, spk_wvf, corr, stats, plot,\
              behav, merger, circuitProphyler, feat, metrics,\
              info, model, h5, testing, ml, datasets

from .c4 import acg_augmentations, dataset_init, dataset_init, dl_utils, predict_cell_types, run_deep_classifier

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
from .metrics import *
from .info import *
from .model import *
from .h5 import *
from .ml import *
from .testing import *

__doc__ = """

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

__version__ = "3.0.2"

print(f"\n\033[32;1mnpyx version {__version__} imported.\033[0m")
