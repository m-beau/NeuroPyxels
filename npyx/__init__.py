# -*- coding: utf-8 -*-

from . import utils, inout, gl, spk_t, spk_wvf, corr, stats, plot,\
              behav, merger, circuitProphyler, feat, metrics,\
              info, model, h5, testing, ml, datasets

from .c4 import dataset_init, misc

try:
    import torch
    from .c4 import acg_augmentations, acg_vs_firing_rate, dl_utils, encode_features,\
                    monkey_dataset_init, plots_functions, predict_cell_types, run_baseline_classifier,\
                    run_deep_classifier, dl_transforms, waveform_augmentations, run_cell_types_classifier
    C4_IMPORTED = True
except ImportError:
    # Do not import extra C4 functionality in the main namespace if torch is not installed.
    # Also affects import printing.
    C4_IMPORTED = False
    pass

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

__version__ = "4.0.2"

npyx_build = "npyx[c4]" if C4_IMPORTED else "npyx"

print(f"\n\033[32;1m{npyx_build} version {__version__} imported.\033[0m")
