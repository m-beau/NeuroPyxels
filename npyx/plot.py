# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os
import os.path as op; opj=op.join
import subprocess
import sys
from pathlib import Path

import pickle as pkl

import numpy as np

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from cmcrameri import cm as cmcr
cmcr=cmcr.__dict__
from IPython.core.display import HTML,display


mpl.rcParams['figure.dpi']=100

import holoviews as hv
import bokeh as bk
import hvplot.pandas

import seaborn as sns

from npyx.utils import phyColorsDic, npa, zscore, isnumeric, assert_iterable
from npyx.stats import fractile_normal, fractile_poisson

from npyx.io import read_spikeglx_meta, extract_rawChunk, assert_chan_in_dataset, chan_map
from npyx.gl import get_units, assert_multi, get_ds_ids
from npyx.spk_wvf import get_depthSort_peakChans, wvf, wvf_dsmatch, get_peak_chan, templates
from npyx.spk_t import trn, train_quality
from npyx.corr import acg, ccg, gen_sfc, get_cm, scaled_acg
from npyx.behav import align_times, get_processed_ifr, get_processed_popsync
from mpl_toolkits.mplot3d import Axes3D

# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg

import networkx as nx

#%% plotting utilities ##############################################################################################

def save_mpl_fig(fig, figname, saveDir, _format):
    saveDir=op.expanduser(saveDir)
    saveDir=Path(saveDir)
    if not saveDir.exists():
        assert saveDir.parent.exists(), f'WARNING can only create a path of a single directory level, {saveDir.parent} must exist already!'
        saveDir.mkdir()
    p=saveDir/f"{figname}.{_format}"
    fig.savefig(p, dpi=500, bbox_inches='tight')
    platform=sys.platform
    if platform=='linux':
        bashCommand = f'sudo chmod a+rwx {p}'
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


def mplshow(fig):

    # create a dummy figure and use its
    # manager to display "fig"

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

def bkshow(bkfig, title=None, save=0, savePath='~/Downloads'):
    if title is None: title=bkfig.__repr__()
    if save:bk.plotting.output_file(f'{title}.html')
    bk.plotting.show(bkfig)

def hvshow(hvobject, backend='matplotlib', return_mpl=True):
    '''
    Holoview utility which
    - for dynamic display, interaction and data exploration:
        in browser, pops up a holoview object as a bokeh figure
    - for static instanciation, refinement and data exploitation:
        in matplotlib current backend, pops up a holoview object as a matplotlib figure
        and eventually returns it for further tweaking.
    Parameters:
        - hvobject: a Holoviews object e.g. Element, Overlay or Layout.
        - backend: 'bokeh' or 'matplotlib', which backend to use to show figure
        - return_mpl: bool, returns a matplotlib figure

    '''
    assert backend in ['bokeh', 'matplotlib']
    if backend=='matplotlib' or return_mpl:
        mplfig=hv.render(hvobject, backend='matplotlib')
    if backend=='bokeh': bkshow(hv.render(hvobject, backend='bokeh'))
    elif backend=='matplotlib': mplshow(mplfig)
    if return_mpl: return mplfig


def mpl_pickledump(fig, figname, path):
    path=Path(path)
    assert path.exists(), 'WARNING provided target path does not exist!'
    figname+='.pkl'
    pkl.dump(fig, open(path/figname,'wb'))

def mpl_pickleload(fig_path):
    fig_path=Path(fig_path)
    assert fig_path.exists(), 'WARNING provided figure file path does not exist!'
    assert str(fig_path)[-4:]=='.pkl', 'WARNING provided figure file path does not end with .pkl!'
    return pkl.load(  open(fig_path,  'rb')  )

def mpl_axi_axpos(nrows, ncols, i):
    '''Converts matplotlib subplot index (as in 232 = 2x3 grid, 2nd subplot)
       into (row,col) axes grid location (in this example 0,1)'''
    ax_ids=np.arange(nrows*ncols).reshape((nrows,ncols))+1
    pos=np.argwhere(ax_ids==i)[0]
    assert any(pos), f'WARNING index {i} is too big given {nrows} rows and {ncols} columns!'
    return pos

def myround(x, base=5):
    return base * np.round(x/base)

def myceil(x, base=5):
    return base * np.ceil(x/base)

def myfloor(x, base=5):
    return base * np.floor(x/base)

def ceil_power10(x):
    return 10**np.ceil(np.log10(x))

def n_decimals(x):
    return len(str(x).split('.')[1])

def get_bestticks_from_array(arr, step=None, light=False):
    span=arr[-1]-arr[0]
    if step is None:
        upper10=ceil_power10(span)
        if span<=upper10/5:
            step=upper10*0.01
        elif span<=upper10/2:
            step=upper10*0.05
        else:
            step=upper10*0.1
    if light: step=2*step
    assert step<span, f'Step {step} is too large for array span {span}!'
    ticks=np.arange(myceil(arr[0],step),myfloor(arr[-1],step)+step,step)
    ticks=np.round(ticks, n_decimals(step))
    if step==int(step):ticks=ticks.astype(int)

    return ticks

def get_labels_from_ticks(ticks):
    ticks=npa(ticks)
    nflt=0
    for t in ticks:
        t=round(t,4)
        for roundi in range(4):
            if t == round(t, roundi):
                if nflt<(roundi):nflt=roundi
                break
    ticks_labels=ticks.astype(int) if nflt==0 else np.round(ticks.astype(float), nflt)
    jump_n=1 if nflt==0 else 2
    ticks_labels=[str(l)+'0'*(nflt+jump_n-len(str(l).replace('-',''))) for l in ticks_labels]
    return ticks_labels, nflt

def mplp(fig=None, ax=None, figsize=None,
         xlim=None, ylim=None, xlabel=None, ylabel=None,
         xticks=None, yticks=None, xtickslabels=None, ytickslabels=None, reset_xticks=False, reset_yticks=False,
         xtickrot=0, ytickrot=0, xtickha='center', xtickva='top', ytickha='right', ytickva='center',
         axlab_w='bold', axlab_s=20,
         ticklab_w='regular', ticklab_s=16, ticks_direction='out', lw=2,
         title=None, title_w='bold', title_s=20,
         hide_top_right=True, hide_axis=False,
         tight_layout=True, hspace=None, wspace=None):
    '''
    make plots pretty
    matplotlib plots
    '''
    if fig is None: fig=plt.gcf()
    if ax is None: ax=plt.gca()
    hfont = {'fontname':'Arial'}
    if figsize is not None:
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
    # Opportunity to easily hide everything
    if hide_axis:
        ax.axis('off')

    else: ax.axis('on')

    # Axis labels
    if ylabel is None:ylabel=ax.get_ylabel()
    if xlabel is None:xlabel=ax.get_xlabel()
    ax.set_ylabel(ylabel, weight=axlab_w, size=axlab_s, **hfont)
    ax.set_xlabel(xlabel, weight=axlab_w, size=axlab_s, **hfont)

    # Setup limits BEFORE altering the ticks
    # since the limits will alter the ticks
    if xlim is None: xlim=ax.get_xlim()
    if ylim is None: ylim=ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Tick values
    if xticks is None:
        if reset_xticks:
            ax.xaxis.set_major_locator(AutoLocator())
        xticks=ax.get_xticks()
        ax.set_xticks(xticks)
    else: ax.set_xticks(xticks)
    if yticks is None:
        if reset_yticks:
            ax.yaxis.set_major_locator(AutoLocator())
        yticks=ax.get_yticks()
        ax.set_yticks(yticks)
    else: ax.set_yticks(yticks)

    # Tick labels
    fig.canvas.draw() # To force setting of ticklabels
    if xtickslabels is None:
        if any(ax.get_xticklabels()):
            if isnumeric(ax.get_xticklabels()[0].get_text()): xtickslabels,x_nflt=get_labels_from_ticks(xticks)
            else: xtickslabels = ax.get_xticklabels()
    if ytickslabels is None:
        if any(ax.get_yticklabels()):
            if isnumeric(ax.get_yticklabels()[0].get_text()): ytickslabels,y_nflt=get_labels_from_ticks(yticks)
            else: ytickslabels = ax.get_yticklabels()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xtickslabels is not None:
        assert len(xtickslabels)==len(xticks), 'WARNING you provided too many/few xtickslabels! Make sure that the default/provided xticks match them.'
        ax.set_xticklabels(xtickslabels, fontsize=ticklab_s, fontweight=ticklab_w, color=(0,0,0), **hfont, rotation=xtickrot, ha=xtickha, va=xtickva)
    if ytickslabels is not None:
        assert len(ytickslabels)==len(yticks), 'WARNING you provided too many/few ytickslabels! Make sure that the default/provided yticks match them.'
        ax.set_yticklabels(ytickslabels, fontsize=ticklab_s, fontweight=ticklab_w, color=(0,0,0), **hfont, rotation=ytickrot, ha=ytickha, va=ytickva)

    # Title
    if title is None: title=ax.get_title()
    ax.set_title(title, size=title_s, weight=title_w)

    # Ticks and spines aspect
    ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=lw, length=4, direction=ticks_direction)
    spine_keys = list(ax.spines.keys())
    hide_spine_keys = ['polar'] if 'polar' in spine_keys else ['top', 'right']
    lw_spine_keys = ['polar'] if 'polar' in spine_keys else ['left', 'bottom', 'top', 'right']
    if hide_top_right and 'top' in hide_spine_keys: [ax.spines[sp].set_visible(False) for sp in hide_spine_keys]
    else: [ax.spines[sp].set_visible(True) for sp in hide_spine_keys]
    for sp in lw_spine_keys:
        ax.spines[sp].set_lw(lw)


    # Alignement and spacing elements
    if tight_layout:fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if (hspace is not None): fig.subplots_adjust(hspace=hspace)
    if (wspace is not None): fig.subplots_adjust(wspace=wspace)
    axis_to_align = [AX for AX in fig.axes if 'AxesSubplot' in AX.__repr__()]
    fig.align_ylabels(axis_to_align)
    fig.align_xlabels(axis_to_align)

    return fig, ax

def get_all_mpl_colors():
    mpl_colors=get_mpl_css_colors(sort=True, aslist=False)
    mpl_colors={**mpl_colors,**matplotlib.colors.BASE_COLORS}
    mpl_colors={**mpl_colors,**matplotlib.colors.TABLEAU_COLORS}
    return mpl_colors

def get_mpl_css_colors(sort=True, aslist=False):
    colors=matplotlib.colors.CSS4_COLORS
    if sort:
        by_hsv = sorted((tuple(mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        colors={name:colors[name] for hsv, name in by_hsv}
    if aslist: colors=list(colors.values())
    return colors

def mpl_hex(color):
    'converts a matplotlib string name to its hex representation.'
    mpl_colors=get_all_mpl_colors()
    message='color should be a litteral string recognized by matplotlib.'
    assert isinstance(color, str), message
    basecolors={'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow', 'k': 'black', 'w': 'white'}
    if color in basecolors.keys(): color=basecolors[color]
    assert color in mpl_colors.keys(), message
    return mpl_colors[color]

def hex_rgb(color):
    'converts a hex color to its rgb representation.'
    message='color must be a hex string starting with #.'
    assert color[0]=='#', message
    return tuple(int(color[1:][i:i+2], 16)/255 for i in (0, 2, 4))

def to_rgb(color):
    'converts a matplotlib string name or hex string to its rgb representation.'
    message='color must either be a litteral matplotlib string name or a hex string starting with #.'
    assert isinstance(color, str), message
    mpl_colors=get_all_mpl_colors()
    if color in mpl_colors.keys(): color=mpl_hex(color)
    assert color[0]=='#', message
    return hex_rgb(color)

def to_hsv(color):
    if isinstance(color,str):
        color=to_rgb(color)
    return mpl.colors.rgb_to_hsv(color)

def to_hex(color):
    'rgb or matplotlib litteral representation to hex representation'
    if isinstance(color,str):
        return mpl_hex(color)
    return rgb_hex(color)

def rgb_hex(color):
    '''converts a (r,g,b) color (either 0-1 or 0-255) to its hex representation.
    for ambiguous pure combinations of 0s and 1s e,g, (0,0,1), (1/1/1) is assumed.'''
    message='color must be an iterable of length 3.'
    assert assert_iterable(color), message
    assert len(color)==3, message
    if all([(c<=1)&(c>=0) for c in color]): color=[int(round(c*255)) for c in color] # in case provided rgb is 0-1
    color=tuple(color)
    return '#%02x%02x%02x' % color

def html_palette(colors, maxwidth=20, as_str=False, show=True):
    'colors must be a list of (r,g,b) values ([0-255 or 0-1]) or hex strings.'
    s=55
    n=min(len(colors),maxwidth)
    col_rows=[colors[i*maxwidth:i*maxwidth+maxwidth] for i in range(len(colors)//maxwidth+1)]
    col_rows=[c for c in col_rows if any(c)]
    h=len(col_rows)
    palette = f'<svg  width="{n * s}" height="{s * h}">'
    for r,colors in enumerate(col_rows):
        for i, c in enumerate(colors):
            if not isinstance(c,str):c=rgb_hex(c)
            palette += (
                f'<rect x="{i * s}" y="{r*s}" width="{s}" height="{s}" style="fill:{c};'
                f'stroke-width:2;stroke:rgb(255,255,255)"/>'
            )
    palette += '</svg>'
    if not as_str: palette = HTML(palette)
    if show and not as_str: display(palette)
    return palette

def get_cmap(cmap_str):
    if cmap_str in list(cmcr.keys()):
        return cmcr[cmap_str]
    else:
        return mpl.cm.get_cmap(cmap_str)

def get_bounded_cmap(cmap_str, vmin, center, vmax, colorseq='linear'):
    assert vmin<=center<=vmax, 'WARNING vmin >center or center>vmax!!'
    cmap = get_cmap(cmap_str)

    vrange = max(vmax - center, center - vmin)
    if colorseq=='linear':
        vrange=[-vrange,vrange]
        cmin, cmax = (vmin-vrange[0])/(vrange[1]-vrange[0]), (vmax-vrange[0])/(vrange[1]-vrange[0])
        colors_reindex = np.linspace(cmin, cmax, 256)
    elif colorseq=='nonlinear':
        topratio=(vmax - center)/vrange
        bottomratio=abs(vmin - center)/vrange
        colors_reindex=np.append(np.linspace(0, 0.5, int(256*bottomratio/2)),np.linspace(0.5, 1, int(256*topratio/2)))
    cmap = mpl.colors.ListedColormap(cmap(colors_reindex))

    return cmap

def get_ncolors_cmap(cmap_str, n, plot=False):
    '''Returns homogeneously distributed n colors from specified colormap.
    - cmap_str: str, matplotlib or crameri colormap
    - n_ int, n colors
    - plot: bool, whether to display colormap in HTML (works in jupyter npotebooks)
    '''
    assert n==int(n)
    n=int(n)
    cmap = get_cmap(cmap_str)
    ids=np.linspace(0,1,n)
    colors = cmap(ids)[:,:-1].tolist() # get rid of alpha
    if plot:
        html_palette(colors, 20, 0, 1)
    return colors

def get_color_families(ncolors, nfamilies, cmapstr=None, gap_between_families=4):
    '''
    Return nfamilies of ncolors colors which are perceptually closer within than between families.

    Within each family, the colors are neighbours on a perceptually sequential colormap.

    Between each family, stand gap_between_families colors.
    Increase this value to make colors within families more similar and colors between families more distinct.

    If you decide to provide a cmap, it NEEDS to be perceptually sequential,
    as this function assumes that colors with close ranks are perceptually close.
    If no cmap is provided, the matplotlib literal colors sorted by HSV are used by default.

    '''
    if cmapstr is None:
        colors_all=get_mpl_css_colors(sort=True, aslist=True)[15:-10]
        colors=npa(colors_all)[np.linspace(0,len(colors_all)-1,(ncolors+gap_between_families)*nfamilies).astype(int)].tolist()
    else:
        colors=get_ncolors_cmap(cmapstr, (ncolors+gap_between_families//2)*nfamilies, plot=False)
    highsat_colors=[c for c in colors if to_hsv(c)[1]>0.4]
    seed_ids=np.linspace(0, len(highsat_colors)-ncolors, nfamilies).astype(int)

    color_families=[[highsat_colors[si+i] for i in range(ncolors)] for si in seed_ids]

    return color_families

def format_colors(colors):
    '''
    Turns single color or iterable of colors into an iterable of colors.
    '''
    # If string: single letter or hex, can simply use flatten
    if type(npa([colors]).flatten()[0]) in [str, np.str_]:
        colors=npa([colors]).flatten()
    # if list of tuples, cannot flatten them!
    else:
        if type(colors[0]) in [float, np.float16, np.float32, np.float64]:
            colors=npa([colors,])
        else:
            colors=npa(colors)
    return colors

def set_ax_size(ax,w,h):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def hist_MB(arr, a=None, b=None, s=None, title='Histogram', xlabel='', ylabel='', ax=None, color=None):
    if a is None: a=np.min(arr)
    if b is None: b=np.max(arr)
    if s is None: s=(b-a)/100
    hist=np.histogram(arr, bins=np.arange(a,b,s))
    y=hist[0]
    x=hist[1][:-1]
    if ax is None:
        (fig, ax) = plt.subplots()
    else:
        fig, ax = ax.get_figure(), ax
    ax.bar(x=x, height=y, width=s, color=color, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel(xlabel) if len(xlabel)>0 else ax.set_xlabel('Binsize:{}'.format(s))
    ax.set_ylabel(ylabel) if len(ylabel)>0 else ax.set_ylabel('Counts')

    #fig,ax=mplp(fig,ax)

    return fig


#%% Stats plots ##############################################################################################

def plot_pval_borders(Y, p, dist='poisson', Y_pred=None, gauss_baseline_fract=1, x=None, ax=None, color=None,
                      ylabel=None, xlabel=None, title=None):
    '''
    Function to plot array X and the upper and lower borders for a given p value.
    Parameters:
        - X: np array
        - p:float, p value [0-1]
        - dist: whether to assume Poisson or Normal distribution
    '''
    Y=npa(Y)
    assert 0<p<1
    assert dist in ['poisson', 'normal']
    if ax is None: fig, ax = plt.subplots()
    else: fig=ax.get_figure()

    if dist=='poisson':
        assert (Y_pred is not None) and (len(Y_pred)==len(Y)), 'When plotting Poisson distribution, you need to provide a predictor with the same shape as X!'
        fp1=[fractile_poisson(p/2, l=c) for c in Y_pred]
        fp2=[fractile_poisson(1-p/2, l=c) for c in Y_pred]
    elif dist=='normal':
        Y_baseline=np.append(Y[:int(len(Y)*gauss_baseline_fract/2)],Y[int(len(Y)*(1-gauss_baseline_fract/2)):])
        Y_pred=np.ones(Y.shape[0])*np.mean(Y_baseline)
        fp1=np.ones(Y.shape[0])*fractile_normal(p=p/2, m=np.mean(Y_baseline), s=np.std(Y_baseline))
        fp2=np.ones(Y.shape[0])*fractile_normal(p=1-p/2, m=np.mean(Y_baseline), s=np.std(Y_baseline))

    if x is None: x=np.arange(len(Y))
    ax.plot(x,Y, c=color)
    ax.plot(x,Y_pred, c='k', ls='--', label='predictor')
    ax.plot(x,fp1, c='r', ls='--', label='pval:{}'.format(p))
    ax.plot(x,fp2, c='r', ls='--')
    ax.legend(fontsize=14)

    fig, ax = mplp(fig, ax, ylabel=ylabel, xlabel=xlabel, title=title)

    return fig

#%% Waveforms or raw data ##############################################################################################

def plot_wvf(dp, u=None, Nchannels=8, chStart=None, n_waveforms=100, t_waveforms=2.8,
             subset_selection='regular', spike_ids=None, wvf_batch_size=10, ignore_nwvf=True, again=False,
             whiten=False, med_sub=False, hpfilt=False, hpfiltf=300, nRangeWhiten=None, nRangeMedSub=None,
             title = '', plot_std=True, plot_mean=True, plot_templates=False, color=phyColorsDic[0],
             labels=False, scalebar_w=5, ticks_lw=1, sample_lines=0, ylim=[0,0],
             saveDir='~/Downloads', saveFig=False, saveData=False, _format='pdf',
             ignore_ks_chanfilt = True,
             ax_edge_um_x=22, ax_edge_um_y=18, margin=0.12, figw_inch=6, figh_inch=None,
             as_heatmap=False):
    '''
    To plot main channel alone: use Nchannels=1, chStart=None
    Parameters:
        - dp: string, datapath to kilosort directory
        - u: int, unit index
        - Nchannels: int, number of channels where waveform is plotted
        - chStart: int, channel from which to plot consecutive Nchannels | Default None, will then center on the peak channel.
        - n_waveforms: int, number of randomly sampled waveforms from which the mean and std are computed
        - t_waveforms: float, time span of the waveform samples around spike onset, in ms
        - title: string, plot title
        - std: boolean, whether or not to plot the underlying standard deviation area | default True
        - mean: boolean, whether or not to plot the mean waveform | default True
        - template: boolean, whether or not to plot the waveform template | default True
        - color: (r,g,b) tuple, hex or matplotlib litteral string, color of the mean waveform | default black
        - sample_lines: 'all' or int, whether to plot all or sample_lines individual samples in the background. Set to 0 to plot nothing.
        - labels: boolean, whether to plot or not the axis, axis labels, title...
                  If False, only waveforms are plotted along with a scale bar. | Default False
        - ylim: upper limit of plots, in uV
        - saveDir  | default False
        - saveFig: boolean, save figure source data to saveDir | default Downloads
        - saveData: boolean, save waveforms source data to saveDir | default Downloads
        - _format: string, figure saving format (any format accepted by matplotlib savefig). | Default: pdf
        - ignore_ks_chanfilt: bool, whether to ignore kilosort channel filtering (some are jumped if low activity)
        - ax_edge_um_x: float, width of subplot (electrode site) in micrometers, relatively to the electrode channel map | Default 20
        - ax_edge_um_y: float, height of subplot.
        - margin: [0-1], figure margin (in proportion of figure)
        - figw_inch: float, figure width in inches (height is derived from width, in inches)
    Returns:
        - matplotlib figure with Nchannels subplots, plotting the mean
    '''

    # Get metadata
    saveDir=op.expanduser(saveDir)
    fs=read_spikeglx_meta(dp, subtype='ap')['sRateHz']
    pv=None if ignore_ks_chanfilt else 'local'
    cm=chan_map(dp, y_orig='tip', probe_version=pv)

    peak_chan=get_peak_chan(dp, u, use_template=True)
    peak_chan_i = int(np.argmin(np.abs(cm[:,0]-peak_chan)))
    t_waveforms_s=int(t_waveforms*(fs/1000))

    # Get data
    use_dsmatch=False ##TODO make sure that ds_match returns the waveforms std
    if not use_dsmatch:
        waveforms=wvf(dp, u=u, n_waveforms=n_waveforms, t_waveforms=t_waveforms_s,
                        subset_selection=subset_selection, spike_ids=spike_ids, wvf_batch_size=wvf_batch_size, ignore_nwvf=ignore_nwvf, again=again,
                        whiten=whiten, med_sub=med_sub, hpfilt=hpfilt, hpfiltf=hpfiltf, nRangeWhiten=nRangeWhiten, nRangeMedSub=nRangeMedSub,
                        ignore_ks_chanfilt = ignore_ks_chanfilt,
                        use_old=False, loop=True, parallel=False, memorysafe=False)
    else:
        waveforms=wvf_dsmatch(dp, u, n_waveforms=n_waveforms,
                  t_waveforms=t_waveforms_s, subset_selection=subset_selection,
                  wvf_batch_size=wvf_batch_size, ignore_nwvf=True, spike_ids = None,
                  save=True, prnt=False, again=again,
                  whiten=whiten, med_sub=med_sub, hpfilt=hpfilt, hpfiltf=hpfiltf,
                  nRangeWhiten=nRangeWhiten, nRangeMedSub=nRangeMedSub,
                  use_old=False, parallel=False,
                  memorysafe=False, fast = False )[1]
    assert waveforms.shape[0]!=0,'No waveforms were found in the provided subset_selection!'
    assert waveforms.shape[1:]==(t_waveforms_s, cm.shape[0])
    tplts=templates(dp, u, ignore_ks_chanfilt=ignore_ks_chanfilt)
    assert tplts.shape[2]==waveforms.shape[2]==cm.shape[0]

    # Filter the right channels
    if chStart is None:
        chStart_i = int(max(peak_chan_i-Nchannels//2, 0))
        chStart=cm[chStart_i,0]
    else:
        chStart_i = int(max(int(np.argmin(np.abs(cm[:,0]-chStart))), 0)) # finds closest chStart given kilosort chanmap
        chStart=cm[chStart_i,0] # Should remain the same, unless chStart was capped to 384 or is a channel ignored to kilosort

    chStart_i=int(min(chStart_i, waveforms.shape[2]-Nchannels-1))
    chEnd_i = int(chStart_i+Nchannels) # no lower capping needed as
    assert chEnd_i <= waveforms.shape[2]-1

    data = waveforms[:, :, chStart_i:chEnd_i]
    data=data[~np.isnan(data[:,0,0]),:,:] # filter out nan waveforms
    datam = np.mean(data,0).T
    datastd = np.std(data,0).T
    tplts=tplts[:, :, chStart_i:chEnd_i]
    subcm=cm[chStart_i:chEnd_i,:].copy().astype(np.float32)

    # Format plotting parameters
    if type(sample_lines) is str:
        assert sample_lines=='all'
        sample_lines=min(waveforms.shape[0],n_waveforms)
    elif type(sample_lines) in [int, float]:
        sample_lines=min(waveforms.shape[0],sample_lines, n_waveforms)

    title = 'waveforms of {}'.format(u) if title=='' else title
    if isinstance(color, str):
        color=to_rgb(color)
    color_dark=(max(color[0]-0.08,0), max(color[1]-0.08,0), max(color[2]-0.08,0))
    ylim1, ylim2 = (np.nanmin(datam-datastd)-50, np.nanmax(datam+datastd)+50) if ylim==[0,0] else (ylim[0], ylim[1])
    x = np.linspace(0, data.shape[1]/(fs/1000), data.shape[1]) # Plot t datapoints between 0 and t/30 ms
    x_tplts = x[(data.shape[1]-tplts.shape[1])//2:(data.shape[1]-tplts.shape[1])//2+tplts.shape[1]] # Plot 82 datapoints between 0 and 82/30 ms

    #Plot
    if as_heatmap:
        hm_yticks=get_bestticks_from_array(subcm[:,0], step=None)[::-1]
        hm_xticks=get_bestticks_from_array(x, step=None)
        if figh_inch is None: figh_inch=figw_inch/4+0.04*subcm.shape[0]
        fig=imshow_cbar(datam, origin='bottom', xevents_toplot=[], yevents_toplot=[], events_color='k', events_lw=2,
                xvalues=x, yvalues=subcm[::-1,0], xticks=hm_xticks, yticks=hm_yticks,
                xticklabels=hm_xticks, yticklabels=hm_yticks, xlabel='Time (ms)', ylabel='Channel', xtickrot=0, title=title,
                cmapstr="RdBu_r", vmin=ylim1*0.5, vmax=ylim2*0.5, center=0, colorseq='linear',
                clabel='Voltage (\u03bcV)', extend_cmap='neither', cticks=None,
                figsize=(figw_inch/2,figh_inch), aspect='auto', function='imshow',
                ax=None)
    else:
        # Initialize figure and subplots layout
        assert 0<=margin<1
        fig_hborder=[margin,1-margin] # proportion of figure used for plotting
        fig_wborder=[margin,1-margin] # proportion of figure used for plotting
        minx_um,maxx_um=min(subcm[:,1])-ax_edge_um_x/2, max(subcm[:,1])+ax_edge_um_x/2
        miny_um,maxy_um=min(subcm[:,2])-ax_edge_um_y/2, max(subcm[:,2])+ax_edge_um_y/2
        if figh_inch is None: figh_inch=figw_inch*(maxy_um-miny_um)/(maxx_um-minx_um)
        fig=plt.figure(figsize=(figw_inch, figh_inch))

        subcm[:,1]=((subcm[:,1]-minx_um)/(maxx_um-minx_um)*np.diff(fig_wborder)+fig_wborder[0]).round(2)
        subcm[:,2]=((subcm[:,2]-miny_um)/(maxy_um-miny_um)*np.diff(fig_hborder)+fig_hborder[0]).round(2)
        axw=(ax_edge_um_x/(maxx_um-minx_um)*np.diff(fig_wborder))[0] # in ratio of figure size
        axh=(ax_edge_um_y/(maxy_um-miny_um)*np.diff(fig_hborder))[0] # in ratio of figure size

        ax=np.empty((subcm.shape[0]), dtype='O')
        # i is the relative raw data /channel index (low is bottom channel)
        i_bottomleft=np.nonzero((subcm[:2,1]==min(subcm[:2,1]))&(subcm[:2,2]==min(subcm[:2,2])))[0]
        i_bottomleft=np.argmin(subcm[:2,2]) if i_bottomleft.shape[0]==0 else i_bottomleft[0]
        for i in range(subcm.shape[0]):
            x0,y0 = subcm[i,1:]
            ax[i] =fig.add_axes([x0-axw/2,y0-axh/2,axw,axh], autoscale_on=False)

        # Plot on subplots
        for i in range(subcm.shape[0]):
            for j in range(sample_lines):
                ax[i].plot(x, data[j,:, i], linewidth=0.3, alpha=0.3, color=color)
            if plot_templates:
                pci_rel=peak_chan_i-chStart_i if chStart is None else np.argmax(np.max(datam, 1)-np.min(datam, 1))
                tpl_scalings=[(max(datam[pci_rel, :])-min(datam[pci_rel, :]))/(max(tpl[:,pci_rel])-min(tpl[:,pci_rel])) for tpl in tplts]
                if np.inf in tpl_scalings:
                    tpl_scalings[tpl_scalings==np.inf]=1
                    print('WARNING manually selected channel range does not comprise template (all zeros).')
                for tpl_i, tpl in enumerate(tplts):
                    ax[i].plot(x_tplts, tpl[:,i]*tpl_scalings[tpl_i], linewidth=1, color=(0,0,0), alpha=0.7, zorder=10000)
            if plot_mean:
                ax[i].plot(x, datam[i, :], linewidth=2, color=color_dark, alpha=1)
            if plot_std:
                ax[i].plot(x, datam[i, :]+datastd[i,:], linewidth=1, color=color, alpha=0.5)
                ax[i].plot(x, datam[i, :]-datastd[i,:], linewidth=1, color=color, alpha=0.5)
                ax[i].fill_between(x, datam[i, :]-datastd[i,:], datam[i, :]+datastd[i,:], facecolor=color, interpolate=True, alpha=0.2)
            ax[i].set_ylim([ylim1, ylim2])
            ax[i].set_xlim([x[0], x[-1]])
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['left'].set_lw(ticks_lw)
            ax[i].spines['bottom'].set_lw(ticks_lw)
            if labels:
                ax[i].text(0.99, 0.99, int(subcm[i,0]),
                                size=12, weight='regular', ha='right', va='top', transform = ax[i].transAxes)
                ax[i].tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=ticks_lw, length=3*ticks_lw, labelsize=12)
                if i==i_bottomleft:
                    ax[i].set_ylabel('Voltage (\u03bcV)', size=12, weight='bold')
                    ax[i].set_xlabel('Time (ms)', size=12, weight='bold')
                else:
                    ax[i].set_xticklabels([])
                    ax[i].set_yticklabels([])
            else:
                ax[i].axis('off')
        if not labels:
            xlimdiff=np.diff(ax[i_bottomleft].get_xlim())
            ylimdiff=ylim2-ylim1
            y_scale=int(ylimdiff*0.3-(ylimdiff*0.3)%10)
            ax[i_bottomleft].plot([0,1],[ylim1,ylim1], c='k', lw=scalebar_w)
            ax[i_bottomleft].text(0.5, ylim1-0.05*ylimdiff, '1 ms', weight='bold', size=18, va='top', ha='center')
            ax[i_bottomleft].plot([0,0],[ylim1,ylim1+y_scale], c='k', lw=scalebar_w)
            ax[i_bottomleft].text(-0.05*xlimdiff, ylim1+y_scale*0.5, f'{y_scale} \u03bcV', weight='bold', size=18, va='center', ha='right')

        if labels: fig.suptitle(t=title, x=0.5, y=0.92+0.02*(len(title.split('\n'))-1), size=18, weight='bold', va='top')

    # Save figure
    if saveFig:
        save_mpl_fig(fig, title, saveDir, _format)
    if saveData:
        np.save(Path(saveDir, title+'.npy'), waveforms)

    return fig

def plot_raw(dp, times=None, alignement_events=None, window=None, channels=np.arange(384), subtype='ap',
             offset=450, color='multi', lw=1, bg_alpha=0.8,
             title=None, _format='pdf',  saveDir='~/Downloads', saveData=0, saveFig=0, figsize=(20,8),
             whiten=False, nRangeWhiten=None, med_sub=False, nRangeMedSub=None, hpfilt=0, hpfiltf=300, ignore_ks_chanfilt=0,
             plot_ylabels=True, show_allyticks=0, yticks_jump=50, plot_baselines=False,
             events=[], set0atEvent=1,
             ax=None, ext_data=None, ext_datachans=np.arange(384),
             as_heatmap=False, vmin=-50,vmax=50,center=0):
    '''
    ## PARAMETERS
    - bp: binary path (files must ends in .bin, typically ap.bin)
    - times: list of boundaries of the time window, in seconds [t1, t2].
    - alignement_events: list of events to align the stimulus to compute an average, in seconds
    - window: [w1,w2], boundaries of mean raw trace if alignement_events is provides (ms) | Default: [-10,10]
    - channels (default: np.arange(0, 385)): list of channels of interest, in 0 indexed integers [c1, c2, c3...]
    - offset: graphical offset between channels, in uV
    - saveDir: directory where to save either the figure or the data (default: ~/Downloads)
    - saveData (default 0): save the raw chunk in the bdp directory as '{bdp}_t1-t2_c1-c2.npy'
    - saveFig: save the figure at saveDir
    - _format: format of the figure to save | default: pdf
    - color: color to plot all the lines. | default: multi, will use 20DistinctColors iteratively to distinguish channels by eye
    - whiten: boolean, whiten data or not
    - pyqtgraph: boolean, whether to use pyqtgraph backend instead of matplotlib (faster to plot and interact, use to explore data before saving nice plots with matplotlib) | default 0
    - show_allyticks: boolean, whetehr to show all y ticks or not (every 50uV for each individual channel), only use if exporing data | default 0
    - events: list of times where to plot vertical lines, in seconds.
    - set0atEvent: boolean, set time=0 as the time of the first event provided in the list events, if any is provided.
    - figsize: figure size
    - plot_ylabels
    - ax
    - title
    - data: array of shape (N channels, N time samples), externally porovided data to plot | Default: None
    -
    PS: if you wish to center the plot on the event, ensure that the event is exactly between times[0] and times[1].
    ## RETURNS
    fig: a matplotlib figure with channel 0 being plotted at the bottom and channel 384 at the top.

    '''
    pyqtgraph=0
    meta=read_spikeglx_meta(dp, subtype)
    fs = int(meta['sRateHz'])
    assert assert_iterable(events)
    # Get data
    if ext_data is None:
        channels=assert_chan_in_dataset(dp, channels)
        if times is not None:
            assert alignement_events is None, 'You can either provide a window of 2 times or a list of alignement_events + a single window to compute an average, but not both!'
            rc = extract_rawChunk(dp, times, channels, subtype, saveData, 1, whiten, hpfilt=hpfilt, hpfiltf=hpfiltf, nRangeWhiten=nRangeWhiten)
        if alignement_events is not None:
            assert window is not None
            window[1]=window[1]+1*1000/fs # to make actual window[1] tick visible
            assert times is None, 'You can either provide a window of 2 times or a list of alignement_events + a single window to compute an average, but not both!'
            assert len(alignement_events)>1
            rc = extract_rawChunk(dp, alignement_events[0]+npa(window)/1e3, channels, subtype, saveData,
                                  whiten, med_sub, hpfilt, hpfiltf, nRangeWhiten, nRangeMedSub, ignore_ks_chanfilt)
            for e in alignement_events[1:]:
                times=e+npa(window)/1e3
                rc+=extract_rawChunk(dp, times, channels, subtype, saveData,
                                     whiten, med_sub, hpfilt, hpfiltf, nRangeWhiten, nRangeMedSub, ignore_ks_chanfilt)
            rc/=len(alignement_events)
    else:
        channels=assert_chan_in_dataset(dp, ext_datachans)
        assert len(channels)==ext_data.shape[0]
        assert window is not None, 'You must tell the plotting function to which time window the external data corresponds to!'
        times=window
        rc=ext_data.copy()
        assert not pyqtgraph

    # Define y ticks
    plt_offsets = np.arange(0, rc.shape[0]*offset, offset)
    y_ticks = np.arange(0, rc.shape[0], 1) if as_heatmap else np.arange(0, rc.shape[0]*offset, offset)
    y_ticks_labels=channels

    # Sparsen y tick labels to declutter y axis
    if not show_allyticks:
        y_ticks_labels=y_ticks_labels[np.arange(len(y_ticks))%yticks_jump==0]
        y_ticks=y_ticks[np.arange(len(y_ticks))%yticks_jump==0]

    # Plot data
    t=np.arange(rc.shape[1])*1000./fs # in milliseconds
    if any(events) and times is not None:
        events=[e-times[0] for e in events] # offset to times[0]
        if set0atEvent:
            t=t-events[0]*1000
            events=[e-events[0] for e in events]
    if alignement_events is not None:
        t=t+window[0]
        events=[0]
    if not pyqtgraph:
        if isinstance(color, str):
            if color=='multi':color=None
            else:color=to_rgb(color)
        if as_heatmap:
            xticklabels = get_bestticks_from_array(t, step=None)
            xticks=xticklabels*fs/1000
            y_ticks_labels=npa([x*10 if x%2==0 else x*10-10 for x in y_ticks_labels])

            fig=imshow_cbar(im=rc, origin='top', xevents_toplot=[], events_color='k',
                            xvalues=None, yvalues=None, xticks=xticks-xticks[0], yticks=y_ticks,
                            xticklabels=xticklabels, yticklabels=y_ticks_labels, xlabel=None, ylabel=None,
                            cmapstr="RdBu_r", vmin=vmin, vmax=vmax, center=center, colorseq='nonlinear',
                            clabel='Voltage (\u03BCV)', extend_cmap='neither', cticks=None,
                            figsize=(4,10), aspect='auto', function='imshow', ax=None)
            ax=fig.axes[0]
            ax.set_ylabel('Depth (\u03BCm)', size=14, weight='bold')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            t=np.tile(t, (rc.shape[0], 1))
            rc+=plt_offsets[:,np.newaxis]
            if plot_baselines:
                for i in np.arange(rc.shape[0]):
                    y=i*offset
                    ax.plot([t[0,0], t[0,-1]], [y, y], color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
            ax.plot(t.T, rc.T, linewidth=lw, color=color, alpha=bg_alpha)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks_labels) if plot_ylabels else ax.set_yticklabels([])
            ax.set_ylabel('Channel', size=14, weight='bold')

        ax.set_xlabel('Time (ms)', size=14, weight='bold')
        ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=2, length=6, labelsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_lw(2)
        ax.spines['bottom'].set_lw(2)

        if title is not None: ax.set_title(title, size=20, weight='bold', va='bottom')

        yl=ax.get_ylim() if as_heatmap else [min(rc[0,:])-2*offset,max(rc[-1,:])+2*offset]
        xl=[0, (t[-1]-t[0])*fs/1000] if as_heatmap else [t[0,0], t[0,-1]]
        for e in events:
            if as_heatmap: e=(e-t[0])*(fs/1000)
            ax.plot([e,e], yl, color=(0.3, 0.3, 0.3), linestyle='--', linewidth=1.5)
        ax.set_ylim(yl)
        ax.set_xlim(xl)

        # fig.tight_layout()

        if saveFig:
            saveDir=op.expanduser(saveDir)
            rcn = '{}_t{}-{}_ch{}-{}'.format(op.basename(dp), times[0], times[1], channels[0], channels[-1]) # raw chunk name
            rcn=rcn+'_whitened' if whiten else rcn+'_raw'
            if title is not None: rcn=title
            fig.savefig(Path(saveDir, '{}.{}'.format(rcn, _format)), format=_format, dpi=500, bbox_inches='tight')

        return fig

def plot_raw_units(dp, times, units=[], channels=np.arange(384), offset=450,
                   Nchan_plot=5, spk_window=82, colors='phy', bg_color='k', lw=1, bg_alpha=0.8, lw_color=1.1,
                   title=None, saveDir='~/Downloads', saveData=0, saveFig=0, _format='pdf', figsize=(20,8),
                   whiten=False, nRangeWhiten=None, med_sub=False, nRangeMedSub=None, hpfilt=0, hpfiltf=300, ignore_ks_chanfilt=0,
                   show_allyticks=0, yticks_jump=50, plot_ylabels=True, events=[], set0atEvent=1):
    '''
    ## PARAMETERS
    - bp: binary path (files must ends in .bin, typically ap.bin)
    - times: list of boundaries of the time window, in seconds [t1, t2]. If 'all', whole recording.
    - channels (default: np.arange(0, 385)): list of channels of interest, in 0 indexed integers [c1, c2, c3...]
    - offset: graphical offset between channels, in uV
    - saveDir: directory where to save either the figure or the data (default: ~/Downloads)
    - saveData (default 0): save the raw chunk in the bdp directory as '{bdp}_t1-t2_c1-c2.npy'
    - saveFig: save the figure at saveDir
    - _format: format of the figure to save | default: pdf
    - color: color to plot all the lines. | default: multi, will use 20DistinctColors iteratively to distinguish channels by eye
    ## RETURNS
    fig: a matplotlib figure with channel 0 being plotted at the bottom and channel 384 at the top.

    '''
    pyqtgraph=0
    # if channels is None:
    #     peakChan=get_peak_chan(dp,units[0])
    #     channels=np.arange(peakChan-Nchan_plot//2-1, peakChan+Nchan_plot//2+2)
    channels=assert_chan_in_dataset(dp, channels)
    rc = extract_rawChunk(dp, times, channels, 'ap', saveData,
                          whiten, med_sub, hpfilt, hpfiltf, nRangeWhiten, nRangeMedSub, ignore_ks_chanfilt)
    # Offset data
    plt_offsets = np.arange(0, len(channels)*offset, offset)
    plt_offsets = np.tile(plt_offsets[:,np.newaxis], (1, rc.shape[1]))
    rc+=plt_offsets


    fig=plot_raw(dp, times, None, None, channels,
             subtype='ap', offset=450, saveDir=saveDir, saveData=saveData, saveFig=0,
             _format=_format, color=bg_color,
             whiten=whiten, nRangeWhiten=nRangeWhiten, med_sub=med_sub, nRangeMedSub=nRangeMedSub, hpfilt=hpfilt, hpfiltf=hpfiltf, ignore_ks_chanfilt=ignore_ks_chanfilt,
             show_allyticks=show_allyticks, yticks_jump=yticks_jump, events=events, set0atEvent=set0atEvent, figsize=figsize,
             plot_ylabels=plot_ylabels, ax=None, title=title, lw=lw, bg_alpha=bg_alpha)

    if not pyqtgraph: ax=fig.get_axes()[0]
    assert assert_iterable(units)
    assert len(units)>=1
    fs=read_spikeglx_meta(dp, 'ap')['sRateHz']
    spk_w1 = spk_window // 2
    spk_w2 = spk_window - spk_w1
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))

    if isinstance(colors, str):
        assert colors=='phy', 'You can only use phy as colors palette keyword.'
        phy_c=list(phyColorsDic.values())[:-1]
        colors=[phy_c[ci%len(phy_c)] for ci in range(len(units))]
    else:
        colors=list(colors)
        assert len(colors)==len(units), 'The length of the list of colors should be the same as the list of units!!'
        for ic, c in enumerate(colors):
            if isinstance(c, str): colors[ic]=to_rgb(c)

    tx=np.tile(np.arange(rc.shape[1]), (rc.shape[0], 1))[0] # in samples
    tx_ms=np.tile(np.arange(rc.shape[1])*1000./fs, (rc.shape[0], 1)) # in ms
    if any(events):
        events=[e-times[0] for e in events] # offset to times[0]
        if set0atEvent:
            tx_ms=tx_ms-events[0]*1000
            events=[e-events[0] for e in events]
    if pyqtgraph:fig[1].disableAutoRange()
    for iu, u in enumerate(units):
        print('plotting unit {}...'.format(u))
        peakChan=get_peak_chan(dp,u, use_template=False)
        assert peakChan in channels, "WARNING the peak channel of {}, {}, is not in the set of channels plotted here!".format(u, peakChan)
        peakChan_rel=np.nonzero(peakChan==channels)[0][0]
        ch1, ch2 = max(0,peakChan_rel-Nchan_plot//2), min(rc.shape[0], peakChan_rel-Nchan_plot//2+Nchan_plot)
        t=trn(dp,u) # in samples
        twin=t[(t>t1+spk_w1)&(t<t2-spk_w2)] # get spikes starting minimum spk_w1 after window start and ending maximum spk_w2 before window end
        twin-=t1 # set t1 as time 0
        for t_spki, t_spk in enumerate(twin):
            print('plotting spike {}/{}...'.format(t_spki, len(twin)))
            spk_id=(tx>=t_spk-spk_w1)&(tx<=t_spk+spk_w2)
            if pyqtgraph:
                win,p = fig
                for line in np.arange(ch1, ch2, 1):
                    p.plot(tx_ms[line, spk_id].T, rc[line, spk_id].T, linewidth=1, pen=tuple(npa(colors[iu])*255))
                fig = win,p
            else:
                ax.plot(tx_ms[ch1:ch2, spk_id].T, rc[ch1:ch2, spk_id].T, lw=lw_color, color=colors[iu])
                #ax.plot(tx_ms[peakChan_rel, spk_id].T, rc[peakChan_rel, spk_id].T, lw=1.5, color=color)
                fig.tight_layout()

    if saveFig and not pyqtgraph:
        rcn = '{}_{}_t{}-{}_ch{}-{}'.format(op.basename(dp), list(units), times[0], times[1], channels[0], channels[-1]) # raw chunk name
        rcn=rcn+'_whitened' if whiten else rcn+'_raw'
        if title is not None: rcn=title
        save_mpl_fig(fig, rcn, saveDir, _format)

    if pyqtgraph:fig[1].autoRange()
    return fig

#%% Peri-event plots ##############################################################################################

def psth_popsync_plot(trains, events, psthb=10, window=[-1000,1000],
                        events_tiling_frac=0.1, sync_win=2, fs=30000, t_end=None,
                        b=1, sd=1000, th=0.02,
                        again=False, dp=None, U=None,

                        zscore=False, zscoretype='within',
                        convolve=False, gsd=1, method='gaussian',
                        bsl_subtract=False, bsl_window=[-4000, 0], process_y=False,

                        events_toplot=[0], events_color='r',
                        title='', color='darkgreen', figsize=None,
                        saveDir='~/Downloads', saveFig=0, saveData=0, _format='pdf',
                        xticks=None, xticklabels=None, xlabel='Time (ms)', ax=None):

    x, y, y_p, y_p_var=get_processed_popsync(trains, events, psthb, window,
                          events_tiling_frac, sync_win, fs, t_end,
                          b, sd, th,
                          again, dp, U,
                          zscore, zscoretype,
                          convolve, gsd, method,
                          bsl_subtract, bsl_window, process_y)

    ylabel='Population synchrony\n(zscore of fraction firing)' if zscore \
        else r'$\Delta$ pop synchrony\n(fraction firing)' if bsl_subtract else 'Population synchrony\n(fraction firing)'
    return psth_plt(x, y_p, y_p_var, window, events_toplot, events_color,
           title, color, figsize,
           saveDir, saveFig, saveData, _format,
           zscore, bsl_subtract,
           convolve, xticks, xticklabels, xlabel, ylabel, None, False,
           ax)

def psth_plot(times, events, psthb=5, psthw=[-1000, 1000], remove_empty_trials=True, events_toplot=[0], events_color='r',
           title='', color='darkgreen', legend_label=None, legend=False,
           saveDir='~/Downloads', saveFig=0, ret_data=0, _format='pdf',
           zscore=False, bsl_subtract=False, bsl_window=[-2000,-1000], ylim=None,
           convolve=True, gsd=2, xticks=None, xticklabels=None, xlabel='Time (ms)', ylabel=None,
           ax=None, figsize=None, tight_layout=True, hspace=None, wspace=None):

    x, y, y_p, y_p_var = get_processed_ifr(times, events, b=psthb, window=psthw, remove_empty_trials=remove_empty_trials,
                                      zscore=zscore, zscoretype='within',
                                      convolve=convolve, gsd=gsd, method='gaussian_causal',
                                      bsl_subtract=bsl_subtract, bsl_window=bsl_window)

    fig = psth_plt(x, y_p, y_p_var, psthw, events_toplot, events_color,
           title, color,
           saveDir, saveFig, _format,
           zscore, bsl_subtract, ylim,
           convolve, xticks, xticklabels,
           xlabel, ylabel, legend_label, legend,
           ax, figsize, tight_layout, hspace, wspace)

    return (x,y,y_p,y_p_var) if ret_data else fig

def psth_plt(x, y_p, y_p_var, psthw, events_toplot=[0], events_color='r',
           title='', color='darkgreen',
           saveDir='~/Downloads', saveFig=0, _format='pdf',
           zscore=False, bsl_subtract=False, ylim=None,
           convolve=True, xticks=None, xticklabels=None,
           xlabel='Time (ms)', ylabel='IFR (Hz)', legend_label=None, legend=False,
           ax=None, figsize=None, tight_layout=True, hspace=None, wspace=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()

    areasteps=None if convolve else 'post'

    if zscore or bsl_subtract:
        ax.fill_between(x, y_p-y_p_var, y_p+y_p_var,
                        color=color, alpha=0.8, step=areasteps, label=legend_label)
    else:
        ax.fill_between(x, y_p-y_p_var, y_p+y_p_var, color=color, alpha=0.5, step=areasteps)
        ax.fill_between(x, y_p*0, y_p,
                        color=color, alpha=1, step=areasteps, label=legend_label)
    if legend: ax.legend()
    if convolve:
        if zscore or bsl_subtract: ax.plot(x, y_p-y_p_var, color='black', lw=0.5)
        ax.plot(x, y_p+y_p_var, color='black', lw=0.5)
        ax.plot(x, y_p, color='black', lw=2)
    else:
        if zscore or bsl_subtract: ax.step(x, y_p-y_p_var, color='black', lw=0.5, where='post')
        ax.step(x, y_p+y_p_var, color='black', lw=0.5, where='post')
        ax.step(x, y_p, color='black', lw=2,where='post')

    yl=ax.get_ylim() if ylim is None else ylim
    assert assert_iterable(yl), 'WARNING the provided ylim need to be of format [ylim1, ylim2]!'
    if not (zscore or bsl_subtract): yl=[0,yl[1]]
    for etp in events_toplot:
        ax.plot([etp,etp], yl, ls='--', lw=1, c=events_color)
        ax.set_ylim(yl)

    xl=psthw
    if bsl_subtract or zscore:
        ax.plot(xl,[0,0],lw=1,ls='--',c='black',zorder=-1)
        if zscore:
            if yl[0]<-2: ax.plot(xl,[-2,-2],lw=1,ls='--',c='black',zorder=-1)
            if yl[1]>2: ax.plot(xl,[2,2],lw=1,ls='--',c='black',zorder=-1)
    ax.set_xlim(xl)

    if ylabel is None:
        ylabel='IFR\n(zscore)' if zscore else r'$\Delta$ FR (Hz)' if bsl_subtract else 'IFR (Hz)'
    if xlabel is None: xlabel=''

    fig,ax=mplp(fig=fig, ax=ax, figsize=figsize,
     xlim=psthw, ylim=yl, xlabel=xlabel, ylabel=ylabel,
     xticks=xticks, xtickslabels=xticklabels,
     axlab_w='bold', axlab_s=20,
     ticklab_w='regular',ticklab_s=16, lw=1,
     title=title, title_w='bold', title_s=20,
     hide_top_right=True, tight_layout=tight_layout, hspace=hspace, wspace=wspace)

    if saveFig:
        figname=title
        save_mpl_fig(fig, figname, saveDir, _format)

    return fig

def raster_plot(times, events, window=[-1000, 1000], events_toplot=[0], events_color='r',
                trials_toplot=[], remove_empty_trials=False,
                title='', color='darkgreen', palette='batlow',
                marker='|', malpha=0.9, size=None, lw=3, sparseylabels=True, figsize=None,
                saveDir='~/Downloads', saveFig=0, ret_data=0, _format='pdf',
                as_heatmap=False, vmin=None, center=None, vmax=None, cmap_str=None,
                show_psth=False, psthb=10,
                zscore=False, bsl_subtract=False, bsl_window=[-2000,-1000], ylim_psth=None,
                convolve=True, gsd=2):
    '''
    Make a raster plot of the provided 'times' aligned on the provided 'events', from window[0] to window[1].
    By default, there will be len(events) lines. you can pick a subset of events to plot
    by providing their indices as a list.array with 'events_toplot'.

    Parameters:
        - times: list/array of spike times, in seconds. If list of lists/arrays,
                 each item of the list is considered an individual spike train.
        - events: list/array of events, in seconds. TRIALS WILL BE PLOTTED ACORDING TO EVENTS ORDER.
        - events_toplot: list/array of events indices to display on the raster | Default: None (plots everything)
        - window: list/array of shape (2,): the raster will be plotted from events-window[0] to events-window[1] | Default: [-1000,1000]
        - remove_empty_trials: boolean, if True does not use empty trials to compute psth
        - title: string, title of the plot + if saved file name will be raster_title._format.
        - color: string or list of strings of size
        - figsize: tuple, (x,y) figure size
        - saveDir: save directory to save data and figure
        - saveFig: boolean, if 1 saves figure with name raster_title._format at saveDir
        - ret_data: boolean, whether to return data (x,y,y_p,y_p_var) instead of matplotlib figure.
        - _format: string, format used to save figure if saveFig=1 | Default: 'pdf'

    Returns:
        - fig: matplotlib figure.
    '''

    events_order=np.argsort(events)
    events=np.sort(events)

    n_cells=len(times) if isinstance(times[0], np.ndarray) else 1
    if n_cells==1: times=[times]
    print(f'{n_cells} cell(s) detected.')
    if isinstance(color, str):
        if n_cells==1: color=[color]
        else: color=get_ncolors_cmap(palette, n_cells, plot=False)
    else: assert len(color)==n_cells, 'WARNING the number of colors needs to match the number of cells provided!'
    subplots_ratio=[4*n_cells,n_cells]

    if show_psth:
        grid = plt.GridSpec(sum(subplots_ratio), 1, wspace=0.2, hspace=0.2)
        fig = plt.figure()
        ax=fig.add_subplot(grid[:-n_cells, :])
    else:
        fig, ax = plt.subplots()


    # Define y ticks according to n_cells and trials order
    y_ticks=np.arange(len(events)*n_cells)+1
    y_ticks_labels=(np.arange(len(events)))[events_order]
    y_ticks_labels=np.hstack([y_ticks_labels[np.newaxis, :].T for i in range(n_cells)]).ravel()

    # Sparsen y tick labels to declutter y axis
    wrong_order=np.all(events_order==np.arange(events_order.shape[0]))
    if wrong_order: print('Events provided not sorted by time - this might be voluntary, just letting you know.')
    if sparseylabels and wrong_order:
        y_ticks_labels_sparse=[]
        for yi,yt in enumerate(y_ticks_labels):
            if yi%(5*n_cells)==0:y_ticks_labels_sparse.append(yt)
            else:y_ticks_labels_sparse.append('')
        y_ticks_labels=y_ticks_labels_sparse
    elif n_cells>1:
        y_ticks_labels_sparse=[]
        for yi,yt in enumerate(y_ticks_labels):
            if yi%(n_cells)==0:y_ticks_labels_sparse.append(yt)
            else:y_ticks_labels_sparse.append('')
        y_ticks_labels=y_ticks_labels_sparse

    # Plot raster
    if size is None: size=max(10,5400//len(events)) # 180 for 30 events
    if show_psth:size-=30; size=max(size,10)
    if title == '':
        title='raster' if not as_heatmap else 'heatmap'
    xlabel='Time (ms)'
    xticks=get_bestticks_from_array(np.arange(window[0], window[1], psthb), light=1)
    xlabel_plot=xlabel if not show_psth else None
    if figsize is None: figsize=[5,subplots_ratio[0]*2]
    if show_psth: figsize[1]=figsize[1]+figsize[1]//subplots_ratio[0]
    for ci in range(n_cells):
        if as_heatmap:
            x, y, y_p, y_p_var = get_processed_ifr(times[ci], events, b=psthb, window=window, remove_empty_trials=remove_empty_trials,
                                      zscore=zscore, zscoretype='within',
                                      convolve=convolve, gsd=gsd, method='gaussian_causal',
                                      bsl_subtract=bsl_subtract, bsl_window=bsl_window, process_y=True)
            extremum=max(abs(0.9*y.min()),abs(0.9*y.max()))
            if vmin is None: vmin = 0 if not (zscore|bsl_subtract) else -extremum
            if vmax is None: vmax = 0.8*y.max() if not (zscore|bsl_subtract) else extremum
            if center is None:
                center=vmin+((vmax-vmin)/2)
            if cmap_str is None: cmap_str = 'viridis' if not (zscore|bsl_subtract) else 'RdBu_r'
            ntrials=y.shape[0]
            clab='Inst. firing rate (Hz)' if not zscore else 'Inst. firing rate (zscore)'
            imshow_cbar(y, origin='top', xevents_toplot=events_toplot, events_color=events_color,
                        xvalues=np.arange(window[0], window[1], psthb), yvalues=np.arange(ntrials)+1,
                        xticks=xticks, yticks=y_ticks,
                        xticklabels=None, yticklabels=y_ticks_labels, xlabel=xlabel_plot, ylabel='Trials', title=title,
                        cmapstr=cmap_str, vmin=vmin, vmax=vmax, center=center, colorseq='nonlinear',
                        clabel=clab, extend_cmap='neither', cticks=None,
                        figsize=figsize, aspect='auto', function='imshow', ax=ax)

        else:
            at, atb = align_times(times[ci], events, window=window, remove_empty_trials=remove_empty_trials)
            ntrials=len(at)
            col=color if n_cells==1 else color[ci]
            for e, ts in at.items():
                i=events_order[np.nonzero(e==events)[0][0]]
                y=[y_ticks[i*n_cells+ci]]*len(ts)
                ts=npa(ts)*1000 # convert to ms
                ax.scatter(ts, y, s=size, color=col, alpha=malpha, marker=marker, lw=lw)
            fig,ax=mplp(fig=fig, ax=ax, figsize=figsize,
                 xlim=window, ylim=[y_ticks[-1]+1, 0], xlabel=xlabel_plot, ylabel="Trials",
                 xticks=xticks, yticks=y_ticks, xtickslabels=None, ytickslabels=y_ticks_labels,
                 axlab_w='bold', axlab_s=20,
                 ticklab_w='regular',ticklab_s=16, lw=1,
                 title=title, title_w='bold', title_s=24,
                 hide_top_right=True, hide_axis=False)
    print(f'{ntrials} trials.')
    xl=ax.get_xlim()
    yl=ax.get_ylim()
    for etp in events_toplot:
        ax.plot([etp,etp], yl, ls='--', lw=1, color=events_color)
    if any(trials_toplot):
        for ttp in trials_toplot:
            ax.plot(xl, [ttp,ttp], ls='--', lw=1, color='k')
    ax.set_ylim(yl)
    ax.set_xlim(xl)

    if show_psth:
        xticks=ax.get_xticks()
        xticklabels=get_labels_from_ticks(xticks)[0]
        ax.set_xticklabels([])
        for ci in range(n_cells):
            ax_psth=fig.add_subplot(grid[-n_cells+ci, :])
            xticklabels_subplot=xticklabels if ci==n_cells-1 else ['' for i in xticklabels]
            xlabel_subplot=xlabel if ci==n_cells-1 else ''
            psth_plot(times[ci], events, psthb=psthb, psthw=window,
                      remove_empty_trials=remove_empty_trials, events_toplot=events_toplot, events_color=events_color,
                       title=None, color=color[ci], legend_label=None, legend=False,
                       saveDir=saveDir, saveFig=0, ret_data=0, _format='pdf',
                       zscore=zscore, bsl_subtract=bsl_subtract, bsl_window=bsl_window, ylim=ylim_psth,
                       convolve=convolve, gsd=gsd,
                       xticks=xticks, xticklabels=xticklabels_subplot, xlabel=xlabel_subplot, ylabel=None,
                       ax=ax_psth, figsize=None, tight_layout=True, hspace=None, wspace=None)

    if saveFig:
        figname=title
        save_mpl_fig(fig, figname, saveDir, _format)

    return (x,y,y_p,y_p_var) if ret_data else fig

def summary_psth(trains, trains_str, events, events_str, psthb=5, psthw=[-1000,1000],
                 zscore=False, bsl_subtract=False, bsl_window=[-2000,-1000], convolve=True, gsd=2,
                 events_toplot=[0], events_col=None, trains_col_groups=None, overlap_events=False,
                 title=None, saveFig=0, saveDir='~/Downloads', _format='pdf',
                 figh=None, figratio=None, transpose=False, ylim=None,
                 as_heatmap=False,  vmin=None, center=None, vmax=None, cmap_str=None):
    '''
    Function to plot a bunch of PSTHs, all trains aligned to all sets of events, in a single summary figure.

    Parameters:
        Related to PSTH data:
            - trains: list of np arrays (s), spike trains
            - trains_str: list of str, name of trains units
            - events: list of np arrays (s), sets of events
            - events_str: list of str, name of event types
            - psthb: float (ms), psth binsize | Default 5
            - psthw: list of floats [w1,w2] (ms), psth window | Default [-1000,1000]
        Related to data processing:
            - zscore: bool, whether to zscore the data (mean/std calculated in bsl_window) | Default False
            - bsl_subtract: bool, whether to baseline_subtract | Default False
            - bsl_window: list of floats [w1,w2], window used to compute mean and std for zscoring | Default [-2000,-1000]
            - convolve: bool, whether to convolve the data with a causal gaussian window | Default True
            - gsd: float (ms), std of causal gaussian window | Default 2
        Related to events coloring/display:
            - events_toplot: list of floats, times at which to draw a vertical line | Default [0]
            - events_col: list of str/(r,g,b)/hex strings of len n_events, color of PSTHs (1 per event)
                          or str, matplotlib / crameri colormap | Default None
            - trains_col_groups: list of int of len n_trains, groups of units which should be colored alike | Default None
            - overlap_events: whether to overlap PSTHs across events. If True, it is advised to also use events_col to distinguish events properly.
        Related to figure saving:
            - title: str, figure suptitle also used as filename if saveFig is True | Default None
            - saveFig: bool, whether to save figure as saveDir/title._format | Default 0
            - saveDir: str, path to directory to save figure | Default '~/Downloads'
            - _format: str, format to save figure with | Default 'pdf'
        Related to plotting layout:
            - figh: fig height in inches | Default None
            - figratio: float, fig_width=fig_height*n_columns*fig_ratio | Default None
            - transpose: bool, whether to transpose rows/columnP (by defaults, events are rows and units columns) | Default False
        Related to heatmap plotting:
            - as_heatmap: bool, whether to represent data as heatmaps rather than columns of 2D PSTHs | Default True
            - vmin: float, min value of colormap of heatmap | Default None
            - center: float, center value of colormap of heatmap | Default None
            - vmax: float, max value of colormap of heatmap  | Default None
            - cmap_str: str, colormap of heatmap  | Default None
    '''
    ## TODO overlay=False, overlay_dim='events',
    nevents=len(events)
    ntrains=len(trains)
    if trains_col_groups is None: trains_col_groups=[0]*ntrains
    ntraingroups=np.unique(trains_col_groups).shape[0]

    if events_col is None:
        colorfamilies = get_color_families(ntraingroups, nevents)
    else:#convert to rgb
        if isinstance(events_col,str): # assumes colormap if str
            colorfamilies = get_color_families(ntraingroups, nevents, events_col)
        elif assert_iterable(events_col):
            if isinstance(events_col[0],str):
                events_col=[to_rgb(c) for c in events_col]
            colorfamilies = [[c]*ntraingroups for c in events_col]
        else:
            raise TypeError('You must provide a LIST of colors or a colormap string.')


    assert ntrains==len(trains_str)==len(trains_col_groups)
    assert nevents==len(events_str)==len(colorfamilies)

    assert len(psthw)==2
    psthw=[psthw[0], psthw[1]+psthb]

    # Plot as 2D grid of PSTHs
    if not as_heatmap:
        if figh is None: figh=8
        if figratio is None: figratio=1.2
        (nrows, ncols) = (len(events), len(trains)) if not transpose else (len(trains), len(events))
        if overlap_events: (nrows, ncols) = (1,ncols) if not transpose else (nrows, 1)
        figh=nrows*3
        figw=ncols*3*figratio
        figsize=(figw,figh)

        # pre-generate figure axes
        fig = plt.figure(figsize=figsize)
        ax_ids=np.arange(nrows*ncols).reshape((nrows,ncols))+1
        axes=ax_ids.copy().astype(object)
        for ei in range(len(events)):
            if overlap_events and (ei != 0): continue
            for ti in range(len(trains)):
                if transpose:
                    ax_id=ax_ids[ti,ei]
                    axes[ti,ei]=fig.add_subplot(nrows, ncols, ax_id)
                else:
                    ax_id=ax_ids[ei,ti]
                    axes[ei,ti]=fig.add_subplot(nrows, ncols, ax_id)

        # plot
        for ei, (e, es, cf) in enumerate(zip(events, events_str, colorfamilies)):
            for ti, (t, ts) in enumerate(zip(trains, trains_str)):
                if overlap_events: ei=0
                ax_id=ax_ids[ei,ti] if not transpose else ax_ids[ti,ei]
                ax_psth=axes[ei,ti] if not transpose else axes[ti,ei]

                legend_label=es if ((ax_id==ax_ids[-1,-1])&(overlap_events)) else None

                xlab='Time (ms)' if ax_id in ax_ids[-1,:] else ''
                ylab='IFR\n(zscore)' if zscore else r'$\Delta$ FR (Hz)' if bsl_subtract else 'IFR (Hz)'
                event_str = es if not overlap_events else ''
                (ttl_s, y_s) = (ts, event_str) if not transpose else (event_str, ts)
                ylab= f'{y_s}\n{ylab}' if ax_id in ax_ids[:,0] else ''
                ttl=ttl_s if ax_id in ax_ids[0,:] else None

                tc=cf[trains_col_groups[ti]]

                psth_plot(t, e, psthb, psthw,
                          remove_empty_trials=True, events_toplot=events_toplot, events_color='k',
                           title=ttl, color=tc, legend_label=legend_label, legend=False,
                           saveDir=saveDir, saveFig=False, ret_data=False, _format=_format,
                           zscore=zscore, bsl_subtract=bsl_subtract, bsl_window=bsl_window, ylim=ylim,
                           convolve=convolve, gsd=gsd, xticks=None, xticklabels=None, xlabel=xlab, ylabel=ylab,
                           ax=ax_psth, figsize=None, tight_layout=False, hspace=0.5, wspace=0.5)

        fig.tight_layout()
        if overlap_events: plt.legend()

        if saveFig:save_mpl_fig(fig, title, saveDir, _format)
        return fig

    # Plot as heatmaps
    if figratio is None: figratio=6
    if figh is None: figh = 2.5
    figw=figratio*figh/len(events)
    fig = plt.figure(figsize=(figw,figh))
    nmaps=len(events) if not transpose else len(trains)
    grid = plt.GridSpec(nmaps, 1, wspace=0.2, hspace=0.3)
    (l1,ls1,l2,ls2)=(events, events_str, trains, trains_str)
    if transpose:(l1,ls1,l2,ls2)=(l2,ls2,l1,ls1)
    for _i1, (_1, _s1) in enumerate(zip(l1,ls1)):
        Y=None
        ax_im=fig.add_subplot(grid[_i1,:])
        for _i2, (_2, _s2) in enumerate(zip(l2,ls2)):
            (e,t)=(_1,_2) if not transpose else (_2,_1)
            x, y, y_p, y_p_var = get_processed_ifr(t, e, b=psthb, window=psthw, remove_empty_trials=True,
                                                      zscore=zscore, zscoretype='within',
                                                      convolve=convolve, gsd=gsd, method='gaussian_causal',
                                                      bsl_subtract=bsl_subtract, bsl_window=bsl_window)
            Y=y_p if Y is None else np.vstack([Y,y_p])
        Y=npa(Y)
        if Y.ndim==1: Y=Y[np.newaxis,:] # handles case where 1 unit
        if vmin is None: vmin1 = 0 if not (zscore|bsl_subtract) else -max(abs(0.9*Y.min()),abs(0.9*Y.max()))
        else: vmin1 = vmin
        if center is None: center1 = 0.4*Y.max() if not (zscore|bsl_subtract) else 0
        else: center1 = center
        if vmax is None: vmax1 = 0.8*Y.max() if not (zscore|bsl_subtract) else max(abs(0.9*Y.min()),abs(0.9*Y.max()))
        else: vmax1 = vmax
        if cmap_str is None: cmap_str = 'viridis' if not (zscore|bsl_subtract) else 'RdBu_r'
        nunits=Y.shape[0]
        y_ticks_labels=trains_str if not transpose else events_str
        clab='Inst. firing rate (Hz)' if not zscore else 'Inst. firing rate (zscore)'
        ylab=f'Units\n{_s1}' if not transpose else f'Events\n{_s1}'
        tc=colorfamilies[_i1][trains_col_groups[_i2]] if not transpose else colorfamilies[_i2][trains_col_groups[_i1]]
        xlab='Time (ms)' if _i1==len(l1)-1 else None
        imshow_cbar(Y, origin='top', xevents_toplot=events_toplot, events_color=tc,
                    xvalues=np.arange(psthw[0], psthw[1], psthb), yvalues=np.arange(nunits)+1,
                    xticks=None, yticks=np.arange(Y.shape[0])+1,
                    xticklabels=None, yticklabels=y_ticks_labels, xlabel=xlab, xtickrot=0,
                    ylabel=ylab, title=None,
                    cmapstr=cmap_str, vmin=vmin1, vmax=vmax1, center=center1, colorseq='nonlinear',
                    clabel=clab, extend_cmap='neither', cticks=None,
                    figsize=figsize, aspect='auto', function='imshow', ax=ax_im, tight_layout=False,
                    cmap_h=0.6/nmaps)

    if title is not None: fig.suptitle(title)
    fig.tight_layout()
    if saveFig:save_mpl_fig(fig, title, saveDir, _format)
    return fig

#%% Correlograms ##############################################################################################

def plt_ccg(uls, CCG, cbin=0.04, cwin=5, bChs=None, fs=30000, saveDir='~/Downloads', saveFig=True,
            _format='pdf', subset_selection='all', labels=True, std_lines=True, title=None, color=-1,
            saveData=False, ylim1=0, ylim2=0, normalize='Hertz', ccg_mn=None, ccg_std=None,
            figsize=(4.5,4), show_hz=False):
    '''Plots acg and saves it given the acg array.
    unit: int.
    ACG: acg array in non normalized counts.
    cwin and cbin: full window and bin in ms.
    phycolor: index (0 to 5) of the phy colorchart.
    savedir: plot saving destination.
    save: boolean, to save the figure or not.
    '''
    global phyColorsDic

    cbin = np.clip(cbin, 1000*1./fs, 1e8)
    if isinstance(color, int): # else, an actual color is passed
        color=phyColorsDic[color]
    fig, ax = plt.subplots(figsize=(10,8))
    x=np.linspace(-cwin*1./2, cwin*1./2, CCG.shape[0])
    assert x.shape==CCG.shape
    if ylim1==0 and ylim2==0:
        if normalize in ['Hertz','Counts']:
            ylim1=0
            yl=max(CCG); ylim2=int(yl)+5-(yl%5);
        elif normalize=='Pearson':
            ylim1=0
            yl=max(CCG); ylim2=yl+0.01-(yl%0.01);
        elif normalize=='zscore':
            yl1=min(CCG);yl2=max(CCG)
            ylim1=yl1-0.5+(abs(yl1)%0.5);ylim2=yl2+0.5-(yl2%0.5)
            ylim1, ylim2 = min(-3, ylim1), max(3, ylim2)
            ylim1, ylim2 = -max(abs(ylim1), abs(ylim2)), max(abs(ylim1), abs(ylim2))
    ax.set_ylim([ylim1, ylim2])

    if ccg_mn is not None and ccg_std is not None and show_hz:
        ax2 = ax.twinx()
        ax2.set_ylabel('Crosscorrelation (Hz)', fontsize=20, rotation=270, va='bottom')
        ax2ticks=[np.round(ccg_mn+tick*ccg_std,1) for tick in ax.get_yticks()]
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ax2ticks, fontsize=20)
        ax2.set_ylim([ylim1, ylim2])

    if normalize in ['Hertz', 'Pearson', 'Counts']:
        y=CCG.copy()
    elif normalize in ['zscore']:
        y=CCG.copy()+abs(ylim1)
    ax.bar(x=x, height=y, width=cbin, color=color, edgecolor=color, bottom=ylim1) # Potentially: set bottom=0 for zscore

    ax.plot([0,0], ax.get_ylim(), ls="--", c=[0.5,0.5,0.5], lw=1, zorder=1000)
    if labels:
        if std_lines:
            if (normalize!='zscore'):
                mn = np.mean(np.append(CCG[:int(len(CCG)*2./5)], CCG[int(len(CCG)*3./5):]))
                std = np.std(np.append(CCG[:int(len(CCG)*2./5)], CCG[int(len(CCG)*3./5):]))
                ax.plot([x[0], x[-1]], [mn,mn], ls="--", c=[0,0,0], lw=2)
                for st in [1,2,3]:
                    ax.plot([x[0], x[-1]], [mn+st*std,mn+st*std], ls="--", c=[0.5,0,0], lw=0.5)
                    ax.plot([x[0], x[-1]], [mn-st*std,mn-st*std], ls="--", c=[0,0,0.5], lw=0.5)
            else:
                ax.plot([x[0], x[-1]], [0,0], ls="--", c=[0,0,0], lw=2)
        if normalize=='Counts':
            ax.set_ylabel("Crosscorrelation (Counts)", size=20)
        if normalize=='Hertz':
            ax.set_ylabel("Crosscorrelation (Hz)", size=20)
        elif normalize=='Pearson':
            ax.set_ylabel("Crosscorrelation (Pearson)", size=20)
        elif normalize=='zscore':
            ax.set_ylabel("Crosscorrelation (z-score)", size=20)
        ax.set_xlabel('Time (ms)', size=20)
        ax.set_xlim([-cwin*1./2, cwin*1./2])
        if not isinstance(title, str):
            if bChs is None:
                title="Units {}->{} ({})s".format(uls[0], uls[1], str(subset_selection)[0:50].replace(' ',  ''))
            else:
                title="Units {}@{}->{}@{} ({})s".format(uls[0], bChs[0], uls[1], bChs[1], str(subset_selection)[0:50].replace(' ',  ''))
        ax.set_title(title, size=22)
        ax.tick_params(labelsize=20)
    mplp(fig, ax, figsize=figsize)

    if saveFig:
        save_mpl_fig(fig, 'ccg{0}-{1}_{2}_{3:.2f}'.format(uls[0], uls[1], cwin, cbin), saveDir, _format)
    if saveData:
        np.save(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}_x.npy'.format(uls[0], uls[1], cwin, cbin), x)
        np.save(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}_y.npy'.format(uls[0], uls[1], cwin, cbin), CCG)

    return fig

def plt_acg(unit, ACG, cbin=0.2, cwin=80, bChs=None, color=0, fs=30000, saveDir='~/Downloads', saveFig=True,
            _format='pdf', subset_selection='all', labels=True, title=None, ref_per=True, saveData=False,
            ylim1=0, ylim2=0, normalize='Hertz', acg_mn=None, acg_std=None, figsize=(4.5,4)):
    '''Plots acg and saves it given the acg array.
    unit: int.
    ACG: acg array in non normalized counts.
    cwin and cbin: full window and bin in ms.
    phycolor: index (0 to 5) of the phy colorchart.
    savedir: plot saving destination.
    saveFig: boolean, to save the figure or not.
    '''
    global phyColorsDic
    cbin = np.clip(cbin, 1000*1./fs, 1e8)
    if isinstance(color, int): # else, an actual color is passed
        color=phyColorsDic[color]
    fig, ax = plt.subplots(figsize=figsize)
    x=np.linspace(-cwin*1./2, cwin*1./2, ACG.shape[0])
    assert x.shape==ACG.shape
    if ylim1==0 and ylim2==0:
        if normalize in ['Hertz','Counts']:
            ylim1=0
            yl=max(ACG); ylim2=int(yl)+5-(yl%5);
        elif normalize=='Pearson':
            ylim1=0
            yl=max(ACG); ylim2=yl+0.01-(yl%0.01);
        elif normalize=='zscore':
            yl1=min(ACG);yl2=max(ACG)
            ylim1=yl1-0.5+(abs(yl1)%0.5);ylim2=yl2+0.5-(yl2%0.5)
            ylim1, ylim2 = min(-3, ylim1), max(3, ylim2)
            ylim1, ylim2 = -max(abs(ylim1), abs(ylim2)), max(abs(ylim1), abs(ylim2))

    if acg_mn is not None and acg_std is not None:
        ax2 = ax.twinx()
        ax2.set_ylabel('Autocorrelation (Hz)', fontsize=20, rotation=270, va='bottom')
        ax2ticks=[np.round(acg_mn+tick*acg_std,1) for tick in ax.get_yticks()]
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ax2ticks, fontsize=20)
        ax2.set_ylim([ylim1, ylim2])

    if normalize in ['Hertz', 'Pearson', 'Counts']:
        y=ACG.copy()
    elif normalize=='zscore':
        y=ACG.copy()+abs(ylim1)
    ax.fill_between(x, y*0, y, color=color, step='mid')
    ax.step(x, y, where='mid', color='black', lw=1)

    if labels:
        if not isinstance(title, str):
            if  bChs is None:
                title="Unit {} ({})s".format(unit, str(subset_selection)[0:50].replace(' ',  ''))
            else:
                assert len(bChs)==1
                title="Unit {}@{} ({})s".format(unit, bChs[0], str(subset_selection)[0:50].replace(' ',  ''))
        if ref_per:
            ax.plot([-1, -1], [ylim1, ylim2], color='black', linestyle='--', linewidth=1)
            ax.plot([1, 1], [ylim1, ylim2], color='black', linestyle='--', linewidth=1)
    ylabdic={'Counts':'Counts',
                    'Hertz':'Hz',
                    'Pearson':'Pearson',
                    'zscore':'z-score'}
    ylabel=f"Autocorrelation ({ylabdic[normalize]})" if labels else None
    mplp(fig, figsize=figsize,
         title=title,
         xlabel='Time (ms)', ylabel=ylabel,
         title_s=16, axlab_s=16, ticklab_s=16,
         xlim=[-cwin*1./2, cwin*1./2], ylim=[ylim1, ylim2])

    if saveFig:
        save_mpl_fig(fig, 'acg{}-{}_{:.2f}'.format(unit, cwin, cbin), saveDir, _format)
    if saveData:
        np.save(saveDir+'/acg{}-{}_{:.2f}_x.npy'.format(unit, cwin, cbin), x)
        np.save(saveDir+'/acg{}-{}_{:.2f}_y.npy'.format(unit, cwin, cbin), ACG)

    return fig


def plt_ccg_subplots(units, CCGs, cbin=0.2, cwin=80, bChs=None, saveDir='~/Downloads',
                     saveFig=False, prnt=False, _format='pdf', figsize=None, subset_selection='all',
                     labels=True, show_ttl=True, title=None, std_lines=False, ylim1=0, ylim2=0, normalize='zscore'):
    bChs=npa(bChs).astype(int)
    l=len(units)
    x=np.arange(-cwin/2, cwin/2+cbin, cbin)

    if figsize is None: figsize=(4.5*l/2,4*l/2)
    fig = plt.figure(figsize=figsize)
    for row in range(l):
        for col in range(l):
            ax=fig.add_subplot(l, l, 1+row*l+col%l)
            if normalize!='mixte':normalize1=normalize
            if row>col:
                mplp(ax=ax, hide_axis=True)
                continue
            if (row==col):
                color=phyColorsDic[row%6]
                y=CCGs[row,col,:]
                if normalize=='mixte':
                    normalize1='Hertz'
            else:
                color=phyColorsDic[-1]
                if normalize=='mixte':
                    y=zscore(CCGs[row,col,:], 4./5)
                    normalize1='zscore'
                else:
                    y=CCGs[row,col,:]

            ax.plot(x, y, color=color, alpha=0)
            ax.set_xlim([-cwin*1./2, cwin*1./2])

            if normalize1 in ['Hertz','Pearson','Counts']:
                ax.set_ylim([0, ax.get_ylim()[1]])
                ax.fill_between(x, np.zeros(len(x)), y, color=color)
            elif normalize1=='zscore':
                ylmax=max(np.abs(ax.get_ylim()))
                ax.set_ylim([-ylmax, ylmax])
                ax.fill_between(x, -ylmax*np.ones(len(x)), y, color=color)

            if labels:
                if row==col==0:
                    ax.set_ylabel("Crosscorr. ({})".format(normalize), size=12)
                if col==l-1 and row==l-1:
                    ax.set_xlabel('Time (ms)', size=12)

                if any(bChs):
                    ttl="{}@{}>{}@{}".format(units[row], bChs[row], units[col], bChs[col])
                else:
                    ttl="{}>{}".format(units[row], units[col])
            else:
                ttl=None
            if not show_ttl: ttl=None

            mplp(ax=ax, figsize=figsize, lw=1,
                 title=ttl, title_s=8, title_w='regular',
                 axlab_s=12, axlab_w='regular',
                 ticklab_s=12, ticklab_w='regular',
                 tight_layout=False)

    if title is not None:
        fig.suptitle(title, size=20, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if saveFig:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        save_mpl_fig(fig, f"ccg{str(units).replace(' ', '')}-{cwin}_{cbin}", saveDir, _format)

    return fig

def plot_acg(dp, unit, cbin=0.2, cwin=80, normalize='Hertz', color=0, saveDir='~/Downloads', saveFig=True, prnt=False,
             _format='pdf', subset_selection='all', labels=True, title=None, ref_per=True, saveData=False, ylim=[0,0],
             acg_mn=None, acg_std=None, again=False,
             figsize=(4.5,4)):
    saveDir=op.expanduser(saveDir)
    bChs=get_depthSort_peakChans(dp, units=[unit])[:,1].flatten()
    ylim1, ylim2 = ylim[0], ylim[1]
    ACG=acg(dp, unit, cbin, cwin, fs=30000, normalize=normalize, prnt=prnt, subset_selection=subset_selection, again=again)
    if normalize=='zscore':
        ACG_hertz=acg(dp, unit, cbin, cwin, fs=30000, normalize='Hertz', prnt=prnt, subset_selection=subset_selection)
        acg25, acg35 = ACG_hertz[:int(len(ACG_hertz)*2./5)], ACG_hertz[int(len(ACG_hertz)*3./5):]
        acg_std=np.std(np.append(acg25, acg35))
        acg_mn=np.mean(np.append(acg25, acg35))
    fig=plt_acg(unit, ACG, cbin, cwin, bChs, color, 30000, saveDir, saveFig, _format=_format,
            subset_selection=subset_selection, labels=labels, title=title, ref_per=ref_per, saveData=saveData, ylim1=ylim1, ylim2=ylim2,
            normalize=normalize, acg_mn=acg_mn, acg_std=acg_std, figsize=figsize)

    return fig

def plot_ccg(dp, units, cbin=0.2, cwin=80, normalize='mixte', saveDir='~/Downloads', saveFig=False, prnt=False,
             _format='pdf', figsize=None,subset_selection='all', labels=True, std_lines=True,
             title=None, show_ttl=True, color=-1, CCG=None, saveData=False,
             ylim=[0,0], ccg_mn=None, ccg_std=None, again=False, trains=None, as_grid=False,
             use_template=True):
    assert assert_iterable(units)
    units=list(units)
    _, _idx=np.unique(units, return_index=True)
    units=npa(units)[np.sort(_idx)].tolist()
    assert normalize in ['Counts', 'Hertz', 'Pearson', 'zscore', 'mixte'],"WARNING ccg() 'normalize' argument should be a string in ['Counts', 'Hertz', 'Pearson', 'zscore', 'mixte']."#
    if normalize=='mixte' and len(units)==2 and not as_grid: normalize='zscore'
    saveDir=op.expanduser(saveDir)
    bChs=get_depthSort_peakChans(dp, units=units, use_template=use_template)[:,1].flatten()
    ylim1, ylim2 = ylim[0], ylim[1]

    if CCG is None:
        normalize1 = normalize if normalize!='mixte' else 'Hertz'
        CCG=ccg(dp, units, cbin, cwin, fs=30000, normalize=normalize1, prnt=prnt, subset_selection=subset_selection, again=again, trains=trains)
    assert CCG is not None

    if CCG.shape[0]==2 and not as_grid:
        if normalize=='zscore':
            CCG_hertz=ccg(dp, units, cbin, cwin, fs=30000, normalize='Hertz', prnt=prnt, subset_selection=subset_selection, again=again, trains=trains)[0,1,:]
            ccg25, ccg35 = CCG_hertz[:int(len(CCG_hertz)*2./5)], CCG_hertz[int(len(CCG_hertz)*3./5):]
            ccg_std=np.std(np.append(ccg25, ccg35))
            ccg_mn=np.mean(np.append(ccg25, ccg35))

        if figsize is None: figsize=(4.5,4)
        fig = plt_ccg(units, CCG[0,1,:], cbin, cwin, bChs, 30000, saveDir, saveFig, _format, subset_selection=subset_selection,
                      labels=labels, std_lines=std_lines, title=title, color=color, saveData=saveData, ylim1=ylim1, ylim2=ylim2,
                      normalize=normalize, ccg_mn=ccg_mn, ccg_std=ccg_std, figsize=figsize)
    else:
        fig = plt_ccg_subplots(units, CCG, cbin, cwin, bChs, saveDir, saveFig, prnt, _format, figsize, subset_selection=subset_selection,
                               labels=labels, show_ttl=show_ttl,title=title, std_lines=std_lines, ylim1=ylim1, ylim2=ylim2, normalize=normalize)

    return fig

def plot_scaled_acg( dp, units, cut_at = 150, bs = 0.5, min_sec = 180, again = False):
    """
    Make the plot used for showing different ACG shapes
    Return: plot
    """
    # check if units are a list
    if isinstance(units, (int, np.int16, np.int32, np.int64)):
        # check if it's len 1
        units = [units]
    elif isinstance(units, str):
        if units.strip() == 'all':
            units = get_units(dp, quality = 'good')
        else:
            raise ValueError("You can only pass 'all' as a string")
    elif isinstance(units, list):
        pass
    else:
            raise TypeError("Only the string 'all', ints, list of ints or ints disguised as floats allowed")

    rec_name = str(dp).split('/')[-1]

    normed_new, isi_mode, isi_hist_counts, isi_hist_range, acg_unnormed  = scaled_acg(dp, units, cut_at = cut_at, bs = bs, min_sec = min_sec, again = again)

    # find the units where the normed_new values pass our filter
    good_ones = np.sum(normed_new, axis = 1) !=0
    good_acgs = normed_new[good_ones]
    good_units = np.array(units)[good_ones]
    good_isi_mode = isi_mode[good_ones]
    good_isi_hist_counts = isi_hist_counts[good_ones]
    good_isi_hist_range = isi_hist_range[good_ones]
    good_acg_unnormed = acg_unnormed[good_ones]

    for unit_id in range(good_units.shape[0]):
        unit = good_units[unit_id]
        fig,ax = plt.subplots(3)
        fig.suptitle(f"Unit {unit} on dp \n {rec_name} \n and mfr mean_fr and isi_hist_mode isi_hist_mode len acg.shape[0]")
        ax[0].vlines(good_isi_mode[unit_id], 0, np.nanmax(good_isi_hist_counts[unit_id]), color = 'red')
        ax[0].bar(good_isi_hist_range[unit_id],good_isi_hist_counts[unit_id])
        ax[1].vlines(good_isi_mode[unit_id], 0,np.nanmax(good_acg_unnormed[unit_id]), color = 'red')
        ax[1].plot(np.arange(0, good_acg_unnormed[unit_id].shape[0]*bs, bs),good_acg_unnormed[unit_id])
#                    ax[2].plot(smooth_new)
        ax[2].plot(good_acgs[unit_id])
#                    ax[2].plot(unit_normed)
        ax[2].vlines(100, 0,np.max(good_acgs[unit_id]), color = 'red')
        fig.tight_layout()


#%% Heatmaps including correlation matrices ##############################################################################################

def imshow_cbar(im, origin='top', xevents_toplot=[], yevents_toplot=[], events_color='k', events_lw=2,
                xvalues=None, yvalues=None, xticks=None, yticks=None,
                xticklabels=None, yticklabels=None, xlabel=None, ylabel=None, xtickrot=45, title='',
                cmapstr="RdBu_r", vmin=None, vmax=None, center=None, colorseq='nonlinear',
                clabel='', extend_cmap='neither', cticks=None,
                figsize=(6,4), aspect='auto', function='imshow',
                ax=None, tight_layout=True, cmap_h=0.3, **kwargs):
    '''
    Essentially plt.imshow(im, cmap=cmapstr), but with a nicer and actually customizable colorbar.

    Parameters:
        - im: 2D array def to matplotlib.pyplot.imshow
        - origin: y axis origin, either top or bottom | Default: top
        - xvalues, yvalues: lists/arrays of lengths im.shape[1] and im.shape[0], respectively.
                            Allows to alter the value to which pixel positions are mapped (which are dumb pixel ranks by default).
        - xticks, yticks: allows to alter the position of the ticks (in [0,npixels] space by default, in xvalues/yvalues space if they are provided)
        - xticklabels, yticklabels: allows to alter the label of the ticks - should have the same size as xticks/yticks
        - cmapstr: string, colormap name from matplotlib ('RdBu_r') or Fabio Crameri package ('batlow')
        - vmin: value to which the lower boundary of the colormap corresponds
        - vmax: value to which the upper boundary of the colormap corresponds
        - center: value to which the center of the colormap corresponds
        - colorseq: string, {'linear', 'nonlinear'}, whether to shrink or not the colormap between the center and the closest boundary
                    when 'center' is not None and isn't equidistant between vmax and vmin
        - clabel: string, colormap label
        - extend_cmap: tring, {'neither', 'both', 'min', 'max'}. If not 'neither', make pointed end(s) for out-of- range values.
        - cticks: list of ticks to show
        - aspect: {'equal', 'auto'}, see imshow documentation
    '''
    assert colorseq in ['linear', 'nonlinear']
    if im.ndim==1:
        print('Single row of pixels detected - plotting it horizontally.')
        im=im[np.newaxis,:]
    assert im.ndim==2
    assert isinstance(cmapstr,str), 'cmap must be a string!'

    minimum=im.min() if vmin is None else vmin
    maximum=im.max() if vmax is None else vmax
    rng=maximum-minimum
    if vmin is None: vmin = minimum+0.1*rng if center is None else min(minimum+0.1*rng,center-0.01*rng)
    if vmax is None: vmax = maximum-0.1*rng if center is None else max(maximum-0.1*rng, center+0.01*rng)
    if center is None: center=vmin+((vmax-vmin)/2)
    if cticks is not None: assert cticks[-1]<=vmax and cticks[0]>=vmin

    # Make custom colormap.
    # If center if provided, reindex colors accordingly
    if center is None:
        cmap = get_cmap(cmapstr)
    else:
        cmap=get_bounded_cmap(cmapstr, vmin, center, vmax, colorseq)

    # Define pixel coordinates (default is 0 to n_rows-1 for y and n_columns=1 for x)
    if xvalues is None: xvalues=np.arange(im.shape[1])
    assert len(xvalues)==im.shape[1], f'xvalues should contain {im.shape[1]} values but contains {len(xvalues)}!'
    dx = (xvalues[1]-xvalues[0])/2 if len(xvalues)>1 else xvalues[0]
    if yvalues is None: yvalues=np.arange(im.shape[0])
    assert len(yvalues)==im.shape[0], f'yvalues should contain {im.shape[0]} values but contains {len(yvalues)}!'
    dy = (yvalues[1]-yvalues[0])/2 if len(yvalues)>1 else yvalues[0]
    extent = [xvalues[0]-dx, xvalues[-1]+dx, yvalues[-1]+dy, yvalues[0]-dy]

    # Plot image with custom colormap
    fig,ax=plt.subplots(figsize=figsize) if ax is None else (ax.get_figure(), ax)
    if function=='imshow': axim=ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
                                          origin={'top':'upper', 'bottom':'lower'}[origin], extent=extent, interpolation='none',
                                          **kwargs)
    elif function=='pcolor': axim=ax.pcolormesh(im, X=xvalues, Y=yvalues,
                                                cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if any(xevents_toplot):
        for e in xevents_toplot:
            yl=ax.get_ylim()
            ax.plot([e,e],yl,lw=events_lw,ls='--',c=events_color)
            ax.set_ylim(yl)
    if any(yevents_toplot):
        for e in yevents_toplot:
            xl=ax.get_xlim()
            ax.plot(xl,[e,e],lw=events_lw,ls='--',c=events_color)
            ax.set_xlim(xl)

    if xticks is None:
        xticks=np.arange(im.shape[1]) if im.shape[1]<=6 else get_bestticks_from_array(np.arange(im.shape[1]), step=None, light=0)
    if yticks is None:
        yticks=np.arange(im.shape[0]) if im.shape[0]<=6 else get_bestticks_from_array(np.arange(im.shape[0]), step=None, light=0)
    mplp(fig, ax, figsize=figsize,
          xlim=None, ylim=None, xlabel=xlabel, ylabel=ylabel,
          xticks=xticks, yticks=yticks, xtickslabels=xticklabels, ytickslabels=yticklabels,
          reset_xticks=False, reset_yticks=False, xtickrot=xtickrot, ytickrot=0,
          xtickha={0:'center',45:'right'}[xtickrot], xtickva='top', ytickha='right', ytickva='center',
          axlab_w='bold', axlab_s=14,
          ticklab_w='regular', ticklab_s=10, ticks_direction='out', lw=1,
          title=title, title_w='bold', title_s=14,
          hide_top_right=False, hide_axis=False, tight_layout=tight_layout)

    # Add colorbar, nicely formatted
    axpos=ax.get_position()
    cbaraxx0,cbaraxy0 = float(max(axpos.x1, 0.85)+0.005), float(axpos.y0)
    cbar_ax = fig.add_axes([cbaraxx0, cbaraxy0, .01, cmap_h])
    if cticks is None: cticks=get_bestticks_from_array(np.arange(vmin,vmax+(vmax-vmin)/10,(vmax-vmin)/10), light=True)
    fig.colorbar(axim, cax=cbar_ax, ax=ax,
             orientation='vertical', label=clabel,
             extend=extend_cmap, ticks=cticks, use_gridspec=True)
    if clabel is not None:
        cbar_ax.yaxis.label.set_font_properties(mpl.font_manager.FontProperties(family='arial',weight='bold', size=14))
        cbar_ax.yaxis.label.set_rotation(-90)
        cbar_ax.yaxis.label.set_va('bottom')
        cbar_ax.yaxis.label.set_ha('center')
        cbar_ax.yaxis.labelpad=5
    fig.canvas.draw()
    set_ax_size(ax,*fig.get_size_inches())
    #cticks=[t.get_text() for t in cbar_ax.yaxis.get_ticklabels()]
    cbar_ax.yaxis.set_ticklabels(cticks, ha='left')
    cbar_ax.yaxis.set_tick_params(pad=5, labelsize=12)

    return fig

# Plot correlation matrix of variables x observations 2D array

def plot_cm(dp, units, cwin=100, cbin=0.2, b=5, corrEvaluator='CCG', vmax=5, vmin=0, cmap='viridis', subset_selection='all',
            saveDir='~/Downloads', saveFig=False, _format='pdf', title=None, ret_cm=False):
    '''Plot correlation matrix.
    dp: datapath
    units: units list of the same dataset
    b: bin, in milliseconds'''

    # Sanity checks
    allowedCorEvals = ['CCG', 'covar', 'corrcoeff', 'corrcoeff_MB']
    try:
        assert corrEvaluator in allowedCorEvals
    except:
        print('WARNING: {} should be in {}. Exiting now.'.format(corrEvaluator, allowedCorEvals))
        return

    # Sort units by depth
    mainChans = get_depthSort_peakChans(dp, units)
    units, channels = mainChans[:,0], mainChans[:,1]

    # make correlation matrix of units sorted by depth
    cm = get_cm(dp, units, cbin, cwin, b, corrEvaluator, subset_selection)

    # Plot correlation matrix
    ttl = f"Dataset: {dp.split('/')[-1]}" if title is None else title
    fig=imshow_cbar(cm, origin='top',
                xticks=np.arange(len(units)), yticks=np.arange(len(units)),
                xticklabels=[f'{units[i]}'for i in range(len(units))],
                yticklabels=[f'{units[i]}@{channels[i]}' for i in range(len(units))],
                xlabel=None, ylabel=None, title=ttl,
                cmapstr=cmap, vmin=vmin, vmax=vmax, colorseq='nonlinear',
                clabel='Crosscorr. [-0.5-0.5]ms (s.d.)', aspect='equal')

    if saveFig:
        if saveDir is None: saveDir=dp
        assert title is not None, 'You need to provide a title parameter to save the figure!'
        fig.savefig(Path(saveDir,title,f'.{_format}'))

    if ret_cm:
        return cm, units, channels # depth-sorted
    return fig

# Connectivity inferred from correlograms

def plot_sfcm(dp, corr_type='connections', metric='amp_z', cbin=0.5, cwin=100,
              p_th=0.02, n_consec_bins=3, fract_baseline=4./5, W_sd=10, test='Poisson_Stark',
              drop_seq=['sign', 'time', 'max_amplitude'], units=None, name=None,
              text=False, markers=False, ticks=True, depth_ticks=False,
              regions={}, reg_colors={}, vminmax=[-7,7], figsize=(6,6),
              saveFig=False, saveDir=None, _format='pdf',
              again=False, againCCG=False, use_template_for_peakchan=False, subset_selection='all'):
    '''
    Visually represents the connectivity datafrane outputted by 'gen_sfc'.
    Each line/row is a good unit.
    Each intersection is a square split in a varying amount of columns,
    each column representing a positive or negatively significant peak collored accordingly to its size s.
    '''

    sfc, sfcm, peakChs, sigstack, sigustack = gen_sfc(dp, corr_type, metric, cbin, cwin,
                                 p_th, n_consec_bins, fract_baseline, W_sd, test,
                                 again, againCCG, drop_seq, None, None, units=units, name=name,
                                 use_template_for_peakchan=use_template_for_peakchan,
                                 subset_selection=subset_selection)
    gu = peakChs[:,0]
    ch = peakChs[:,1].astype(int)

    if corr_type=='synchrony':
        vminmax=[0,vminmax[1]]
    elif corr_type=='excitations':
        vminmax=[0,vminmax[1]]
    elif corr_type=='inhibitions':
        vminmax=[vminmax[0],0]

    if depth_ticks:
        labs=['{}'.format(3840-ch[i]*10) for i in range(len(gu)) if i%10==0]
        tks=[i for i in range(len(gu)) if i%10==0]
        lab = 'Depth on probe (\u03BCm)'
    else:
        labs=['{}@{}'.format(gu[i], ch[i]) for i in range(len(gu))]
        tks=np.arange(len(labs))
        lab = 'unit.dataset@channel'

    mpl.rcParams['figure.dpi']=100
    ttl='Significant functional correlation matrix\n{}\n{}-{}-{}-{}-{}\n({})'.format(op.basename(dp),test, p_th, n_consec_bins, fract_baseline, W_sd, corr_type)
    dataset_borders = list(np.nonzero(np.diff(get_ds_ids(peakChs[:,0])))[0]) if assert_multi(dp) else []
    fig=imshow_cbar(sfcm, origin='top', xevents_toplot=dataset_borders, yevents_toplot=dataset_borders, events_color=[0.5,0.5,0.5],events_lw=1,
                xvalues=None, yvalues=None, xticks=tks, yticks=tks,
                xticklabels=labs, yticklabels=labs, xlabel=lab, ylabel=lab, title=ttl,
                cmapstr="RdBu_r", vmin=vminmax[0], vmax=vminmax[1], center=0, colorseq='nonlinear',
                clabel='Crosscorr. modulation (s.d.)', extend_cmap='neither', cticks=None,
                figsize=figsize, aspect='auto', function='imshow',
                ax=None)

    ax=fig.axes[0]
    ax.plot(ax.get_xlim(), ax.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
    [ax.spines[sp].set_visible(True) for sp in ['left', 'bottom', 'top', 'right']]

    if not ticks:
        [tick.set_visible(False) for tick in ax.xaxis.get_major_ticks()]
        [tick.set_visible(False) for tick in ax.yaxis.get_major_ticks()]

    if any(regions):
        xl,yl=ax.get_xlim(), ax.get_ylim()
        if reg_colors=={}:
            reg_colors={k:(1,1,1) for k in regions.keys()}
        for region, rng in regions.items():
            rngi=[np.argmin(abs(r-ch)) for r in rng[::-1]]
            ax.plot([rngi[0]-0.5,rngi[0]-0.5], [yl[0],yl[1]], ls="-", c=[0.5,0.5,0.5], lw=1)
            ax.plot([rngi[1]+0.5,rngi[1]+0.5], [yl[0],yl[1]], ls="-", c=[0.5,0.5,0.5], lw=1)
            ax.plot([xl[0],xl[1]], [rngi[0]-0.5,rngi[0]-0.5], ls="-", c=[0.5,0.5,0.5], lw=1)
            ax.plot([xl[0],xl[1]], [rngi[1]+0.5,rngi[1]+0.5], ls="-", c=[0.5,0.5,0.5], lw=1)
            rect_y = mpl.patches.Rectangle((xl[0],rngi[0]-0.5), 1, np.diff(rngi)+1, linewidth=1, edgecolor=(0,0,0,0), facecolor=reg_colors[region])
            rect_x = mpl.patches.Rectangle((rngi[0]-0.5, yl[0]-1), np.diff(rngi)+1, 1, linewidth=1, edgecolor=(0,0,0,0), facecolor=reg_colors[region])
            ax.add_patch(rect_y)
            ax.add_patch(rect_x)
            ax.text(x=2, y=rngi[0]+np.diff(rngi)/2, s=region, c=reg_colors[region], fontsize=18, fontweight='bold', rotation=90, va='center')

    if markers:
        for i in range(sfcm.shape[0]):
            for j in range(sfcm.shape[0]):
                if i!=j:
                    ccgi=(gu[i]==sfc['uSrc'])&(gu[j]==sfc['uTrg'])
                    if np.any(ccgi):
                        pkT = sfc.loc[ccgi, 't_ms']
                        if pkT>0.5:
                            ax.scatter(j, i, marker='>', s=20, c="black")
                        elif pkT<-0.5:
                            ax.scatter(j, i, marker='<', s=20, c="black")
                        elif -0.5<=pkT and pkT<=0.5:
                            ax.scatter(j, i, marker='o', s=20, c="black")
    if text:
        for i in range(sfcm.shape[0]):
            for j in range(sfcm.shape[0]):
                ccgi=(gu[i]==sfc['uSrc'])&(gu[j]==sfc['uTrg'])
                if np.any(ccgi):
                    pkT = sfc.loc[ccgi, 't_ms']
                    if i!=j and (min(pkT)<=0 or max(pkT)>0):
                        ax.text(x=j, y=i, s=str(pkT), size=12)

    if saveFig:
        if saveDir is None: saveDir=dp
        save_mpl_fig(fig, ttl.replace('\n', '_'), saveDir, _format)

    return fig

# def plot_sfcm_old(dp, corr_type='connections', metric='amp_z', cbin=0.5, cwin=100,
#               p_th=0.02, n_consec_bins=3, fract_baseline=4./5, W_sd=10, test='Poisson_Stark',
#               drop_seq=['sign', 'time', 'max_amplitude'], units=None, name=None,
#               text=False, markers=False, ticks=True, depth_ticks=False,
#               regions={}, reg_colors={}, vminmax=[-7,7], figsize=(7,7),
#               saveFig=False, saveDir=None, again=False, againCCG=False, use_template_for_peakchan=False):
#     '''
#     Visually represents the connectivity datafrane outputted by 'gen_sfc'.
#     Each line/row is a good unit.
#     Each intersection is a square split in a varying amount of columns,
#     each column representing a positive or negatively significant peak collored accordingly to its size s.
#     '''

#     sfc, sfcm, peakChs = gen_sfc(dp, corr_type, metric, cbin, cwin,
#                                  p_th, n_consec_bins, fract_baseline, W_sd, test,
#                                  again, againCCG, drop_seq, units, name,
#                                  cross_cont_proof=False, use_template_for_peakchan=use_template_for_peakchan)

#     gu = peakChs[:,0]
#     ch = peakChs[:,1].astype(int)

#     if corr_type=='synchrony':
#         vminmax=[0,vminmax[1]]
#     elif corr_type=='excitations':
#         vminmax=[0,vminmax[1]]
#     elif corr_type=='inhibitions':
#         vminmax=[vminmax[0],0]

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
#     axpos=ax.get_position()
#     cbar_ax = fig.add_axes([axpos.x0+axpos.width+0.01, axpos.y0, .02, .3])
#     sns.heatmap(sfcm, yticklabels=True, xticklabels=True, cmap="RdBu_r", center=0, vmin=vminmax[0], vmax=vminmax[1],
#                      cbar_kws={'label': 'Crosscorr. modulation (s.d.)'}, ax=ax, cbar_ax=cbar_ax)
#     cbar_ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(family='arial',weight='bold', size=12))
#     cbar_ax.yaxis.label.set_rotation(90)
#     cbar_ax.yaxis.label.set_va('top')
#     cbar_ax.yaxis.labelpad=5
#     cbar_ax.yaxis.set_ticklabels(cbar_ax.yaxis.get_ticklabels(), ha='center')
#     cbar_ax.yaxis.set_tick_params(pad=11)
#     set_ax_size(ax,*figsize)

#     ax.plot(ax.get_xlim(), ax.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
#     ttl='Significant functional correlation matrix\n{}\n{}-{}-{}-{}-{}\n({})'.format(op.basename(dp),test, p_th, n_consec_bins, fract_baseline, W_sd, corr_type)
#     ax.set_title(ttl, fontsize=16, fontweight='bold')

#     if depth_ticks:
#         labs=['{}'.format(3840-ch[i]*10) for i in range(len(gu)) if i%10==0]
#         tks=[i for i in range(len(gu)) if i%10==0]
#         ax.set_xticks(tks)
#         ax.set_yticks(tks)
#         ax.set_yticklabels(labs, fontsize=14, fontweight='bold', rotation=45)
#         ax.set_xticklabels(labs, fontsize=14, fontweight='bold', rotation=45)
#         ax.set_ylabel('Depth on probe (\u03BCm)', fontsize=16, fontweight='bold')
#         ax.set_xlabel('Depth on probe (\u03BCm)', fontsize=16, fontweight='bold')
#     else:
#         labs=['{}@{}'.format(gu[i], ch[i]) for i in range(len(gu))]
#         ax.set_yticklabels(labs, fontsize=12, fontweight='regular')
#         ax.set_xticklabels(labs, fontsize=12, fontweight='regular')

#     [ax.spines[sp].set_visible(True) for sp in ['left', 'bottom', 'top', 'right']]

#     if not ticks:
#         [tick.set_visible(False) for tick in ax.xaxis.get_major_ticks()]
#         [tick.set_visible(False) for tick in ax.yaxis.get_major_ticks()]

#     if any(regions):
#         for region, rng in regions.items():
#             rngi=[np.nonzero(abs(r-ch)==min(abs(r-ch)))[0][0] for r in rng[::-1]]
#             for r in rngi:
#                 ax.plot([r,r], [0,len(ch)], ls="-", c=[0.5,0.5,0.5], lw=1)
#                 ax.plot([0,len(ch)], [r,r], ls="-", c=[0.5,0.5,0.5], lw=1)
#             ax.plot(rngi,[len(ch),len(ch)], ls="-", c=reg_colors[region], lw=10, solid_capstyle='butt')
#             ax.plot([0,0], rngi, ls="-", c=reg_colors[region], lw=10, solid_capstyle='butt')
#             ax.text(x=2, y=rngi[0]+np.diff(rngi)/2, s=region, c=reg_colors[region], fontsize=18, fontweight='bold', rotation=90, va='center')

#     if markers:
#         for i in range(sfcm.shape[0]):
#             for j in range(sfcm.shape[0]):
#                 if i!=j:
#                     ccgi=(gu[i]==sfc['uSrc'])&(gu[j]==sfc['uTrg'])
#                     if np.any(ccgi):
#                         pkT = sfc.loc[ccgi, 't_ms']
#                         if pkT>0.5:
#                             ax.scatter(j, i, marker='>', s=20, c="black")
#                         elif pkT<-0.5:
#                             ax.scatter(j, i, marker='<', s=20, c="black")
#                         elif -0.5<=pkT and pkT<=0.5:
#                             ax.scatter(j, i, marker='o', s=20, c="black")
#     if text:
#         for i in range(sfcm.shape[0]):
#             for j in range(sfcm.shape[0]):
#                 ccgi=(gu[i]==sfc['uSrc'])&(gu[j]==sfc['uTrg'])
#                 if np.any(ccgi):
#                     pkT = sfc.loc[ccgi, 't_ms']
#                     if i!=j and (min(pkT)<=0 or max(pkT)>0):
#                         ax.text(x=j, y=i, s=str(pkT), size=12)

#     if saveFig:
#         if saveDir is None: saveDir=dp
#         fig.savefig(Path(saveDir,ttl.replace('\n', '_')+'.pdf'))

#     return fig

#%% Save matplotlib animations
# https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1
##### TO CREATE A SERIES OF PICTURES

def make_views(ax,angles,width, height, elevation=None,
                prefix='tmprot_',**kwargs):
    """
    Makes jpeg pictures of the given 3d ax, with different angles.
    Args:
        ax (3D axis): te ax
        angles (list): the list of angles (in degree) under which to
                       take the picture.
        width,height (float): size, in inches, of the output images.
        prefix (str): prefix for the files created.

    Returns: the list of files created (for later removal)
    """

    files = []
    ax.figure.set_size_inches(width,height)

    for i,angle in enumerate(angles):

        ax.view_init(elev = elevation, azim=angle)
        ax.set_xlim3d([206, 212])
        ax.set_ylim3d([208, 213])
        ax.set_zlim3d([207, 213])
        fname = '%s%03d.png'%(prefix,i)
        ax.figure.savefig(fname)
        files.append(fname)

    return files



##### TO TRANSFORM THE SERIES OF PICTURE INTO AN ANIMATION

def make_movie(files,output, fps=10,bitrate=1800,**kwargs):
    """
    Uses mencoder, produces a .mp4/.ogv/... movie from a list of
    picture files.
    """

    output_name, output_ext = os.path.splitext(output)
    command = { '.mp4' : 'mencoder "mf://%s" -mf fps=%d -o %s.mp4 -ovc lavc\
                         -lavcopts vcodec=msmpeg4v2:vbitrate=%d'
                         %(",".join(files),fps,output_name,bitrate)}

    command['.ogv'] = command['.mp4'] + '; ffmpeg -i %s.mp4 -r %d %s'%(output_name,fps,output)

    print(command[output_ext])
    output_ext = os.path.splitext(output)[1]
    os.system(command[output_ext])



def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """

    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))




def make_strip(files,output,**kwargs):
    """
    Uses imageMagick to produce a .jpeg strip from a list of
    picture files.
    """

    os.system('montage -tile 1x -geometry +0+0 %s %s'%(" ".join(files),output))



##### MAIN FUNCTION

def rotanimate(ax, width, height, angles, output, **kwargs):
    """
    Produces an animation (.mp4,.ogv,.gif,.jpeg,.png) from a 3D plot on
    a 3D ax

    Args:
        ax (3D axis): the ax containing the plot of interest
        angles (list): the list of angles (in degree) under which to
                       show the plot.
        output : name of the output file. The extension determines the
                 kind of animation used.
        **kwargs:
            - width : in inches
            - heigth: in inches
            - framerate : frames per second
            - delay : delay between frames in milliseconds
            - repeat : True or False (.gif only)
    """

    output_ext = os.path.splitext(output)[1]

    files = make_views(ax,angles, width, height, **kwargs)

    D = { '.mp4' : make_movie,
          '.ogv' : make_movie,
          '.gif': make_gif ,
          '.jpeg': make_strip,
          '.png':make_strip}

    D[output_ext](files,output,**kwargs)

    for f in files:
        os.remove(f)

def make_mpl_animation(ax, Nangles, delay, width=10, height=10, saveDir='~/Downloads', title='movie', frmt='gif', axis=True):
    '''
    ax is the figure axes that you will make rotate along its vertical axis,
    on Nangles angles (default 300),
    separated by delay time units (default 2),
    at a resolution of widthxheight pixels (default 10x10),
    saved in saveDir directory (default Downloads) with the title title (default movie) and format frmt (gif).
    '''
    assert frmt in ['gif', 'mp4', 'ogv']
    if not axis: plt.axis('off') # remove axes for visual appeal
    oldDir=os.getcwd()
    saveDir=op.expanduser(saveDir)
    os.chdir(saveDir)
    angles = np.linspace(0,360,Nangles)[:-1] # Take 20 angles between 0 and 360
    ttl='{}.{}'.format(title, frmt)
    rotanimate(ax, width, height, angles,ttl, delay=delay)

    os.chdir(oldDir)


def plot_filtered_times(dp, unit, first_n_minutes=20, consecutive_n_seconds = 180, acg_window_len=3, acg_chunk_size = 10, gauss_window_len = 3, gauss_chunk_size = 10, use_or_operator = False):
    unit_size_s = first_n_minutes * 60

    goodsec, acgsec, gausssec = train_quality(dp, unit, first_n_minutes, consecutive_n_seconds, acg_window_len, acg_chunk_size, gauss_window_len, gauss_chunk_size, use_or_operator)

    good_sec = []
    for i in goodsec:
        good_sec.append(list(range(i[0], i[1]+1)))
    good_sec = np.hstack((good_sec))

    acg_sec = []
    for i in acgsec:
        acg_sec.append(list(range(i[0], i[1]+1)))
    acg_sec = np.hstack((acg_sec))

    gauss_sec = []
    for i in gausssec:
        gauss_sec.append(list(range(i[0], i[1]+1)))
    gauss_sec = np.hstack((gauss_sec))

    # Parameters
    fs = read_spikeglx_meta(dp, 'ap')['sRateHz']

    samples_fr = unit_size_s * fs
    spike_clusters = np.load(dp/'spike_clusters.npy')
    amplitudes_sample = np.load(dp/'amplitudes.npy')  # shape N_tot_spikes x 1
    spike_times = np.load(dp/'spike_times.npy')  # in samples

    amplitudes_unit = amplitudes_sample[spike_clusters == unit]
    spike_times_unit = spike_times[spike_clusters == unit]
    unit_mask_20 = (spike_times_unit <= samples_fr)
    spike_times_unit_20 = spike_times_unit[unit_mask_20]
    amplitudes_unit_20 = amplitudes_unit[unit_mask_20]


    plt.figure()
    plt.plot(spike_times_unit_20/fs, amplitudes_unit_20, '.', alpha = 0.5)
    plt.text(0, 3,'Gaussian FN', fontsize = 5, color = 'blue')
    plt.text(0, 1,'FP + FN', fontsize = 5, color = 'green')
    plt.text(0, -3,'ACG FP', fontsize = 5, color = 'red')
    plt.title(f"Amplitudes in first 20 min for {unit}")

    for i in good_sec:
        s_time, e_time = i ,(i+1)
        plt.hlines(0, s_time, e_time, color = 'green')
#     # find the longest consecutive section
# # check if this is longer than 3 minutes, 18 sections
#
    for i in acg_sec:
        s_time, e_time = i ,(i+1)
        plt.hlines(-2, s_time, e_time, color = 'red')
#
    for i in gauss_sec:
        s_time, e_time = i ,(i+1)
        plt.hlines(2, s_time, e_time, color = 'blue')
    plt.show()

#%% How to plot 2D things with pyqtplot



# #QtGui.QApplication.setGraphicsSystem('raster')
# app = QtGui.QApplication([])
# #mw = QtGui.QMainWindow()
# #mw.resize(800,800)

# win = pg.GraphicsWindow(title="Basic plotting examples")
# win.resize(1000,600)
# win.setWindowTitle('pyqtgraph example: Plotting')

# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)

# p1 = win.addPlot(title="Basic array plotting", y=np.random.normal(size=100))

# p2 = win.addPlot(title="Multiple curves")
# p2.plot(np.random.normal(size=100), pen=(255,0,0), name="Red curve")
# p2.plot(np.random.normal(size=110)+5, pen=(0,255,0), name="Green curve")
# p2.plot(np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")

# p3 = win.addPlot(title="Drawing with points")
# p3.plot(np.random.normal(size=100), pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')


# win.nextRow()

# p4 = win.addPlot(title="Parametric, grid enabled")
# x = np.cos(np.linspace(0, 2*np.pi, 1000))
# y = np.sin(np.linspace(0, 4*np.pi, 1000))
# p4.plot(x, y)
# p4.showGrid(x=True, y=True)

# p5 = win.addPlot(title="Scatter plot, axis labels, log scale")
# x = np.random.normal(size=1000) * 1e-5
# y = x*1000 + 0.005 * np.random.normal(size=1000)
# y -= y.min()-1.0
# mask = x > 1e-15
# x = x[mask]
# y = y[mask]
# p5.plot(x, y, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
# p5.setLabel('left', "Y Axis", units='A')
# p5.setLabel('bottom', "Y Axis", units='s')
# p5.setLogMode(x=True, y=False)

# p6 = win.addPlot(title="Updating plot")
# curve = p6.plot(pen='y')
# data = np.random.normal(size=(10,1000))
# ptr = 0
# def update():
#     global curve, data, ptr, p6
#     curve.setData(data[ptr%10])
#     if ptr == 0:
#         p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
#     ptr += 1
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)


# win.nextRow()

# p7 = win.addPlot(title="Filled plot, axis disabled")
# y = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(size=1000, scale=0.1)
# p7.plot(y, fillLevel=-0.3, brush=(50,50,200,100))
# p7.showAxis('bottom', False)


# x2 = np.linspace(-100, 100, 1000)
# data2 = np.sin(x2) / x2
# p8 = win.addPlot(title="Region Selection")
# p8.plot(data2, pen=(255,255,255,200))
# lr = pg.LinearRegionItem([400,700])
# lr.setZValue(-10)
# p8.addItem(lr)

# p9 = win.addPlot(title="Zoom on selected region")
# p9.plot(data2)
# def updatePlot():
#     p9.setXRange(*lr.getRegion(), padding=0)
# def updateRegion():
#     lr.setRegion(p9.getViewBox().viewRange()[0])
# lr.sigRegionChanged.connect(updatePlot)
# p9.sigXRangeChanged.connect(updateRegion)
# updatePlot()
#from npyx.spk_t import trn
#from npyx.corr import acg, ccg, gen_sfc, get_ccg_sig, get_cm
