# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os
import os.path as op; opj=op.join
from pathlib import Path

import pickle as pkl

import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator

import holoviews as hv
import bokeh as bk

import seaborn as sns

from npyx.utils import phyColorsDic, DistinctColors20, npa, zscore, isnumeric
from npyx.stats import fractile_normal, fractile_poisson

from npyx.io import read_spikeglx_meta, extract_rawChunk, assert_chan_in_dataset, chan_map
from npyx.gl import get_units
from npyx.spk_wvf import get_depthSort_peakChans, wvf, get_peak_chan, templates
from npyx.spk_t import trn
from npyx.corr import acg, ccg, gen_sfc, get_ccg_sig, get_cm
from npyx.behav import align_times, get_processed_ifr, get_events
from mpl_toolkits.mplot3d import Axes3D

# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg

import networkx as nx

#%% plotting utilities

def save_mpl_fig(fig, figname, saveDir, _format):
    saveDir=Path(saveDir)
    if not saveDir.exists():
        assert saveDir.parent.exists(), f'WARNING can only create a path of a single directory level, {saveDir.parent} must exist already!'
        saveDir.mkdir()
    fig.savefig(saveDir/f"{figname}.{_format}", dpi=300, bbox_inches='tight')
            

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
    

def myround(x, base=5):
    return base * np.round(x/base)

def myceil(x, base=5):
    return base * np.ceil(x/base)

def myfloor(x, base=5):
    return base * np.floor(x/base)

def get_bestticks_from_array(arr, step=None):
    span=arr[-1]-arr[0]
    if step is None:
        upper10=10**np.ceil(np.log10(span))
        if span<=upper10/5:
            step=upper10/100
        elif span<=upper10/2:
            step=upper10/10/2
        else:
            step=upper10/10
    assert step<span, 'Step {step} is too large for array span {span}!'
    ticks=np.arange(myceil(arr[0],step),myfloor(arr[-1],step)+step,step)
    if step==int(step):ticks=ticks.astype(int)
    return ticks

def get_labels_from_ticks(ticks):
    ticks=npa(ticks)
    nflt=0
    for i, t in enumerate(ticks):
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
         xticks=None, yticks=None, xtickslabels=None, ytickslabels=None, reset_xticks=False, reset_yticks=False, xtickrot=0, ytickrot=0,
         axlab_w='bold', axlab_s=20,
         ticklab_w='regular', ticklab_s=16, ticks_direction='out', lw=2,
         title=None, title_w='bold', title_s=24,
         hide_top_right=True, hide_axis=False):
    '''
    make plots pretty
    matplotlib plots
    '''
    if fig is None: fig=plt.gcf()
    if ax is None: ax=fig.axes[0]
    hfont = {'fontname':'Arial'}
    if figsize is not None:
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
    # Opportunity to easily hide everything
    if hide_axis:
        ax.axis('off')
        return fig, ax
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
        assert len(xtickslabels)==len(xticks), 'WARNING you provided too many/few xtickslabels!'
        ax.set_xticklabels(xtickslabels, fontsize=ticklab_s, fontweight=ticklab_w, color=(0,0,0), **hfont, rotation=xtickrot)
    if ytickslabels is not None:
        assert len(ytickslabels)==len(yticks), 'WARNING you provided too many/few ytickslabels!'
        ax.set_yticklabels(ytickslabels, fontsize=ticklab_s, fontweight=ticklab_w, color=(0,0,0), **hfont, rotation=ytickrot)
    
    # Title
    if title is None: title=ax.get_title()
    ax.set_title(title, size=title_s, weight=title_w)
    
    # Ticks and spines aspect
    ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=lw, length=4, direction=ticks_direction)
    if hide_top_right: [ax.spines[sp].set_visible(False) for sp in ['top', 'right']]
    else: [ax.spines[sp].set_visible(True) for sp in ['top', 'right']]
    for sp in ['left', 'bottom', 'top', 'right']:
        ax.spines[sp].set_lw(lw)
    
    # Alignement and spacing elements
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axis_to_align = [AX for AX in fig.axes if AX.__repr__().split('.')[3].split(' ')[0]=='AxesSubplot']
    fig.align_ylabels(axis_to_align)
    fig.align_xlabels(axis_to_align)
        
    return fig, ax

    
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
    ax.bar(x=x, height=y, width=s, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel) if len(xlabel)>0 else ax.set_xlabel('Binsize:{}'.format(s))
    ax.set_ylabel(ylabel) if len(ylabel)>0 else ax.set_ylabel('Counts')
    
    fig,ax=mplp(fig,ax)
    
    return fig


#%% Stats related plots
    
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

#%% Waveforms or raw data

def plot_wvf(dp, u, Nchannels=8, chStart=None, n_waveforms=100, t_waveforms=2.8, ignore_nwvf=True, bpfilter=False, whiten=False, med_sub=False, again=False, subset_selection='regular',
               title = '', plot_std=True, plot_mean=True, plot_templates=False, color=phyColorsDic[0],
               labels=True, sample_lines='all', ylim=[0,0], saveDir='~/Downloads', saveFig=False, saveData=False, _format='pdf', ax=None):
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
        - color: (r,g,b) tuple or [0 to 1] floats, color of the mean waveform | default black
        - sample_lines: 'all' or int, whether to plot all or sample_lines individual samples in the background. Set to 0 to plot nothing.
        - labels: boolean, whether to plot or not the axis, axis labels, title... If False, only lines are plotted
        - ylim: upper limit of plots, in uV
        - saveDir  | default False
        - saveFig: boolean, save figure source data to saveDir | default Downloads
        - saveData: boolean, save waveforms source data to saveDir | default Downloads
    Returns:
        - matplotlib figure with Nchannels subplots, plotting the mean
    '''

        
    fs=read_spikeglx_meta(dp, subtype='ap')['sRateHz']
    cm=chan_map(dp, y_orig='surface', probe_version='local')
    peak_chan=get_peak_chan(dp, u)
    peak_chan_i = int(np.nonzero(np.abs(cm[:,0]-peak_chan)==min(np.abs(cm[:,0]-peak_chan)))[0][0]);
    t_waveforms_s=int(t_waveforms*(fs/1000))
    waveforms=wvf(dp, u, n_waveforms, t_waveforms_s, subset_selection=subset_selection, wvf_batch_size=10, again=again,
                  ignore_nwvf=ignore_nwvf, bpfilter=bpfilter, whiten=whiten, med_sub=med_sub)
    tplts=templates(dp, u)
    if waveforms.shape[0]==0:
        raise ValueError('No waveforms were found in the provided subset_selection!')
    assert waveforms.shape[1:]==(t_waveforms_s, cm.shape[0])
    
    if type(sample_lines) is str:
        assert sample_lines=='all'
        sample_lines=min(waveforms.shape[0],n_waveforms)
    elif type(sample_lines) is float or type(sample_lines) is int:
        sample_lines=min(waveforms.shape[0],sample_lines, n_waveforms)
    
    saveDir=op.expanduser(saveDir)
    if Nchannels>=2:
        Nchannels=Nchannels+Nchannels%2
        fig, ax = plt.subplots(int(Nchannels*1./2), 2, figsize=(8, 2*Nchannels), dpi=80)
        if Nchannels==2:
            ax=ax.reshape((1, 2))#to handle case of 2 subplots
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=80)
        ax=np.array(ax).reshape((1, 1))
    if chStart is None:
        chStart_i = peak_chan_i-Nchannels//2
        chStart=cm[chStart_i,0]
    else:
        chStart_i = int(np.nonzero(np.abs(cm[:,0]-chStart)==min(np.abs(cm[:,0]-chStart)))[0][0]) # if not all channels were processed by kilosort,
    chStart_i=int(min(chStart_i, waveforms.shape[2]-Nchannels-1))
    chEnd_i = int(chStart_i+Nchannels)
    pci_rel=peak_chan_i-chStart_i

    data = waveforms[:, :, chStart_i:chEnd_i]
    data=data[~np.isnan(data[:,0,0]),:,:] # filter out nan waveforms
    datam = np.rollaxis(data.mean(0),1)
    datastd = np.rollaxis(data.std(0),1)
    tplts=tplts[:, :, chStart_i:chEnd_i]

    color_dark=(max(color[0]-0.08,0), max(color[1]-0.08,0), max(color[2]-0.08,0))
    ylim1, ylim2 = (np.nanmin(datam-datastd)-50, np.nanmax(datam+datastd)+50) if ylim==[0,0] else (ylim[0], ylim[1])
    x = np.linspace(0, data.shape[1]/(fs/1000), data.shape[1]) # Plot t datapoints between 0 and t/30 ms
    x_tplts = x[(data.shape[1]-tplts.shape[1])//2:(data.shape[1]-tplts.shape[1])//2+tplts.shape[1]] # Plot 82 datapoints between 0 and 82/30 ms
    for i in range(data.shape[2]):
        i1, i2 = max(0,data.shape[2]//2-1-i//2), i%2
        ax[i1, i2].set_ylim([ylim1, ylim2])
        for j in range(sample_lines):
            ax[i1, i2].plot(x, data[j,:, i], linewidth=0.3, alpha=0.3, color=color)
        #r, c = int(Nchannels*1./2)-1-(i//2),i%2
        if plot_templates:
            tpl_scalings=[(max(datam[pci_rel, :])-min(datam[pci_rel, :]))/(max(tpl[:,pci_rel])-min(tpl[:,pci_rel])) for tpl in tplts]
            for tpl_i, tpl in enumerate(tplts):
                ax[i1, i2].plot(x_tplts, tpl[:,i]*tpl_scalings[tpl_i], linewidth=1, color=(0,0,0), alpha=0.4)
        if plot_mean:
            ax[i1, i2].plot(x, datam[i, :], linewidth=2, color=color_dark, alpha=1)
        if plot_std:
            ax[i1, i2].plot(x, datam[i, :]+datastd[i,:], linewidth=1, color=color, alpha=0.5)
            ax[i1, i2].plot(x, datam[i, :]-datastd[i,:], linewidth=1, color=color, alpha=0.5)
            ax[i1, i2].fill_between(x, datam[i, :]-datastd[i,:], datam[i, :]+datastd[i,:], facecolor=color, interpolate=True, alpha=0.2)
        ax[i1, i2].spines['right'].set_visible(False)
        ax[i1, i2].spines['top'].set_visible(False)
        ax[i1, i2].spines['left'].set_lw(2)
        ax[i1, i2].spines['bottom'].set_lw(2)
        if labels:
            ax[i1, i2].set_title(cm[:,0][chStart_i+(2*(i//2)+i%2)], size=12, loc='right', weight='bold')
            ax[i1, i2].tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=2, length=6, labelsize=14)
            if i2==0 and i1==max(0,int(Nchannels/2)-1):#start plotting from top
                ax[i1, i2].set_ylabel('EC V (\u03bcV)', size=14, weight='bold')
                ax[i1, i2].set_xlabel('t (ms)', size=14, weight='bold')
            else:
                ax[i1, i2].set_xticklabels([])
                ax[i1, i2].set_yticklabels([])
        else:
            ax[i1, i2].axis('off')

    title = 'waveforms of {}'.format(u) if title=='' else title
    if labels: fig.suptitle(title, size=18, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95-0.07*(len(title.split('\n'))-1)])
    
    if saveFig:
        assert _format in ['png', 'pdf', 'eps']
        fig.savefig(Path(saveDir, title+'.{}'.format(_format)), format=_format)
    if saveData:
        np.save(Path(saveDir, title+'.npy'), waveforms)
    
    return fig

def plot_raw(dp, times=None, alignement_events=None, window=None, channels=np.arange(384), subtype='ap',
             offset=450, color='multi', lw=1,
             title=None, _format='pdf',  saveDir='~/Downloads', saveData=0, saveFig=0, figsize=(20,8),
             whiten=False, nRangeWhiten=None, med_sub=False, nRangeMedSub=None, hpfilt=0, hpfiltf=300, ignore_ks_chanfilt=0,
             plot_ylabels=True, show_allyticks=0, yticks_jump=50,
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
    assert type(events) is list
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
        if color=='multi':
            color=None
        if as_heatmap:
            xticklabels = get_bestticks_from_array(t, step=None)
            xticks=xticklabels*fs/1000
            y_ticks_labels=npa([x*10 if x%2==0 else x*10-10 for x in y_ticks_labels])
            
            fig=imshow_cbar(im=rc, origin='top', events_toplot=[], events_color='k',
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
            for i in np.arange(rc.shape[0]):
                y=i*offset
                ax.plot([t[0,0], t[0,-1]], [y, y], color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
            ax.plot(t.T, rc.T, linewidth=lw, color=color)
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
    
    # # PyQt plotting if no matplotlib fig was returned
    # win = pg.GraphicsWindow(title="Raw data - {}-{}ms, channels {}-{}".format(times[0], times[1], channels[0], channels[-1]))
    # win.setBackground('w')
    # win.resize(1500,600)
    # p = win.addPlot()
    # p.setTitle("Raw data - {}-{}ms, channels {}-{}".format(times[0], times[1], channels[0], channels[-1]), color='k')
    # p.disableAutoRange()
    # # Enable antialiasing for prettier plots
    # pg.setConfigOptions(antialias=True)

    # for i in np.arange(rc.shape[0]):
    #     y=i*offset
    #     pen=pg.mkPen(color=(125,125,125), style=QtCore.Qt.DashLine, width=1.5)
    #     p.plot([0, t[0,-1]], [y, y], pen=pen)
    # for e in events:
    #     p.plot([e,e], [p.rect().getCoords()[1], p.rect().getCoords()[3]], color=(0.3, 0.3, 0.3), linestyle='--', linewidth=1.5)
    # if color=='multi':
    #     color=[DistinctColors20[ci%(len(DistinctColors20)-1)] for ci in range(rc.shape[0])]
    # else:
    #     if color in ['k', 'black']:
    #         color=[(0,0,0)]*rc.shape[0]
    #     else:
    #         assert npa(color).shape[0]==3
    #         color=[npa(color)]*rc.shape[0]
            
    # for line in range(rc.shape[0]):
    #     pen=pg.mkPen(color=tuple(npa(color[line])*255), width=1)
    #     p.plot(t[line,:].T, rc[line,:].T, pen=pen)
    # pen=pg.mkPen(color=(0,0,0), width=2)
    # p.getAxis('left').setTicks([[(y_ticks[i], y_ticks_labels[i]) for i in range(len(y_ticks))],[]])
    # p.getAxis('bottom').setLabel('Time (ms)')
    # p.getAxis('left').setLabel('Extracellular potential (\u03bcV)')
    # p.getAxis('left').setPen(pen)
    # p.getAxis('bottom').setPen(pen)
    # font=QtGui.QFont()
    # font.setPixelSize(14)
    # p.getAxis("bottom").setTickFont(font)
    # p.getAxis("left").setTickFont(font)
    # p.getAxis("bottom").setStyle(tickTextOffset = 5)
    # p.getAxis("left").setStyle(tickTextOffset = 5)
    # p.autoRange() # adding it only after having plotted everything makes it way faster
    
    # return win,p

def plot_raw_units(dp, times, units=[], channels=np.arange(384), offset=450,
                   Nchan_plot=5, spk_window=82, colors='phy', back_color='k', lw=1,
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
             _format=_format, color=back_color,
             whiten=whiten, nRangeWhiten=nRangeWhiten, med_sub=med_sub, nRangeMedSub=nRangeMedSub, hpfilt=hpfilt, hpfiltf=hpfiltf, ignore_ks_chanfilt=ignore_ks_chanfilt,
             show_allyticks=show_allyticks, yticks_jump=50, events=events, set0atEvent=set0atEvent, figsize=figsize,
             plot_ylabels=True, ax=None, title=title, lw=lw)
             
    if not pyqtgraph: ax=fig.get_axes()[0]
    assert type(units) is list
    assert len(units)>=1
    fs=read_spikeglx_meta(dp, 'ap')['sRateHz']
    spk_w1 = spk_window // 2
    spk_w2 = spk_window - spk_w1
    t1, t2 = int(np.round(times[0]*fs)), int(np.round(times[1]*fs))
    
    if colors=='phy':
        phy_c=list(phyColorsDic.values())[:-1]
        colors=[phy_c[ci%len(phy_c)] for ci in range(len(units))]
    else:
        assert type(colors) is str
        assert len(colors)==len(units), 'The length of the list of colors should be the same as the list of units!!'
    
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
        peakChan=get_peak_chan(dp,u)
        assert peakChan in channels, "WARNING the peak channel of {}, {}, is not in the set of channels plotted here!".format(u, peakChan)
        peakChan_rel=np.nonzero(peakChan==channels)[0][0]
        ch1, ch2 = max(0,peakChan_rel-Nchan_plot//2), min(rc.shape[0], peakChan_rel-Nchan_plot//2+Nchan_plot)
        t=trn(dp,u) # in samples
        twin=t[(t>t1+spk_w1)&(t<t2-spk_w2)] # get spikes starting minimum spk_w1 after window start and ending maximum spk_w2 before window end
        twin-=t1 # set t1 as time 0
        for t_spki, t_spk in enumerate(twin):
            print('plotting spike {}/{}...'.format(t_spki, len(twin)))
            spk_id=(tx>=t_spk-spk_w1)&(tx<=t_spk+spk_w2)
            color=colors[iu]
            if pyqtgraph:
                win,p = fig
                for line in np.arange(ch1, ch2, 1):
                    p.plot(tx_ms[line, spk_id].T, rc[line, spk_id].T, linewidth=1, pen=tuple(npa(color)*255))
                fig = win,p
            else:
                ax.plot(tx_ms[ch1:ch2, spk_id].T, rc[ch1:ch2, spk_id].T, lw=lw+0.1, color=color)
                #ax.plot(tx_ms[peakChan_rel, spk_id].T, rc[peakChan_rel, spk_id].T, lw=1.5, color=color)
                fig.tight_layout()
            
    if saveFig and not pyqtgraph:
        saveDir=op.expanduser(saveDir)
        rcn = '{}_{}_t{}-{}_ch{}-{}'.format(op.basename(dp), units, times[0], times[1], channels[0], channels[-1]) # raw chunk name
        rcn=rcn+'_whitened' if whiten else rcn+'_raw'
        if title is not None: rcn=title
        fig.savefig(Path(saveDir, '{}.{}'.format(rcn, _format)), format=_format)
    
    if pyqtgraph:fig[1].autoRange()
    return fig

#%% Peri-event time plots: rasters, psths...

def psth_plot(times, events, events_toplot=[0], events_color='r', psthb=5, psthw=[-1000, 1000], remove_empty_trials=False,
           title='', color='darkgreen', figsize=None,
           saveDir='~/Downloads', saveFig=0, saveData=0, _format='pdf',
           zscore=False, bsl_subtract=False, bsl_window=[-2000,-1000],
           convolve=True, gsd=2, xticks=None, xticklabels=None, xlabel='Time (ms)', ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
    
    x, y, y_p, y_p_var = get_processed_ifr(times, events, b=psthb, window=psthw, remove_empty_trials=remove_empty_trials,
                                      zscore=zscore, zscoretype='within', 
                                      convolve=convolve, gsd=gsd, method='gaussian_causal',
                                      bsl_subtract=bsl_subtract, bsl_window=bsl_window)
    areasteps=None if convolve else 'post'
    if zscore or bsl_subtract:
        ax.fill_between(x, y_p-y_p_var, y_p+y_p_var, color=color, alpha=0.7, step=areasteps)
    else:
        ax.fill_between(x, y_p-y_p_var, y_p+y_p_var, color=color, alpha=0.5, step=areasteps)
        ax.fill_between(x, y_p*0, y_p, color=color, alpha=1, step=areasteps)
    if convolve:
        if zscore or bsl_subtract: ax.plot(x, y_p-y_p_var, color='black', lw=0.5)
        ax.plot(x, y_p+y_p_var, color='black', lw=0.5)
        ax.plot(x, y_p, color='black', lw=1)
    else:
        if zscore or bsl_subtract: ax.step(x, y_p-y_p_var, color='black', lw=0.5, where='post')
        ax.step(x, y_p+y_p_var, color='black', lw=0.5, where='post')
        ax.step(x, y_p, color='black', lw=1,where='post')
    
    yl=ax.get_ylim()
    if not (zscore or bsl_subtract): yl=[0,yl[1]]
    for etp in events_toplot:
        ax.plot([etp,etp], yl, ls='--', lw=1, c=events_color)
        ax.set_ylim(yl)
        
    xl=psthw
    if bsl_subtract or zscore:
        ax.plot(xl,[0,0],lw=1,ls='--',c='black',zorder=-1)
        if zscore:
            if yl[0]<-2: ax.plot(xl,[-2,-2],lw=1,ls='--',c='red',zorder=-1)
            if yl[1]>2: ax.plot(xl,[2,2],lw=1,ls='--',c='red',zorder=-1)
    ax.set_xlim(xl)
        
    ylabel='IFR\n(zscore)' if zscore else r'$\Delta$ FR (Hz)' if bsl_subtract else 'IFR (Hz)'
    
    fig,ax=mplp(fig=fig, ax=ax, figsize=figsize,
     xlim=psthw, ylim=yl, xlabel=xlabel, ylabel=ylabel,
     xticks=xticks, xtickslabels=xticklabels,
     axlab_w='bold', axlab_s=20,
     ticklab_w='regular',ticklab_s=16, lw=1,
     title=None, title_w='bold', title_s=24,
     hide_top_right=True, hide_axis=False)
        
    if saveFig:
        figname=title
        save_mpl_fig(fig, figname, saveDir, _format)
    
    return fig

def raster_plot(times, events, events_toplot=[0], events_color='r', trials_toplot=[], window=[-1000, 1000], remove_empty_trials=False,
           title='', color='darkgreen', colorpalette="tab10", marker='|', malpha=0.9, size=None, lw=3, sparseylabels=True, figsize=None,
           saveDir='~/Downloads', saveFig=0, saveData=0, _format='pdf',
           as_heatmap=False, vmin=None, center=None, vmax=None, cmap_str=None,
           show_psth=False, psthb=10,
           zscore=False, bsl_subtract=False, bsl_window=[-2000,-1000],
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
        - saevFig: boolean, if 1 saves figure with name raster_title._format at saveDir
        - saveData: boolean, if 1 saves data as 2D array 2xlen(times), with first line being the event index and second line the relative timestamp time in seconds.
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
        else: color=sns.color_palette(colorpalette, n_cells).as_hex()
    else: assert len(color)==n_cells
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
    if sparseylabels and np.all(events_order==np.arange(events_order.shape[0])):
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
    xlabel_plot=xlabel if not show_psth else None
    if figsize is None: figsize=[5,subplots_ratio[0]*2]
    if show_psth: figsize[1]=figsize[1]+figsize[1]//subplots_ratio[0]
    for ci in range(n_cells):
        if as_heatmap:
            x, y, y_p, y_p_var = get_processed_ifr(times[ci], events, b=psthb, window=window, remove_empty_trials=remove_empty_trials,
                                      zscore=zscore, zscoretype='within', 
                                      convolve=convolve, gsd=gsd, method='gaussian_causal',
                                      bsl_subtract=bsl_subtract, bsl_window=bsl_window, process_y=True)
            if vmin is None: vmin = 0 if not (zscore|bsl_subtract) else -max(abs(0.9*y.min()),abs(0.9*y.max()))
            if center is None: center = 0.4*y.max() if not (zscore|bsl_subtract) else 0
            if vmax is None: vmax = 0.8*y.max() if not (zscore|bsl_subtract) else max(abs(0.9*y.min()),abs(0.9*y.max()))
            if cmap_str is None: cmap_str = 'viridis' if not (zscore|bsl_subtract) else 'RdBu_r'
            ntrials=y.shape[0]
            clab='Inst. firing rate (Hz)' if not zscore else 'Inst. firing rate (zscore)'
            imshow_cbar(y, origin='top', events_toplot=events_toplot, events_color=events_color,
                        xvalues=np.arange(window[0], window[1], psthb), yvalues=np.arange(ntrials)+1,
                        xticks=None, yticks=y_ticks,
                        xticklabels=None, yticklabels=y_ticks_labels, xlabel=xlabel_plot, ylabel='Trials', title=title,
                        cmapstr=cmap_str, vmin=vmin, vmax=vmax, center=center, colorseq='nonlinear',
                        clabel=clab, extend_cmap='neither', cticks=None,
                        figsize=figsize, aspect='auto', function='imshow', ax=ax)
            
        else:
            at, atb = align_times(times[ci], events, window=window, remove_empty_trials=remove_empty_trials)
            ntrials=len(at)
            col='black' if n_cells==1 else color[ci]
            for e, ts in at.items():
                i=events_order[np.nonzero(e==events)[0][0]]
                y=[y_ticks[i*n_cells+ci]]*len(ts)
                ts=npa(ts)*1000 # convert to ms
                ax.scatter(ts, y, s=size, c=col, alpha=malpha, marker=marker, lw=lw)
            fig,ax=mplp(fig=fig, ax=ax, figsize=figsize,
                 xlim=window, ylim=[y_ticks[-1]+1, 0], xlabel=xlabel_plot, ylabel="Trials",
                 xticks=None, yticks=y_ticks, xtickslabels=None, ytickslabels=y_ticks_labels,
                 axlab_w='bold', axlab_s=20,
                 ticklab_w='regular',ticklab_s=16, lw=1,
                 title=title, title_w='bold', title_s=24,
                 hide_top_right=True, hide_axis=False)
    print(f'{ntrials} trials.')
    xl=ax.get_xlim()
    yl=ax.get_ylim()
    for etp in events_toplot:
        ax.plot([etp,etp], yl, ls='--', lw=1, c=events_color)
    if any(trials_toplot):
        for ttp in trials_toplot:
            ax.plot(xl, [ttp,ttp], ls='--', lw=1, c='k')
    ax.set_ylim(yl)
    ax.set_xlim(xl)

    if show_psth:
        xticks=ax.get_xticks()
        xticklabels=get_labels_from_ticks(xticks)[0]
        ax.set_xticklabels([])
        for ci in range(n_cells):
            ax_psth=fig.add_subplot(grid[-n_cells+ci, :])
            xticklabels_subplot=xticklabels if ci==n_cells-1 else ['' for i in xticklabels]
            xlabel_subplot=xlabel if ci==n_cells-1 else None
            psth_plot(times[ci], events, events_toplot=events_toplot, events_color=events_color, psthb=psthb, psthw=window,
                     remove_empty_trials=remove_empty_trials,
                       title=None, color=color[ci], figsize=None,
                       saveDir=saveDir,
                       zscore=zscore, bsl_subtract=bsl_subtract, bsl_window=bsl_window,
                       convolve=convolve, gsd=gsd,
                       xticks=xticks, xticklabels=xticklabels_subplot, xlabel=xlabel_subplot,ax=ax_psth)
        
    if saveFig:
        figname=title
        save_mpl_fig(fig, figname, saveDir, _format)
        
    return fig

def summary_psth_wrap(dp, u_u_str, events_types, psthb=5, psthw=[-1000,1000], events_toplot=[0],
                      saveFig=0, saveDir='~/Downloads', _format='pdf',
                      zscore=False, bsl_subtract=False, bsl_window=[-2000, -1000],
                      convolve=True, gsd=2, figw=6, fig_wh_ratio=2, vspace=0.6,
                      ret_data=False, overlay=False, overlay_dim='events',
                      events_colors=None, trains_colors=None, order=['event','unit'], column=False):
    '''
    Usage example:
    dsi='DK153day3probe1'
    dp=DSs[dsi]['dp']
    u_u_str = [[573,'ss'],[613,'ss']]
    events_types=['rr', 'cr_r']
    '''
    if trains_colors is None: trains_colors={us[0]:sns.color_palette('muted',len(u_u_str)).as_hex()[usi] for usi,us in enumerate(u_u_str)}
    units=[]
    trains_str=[]
    trains_col=[]
    for us in u_u_str:
        units.append(us[0])
        trains_str.append(us[1]+' '+str(us[0]))
        trains_col.append(trains_colors[us[0]])
            
    trains=[trn(dp, u)/30000 for u in units]
    
    if events_colors is None: events_colors={e:sns.color_palette('muted',len(events_types)).as_hex()[ei] for ei,e in enumerate(events_types)}
    events=[]
    events_str=events_types
    events_col=[]
    for et in events_types:
        events.append(get_events(dp, et, f_behav=None, include_wheel_data=False, add_spont_licks=False,
                wheel_gain=3, rew_zone=12.5, rew_frames=3, vr_rate=30,
                wheel_diam=45, difficulty=2, ballistic_thresh=100, plot=False)/30000)
        events_col.append(events_colors[et])
        
    return summary_psth(trains, trains_str, events, events_str, psthb, psthw, events_toplot,
                      saveFig, saveDir, _format,
                      zscore, bsl_subtract, bsl_window,
                      convolve, gsd, figw, fig_wh_ratio, vspace,
                      ret_data, overlay, overlay_dim,
                      events_col, trains_col, order=order, column=column)


def summary_psth(trains, trains_str, events, events_str, psthb=5, psthw=[-1000,1000], events_toplot=[0],
                      saveFig=0, saveDir='~/Downloads', _format='pdf',
                      zscore=False, bsl_subtract=False, bsl_window=[-2000,-1000],
                      convolve=True, gsd=2, figw=6, fig_wh_ratio=2, vspace=0.6,
                      ret_data=False, overlay=False, overlay_dim='events',
                      events_col=None, trains_col=None, order=['event','unit'],
                      column=False):
    '''
    events: in s    
    
    
    events_col will be used if no overlay or overlay by event.
    trains_col will be used only if overlay by train.
    '''
    assert overlay_dim in ['trains', 'events']
    
    if ret_data: assert len(trains)==1 and len(events)==1, 'WARNING in order to use argument ret_data, you should plot a single PSTH, not a collection of them -> provide a single cell and event type.'
    
    assert 'event' in order and 'unit' in order, "WARNING order MUST a list containing 'unit' AND 'event' (either ['event', 'unit'], or ['unit', 'event']."
    
    if trains_col is None:
        trains_col =  sns.color_palette("tab10", len(trains)).as_hex()
    if events_col is None:
        events_col = sns.color_palette("tab10", len(events)).as_hex()
        
    assert len(trains)==len(trains_str)==len(trains_col)
    assert len(events)==len(events_str)==len(events_col)
    
    assert len(psthw)==2
    psthw=[psthw[0], psthw[1]+psthb]
    (lw1, lw2) = (0.5, 1) if (zscore or bsl_subtract) else (0.5, 1)
    
    # Populate dataframe
    en_str={}
    df=pd.DataFrame({'unit':[], 'event':[], 't':[], 'y':[], 'y_var1':[], 'y_var2':[], 'unit_c':[], 'event_c':[]})
    for ti, t in enumerate(trains):
        for ei, e in enumerate(events):
            x, y, y_p, y_p_var = get_processed_ifr(t, e, b=psthb, window=psthw, remove_empty_trials=True,
                                                      zscore=zscore, zscoretype='within', 
                                                      convolve=convolve, gsd=gsd, method='gaussian_causal',
                                                      bsl_subtract=bsl_subtract, bsl_window=bsl_window)
            n=len(x)
            u=trains_str[ti]
            en_str[events_str[ei]]=f' (n={len(e)})'
            e=events_str[ei]+en_str[events_str[ei]]
            c = trains_col[ti] if overlay and overlay_dim=='trains' else events_col[ei] # only use trains_col if there is an overlay by trains
            df=df.append(pd.DataFrame({'unit':[u]*n, 'event':[e]*n, 't':x, 'y':y_p, 'y_var1':y_p-y_p_var, 'y_var2':y_p+y_p_var, 'area_c':[c]*n}), ignore_index=True)
    df['0']=0 # to plot baselines in holoviews
        
    # Plot with holoviews
    ylabel='IFR (zscore)' if zscore else r'$\Delta$ FR (Hz)' if bsl_subtract else 'IFR (Hz)'
    alpha=0.7 if overlay else 1
    
    # hv.extension('matplotlib')
    interp='linear' if convolve else 'steps-post'
    mean=df.hvplot.line(x='t', y='y',
                        xlabel='Time(ms)', ylabel=ylabel,
                        groupby=order, dynamic=False, legend=False)
    mean.opts(linewidth=lw2, c='black', interpolation=interp, backend='matplotlib')
    var1=df.hvplot.line(x='t', y='y_var1',
                    groupby=order, dynamic=False, legend=False)
    var1.opts(linewidth=lw1, c='black', interpolation=interp, backend='matplotlib')
    var2=df.hvplot.line(x='t', y='y_var2',
                groupby=order, dynamic=False, legend=False)
    var2.opts(linewidth=lw1, c='black', interpolation=interp, backend='matplotlib')
    if convolve:
        if zscore or bsl_subtract:
            var12=df.hvplot.area(x='t', y='y_var1', y2='y_var2',
                                color='area_c',alpha=alpha, 
                                groupby=order, dynamic=False, legend=True)
        else:
            var12=df.hvplot.area(x='t', y='0', y2='y',
                        color='area_c',alpha=alpha,
                        groupby=order, dynamic=False, legend=True)
            var12bis=df.hvplot.area(x='t', y='y_var1', y2='y_var2',
                                color='grey',alpha=0.7,
                                groupby=order, dynamic=False, legend=False)
    else:
        if zscore or bsl_subtract:
            var12=df.hvplot.bar(x='t', y='y_var2',
                                color='area_c', fill_color='area_c', alpha=alpha, 
                                groupby=order, dynamic=False, legend=True)
        else:
            var12=df.hvplot.bar(x='t', y='y',
                        color='area_c', fill_color='area_c', alpha=alpha,
                        groupby=order, dynamic=False, legend=True)
            var12bis=df.hvplot.bar(x='t', y='y_var2',
                                color='grey', fill_color='grey', alpha=0.7,
                                groupby=order, dynamic=False, legend=False)
    mean.opts(fig_inches=figw, aspect=fig_wh_ratio) # will apply to all!
        
    # Compose Holomap
    if zscore or bsl_subtract:
        psth=var12*var1*var2*mean
    else:
        psth=var12*mean if overlay else var12bis*var12*var2*mean
        
    # Sort holomap so that subplots appear in correct order
    # Else, the order of subplots across dimensions will follow the keyword 'groupby' (here ['unit', 'event']
    # then the alphabetical order within dimensions (e.g. 'cr_r' then 'rr', even if not provided in this order)
    order_dic={'event':[es+en_str[es] for es in events_str],'unit':trains_str}
    psth_sorted=hv.HoloMap(kdims=order, sort=False)
    for i in order_dic[order[0]]:
        for j in order_dic[order[1]]:
            if (i,j) in list(psth.data.keys()): psth_sorted[(i,j)]=psth[(i,j)]
            elif (j,i) in list(psth.data.keys()): psth_sorted[(j,i)]=psth[(j,i)]
        
    # turn holomap into a plottable layout/overlay
    if overlay:
        ncols=1
        nrows = len(events) if overlay_dim == 'events' else len(trains)
    else:
        if order[0]=='event': ncols,nrows=len(trains),len(events)
        elif order[0]=='unit': ncols,nrows=len(events),len(trains)
    if column: ncols=1
        
    trsps=True if ncols>1 else False
    
    if overlay:
        if overlay_dim=='events':
            psth=psth_sorted.overlay('event', sort=False).layout('unit', sort=False).cols(ncols)
        elif overlay_dim=='trains':
            psth=psth_sorted.overlay('unit', sort=False).layout('event', sort=False).cols(ncols)
    else:
        psth=psth_sorted.layout(order, sort=False).cols(ncols)

    # Holoviews bug, does not transpose titles...
    # Need to first reconstruct them!!
    if trsps:
        kdims=[kd.label for kd in psth.kdims]
        keys=list(psth.data.keys())
        titles=[[f'{kdims[ki]}: {k}' for ki,k in enumerate(key)] for key in keys]
        titles=[', '.join(ttl) for ttl in titles]
        titles_grid=pd.DataFrame() # will have ncols rows and nrows columns
        for axi, ttl in enumerate(titles):
            i,j=axi//ncols,axi%ncols
            titles_grid.loc[i,j]=ttl.replace(', ','\n')
        titles_grid=titles_grid.T
        
    # Render figure
    psth.opts(transpose=trsps,vspace=0.6)
    fig = hv.render(psth, backend='matplotlib')
    
    # Add dashed landmarks and fine formatting in matplotlib
    for axi, ax in enumerate(fig.axes):
        if axi<len(fig.axes)-1:
            if ax.get_legend() is not None: ax.get_legend().remove()
        i,j=axi//nrows,axi%nrows
        ttl=ax.get_title().replace(', ','\n') if not trsps else titles_grid.loc[i,j]
        ax.set_title(ttl, loc='left')
        ax.set_title(None)
        yl=ax.get_ylim()
        if not (zscore or bsl_subtract): yl=(0,yl[1])
        for etp in events_toplot:
            ax.plot([etp,etp], yl, ls='--', lw=1, c='k')
        ax.set_ylim(yl)
        if bsl_subtract or zscore:
            xl=ax.get_xlim()
            ax.plot(xl,[0,0],lw=1,ls='--',c='black',zorder=-1)
            if zscore:
                if yl[0]<-2: ax.plot(xl,[-2,-2],lw=1,ls='--',c='red',zorder=-1)
                if yl[1]>2: ax.plot(xl,[2,2],lw=1,ls='--',c='red',zorder=-1)
            ax.set_xlim(xl)
        mplp(fig,ax, axlab_s=12, axlab_w='regular', ticklab_s=12)
    
    # Make pretty
    # fig.align_ylabels()
    # fig.tight_layout()
    mplshow(fig)
    
    # Save figure
    if saveFig:
        event_types_stack_str=''
        for es in events_str: event_types_stack_str+=es+'-'
        event_types_stack_str=event_types_stack_str[:-1]
        units_stack_str=''
        for us in trains_str: units_stack_str+=us+'-'
        units_stack_str=units_stack_str[:-1]
        figname=f"psth {units_stack_str}_{zscore}{bsl_subtract}_{event_types_stack_str}"
        save_mpl_fig(fig, figname, saveDir, _format)
    
    if ret_data:
        return x, y, y_p, y_p_var
    return fig

#%% Correlograms

def plt_ccg(uls, CCG, cbin=0.04, cwin=5, bChs=None, fs=30000, saveDir='~/Downloads', saveFig=True, 
            show=True, _format='pdf', subset_selection='all', labels=True, std_lines=True, title=None, color=-1, 
            saveData=False, ylim1=0, ylim2=0, normalize='Hertz', ccg_mn=None, ccg_std=None):
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
    if type(color)==int: # else, an actual color is passed
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

    if ccg_mn is not None and ccg_std is not None:
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
    
    ax.plot([0,0], ax.get_ylim(), ls="--", c=[0,0,0], lw=2)
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
        if type(title)!=str:
            if bChs is None:
                title="Units {}->{} ({})s".format(uls[0], uls[1], str(subset_selection)[0:50].replace(' ',  ''))
            else:
                title="Units {}@{}->{}@{} ({})s".format(uls[0], bChs[0], uls[1], bChs[1], str(subset_selection)[0:50].replace(' ',  ''))
        ax.set_title(title, size=22)
        ax.tick_params(labelsize=20)
    fig.tight_layout()
    plt.close() if not show else plt.show()
    if saveFig or saveData:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        if saveFig:
            fig.savefig(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}.{4}'.format(uls[0], uls[1], cwin, cbin,_format))
        if saveData:
            np.save(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}_x.npy'.format(uls[0], uls[1], cwin, cbin), x)
            np.save(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}_y.npy'.format(uls[0], uls[1], cwin, cbin), CCG)
        
    return fig
        
def plt_acg(unit, ACG, cbin=0.2, cwin=80, bChs=None, color=0, fs=30000, saveDir='~/Downloads', saveFig=True, 
            show=True, _format='pdf', subset_selection='all', labels=True, title=None, ref_per=True, saveData=False, 
            ylim1=0, ylim2=0, normalize='Hertz', acg_mn=None, acg_std=None):
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
    if type(color)==int: # else, an actual color is passed
        color=phyColorsDic[color]
    fig, ax = plt.subplots(figsize=(10,8))
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
    ax.set_ylim([ylim1, ylim2])

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
    ax.fill_between(x, y*0, y, color=color)
    ax.step(x, y, where='mid', color='black', lw=1)
    
    if labels:
        if normalize=='Counts':
            ax.set_ylabel("Autocorrelation (Counts)", size=20)
        if normalize=='Hertz':
            ax.set_ylabel("Autocorrelation (Hz)", size=20)
        elif normalize=='Pearson':
            ax.set_ylabel("Autocorrelation (Pearson)", size=20)
        elif normalize=='zscore':
            ax.set_ylabel("Autocorrelation (z-score)", size=20)
        ax.set_xlabel('Time (ms)', size=20)
        ax.set_xlim([-cwin*1./2, cwin*1./2])
        if type(title)!=str:
            if  bChs is None:
                title="Unit {} ({})s".format(unit, str(subset_selection)[0:50].replace(' ',  ''))
            else:
                assert len(bChs)==1
                title="Unit {}@{} ({})s".format(unit, bChs[0], str(subset_selection)[0:50].replace(' ',  ''))
        ax.set_title(title, size=22)
        ax.tick_params(labelsize=20)
        if ref_per:
            ax.plot([-1, -1], [ylim1, ylim2], color='black', linestyle='--', linewidth=1)
            ax.plot([1, 1], [ylim1, ylim2], color='black', linestyle='--', linewidth=1)
    mplp(fig, figsize=(9,8))
    
    plt.close() if not show else plt.show()
    if saveFig or saveData:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        if saveFig:
            fig.savefig(saveDir+'/acg{}-{}_{:.2f}.{}'.format(unit, cwin, cbin, _format))
        if saveData:
            np.save(saveDir+'/acg{}-{}_{:.2f}_x.npy'.format(unit, cwin, cbin), x)
            np.save(saveDir+'/acg{}-{}_{:.2f}_y.npy'.format(unit, cwin, cbin), ACG)

    return fig
        
    
def plt_ccg_subplots(units, CCGs, cbin=0.2, cwin=80, bChs=None, Title=None, saveDir='~/Downloads', 
                     saveFig=False, prnt=False, show=True, _format='pdf', subset_selection='all', 
                     labels=True, title=None, std_lines=False, ylim1=0, ylim2=0, normalize='zscore'):
    
    fig, ax = plt.subplots(len(units), len(units), figsize=(3*len(units), 3*len(units)), dpi=80)
    for i in range(len(units)):
        for j in range(len(units)):
            r, c = i, j
            x=np.arange(-cwin/2, cwin/2+cbin, cbin)
            if normalize!='mixte':normalize1=normalize
            if i>j:
                mplp(ax=ax[r, c], hide_axis=True)
            else:
                if (i==j):
                    color=phyColorsDic[i%6]
                    y=CCGs[i,j,:]
                    if normalize=='mixte':
                        normalize1='Hertz'
                else:
                    color=phyColorsDic[-1]
                    if normalize=='mixte':
                        y=zscore(CCGs[i,j,:], 4./5)
                        normalize1='zscore'
                    else:
                        y=CCGs[i,j,:]
                
                ax[r, c].plot(x, y, color=color, alpha=0)
                ax[r, c].set_xlim([-cwin*1./2, cwin*1./2])
                
                if normalize1 in ['Hertz','Pearson','Counts']:
                    ax[r, c].set_ylim([0, ax[r, c].get_ylim()[1]])
                    ax[r, c].fill_between(x, np.zeros(len(x)), y, color=color)
                elif normalize1=='zscore':
                    ylmax=max(np.abs(ax[r, c].get_ylim()))
                    ax[r, c].set_ylim([-ylmax, ylmax])
                    ax[r, c].fill_between(x, -ylmax*np.ones(len(x)), y, color=color)
                    
                if labels:
                    if j==0 and i==0:
                        ax[r, c].set_ylabel("Crosscorr. ({})".format(normalize), size=12)
                    if j==len(units)-1 and i==len(units)-1:
                        ax[r, c].set_xlabel('Time (ms)', size=12)
                    
                    if any(bChs):
                        title="{}@{}->{}@{}".format(units[i], bChs[i], units[j], bChs[j])
                    else:
                        title="{}->{}".format(units[i], units[j])
                    
                    mplp(ax=ax[r, c], figsize=(3*len(units), 3*len(units)),
                         title=title, title_s=12, title_w='regular',
                         axlab_s=12, axlab_w='regular',
                         ticklab_s=12, ticklab_w='regular')
    
    
    if Title:
        fig.suptitle(Title, size=20, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.close() if not show else plt.show()
    if saveFig:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        fig.savefig(saveDir+'/ccg{0}-{1}_{2:.2f}.{3}'.format(str(units).replace(' ', ''), cwin, cbin, _format))
        
    return fig

def plot_acg(dp, unit, cbin=0.2, cwin=80, normalize='Hertz', color=0, saveDir='~/Downloads', saveFig=True, prnt=False, show=True, 
             _format='pdf', subset_selection='all', labels=True, title=None, ref_per=True, saveData=False, ylim=[0,0], acg_mn=None, acg_std=None, again=False):
    assert type(unit)==int or type(unit)==str
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
            subset_selection=subset_selection, labels=labels, title=title, ref_per=ref_per, saveData=saveData, ylim1=ylim1, ylim2=ylim2, normalize=normalize, acg_mn=acg_mn, acg_std=acg_std)
    
    return fig
    
def plot_ccg(dp, units, cbin=0.2, cwin=80, normalize='Hertz', saveDir='~/Downloads', saveFig=False, prnt=False, show=True, 
             _format='pdf', subset_selection='all', labels=True, std_lines=True, title=None, color=-1, CCG=None, saveData=False,
             ylim=[0,0], ccg_mn=None, ccg_std=None, again=False, trains=None, ccg_grid=False):
    assert type(units) in [list, np.ndarray]
    units=list(units)
    _, _idx=np.unique(units, return_index=True)
    units=npa(units)[np.sort(_idx)].tolist()
    assert normalize in ['Counts', 'Hertz', 'Pearson', 'zscore', 'mixte'],"WARNING ccg() 'normalize' argument should be a string in ['Counts', 'Hertz', 'Pearson', 'zscore', 'mixte']."#
    if normalize=='mixte' and len(units)==2 and not ccg_grid: normalize='zscore'
    saveDir=op.expanduser(saveDir)
    bChs=get_depthSort_peakChans(dp, units=units)[:,1].flatten()
    ylim1, ylim2 = ylim[0], ylim[1]

    if CCG is None:
        normalize1 = normalize if normalize!='mixte' else 'Hertz'
        CCG=ccg(dp, units, cbin, cwin, fs=30000, normalize=normalize1, prnt=prnt, subset_selection=subset_selection, again=again, trains=trains)
    assert CCG is not None
    if CCG.shape[0]==2 and not ccg_grid:
        if normalize=='zscore':
            CCG_hertz=ccg(dp, units, cbin, cwin, fs=30000, normalize='Hertz', prnt=prnt, subset_selection=subset_selection, again=again, trains=trains)[0,1,:]
            ccg25, ccg35 = CCG_hertz[:int(len(CCG_hertz)*2./5)], CCG_hertz[int(len(CCG_hertz)*3./5):]
            ccg_std=np.std(np.append(ccg25, ccg35))
            ccg_mn=np.mean(np.append(ccg25, ccg35))
        fig = plt_ccg(units, CCG[0,1,:], cbin, cwin, bChs, 30000, saveDir, saveFig, show, _format, subset_selection=subset_selection, 
                      labels=labels, std_lines=std_lines, title=title, color=color, saveData=saveData, ylim1=ylim1, ylim2=ylim2,
                      normalize=normalize, ccg_mn=ccg_mn, ccg_std=ccg_std)
    else:
        fig = plt_ccg_subplots(units, CCG, cbin, cwin, bChs, None, saveDir, saveFig, prnt, show, _format, subset_selection=subset_selection, 
                               labels=labels, title=title, std_lines=std_lines, ylim1=ylim1, ylim2=ylim2, normalize=normalize)
        
    return fig

#%% Heatmaps including correlation matrices

def imshow_cbar(im, origin='top', events_toplot=[], events_color='k',
                xvalues=None, yvalues=None, xticks=None, yticks=None,
                xticklabels=None, yticklabels=None, xlabel=None, ylabel=None, title='',
                cmapstr="RdBu_r", vmin=-1, vmax=1, center=0, colorseq='nonlinear',
                clabel=None, extend_cmap='neither', cticks=None,
                figsize=(6,4), aspect='auto', function='imshow',
                ax=None, **kwargs):
    '''
    Essentially plt.imshow(im, cmap=cmapstr), but with a nicer and actually customizable colorbar.
    
    Parameters:
        - im: 2D array def to matplotlib.pyplot.imshow
        - origin: y axis origin, either top or bottom | Default: top
        - xvalues, yvalues: lists/arrays of lengths im.shape[1] and im.shape[0], respectively.
                            Allows to alter the value to which pixel positions are mapped (which are dumb pixel ranks by default).
        - xticks, yticks: allows to alter the position of the ticks (in [0,npixels] space by default, in xvalues/yvalues space if they are provided)
        - xticklabels, yticklabels: allows to alter the label of the ticks - should have the same size as xticks/yticks
        - cmapstr: string, colormap name
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
    assert im.ndim==2
    assert isinstance(cmapstr,str), 'cmap must be a string!'
    if cticks is not None: assert cticks[-1]<=vmax and cticks[0]>=vmin
    
    # Make custom colormap.
    # If center if provided, reindex colors accordingly
    cmap = mpl.cm.get_cmap(cmapstr)
    if center is not None:
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
        
    # Define pixel coordinates (default is 0 to n_rows-1 for y and n_columns=1 for x)
    if xvalues is None: xvalues=np.arange(im.shape[1])
    assert len(xvalues)==im.shape[1], f'xvalues should contain {im.shape[1]} values but contains {len(xvalues)}!'
    dx = (xvalues[1]-xvalues[0])/2.
    if yvalues is None: yvalues=np.arange(im.shape[0])
    assert len(yvalues)==im.shape[0], f'yvalues should contain {im.shape[0]} values but contains {len(yvalues)}!'
    dy = (yvalues[1]-yvalues[0])/2.
    extent = [xvalues[0]-dx, xvalues[-1]+dx, yvalues[-1]+dy, yvalues[0]-dy]
    
    # Plot image with custom colormap
    fig,ax=plt.subplots(figsize=figsize) if ax is None else (ax.get_figure(), ax)
    if function=='imshow': axim=ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, 
                                          origin={'top':'upper', 'bottom':'lower'}[origin], extent=extent, **kwargs)
    elif function=='pcolor': axim=ax.pcolormesh(im, X=xvalues, Y=yvalues,
                                                cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if any(events_toplot):
        for e in events_toplot:
            yl=ax.get_ylim()
            ax.plot([e,e],yl,lw=2,ls='--',c=events_color)
            ax.set_ylim(yl)
            
    mplp(fig, ax, figsize=figsize,
          xlim=None, ylim=None, xlabel=xlabel, ylabel=ylabel,
          xticks=xticks, yticks=yticks, xtickslabels=xticklabels, ytickslabels=yticklabels,
          reset_xticks=False, reset_yticks=False, xtickrot=0, ytickrot=0,
          axlab_w='bold', axlab_s=20,
          ticklab_w='regular', ticklab_s=16, ticks_direction='out', lw=1,
          title=title, title_w='bold', title_s=24,
          hide_top_right=False, hide_axis=False)

    # Add colorbar, nicely formatted
    axpos=ax.get_position()
    cbaraxx0,cbaraxy0 = float(axpos.x0+axpos.width+0.01), float(axpos.y0)
    cbar_ax = fig.add_axes([cbaraxx0, cbaraxy0, .02, .3])
    fig.colorbar(axim, cax=cbar_ax, ax=ax,
             orientation='vertical', label=clabel,
             extend=extend_cmap, ticks=cticks, use_gridspec=True)
    if clabel is not None:
        cbar_ax.yaxis.label.set_font_properties(mpl.font_manager.FontProperties(family='arial',weight='bold', size=16))
        cbar_ax.yaxis.label.set_rotation(-90)
        cbar_ax.yaxis.label.set_va('bottom')
        cbar_ax.yaxis.label.set_ha('center')
        cbar_ax.yaxis.labelpad=5
    fig.canvas.draw()
    cbar_ax.yaxis.set_ticklabels([t.get_text() for t in cbar_ax.yaxis.get_ticklabels()], ha='center')
    cbar_ax.yaxis.set_tick_params(pad=11, labelsize=14)
    set_ax_size(ax,*fig.get_size_inches())

    return fig

# Plot correlation matrix of variables x observations 2D arrray
    
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
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    axpos=ax.get_position()
    cbar_ax = fig.add_axes([axpos.x0+axpos.width-0.1, axpos.y0, .02, .3])
    hm = sns.heatmap(cm, vmin=vmin, vmax=vmax, cmap=cmap,
                     cbar_kws={'label': 'Crosscorr. [-0.5-0.5]ms (s.d.)'}, ax=ax, cbar_ax=cbar_ax)
    
    # Main plot params
    hm.axes.plot(hm.axes.get_xlim(), hm.axes.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
    hm.axes.set_yticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=0)
    hm.axes.set_xticklabels(['{}'.format(units[i]) for i in range(len(units))], rotation=45, ha='right')
    if title is None:
        hm.axes.set_title('Dataset: {}'.format(dp.split('/')[-1]))
    else:
        hm.axes.set_title(title)
    hm.axes.set_aspect('equal','box')
    
    # Colorbar params
    cbar_ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(family='arial',weight='bold', size=10))
    cbar_ax.yaxis.label.set_rotation(-90)
    cbar_ax.yaxis.label.set_va('bottom')
    cbar_ax.yaxis.labelpad=5
    cbar_ax.yaxis.set_ticklabels(cbar_ax.yaxis.get_ticklabels(), ha='center')
    cbar_ax.yaxis.set_tick_params(pad=11)
    
    fig = hm.get_figure()
    plt.tight_layout()
    
    if saveFig:
        if saveDir is None: saveDir=dp
        assert title is not None, 'You need to provide a title parameter to save the figure!'
        fig.savefig(Path(saveDir,title,f'.{_format}'))
    
    if ret_cm:
        return cm, units, channels # depth-sorted
    return fig

## Connectivity inferred from correlograms
def plot_sfcm(dp, corr_type='connections', metric='amp_z', cbin=0.5, cwin=100, 
              p_th=0.02, n_consec_bins=3, fract_baseline=4./5, W_sd=10, test='Poisson_Stark',
              drop_seq=['sign', 'time', 'max_amplitude'], units=None, name=None,
              text=False, markers=False, ticks=True, depth_ticks=False,
              regions={}, reg_colors={}, vminmax=[-7,7], figsize=(7,7),
              saveFig=False, saveDir=None, again=False, againCCG=False):
    '''
    Visually represents the connectivity datafrane outputted by 'gen_sfc'.
    Each line/row is a good unit.
    Each intersection is a square split in a varying amount of columns,
    each column representing a positive or negatively significant peak collored accordingly to its size s.
    '''
    sfc, sfcm, peakChs = gen_sfc(dp, corr_type, metric, cbin, cwin,
                                 p_th, n_consec_bins, fract_baseline, W_sd, test, 
                                 again, againCCG, drop_seq, units, name)
    gu = peakChs[:,0]
    ch = peakChs[:,1].astype(int)
    
    if corr_type=='synchrony':
        vminmax=[0,vminmax[1]]
    elif corr_type=='excitations':
        vminmax=[0,vminmax[1]]
    elif corr_type=='inhibitions':
        vminmax=[vminmax[0],0]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    axpos=ax.get_position()
    cbar_ax = fig.add_axes([axpos.x0+axpos.width+0.01, axpos.y0, .02, .3])
    sns.heatmap(sfcm, yticklabels=True, xticklabels=True, cmap="RdBu_r", center=0, vmin=vminmax[0], vmax=vminmax[1],
                     cbar_kws={'label': 'Crosscorr. modulation (s.d.)'}, ax=ax, cbar_ax=cbar_ax)
    cbar_ax.yaxis.label.set_font_properties(matplotlib.font_manager.FontProperties(family='arial',weight='bold', size=12))
    cbar_ax.yaxis.label.set_rotation(90)
    cbar_ax.yaxis.label.set_va('top')
    cbar_ax.yaxis.labelpad=5
    cbar_ax.yaxis.set_ticklabels(cbar_ax.yaxis.get_ticklabels(), ha='center')
    cbar_ax.yaxis.set_tick_params(pad=11)
    set_ax_size(ax,*figsize)
    
    ax.plot(ax.get_xlim(), ax.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
    ttl='Significant functional correlation matrix\n{}\n{}-{}-{}-{}-{}\n({})'.format(op.basename(dp),test, p_th, n_consec_bins, fract_baseline, W_sd, corr_type)
    ax.set_title(ttl, fontsize=16, fontweight='bold')
    
    if depth_ticks:
        labs=['{}'.format(3840-ch[i]*10) for i in range(len(gu)) if i%10==0]
        tks=[i for i in range(len(gu)) if i%10==0]
        ax.set_xticks(tks)
        ax.set_yticks(tks)
        ax.set_yticklabels(labs, fontsize=14, fontweight='bold', rotation=45)
        ax.set_xticklabels(labs, fontsize=14, fontweight='bold', rotation=45)
        ax.set_ylabel('Depth on probe (\u03BCm)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Depth on probe (\u03BCm)', fontsize=16, fontweight='bold')
    else:
        labs=['{}@{}'.format(gu[i], ch[i]) for i in range(len(gu))]
        ax.set_yticklabels(labs, fontsize=12, fontweight='regular')
        ax.set_xticklabels(labs, fontsize=12, fontweight='regular')

    [ax.spines[sp].set_visible(True) for sp in ['left', 'bottom', 'top', 'right']]
    
    if not ticks:
        [tick.set_visible(False) for tick in ax.xaxis.get_major_ticks()]
        [tick.set_visible(False) for tick in ax.yaxis.get_major_ticks()]
    
    if any(regions):
        for region, rng in regions.items():
            rngi=[np.nonzero(abs(r-ch)==min(abs(r-ch)))[0][0] for r in rng[::-1]]
            for r in rngi:
                ax.plot([r,r], [0,len(ch)], ls="-", c=[0.5,0.5,0.5], lw=1)
                ax.plot([0,len(ch)], [r,r], ls="-", c=[0.5,0.5,0.5], lw=1)
            ax.plot(rngi,[len(ch),len(ch)], ls="-", c=reg_colors[region], lw=10, solid_capstyle='butt')
            ax.plot([0,0], rngi, ls="-", c=reg_colors[region], lw=10, solid_capstyle='butt')
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
        fig.savefig(Path(saveDir,ttl.replace('\n', '_')+'.pdf'))
    
    return fig

def plot_dataset_CCGs(dp, cbin=0.1, cwin=10, threshold=2, n_consec_bins=3, subset_selection='all'):
    gu = get_units(dp, quality='good') # get good units
    prct=0; sig=0;
    for i1, u1 in enumerate(gu):
        for i2, u2 in enumerate(gu):
            if prct!=int(100*((i1*len(gu)+i2+1)*1./(len(gu)**2))):
                prct=int(100*((i1*len(gu)+i2+1)*1./(len(gu)**2)))
                #print('{}%...'.format(prct), end=end)
            if i1<i2:
                print('Assessing CCG {}x{}... {}%'.format(u1, u2, prct))
                hist=ccg(dp, [u1,u2], cbin, cwin, fs=30000, normalize='Hertz', prnt=False, subset_selection=subset_selection)[0,1,:]
                pks = get_ccg_sig(hist, cbin, threshold, n_consec_bins, ext_mn=None, ext_std=None, pkSgn='all')
                if np.array(pks).any():
                    sig+=1
                    print("{}th significant CCG...".format(sig))
                    plot_ccg(dp, [u1,u2], cbin, cwin, savedir=dp+'/significantCCGs_{}_{}_{}_{}'.format(cbin, cwin, threshold, n_consec_bins), \
                             saveFig=True, prnt=False, show=False, pdf=False, png=True, subset_selection=subset_selection, CCG=hist)
    return

def plot_dataset_ACGs(dp, cbin=0.5, cwin=80):
    gu = get_units(dp, quality='good') # get good units
    prct=0
    for i1, u1 in enumerate(gu):
        prct=i1*100./len(gu)
        print('{}%...'.format(prct))
        plot_acg(dp, u1, cbin, cwin, savedir=dp+'/allACGs', saveFig=True, prnt=False, show=False, pdf=False, png=True)
    return


#%% Graphs
    
def network_plot_3D(G, angle, save=False):
    '''https://www.idtools.com.au/3d-network-graphs-python-mplot3d-toolkit'''
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)] 

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            
            # Scatter plot
            ax.scatter(xi, yi, zi, c=colors[key], s=20+20*G.degree(key), edgecolors='k', alpha=0.7)
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
    
    # Set the initial view
    ax.view_init(30, angle)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
        plt.savefig(str(angle).zfill(3)+".png")
        plt.close('all')
    else:
         plt.show()
    
    return

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
