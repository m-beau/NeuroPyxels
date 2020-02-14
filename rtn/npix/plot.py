# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
"""
import os
import os.path as op; opj=op.join
import ast

import numpy as np
import pandas as pd
from scipy import signal as sgnl

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
import seaborn as sns

from rtn.utils import phyColorsDic, seabornColorsDic, DistinctColors20, DistinctColors15, mark_dict,\
                    npa, sign, minus_is_1, thresh, smooth, \
                    _as_array, _unique, _index_of, mpl_colors
                    
from rtn.npix.io import read_spikeglx_meta, extract_rawChunk, assert_chan_in_dataset, chan_map
from rtn.npix.gl import get_units
from rtn.npix.spk_wvf import get_depthSort_peakChans, wvf, get_peak_chan, templates
from rtn.npix.spk_t import trn
from rtn.npix.corr import acg, ccg, gen_sfc, extract_hist_modulation_features, make_cm, make_matrix_2xNevents, crosscorr_cyrille
from rtn.npix.behav import align_times, get_processed_ifr
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import networkx as nx

#%% regular histogram, utilities
def get_labels_from_ticks(ticks):
    nflt=0
    for i, t in enumerate(ticks):
        if t != round(t, 0):
            nflt=1
            if t != round(t, 1):
                nflt=2
    ticks_labels=npa(ticks).astype(int) if nflt==0 else np.round(ticks, nflt)
    if nflt!=0: ticks_labels=[str(l)+'0'*(nflt+2-len(str(l))) for l in ticks_labels]
    return ticks_labels, nflt
    
def mplp(fig=None, ax=None, figsize=(8,6),
         xlim=None, ylim=None, xlabel=None, ylabel=None,
         xticks=None, yticks=None, xtickslabels=None, ytickslabels=None,
         axlab_w='bold', axlab_s=20,
         ticklab_w='regular',ticklab_s=16, lw=2,
         title=None, title_w='bold', title_s=24,
         hide_top_right=False, hide_axis=False):
    '''
    make plots pretty
    matplotlib plots
    '''
    if fig is None: fig=plt.gcf()
    if ax is None: ax=plt.gca()
    
    hfont = {'fontname':'Arial'}
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    # Opportunity to easily hide everything
    if hide_axis:
        ax.axis('off')
        return fig, ax
    else:
        ax.axis('on')
    
    # Axis labels
    if ylabel is None:ylabel=ax.get_ylabel()
    if xlabel is None:xlabel=ax.get_xlabel()
    ax.set_ylabel(ylabel, weight=axlab_w, size=axlab_s, **hfont)
    ax.set_xlabel(xlabel, weight=axlab_w, size=axlab_s, **hfont)
    
    # Setup limits BEFORE changing tick labels because tick labels remain unchanged despite limits change!
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    
    # Ticks and ticks labels
    if xticks is None:
        ax.xaxis.set_major_locator(AutoLocator())
        xticks= ax.get_xticks()
        if xlim is None:xticks=xticks[1:-1]
    if yticks is None:
        ax.yaxis.set_major_locator(AutoLocator())
        yticks= ax.get_yticks()
        if ylim is None: yticks=yticks[1:-1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    x_nflt=0
    if xtickslabels is None:
        xtickslabels,x_nflt=get_labels_from_ticks(xticks)
        assert len(xtickslabels)==len(xticks), 'WARNING you provided too many/fey xtickslabels!'
    if ytickslabels is None:
        ytickslabels,y_nflt=get_labels_from_ticks(yticks)
    else:
        assert len(ytickslabels)==len(yticks), 'WARNING you provided too many/fey ytickslabels!'
    
    rot=45 if (x_nflt==2) and (fig.get_figwidth()<=6) else 0
    ax.set_xticklabels(xtickslabels, fontsize=ticklab_s, fontweight=ticklab_w, color=(0,0,0), **hfont, rotation=rot)
    ax.set_yticklabels(ytickslabels, fontsize=ticklab_s, fontweight=ticklab_w, color=(0,0,0), **hfont)

    if title is None: title=ax.get_title()
    ax.set_title(title, size=title_s, weight=title_w)
    
    ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=lw, length=6, direction='in')
    if hide_top_right: [ax.spines[sp].set_visible(False) for sp in ['top', 'right']]
    else: [ax.spines[sp].set_visible(True) for sp in ['top', 'right']]
        
    for sp in ['left', 'bottom', 'top', 'right']:
        ax.spines[sp].set_lw(lw)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, ax

def hist_MB(arr, a, b, s, title='Histogram', xlabel='', ylabel='', ax=None):
    hist=np.histogram(arr, bins=np.arange(a,b,s))
    y=hist[0]
    x=hist[1][:-1]
    if ax is None:
        (fig, ax) = plt.subplots()
    else:
        fig, ax = ax.get_figure(), ax
    ax.bar(x=x, height=y, width=s)
    ax.set_title(title)
    ax.set_xlabel(xlabel) if len(xlabel)>0 else ax.set_xlabel('Binsize:{}'.format(s))
    ax.set_ylabel(ylabel) if len(ylabel)>0 else ax.set_ylabel('Counts')
    
    fig,ax=mplp(fig,ax)
    
    return fig

#%% Waveforms or raw data

def plot_wvf(dp, u, Nchannels=8, chStart=None, n_waveforms=100, t_waveforms=2.8,
               title = '', plot_std=True, plot_mean=True, plot_templates=False, color=phyColorsDic[0],
               labels=True, sample_lines='all', ylim=[0,0], saveDir='~/Downloads', saveFig=False, saveData=False, _format='pdf'):
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
    if type(sample_lines) is str:
        assert sample_lines=='all'
        sample_lines=n_waveforms
    elif type(sample_lines) is float or type(sample_lines) is int:
        sample_lines=min(sample_lines, n_waveforms)
        
    fs=read_spikeglx_meta(dp, subtype='ap')['sRateHz']
    cm=chan_map(dp, y_orig='surface', probe_version='local')
    peak_chan=get_peak_chan(dp, u); peak_chan_i = int(np.nonzero(np.abs(cm[:,0]-peak_chan)==min(np.abs(cm[:,0]-peak_chan)))[0][0]);
    t_waveforms_s=int(t_waveforms*(fs/1000))
    waveforms=wvf(dp, u, n_waveforms, t_waveforms_s, wvf_subset_selection='regular', wvf_batch_size=10)
    tplts=templates(dp, u)
    assert waveforms.shape==(n_waveforms, t_waveforms_s, cm.shape[0])
    
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
    datam = np.rollaxis(data.mean(0),1)
    datastd = np.rollaxis(data.std(0),1)
    tplts=tplts[:, :, chStart_i:chEnd_i]
    tpl_scalings=[(max(datam[pci_rel, :])-min(datam[pci_rel, :]))/(max(tpl[:,pci_rel])-min(tpl[:,pci_rel]))for tpl in tplts]

    color_dark=(max(color[0]-0.08,0), max(color[1]-0.08,0), max(color[2]-0.08,0))
    ylim1, ylim2 = (np.min(datam-datastd)-50, np.max(datam+datastd)+50) if ylim==[0,0] else (ylim[0], ylim[1])
    x = np.linspace(0, data.shape[1]/(fs/1000), data.shape[1]) # Plot t datapoints between 0 and t/30 ms
    x_tplts = x[(data.shape[1]-tplts.shape[1])//2:(data.shape[1]-tplts.shape[1])//2+tplts.shape[1]] # Plot 82 datapoints between 0 and 82/30 ms
    for i in range(data.shape[2]):
        i1, i2 = max(0,data.shape[2]//2-1-i//2), i%2
        ax[i1, i2].set_ylim([ylim1, ylim2])
        for j in range(sample_lines):
            ax[i1, i2].plot(x, data[j,:, i], linewidth=0.3, alpha=0.3, color=color)
        #r, c = int(Nchannels*1./2)-1-(i//2),i%2
        if plot_templates:
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
        fig.savefig(opj(saveDir, title+'.{}'.format(_format)), format=_format)
    if saveData:
        np.save(opj(saveDir, title+'.npy'), waveforms)
    
    return fig

def plot_raw(dp, times, channels=np.arange(384), subtype='ap', offset=450, saveDir='~/Downloads', saveData=0, saveFig=0,
             _format='pdf', color='multi', whiten=True, pyqtgraph=1, show_allyticks=0, events=[], set0atEvent=1, figsize=(20,8), plot_ylabels=True):
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
    - whiten: boolean, whiten data or not
    - pyqtgraph: boolean, whether to use pyqtgraph backend instead of matplotlib (faster to plot and interact, use to explore data before saving nice plots with matplotlib) | default 0
    - show_allyticks: boolean, whetehr to show all y ticks or not (every 50uV for each individual channel), only use if exporing data | default 0
    - events: list of times where to plot vertical lines, in seconds.
    - set0atEvent: boolean, set time=0 as the time of the first event provided in the list events, if any is provided.
    PS: if you wish to center the plot on the event, ensure that the event is exactly between times[0] and times[1].
    ## RETURNS
    fig: a matplotlib figure with channel 0 being plotted at the bottom and channel 384 at the top.
    
    '''
    assert type(events) is list
    # Get data
    channels=assert_chan_in_dataset(dp, channels)
    rc = extract_rawChunk(dp, times, channels, subtype, saveData, 1, whiten)
    meta=read_spikeglx_meta(dp, subtype)
    fs = int(meta['sRateHz'])
    # Offset data
    plt_offsets = np.arange(0, rc.shape[0]*offset, offset)
    plt_offsets = np.tile(plt_offsets[:,np.newaxis], (1, rc.shape[1]))
    rc+=plt_offsets
    
    # Plot data
    y_subticks = np.arange(50, offset/2, 50) if show_allyticks else npa([100])
    y_ticks=[plt_offsets[:,0]] # len(y_ticks)==len(channels) here
    for ys in y_subticks:
        y_ticks=y_ticks+[plt_offsets[:,0]-ys, plt_offsets[:,0]+ys] if show_allyticks else y_ticks+[npa([plt_offsets[0,0]+ys])]
    y_labels_ch=['#{}'.format(channels[i]) for i in range(len(channels))]
    y_labels=[]
    for i in range(len(y_labels_ch)):
        y_labels+=[y_labels_ch[i]]
        if i==0 or show_allyticks:
            for j in range(len(y_ticks)-1):
                if y_ticks[j+1][0]<0:
                    y_labels=y_labels[:-1]+[str(y_ticks[j+1][0])]+[y_labels[-1]]
                else:
                    y_labels=y_labels+[str(y_ticks[j+1][0])]
    y_ticks = np.sort(np.concatenate(y_ticks).flatten())
    assert len(y_ticks)==len(y_labels)

    t=np.tile(np.arange(rc.shape[1])*1000./fs, (rc.shape[0], 1)) # in milliseconds
    if any(events):
        events=[e-times[0] for e in events] # offset to times[0]
        if set0atEvent:
            t=t-events[0]*1000
            events=[e-events[0] for e in events]
    if not pyqtgraph:
        fig, ax = plt.subplots(figsize=figsize)
        if color=='multi':
            color=None
        for i in np.arange(rc.shape[0]):
            y=i*offset
            ax.plot([t[0,0], t[0,-1]], [y, y], color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1)
        ax.plot(t.T, rc.T, linewidth=1, color=color)
        ax.set_xlabel('Time (ms)', size=14, weight='bold')
        ax.set_ylabel('Extracellular potential (\u03bcV)', size=14, weight='bold')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels) if plot_ylabels else ax.set_yticklabels([])
        ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=2, length=6, labelsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_lw(2)
        ax.spines['bottom'].set_lw(2)
        for e in events:
            ax.plot([e,e], ax.get_ylim(), color=(0.3, 0.3, 0.3), linestyle='--', linewidth=1.5)
        if len(channels)==1:
            ax.set_xlim([t[0,0], t[0,-1]])
        fig.tight_layout()
    
        if saveFig:
            saveDir=op.expanduser(saveDir)
            rcn = '{}_t{}-{}_ch{}-{}'.format(op.basename(dp), times[0], times[1], channels[0], channels[-1]) # raw chunk name
            rcn=rcn+'_whitened' if whiten else rcn+'_raw'
            fig.savefig(opj(saveDir, '{}.{}'.format(rcn, _format)), format=_format)
        
        return fig
    
    else:
        win = pg.GraphicsWindow(title="Raw data - {}-{}ms, channels {}-{}".format(times[0], times[1], channels[0], channels[-1]))
        win.setBackground('w')
        win.resize(1500,600)
        p = win.addPlot()
        p.setTitle("Raw data - {}-{}ms, channels {}-{}".format(times[0], times[1], channels[0], channels[-1]), color='k')
        p.disableAutoRange()
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)
    
        for i in np.arange(rc.shape[0]):
            y=i*offset
            pen=pg.mkPen(color=(125,125,125), style=QtCore.Qt.DashLine, width=1.5)
            p.plot([0, t[0,-1]], [y, y], pen=pen)
        for e in events:
            p.plot([e,e], [p.rect().getCoords()[1], p.rect().getCoords()[3]], color=(0.3, 0.3, 0.3), linestyle='--', linewidth=1.5)
        if color=='multi':
            color=[DistinctColors20[ci%(len(DistinctColors20)-1)] for ci in range(rc.shape[0])]
        else:
            if color=='k':
                color=[(0,0,0)]*rc.shape[0]
            else:
                assert npa(color).shape[0]==3
                color=[npa(color)]*rc.shape[0]
                
        for line in range(rc.shape[0]):
            pen=pg.mkPen(color=tuple(npa(color[line])*255), width=1)
            p.plot(t[line,:].T, rc[line,:].T, pen=pen)
        pen=pg.mkPen(color=(0,0,0), width=2)
        p.getAxis('left').setTicks([[(y_ticks[i], y_labels[i]) for i in range(len(y_ticks))],[]])
        p.getAxis('bottom').setLabel('Time (ms)')
        p.getAxis('left').setLabel('Extracellular potential (\u03bcV)')
        p.getAxis('left').setPen(pen)
        p.getAxis('bottom').setPen(pen)
        font=QtGui.QFont()
        font.setPixelSize(14)
        p.getAxis("bottom").setTickFont(font)
        p.getAxis("left").setTickFont(font)
        p.getAxis("bottom").setStyle(tickTextOffset = 5)
        p.getAxis("left").setStyle(tickTextOffset = 5)
        p.autoRange() # adding it only after having plotted everything makes it way faster
        
        
        return win,p

def plot_raw_units(dp, times, units=[], channels=None, offset=450, saveDir='~/Downloads', saveData=0, saveFig=0, whiten=1,
             _format='pdf', colors='phy', Nchan_plot=5, spk_window=82, pyqtgraph=1, show_allyticks=0, events=[], set0atEvent=1, figsize=(20,8)):
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
    # if channels is None:
    #     peakChan=get_peak_chan(dp,units[0])
    #     channels=np.arange(peakChan-Nchan_plot//2-1, peakChan+Nchan_plot//2+2)
    channels=assert_chan_in_dataset(dp, channels)
    rc = extract_rawChunk(dp, times, channels, 'ap', saveData, 1, whiten)
    # Offset data
    plt_offsets = np.arange(0, len(channels)*offset, offset)
    plt_offsets = np.tile(plt_offsets[:,np.newaxis], (1, rc.shape[1]))
    rc+=plt_offsets
    
    back_color='k'
    fig=plot_raw(dp, times, channels, 'ap', offset, saveDir, saveData, 0, color=back_color, whiten=whiten, pyqtgraph=pyqtgraph,
                 show_allyticks=show_allyticks, events=events, set0atEvent=set0atEvent, figsize=figsize)
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
                ax.plot(tx_ms[ch1:ch2, spk_id].T, rc[ch1:ch2, spk_id].T, lw=1.1, color=color)
                #ax.plot(tx_ms[peakChan_rel, spk_id].T, rc[peakChan_rel, spk_id].T, lw=1.5, color=color)
                fig.tight_layout()
            
    if saveFig and not pyqtgraph:
        saveDir=op.expanduser(saveDir)
        rcn = '{}_{}_t{}-{}_ch{}-{}'.format(op.basename(dp), units, times[0], times[1], channels[0], channels[-1]) # raw chunk name
        rcn=rcn+'_whitened' if whiten else rcn+'_raw'
        fig.savefig(opj(saveDir, '{}.{}'.format(rcn, _format)), format=_format)
    
    if pyqtgraph:fig[1].autoRange()
    return fig

#%% Peri-event time plots: rasters, psths...

def ifr_subplots(times_list, events_list, titles_list, figsize=(8,4)):
    assert len(times_list)==len(events_list)==len(titles_list)
    
    fig, ax = plt.subplots(len(times_list), figsize=figsize)
    
    for i, (tm, ev, tt) in enumerate(zip(times_list, events_list, titles_list)):
        ax[i]=ifr_plot(tm, ev).get_axes()[0]
        ax[i].set_title(tt)
    
    return fig

def ifr_plot(dp, unit, events, b=5, window=[-1000,1000], remove_empty_trials=False,
             zscore=False, zscoretype='overall', convolve=True, gw=64, gsd=1,
             title='', figsize=(10,4), color=seabornColorsDic[0],
             plot_all_traces=False, zslines=False, plot_sem=True,
             saveDir='~/Downloads', saveFig=False, saveData=False, _format='pdf'):
    '''
    '''
    
    times=trn(dp, unit)/read_spikeglx_meta(dp, subtype='ap')['sRateHz']
    
    if title == '':
        title='psth_{}'.format(unit)
        
    return ifr_plt(times, events, b, window, remove_empty_trials,
             zscore, zscoretype, convolve, gw, gsd, title, figsize, 
             color, plot_all_traces, zslines, plot_sem, saveDir,
             saveFig, saveData, _format)

def ifr_plt(times, events, b=5, window=[-1000,1000], remove_empty_trials=False,
             zscore=False, zscoretype='overall', convolve=True, gw=64, gsd=1,
             title='', figsize=(10,4), color=seabornColorsDic[0],
             plot_all_traces=False, zslines=False, plot_sem=True,
             saveDir='~/Downloads', saveFig=False, saveData=False, _format='pdf', ax=None):
    '''
    '''
    
    # Get ifr +- zscored +- smoothed (processed)
    x, y, y_mn, y_p, y_p_sem = get_processed_ifr(times, events, b, window, remove_empty_trials,
                      zscore, zscoretype, convolve, gw, gsd)
    # plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig=ax.get_figure()
    ylims=[]
    if zscore:
        if not convolve:
            if not plot_sem:
                ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
            else:
                ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
                ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0))
                ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.1, facecolor=color)
                ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
                ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
        else:
            if plot_all_traces:
                for i in range(y.shape[0]):
                    gaussWin=sgnl.gaussian(gw, gsd)
                    gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
                    trace = np.convolve(y[i,:], gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
                    ax.plot(x, trace, lw=0.3, color=color, alpha=0.2)
            ax.plot(x, y_p, lw=1, color=color)
            if plot_sem:
                ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
                ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
                ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)
                
        ax.plot([x[0], x[-1]], [0,0], ls="--", c=(0,0,0), lw=0.5)
        if zslines:
            ax.plot([x[0], x[-1]], [1,1], ls="--", c=[1,0,0], lw=1)
            ax.plot([x[0], x[-1]], [2,2], ls="--", c=[1,0,0], lw=1)
            ax.plot([x[0], x[-1]], [3,3], ls="--", c=[1,0,0], lw=1)
            ax.plot([x[0], x[-1]], [-1,-1], ls="--", c=[0,0,1], lw=1)
            ax.plot([x[0], x[-1]], [-2,-2], ls="--", c=[0,0,1], lw=1)
            ax.plot([x[0], x[-1]], [-3,-3], ls="--", c=[0,0,1], lw=1)
        ax.set_ylim([-1, 2])
        ax.set_ylabel('Inst.F.R. (s.d.)')
    
    elif not zscore:
        if plot_all_traces:
            for i in range(y.shape[0]):
                    ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
        if not convolve:
            if not plot_sem:
                ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
            else:
                ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
                ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0), zorder=3)
                ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.2, facecolor=color)
                ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
                ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
        else:
            ax.plot(x, y_p, lw=1.5, color=color, alpha=1)
            if plot_sem:
                ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
                ax.plot(x, y_p-y_p_sem, lw=1, color=color)
                ax.plot(x, y_p+y_p_sem, lw=1, color=color)
        yl=max(y_p+y_p_sem); ylims.append(int(yl)+5-(yl%5));
        ax.set_ylabel('Inst.F.R. (Hz)')
    
    ax.plot([0,0], ax.get_ylim(), color=(0.3, 0.3, 0.3), linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time from event (ms).')
    if title == '':
        title='psth'
    ax.set_title(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    ax, fig = mplp(ax,fig)

    # Save data and.or figure
    saveDir=op.expanduser(saveDir)
    if not os.path.isdir(saveDir): os.mkdir(saveDir)
    if saveData:
        np.save(opj(saveDir, title+'_x.npy'), x)
        np.save(opj(saveDir,title+'_y.npy'), y)
        np.save(opj(saveDir,title+'_y_processed.npy'), y_p)
        np.save(opj(saveDir,title+'_y_p_sem.npy'), y_p_sem)
    if saveFig:
        fig.savefig(opj(saveDir, '{}.{}'.format(title, _format)), format=_format)

    return fig

def raster_plot(dp, units, events, events_toplot=None, window=[-1000, 1000], remove_empty_trials=False,
           title='', figsize=(10,5), saveDir='~/Downloads', saveFig=0, saveData=0, _format='pdf'):
    
    fig,ax=plt.subplots()
    if title == '':
        title='raster_{}'.format(units)
    
    for i,u in enumerate(units):
        times=trn(dp, u)/read_spikeglx_meta(dp, subtype='ap')['sRateHz']
        fig=raster_plt(times, events, events_toplot, window, remove_empty_trials,
           title, mpl_colors[i], 10, figsize, saveDir, saveFig, saveData, _format, ax=ax)
    
    return fig

def raster_plt(times, events, events_toplot=None, window=[-1000, 1000], remove_empty_trials=False,
           title='', color='k', size=10, figsize=(10,5),
           saveDir='~/Downloads', saveFig=0, saveData=0, _format='pdf', ax=None):
    '''
    Make a raster plot of the provided 'times' aligned on the provided 'events', from window[0] to window[1].
    By default, there will be len(events) lines. you can pick a subset of events to plot
    by providing their indices as a list.array with 'events_toplot'.
    
    Parameters:
        - times: list/array of time points, in seconds.
        - events: list/array of events, in seconds.
        - events_toplot: list/array of events indices to display on the raster | Default: None (plots everything)
        - window: list/array of shape (2,): the raster will be plotted from events-window[0] to events-window[1] | Default: [-1000,1000]
        - remove_empty_trials: boolean, if True only plots trials with at least one spike
        - title: string, title of tehe plot + if saved file name will be raster_title._format.
        - figsize: tuple, (x,y) figure size
        - saveDir: save directory to save data and figure
        - saevFig: boolean, if 1 saves figure with name raster_title._format at saveDir
        - saveData: boolean, if 1 saves data as 2D array 2xlen(times), with first line being the event index and second line the relative timestamp time in seconds.
        - _format: string, format used to save figure if saveFig=1 | Default: 'pdf'
    
    Returns:
        - fig: matplotlib figure.
    '''
    
    if events_toplot is None:
        events_toplot=npa([0])
    
    at, atb = align_times(times, events, window=window, remove_empty_trials=remove_empty_trials)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()
    
    # Handles indexing of empty trials
    y_ticks=np.arange(len(at))+1
    y_ticks_labels=np.nonzero(np.isin(np.sort(events),np.sort(list(at.keys()))))[0]+1
    
    for e, ts in at.items():
        i=np.argsort(list(at.keys()))[npa(list(at.keys()))==e][0]
        y=[y_ticks[i]]*len(ts)
        ts=npa(ts)*1000 # convert to ms
        ax.scatter(ts, y, s=10, c='k', alpha=1)
    
    if title == '':
        title='raster'
    
    fig,ax=mplp(fig=fig, ax=ax, figsize=(figsize[0], max(figsize[1], len(at)//4)),
         xlim=window, ylim=[y_ticks[-1]+1, 0], xlabel="Time (ms)", ylabel="Trials",
         xticks=None, yticks=y_ticks, xtickslabels=None, ytickslabels=y_ticks_labels,
         axlab_w='bold', axlab_s=20,
         ticklab_w='regular',ticklab_s=16, lw=2,
         title=title, title_w='bold', title_s=24,
         hide_top_right=True, hide_axis=False)
    
    for etp in events_toplot:
        ax.plot([etp,etp], ax.get_ylim(), ls='--', lw=3, c='r', zorder=-1)

    if saveFig:
        fig.savefig(opj(saveDir, '{}.{}'.format(title, _format)), format=_format)
        
    return fig

#%% Correlograms

def plt_ccg(uls, CCG, cbin=0.04, cwin=5, bChs=None, fs=30000, saveDir='~/Downloads', saveFig=True, 
            show=True, pdf=True, png=False, rec_section='all', labels=True, std_lines=True, title=None, color=-1, 
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
        if normalize=='Hertz':
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
        ax2.set_ylabel('Crosscorrelation (Hz)', fontsize=20, rotation=270)
        ax2ticks=[np.round(ccg_mn+tick*ccg_std,1) for tick in ax.get_yticks()]
        ax2.set_yticks(ax.get_yticks())
        ax2.set_yticklabels(ax2ticks, fontsize=20)
        ax2.set_ylim([ylim1, ylim2])
        
    if normalize=='Hertz' or normalize=='Pearson':
        y=CCG.copy()
    elif normalize=='zscore':
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
                title="Units {}->{} ({})s".format(uls[0], uls[1], str(rec_section)[0:50].replace(' ',  ''))
            else:
                title="Units {}@{}->{}@{} ({})s".format(uls[0], bChs[0], uls[1], bChs[1], str(rec_section)[0:50].replace(' ',  ''))
        ax.set_title(title, size=22)
        ax.tick_params(labelsize=20)
    fig.tight_layout()
    plt.close() if not show else plt.show()
    if saveFig or saveData:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        if saveFig:
            if pdf: fig.savefig(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}.pdf'.format(uls[0], uls[1], cwin, cbin))
            if png: fig.savefig(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}.png'.format(uls[0], uls[1], cwin, cbin))
        if saveData:
            np.save(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}_x.npy'.format(uls[0], uls[1], cwin, cbin), x)
            np.save(saveDir+'/ccg{0}-{1}_{2}_{3:.2f}_y.npy'.format(uls[0], uls[1], cwin, cbin), CCG)
        
    return fig
        
def plt_acg(unit, ACG, cbin=0.2, cwin=80, bChs=None, color=0, fs=30000, saveDir='~/Downloads', saveFig=True, 
            show=True, pdf=True, png=False, rec_section='all', labels=True, title=None, ref_per=True, saveData=False, 
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
        if normalize=='Hertz':
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

    if normalize=='Hertz' or normalize=='Pearson':
        y=ACG.copy()
    elif normalize=='zscore':
        y=ACG.copy()+abs(ylim1)
    ax.bar(x=x, height=y, width=cbin, color=color, edgecolor=color, bottom=ylim1) # Potentially: set bottom=0 for zscore
    
    if labels:
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
                title="Unit {} ({})s".format(unit, str(rec_section)[0:50].replace(' ',  ''))
            else:
                assert len(bChs)==1
                title="Unit {}@{} ({})s".format(unit, bChs[0], str(rec_section)[0:50].replace(' ',  ''))
        ax.set_title(title, size=22)
        ax.tick_params(labelsize=20)
        if ref_per:
            ax.plot([-1, -1], [ylim1, ylim2], color='black', linestyle='--', linewidth=1)
            ax.plot([1, 1], [ylim1, ylim2], color='black', linestyle='--', linewidth=1)
    fig.tight_layout()
    plt.close() if not show else plt.show()
    if saveFig or saveData:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        if saveFig:
            if pdf: fig.savefig(saveDir+'/acg{}-{}_{:.2f}.pdf'.format(unit, cwin, cbin))
            if png: fig.savefig(saveDir+'/acg{}-{}_{:.2f}.png'.format(unit, cwin, cbin))
        if saveData:
            np.save(saveDir+'/acg{}-{}_{:.2f}_x.npy'.format(unit, cwin, cbin), x)
            np.save(saveDir+'/acg{}-{}_{:.2f}_y.npy'.format(unit, cwin, cbin), ACG)

    return fig
        
    
def plt_ccg_subplots(units, CCGs, cbin=0.2, cwin=80, bChs=None, Title=None, saveDir='~/Downloads', 
                     saveFig=False, prnt=False, show=True, pdf=True, png=False, rec_section='all', 
                     labels=True, title=None, std_lines=True, ylim1=0, ylim2=0, normalize='zscore'):
    colorsDic = {
    0:(53./255, 127./255, 255./255),
    1:(255./255, 0./255, 0./255),
    2:(255./255,215./255,0./255),
    3:(238./255, 53./255, 255./255),
    4:(84./255, 255./255, 28./255),
    5:(255./255,165./255,0./255),
    -1:(0., 0., 0.),
    }
    #print(cwin*1./CCGs.shape[2])
    #assert cbin==cwin*1./CCGs.shape[2]

    fig, ax = plt.subplots(len(units), len(units), figsize=(16, 10), dpi=80)
    for i in range(len(units)):
        for j in range(len(units)):
            r, c = i, j
            if (i==j):
                color=colorsDic[i]#'#A785BD'#
            else:
                color=colorsDic[-1]
            x=np.linspace(-cwin*1./2, cwin*1./2, CCGs.shape[2])
            y=CCGs[i,j,:]
            ax[r, c].bar(x=x, height=y, width=cbin, color=color, edgecolor=color)
            if ylim1==0 and ylim2==0:
                if normalize=='Hertz':
                    ylim1=0
                    yl=max(y); ylim2=int(yl)+5-(yl%5);
                elif normalize=='Pearson':
                    ylim1=0
                    yl=max(y); ylim2=yl+0.01-(yl%0.01);
                elif normalize=='zscore':
                    yl1=min(y);yl2=max(y)
                    ylim1=yl1-0.5+(abs(yl1)%0.5);ylim2=yl2+0.5-(yl2%0.5)
                    ylim1, ylim2 = min(-3, ylim1), max(3, ylim2)
                    ylim1, ylim2 = -max(abs(ylim1), abs(ylim2)), max(abs(ylim1), abs(ylim2))
            ax[r, c].set_ylim([ylim1, ylim2])
            
            if normalize=='Hertz' or normalize=='Pearson':
                yy=y.copy()
            elif normalize=='zscore':
                yy=y.copy()+abs(ylim1)
            ax[r, c].bar(x=x, height=yy, width=cbin, color=color, edgecolor=color, bottom=ylim1) # Potentially: set bottom=0 for zscore
            
            if labels:
                if j==0:
                    if normalize=='Hertz':
                        ax[r, c].set_ylabel("Crosscorr. (Hz)", size=12)
                    elif normalize=='Pearson':
                        ax[r, c].set_ylabel("Crosscorr. (Pearson)", size=12)
                    elif normalize=='zscore':
                        ax[r, c].set_ylabel("Crosscorr. (z-score)", size=12)
                if i==len(units)-1:
                    ax[r, c].set_xlabel('Time (ms)', size=12)
                ax[r, c].set_xlim([-cwin*1./2, cwin*1./2])
                if type(bChs)!=list:
                    title="{}->{} ({})s".format(units[i], units[j], str(rec_section)[0:50].replace(' ',  ''))
                else:
                    title="{}@{}->{}@{} ({})s".format(units[i], bChs[i], units[j], bChs[j], str(rec_section)[0:50].replace(' ',  ''))
                ax[r, c].set_title(title, size=12)
                ax[r, c].tick_params(labelsize=20)
                if i!=j:
                    mn = np.mean(np.append(y[:int(len(y)*2./5)], y[int(len(y)*3./5):]))
                    std = np.std(np.append(y[:int(len(y)*2./5)], y[int(len(y)*3./5):]))
                    if std_lines:
                        ax[r, c].plot([x[0], x[-1]], [mn,mn], ls="--", c=[0,0,0], lw=1)
                        for st in [1,2,3]:
                            ax[r, c].plot([x[0], x[-1]], [mn+st*std,mn+st*std], ls="--", c=[0.5,0,0], lw=0.5)
                            ax[r, c].plot([x[0], x[-1]], [mn-st*std,mn-st*std], ls="--", c=[0,0,0.5], lw=0.5)
    
    
    if Title:
        fig.suptitle(Title, size=20, weight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.close() if not show else plt.show()
    if saveFig:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        if pdf: fig.savefig(saveDir+'/ccg{}-{}_{2:.2f}.pdf'.format(str(units).replace(' ', ''), cwin, cbin))
        if png: fig.savefig(saveDir+'/ccg{}_{}_{2:.2f}.png'.format(str(units).replace(' ', ''), cwin, cbin))
        
    return fig

def plot_acg(dp, unit, cbin=0.2, cwin=80, normalize='Hertz', color=0, saveDir='~/Downloads', saveFig=True, prnt=False, show=True, 
             pdf=True, png=False, rec_section='all', labels=True, title=None, ref_per=True, saveData=False, ylim=[0,0], acg_mn=None, acg_std=None, again=False):
    assert type(unit)==int or type(unit)==str
    saveDir=op.expanduser(saveDir)
    bChs=get_depthSort_peakChans(dp, units=unit)[:,1].flatten()
    ylim1, ylim2 = ylim[0], ylim[1]

    ACG=acg(dp, unit, cbin, cwin, fs=30000, normalize=normalize, prnt=prnt, rec_section=rec_section, again=again)
    if normalize=='zscore':
        ACG_hertz=acg(dp, unit, cbin, cwin, fs=30000, normalize='Hertz', prnt=prnt, rec_section=rec_section)
        acg25, acg35 = ACG_hertz[:int(len(ACG_hertz)*2./5)], ACG_hertz[int(len(ACG_hertz)*3./5):]
        acg_std=np.std(np.append(acg25, acg35))
        acg_mn=np.mean(np.append(acg25, acg35))
    fig=plt_acg(unit, ACG, cbin, cwin, bChs, color, 30000, saveDir, saveFig, pdf=pdf, png=png, 
            rec_section=rec_section, labels=labels, title=title, ref_per=ref_per, saveData=saveData, ylim1=ylim1, ylim2=ylim2, normalize=normalize, acg_mn=acg_mn, acg_std=acg_std)
    
    return fig
    
def plot_ccg(dp, units, cbin=0.2, cwin=80, normalize='Hertz', saveDir='~/Downloads', saveFig=False, prnt=False, show=True, 
             pdf=True, png=False, rec_section='all', labels=True, std_lines=True, title=None, color=-1, CCG=None, saveData=False, ylim=[0,0], ccg_mn=None, ccg_std=None, again=False):
    assert type(units)==list
    saveDir=op.expanduser(saveDir)
    bChs=get_depthSort_peakChans(dp, units=units)[:,1].flatten()
    ylim1, ylim2 = ylim[0], ylim[1]

    if CCG is None:
        CCG=ccg(dp, units, cbin, cwin, fs=30000, normalize=normalize, prnt=prnt, rec_section=rec_section, again=again)
        assert CCG is not None
        if normalize=='zscore':
            CCG_hertz=ccg(dp, units, cbin, cwin, fs=30000, normalize='Hertz', prnt=prnt, rec_section=rec_section, again=again)[0,1,:]
            ccg25, ccg35 = CCG_hertz[:int(len(CCG_hertz)*2./5)], CCG_hertz[int(len(CCG_hertz)*3./5):]
            ccg_std=np.std(np.append(ccg25, ccg35))
            ccg_mn=np.mean(np.append(ccg25, ccg35))
        if CCG.shape[0]==2:
            fig = plt_ccg(units, CCG[0,1,:], cbin, cwin, bChs, 30000, saveDir, saveFig, show, pdf, png, rec_section=rec_section, 
                          labels=labels, std_lines=std_lines, title=title, color=color, saveData=saveData, ylim1=ylim1, ylim2=ylim2, normalize=normalize, ccg_mn=ccg_mn, ccg_std=ccg_std)
    else:
        fig = plt_ccg_subplots(units, CCG, cbin, cwin, bChs, None, saveDir, saveFig, prnt, show, pdf, png, rec_section=rec_section, 
                               labels=labels, title=title, std_lines=std_lines, ylim1=ylim1, ylim2=ylim2, normalize=normalize)
        
    return fig

# Plot correlation matrix of variables x ovservations 2D arrray
    
def plot_cm(dp, units, b=5, cwin=100, cbin=1, corrEvaluator='CCG', vmax=5, vmin=0, cmap='viridis', rec_section='all', ret_cm=False, saveDir='~/Downloads', saveFig=False, pdf=1, png=0):
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
    cm = make_cm(dp, units, b, cwin, cbin, corrEvaluator, vmax, vmin, rec_section)
    
    # Plot correlation matrix
    cm[np.eye(cm.shape[0], dtype=bool)]=0 # set diag values to 0
    plt.figure()
    hm = sns.heatmap(cm, vmin=vmin, vmax=vmax, cmap=cmap)
    hm.axes.plot(hm.axes.get_xlim(), hm.axes.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
    hm.axes.set_yticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=0)
    hm.axes.set_xticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=45, horizontalalignment='right')
    hm.axes.set_title('Dataset: {}'.format(dp.split('/')[-1]))
    hm.axes.set_aspect('equal','box-forced')
    fig = plt.gcf()
    plt.tight_layout()
    print(units)
    
    if saveFig:
        saveDir=op.expanduser(saveDir)
        if not os.path.isdir(saveDir): os.mkdir(saveDir)
        if pdf: fig.savefig(saveDir+'/ccg%s-%d_%.2f.pdf'%(str(units).replace(' ', ''), cwin, cbin))
        if png: fig.savefig(saveDir+'/ccg%s_%d_%.2f.png'%(str(units).replace(' ', ''), cwin, cbin))
    
    if ret_cm:
        return cm
    else:
        return fig

## Connectivity inferred from correlograms
def plot_sfcdf(dp, cbin=0.2, cwin=100, threshold=3, n_consec_bins=3, text=True, markers=False, rec_section='all', 
               ticks=True, again = False, saveFig=True, saveDir=None, againCCG=False):
    '''
    Visually represents the connectivity datafrane otuputted by 'gen_cdf'.
    Each line/row is a good unit.
    Each intersection is a square split in a varying amount of columns,
    each column representing a positive or negatively significant peak collored accordingly to its size s.
    '''
    df, hmm, gu, bestChs, hmmT = gen_sfc(dp, cbin, cwin, threshold, n_consec_bins, rec_section=rec_section, _format='peaks_infos', again=again, againCCG=againCCG)
    plt.figure()
    hm = sns.heatmap(hmm, yticklabels=True, xticklabels=True, cmap="RdBu_r", center=0, vmin=-5, vmax=5, cbar_kws={'label': 'Crosscorrelogram peak (s.d.)'})
    #hm = sns.heatmap(hmmT, yticklabels=False, xticklabels=False, alpha=0.0)
    
    hm.axes.plot(hm.axes.get_xlim(), hm.axes.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
    hm.axes.set_yticklabels(['{}@{}'.format(gu[i], bestChs[i]) for i in range(len(gu))])
    hm.axes.set_xticklabels(np.array([['{}@{}'.format(gu[i], bestChs[i])]*12 for i in range(len(gu))]).flatten())
    for i in range(hmm.shape[1]):
        if (i-5)%12!=0: hm.axes.xaxis.get_major_ticks()[i].set_visible(False)
        else: hm.axes.xaxis.get_major_ticks()[i].set_visible(True)
    if not ticks:
        [tick.set_visible(False) for tick in hm.axes.xaxis.get_major_ticks()]
        [tick.set_visible(False) for tick in hm.axes.yaxis.get_major_ticks()]

    if markers:
        for i in range(hmm.shape[0]):
            for j in range(hmm.shape[0]):
                if i!=j:
                    pkT = hmmT[i,j*12]
                    if pkT>0:
                        hm.axes.scatter(j*12+5, i, marker='^', s=20, c="black")
                    elif pkT<0:
                        hm.axes.scatter(j*12+5, i, marker='v', s=20, c="black")
                    elif pkT==0:
                        hm.axes.scatter(j*12+5, i, marker='*', s=20, c="black")
    if text:
        for i in range(hmm.shape[0]):
            for j in range(hmm.shape[0]):
                pkT = np.unique(hmmT[i,j*12:j*12+12])
                if i!=j and (min(pkT)<=0 or max(pkT)>0):
                    hm.axes.text(x=j*12+2, y=i, s=str(pkT), size=6)
    fig = hm.get_figure()
    if saveFig:
        if saveDir is None: saveDir=dp
        fig.savefig(opj(saveDir,'heatmap_{}_{}_{}_{}.pdf'.format(cbin, cwin, threshold, n_consec_bins)))
    
    return fig

def plot_dataset_CCGs(dp, cbin=0.1, cwin=10, threshold=2, n_consec_bins=3, rec_section='all'):
    gu = get_units(dp, quality='good') # get good units
    prct=0; sig=0;
    for i1, u1 in enumerate(gu):
        for i2, u2 in enumerate(gu):
            if prct!=int(100*((i1*len(gu)+i2+1)*1./(len(gu)**2))):
                prct=int(100*((i1*len(gu)+i2+1)*1./(len(gu)**2)))
                #print('{}%...'.format(prct), end=end)
            if i1<i2:
                print('Assessing CCG {}x{}... {}%'.format(u1, u2, prct))
                hist=ccg(dp, [u1,u2], cbin, cwin, fs=30000, normalize='Hertz', prnt=False, rec_section=rec_section)[0,1,:]
                pks = extract_hist_modulation_features(hist, cbin, threshold, n_consec_bins, ext_mn=None, ext_std=None, pkSgn='all')
                if np.array(pks).any():
                    sig+=1
                    print("{}th significant CCG...".format(sig))
                    plot_ccg(dp, [u1,u2], cbin, cwin, savedir=dp+'/significantCCGs_{}_{}_{}_{}'.format(cbin, cwin, threshold, n_consec_bins), \
                             saveFig=True, prnt=False, show=False, pdf=False, png=True, rec_section=rec_section, CCG=hist)
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
        plt.savefig("C:\scratch\\data\ "+str(angle).zfill(3)+".png")
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
