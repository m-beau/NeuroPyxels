# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
Behavior analysis tools.
"""

import os

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.signal as sgnl
import scipy.stats as stats
import paq2py
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from rtn.utils import seabornColorsDic, npa, thresh

from rtn.npix.spk_t import trn, trnb, isi
from rtn.npix.corr import ccg
from rtn.npix.plot import hist_MB


def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))

def import_PAQdata(dp, variables='all'):
    '''Used to align: Analog variables (wheel position...), noisy digital variables (piezo lick...).
    If variables is not a list, all PackIO variables will be exported.'''
    
    vtypes = {'RECON':'digital', 'GAMEON':'digital', 'TRIALON':'digital', 
              'REW':'digital', 'CUE':'digital', 'LICKS':'digital', 
              'VRframes':'digital', 'REW_GHOST':'digital', 'ROT':'analog', 
              'ROTreal':'analog', 'CameraFrames':'digital', 'LICKS_Piezo':'digital'}
    paq_dp = dp+'/behavior'
    paq_f = next(list_files(paq_dp, 'paq'))
    paq = paq2py.paq_read(paq_dp+'/'+paq_f)
    allVariables = np.array(paq['chan_names'])
    if type(variables)==list:
        variables=np.array(variables)
        areIn = np.isin(variables, allVariables)
        if not np.all(areIn):
            print('WARNING: {} is not in the list of accepted variables {}. Exitting now.'.format(variables[~areIn], allVariables))
            return
    else:
        variables = allVariables
    variables = {variables[i]:vtypes[variables[i]] for i in range(len(variables))}
    
    rawPAQVariables = {}
    print('>> PackIO acquired channels: {}, of which {} will be extracted...'.format(allVariables, list(variables.keys())))
    for v in variables.keys():
        (i, ) = np.nonzero(v==np.array(allVariables))[0]
        print('Extracting PackIO channel {}.'.format(v))
        data = paq['data'][i]
        rawPAQVariables[v] = data
        if variables[v]=='digital':
            th = (max(data)-min(data))*1./2
            rawPAQVariables[v+'_ON'] = thresh(data, th, 1)*1/5000 # PAQIO: 5kHz acquisition
            rawPAQVariables[v+'_OFF'] = thresh(data, th, -1)*1/5000 # PAQIO: 5kHz acquisition
    del paq
    return rawPAQVariables

def mk_PAQtriggersDic(dp):
    '''Used to split up 4 kinds of trial types and bond them their respective reward onset times, 
    in PackIO time reference frame:
    Random Real RR, Cued Real CR, Random Fictive RF and Correct Omission CO reward trials.'''
    
    var_defining_trial_types = ['TRIALON','REW','CUE','REW_GHOST']
    rawPAQVariables = import_PAQdata(dp, var_defining_trial_types)
    
    # Make triggers dictionnary sorted by trial type
    # IMPORTANT: if cued trial, trial_on_on is the cue onset, reward is +500ms. else, trial_on_on is reward time.
    # 1st R: Random, 1st C: Cued, 2nd R: Real, 2nd F: Fictive (ghost), 2nd O: Omitted
    PAQtriggersDic = {'RR_trials':[],'RR':[],
    'CR_trials':[], 'CR':[],
    'RF_trials':[], 'RF':[],
    'CO_trials':[], 'CO':[]}
    for i in range(len(rawPAQVariables['TRIALON_ON'][:-1])):
        ton=rawPAQVariables['TRIALON_ON'][i]
        toff=rawPAQVariables['TRIALON_OFF'][i]
        cue=np.any((rawPAQVariables['CUE_ON']>=ton)&(rawPAQVariables['CUE_ON']<=toff))
        rew=np.any((rawPAQVariables['REW_ON']>=ton)&(rawPAQVariables['REW_ON']<=toff))
        ghost=np.any((rawPAQVariables['REW_GHOST_ON']>=ton)&(rawPAQVariables['REW_GHOST_ON']<=toff))
        if cue:
            PAQtriggersDic['CR_trials'].append(i) if rew else PAQtriggersDic['CO_trials'].append(i)
        else:
            PAQtriggersDic['RR_trials'].append(i) if rew else PAQtriggersDic['RF_trials'].append(i) if ghost else 0
    
    for trg in ['RR', 'CR', 'RF', 'CO']:
        trg_t = np.array(PAQtriggersDic[trg+'_trials'], dtype=np.int64)
        PAQtriggersDic[trg+'_trials']=trg_t
        trg_times = rawPAQVariables['TRIALON_ON'][trg_t].flatten()
        if trg[0]=='C':
            trg_times=trg_times+0.5 # on cued trials, reward (or reward omission) onset is 500ms after onset
        PAQtriggersDic[trg]=trg_times
    
    return PAQtriggersDic

#%% 2) Neuropixels sync channel binary variables onsets and offsets, extracted with Nick's and my scripts in matlab 
### (/home/ms047/Documents/cortex-git/spikes/preprocessing/extractSyncChannel.m
###  and /home/ms047/Dropbox/Science/PhD/Data_Analysis/MatLab_scripts/exportSyncdataToNumpy.m)

def import_GLXdata(dp, variables='all'):
    '''Used to align: clean digital variables recorded on FPGA, units..
    If variables is not a list, user defined Spike GLX variables will be imported.'''
    
    # Import variables from matlab-exported np arrays
    npy_dp = dp+'/exported_syncdat_npy'
    if not os.path.isdir(npy_dp):
        print('WARNING triggers have not been exported in {}. Exitting now.'.format(npy_dp))
        return
    allVariables=['piezo_lick', 'real_reward', 'buzz_cue', 'ghost_reward', 'trial_on']
    if type(variables)==list:
            variables=np.array(variables)
            areIn = np.isin(variables, allVariables)
            if not np.all(areIn):
                print('WARNING: {} is not in the list of accepted variables {}. Exiting now.'.format(variables[~areIn], allVariables))
                return
    else:
        variables = allVariables
        
    rawGLXVariables = {}
    for v in variables:
        fn=npy_dp+'/'+str(v)
        if not os.path.isfile(fn+'_on.npy'):
            print('WARNING triggers have not been exported to {} via MATLAB. Exitting now.'.format(npy_dp))
            #return
        rawGLXVariables[v+'_on']=np.load(fn+'_on.npy').flatten()
        rawGLXVariables[v+'_off']=np.load(fn+'_off.npy').flatten()
    
    return rawGLXVariables

def mk_GLXtriggersDic(dp):
    '''Used to split up 4 kinds of trial types and bond them their respective reward onset times, 
    in SpikeGLX time reference frame:
    Random Real RR, Cued Real CR, Random Fictive RF and Correct Omission CO reward trials.'''

    rawGLXVariables = import_GLXdata(dp, variables='all')
    # Make triggers dictionnary sorted by trial type
    # IMPORTANT: if cued trial, trial_on_on is the cue onset, reward is +500ms. else, trial_on_on is reward time.
    # 1st R: Random, 1st C: Cued, 2nd R: Real, 2nd F: Fictive (ghost), 2nd O: Omitted
    GLXtriggersDic = {'RR_trials':[],'RR':[],
    'CR_trials':[], 'CR':[],
    'RF_trials':[], 'RF':[],
    'CO_trials':[], 'CO':[]}
    for i in range(len(rawGLXVariables['trial_on_on'][:-1])):
        ton=rawGLXVariables['trial_on_on'][i]
        toff=rawGLXVariables['trial_on_off'][i]
        cue=np.any((rawGLXVariables['buzz_cue_on']>=ton)&(rawGLXVariables['buzz_cue_on']<=toff))
        rew=np.any((rawGLXVariables['real_reward_on']>=ton)&(rawGLXVariables['real_reward_on']<=toff))
        ghost=np.any((rawGLXVariables['ghost_reward_on']>=ton)&(rawGLXVariables['ghost_reward_on']<=toff))
        if cue:
            GLXtriggersDic['CR_trials'].append(i) if rew else GLXtriggersDic['CO_trials'].append(i)
        else:
            GLXtriggersDic['RR_trials'].append(i) if rew else GLXtriggersDic['RF_trials'].append(i) if ghost else 0
    
    for trg in ['RR', 'CR', 'RF', 'CO']:
        GLXtriggersDic[trg+'_trials']=np.array(GLXtriggersDic[trg+'_trials'])
        trg_times = rawGLXVariables['trial_on_on'][GLXtriggersDic[trg+'_trials']]
        if trg[0]=='C':
            trg_times=trg_times+0.5 # on cued trials, reward (or reward omission) onset is 500ms after onset
        GLXtriggersDic[trg]=trg_times
    return GLXtriggersDic

#%% 3) Virmen files processed by Dim's script in matlab
### (/home/ms047/Dropbox/Science/PhD/Data_Analysis/5_EncodingDetermination/Rot2p analysis/Rot2p_main.m)
importVIRMEN=False
if __name__=='__main__' and importVIRMEN:
    fd='/media/ms047/DK_probes_backup/DK105_day2'
    df='/behavior/aligned_data.mat'
    aligned_data = sp.io.loadmat(fd+df)['aligned_data']
    trials, summaries = aligned_data['trials'][0,0], aligned_data['summaries'][0,0]
    del aligned_data

#%% Plot behavior quality characteristics

def extract_licks(dp, source='PAQ'):
    if source=='PAQ':
        lick_var = 'LICKS_Piezo'
        lick_var_on, lick_var_off = lick_var+'_ON', lick_var+'_OFF'
        licksDic = import_PAQdata(dp, variables=[lick_var])
    elif source=='GLX':
        lick_var = 'piezo_lick'
        lick_var_on, lick_var_off = lick_var+'_on', lick_var+'_off'
        licksDic = import_GLXdata(dp, variables=[lick_var])
    else:
        print('WARNING source must be either PAQ or GLX. Exitting now.')
        return
    on, off = licksDic[lick_var_on], licksDic[lick_var_off] # seconds
    min_interlickinterval = 70e-3 # seconds
    ON = on[(len(on)-1-np.nonzero(np.diff(abs(on-on[-1])[::-1])>min_interlickinterval)[0])[::-1]]
    OFF = off[np.nonzero(np.diff(off)>min_interlickinterval)[0]]
    ON, OFF = np.append(on[0], ON), np.append(OFF, off[-1])
    licksDic[lick_var_on], licksDic[lick_var_off] = ON, OFF
    return licksDic

def licks_hist(dp, source='PAQ'):
    '''Function plotting the distribution of the interlick intervals.
       Source should be either 'PAQ' or 'GLX'.'''
    licksDic = extract_licks(dp, source) # in seconds
    lick_var_on='LICKS_Piezo_ON' if source=='PAQ' else 'piezo_lick_on'
    fig = hist_MB(np.diff(licksDic[lick_var_on])*1000, 0, 250, 2, title='Interlicks intervals distribution.', xlabel='InterLicks intervals (ms)')
    return fig

def extract_wheel(dp):
    fig, ax = plt.subplots()
    wheelDic = import_PAQdata(dp, variables=['TRIALON', 'ROT', 'ROTreal'])
    #TODO Clip +/- 4 seconds around trial onsets and offsets
    #TODO Define wheel onset for correct trials only, offsets = trials offsets
    return wheelDic

#%% Format spike trains: clip (-4s, +4s) around alignement event

# Either directly from the matlab file, alrady done
def get_npy_export(unit, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports'):
    assert type(unit)==int
    assert alignement_event in ['movon' ,'off']
    assert start_format in ['ifr', 'bst', 'meanifr']
    arr=np.load('{}/{}_{}_{}_cor.npy'.format(dp, unit, start_format, alignement_event))
    return arr

# Or clip it here
def align_unit(dp, u, triggers, b=2, window=[-1000,1000], rem_emptyTrials=False):
    ''' b: binarized train bin in millisecond
        window is in milliseconds
        triggers in seconds
        u is int or list of ints -> population rate!'''
    if type(u)==int:
        t = trn(dp, u) # in samples (fs: 30000Hz)
    elif type(u)==list:
        t = npa(empty=(0))
        for unt in u:
            t = np.append(t, trn(dp, unt))
        t = np.sort(t)
    aligned_t = []
    aligned_tb = np.zeros((len(triggers), int((window[1]-window[0])*1./b)))
    for i, trg in enumerate(triggers):
        ts = t-(trg*30000) # ts: t shifted
        tsc = ts[(ts>=window[0]*30)&(ts<=window[1]*30)]*1./30 # tsc: ts clipped + conv in ms
        if np.any(tsc) or not rem_emptyTrials:
            aligned_t.append(tsc.tolist())
            tscb = np.histogram(tsc, bins=np.arange(window[0],window[1]+b,b))[0] # tscb: tsc binned
            aligned_tb[i,:] = tscb
        else:
            assert aligned_tb.dtype==float
            aligned_tb[i,:] = np.nan
    aligned_tb=aligned_tb[~np.isnan(aligned_tb).any(axis=1)]
    return aligned_t, aligned_tb

def align_licks(dp, triggers, b=2, window=[-1000,1000], source='GLX'):
    ''' b: binarized train bin in millisecond
        window is in milliseconds
        triggers in seconds
        Source should be either 'PAQ' or 'GLX'.'''
    licksDic = extract_licks(dp, source) # in seconds
    lick_var = 'LICKS_Piezo_ON' if source=='PAQ' else 'piezo_lick_on'
    t = licksDic[lick_var]*1000 # in milliseconds
    aligned_t = []
    aligned_tb = np.zeros((len(triggers), int((window[1]-window[0])*1./b)))
    for i, trg in enumerate(triggers*1000): # in milliseconds
        ts = t-(trg) # ts: t shifted
        tsc = ts[(ts>=window[0])&(ts<=window[1])] # tsc: ts clipped + conv in ms
        aligned_t.append(tsc.tolist())
        tscb = np.histogram(tsc, bins=np.arange(window[0],window[1]+b,b))[0] # tscb: tsc binned
        aligned_tb[i,:] = tscb
    
    return aligned_t, aligned_tb

#%% Plot signle units (and licks) raster plots or PSTH


def raster(dp, u, triggersnames, title='', window=[-1000,1000], show=True, licks_source = 'GLX'):
    
    # Sanity check triggers
    if type(u)!=list:
        if u =='licks' and licks_source=='PAQ':
            triggersDic = mk_PAQtriggersDic(dp)
        elif type(u)==int or type(u)==float or (u =='licks' and licks_source=='GLX'):
            triggersDic = mk_GLXtriggersDic(dp)
        else:
            print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
            return
    else:
        for unt in u:
            if type(unt)!=int:
                print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
                return
        triggersDic = mk_GLXtriggersDic(dp)
    
    trgnDic = {'RR':'random real reward onset', 'CR':'cued real reward onset', 
               'RF':'random fictive reward onset', 'CO':'cued omitted reward onset'}
    if type(triggersnames)!=list: triggersnames = list(triggersnames)
    try:
        for trgn in triggersnames:
            assert trgn in trgnDic.keys()
    except:
        print('WARNING the triggersname should be one of: {}. Exit now.'.format(trgnDic.keys()))
        return
    # plot
    fig, axes = plt.subplots(len(triggersnames), figsize=(8,2.5*len(triggersnames)))
    for ti, trg in enumerate(triggersnames):
        ax=axes[ti] if len(triggersnames)>1 else axes
        
        triggers = triggersDic[trg]
        at, atb = align_unit(dp, u, triggers, window=window) if (type(u)==int or type(u)==list) else align_licks(dp, triggers, window=window, source=licks_source)
        print('Number of licks/spikes:', len([item for sublist in at for item in sublist]))
        for i, trial in enumerate(at):
            ax.scatter(trial, i+1+np.zeros((len(trial))), color='black', s=2)
        ax.plot([0,0], ax.get_ylim(), ls='--', lw=1, color='black')
        if trg[0]=='C':
            ax.plot([-500, -500], ax.get_ylim(), ls='--', lw=1, color='black')
        ax.set_ylim([0, len(at)])
        ax.invert_yaxis()
        ax.set_ylabel('Trial')
        ax.set_xlabel('Time from {} (ms)'.format(trgnDic[trg]))
        ax.set_xlim(window[0], window[1])
    fig.suptitle(title) if len(title)!=0 else fig.suptitle('Unit {}.'.format(u))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if not show:
        plt.close(fig)
    return fig
#%%

def get_ifr(dp, u, triggers, b=5, window=[-1000,1000], licks_source='GLX'):
    '''
    dp: string, path to dataset with kilosort/phy output.
    u: integer, unit index or list of ints -> population rate!
    triggers: list, time stamps to trigger alignement, in seconds.
    b: float, bin to make the aligned binned train, in milliseconds.
    window: [w1, w2], where w1 and w2 are in milliseconds.
    convolve: boolean, set to True to convolve the aligned binned train with a gaussian window to smooth the ifr
    gw: integer, gaussian window width, only used if convolve is True
    gsd: float, gaussian window standard deviation, only used if convolve is True
    '''    
    # Get aligned binned train
    at, atb = align_unit(dp, u, triggers, b, window=window) if (type(u)==int or type(u)==list) else align_licks(dp, triggers, b, window=window, source=licks_source) if type(u)==str else print("WARNING u must be an int or 'licks'.")
    # Make ifr
    ifr = np.zeros((atb.shape[0], atb.shape[1]))
    for i in range(atb.shape[0]):
        ifr[i,:] = atb[i,:]/(b*1e-3)
    ifr_mn = np.array([np.mean(ifr, axis=1), ]*ifr.shape[1]).transpose()
    ifr_sd = np.array([np.std(ifr, axis=1), ]*ifr.shape[1]).transpose()
    
    # Set 0 sd to 1 so that dividing does not change anything
    for i in range(ifr_sd.shape[0]):
        if np.all(ifr_sd[i,:]==0): ifr_sd[i,:]=1
    
    return ifr, ifr_mn, ifr_sd

def get_processed_ifr(dp, u, triggers, b=5, window=[-1000,1000], zscore=False, zscoretype='overall', convolve=False, gw=64, gsd=1, licks_source='GLX'):
    '''u can be a list of units -> population rate!'''
    ifr, ifr_mn, ifr_sd = get_ifr(dp, u, triggers, b, window, licks_source)
    
    # Window and bins translation
    maxWin=4000; minWin=-4000;
    window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
    x = np.arange(window[0], window[1], b)
    y = ifr[:, int(ifr.shape[1]/2)+int(window[0]/b):int(ifr.shape[1]/2)+int(window[1]/b)+1]
    if x.shape[0]>y.shape[1]:
        x=x[:-1]
    assert x.shape[0]==y.shape[1]
        
    if zscore:
        assert zscoretype in ['overall', 'trialwise']
        if zscoretype=='overall':
            y_mn=np.mean(ifr.flatten())
            y_sd=np.std(ifr.flatten())
            print('overall mean:{}, sd:{}'.format(y_mn, y_sd))
        if zscoretype=='trialwise':
            y_mn = ifr_mn[:, int(ifr.shape[1]/2)+int(window[0]/b)-1:int(ifr.shape[1]/2)+int(window[1]/b)+1]
            y_sd = ifr_sd[:, int(ifr.shape[1]/2)+int(window[0]/b)-1:int(ifr.shape[1]/2)+int(window[1]/b)+1]
            print('trialwise mean:{}, sd:{}'.format(y_mn[:,0], y_sd[:,0]))
        y_p = (y-y_mn)/y_sd
        y_p=np.mean(y_p, axis=0)
        y_p_sem=stats.sem(y, axis=0)
    else:
        y_p = y_mn = np.mean(y, axis=0)
        y_p_sem = stats.sem(y, axis=0)
        
    if convolve:
        gaussWin=sgnl.gaussian(gw, gsd)
        gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
        y_p = np.convolve(y_p, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]

    return x, y, y_mn, y_p, y_p_sem

def ifr_plot(dp, u, triggersnames, title='', b=5, window=[-1000,1000], color=seabornColorsDic[0], 
             zscore=False, plot_all_traces=False, zslines=False, zscoretype='overall', 
             convolve=True, error=True, show=True, ylim=None, licks_source = 'GLX', gw=64, gsd=1, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
    '''Window has to be in milliseconds. b as well.
    
    if u is a list of units, the population rate of this list will be computed.'''
    
    # Sanity check triggers
    if type(u)!=list:
        if u =='licks' and licks_source=='PAQ':
            triggersDic = mk_PAQtriggersDic(dp)
        elif type(u)==int or type(u)==float or (u =='licks' and licks_source=='GLX'):
            triggersDic = mk_GLXtriggersDic(dp)
        else:
            print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
            return
    else:
        for unt in u:
            if type(unt)!=int:
                print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
                return
        triggersDic = mk_GLXtriggersDic(dp)
    
    trgnDic = {'RR':'random real reward onset', 'CR':'cued real reward onset', 
               'RF':'random fictive reward onset', 'CO':'cued omitted reward onset'}
    if type(triggersnames)!=list: triggersnames = list(triggersnames)
    try:
        for trgn in triggersnames:
            assert trgn in trgnDic.keys()
    except:
        print('WARNING the triggersname should be one of: {}. Exit now.'.format(trgnDic.keys()))
        return
    
    # plot
    
    if saveFig or saveData:
        unit_n = str(u)+'_'+dp.split('/')[-1]
        fig_n = 'IFRsingleUnit{}_'.format(unit_n)
        Dir = saveDir+'/'+fig_n+str(triggersnames)
        if not os.path.isdir(Dir): os.mkdir(Dir)
    fig, axes = plt.subplots(len(triggersnames), figsize=(8,2.5*len(triggersnames)))
    ylims=[]
    for ti, trg in enumerate(triggersnames):
        ax=axes[ti] if len(triggersnames)>1 else axes
        
        triggers = triggersDic[trg]
        x, y, y_mn, y_p, y_p_sem = get_processed_ifr(dp, u, triggers, b, window, zscore, zscoretype, convolve, gw, gsd, licks_source)
        if saveData:
            np.save(Dir+'/'+fig_n+'{}aligned_x.npy'.format(trg), x)
            np.save(Dir+'/'+fig_n+'{}aligned_y.npy'.format(trg), y)
            np.save(Dir+'/'+fig_n+'{}aligned_y_processed.npy'.format(trg), y_p)
            np.save(Dir+'/'+fig_n+'{}aligned_y_p_sem.npy'.format(trg), y_p_sem)
        if zscore:
            if not convolve:
                if not error:
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
                if error:
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
                if not error:
                    ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
                else:
                    ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
                    ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0), zorder=3)
                    ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.2, facecolor=color)
                    ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
                    ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
            else:
                ax.plot(x, y_p, lw=1, color=color, alpha=1)
                if error:
                    ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
                    ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
                    ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)
            yl=max(y_p+y_p_sem); ylims.append(int(yl)+5-(yl%5));
            ax.set_ylabel('Inst.F.R. (Hz)')
            
        ax.set_xlabel('Time from {} (ms).'.format(trgnDic[trg]))
        

    AXES=axes if len(triggersnames)>1 else [axes]
    for ax, trg in zip(AXES, triggersnames):
        if not zscore:
            ylim = max(ylims) if not ylim else ylim
        ax.set_ylim([0, ylim])
        ax.set_xlim(window[0], window[1])
        ax.plot([0,0], ax.get_ylim(), ls='--', lw=1, color='black')
        if trg[0]=='C':
            ax.plot([-500, -500], ax.get_ylim(), ls='--', lw=1, color='black')
    fig.suptitle(title) if len(title)!=0 else fig.suptitle('Unit {}.'.format(u))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if saveFig:
        fig.savefig(Dir+'/'+fig_n+'{}aligned.pdf'.format(triggersnames))
        fig.savefig(Dir+'/'+fig_n+'{}aligned.png'.format(triggersnames))

    if not show:
        plt.close(fig)
    
    return fig


def plot_CS_dataset(CS_MB021,CS_MB022,CS_MB023):
    dps=['/media/ms047/DK_probes_backup/Conditioning/MB021/Kilosort50iters',
         '/media/ms047/DK_probes_backup/Conditioning/MB022',
         '/media/ms047/DK_probes_backup/Conditioning/MB023']
    CSs=[CS_MB021,
         CS_MB022,
         CS_MB023]
    trg_ls=[['RR','RF', 'CR', 'CO']]
    for dp, CS in zip(dps, CSs):
        if not os.path.isdir(dp+'/AlignedComplexSpikes'): os.mkdir(dp+'/AlignedComplexSpikes')
        for cs in CS:
            for trg_l in trg_ls:
                fig = ifr_plot(dp, cs, ['RR'],  b=10, window=[-750,750], color=seabornColorsDic[0], convolve=True, error=True, show=False, ylim=None)
                fig.savefig(dp+'/AlignedComplexSpikes/{}_{}_aligned.pdf'.format(cs, str(trg_l).replace(', ', ' x ').replace("'", '')))
                plt.close()

def plot_CS_selected(selected_units):
    DP = '/home/ms047/Dropbox/Science/PhD/Data_Presentation/Reward paper/MainFig/CS_patterns'
    if not os.path.isdir(DP): os.mkdir(DP)
    dps={'MB021':'/media/ms047/DK_probes_backup/Conditioning/MB021/Kilosort50iters',
         'MB022':'/media/ms047/DK_probes_backup/Conditioning/MB022',
         'MB023':'/media/ms047/DK_probes_backup/Conditioning/MB023'}
    trg_l=['RR','RF', 'CR', 'CO']
    for ap, ap_val in selected_units.items():
        ylim = 10 if ap=='RR_minus' else None
        if not os.path.isdir(DP+'/'+ap): os.mkdir(DP+'/'+ap)
        for dataset, dataset_units in ap_val.items():
            if not os.path.isdir(DP+'/'+ap+'/'+dataset): os.mkdir(DP+'/'+ap+'/'+dataset)
            for cs in dataset_units:
                fig = ifr_plot(dps[dataset], cs, trg_l,  b=10, window=[-750,750], color=seabornColorsDic[0], convolve=True, error=True, show=False, ylim=ylim)
                fig.savefig(DP+'/'+ap+'/'+dataset+'/{}_{}_aligned.pdf'.format(cs, str(trg_l).replace(', ', ' x ').replace("'", '')))
                plt.close()

#%% IFR population plots
                
def make_ifr_matrix(dp, units, triggersname, b=5, window=[-1000,1000],
                       zscore=True, zscoretype='overall', convolve=True, gw=64, gsd=1):
    '''triggersname: one of the keys of GLXtriggersDic.'''
    assert zscoretype in ['overall', 'trialwise']
    if type(units)==int:
        units=[units]
    # Adjust window, get triggers
    maxWin=4000; minWin=-4000;
    window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
    triggersDic = mk_GLXtriggersDic(dp)
    triggers = triggersDic[triggersname]
    
    # Populate matrix
    x, y, y_mn, y_p, y_p_sem = get_processed_ifr(dp, units[0], triggers, b, window, zscore, zscoretype, convolve, gw, gsd)
    ifr_matrix=np.zeros((len(units), len(x)))
    for i, u in enumerate(units):
        x, y, y_mn, y_p, y_p_sem = get_processed_ifr(dp, u, triggers, b, window, zscore, zscoretype, convolve, gw, gsd)
        ifr_matrix[i, :] = y_p
        
    return ifr_matrix, x



def av_ifr_plot_acrossDP(DPs, unitsPerDataset, triggersname, title='', b=5, window=[-1000,1000], color=seabornColorsDic[0],
             zscore=True, zscoretype='overall', plot_all_units=False, zslines=False, 
             convolve=True, error=True, show=True, ylim=None, gw=64, gsd=1, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
    
    for initDataset in DPs.keys():
        dp, units = DPs[initDataset], unitsPerDataset[initDataset]
        if len(units)>0: break
    if len(units)==0:
        return plt.figure() # empty figure if no unit at all across all datasets
    ifr_matrix, x = make_ifr_matrix(dp, units, triggersname, b, window,
                           zscore, zscoretype=zscoretype, convolve=False)
    ifrs_matrix=np.zeros((1, len(x)))
    totalUnits=0
    for dataset in unitsPerDataset.keys():
        dp = DPs[dataset]
        units = unitsPerDataset[dataset]
        if len(units)>0:
            totalUnits+=len(units)
            # DO NOT AVERAGE PRE-CONVOLVED TRACES!!
            ifr_matrix, x1 = make_ifr_matrix(dp, units, triggersname, b, window,
                               zscore, zscoretype=zscoretype, convolve=False)
            ifrs_matrix=np.append(ifrs_matrix, ifr_matrix, axis=0) # vstack
    
    y = ifrs_matrix
    y_p, y_p_sem = np.mean(y, axis=0), stats.sem(y, axis=0) # Zscored or not, convolved or not from within make_ifr_matrix -> get_processed_ifr
    
    # plot
    fig, ax = plt.subplots(1, figsize=(8,2.5))
    ylims=[]
        
    if zscore:
        if not convolve:
            if not error:
                ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
            else:
                ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
                ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0))
                ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.1, facecolor=color)
                ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
                ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
        else:
            # CONVOLUTION HAS TO BE DONE OUTSIDE OF get_processed_ifr 
            # BECAUSE IT HAS TO BE DONE AFTER AVERAGING ACROSS DATASETS
            gaussWin=sgnl.gaussian(gw, gsd)
            print(gsd)
            gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
            y_p = np.convolve(y_p, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
            if plot_all_units:
                for i, yi in enumerate(y):
                    y[i,:] = np.convolve(yi, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
                for i in range(y.shape[0]):
                    ax.plot(x, y[i,:], lw=0.5, color=color, alpha=0.8)
            ax.plot(x, y_p, lw=1, color=color)
            if error:
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
        if plot_all_units:
            for i in range(y.shape[0]):
                    ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
        if not convolve:
            if not error:
                ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
            else:
                ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
                ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0), zorder=3)
                ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.2, facecolor=color)
                ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
                ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
        else:
            gaussWin=sgnl.gaussian(gw, gsd)
            print(gsd)
            gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
            y_p = np.convolve(y_p, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
            ax.plot(x, y_p, lw=1, color=color, alpha=1)
            if error:
                ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
                ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
                ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)
        yl=max(y_p+y_p_sem); ylims.append(int(yl)+5-(yl%5));
        ax.set_ylabel('Inst.F.R. (Hz)')
    
    ax.set_xlabel('Time from {} (ms).'.format(triggersname))
    ax.set_title('{} (n={})'.format(title, totalUnits))
    if not zscore:
        ylim = max(ylims) if not ylim else ylim
        ax.set_ylim([0, ylim])
    ax.set_xlim(window[0], window[1])
    ax.plot([0,0], ax.get_ylim(), ls='--', lw=1, color='black')
    if triggersname[0]=='C':
        ax.plot([-500, -500], ax.get_ylim(), ls='--', lw=1, color='black')
    fig.tight_layout()
    
    if saveFig or saveData:
        fig_n = 'IFRpop_'
        Dir = saveDir+'/'+fig_n+str(triggersname)+'aligned'
        if not os.path.isdir(Dir): os.mkdir(Dir)
        if saveData:
            np.save(Dir+'/'+fig_n+'{}aligned_x.npy'.format(triggersname), x)
            np.save(Dir+'/'+fig_n+'{}aligned_y.npy'.format(triggersname), y)
            np.save(Dir+'/'+fig_n+'{}aligned_y_processed.npy'.format(triggersname), y_p)
            np.save(Dir+'/'+fig_n+'{}aligned_y_p_sem.npy'.format(triggersname), y_p_sem)
        if saveFig:
            fig.savefig(Dir+'/'+fig_n+'{}aligned.pdf'.format(triggersname))
    if not show:
        plt.close(fig)
        
    return fig


def ifr_barplot_compWind_acrossDP(DPs, unitsPerDataset, triggersnames, winds, title='', b=5, window=[-750,750],
         zscore=False, zscoretype='overall', show=False, ylim=15, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
    '''winds format should be [[w1, w2], [w3,w4]... in ms]
    Test: sum ranked wilcoxon from scipy.stats'''
    for wind in winds:
        assert wind[0]<=wind[1]
    # Get concatenated of av IFR across datasets
    totalUnits=[]
    for dataset in unitsPerDataset.keys():
        units = [dataset+'_'+str(u) for u in unitsPerDataset[dataset]]
        totalUnits+=units
    DF = pd.DataFrame(columns=["Unit", "Triggers"]+[str(i) for i in winds], index = np.arange(len(totalUnits*len(triggersnames))))
    DF["Unit"]=totalUnits*len(triggersnames)
    for trg_i, triggersname in enumerate(triggersnames):
        for initDataset in DPs.keys():
            dp, units = DPs[initDataset], unitsPerDataset[initDataset]
            if len(units)>0: break
#        if len(units)==0:
#            return plt.figure() # empty figure if no unit at all across all datasets
        ifr_matrix, x = make_ifr_matrix(dp, units, triggersname, b, window,
                               zscore, zscoretype='overall', convolve=False, gw=64, gsd=1)
        ifrs_matrix=np.zeros((0, len(x)))
    
        for dataset in unitsPerDataset.keys():
            dp = DPs[dataset]
            units = unitsPerDataset[dataset]
            if len(units)>0:
                ifr_matrix, x1 = make_ifr_matrix(dp, units, triggersname, b, window,
                                   zscore, zscoretype='overall', convolve=False, gw=64, gsd=1)
                ifrs_matrix=np.append(ifrs_matrix, ifr_matrix, axis=0) # vstack
        y = ifrs_matrix
        i1=len(totalUnits)
        i2 = len(totalUnits)*trg_i # jump to the next 'indices slice' of size totalUnits i.e. next trigger, all units again
        DF.iloc[0+i2:i1+i2, DF.columns.get_loc("Triggers")]=triggersname
        for wind in winds:
            w1, w2 = int((wind[0]-window[0])/b), int((wind[1]-window[0])/b)
            av_wind = np.mean(y[:, w1:w2], axis=1)
            DF.iloc[0+i2:i1+i2, DF.columns.get_loc(str(wind))]=av_wind
    # Reshape the pandas dataframe, convenient to then make the barplot with seaborn
    DF = pd.melt(DF, id_vars=["Triggers", "Unit"], var_name="Window", value_name="Average IFR")
    
    # Make paired t-test
    pt_table = pd.DataFrame(columns=["Trigger 1", "Window 1", "Trigger 2", "Window 2", "Statistic", "Pval"])
    i=0
    for trg1 in triggersnames:
        for win1 in winds:
            for trg2 in triggersnames:
                for win2 in winds:
                    if (trg1!=trg2 or win1!=win2):
                        # Assert that the units match to allow you to do a paired test!!
                        units1 = DF.loc[(DF["Triggers"]==trg1) & (DF["Window"]==str(win1))]["Unit"]
                        units2 = DF.loc[(DF["Triggers"]==trg2) & (DF["Window"]==str(win2))]["Unit"]
                        assert np.all(units1.values == units2.values)
                        dist1 = DF.loc[(DF["Triggers"]==trg1) & (DF["Window"]==str(win1))]["Average IFR"]
                        dist2 = DF.loc[(DF["Triggers"]==trg2) & (DF["Window"]==str(win2))]["Average IFR"]
                        statistic, Pval = stats.wilcoxon(dist1, dist2) # non parametric paired test
                        pt_table.loc[i, :]=[trg1, str(win1), trg2, str(win2), statistic, Pval]
                        i+=1
                        
    # Plot barplot with seaborn
    fig, ax = plt.subplots()
    sns.barplot(x="Triggers", y="Average IFR", hue="Window", data=DF, order = triggersnames, hue_order=[str(i) for i in winds])#yerrs)
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    sns.stripplot(x="Triggers", y="Average IFR", hue="Window", data=DF, order = triggersnames, hue_order=[str(i) for i in winds],
                    size=6, jitter=False, dodge=True, color=(0.4, 0.4, 0.4), alpha=0.6, marker="D")
    ax.legend(leg_handles, leg_labels, title='Window')
    ax.set_title('Comp.of av. IFR \n windows {} for {} units \n of pattern {}.'.format(winds, len(totalUnits), title))
    ax.set_ylim([0, ylim])
    if saveFig or saveData:
        fig_n = 'barplotMeanIFR_{}aligned_{}windows'.format(triggersnames, winds)
        Dir = saveDir+'/'+fig_n
        if not os.path.isdir(Dir): os.mkdir(Dir)
        if saveData:
            DF.to_csv(Dir+'/'+fig_n+'_values.csv')
            pt_table.to_csv(Dir+'/'+fig_n+'_stats.csv')
        if saveFig:
            fig.savefig(Dir+'/'+fig_n+'.pdf')
    if not show:
        plt.close(fig)
    
    return fig, pt_table

### Modify here - parameters
DP = '/home/ms047/Dropbox/Science/PhD/Data_Presentation/Reward paper/MainFig/CS_pop'
DPs={'MB021':'/media/ms047/DK_probes_backup/Conditioning/MB021/Kilosort50iters',
 'MB022':'/media/ms047/DK_probes_backup/Conditioning/MB022',
 'MB023':'/media/ms047/DK_probes_backup/Conditioning/MB023'}
selected_units = {# Random Fictive error can be 1) similar 2) same time, higher amplitude 3) same amplitude, delayed, 4) sharper + delayed, but rarely two bumps...
            'RRxRF2bumps_plus':{'MB021':[232,  286, 233, 229, 222, 221, 220, 219, 285], 
                          'MB022':[280, 277,  205, 216,  149], # Can tell the difference (RR response never like RF response)
                          'MB023':[525, 243, 268, 206,  231, 195, 192, 91]},
            'RRxRF2bumps_minus':{'MB021':[], 
                        'MB022':[169, 351, 209], 
                        'MB023':[225, 232,181]},
            'RRxRFbiggerbump_plus':{'MB021':[230, 232, 233,  229, 221, 223], 
                          'MB022':[280, 277, 205, 153, 149], # Can tell the difference (RR response never like RF response)
                          'MB023':[243, 358, 204]},
            'RRxRFdelayedbump_plus':{'MB021':[232, 225, 230,  232, 233, 234, 226, 229, 222, 221, 223, 219], 
                          'MB022':[216, 153, 149, 190], # Can tell the difference (RR response never like RF response)
                          'MB023':[ 207]},
            'RRxRFall_plus':{'MB021':[232, 225, 230, 286, 232, 233, 234, 226, 229, 222, 221, 223, 220, 219, 285], 
                          'MB022':[280, 277, 216, 153, 149, 190], # Can tell the difference (RR response never like RF response)
                          'MB023':[243, 268, 358, 206, 204, 231, 207, 195, 192, 91, 525]},
            'RRxRFall_minus':{'MB021':[], 
                        'MB022':[169, 351, 209], 
                        'MB023':[225, 232,181]},
            # Cued Real shift
            'RRxCR_plus':{'MB021':[290, 295, 287, 289, 232, 234, 284, 231, 286, 233, 226, 229, 222, 221, 223, 220, 225, 230, 219, 321], 
                          'MB022':[872, 1078, 1063, 319, 874, 763, 783, 349, 280, 277, 266,  186, 205, 156, 216], 
                          'MB023':[249, 211, 206, 83, 293, 379,  268, 249, 206, 204, 199, 207, 209, 195, 168, 525, 550]},
            'RRxCR_plusNew':{'MB021':[219, 220, 221, 222, 223, 225, 226, 229, 230, 231, 232, 233, 234, 
                                      284, 286, 287, 289, 290, 295, 321], 
                          'MB022':[ 156,  186,  205,  216,  266,  277,  280,  319,  349,  763,  783, 864, 872,  874, 1063, 1078], 
                          'MB023':[ 83, 91, 160, 166, 168, 179, 195, 199, 204, 206, 206, 207, 209, 211, 249, 284, 379, 525, 550, 552]},
            'RRxRR_minusNew':{'MB021':[285], 
                           'MB022':[144],
                           'MB023':[93, 551, 286]},#246, 286]},
            # Cued Omission error
            'CRxCOearly_plusNew':{'MB021':[225], 
                          'MB022':[186], 
                          'MB023':[337, 83, 550, 179, 91]}, #293or411, 179, 91]},
            'CRxCOearly_minus':{'MB021':[], 
                           'MB022':[209, 169], 
                           'MB023':[95]},
            'CRxCOlate_plus':{'MB021':[], 
                          'MB022':[190, 205], 
                          'MB023':[195]},
            'CRxCOall_plus':{'MB021':[225], 
                          'MB022':[ 190, 186, 205], 
                          'MB023':[337, 83, 293, 195, 179, 91]},
            'CRxCOall_minus':{'MB021':[], 
                           'MB022':[209, 169], 
                           'MB023':[95]},
            'RRxRFlicking':{'MB021':['licks'], 
                           'MB022':['licks'], 
                           'MB023':['licks']}}
patterns_compWindows = {# Random Fictive error can be 1) similar 2) same time, higher amplitude 3) same amplitude, delayed, 4) sharper + delayed, but rarely two bumps...
            'RRxRF2bumps_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
            'RRxRF2bumps_minus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
            'RRxRFbiggerbump_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
            'RRxRFdelayedbump_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
            'RRxRFall_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
            'RRxRFall_minus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
            # Cued Real shift
            'RRxCR_plus':[[-750, -500], [0,100]],
            'RRxRR_minus':[[-750, -500], [0, 100]],
            # Cued Omission error
            'CRxCOearly_plus':[[-750, -500], [-500,-450], [0, 200]],
            'CRxCOearly_minus':[[-750, -500], [-500,-450], [0, 200]],
            'CRxCOlate_plus':[[-750, -500], [-500,-450], [200, 250]],
            'CRxCOall_plus':[[-750, -500], [-500,-450], [0, 200], [200, 250]],
            'CRxCOall_minus':[[-750, -500], [-500,-450], [100, 200]],
            'RRxRFlicking':[[-750, -500], [100,600]]}
    

def plot_all_avIFRpatterns(DP, DPs, selected_units):
    '''
    selectedUnits has to be of the form:
        {'ABxCD...':{'datasetName1':[u1,u2,...uN], 'dataset2Name':[u1,u2,...uN]...},{}...}
        
        and the dictionnary DPs = {'dataset1Name':'dataset1Path', ...}'''
    # Loop through patterns
    for pattern, unitsPerDataset in selected_units.items():
        trgs=[pattern[0:2], pattern[3:5]]
        print('\nTriggers for pattern {}: {}'.format(pattern, trgs))
        avPlotPath=DP+'/'+pattern
        if not os.path.isdir(avPlotPath): os.mkdir(avPlotPath)
        ttl='Av. IFR across units displaying pattern {}'.format(pattern)
        for trg in trgs:
            fig1 = av_ifr_plot_acrossDP(DPs, unitsPerDataset, trg, title=ttl, b=10, 
                                        window=[-750,750], color=seabornColorsDic[0],
                                        zscore=False, zscoretype='overall', plot_all_units=False, 
                                        zslines=False, convolve=True, error=True, show=False, ylim=None)
            fig1.savefig(avPlotPath+'/{}_aligned:{}(avgAcrossUnits).pdf'.format(str(trg), pattern))
        winds = patterns_compWindows[pattern]
        fig2, statsTable = ifr_barplot_compWind_acrossDP(DPs, unitsPerDataset, trgs, winds, title=ttl, 
                                             b=10, window=[-750,750], color=seabornColorsDic[0],
                                             zscore=False, zscoretype='overall', convolve=False, show=False)
        fig2.savefig(avPlotPath+'/{}vs{}@{}:{}.pdf'.format(trgs[0], trgs[1], winds, pattern))
        statsTable.to_csv(avPlotPath+'/{}vs{}@{}:{}.csv'.format(trgs[0], trgs[1], winds, pattern))

    
def ifr_heatmap(dp, units, selected_units, title='', b=5, window=[-1000,1000],
                zscoretype='overall', convolve=True, error=True, show=True, ylim=None, PCAsort=1, PCAwindow=[-1000,1000]):
    sns.set_style('white')
    
    fig, axes = plt.subplots(len(triggersnames))
    for pattern, datasets in selected_units.items():
        zscore=True
        ifr_matrix, x = make_av_ifr_matrix(dp, units, trg, b, window, zscore, zscoretype, convolve)
        ifr_matrixPCA, xPCA = make_av_ifr_matrix(dp, units, trg, b, PCAwindow, zscore, zscoretype, convolve)
        # Sort units per first principal component coefficient
        pca = PCA(n_components=5)
        Xproj = pca.fit_transform(ifr_matrixPCA)
        coefPC1 = Xproj[:,0]
        if PCAsort==1:
            PC1sorted = np.argsort(coefPC1)
        elif PCAsort==-1:
            PC1sorted = np.argsort(coefPC1)[::-1]
        #mean = pca.mean_
        #comps = pca.components_
        #exp_var = pca.explained_variance_ratio_
        ifr_matrix = ifr_matrix[PC1sorted,:]
        units = np.array(units)[PC1sorted]
        
        #cmap = sns.palplot(sns.diverging_palette(12, 255, l=40, n=100, center="dark"))
        
        hm = sns.heatmap(ifr_matrix, vmin=-2, vmax=2, cmap="RdBu_r", center=0, cbar_kws={'label': 'Instantaneous Firing Rate (s.d.)'})
        
        if window[0]<0:
            zero=int(len(x)*(-window[0])/(window[1]-window[0]))
            hm.axes.plot([zero,zero], hm.axes.get_ylim()[::-1], ls="--", c=[0,0,0], lw=1)
            if alignement_event=='off':
                rewt=zero+int(len(x)*400/(window[1]-window[0]))
                hm.axes.plot([rewt,rewt], hm.axes.get_ylim()[::-1], ls="--", c=[30/255,144/255,255/255], lw=1)
        hm.axes.set_yticklabels(['{}'.format(u) for u in units], rotation='horizontal')
    #    x_hmm = np.zeros((1,8064))
    #    hm.axes.set_xticklabels([str[i] for i in x], rotation='vertical')
    #    for i in range(hmm.shape[1]):
    #        if (i-8064/(window[1]-window[0]))%12!=0: hm.axes.xaxis.get_major_ticks()[i].set_visible(False)
    #        else: hm.axes.xaxis.get_major_ticks()[i].set_visible(True)
        
    fig = plt.gcf()
    spt = 'Putative Purkinje cells' if region == 'cortex' else 'Cerebellar Nuclear Cells'
    fig.suptitle(spt)
    ax = plt.gca()
    
    return fig, ax
#%% Plot correlation matrix of units list
from elephant.spike_train_generation import SpikeTrain
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import covariance, corrcoef
from quantities import s, ms

usedUnits = {'MB021':[], 'MB022':[], 'MB023':[]}
for ptrn, DSs in selected_units.items():
    for DS, unts in DSs.items():
        for u in unts:
            if type(u)==int: usedUnits[DS].append(u)
for k, v in usedUnits.items():
    usedUnits[k]=np.unique(v)

# 62 units total: 13 sure, 49 unsure.
paperUnits = {'MB021':npa([219, 220, 221, 222, 223, 225, 226, 229, 230, 231, 232, 233, 234,
                           284, 285, 286, 287, 289, 290, 295, 321]), 
              'MB022':npa([ 144,  156,  186,  205,  216,  266,  277,  280,  319,  349,  763,
                           783,  864, 872,  874, 1063, 1078]), 
              'MB023':npa([ 83,  91,  93, 160, 166, 168, 179, 195, 199, 204, 206, 207, 209, 211, 249,
                           284, 286, 337, 379, 525, 550, 551, 552])}
sureUnits= {'MB021':npa([321, 290, 220, 221]), 
             'MB022':npa([142, 812, 763, 783, 764, 480]), 
             'MB023':npa([268, 258, 337])}

def plot_cm(dp, units, b=5, cwin=100, cbin=1, corrEvaluator='CCG', vmax=0):
    '''Plot correlation matrix.
    dp: datapath
    units: units list of the same dataset
    b: bin, in milliseconds'''
    try:
        assert corrEvaluator in ['CCG', 'corrcoeff']
    except:
        print('WARNING: {} should be in {}. Exiting now.'.format(corrEvaluator, ['CCG', 'corrcoeff']))
        return
    # Sort units by depth
    if os.path.isfile(dp+'/FeaturesTable/FeaturesTable_good.csv'):
        ft = pd.read_csv(dp+'/FeaturesTable/FeaturesTable_good.csv', sep=',', index_col=0)
        bestChs=np.array(ft["WVF-MainChannel"])
        depthIdx = np.argsort(bestChs)[::-1] # From surface (high ch) to DCN (low ch)
        table_units=np.array(ft.index, dtype=np.int64)[depthIdx]
        table_channels = bestChs[depthIdx]
    else:
        print('You need to export the features tables using phy first!!')
        return
    #TODO make all CS clusters 'good'
    units = table_units[np.isin(table_units, units)]
    channels = table_channels[np.isin(table_units, units)]
    # Get correlation matrix
    cmCCG=np.empty((len(units), len(units)))
    trnLs = []
    for i1, u1 in enumerate(units):
        tb1=trnb(dp, u1, 1) # 1 in ms
        t1 = SpikeTrain(trn(dp, u1)*1./30*ms, t_stop=len(tb1)*1)
        trnLs.append(t1)
        for i2, u2 in enumerate(units):
            if u1!=u2:
                CCG = ccg(dp, [u1, u2], cbin, cwin)[0,1,:]
                coeffCCG = CCG[len(CCG)//2+1]
                #coeffCCG/=np.sqrt((1000./np.mean(isi(dp1, u1)))*(1000./np.mean(isi(dp1, u2))))
            else:
                coeffCCG=0
            cmCCG[i1, i2]=coeffCCG
    
    if corrEvaluator == 'CCG':
        cm = cmCCG
        vmax = 10 if vmax == 0 else vmax
    elif corrEvaluator == 'corrcoeff':
        cm = covariance(BinnedSpikeTrain(trnLs, binsize=b*ms))
        vmax = 0.05 if vmax == 0 else vmax
    
    # Plot correlation matrix
    plt.figure()
    hm = sns.heatmap(cm, vmin=0, vmax=vmax, cmap='viridis')
    hm.axes.plot(hm.axes.get_xlim(), hm.axes.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
    hm.axes.set_yticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=0)
    hm.axes.set_xticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=45, horizontalalignment='right')
    hm.axes.set_title('Dataset: {}'.format(dp.split('/')[-1]))
    hm.axes.set_aspect('equal','box-forced')
    fig = plt.gcf()
    plt.tight_layout()
    print(units)
    return fig

def get_summaryMFR(dp, units):
    allMFR = []
    for u in units:
        print('UNIT {}'.format(u))
        isint_s = isi(dp, u)*1./1000
        mfr = 1./np.mean(isint_s)
        allMFR.append(mfr)
        if mfr>=3:
            print('WARNING {} mfr >=3!'.format(u))
    return np.array(allMFR)

def plot_summary_MFR(DPs, paperUnits, ylim=5, jitter=True, show=True, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
    allMFRs = np.array([])
    for DS, units in paperUnits.items():
        mfrs = get_summaryMFR(DPs[DS], units)
        allMFRs = np.append(allMFRs, mfrs)
    fig, ax = plt.subplots(figsize=(5,2))
    sns.stripplot(data=allMFRs, size=6, jitter=jitter, dodge=True, color=seabornColorsDic[3], alpha=0.6, marker="D", orient='h', zorder=1)
    sns.pointplot(data=allMFRs, dodge=.532, join=False, color=(139/256,0,0), markers="D", scale=1, orient='h', ci='sd')
    ax.set_title('Mean firing rate of all putative Complex Spikes')
    ax.set_xlim([0, ylim])
    ax.set_xlabel('Mean Firing Rate (Hz)')
    ax.set_yticklabels([])
    print('All MFR mean: {} +/- {} Hz'.format(np.mean(allMFRs), np.std(allMFRs)))
    if saveFig or saveData:
        fig_n = 'summaryIFR'
        Dir = saveDir+'/'+fig_n
        if not os.path.isdir(Dir): os.mkdir(Dir)
        if saveData:
            np.save(Dir+'.npy', allMFRs)
        if saveFig:
            fig.savefig(Dir+'.pdf')
    if not show:
        plt.close(fig)
    return fig

#%% Align spike train on given events from matlab generated alignements

def get_ifr_trace_old(unit, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports'):
    arr = get_npy_export(unit, alignement_event, start_format, dp)
    if start_format=='bst': # else, already ifr
        bin_=0.001 # in seconds
        gaussWin=sgnl.gaussian(64, 4)
        gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
        ifr = np.zeros((arr.shape[0], 8064))
        for i in range(arr.shape[0]):
            ifr[i,:] = np.convolve(arr[i,:]/bin_, gaussWin)
            
    elif start_format=='ifr':
        ifr=arr.copy()
    ifr_mn = np.array([np.mean(ifr, axis=1), ]*ifr.shape[1]).transpose()
    ifr_sd = np.array([np.std(ifr, axis=1), ]*ifr.shape[1]).transpose()
    # Set 0 sd to 1 so that dividing does not change anything
    for i in range(ifr_sd.shape[0]):
        if np.all(ifr_sd[i,:]==0): ifr_sd[i,:]=1
        
    assert ifr.shape == ifr_mn.shape == ifr_sd.shape
    return ifr, ifr_mn, ifr_sd

def make_av_ifr_matrix_old(units, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports', window=[-4000, 4000], zscoretype='overall'):
    
    assert zscoretype in ['overall', 'trialwise']
    # Window and bins translation
    maxWin=4000; minWin=-4000;
    window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
    bin_=1 # 1 ms bins
    convSamples=8064
    bin_ifr = bin_*(maxWin-minWin+bin_)*1./convSamples # to compensate for the convolution resampling
    
    ifr, ifr_mn, ifr_sd = get_ifr_trace_old(units[0], alignement_event, start_format, dp)
    
    y = ifr[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
    
    x = np.arange(window[0], window[1]+bin_ifr, bin_ifr)
    if x.shape[0]>y.shape[1]:
        x=x[:-1]
    assert x.shape[0]==y.shape[1]
        
    
    ifr_matrix=np.zeros((len(units), len(x)))
    
    for i, u in enumerate(units):
        print('for unit {}'.format(u))
        ifr, ifr_mn, ifr_sd = get_ifr_trace(u, alignement_event, start_format, dp)
        y = ifr[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
        
        if zscoretype=='overall':
            ifr_fl = ifr.flatten()
            y_mn = np.mean(ifr_fl)
            y_sd = np.std(ifr_fl)
            print('overall mean:{}, sd:{}'.format(y_mn, y_sd))
        elif zscoretype=='trialwise':
            y_mn = ifr_mn[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
            y_sd = ifr_sd[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
            print('trialwise mean:{}, sd:{}'.format(y_mn[:,0], y_sd[:,0]))
       
        y = (y-y_mn)/y_sd
        y_zs=np.mean(y, axis=0)
        
        ifr_matrix[i, :] = y_zs
        
    return ifr_matrix, x

def ifr_trace_old(units, alignement_event, window=[-4000, 4000], start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports', 
              colors=[(0,0,0), (1,0,0), (0,1,0), (0,0,1)], zscore=False, plot_all_traces=False, zslines=False, offset=True, title=None, zscoretype='overall'):
    '''Window has to be in milliseconds.'''
    if type(units)!=list: units=[units]
    if offset:
        fig, axes = plt.subplots(len(units), 1, figsize=(10,3*len(units)))
    else:
        fig, axes = plt.subplots(1, 1)
    
    for ui, unit in enumerate(units):
        ax=axes[ui] if offset else axes
        color=colors[ui]
        print(unit, type(unit))
        if start_format in ['bst','ifr']:
            ifr, ifr_mn, ifr_sd = get_ifr_trace_old(unit, alignement_event, start_format, dp)
        else:
            return
        # Window and bins translation
        maxWin=4000; minWin=-4000;
        window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
        bin_=1 # 1 ms bins
        convSamples=8064
        bin_ifr = bin_*(maxWin-minWin+bin_)*1./convSamples # to compensate for the convolution resampling
        x = np.arange(window[0], window[1]+bin_ifr, bin_ifr)
        y = ifr[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
        if x.shape[0]>y.shape[1]:
            x=x[:-1]
        assert x.shape[0]==y.shape[1]
        ax.set_title(str(unit)) if offset else ax.set_title(str(units))
            
        
        if zscore:
            assert zscoretype in ['overall', 'trialwise']
            if zscoretype=='overall':
                y_mn=np.mean(ifr.flatten())
                y_sd=np.std(ifr.flatten())
                print('overall mean:{}, sd:{}'.format(y_mn, y_sd))
            if zscoretype=='trialwise':
                y_mn = ifr_mn[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
                y_sd = ifr_sd[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
                print('trialwise mean:{}, sd:{}'.format(y_mn[:,0], y_sd[:,0]))
            y = (y-y_mn)/y_sd
            y_zs=np.mean(y, axis=0)
            y_zs_sem=stats.sem(y, axis=0)
            if plot_all_traces:
                for i in range(ifr.shape[0]):
                        ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
            ax.plot(x, y_zs, lw=1, color=color)
            ax.fill_between(x, y_zs-y_zs_sem, y_zs+y_zs_sem, facecolor=color, interpolate=True, alpha=0.2)
            ax.plot(x, y_zs-y_zs_sem, lw=0.5, color=color)
            ax.plot(x, y_zs+y_zs_sem, lw=0.5, color=color)
            ax.plot([x[0], x[-1]], [0,0], ls="--", c=(0,0,0), lw=0.5)
            if zslines:
                ax.plot([x[0], x[-1]], [1,1], ls="--", c=[1,0,0], lw=1)
                ax.plot([x[0], x[-1]], [2,2], ls="--", c=[1,0,0], lw=1)
                ax.plot([x[0], x[-1]], [3,3], ls="--", c=[1,0,0], lw=1)
                ax.plot([x[0], x[-1]], [-1,-1], ls="--", c=[0,0,1], lw=1)
                ax.plot([x[0], x[-1]], [-2,-2], ls="--", c=[0,0,1], lw=1)
                ax.plot([x[0], x[-1]], [-3,-3], ls="--", c=[0,0,1], lw=1)
            ax.plot([0,0], [-3, 3], ls="--", c=[0,0,0], lw=0.5)
            if alignement_event=='off':
                ax.plot([400,400], [-3, 3], ls="--", c=[30/255,144/255,255/255], lw=0.5)
            ax.set_ylim([-1.5, 1.5])
            ax.set_ylabel('Inst.F.R (s.d.)')
            ax.set_xlim(window[0], window[1])
            ax.set_xlabel('Time (ms)')
        
        elif not zscore:
            y_mn = np.mean(y, axis=0)
            y_sem = stats.sem(y, axis=0)
            if plot_all_traces:
                for i in range(ifr.shape[0]):
                        ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
            ax.plot(x, y_mn, lw=1, color=color, alpha=1)
            ax.fill_between(x, y_mn-y_sem, y_mn+y_sem, facecolor=color, interpolate=True, alpha=0.2)
            ax.plot(x, y_mn-y_sem, lw=0.5, color=color)
            ax.plot(x, y_mn+y_sem, lw=0.5, color=color)
            ax.plot([0,0], ax.get_ylim(), ls="--", c=[0,0,0], lw=0.5)
            if alignement_event=='off':
                ax.plot([400,400], ax.get_ylim(), ls="--", c=[30/255,144/255,255/255], lw=0.5)
            ax.set_xlim(window[0], window[1])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Inst.F.R (Hz)')
    
    fig.tight_layout()
    return fig, axes


def ifr_heatmap_old(region, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports', window=[-4000,4000], sort_dir=1):
    assert region in ['cortex', 'nuclei']
    sns.set_style('white')
    if region=='cortex':
        units=[29, 263, 292, 363, 611, 710, 32039, 33145, 34469, 50838, 50839, 75046]
    elif region=='nuclei':
        units=[1348, 3725, 7620, 15097, 15110, 15112, 16421, 18206, 18944, 20041, 20610, 26610, 50842, 63368, 79513]
    ifr_matrix, x = make_av_ifr_matrix(units, alignement_event, start_format, dp, window)
    if alignement_event=='off':
        ifr_matrixPCA, xPCA = make_av_ifr_matrix(units, alignement_event, start_format, dp, [400, 600])
    if alignement_event=='movon':
        ifr_matrixPCA, xPCA = make_av_ifr_matrix(units, alignement_event, start_format, dp, [0, 1000])
    # Sort units per first principal component coefficient
    pca = PCA(n_components=5)
    Xproj = pca.fit_transform(ifr_matrixPCA)
    coefPC1 = Xproj[:,0]
    if sort_dir==1:
        PC1sorted = np.argsort(coefPC1)
    elif sort_dir==-1:
        PC1sorted = np.argsort(coefPC1)[::-1]
    #mean = pca.mean_
    #comps = pca.components_
    #exp_var = pca.explained_variance_ratio_
    ifr_matrix = ifr_matrix[PC1sorted,:]
    units = np.array(units)[PC1sorted]
    
    #cmap = sns.palplot(sns.diverging_palette(12, 255, l=40, n=100, center="dark"))
    fig = plt.figure(figsize=(15, 0.3*len(units)))
    hm = sns.heatmap(ifr_matrix, vmin=-2, vmax=2, cmap="RdBu_r", center=0, cbar_kws={'label': 'Instantaneous Firing Rate (s.d.)'})
    
    if window[0]<0:
        zero=int(len(x)*(-window[0])/(window[1]-window[0]))
        hm.axes.plot([zero,zero], hm.axes.get_ylim()[::-1], ls="--", c=[0,0,0], lw=1)
        if alignement_event=='off':
            rewt=zero+int(len(x)*400/(window[1]-window[0]))
            hm.axes.plot([rewt,rewt], hm.axes.get_ylim()[::-1], ls="--", c=[30/255,144/255,255/255], lw=1)
    hm.axes.set_yticklabels(['{}'.format(u) for u in units], rotation='horizontal')
#    x_hmm = np.zeros((1,8064))
#    hm.axes.set_xticklabels([str[i] for i in x], rotation='vertical')
#    for i in range(hmm.shape[1]):
#        if (i-8064/(window[1]-window[0]))%12!=0: hm.axes.xaxis.get_major_ticks()[i].set_visible(False)
#        else: hm.axes.xaxis.get_major_ticks()[i].set_visible(True)
    
    fig = plt.gcf()
    spt = 'Putative Purkinje cells' if region == 'cortex' else 'Cerebellar Nuclear Cells'
    fig.suptitle(spt)
    ax = plt.gca()
    
    return fig, ax
