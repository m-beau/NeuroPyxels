# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London
Behavior analysis tools.
"""

import os.path as op; opj=op.join
from pathlib import Path
import pickle

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import scipy.stats as stats
from numpy import pi, cos, sin

import cv2

from npyx.utils import npa, thresh, thresh_consec, smooth, sign, align_timeseries, assert_int, get_bins

from npyx.inout import read_metadata, get_npix_sync, paq_read, list_files
from npyx.spk_t import mean_firing_rate
from npyx.corr import crosscorr_cyrille, frac_pop_sync
from npyx.merger import assert_multi, get_ds_table

#%% Generate trials dataframe from either paqIO file or matlab datastructure

def behav_dic(dp, f_behav=None, vid_path=None, again=False, again_align=False, again_rawpaq=False,
                     lick_ili_th=0.075, n_ticks=1024, diam=200, gsd=25,
                     plot=False, drop_raw=True, cam_paqi_to_use=None):
    '''
    Remove artefactual licking, processes rotary encoder and wheel turn trials.
    - dp: str, path of kilosort dataset.
    - f_behav: str, paqIO behavioural file. If None, assumed to be at ../behavior relative to dp.
    - again: bool, whether to recompute paqdic from paq file.
    - plot: bool, whether to load plots regarding licking preprocessing to ensure all goes well.
    - drop_raw: drop raw paqIO data for digital channels (huge memory saver)
    - cam_paqi_to_use: [int, int], indices of first and last camera frames to consider for urnning analysis
    '''

    ## Try to load presaved df
    dp=Path(dp)
    fn=dp.parent/'behavior'/f'paq_dic_proc_lickth{lick_ili_th}_raw{drop_raw}.pkl'
    if again_rawpaq: again_align=True
    if again_align: again=True
    if fn.exists() and not again:
        print(f'Behavioural data found at {fn}.')
        return pickle.load(open(str(fn),"rb"))

    paqdic=npix_aligned_paq(dp,f_behav=f_behav, again=again_align, again_rawpaq=again_rawpaq)
    paq_fs=paqdic['paq_fs']
    fs=paqdic['npix_fs']
    if assert_multi(dp):
        dp_source=get_ds_table(dp)['dp'][0]
        npix_fs=read_metadata(dp_source)['highpass']['sampling_rate']
    else:
        npix_fs=read_metadata(dp)['highpass']['sampling_rate']

    # Preprocessing of extracted behavioural data (only lick onsets so far)
    licks_on=paqdic['LICKS_Piezo_ON_npix'].copy()
    paqdic['LICKS_Piezo_ON_npix']=licks_on[np.diff(np.append([0],licks_on))>lick_ili_th*npix_fs] # 0.075 is the right threshold, across mice!
    paqdic['licks']=paqdic['LICKS_Piezo_ON_npix'] # for convenience

    licks_diff = np.diff(paqdic['licks'])
    m_pre = np.append([True], licks_diff>0.4*fs).astype(bool)
    m_post = np.append((licks_diff<0.25*fs)&(licks_diff>0.1*fs), [True]).astype(bool)
    m_noprepost = m_pre&m_post
    paqdic['lick_salve_onsets'] = paqdic['licks'][m_noprepost]

    print(f'\nInter lick interval lower threshold set at {lick_ili_th} seconds.\n')
    if plot:
        hbins=np.logspace(np.log10(0.005),np.log10(10), 500)
        fig,ax=plt.subplots()
        ax.set_title('Licks distribution before filtering')
        hist=np.histogram(np.diff(licks_on/npix_fs), bins=hbins)
        ax.hist(np.diff(licks_on/npix_fs), bins=hbins)
        ax.set_xscale('log')
        plt.xlim(0.005,10)
        plt.ylim(0,max(hist[0][hist[1][:-1]>0.05])+10)
        fig,ax=plt.subplots()
        ax.set_title('Licks distribution after filtering')
        hist=np.histogram(np.diff(paqdic['LICKS_Piezo_ON_npix']/npix_fs), bins=hbins)
        ax.hist(np.diff(paqdic['LICKS_Piezo_ON_npix']/npix_fs), bins=hbins)
        ax.set_xscale('log')
        plt.xlim(0.005,10)
        plt.ylim(0,max(hist[0][hist[1][:-1]>0.05])+10)

    # Process rotary encoder if running dataset
    # Get periods with/without
    if not 'ROT_A' in paqdic.keys():
        print('No rotary encoder data found in paqdic. Assuming no running, no camera.')
    else:
        print('Running dataset detected.')
        v=decode_rotary(paqdic['ROT_A'], paqdic['ROT_B'], paq_fs, n_ticks, diam, gsd, True)
        v*=sign(abs(np.max(v))-abs(np.min(v))) # 1 if running forward is positive
        paqdic['ROT_SPEED']=v

        # Process extra camera frames
        if vid_path is None:
            vid_path=dp.parent/'videos'
        videos=list_files(vid_path, 'avi', 1) if vid_path.exists() else []

        if not any(videos):
            print(f'No videos found at {vid_path} - camera triggers not processed.\n')
        else:
            if cam_paqi_to_use is not None:
                assert len(cam_paqi_to_use)==2
                assert cam_paqi_to_use[0]>=0, 'cam_paqi_to_use[0] cannot be negative!'
                assert cam_paqi_to_use[1]<=len(paqdic['CameraFrames'])-1,\
                    f"cam_paqi_to_use[0] cannot be higher than {len(paqdic['CameraFrames'])-1}!"
                if cam_paqi_to_use[1]<0:cam_paqi_to_use[1]=len(paqdic['CameraFrames'])+cam_paqi_to_use[1]
                ON=paqdic['CameraFrames_ON']
                OFF=paqdic['CameraFrames_OFF']
                mon=(ON>=cam_paqi_to_use[0])&(ON<=cam_paqi_to_use[1])
                mof=(OFF>=cam_paqi_to_use[0])&(OFF<=cam_paqi_to_use[1])
                paqdic['CameraFrames_ON_npix']=paqdic['CameraFrames_ON_npix'][mon]
                paqdic['CameraFrames_OFF_npix']=paqdic['CameraFrames_OFF_npix'][mof]
            frames_npix=paqdic['CameraFrames_ON_npix']/paqdic['npix_fs']
            nframes=[get_nframes(str(v)) for v in videos]
            print(f'Videos nframes:{nframes} -> {np.sum(nframes)} frames in video files.')
            print(f'{frames_npix.shape[0]} triggers in paqIO file')
            print(f'PaqIO inter-trigger intervals below 1ms: {sum(np.diff(frames_npix)<0.001)}.')
            print((f"\n**{sum(npa(nframes)!=60000)}** seemingly manually aborted videos and "
                   f"\n**{frames_npix.shape[0]-sum(nframes)}** unexpected camera triggers."
                   f"\nIf these numbers do not match, figure out why. "
                    "There might be some trashed video triggers somewhere."))

            fi=0
            n_man_abort=0
            last=0
            for i,nf in enumerate(nframes):
                if nf==nframes[-1]:last=1
                fi+=nf
                ii=fi+n_man_abort
                if nf!=60000:#manually aborted
                    n_man_abort+=1
                    # ensure that where the gap between videos (>50ms)
                    # is expected, you actually have a short interval
                    assert frames_npix[ii]-frames_npix[ii-1]<0.05
                    if not last:assert frames_npix[ii+1]-frames_npix[ii]>0.05
                    frames_npix[ii]=np.nan
                else:
                    # ensure that the gap between videos (>50ms)
                    # is where it is expected
                    assert frames_npix[ii-1]-frames_npix[ii-2]<0.05
                    if not last:assert frames_npix[ii]-frames_npix[ii-1]>0.05
            frames_npix=(frames_npix[~np.isnan(frames_npix)]*paqdic['npix_fs']).astype(np.int64)
            print(f'After correction the delta between expected/actual frames is **{frames_npix.shape[0]-sum(nframes)}**.')
            paqdic['CameraFrames_ON_npix']=frames_npix

        # load swing onsets if already computed
        swings_fn = dp.parent/'behavior'/'leftfrontpaw_swings.npy'
        if swings_fn.exists():
            print('Loaded swing onsets from deeplabcut data.')
            swings_onof_i=np.load(swings_fn)
            paqdic['swing_ON_npix']=paqdic['CameraFrames_ON_npix'][swings_onof_i[:,0].ravel()]
            paqdic['swing_OFF_npix']=paqdic['CameraFrames_ON_npix'][swings_onof_i[:,1].ravel()]

        swings_fn = dp.parent/'behavior'/'swing_onsets_phasethreshold.npy'
        if swings_fn.exists():
            print('Loaded swing phase onsets from deeplabcut data.')
            swing_on = np.load(swings_fn)
            stance_on = np.load(swings_fn.parent/'stance_onsets_phasethreshold.npy')
            paqdic['swing_phase_ON_npix'] = paqdic['CameraFrames_ON_npix'][swing_on]
            paqdic['stance_phase_ON_npix'] = paqdic['CameraFrames_ON_npix'][stance_on]
            paqdic['deeplabcut_df'] = pd.read_csv(swings_fn.parent/'locomotion_processed.csv')


    # Process cues and rewards - only engaged events are considered!
    cues=paqdic['CUE_ON_npix']
    rewards=paqdic['REW_ON_npix']
    licks=paqdic['LICKS_Piezo_ON_npix']

    rewarded_cues=[]
    engaged_rewards=[]
    random_rewards=[]
    for r in rewards:
        licks_m=(r<licks)&(licks<r+2*fs)
        if any(licks_m): #engaged_rewards
            engaged_rewards.append(r)
            cues_m=(r-2*fs<cues)&(cues<r)
            if any(cues_m):
                c=cues[cues_m][0]
                rewarded_cues.append(c)
            else:
                random_rewards.append(r)

    omitted_cues=[]
    engaged_cues=[]
    for c in cues:
        licks_m=(c<licks)&(licks<c+2*fs)
        if any(licks_m):
            engaged_cues.append(c)
            rew_m=(c<rewards)&(rewards<c+2*fs)
            if not any(rew_m):
                omitted_cues.append(c)

    paqdic['random_rewards']=npa(random_rewards)    
    paqdic['omitted_cues']=npa(omitted_cues)
    paqdic['rewarded_cues']=npa(rewarded_cues)
    paqdic['engaged_rewards']=npa(engaged_rewards)
    paqdic['engaged_cues']=npa(engaged_cues)

    # Process turning wheel events if turning wheel dataset
    if 'ROTreal' in paqdic.keys():
        print('Wheel turning dataset detected.')
        wheel_turn_dic = get_wheelturn_df_dic(dp, paqdic,
                        include_wheel_data=False, add_spont_licks=False,
                        wheel_gain=3, rew_zone=12.5, rew_frames=3, vr_rate=30,
                        wheel_diam=45, difficulty=2, ballistic_thresh=100,
                        plot=False, again=again, verbose=False)[1]
        # merge giving priority to rr, cr_c and cr_r from wheel_turn_dic
        paqdic={**paqdic,**wheel_turn_dic}

    # Save behav dict
    if drop_raw:
        undesired=['RECON', 'RECON_ON', 'RECON_OFF',
                   'GAMEON', 'GAMEON_ON', 'GAMEON_OFF', 'TRIALON', 'TRIALON_ON', 'TRIALON_OFF',
                   'REW', 'CUE', 'VRframes', 'VRframes_ON', 'VRframes_OFF', 'GHOST_REW',
                   'ROT_A', 'ROT_B', 'CameraFrames', 'LICKS_Piezo']
        for k in undesired:
            if k in paqdic.keys():paqdic.pop(k)
        print(f'\nDropped: {undesired}.\n')
    pickle.dump(paqdic, open(str(fn),"wb"))

    return paqdic

def npix_aligned_paq(dp, f_behav=None, again=False, again_rawpaq=False):
    '''Aligns thresholded paqIO data at f_behav to npix data at dp.
    '''
    if again_rawpaq: again=True
    dp=Path(dp)

    behav_dir = dp.parent/'behavior'
    fn=behav_dir/'paq_dic.pkl'
    if fn.exists() and not again:
        print(f'Paq aligned data found at {fn}.')
        return pickle.load(open(str(fn),"rb"))

    ## Load paq data and npix sync channel data
    if f_behav is None:
        files=list_files(behav_dir, 'paq')
        assert len(files)>0, (f"WARNING no files with extension 'paq' were found at {behav_dir}"
                               " - either add one there or explicitely declare a file path with f_behav parameter.")
        assert len(files)==1, (f"WARNING more than 1 file with extension 'paq' were found at '{behav_dir}' - "
                                "clean up your directory or use f_behav argument and try again.")
        f_behav=behav_dir/files[0]
        print(f'Behavioural data loaded from: {f_behav}')
    paqdic=load_PAQdata(f_behav, variables='all', unit='samples', again=again_rawpaq)
    paq_fs=paqdic['paq_fs']
    npix_ons, npix_ofs = get_npix_sync(dp, output_binary = False, filt_key='highpass', unit='samples')
    if assert_multi(dp):
        dp_source=get_ds_table(dp)['dp'][0]
    else: dp_source=dp
    npix_fs = read_metadata(dp_source)['highpass']['sampling_rate']
    paqdic['npix_fs']=npix_fs

    ## Match Paq data to npix data - convert paq onsets/offsets to npix time frame (or directly use it if available)
    # First, match npix sync channels to paqIO channels through exhaustive screening
    # (only one match for 3B recordings, several for 3A recordings)
    paq_npix_df=pd.DataFrame(columns=['npix', 'paq', 'p', 'len_match'])
    for npixk, npixv in npix_ons.items():
        for paqk, paqv in paqdic.items():
            if '_ON' in paqk and len(paqv)>1:
                p=stats.kstest(np.diff(npixv)/npix_fs, np.diff(paqv)/paq_fs)[1]
                lenmatch=int(len(paqdic[paqk])==len(npix_ons[npixk]))
                paq_npix_df=paq_npix_df.append({'npix':npixk, 'paq':paqk, 'p':p, 'len_match':lenmatch}, ignore_index=True)
    npix_paq={}
    for npixk in npix_ons.keys():
        match_p=(paq_npix_df.loc[paq_npix_df['npix']==npixk, 'p']>0.99)
        match_l=(paq_npix_df.loc[paq_npix_df['npix']==npixk, 'len_match']==1)
        if match_p.any() or match_l.any():
            paqk=paq_npix_df.loc[paq_npix_df['npix']==npixk, 'paq'][match_p|match_l].values
            assert paqk.shape[0]==1, f'WARNING, more than one match found ({paqk}) between npix sync channel and PaqIO!!'
            paqk=paqk[0]
            print(f'\n>>> Match found between npix channel {npixk} and paqIO channel {paqk} ({len(npix_ons[npixk])} events)!\n')
            npix_paq[npixk]=paqk
            npix_paq[paqk]=npixk
    assert len(npix_paq)>0, 'WARNING no match was found between paqIO file and npix sync channel!'

    # Then, pick the longest matching sync channel to align paqIO channels not acquired with npix
    len_arr=npa([[k,len(npix_ons[k])] for k in npix_paq.keys() if k in npix_ons.keys()])
    sync_npix_k=len_arr[np.argmax(len_arr[:,1]),0]
    sync_npix=npix_ons[sync_npix_k]
    sync_paq=paqdic[npix_paq[sync_npix_k]]

    # Model drift: npix_sync = a * paq_sync + b
    (a, b) = np.polyfit(sync_paq, sync_npix, 1)
    paqdic['a']=a
    paqdic['b']=b
    print(f'Drift (assumed linear) of {round(abs(a*paq_fs/npix_fs-1)*3600*1000,2)}ms/h, \noffset of {round(b/npix_fs,2)}s between ephys and paq files.\n')
    for paqk in list(paqdic.keys()):
        paqv=paqdic[paqk]
        if '_ON' in paqk and len(paqv)>1:
            off_key=f"{paqk.split('_ON')[0]+'_OFF'}"
            if paqk in npix_paq.keys():
                paqdic[f'{paqk}_npix']=npix_ons[npix_paq[paqk]]
                paqdic[f"{off_key}_npix"]=npix_ofs[npix_paq[paqk]] # same key for onsets and offsets
            else:
                paqdic[f'{paqk}_npix_old']=align_timeseries([paqv], [sync_paq, sync_npix], [paq_fs, npix_fs]).astype(np.int64)
                paqdic[f"{off_key}_npix_old"]=align_timeseries([paqdic[off_key]], [sync_paq, sync_npix], [paq_fs, npix_fs]).astype(np.int64)
                paqdic[f'{paqk}_npix']=np.round(a*paqv+b, 0).astype(np.int64)
                paqdic[f"{off_key}_npix"]=np.round(a*paqdic[off_key]+b, 0).astype(np.int64)

    pickle.dump(paqdic, open(str(fn),"wb"))

    return paqdic

def load_PAQdata(paq_f, variables='all', again=False, unit='seconds', th_frac=0.2):
    '''
    Used to load analog (wheel position...)
    and threshold digital (piezo lick...) variables from paqIO file.
    If variables is not a list, all PackIO variables will be exported.

    Parameters:
        - paq_f: string or PosixPath, path to .paq file.
        - variables: 'all' or list of strings, paqIO variables to output.
        - again: boolean, if True does not try to load pre-saved data.
        - units: units of the returned thresholded arrays
        - th_frac: threshold used on the raw signal, in fraction of min to max.
    Returns:
        - paqdic, dictionnary of all variables (under key var)
          as well as onset/offsets of digital variables (under keys var_ON and var_OFF)
    '''

    # Load paq data
    paq = paq_read(paq_f)

    # Attempt to load pre-saved paqdata
    paq_f=Path(paq_f)
    fn=paq_f.parent/(paq_f.name.split('.')[0]+'_all_samples.pkl')
    if fn.exists() and not again:
        rawPAQVariables = pickle.load(open(fn,"rb"))
        if type(variables) is str: assert variables=='all'
        else:
            assert type(variables) is list
            assert np.all(np.isin(variables, list(rawPAQVariables.keys())))
            variables=variables+[v+'_ON' for v in variables]+[v+'_OFF' for v in variables]
            rawPAQVariables = {k:rawPAQVariables[k] for k in rawPAQVariables.keys() if k in variables}
    else:
        # Load raw packIO data and process variables
        allVariables = np.array(paq['chan_names'])
        vtypes = {'RECON':'digital', 'GAMEON':'digital', 'TRIALON':'digital',
          'REW':'digital', 'GHOST_REW':'digital', 'CUE':'digital', 'LICKS':'digital',
          'VRframes':'digital', 'REW_GHOST':'digital', 'ROT':'analog',
          'ROTreal':'analog', 'CameraFrames':'digital', 'LICKS_Piezo':'digital', 'LICKS_elec':'digital',
          'opto_stims':'digital', 'ROT_A':'digital', 'ROT_B':'digital'}
        if type(variables)==list:
            variables=np.array(variables)
            areIn = np.isin(variables, allVariables)
            if not np.all(areIn):
                print('WARNING: {} is not in the list of accepted variables {}. Exitting now.'.format(variables[~areIn], allVariables))
                return
        else:
            assert variables=='all'
            areIn=np.isin(allVariables, list(vtypes.keys()))

            assert np.all(areIn), f'''WARNING variables found in PaqIO are not characterized as digital or analog in function!
            Variables not characterized in function: {npa(allVariables)[~areIn]}'''
            variables = allVariables
        variables = {variables[i]:vtypes[variables[i]] for i in range(len(variables))}

        # Process packIO data and store it in a dict
        rawPAQVariables = {}
        rate_key=[k for k in list(paq.keys()) if 'rate' in k][0]
        rawPAQVariables['paq_fs']=paq[rate_key]
        print('>> PackIO acquired channels: {}, of which {} will be extracted...'.format(allVariables, list(variables.keys())))
        for v in variables.keys():
            (i, ) = np.nonzero(v==np.array(allVariables))[0]
            print('Extracting PackIO channel {}...'.format(v))
            data = paq['data'][i]
            rawPAQVariables[v] = data
            if variables[v]=='digital':
                print('    Thresholding...')
                th = min(data)+(max(data)-min(data))*th_frac
                rawPAQVariables[v+'_ON'] = thresh(data, th, 1).astype(np.int64)
                rawPAQVariables[v+'_OFF'] = thresh(data, th, -1).astype(np.int64)

        # Pickle it
        if np.all(list(variables.keys())==allVariables):
            pickle.dump(rawPAQVariables, open(fn,"wb"))
        if not fn.exists(): print('WARNING There was a pickle dumping issue, do it manually!!')

    assert unit in ['seconds', 'samples']
    conv=paq['rate'] # PAQIO: 5kHz acquisition
    if unit=='seconds': rawPAQVariables={k:v/conv if (('_ON' in k)|('_OFF' in k)) else v for k, v in rawPAQVariables.items()}

    return rawPAQVariables

def get_wheelturn_df_dic(dp, paqdic, include_wheel_data=False, add_spont_licks=False,
                    wheel_gain=3, rew_zone=12.5, rew_frames=3, vr_rate=30,
                    wheel_diam=45, difficulty=2, ballistic_thresh=100,
                    plot=False, again=False, verbose=True):
    '''
    Parameters:
        - dp: str, path to neuropixels data directory (behavioural data is in dp/behavior)
        - paqdic: paqIO dictionnary, with data cleaned up
          including 'ROT' (wheel pos in degrees, 0 is center), 'ROTreal' (object pos in degrees, 0 is center)

        - include_wheel_data: bool, whether to add memory-heavy object position (in degrees) and wheel position (in mm) to the dataframe,
                              sampled at paqIO sampling rate, and related metrics (movement onset in relative paqIO units etc)
        - add_spont_licks: bool, whether to add memory-heavy spontaneous lick onsets
                           at the end of the dataframe as trial_type 'spontaneous_licks'

        - wheel_gain: float, gain between the wheel and the object (vr.rotgain virmen setting) | Default 3
        - rew_zone: float, target zone at the center (vr.rewarddeg virmen setting) | Default 12.5
        - rew_frames: int, number of frames where speed needs to be 0 (vr.rewardframes virmen setting) | Default 3 (about 100ms)
        - vr_rate: int, refresh rate of virmen engine (Hz) | Default 30
        - wheel_diam: float, lego wheel diameter in mm | Default 45
        - difficulty: int, difficulty level of the task (vr.version virmen setting).
                      2: 2 sided, overshoot allowed. 3: overshoot not allowed, probably 1 sided. | Default 2
        - ballistic_thresh: how short a movement must to be called 'ballistic', in milliseconds | Default 100

        - plot: bool, whether to plot wheel position, speed and detected movement onset/offset as the dataframe is populated | Default True

    Returns:
        behav_df: session summary pandas dataframe, with columns
            'trial_type'        - wheel_turn, random_reward, cued_reward or cue_alone
            'trialnum'          - wheel turning trial number
            'trialside'         - 1 or -1 (sign matches paqIO ROT channel start value)
            'trial_onset'       - trial onset, in neuropixels temporal reference frame
            'object_onset'      - object appearance onset, npix frame
            'movement_onset'    - detected movement onset (norm. speed threshold), npix frame
            'movement_offset'   - detected movement ofset (enters reward zone), npix frame
            'movement_duration' - movement duration in ms (offset-onset)
            'ballistic'         - True if movement is shorter than ballistic_thresh
            'correct'           - 1 if correct, 0 if not
            'trial_offset'      - trial offset, in npix frame
            'reward_onset'      - reward onset, in npix frame
            'cue_onset'         - cue onset, in npix frame
            'ghost_onset'       - empty solenoid (fake reward) onset, in npix frame
            'lick_onsets'       - array of lick onsets happening between trial onset and 5s after trial offset
                                  or reward and 5s after reward for random rewards RR
                                  or cue and 5s after reward for cued rewards CR
                                  or cue and 5s after cue for omitted rewards OR
            and if include_wheel_data:
                'object_position'        - in degrees, in paqIO samples (5000Hz), clipped from to 4s before trial onset to 4s after offset
                'wheel_position'         - in mm, in paqIO samples (5000Hz), clipped similarly
                'trial_onset_relpaq'     - trial onset, in object_position and wheel_position relative paqIO samples
                'movement_onset_relpaq'  - movement onset, in object_position and wheel_position relative paqIO samples
                'movement_offset_relpaq' - movement offset, in object_position and wheel_position relative paqIO samples
                'trial_offset_relpaq'    - trial offset, in object_position and wheel_position relative paqIO samples
                'lick_trace'             - piezo_licks trace, in paqIO samples (5000Hz), clipped similarly

        behav_dic: behav_df with events extracted in dictionnary with the following keys
          > 'olc_on'/'oli_on': visual object onsets for correct trials
          > 'orc_on'/orr_on: visual object onsets for incorrect trials

          > 'lc_on'/'lc_of: movement on/offsets for left correct trials
          > 'rc_on'/'rc_of: movement on/offsets for right correct trials

          > 'c_r: reward onset for correct trials
          > 'i_of: trial offsets for correct trials (timeout)
          > all other keys in paqdic
    '''

    ## Organize them in dataset, all in NEUROPIXELS time frame
    # i.e. (use above-aligned paqdic[f'{paqk}_npix'] as onsets)
    dp = Path(dp)
    fn=dp.parent/f"behavior/trials_df-{include_wheel_data}-{add_spont_licks}-{difficulty}-{ballistic_thresh}.csv"

    ## Conversions and parameters processing
    paq_fs, npix_fs = paqdic['paq_fs'], paqdic['npix_fs']
    # ROTreal is object displacement in degrees
    # ROT is wheel displacement in degrees
    wheel_turn_dic={}
    wheel_turn_dic['vr_fs']=vr_rate
    wheel_turn_dic['wheel_position_mm']=paqdic['ROTreal']*(np.pi*wheel_diam)/360/wheel_gain
    wheel_turn_dic['wheel_speed_cm/s']=np.diff(wheel_turn_dic['wheel_position_mm'])*vr_rate/10 # mm to cm/s

    if Path(fn).exists() and not again:
        df = pd.read_csv(fn)
    else:
        ## Make and fill trials dataframe
        df=pd.DataFrame(columns=['trial_type', 'trialnum', 'trialside', 'trial_onset', 'object_onset',
                                'movement_onset', 'movement_offset', 'movement_duration', 'ballistic', 'correct', 'trial_offset',
                                'reward_onset', 'cue_onset', 'ghost_onset', 'lick_onsets'])

        # Process wheel trials
        nwheeltrials=len(paqdic['TRIALON_ON'])
        df['trial_type']=['wheel_turn']*nwheeltrials
        df["lick_onsets"]=df["lick_onsets"].astype(object) # to be able to store array
        if include_wheel_data:
            df['object_position']=np.nan
            df['wheel_position']=np.nan
            df['trial_onset_relpaq']=np.nan
            df['movement_onset_relpaq']=np.nan
            df['movement_offset_relpaq']=np.nan
            df['trial_offset_relpaq']=np.nan
            df['lick_trace']=np.nan
            df["object_position"]=df["object_position"].astype(object)
            df["wheel_position"]=df["wheel_position"].astype(object)
            df['lick_trace']=df["wheel_position"].astype(object)
        pad=4
        assert difficulty in [2,3]
        for tr in df.index:
            if verbose: print(f'  Wheel steering trial {tr}/{len(df.index)}...')
            npixon=paqdic['TRIALON_ON_npix'][tr]
            npixof=paqdic['TRIALON_OFF_npix'][tr]
            paqon=int(paqdic['TRIALON_ON'][tr])
            paqoff=int(paqdic['TRIALON_OFF'][tr])
            i1,i2 = int(paqon-pad*paq_fs),int(paqoff+pad*paq_fs)
            ob_on_vel=np.diff(paqdic['ROTreal'][paqon:paqon+500])
            ob_on_vel=abs(ob_on_vel)/max(abs(ob_on_vel))
            ob_on_paq=thresh(ob_on_vel, 0.5, 1)[0]+1 # add 1 because velocity is thresholded
            start_side=sign(paqdic['ROT'][paqon+ob_on_paq+1]) # 1 or -1
            # wheel and object positions are clipped between -4s before trial onset and 4s after trial offset
            opos=paqdic['ROT'][i1:i2] # wheel pos in degrees
            wpos=wheel_turn_dic['wheel_position_mm'][i1:i2] # in mm
            wpos_mvt=paqdic['ROTreal'][paqon+ob_on_paq:paqoff]
            wvel=np.diff(wpos)
            wvel_mvt=wvel[int(pad*paq_fs+ob_on_paq):-int(pad*paq_fs)]
            wvel_mvt_norm=wvel_mvt/min(abs({-1:max, 1:min}[start_side](wvel_mvt)), 2)

            # assess trial outcome from wheel kinetics
            wpos_outeval=paqdic['ROT'][int(paqon+ob_on_paq):int(paqoff)]
            stay_for_rew=int(rew_frames/vr_rate*paq_fs)
            correct=0
            center_crossed=False
            th0=np.append(thresh(wpos_outeval, 0, 1), thresh(wpos_outeval, 0, -1))
            if np.any(th0):
                jump=wpos_outeval[th0[0]+1]-wpos_outeval[th0[0]-1]
                if jump<300: # handles cases where the object went all the way around -> false positive threshold cross
                    center_crossed = True
            stayed_rew_zone=np.all((paqdic['ROT'][int(paqoff)-stay_for_rew:int(paqoff)]<=rew_zone)&\
                                (paqdic['ROT'][int(paqoff)-stay_for_rew:int(paqoff)]>=-rew_zone))
            #non_responsive=(paqdic['ROT'][int(paqoff)]<=rew_zone or paqdic['ROT'][int(paqoff)]>=-rew_zone)
            if difficulty>=2: # rule: crossing threshold or ending in reward zone
                if center_crossed or stayed_rew_zone: correct=1
            if difficulty>=3: # rule: ending in reward zone
                if stayed_rew_zone: correct=1
            #if non_responsive: correct=np.nan

            # Fill dataframe
            df.loc[tr, 'trialnum']=tr
            df.loc[tr, 'trial_onset']=npixon
            df.loc[tr, 'trial_offset']=npixof
            df.loc[tr, 'object_onset']=int(npixon+ob_on_paq*npix_fs/paq_fs)
            df.loc[tr, 'trialside']=start_side
            df.loc[tr, 'correct']=correct
            lickmask=(paqdic['LICKS_Piezo_ON_npix']>npixon)&(paqdic['LICKS_Piezo_ON_npix']<npixof+5*npix_fs)
            df.loc[tr, "lick_onsets"]=paqdic['LICKS_Piezo_ON_npix'][lickmask]
            if correct:
                movonpaq=ob_on_paq+thresh(wvel_mvt_norm, -start_side*0.5, -start_side)[0]
                movofpaq=ob_on_paq+np.append(thresh(wpos_mvt, rew_zone, -1), thresh(wpos_mvt, -rew_zone, 1)).min()
                mov_dur=(movofpaq-movonpaq)*1000/paq_fs
                ballistic=(mov_dur<ballistic_thresh) # in ms
                if plot:
                    plt.figure()
                    plt.plot(wvel[int(pad*paq_fs+500):-int(pad*paq_fs)], c='grey')
                    plt.plot(wpos[int(pad*paq_fs+500):-int(pad*paq_fs)], c='k')
                    ls='--' if ballistic else '-'
                    plt.plot([movonpaq-500, movonpaq-500], [min(wpos), max(wpos)], c='g', ls=ls)
                    plt.plot([movofpaq-500, movofpaq-500], [min(wpos), max(wpos)], c='r', ls=ls)
                    plt.title(f'trial {tr}\nduration:{mov_dur}ms')
                df.loc[tr, 'movement_onset']=int(npixon+(movonpaq)*npix_fs/paq_fs)
                df.loc[tr, 'movement_offset']=int(npixon+(movofpaq)*npix_fs/paq_fs)
                df.loc[tr, 'movement_duration']=mov_dur
                df.loc[tr, 'ballistic']=ballistic
            if include_wheel_data:
                df.at[tr, 'object_position']=opos
                df.at[tr, 'wheel_position']=wpos
                df.loc[tr, 'trial_onset_relpaq']=int(pad*paq_fs)
                df.loc[tr, 'movement_onset_relpaq']=int(pad*paq_fs)+movonpaq
                df.loc[tr, 'movement_offset_relpaq']=int(pad*paq_fs)+movofpaq
                df.loc[tr, 'trial_offset_relpaq']=len(wpos)-int(pad*paq_fs)
                df.at[tr, 'lick_trace']=paqdic['LICKS_Piezo'][i1:i2]

        # Append trial rewards onsets ##TODO
        for tr in df.index:
            of=df.loc[tr, 'trial_offset']
            rew_mask=(paqdic['REW_ON_npix']>of)&(paqdic['REW_ON_npix']<(of+1*npix_fs)) # happens about 400ms after trial offset
            if np.any(rew_mask):
                if df.loc[tr, 'correct']!=1: print(f'WARNING seems like a reward onset was found after trial {tr} offset, although incorrect...\
                                                Probably beause of 360deg jump of ROT channel not followed by ROTreal.')
                df.loc[tr, 'reward_onset']=paqdic['REW_ON_npix'][rew_mask][0]
            else:
                if df.loc[tr, 'correct']==1: print(f'WARNING no reward was found after trial {tr} offset, although correct!!! Figure out the problem ++!!')

        # Now find random rewards and respective cues if any
        random_rewards=paqdic['REW_ON_npix'][~np.isin(paqdic['REW_ON_npix'], df['reward_onset'])]
        for r in random_rewards:
            cue_mask=(paqdic['CUE_ON_npix']>r-1*npix_fs)&(paqdic['CUE_ON_npix']<r) # cues happen about 500ms before reward
            i=df.index[-1]+1
            if np.any(cue_mask):
                c=paqdic['CUE_ON_npix'][cue_mask][0]
                df.loc[i, 'trial_type']='cued_reward'
                df.loc[i, 'reward_onset']=r
                df.loc[i, 'cue_onset']=c
                lickmask=(paqdic['LICKS_Piezo_ON_npix']>c)&(paqdic['LICKS_Piezo_ON_npix']<r+5*npix_fs) # from cue to 5s after reward
            else:
                df.loc[i, 'trial_type']='random_reward'
                df.loc[i, 'reward_onset']=r
                lickmask=(paqdic['LICKS_Piezo_ON_npix']>r)&(paqdic['LICKS_Piezo_ON_npix']<r+5*npix_fs) # from reward to 5s after reward
            df.loc[i, "lick_onsets"]=paqdic['LICKS_Piezo_ON_npix'][lickmask]

        # Finally, fill in the cues alone (omitted rewards)
        cues_alone=paqdic['CUE_ON_npix'][~np.isin(paqdic['CUE_ON_npix'], df['cue_onset'])]
        for c in cues_alone:
            i=df.index[-1]+1
            df.loc[i, 'trial_type']='cue_alone'
            df.loc[i, 'cue_onset']=c
            lickmask=(paqdic['LICKS_Piezo_ON_npix']>c)&(paqdic['LICKS_Piezo_ON_npix']<c+5*npix_fs) # from cue to 5s after cue
            df.loc[i, "lick_onsets"]=paqdic['LICKS_Piezo_ON_npix'][lickmask]

        # Also add spontaneous licks
        if add_spont_licks:
            allocated_licks=npa([list(df.loc[i, "lick_onsets"]) for i in df.index]).flatten()
            spontaneous_licks=paqdic['LICKS_Piezo_ON_npix'][~np.isin(paqdic['LICKS_Piezo_ON_npix'], allocated_licks)]
            i=df.index[-1]+1
            df.loc[i, 'trial_type']='spontaneous_licks'
            df.loc[i, "lick_onsets"]=spontaneous_licks

        df.to_csv(fn)

    # finally, convert to dictionnary - more convenient
    mask_left=(df['trialside']==1)
    mask_right=(df['trialside']==-1)
    mask_correct=(df['correct']==1)&(df['ballistic']==1)
    mask_incorrect=(df['correct']==0)
    mask_rr=(df['trial_type']=='random_reward')
    mask_cr=(df['trial_type']=='cued_reward')
    wheel_turn_dic={**wheel_turn_dic,
    **{'olc_on':  df.loc[mask_left&mask_correct, 'object_onset'].values.astype(float),
     'oli_on':  df.loc[mask_left&mask_incorrect, 'object_onset'].values.astype(float),
     'orc_on':  df.loc[mask_right&mask_correct, 'object_onset'].values.astype(float),
     'ori_on':  df.loc[mask_right&mask_incorrect, 'object_onset'].values.astype(float),

     'lc_on':  df.loc[mask_left&mask_correct, 'movement_onset'].values.astype(float),
     'rc_on':  df.loc[mask_right&mask_correct, 'movement_onset'].values.astype(float),
     'lc_of':  df.loc[mask_left&mask_correct, 'movement_offset'].values.astype(float),
     'rc_of':  df.loc[mask_right&mask_correct, 'movement_offset'].values.astype(float),

     'c_r':  df.loc[mask_correct, 'reward_onset'].values.astype(float),
     'i_of':  df.loc[mask_incorrect, 'trial_offset'].values.astype(float),

    # random/cued rewards are now handled in get_behav_dic
    'rr':    df.loc[mask_rr, 'reward_onset'].values.astype(float),
    'cr_c':    df.loc[mask_cr, 'cue_onset'].values.astype(float),
    'cr_r':    df.loc[mask_cr, 'reward_onset'].values.astype(float)}}

    return df, wheel_turn_dic

#%% Alignement, binning and processing of time series

def align_variable(events, variable_t, variable, b=2, window=[-1000,1000], remove_empty_trials=False):
    '''
    Parameters:
        - events: list/array in seconds, events to align timestamps to
        - variable_t: list/array in seconds, timestamps to align around events.
        - variable: list/array, variable to bin around event.
        - bin: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - aligned_t: dictionnaries where each key is an event in absolute time and value the times aligned to this event within window.
        - binned_variable: a len(events) x window/b matrix where the variable has been binned, in variable units.
    '''


    events, variable_t, variable = npa(events), npa(variable_t), npa(variable)
    assert np.any(events), 'You provided an empty array of events!'
    assert variable_t.ndim==1
    assert len(variable_t)==len(variable)
    sort_i = np.argsort(variable_t)
    variable_t = variable_t[sort_i]
    variable = variable[sort_i]
    window_s=[window[0]/1000, window[1]/1000]

    aligned_t = {}
    tbins=np.arange(window[0], window[1]+b, b)
    binned_variable = np.zeros((len(events), len(tbins)-1)).astype(float)
    for i, e in enumerate(events):
        ts = variable_t-e # ts: t shifted
        ts_m = (ts>=window_s[0])&(ts<=window_s[1])
        ts_win = ts[ts_m] # tsc: ts in widnow
        variable_win = variable[ts_m]
        if np.any(ts_win) or not remove_empty_trials:
            aligned_t[e]=ts_win.tolist()
            tscb = np.histogram(ts_win*1000, bins=tbins, weights=variable_win)[0] # tscb: tsc binned
            tscb_n = np.histogram(ts_win*1000, bins=tbins)[0]
            tscb_n[tscb_n==0]=1 # no 0 div
            binned_variable[i,:] = tscb/tscb_n
        else:
            binned_variable[i,:] = np.nan
    binned_variable=binned_variable[~np.isnan(binned_variable).any(axis=1)]

    if not np.any(binned_variable): binned_variable = np.zeros((len(events), len(tbins)-1))

    return aligned_t, binned_variable

def align_times(times, events, b=2, window=[-1000,1000], remove_empty_trials=False):
    '''
    Parameters:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - aligned_t: dictionnaries where each key is an event in absolute time and value the times aligned to this event within window.
        - aligned_tb: a len(events) x window/b matrix where the spikes have been aligned, in counts.
    '''
    assert np.any(events), 'You provided an empty array of events!'
    t = np.sort(times)
    aligned_t = {}
    tbins=np.arange(window[0], window[1], b)
    aligned_tb = np.zeros((len(events), len(tbins))).astype(float)
    for i, e in enumerate(events):
        ts = t-e # ts: t shifted
        tsc = ts[(ts>=window[0]/1000)&(ts<=window[1]/1000)] # tsc: ts clipped
        if np.any(tsc) or not remove_empty_trials:
            aligned_t[e]=tsc.tolist()
            tscb = np.histogram(tsc*1000, bins=np.arange(window[0],window[1]+b,b))[0] # tscb: tsc binned
            aligned_tb[i,:] = tscb
        else:
            aligned_tb[i,:] = np.nan
    aligned_tb=aligned_tb[~np.isnan(aligned_tb).any(axis=1)]

    if not np.any(aligned_tb): aligned_tb = np.zeros((len(events), len(tbins)))

    return aligned_t, aligned_tb

def time_to_phase():

    return

def align_times_manyevents(times, events, b=2, window=[-1000,1000], fs=30000):
    '''
    Will run faster than align_times if many events are provided (will run in approx 800ms for 10 or for 600000 events,
                                                                  whereas align_times will run in about 1 second every 2000 event
                                                                  so in 5 minutes for 600000 events!)
    Parameters:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - fs: sampling rate in Hz - does not need to be exact, but the times and events arrays multiplied by that should be integers
    Returns:
        - aligned_tb: a 1 x window/b matrix where the spikes have been aligned, in counts.
    '''
    tfs, efs = np.round(times*fs, 2), np.round(events*fs, 2)
    assert np.all(tfs==tfs.astype(np.int64)), 'WARNING sampling rate must be wrong or provided times are not in seconds!'
    indices=np.append(0*events, 0*times+1).astype(np.int64)
    times=np.append(efs, tfs).astype(np.int64)

    sorti=np.argsort(times)
    indices, times = indices[sorti], times[sorti]

    win_size=np.diff(window)[0]
    bin_size=b

    aligned_tb=crosscorr_cyrille(times, indices, win_size, bin_size, fs=fs, symmetrize=True)[0,1,:]

    return aligned_tb

def jPSTH(spikes1, spikes2, events, b=2, window=[-1000,1000], convolve=False, method='gaussian', gsd=2):
    '''
    From A. M. H. J. AERTSEN, G. L. GERSTEIN, M. K. HABIB, AND G. PALM, 1989, Journal of Neurophysiology
    Dynamics of neuronal firing correlation: modulation of 'effective connectivity'

    Parameters:
        - spikes1, spikes2: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - fs: sampling rate in Hz - does not need to be exact, but the times and events arrays multiplied by that should be integers
    Returns:
        - aligned_tb: a 1 x window/b matrix where the spikes have been aligned, in counts.
    '''

    # psth1 is reversed in time for plotting purposes (see [::-1] all over the place)
    psth1=align_times(spikes1, events, b, window, remove_empty_trials=False)[1]
    psth2=align_times(spikes2, events, b, window, remove_empty_trials=False)[1]
    if convolve:
        psth1 = smooth(psth1, method=method, sd=gsd)
        psth2 = smooth(psth2, method=method, sd=gsd)

    ntrials=psth1.shape[0]
    nbins=psth1.shape[1]

    # Compute raw jPSTH
    # jpsth[u,v] = sum(psth1[:,u]*psth2[:,v])/K where rows are trials=events, so K=len(events)
    # see Aertsen et al. equation 3
    jpsth_raw=np.dot(psth1[:,::-1].T, psth2)/ntrials # Eq 3

    # Compute shift predictor, dot product of the mean psths across trials
    # (mathematically equivalent to averaging the raw jPSTHs
    # for every permutation of trials of 1 psth while keeping the other the same)
    # it shows vertical and hosrizontal features, but no diagonal features
    shift_predictor=np.dot(psth1.mean(0)[np.newaxis,::-1].T, psth2.mean(0)[np.newaxis,:]) # Eq 4
    D_ij=jpsth_raw-shift_predictor # Eq 5

    # Normalize jPSTH: units of correlation coefficient between -1 and 1
    s_ij=np.dot(psth1.std(0)[np.newaxis,::-1].T, psth2.std(0)[np.newaxis,:]) # sd across trials, see Eq 7a
    jpsth=D_ij/s_ij # Eq 9

    # Compute jPSTH interesting projections.
    # Only use subsquare at 45 degress from jPSTH to compute CCG projection.
    rot_jpsth135=sp.ndimage.rotate(jpsth, angle=-135)
    a=rot_jpsth135.shape[0]
    c=a/2 # b**2=2*(a/2)**2 <=> b=a*np.sqrt(2)/2. Similarly, c=b*np.sqrt(2)/2. So c=a/2.
    jpsth_ccg=rot_jpsth135[int(a-c)//2:-int(a-c)//2, int(a-c)//2:-int(a-c)//2]

    rot_jpsth45=sp.ndimage.rotate(jpsth, angle=-45)
    a=rot_jpsth45.shape[0]
    c=a/2 # b**2=2*(a/2)**2 <=> b=a*np.sqrt(2)/2. Similarly, c=b*np.sqrt(2)/2. So c=a/2.
    coincidence_psth=rot_jpsth45[int(a-c)//2:-int(a-c)//2, int(a-c)//2:-int(a-c)//2]

    return jpsth, jpsth_ccg, coincidence_psth

def get_ifr(times, events, b=2, window=[-1000,1000], remove_empty_trials=False):
    '''
    Parameters:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - ifr: a len(events) x window/b matrix where the spikes have been aligned, in Hertz.
    '''
    at, atb = align_times(times, events, b, window, remove_empty_trials)
    return atb/(b*1e-3)

def process_2d_trials_array(y, y_bsl, zscore=False, zscoretype='within',
                            convolve=False, gsd=1, method='gaussian',
                            bsl_subtract=False,
                            process_y=False):
    # zscore or not
    assert zscoretype in ['within', 'across']
    if zscore or bsl_subtract: # use baseline of ifr far from stimulus
        y_mn = np.mean(np.mean(y_bsl, axis=0))
        if zscore:
            assert not bsl_subtract, 'WARNING, cannot zscore AND baseline subtract - pick either!'
            if zscoretype=='within':
                y_mn = np.mean(np.mean(y_bsl, axis=0))
                y_sd = np.std(np.mean(y_bsl, axis=0))
                if y_sd==0 or np.isnan(y_sd): y_sd=1
                y_p = (np.mean(y, axis=0)-y_mn)/y_sd
                y_p_var = stats.sem((y-y_mn)/y_sd, axis=0) # variability across trials in zscore values??
                if process_y: y =  (y-y_mn)/y_sd
            elif zscoretype=='across':
                y_mn = np.mean(y_bsl.flatten())
                y_sd = np.std(y_bsl.flatten())
                if y_sd==0 or np.isnan(y_sd): y_sd=1
                y_p = (np.mean(y, axis=0)-y_mn)/y_sd
                y_p_var = stats.sem((y-y_mn)/y_sd, axis=0) # variability across trials in zscore values??
                if process_y: y = (y-y_mn)/y_sd

        elif bsl_subtract:
            y_p = np.mean(y, axis=0)-y_mn
            y_p_var= stats.sem(y, axis=0)
            if process_y: y = y-y_mn

    else:
        y_p = np.mean(y, axis=0)
        y_p_var = stats.sem(y, axis=0) # sd across trials

    assert not np.any(np.isnan(y_p)), 'WARNING nans found in trials array!'
    # Convolve or not
    if convolve:
        y_p = smooth(y_p, method=method, sd=gsd)
        y_p_var = smooth(y_p_var, method=method, sd=gsd)
        if process_y: y=smooth(y, method=method, sd=gsd)

    if np.any(np.isnan(y_p_var)):
        y_p_var=np.ones(y_p.shape)
        #print('WARNING not enough spikes around events to compute std, y_p_var was filled with nan. Patched by filling with ones.')

    return y, y_p, y_p_var

def get_processed_ifr(times, events, b=10, window=[-1000,1000], remove_empty_trials=False,
                      zscore=False, zscoretype='within',
                      convolve=False, gsd=1, method='gaussian',
                      bsl_subtract=False, bsl_window=[-4000, 0], process_y=False):
    '''
    Parameters:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
        - convolve: boolean, set to True to convolve the aligned binned train with a half-gaussian window to smooth the ifr
        - gsd: float, gaussian window standard deviation in ms
        - method: convolution window shape: gaussian, gaussian_causal, gamma | Default: gaussian
        - bsl_substract: whether to baseline substract the trace. Baseline is taken as the average of the baseline window bsl_window
        - bsl_window: [t1,t2], window on which the baseline is computed, in ms -> used for zscore and for baseline subtraction (i.e. zscoring without dividing by standard deviation)
        - process_y: whether to also process the raw trials x bins matrix y (returned raw by default)
    Returns:
        - x: 1D array tiling bins, in milliseconds
        - y: 2D array NtrialsxNbins, the unprocessed ifr (by default - can be processed if process_y is set to True)
        - y_mn
        - y_p
        - y_p_sem
    '''

    # Window and bins translation
    x = np.arange(window[0], window[1], b)
    y = get_ifr(times, events, b, window)
    y_bsl = get_ifr(times, events, b, bsl_window)
    assert not np.any(np.isnan(y.ravel())), 'WARNING nans found in aligned ifr!!'
    if x.shape[0]>y.shape[1]:
        x=x[:-1]
    assert x.shape[0]==y.shape[1]

    # Get mean firing rate to remove trials with too low fr (prob drift)
    # but fully empty trials were already removed in align_times
    if remove_empty_trials:
        y=y.astype(float)
        low_fr_th=0.2 #%
        consec_time=500 #ms
        consec_time=consec_time//b
        m_fr=mean_firing_rate(times, exclusion_quantile=0.005, fs=1) # in seconds
        for triali, trial in enumerate(y):
            fr_dropped=thresh_consec(trial, m_fr*low_fr_th, sgn=-1, n_consec=consec_time, exclude_edges=True, only_max=False, ret_values=True)
            if len(fr_dropped)>0: y[triali,:]=np.nan
        drop_mask = np.isnan(y[:,0])
        y=y[~drop_mask,:]
        y_bsl=y_bsl[~drop_mask,:]
        if not np.any(y): y = np.zeros((1,x.shape[0]))

    y, y_p, y_p_var = process_2d_trials_array(y, y_bsl, zscore, zscoretype,
                      convolve, gsd, method,
                      bsl_subtract, process_y)

    return x, y, y_p, y_p_var


def get_processed_BTN_matrix(b, w,
                    trains=None, events=None, BT_matrices=None,
                    convolve=False, sd=2, kernel='gaussian_causal',
                    return_poisson=False, remove_empty_trials=True, return_trials_mask=False):
    """
    Returns a matrix M of shape (B,T,N) = (n bins, n trials, n neurons).

    If desired, alongside its random Poisson counterpart (return_poisson = True).
    """

    M = get_BTN_matrix(b, w, trains, events, BT_matrices)
    B, T, N = M.shape

    if remove_empty_trials:
        M, all_active_mask, active_masks = filter_allneurons_active(M, p=0, return_masks=True)
        print(f"{N} Neurons commonly active on {all_active_mask.sum()}/{T} trials.")

    if return_poisson:
        Mo = get_poisson_BTN_matrix(M)

    if convolve:
        sd=int(sd/b) # convert from ms to bins
        sd=max(1,sd) # cap to 1
        for n in range(N):
            M[:,:,n] = smooth(M[:,:,n], kernel, sd=sd, axis=0)
        if return_poisson:
            for n in range(N):
                Mo[:,:,n] = smooth(Mo[:,:,n], kernel, sd=sd, axis=0)
    ret = [M]
    if return_poisson:
        ret.append(Mo)
    if return_trials_mask:
        if not remove_empty_trials:
            all_active_mask = np.ones(T).astype(bool)
        ret.append(all_active_mask)

    return tuple(ret)


def get_BTN_matrix(b, w, trains=None, events=None, BT_matrices=None):
    """
    B: n bins, T: n trials, N: n neurons
    - trains: list of N arrays, in seconds
    - events: (T,) array, in seconds
    """

    assert (trains is not None and events is not None)|(BT_matrices is not None),\
        "You must provide either trains and events or BT_matrices."

    if BT_matrices is None:
        BT_matrices = []
        for t in trains:
            x, y, y_p, y_p_var = get_processed_ifr(t, events, b, w)
            BT_matrices.append((y.T*b/1000).astype(np.int32))

    B, T = BT_matrices[0].shape
    N = len(BT_matrices)
    assert B == len(np.arange(w[0], w[1], b)), \
        "WARNING mismatch between provided BT_matrice and expected bin and window size."

    M = np.zeros((B, T, N))
    for i,y in enumerate(BT_matrices):
        M[:,:,i] = y
    
    return M


def get_poisson_BTN_matrix(M):

    B, T, N = M.shape

    # subselect trials with spikes to make finer estimation
    M_active, all_active_mask, active_masks = filter_allneurons_active(M, p=0, return_masks=True)

    Mo = M.copy()
    for n in range(N):
        active_mask = active_masks[:,n]
        lam = np.mean(M[:, active_mask, n], 1)
        P_lam = np.random.poisson(np.tile(lam[None, :], (active_mask.sum(), 1)))
        Mo[:,active_mask,n] = P_lam.T

    return Mo


def filter_allneurons_active(M, p=0, return_masks=False):
    """
    - M: BxTxN matrix
    - p: [0-1], proportion of mean firing rate above which a trial must be 
                for the neuron to be considered active. Default 0 (there must be a tleast 1 spike in the trial)."""
    
    B, T, N = M.shape

    active_masks = np.zeros((T,N)).astype(np.bool)
    for n in range(N):
        mean_T = np.mean(M[:,:,n], axis=0)
        non0_mask = mean_T > 0
        mean_firing_rate = np.mean(M[:,non0_mask,n].ravel())
        active_mask = np.mean(M[:,:,n], axis=0) > p*mean_firing_rate
        active_masks[:,n] = active_mask.astype(np.bool)
    
    all_active_mask = np.logical_and.reduce(active_masks, axis=1)

    if return_masks:
        return M[:,all_active_mask,:], all_active_mask, active_masks

    return M[:,all_active_mask,:]


def get_processed_popsync(trains, events, psthb=10, window=[-1000,1000],
                          events_tiling_frac=0.1, sync_win=2, fs=30000, t_end=None,
                          b=1, sd=1000, th=0.02,
                          again=False, dp=None, U=None,
                          zscore=False, zscoretype='within',
                          convolve=False, gsd=1, method='gaussian',
                          bsl_subtract=False, bsl_window=[-4000, 0], process_y=False):
    '''
    Parameters:
        - trains: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - psthb: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
        - convolve: boolean, set to True to convolve the aligned binned train with a half-gaussian window to smooth the ifr
        - gsd: float, gaussian window standard deviation in ms
        - method: convolution window shape: gaussian, gaussian_causal, gamma | Default: gaussian
        - bsl_substract: whether to baseline substract the trace. Baseline is taken as the average of the baseline window bsl_window
        - bsl_window: [t1,t2], window on which the baseline is computed, in ms -> used for zscore and for baseline subtraction (i.e. zscoring without dividing by standard deviation)
        - process_y: whether to also process the raw trials x bins matrix y (returned raw by default)
    Returns:
        - x: 1D array tiling bins, in milliseconds
        - y: 2D array NtrialsxNbins, the unprocessed ifr (by default - can be processed if process_y is set to True)
        - y_mn
        - y_p
        - y_p_sem
    '''

    # Window and bins translation
    x = np.arange(window[0], window[1], psthb)
    y = psth_fraction_pop_sync(trains, events, psthb, window,
                                events_tiling_frac, sync_win, fs, t_end,
                                b, sd, th, again, dp, U)
    assert not np.any(np.isnan(y.ravel())), 'WARNING nans found in aligned ifr!!'
    if x.shape[0]>y.shape[1]:
        x=x[:-1]
    assert x.shape[0]==y.shape[1]

    y_bsl = psth_fraction_pop_sync(trains, events, psthb, bsl_window,
                                events_tiling_frac, sync_win, fs, t_end,
                                b, sd, th, again, dp, U)

    y, y_p, y_p_var = process_2d_trials_array(y, y_bsl, zscore, zscoretype,
                      convolve, gsd, method,
                      bsl_subtract, process_y)

    return x, y, y_p, y_p_var

def psth_fraction_pop_sync(trains, events, psthb, psthw, events_tiling_frac=0.1, sync_win=2, fs=30000, t_end=None, b=1, sd=1000, th=0.02, again=False, dp=None, U=None):
    '''
      Computes the population synchrony for a set of events.
        For instance, with pstw=[-100,100], psthb=10 and events_tiling_frac=0.1,
        the fraction of population synchrony will be computed for all time stamps between [-100,100] every 1ms.
    - trains: list of np arrays in samples, timeseries of which fraction_pop_sync will be computed
    - events: np array in samples, events around which fraction_pop_sync will be averaged
      BOTH MUST BE INTEGERS IN SAMPLES
    - psthb: float in ms, binning of psth
    - psthw: list of floats [t1,t2] in ms, window of psth
    - events_tiling_frac: fraction [0-1] of psth bins used to tile the windows around events with time stamps.
    - fs: float in Hz, t1 and trains sampling frequency
    - t_end: int in samples, end of recording of t1 and trains, in samples
    - sync_win: float in ms, synchrony window to define synchrony
    - b: int in ms, binsize defining the binning of timestamps to define 'broad firing periods' (see npyx.spk_t.firing_periods)
    - sd: int in ms, gaussian window sd to convolve the binned timestamps defining 'broad firing periods' (see npyx.spk_t.firing_periods)
    - th: float [0-1], threshold defining the fraction of mean firing rate reached in the 'broad firing periods' (see npyx.spk_t.firing_periods)
    - again: bool, whether to recompute the firing periods of units in U (trains)
    - dp: string, datapath to dataset with units corresponding to trains - optional, to ensure fast loading of firing_periods
    - U: list, units matching trains
    '''
    assert assert_int(events[0]), 'events must be provided in samples!'
    for t in trains:
        assert assert_int(t[0]), 'trains must be provided in samples!'
    assert sync_win>=psthb*events_tiling_frac, 'you are not tiling time in a meaningful way - \
        use a bigger sync window, a smaller psthb or a smaller events_tiling_frac'
    events_tiling_frac=1./int(1/events_tiling_frac) # ensures that downsampling is possible later
    eventiles=(np.arange(psthw[0], psthw[1]+psthb*events_tiling_frac, psthb*events_tiling_frac)*fs/1000).astype(np.int64)
    peri_event_stamps=np.concatenate([events+dt for dt in eventiles])

    # only consider spikes around events
    for ti, t in enumerate(trains.copy()):
        print(f'pre_masking: {len(trains[ti])} spikes.')
        t_mask = (t*0).astype(bool)
        for e in events:
            t_mask = t_mask | ( (t>=e+(psthw[0]*fs/1000)) & (t<=e+(psthw[1]*fs/1000)) )
        trains[ti] = t[t_mask]
        print(f'post_masking: {len(trains[ti])} spikes.')
    fps = frac_pop_sync(peri_event_stamps, trains, fs, t_end, sync_win, b, sd, th, again, dp, U)

    # Now reshape the pop synchrony trial-wise and
    # downsample it (rolling average + downsampling) from psthb*events_tiling_frac to psthb resolution
    n=int(1./events_tiling_frac) # n is the space between downsampled points.
    y_popsync = fps.reshape((len(events), len(eventiles)))
    window = (1.0 / n) * np.ones(n,)
    #y_popsync = np.convolve2d(y_popsync, window, mode='valid')[:,::n]
    y_popsync = np.apply_along_axis(lambda m:np.convolve(m, window, mode='valid'), axis=1, arr=y_popsync)[:,::n]

    assert not np.any(np.isnan(y_popsync.ravel())), 'WARNING nans found in aligned ifr!!'

    return y_popsync

#%% Process video data

def decode_rotary(A,B, fs=5000, n_ticks=1024, diam=200, gsd=25, med_filt=True):
    '''Function to decode velocity from rotary encoder channels.
    - a: np array, analog recording of channel A at sampling frequency fs
    - b: np array, analog recording of channel B at sampling frequency fs
    - fs: float (Hz), sampling frequency
    - n_ticks: int, number of ticks (periods) on rotary encoder (not number of thresholds (4x more))
    - diam: float (mm), outer diameter of wheel coupled to the encoder
    - gsd: float (ms), std of gaussian kernel (mandatory gaussian-causal smoothing)
    - med_filt: bool, whether to median filter on top of mandatory gaussian smoothing
    '''

    ## Compute channels on/offsets
    ath=A.min()+(A.max()-A.min())*0.2
    bth=B.min()+(B.max()-B.min())*0.2
    a=(A>ath).astype(np.int8) # not using thresh_fast as need the bool array later
    b=(B>bth).astype(np.int8)
    da=np.diff(a)
    db=np.diff(b)
    a_on=np.nonzero(da==1)[0]
    a_of=np.nonzero(da==-1)[0]
    b_on=np.nonzero(db==1)[0]
    b_of=np.nonzero(db==-1)[0]

    ## Compute array d: delta in rotary ticks
    # This has the size of record_length, and will be filled with
    # > -1/1 where A or B thresholds were crossed
    # > 0 everywhere else (init. with 0s)
    d=np.zeros((a.shape))

    # If only one channel was recorded, everything isn't lost.
    a_mess=f'WARNING half max of rotary channel A is {round(ath, 3)} -> channel must be dead. Skipping rotary decoding.\n'
    b_mess=f'WARNING half max of rotary channel B is {round(bth, 3)} -> channel must be dead. Skipping rotary decoding.\n'
    if (ath<0.2)&(bth<0.2):
        print(a_mess)
        print(b_mess)

        return npa([np.nan])

    elif (ath<0.2)|(bth<0.2):
        if (bth>0.2):
            print(a_mess)
            print('Only channel B used -> only absolute speed with a 1/2 period precision.\n')
            d[b_on]=1
            d[b_of]=1
        if (ath>0.2):
            print(b_mess)
            print('Only channel A used -> only absolute speed with a 1/2 period precision.\n')
            d[a_on]=1
            d[a_of]=1

        # *2 because 2 threshold crosses per period (Aup,Adown OR Bup,Bdown)
        n_ticks*=2

    elif (ath>0.2)&(bth>0.2):
        # Arbitrary decision:
        # if a is crossed up, and b is high, displacement is 1.
        # everything else necessarily follows.
        for aon in a_on:
            if not b[aon]:d[aon]=1 # if a up and b is up, 1.
            else:d[aon]=-1 # if a up and b is down, -1.
        for aof in a_of:
            if b[aof]:d[aof]=1 # if a down and b is up, -1.
            else:d[aof]=-1 # if a down and b is down, 1.

        for bon in b_on:
            if a[bon]:d[bon]=1 # if b up and a is up, -1.
            else:d[bon]=-1 # if b up and a is down, 1.
        for bof in b_of:
            if not a[bof]:d[bof]=1 # if b down and a is down, -1.
            else:d[bof]=-1 # if b down and a is up, 1.


        # *4 because 4 threshold crosses per period (Aup,Bup,Adown,Bdown)
        n_ticks*=4

    ## Convert array of delta-ticks to mm/s
    # periphery/n_ticks to get mm per tick
    mm_per_tick=np.pi*diam/n_ticks
    # delta rotary ticks to delta mm
    d*=mm_per_tick
    # delta mm to mm/s using sampling rate
    d*=fs

    ## mandatory smooth (it makes no sense to keep an instantaneous speed at resolution fs)
    gsd=int(gsd*fs/1000) # convert from ms to array sampling (1ms -> 5 samples)
    d=smooth(d, method='gaussian_causal', sd=gsd)
    if med_filt:
        msd=int(25*fs/1000)
        msd=msd+(msd%2)-1 # gotta be odd
        d=sp.ndimage.median_filter(d, msd)

    print('\nRotary data decoded.\n')

    return d

def get_nframes(video_path):
    cap = cv2.VideoCapture(video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def frameid_vidpath(frameid, nframes, videos):
    '''Return relative frame id and respective video
    from absolute frame index'''
    assert len(nframes)==len(videos)
    cumfcount=0
    for vid_i in range(len(videos)):
        rel_id=frameid-cumfcount
        vidpath=videos[vid_i]
        cumfcount+=nframes[vid_i]
        if (cumfcount-1)>=frameid:
            return rel_id, vidpath

def frame_from_vid(video_path, frame_i, plot=True):
    cap = cv2.VideoCapture(video_path)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert 0<=frame_i<totalFrames, 'Frame index too high!'
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_i)
    ret, frame = cap.read()
    if plot: plt.imshow(frame)
    return frame

#%% Rig-related

def ellipsis(a, b, x0=0, y0=0, rot=0):
    '''
    - a, b: floats, length of horizontal/vertical axis (for rot=0), respectively
    - x0, y0: floats, (x,y) coordinates of origin
    - rot: float (degrees), ellipsis clockwise rotation
    '''
    rot*=2*pi/360
    t=np.linspace(0, 2*pi, 100)

    ell = npa([a * cos(t), b * sin(t)])
    rotM = np.array([[cos(rot) , -sin(rot)],[sin(rot) , cos(rot)]])
    ellrot = np.zeros((2,ell.shape[1]))
    for i in range(ell.shape[1]):
        ellrot[:,i] = np.dot(rotM,ell[:,i])
    ellrot[0,:]=ellrot[0,:]+x0
    ellrot[1,:]=ellrot[1,:]+y0

    return ellrot

def in_ellipsis(X, Y, a, b, x0=0, y0=0, rot=0, a_axis='major', plot=False):
    f'''
    - X, Y: 1dim np arrays or shape (n,), coordinates to test
    {ellipsis.__doc__}

    returns: m, boolean array of shape (n, ), True for points (X,Y) within ellipsis.
    '''

    assert len(X)==len(Y)
    if a_axis=='minor': X,Y=Y,X

    rot*=-2*pi/360

    m=( ( ((X-x0)*cos(rot)+(Y-y0)*sin(rot))**2 / a**2 ) + ( ((X-x0)*sin(rot)+(Y-y0)*cos(rot))**2 / b**2 ) )-1

    return (m<0)

def ellipsis_string(x, a, b, axis='major'):
    '''
    - x: float, x (or y) coordinate of string on axis 'axis' (mm)
    - a: float, minor axis of ellipsis (mm)
    - b: float, major axis of ellipsis (mm)
    - axis: string, whether x is along the major or minor axis - default major.
    '''

    return a*np.sqrt(1-x**2/b**2) if axis=='major' else b*np.sqrt(1-x**2/a**2)



def draw_wheel_mirror(string=None, depth=None, theta=45, r=95, H=75, plot=True, saveFig=False, saveDir=None):
    '''Homologous to a cylindrical wedge (plane crossing <1 basis of cylinder).
    - string: float, desired lateral coverage below the mouse
    - depth: float, desired depth of mirror below the mouse (alternative to string)
    - theta: float, angle of mirror (degrees) - default 45 degrees
    - r: float, radius of the wheel (mm) - default 97 (mm)
    - H: float, width of the wheel (mm) - default 75
    - plot: bool, whether to plot ellipse or not
    '''

    assert (depth is not None) or (string is not None), 'You need to provide either depth or string.'
    assert not (depth is not None) and (string is not None), 'You can only provide either depth or string - not both.'
    if depth is None:
        assert 0<string<2*r
        h=np.sqrt(r**2-(string/2)**2)
        depth=r-h
    elif string is None:
        assert 0<depth<r
        h=r-depth
        string=2*np.sqrt(r**2-h**2)
    print(f'Border of mirror will be {round(depth, 1)}mm below the mouse, covering {round(string, 1)}mm laterally.')

    # Distances parallel to major axis inside cylinder
    theta=theta*2*pi/360
    e1=h/np.cos(theta)
    E=np.sqrt(2*H**2)
    e2=E-e1

    # Vertical distances
    dH=np.tan(theta)*(h+r)

    # Ellipse axis
    a=r # minor axis
    b=np.sqrt(dH**2+(h+r)**2)-e1
    print(f'Ellipsis minor axis: {round(a, 1)}mm, major axis:{round(b, 1)}mm.')

    # Distances parallel to minor axis inside cylinder
    y1=ellipsis_string(e1, a, b, axis='major')
    assert round(y1)==round(string/2)
    y2=ellipsis_string(e2, a, b, axis='major')
    print(f'Ellipsis strings: {round(2*y1, 1)}mm on one side ({round(e1, 1)}mm away from center), \
    {round(2*y2, 1)}mm on the other ({round(e2, 1)}mm away from center).')

    # Plot ellipse to real-world scale
    if plot:
        figure_width = 2*b/10 # cm
        figure_height = 2*a/10 # cm
        left_right_margin = 0 # cm
        top_bottom_margin = 0 # cm

        left   = left_right_margin / figure_width # Percentage from height
        bottom = top_bottom_margin / figure_height # Percentage from height
        width  = 1 - left*2
        height = 1 - bottom*2
        cm2inch = 1/2.54 # inch per cm

        # specifying the width and the height of the box in inches
        fig = plt.figure(figsize=(figure_width*cm2inch,figure_height*cm2inch))
        ax = fig.add_axes((left, bottom, width, height))

        # limits settings (important)
        plt.xlim(-(figure_width * width)/2, (figure_width * width)/2)
        plt.ylim(-(figure_height * height)/2, (figure_height * height)/2)

        # Ticks settings
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

        # Grid settings
        for spi in ['top', 'right', 'left', 'bottom']: ax.spines[spi].set_visible(False)
        ax.grid(color="gray", which="both", linestyle=':', linewidth=0.5)

        # your Plot (consider above limits)
        ellipse = Ellipse((0, 0), 2*b/10, 2*a/10, angle=0, fill=False, ec='k', lw=2, ls='--')
        ax.add_artist(ellipse)
        ax.plot([-e1/10, -e1/10], [-y1/10, y1/10], c='k', lw=2, ls='--')
        ax.plot([e2/10, e2/10], [-y2/10, y2/10], c='k', lw=2, ls='--')
        ax.scatter([0],[0], c='k', lw=2, s=500, marker='+', zorder=100)

        # save figure ( printing png file had better resolution, pdf was lighter and better on screen)
        if saveFig:
            saveDir=Path.home() if saveDir is None else Path(saveDir)
            assert saveDir.exists()
            fig.savefig(saveDir/f'mirror_string{round(string, 1)}_depth{round(depth, 1)}_a{round(a, 1)}_b{round(b, 1)}.png', dpi=100)
            fig.savefig(saveDir/f'mirror_string{round(string, 1)}_depth{round(depth, 1)}_a{round(a, 1)}_b{round(b, 1)}.pdf')


#%% archive

# def dat_to_dic(dp, variables='all'):
#     '''DEPRECATED loads matlab-exported np arrays...'''

#     # Import variables from matlab-exported np arrays
#     npy_dp = dp+'/exported_syncdat_npy'
#     if not os.path.isdir(npy_dp):
#         print('WARNING triggers have not been exported in {}. Exitting now.'.format(npy_dp))
#         return
#     allVariables=['piezo_lick', 'real_reward', 'buzz_cue', 'ghost_reward', 'trial_on']
#     if type(variables)==list:
#             variables=np.array(variables)
#             areIn = np.isin(variables, allVariables)
#             if not np.all(areIn):
#                 print('WARNING: {} is not in the list of accepted variables {}. Exiting now.'.format(variables[~areIn], allVariables))
#                 return
#     else:
#         variables = allVariables

#     rawGLXVariables = {}
#     for v in variables:
#         fn=npy_dp+'/'+str(v)
#         if not os.path.isfile(fn+'_on.npy'):
#             print('WARNING triggers have not been exported to {} via MATLAB. Exitting now.'.format(npy_dp))
#             #return
#         rawGLXVariables[v+'_on']=np.load(fn+'_on.npy').flatten()
#         rawGLXVariables[v+'_off']=np.load(fn+'_off.npy').flatten()

#     return rawGLXVariables

# def mat_to_trialsdf(dp, f_behav=None, trial_on_i=2, reward_i=5, cue_i=4,
#                     include_wheel_data=False, wheel_position_index=13, wheel_velocity_index=14, object_position_index=9):
#     '''
#     DEPRECATED only orks for neuropixels 3A recordings with trialON, reward and cue recorded directly on sync channel
#     Loads the trials-related information from the matlab data structure outputted by Virmen.
#     Parameters:
#         - dp: string, datapath to Neuropixels binary file (preferably LFP, faster)
#         - f_behav: string, path to the Virmen matlab structure
#         - trial_on_i: neuropixels sync channel index of trial onsets
#         - reward_i: neuropixels sync channel index of rewards onsets
#         - cue_i: neuropixels sync channel index of auditory cue onsets
#     Returns:
#         - trials_df: pandas dataframe with Ntrials rows, and columns.
#     '''
#     dp=Path(dp)
#     if f_behav is None:
#         files=list_files(dp/'behavior', 'mat')
#         assert len(files)>0, f"WARNING no files with extension 'mat' were found at {dp/'behavior'} - either add one there or explicitely declare a file path with f_behav parameter."
#         assert len(files)==1, f"WARNING more than 1 file with extension 'mat' were found at '{dp/'behavior'}' - clean up your directory structure and try again."
#         f_behav=dp/'behavior'/files[0]
#         print(f'Behavioural data loaded from: {f_behav}')

#     with h5py.File(f_behav, 'r') as f:
#         trials_dic={}
#         for col in ['trialnum', 'trialside', 'correct', 'reward', 'ghost']:
#             index=f['datastruct']['paqdata']['behav']['trials'][col]
#             trials_dic[col]=[]
#             for i in range(len(index)):
#                 val=f[index[i,0]][:][0]
#                 try:
#                     val=val[0]
#                 except:
#                     assert val==0, 'WARNING this thing {} was expected to be either a np array or 0 but is not - figure out what the h5py decoding generated.'.format(val)
#                     val=np.nan# looks like '[]' in matlab
#                 trials_dic[col].append(val)#[0,0] for i in range(len(index))]

#     # Load Neuropixels sync channel, in seconds
#     ons, ofs = get_npix_sync(dp, output_binary = False, sourcefile='ap', unit='samples') # sync channels onsets and offsets
#     npix_trialOns=ons[trial_on_i]
#     npix_trialOfs=ofs[trial_on_i]
#     npix_rewards=ons[reward_i]
#     npix_tone_cues=ons[cue_i]

#     # Make trials dataframe
#     df=pd.DataFrame(data=trials_dic)

#     # Add trial types
#     df.insert(0, 'trial_type', ['wheel_turn' for i in range(len(df.index))])

#     # 1 is left, -1 is right
#     for tr in df.index:
#         df.loc[tr, 'trialside'] = 'left' if (df.loc[tr, 'trialside']==1) else 'right'

#     # Add trial onsets and offsets in neuropixels time frame
#     assert len(npix_trialOns)==len(df.index)
#     df['npix_trialOns']=npix_trialOns
#     assert len(npix_trialOfs)==len(df.index)
#     df['npix_trialOfs']=npix_trialOfs

#     # Add wheel movement
#     if include_wheel_data:
#         df["object_position"] = np.nan;df["object_position"]=df["object_position"].astype(object)
#         df["wheel_position"] = np.nan;df["wheel_position"]=df["wheel_position"].astype(object)
#         df["wheel_velocity"] = np.nan;df["wheel_velocity"]=df["wheel_velocity"].astype(object)
#         with h5py.File(f_behav, 'r') as f:
#             for tr in df.index:
#                 index=f['datastruct']['paqdata']['behav']['trials']['data']
#                 obj=f[index[0,0]][:][object_position_index]
#                 wheel_p=f[index[0,0]][:][wheel_position_index]
#                 wheel_v=f[index[0,0]][:][wheel_velocity_index]
#                 df.at[tr, 'object_position'] = obj
#                 df.at[tr, 'wheel_position'] = wheel_p
#                 df.at[tr, 'wheel_velocity'] = wheel_v
#                 ##TODO check indices of actual wheel movements
#                 # if npix_trialOns is not None:
#                 #     df.at[tr, 'npix_movOns'] = get_weel_movement_onset(wheel_velocity=wheel_v, npix_trialOn=df.at[tr, 'npix_trialOns'], paq_fs=5000)

#     # Append random rewards, cued or not, at the end of the dataframe
#     rews=[]
#     for tr in df.index:
#         of=df.loc[tr, 'npix_trialOfs']
#         rew_mask=(npix_rewards>of)&(npix_rewards<(of+1)) # happens about 400ms after trial offset
#         if np.any(rew_mask):
#             assert df.loc[tr, 'reward']==1, 'WARNING seems like a reward onset was found during a trial considered Incorrect...'
#             rews.append(npix_rewards[rew_mask][0])
#         else:
#             rews.append(np.nan)
#     df['npix_rewards']=rews

#     # Now find random rewards
#     if npix_tone_cues is not None:
#         random_rewards=npix_rewards[~np.isin(npix_rewards, df['npix_rewards'])]
#         for r in random_rewards:
#             cue_mask=(npix_tone_cues>r-1)&(npix_tone_cues<r) # cues happen about 500ms before reward
#             i=df.index[-1]+1
#             if np.any(cue_mask):
#                 df.loc[i, 'trial_type']='cued_reward'
#                 df.loc[i, 'npix_rewards']=r
#                 df.loc[i, 'npix_tone_cues']=npix_tone_cues[cue_mask]
#             else:
#                 df.loc[i, 'trial_type']='random_reward'
#                 df.loc[i, 'npix_rewards']=r
#     return df

# def extract_licks(dp, source='PAQ'):
#     if source=='PAQ':
#         lick_var = 'LICKS_Piezo'
#         lick_var_on, lick_var_off = lick_var+'_ON', lick_var+'_OFF'
#         licksDic = load_PAQdata(dp, variables=[lick_var])
#     elif source=='GLX':
#         lick_var = 'piezo_lick'
#         lick_var_on, lick_var_off = lick_var+'_on', lick_var+'_off'
#         licksDic = import_GLXdata(dp, variables=[lick_var])
#     else:
#         print('WARNING source must be either PAQ or GLX. Exitting now.')
#         return
#     on, off = licksDic[lick_var_on], licksDic[lick_var_off] # seconds
#     min_interlickinterval = 70e-3 # seconds
#     ON = on[(len(on)-1-np.nonzero(np.diff(abs(on-on[-1])[::-1])>min_interlickinterval)[0])[::-1]]
#     OFF = off[np.nonzero(np.diff(off)>min_interlickinterval)[0]]
#     ON, OFF = np.append(on[0], ON), np.append(OFF, off[-1])
#     licksDic[lick_var_on], licksDic[lick_var_off] = ON, OFF
#     return licksDic

# def licks_hist(dp, source='PAQ'):
#     '''Function plotting the distribution of the interlick intervals.
#        Source should be either 'PAQ' or 'GLX'.'''
#     licksDic = extract_licks(dp, source) # in seconds
#     lick_var_on='LICKS_Piezo_ON' if source=='PAQ' else 'piezo_lick_on'
#     fig = hist_MB(np.diff(licksDic[lick_var_on])*1000, 0, 250, 2, title='Interlicks intervals distribution.', xlabel='InterLicks intervals (ms)')
#     return fig

# def extract_wheel(dp):
#     fig, ax = plt.subplots()
#     wheelDic = load_PAQdata(dp, variables=['TRIALON', 'ROT', 'ROTreal'])
#     #TODO Clip +/- 4 seconds around trial onsets and offsets
#     #TODO Define wheel onset for correct trials only, offsets = trials offsets
#     return wheelDic

# #%% Format spike trains: clip (-4s, +4s) around alignement event

# # Either directly from the matlab file, alrady done
# def get_npy_export(unit, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports'):
#     assert type(unit)==int
#     assert alignement_event in ['movon' ,'off']
#     assert start_format in ['ifr', 'bst', 'meanifr']
#     arr=np.load('{}/{}_{}_{}_cor.npy'.format(dp, unit, start_format, alignement_event))
#     return arr

# # Or clip it here
# def align_unit(dp, u, triggers, b=2, window=[-1000,1000], rem_emptyTrials=False):
#     ''' b: binarized train bin in millisecond
#         window is in milliseconds
#         triggers in seconds
#         u is int or list of ints -> population rate!'''
#     if type(u)==int:
#         t = trn(dp, u) # in samples (fs: 30000Hz)
#     elif type(u)==list:
#         t = npa(empty=(0))
#         for unt in u:
#             t = np.append(t, trn(dp, unt))
#         t = np.sort(t)
#     aligned_t = []
#     aligned_tb = np.zeros((len(triggers), int((window[1]-window[0])*1./b)))
#     for i, trg in enumerate(triggers):
#         ts = t-(trg*30000) # ts: t shifted
#         tsc = ts[(ts>=window[0]*30)&(ts<=window[1]*30)]*1./30 # tsc: ts clipped + conv in ms
#         if np.any(tsc) or not rem_emptyTrials:
#             aligned_t.append(tsc.tolist())
#             tscb = np.histogram(tsc, bins=np.arange(window[0],window[1]+b,b))[0] # tscb: tsc binned
#             aligned_tb[i,:] = tscb
#         else:
#             assert aligned_tb.dtype==float
#             aligned_tb[i,:] = np.nan
#     aligned_tb=aligned_tb[~np.isnan(aligned_tb).any(axis=1)]
#     return aligned_t, aligned_tb

# def align_licks(dp, triggers, b=2, window=[-1000,1000], source='GLX'):
#     ''' b: binarized train bin in millisecond
#         window is in milliseconds
#         triggers in seconds
#         Source should be either 'PAQ' or 'GLX'.'''
#     licksDic = extract_licks(dp, source) # in seconds
#     lick_var = 'LICKS_Piezo_ON' if source=='PAQ' else 'piezo_lick_on'
#     t = licksDic[lick_var]*1000 # in milliseconds
#     aligned_t = []
#     aligned_tb = np.zeros((len(triggers), int((window[1]-window[0])*1./b)))
#     for i, trg in enumerate(triggers*1000): # in milliseconds
#         ts = t-(trg) # ts: t shifted
#         tsc = ts[(ts>=window[0])&(ts<=window[1])] # tsc: ts clipped + conv in ms
#         aligned_t.append(tsc.tolist())
#         tscb = np.histogram(tsc, bins=np.arange(window[0],window[1]+b,b))[0] # tscb: tsc binned
#         aligned_tb[i,:] = tscb

#     return aligned_t, aligned_tb

# #%% Plot signle units (and licks) raster plots or PSTH


# def raster(dp, u, triggersnames, title='', window=[-1000,1000], show=True, licks_source = 'GLX'):

#     # Sanity check triggers
#     if type(u)!=list:
#         if u =='licks' and licks_source=='PAQ':
#             triggersDic = mk_PAQtriggersDic(dp)
#         elif type(u)==int or type(u)==float or (u =='licks' and licks_source=='GLX'):
#             triggersDic = mk_GLXtriggersDic(dp)
#         else:
#             print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
#             return
#     else:
#         for unt in u:
#             if type(unt)!=int:
#                 print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
#                 return
#         triggersDic = mk_GLXtriggersDic(dp)

#     trgnDic = {'RR':'random real reward onset', 'CR':'cued real reward onset',
#                'RF':'random fictive reward onset', 'CO':'cued omitted reward onset'}
#     if type(triggersnames)!=list: triggersnames = list(triggersnames)
#     try:
#         for trgn in triggersnames:
#             assert trgn in trgnDic.keys()
#     except:
#         print('WARNING the triggersname should be one of: {}. Exit now.'.format(trgnDic.keys()))
#         return
#     # plot
#     fig, axes = plt.subplots(len(triggersnames), figsize=(8,2.5*len(triggersnames)))
#     for ti, trg in enumerate(triggersnames):
#         ax=axes[ti] if len(triggersnames)>1 else axes

#         triggers = triggersDic[trg]
#         at, atb = align_unit(dp, u, triggers, window=window) if (type(u)==int or type(u)==list) else align_licks(dp, triggers, window=window, source=licks_source)
#         print('Number of licks/spikes:', len([item for sublist in at for item in sublist]))
#         for i, trial in enumerate(at):
#             ax.scatter(trial, i+1+np.zeros((len(trial))), color='black', s=2)
#         ax.plot([0,0], ax.get_ylim(), ls='--', lw=1, color='black')
#         if trg[0]=='C':
#             ax.plot([-500, -500], ax.get_ylim(), ls='--', lw=1, color='black')
#         ax.set_ylim([0, len(at)])
#         ax.invert_yaxis()
#         ax.set_ylabel('Trial')
#         ax.set_xlabel('Time from {} (ms)'.format(trgnDic[trg]))
#         ax.set_xlim(window[0], window[1])
#     fig.suptitle(title) if len(title)!=0 else fig.suptitle('Unit {}.'.format(u))
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     if not show:
#         plt.close(fig)
#     return fig
# #%%

# def get_ifr(dp, u, triggers, b=5, window=[-1000,1000], licks_source='GLX'):
#     '''
#     dp: string, path to dataset with kilosort/phy output.
#     u: integer, unit index or list of ints -> population rate!
#     triggers: list, time stamps to trigger alignement, in seconds.
#     b: float, bin to make the aligned binned train, in milliseconds.
#     window: [w1, w2], where w1 and w2 are in milliseconds.
#     convolve: boolean, set to True to convolve the aligned binned train with a gaussian window to smooth the ifr
#     gw: integer, gaussian window width, only used if convolve is True
#     gsd: float, gaussian window standard deviation, only used if convolve is True
#     '''
#     # Get aligned binned train
#     at, atb = align_unit(dp, u, triggers, b, window=window) if (type(u)==int or type(u)==list) else align_licks(dp, triggers, b, window=window, source=licks_source) if type(u)==str else print("WARNING u must be an int or 'licks'.")
#     # Make ifr
#     ifr = np.zeros((atb.shape[0], atb.shape[1]))
#     for i in range(atb.shape[0]):
#         ifr[i,:] = atb[i,:]/(b*1e-3)
#     ifr_mn = np.array([np.mean(ifr, axis=1), ]*ifr.shape[1]).transpose()
#     ifr_sd = np.array([np.std(ifr, axis=1), ]*ifr.shape[1]).transpose()

#     # Set 0 sd to 1 so that dividing does not change anything
#     for i in range(ifr_sd.shape[0]):
#         if np.all(ifr_sd[i,:]==0): ifr_sd[i,:]=1

#     return ifr, ifr_mn, ifr_sd

# def get_processed_ifr(dp, u, triggers, b=5, window=[-1000,1000], zscore=False, zscoretype='overall', convolve=False, gw=64, gsd=1, licks_source='GLX'):
#     '''u can be a list of units -> population rate!'''
#     ifr, ifr_mn, ifr_sd = get_ifr(dp, u, triggers, b, window, licks_source)

#     # Window and bins translation
#     maxWin=4000; minWin=-4000;
#     window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
#     x = np.arange(window[0], window[1], b)
#     y = ifr[:, int(ifr.shape[1]/2)+int(window[0]/b):int(ifr.shape[1]/2)+int(window[1]/b)+1]
#     if x.shape[0]>y.shape[1]:
#         x=x[:-1]
#     assert x.shape[0]==y.shape[1]

#     if zscore:
#         assert zscoretype in ['overall', 'trialwise']
#         if zscoretype=='overall':
#             y_mn=np.mean(ifr.flatten())
#             y_sd=np.std(ifr.flatten())
#             print('overall mean:{}, sd:{}'.format(y_mn, y_sd))
#         if zscoretype=='trialwise':
#             y_mn = ifr_mn[:, int(ifr.shape[1]/2)+int(window[0]/b)-1:int(ifr.shape[1]/2)+int(window[1]/b)+1]
#             y_sd = ifr_sd[:, int(ifr.shape[1]/2)+int(window[0]/b)-1:int(ifr.shape[1]/2)+int(window[1]/b)+1]
#             print('trialwise mean:{}, sd:{}'.format(y_mn[:,0], y_sd[:,0]))
#         y_p = (y-y_mn)/y_sd
#         y_p=np.mean(y_p, axis=0)
#         y_p_sem=stats.sem(y, axis=0)
#     else:
#         y_p = y_mn = np.mean(y, axis=0)
#         y_p_sem = stats.sem(y, axis=0)

#     if convolve:
#         gaussWin=sgnl.gaussian(gw, gsd)
#         gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
#         y_p = np.convolve(y_p, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]

#     return x, y, y_mn, y_p, y_p_sem

# def ifr_plot(dp, u, triggersnames, title='', b=5, window=[-1000,1000], color=seabornColorsDic[0],
#              zscore=False, plot_all_traces=False, zslines=False, zscoretype='overall',
#              convolve=True, error=True, show=True, ylim=None, licks_source = 'GLX', gw=64, gsd=1, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
#     '''Window has to be in milliseconds. b as well.

#     if u is a list of units, the population rate of this list will be computed.'''

#     # Sanity check triggers
#     if type(u)!=list:
#         if u =='licks' and licks_source=='PAQ':
#             triggersDic = mk_PAQtriggersDic(dp)
#         elif type(u)==int or type(u)==float or (u =='licks' and licks_source=='GLX'):
#             triggersDic = mk_GLXtriggersDic(dp)
#         else:
#             print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
#             return
#     else:
#         for unt in u:
#             if type(unt)!=int:
#                 print("WARNING u must be an int, float, 'licks' or list of ints. Exitting now.")
#                 return
#         triggersDic = mk_GLXtriggersDic(dp)

#     trgnDic = {'RR':'random real reward onset', 'CR':'cued real reward onset',
#                'RF':'random fictive reward onset', 'CO':'cued omitted reward onset'}
#     if type(triggersnames)!=list: triggersnames = list(triggersnames)
#     try:
#         for trgn in triggersnames:
#             assert trgn in trgnDic.keys()
#     except:
#         print('WARNING the triggersname should be one of: {}. Exit now.'.format(trgnDic.keys()))
#         return

#     # plot

#     if saveFig or saveData:
#         unit_n = str(u)+'_'+dp.split('/')[-1]
#         fig_n = 'IFRsingleUnit{}_'.format(unit_n)
#         Dir = saveDir+'/'+fig_n+str(triggersnames)
#         if not os.path.isdir(Dir): os.mkdir(Dir)
#     fig, axes = plt.subplots(len(triggersnames), figsize=(8,2.5*len(triggersnames)))
#     ylims=[]
#     for ti, trg in enumerate(triggersnames):
#         ax=axes[ti] if len(triggersnames)>1 else axes

#         triggers = triggersDic[trg]
#         x, y, y_mn, y_p, y_p_sem = get_processed_ifr(dp, u, triggers, b, window, zscore, zscoretype, convolve, gw, gsd, licks_source)
#         if saveData:
#             np.save(Dir+'/'+fig_n+'{}aligned_x.npy'.format(trg), x)
#             np.save(Dir+'/'+fig_n+'{}aligned_y.npy'.format(trg), y)
#             np.save(Dir+'/'+fig_n+'{}aligned_y_processed.npy'.format(trg), y_p)
#             np.save(Dir+'/'+fig_n+'{}aligned_y_p_sem.npy'.format(trg), y_p_sem)
#         if zscore:
#             if not convolve:
#                 if not error:
#                     ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
#                 else:
#                     ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
#                     ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0))
#                     ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.1, facecolor=color)
#                     ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
#                     ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
#             else:
#                 if plot_all_traces:
#                     for i in range(y.shape[0]):
#                         gaussWin=sgnl.gaussian(gw, gsd)
#                         gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
#                         trace = np.convolve(y[i,:], gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
#                         ax.plot(x, trace, lw=0.3, color=color, alpha=0.2)
#                 ax.plot(x, y_p, lw=1, color=color)
#                 if error:
#                     ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
#                     ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
#                     ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)

#             ax.plot([x[0], x[-1]], [0,0], ls="--", c=(0,0,0), lw=0.5)
#             if zslines:
#                 ax.plot([x[0], x[-1]], [1,1], ls="--", c=[1,0,0], lw=1)
#                 ax.plot([x[0], x[-1]], [2,2], ls="--", c=[1,0,0], lw=1)
#                 ax.plot([x[0], x[-1]], [3,3], ls="--", c=[1,0,0], lw=1)
#                 ax.plot([x[0], x[-1]], [-1,-1], ls="--", c=[0,0,1], lw=1)
#                 ax.plot([x[0], x[-1]], [-2,-2], ls="--", c=[0,0,1], lw=1)
#                 ax.plot([x[0], x[-1]], [-3,-3], ls="--", c=[0,0,1], lw=1)
#             ax.set_ylim([-1, 2])
#             ax.set_ylabel('Inst.F.R. (s.d.)')

#         elif not zscore:
#             if plot_all_traces:
#                 for i in range(y.shape[0]):
#                         ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
#             if not convolve:
#                 if not error:
#                     ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
#                 else:
#                     ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
#                     ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0), zorder=3)
#                     ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.2, facecolor=color)
#                     ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
#                     ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
#             else:
#                 ax.plot(x, y_p, lw=1, color=color, alpha=1)
#                 if error:
#                     ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
#                     ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
#                     ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)
#             yl=max(y_p+y_p_sem); ylims.append(int(yl)+5-(yl%5));
#             ax.set_ylabel('Inst.F.R. (Hz)')

#         ax.set_xlabel('Time from {} (ms).'.format(trgnDic[trg]))


#     AXES=axes if len(triggersnames)>1 else [axes]
#     for ax, trg in zip(AXES, triggersnames):
#         if not zscore:
#             ylim = max(ylims) if not ylim else ylim
#         ax.set_ylim([0, ylim])
#         ax.set_xlim(window[0], window[1])
#         ax.plot([0,0], ax.get_ylim(), ls='--', lw=1, color='black')
#         if trg[0]=='C':
#             ax.plot([-500, -500], ax.get_ylim(), ls='--', lw=1, color='black')
#     fig.suptitle(title) if len(title)!=0 else fig.suptitle('Unit {}.'.format(u))
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     if saveFig:
#         fig.savefig(Dir+'/'+fig_n+'{}aligned.pdf'.format(triggersnames))
#         fig.savefig(Dir+'/'+fig_n+'{}aligned.png'.format(triggersnames))

#     if not show:
#         plt.close(fig)

#     return fig


# def plot_CS_dataset(CS_MB021,CS_MB022,CS_MB023):
#     dps=['/media/ms047/DK_probes_backup/Conditioning/MB021/Kilosort50iters',
#          '/media/ms047/DK_probes_backup/Conditioning/MB022',
#          '/media/ms047/DK_probes_backup/Conditioning/MB023']
#     CSs=[CS_MB021,
#          CS_MB022,
#          CS_MB023]
#     trg_ls=[['RR','RF', 'CR', 'CO']]
#     for dp, CS in zip(dps, CSs):
#         if not os.path.isdir(dp+'/AlignedComplexSpikes'): os.mkdir(dp+'/AlignedComplexSpikes')
#         for cs in CS:
#             for trg_l in trg_ls:
#                 fig = ifr_plot(dp, cs, ['RR'],  b=10, window=[-750,750], color=seabornColorsDic[0], convolve=True, error=True, show=False, ylim=None)
#                 fig.savefig(dp+'/AlignedComplexSpikes/{}_{}_aligned.pdf'.format(cs, str(trg_l).replace(', ', ' x ').replace("'", '')))
#                 plt.close()

# def plot_CS_selected(selected_units):
#     DP = '/home/ms047/Dropbox/Science/PhD/Data_Presentation/Reward paper/MainFig/CS_patterns'
#     if not os.path.isdir(DP): os.mkdir(DP)
#     dps={'MB021':'/media/ms047/DK_probes_backup/Conditioning/MB021/Kilosort50iters',
#          'MB022':'/media/ms047/DK_probes_backup/Conditioning/MB022',
#          'MB023':'/media/ms047/DK_probes_backup/Conditioning/MB023'}
#     trg_l=['RR','RF', 'CR', 'CO']
#     for ap, ap_val in selected_units.items():
#         ylim = 10 if ap=='RR_minus' else None
#         if not os.path.isdir(DP+'/'+ap): os.mkdir(DP+'/'+ap)
#         for dataset, dataset_units in ap_val.items():
#             if not os.path.isdir(DP+'/'+ap+'/'+dataset): os.mkdir(DP+'/'+ap+'/'+dataset)
#             for cs in dataset_units:
#                 fig = ifr_plot(dps[dataset], cs, trg_l,  b=10, window=[-750,750], color=seabornColorsDic[0], convolve=True, error=True, show=False, ylim=ylim)
#                 fig.savefig(DP+'/'+ap+'/'+dataset+'/{}_{}_aligned.pdf'.format(cs, str(trg_l).replace(', ', ' x ').replace("'", '')))
#                 plt.close()

# #%% IFR population plots

# def make_ifr_matrix(dp, units, triggersname, b=5, window=[-1000,1000],
#                        zscore=True, zscoretype='overall', convolve=True, gw=64, gsd=1):
#     '''triggersname: one of the keys of GLXtriggersDic.'''
#     assert zscoretype in ['overall', 'trialwise']
#     if type(units)==int:
#         units=[units]
#     # Adjust window, get triggers
#     maxWin=4000; minWin=-4000;
#     window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
#     triggersDic = mk_GLXtriggersDic(dp)
#     triggers = triggersDic[triggersname]

#     # Populate matrix
#     x, y, y_mn, y_p, y_p_sem = get_processed_ifr(dp, units[0], triggers, b, window, zscore, zscoretype, convolve, gw, gsd)
#     ifr_matrix=np.zeros((len(units), len(x)))
#     for i, u in enumerate(units):
#         x, y, y_mn, y_p, y_p_sem = get_processed_ifr(dp, u, triggers, b, window, zscore, zscoretype, convolve, gw, gsd)
#         ifr_matrix[i, :] = y_p

#     return ifr_matrix, x



# def av_ifr_plot_acrossDP(DPs, unitsPerDataset, triggersname, title='', b=5, window=[-1000,1000], color=seabornColorsDic[0],
#              zscore=True, zscoretype='overall', plot_all_units=False, zslines=False,
#              convolve=True, error=True, show=True, ylim=None, gw=64, gsd=1, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):

#     for initDataset in DPs.keys():
#         dp, units = DPs[initDataset], unitsPerDataset[initDataset]
#         if len(units)>0: break
#     if len(units)==0:
#         return plt.figure() # empty figure if no unit at all across all datasets
#     ifr_matrix, x = make_ifr_matrix(dp, units, triggersname, b, window,
#                            zscore, zscoretype=zscoretype, convolve=False)
#     ifrs_matrix=np.zeros((1, len(x)))
#     totalUnits=0
#     for dataset in unitsPerDataset.keys():
#         dp = DPs[dataset]
#         units = unitsPerDataset[dataset]
#         if len(units)>0:
#             totalUnits+=len(units)
#             # DO NOT AVERAGE PRE-CONVOLVED TRACES!!
#             ifr_matrix, x1 = make_ifr_matrix(dp, units, triggersname, b, window,
#                                zscore, zscoretype=zscoretype, convolve=False)
#             ifrs_matrix=np.append(ifrs_matrix, ifr_matrix, axis=0) # vstack

#     y = ifrs_matrix
#     y_p, y_p_sem = np.mean(y, axis=0), stats.sem(y, axis=0) # Zscored or not, convolved or not from within make_ifr_matrix -> get_processed_ifr

#     # plot
#     fig, ax = plt.subplots(1, figsize=(8,2.5))
#     ylims=[]

#     if zscore:
#         if not convolve:
#             if not error:
#                 ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
#             else:
#                 ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
#                 ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0))
#                 ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.1, facecolor=color)
#                 ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
#                 ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
#         else:
#             # CONVOLUTION HAS TO BE DONE OUTSIDE OF get_processed_ifr
#             # BECAUSE IT HAS TO BE DONE AFTER AVERAGING ACROSS DATASETS
#             gaussWin=sgnl.gaussian(gw, gsd)
#             print(gsd)
#             gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
#             y_p = np.convolve(y_p, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
#             if plot_all_units:
#                 for i, yi in enumerate(y):
#                     y[i,:] = np.convolve(yi, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
#                 for i in range(y.shape[0]):
#                     ax.plot(x, y[i,:], lw=0.5, color=color, alpha=0.8)
#             ax.plot(x, y_p, lw=1, color=color)
#             if error:
#                 ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
#                 ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
#                 ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)

#         ax.plot([x[0], x[-1]], [0,0], ls="--", c=(0,0,0), lw=0.5)
#         if zslines:
#             ax.plot([x[0], x[-1]], [1,1], ls="--", c=[1,0,0], lw=1)
#             ax.plot([x[0], x[-1]], [2,2], ls="--", c=[1,0,0], lw=1)
#             ax.plot([x[0], x[-1]], [3,3], ls="--", c=[1,0,0], lw=1)
#             ax.plot([x[0], x[-1]], [-1,-1], ls="--", c=[0,0,1], lw=1)
#             ax.plot([x[0], x[-1]], [-2,-2], ls="--", c=[0,0,1], lw=1)
#             ax.plot([x[0], x[-1]], [-3,-3], ls="--", c=[0,0,1], lw=1)
#         ax.set_ylim([-1, 2])
#         ax.set_ylabel('Inst.F.R. (s.d.)')

#     elif not zscore:
#         if plot_all_units:
#             for i in range(y.shape[0]):
#                     ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
#         if not convolve:
#             if not error:
#                 ax.bar(x, y_p, width=b, color=color, edgecolor=color, linewidth=1)
#             else:
#                 ax.hlines(y_p, xmin=x, xmax=x+b, color='black', linewidth=1, zorder=12)
#                 ax.bar(x, y_p+y_p_sem, width=b, edgecolor=color, linewidth=1, align='edge', fc=(1,1,1,0), zorder=3)
#                 ax.fill_between(x=x, y1=y_p+y_p_sem, y2=y_p-y_p_sem, step='post', alpha=0.2, facecolor=color)
#                 ax.fill_between(x, y_p-y_p_sem, step='post', facecolor='white', zorder=8)
#                 ax.step(x, y_p-y_p_sem, color=color, where='post', linewidth=1, zorder=10)
#         else:
#             gaussWin=sgnl.gaussian(gw, gsd)
#             print(gsd)
#             gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
#             y_p = np.convolve(y_p, gaussWin, mode='full')[int(gw/2):-int(gw/2-1)]
#             ax.plot(x, y_p, lw=1, color=color, alpha=1)
#             if error:
#                 ax.fill_between(x, y_p-y_p_sem, y_p+y_p_sem, facecolor=color, interpolate=True, alpha=0.2)
#                 ax.plot(x, y_p-y_p_sem, lw=0.5, color=color)
#                 ax.plot(x, y_p+y_p_sem, lw=0.5, color=color)
#         yl=max(y_p+y_p_sem); ylims.append(int(yl)+5-(yl%5));
#         ax.set_ylabel('Inst.F.R. (Hz)')

#     ax.set_xlabel('Time from {} (ms).'.format(triggersname))
#     ax.set_title('{} (n={})'.format(title, totalUnits))
#     if not zscore:
#         ylim = max(ylims) if not ylim else ylim
#         ax.set_ylim([0, ylim])
#     ax.set_xlim(window[0], window[1])
#     ax.plot([0,0], ax.get_ylim(), ls='--', lw=1, color='black')
#     if triggersname[0]=='C':
#         ax.plot([-500, -500], ax.get_ylim(), ls='--', lw=1, color='black')
#     fig.tight_layout()

#     if saveFig or saveData:
#         fig_n = 'IFRpop_'
#         Dir = saveDir+'/'+fig_n+str(triggersname)+'aligned'
#         if not os.path.isdir(Dir): os.mkdir(Dir)
#         if saveData:
#             np.save(Dir+'/'+fig_n+'{}aligned_x.npy'.format(triggersname), x)
#             np.save(Dir+'/'+fig_n+'{}aligned_y.npy'.format(triggersname), y)
#             np.save(Dir+'/'+fig_n+'{}aligned_y_processed.npy'.format(triggersname), y_p)
#             np.save(Dir+'/'+fig_n+'{}aligned_y_p_sem.npy'.format(triggersname), y_p_sem)
#         if saveFig:
#             fig.savefig(Dir+'/'+fig_n+'{}aligned.pdf'.format(triggersname))
#     if not show:
#         plt.close(fig)

#     return fig


# def ifr_barplot_compWind_acrossDP(DPs, unitsPerDataset, triggersnames, winds, title='', b=5, window=[-750,750],
#          zscore=False, zscoretype='overall', show=False, ylim=15, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
#     '''winds format should be [[w1, w2], [w3,w4]... in ms]
#     Test: sum ranked wilcoxon from scipy.stats'''
#     for wind in winds:
#         assert wind[0]<=wind[1]
#     # Get concatenated of av IFR across datasets
#     totalUnits=[]
#     for dataset in unitsPerDataset.keys():
#         units = [dataset+'_'+str(u) for u in unitsPerDataset[dataset]]
#         totalUnits+=units
#     DF = pd.DataFrame(columns=["Unit", "Triggers"]+[str(i) for i in winds], index = np.arange(len(totalUnits*len(triggersnames))))
#     DF["Unit"]=totalUnits*len(triggersnames)
#     for trg_i, triggersname in enumerate(triggersnames):
#         for initDataset in DPs.keys():
#             dp, units = DPs[initDataset], unitsPerDataset[initDataset]
#             if len(units)>0: break
# #        if len(units)==0:
# #            return plt.figure() # empty figure if no unit at all across all datasets
#         ifr_matrix, x = make_ifr_matrix(dp, units, triggersname, b, window,
#                                zscore, zscoretype='overall', convolve=False, gw=64, gsd=1)
#         ifrs_matrix=np.zeros((0, len(x)))

#         for dataset in unitsPerDataset.keys():
#             dp = DPs[dataset]
#             units = unitsPerDataset[dataset]
#             if len(units)>0:
#                 ifr_matrix, x1 = make_ifr_matrix(dp, units, triggersname, b, window,
#                                    zscore, zscoretype='overall', convolve=False, gw=64, gsd=1)
#                 ifrs_matrix=np.append(ifrs_matrix, ifr_matrix, axis=0) # vstack
#         y = ifrs_matrix
#         i1=len(totalUnits)
#         i2 = len(totalUnits)*trg_i # jump to the next 'indices slice' of size totalUnits i.e. next trigger, all units again
#         DF.iloc[0+i2:i1+i2, DF.columns.get_loc("Triggers")]=triggersname
#         for wind in winds:
#             w1, w2 = int((wind[0]-window[0])/b), int((wind[1]-window[0])/b)
#             av_wind = np.mean(y[:, w1:w2], axis=1)
#             DF.iloc[0+i2:i1+i2, DF.columns.get_loc(str(wind))]=av_wind
#     # Reshape the pandas dataframe, convenient to then make the barplot with seaborn
#     DF = pd.melt(DF, id_vars=["Triggers", "Unit"], var_name="Window", value_name="Average IFR")

#     # Make paired t-test
#     pt_table = pd.DataFrame(columns=["Trigger 1", "Window 1", "Trigger 2", "Window 2", "Statistic", "Pval"])
#     i=0
#     for trg1 in triggersnames:
#         for win1 in winds:
#             for trg2 in triggersnames:
#                 for win2 in winds:
#                     if (trg1!=trg2 or win1!=win2):
#                         # Assert that the units match to allow you to do a paired test!!
#                         units1 = DF.loc[(DF["Triggers"]==trg1) & (DF["Window"]==str(win1))]["Unit"]
#                         units2 = DF.loc[(DF["Triggers"]==trg2) & (DF["Window"]==str(win2))]["Unit"]
#                         assert np.all(units1.values == units2.values)
#                         dist1 = DF.loc[(DF["Triggers"]==trg1) & (DF["Window"]==str(win1))]["Average IFR"]
#                         dist2 = DF.loc[(DF["Triggers"]==trg2) & (DF["Window"]==str(win2))]["Average IFR"]
#                         statistic, Pval = stats.wilcoxon(dist1, dist2) # non parametric paired test
#                         pt_table.loc[i, :]=[trg1, str(win1), trg2, str(win2), statistic, Pval]
#                         i+=1

#     # Plot barplot with seaborn
#     fig, ax = plt.subplots()
#     sns.barplot(x="Triggers", y="Average IFR", hue="Window", data=DF, order = triggersnames, hue_order=[str(i) for i in winds])#yerrs)
#     leg_handles, leg_labels = ax.get_legend_handles_labels()
#     sns.stripplot(x="Triggers", y="Average IFR", hue="Window", data=DF, order = triggersnames, hue_order=[str(i) for i in winds],
#                     size=6, jitter=False, dodge=True, color=(0.4, 0.4, 0.4), alpha=0.6, marker="D")
#     ax.legend(leg_handles, leg_labels, title='Window')
#     ax.set_title('Comp.of av. IFR \n windows {} for {} units \n of pattern {}.'.format(winds, len(totalUnits), title))
#     ax.set_ylim([0, ylim])
#     if saveFig or saveData:
#         fig_n = 'barplotMeanIFR_{}aligned_{}windows'.format(triggersnames, winds)
#         Dir = saveDir+'/'+fig_n
#         if not os.path.isdir(Dir): os.mkdir(Dir)
#         if saveData:
#             DF.to_csv(Dir+'/'+fig_n+'_values.csv')
#             pt_table.to_csv(Dir+'/'+fig_n+'_stats.csv')
#         if saveFig:
#             fig.savefig(Dir+'/'+fig_n+'.pdf')
#     if not show:
#         plt.close(fig)

#     return fig, pt_table

# ### Modify here - parameters
# DP = '/home/ms047/Dropbox/Science/PhD/Data_Presentation/Reward paper/MainFig/CS_pop'
# DPs={'MB021':'/media/ms047/DK_probes_backup/Conditioning/MB021/Kilosort50iters',
#  'MB022':'/media/ms047/DK_probes_backup/Conditioning/MB022',
#  'MB023':'/media/ms047/DK_probes_backup/Conditioning/MB023'}
# selected_units = {# Random Fictive error can be 1) similar 2) same time, higher amplitude 3) same amplitude, delayed, 4) sharper + delayed, but rarely two bumps...
#             'RRxRF2bumps_plus':{'MB021':[232,  286, 233, 229, 222, 221, 220, 219, 285],
#                           'MB022':[280, 277,  205, 216,  149], # Can tell the difference (RR response never like RF response)
#                           'MB023':[525, 243, 268, 206,  231, 195, 192, 91]},
#             'RRxRF2bumps_minus':{'MB021':[],
#                         'MB022':[169, 351, 209],
#                         'MB023':[225, 232,181]},
#             'RRxRFbiggerbump_plus':{'MB021':[230, 232, 233,  229, 221, 223],
#                           'MB022':[280, 277, 205, 153, 149], # Can tell the difference (RR response never like RF response)
#                           'MB023':[243, 358, 204]},
#             'RRxRFdelayedbump_plus':{'MB021':[232, 225, 230,  232, 233, 234, 226, 229, 222, 221, 223, 219],
#                           'MB022':[216, 153, 149, 190], # Can tell the difference (RR response never like RF response)
#                           'MB023':[ 207]},
#             'RRxRFall_plus':{'MB021':[232, 225, 230, 286, 232, 233, 234, 226, 229, 222, 221, 223, 220, 219, 285],
#                           'MB022':[280, 277, 216, 153, 149, 190], # Can tell the difference (RR response never like RF response)
#                           'MB023':[243, 268, 358, 206, 204, 231, 207, 195, 192, 91, 525]},
#             'RRxRFall_minus':{'MB021':[],
#                         'MB022':[169, 351, 209],
#                         'MB023':[225, 232,181]},
#             # Cued Real shift
#             'RRxCR_plus':{'MB021':[290, 295, 287, 289, 232, 234, 284, 231, 286, 233, 226, 229, 222, 221, 223, 220, 225, 230, 219, 321],
#                           'MB022':[872, 1078, 1063, 319, 874, 763, 783, 349, 280, 277, 266,  186, 205, 156, 216],
#                           'MB023':[249, 211, 206, 83, 293, 379,  268, 249, 206, 204, 199, 207, 209, 195, 168, 525, 550]},
#             'RRxCR_plusNew':{'MB021':[219, 220, 221, 222, 223, 225, 226, 229, 230, 231, 232, 233, 234,
#                                       284, 286, 287, 289, 290, 295, 321],
#                           'MB022':[ 156,  186,  205,  216,  266,  277,  280,  319,  349,  763,  783, 864, 872,  874, 1063, 1078],
#                           'MB023':[ 83, 91, 160, 166, 168, 179, 195, 199, 204, 206, 206, 207, 209, 211, 249, 284, 379, 525, 550, 552]},
#             'RRxRR_minusNew':{'MB021':[285],
#                            'MB022':[144],
#                            'MB023':[93, 551, 286]},#246, 286]},
#             # Cued Omission error
#             'CRxCOearly_plusNew':{'MB021':[225],
#                           'MB022':[186],
#                           'MB023':[337, 83, 550, 179, 91]}, #293or411, 179, 91]},
#             'CRxCOearly_minus':{'MB021':[],
#                            'MB022':[209, 169],
#                            'MB023':[95]},
#             'CRxCOlate_plus':{'MB021':[],
#                           'MB022':[190, 205],
#                           'MB023':[195]},
#             'CRxCOall_plus':{'MB021':[225],
#                           'MB022':[ 190, 186, 205],
#                           'MB023':[337, 83, 293, 195, 179, 91]},
#             'CRxCOall_minus':{'MB021':[],
#                            'MB022':[209, 169],
#                            'MB023':[95]},
#             'RRxRFlicking':{'MB021':['licks'],
#                            'MB022':['licks'],
#                            'MB023':['licks']}}
# patterns_compWindows = {# Random Fictive error can be 1) similar 2) same time, higher amplitude 3) same amplitude, delayed, 4) sharper + delayed, but rarely two bumps...
#             'RRxRF2bumps_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
#             'RRxRF2bumps_minus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
#             'RRxRFbiggerbump_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
#             'RRxRFdelayedbump_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
#             'RRxRFall_plus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
#             'RRxRFall_minus':[[-750, -500], [0,100], [100,200], [50, 100], [100, 150]],
#             # Cued Real shift
#             'RRxCR_plus':[[-750, -500], [0,100]],
#             'RRxRR_minus':[[-750, -500], [0, 100]],
#             # Cued Omission error
#             'CRxCOearly_plus':[[-750, -500], [-500,-450], [0, 200]],
#             'CRxCOearly_minus':[[-750, -500], [-500,-450], [0, 200]],
#             'CRxCOlate_plus':[[-750, -500], [-500,-450], [200, 250]],
#             'CRxCOall_plus':[[-750, -500], [-500,-450], [0, 200], [200, 250]],
#             'CRxCOall_minus':[[-750, -500], [-500,-450], [100, 200]],
#             'RRxRFlicking':[[-750, -500], [100,600]]}


# def plot_all_avIFRpatterns(DP, DPs, selected_units):
#     '''
#     selectedUnits has to be of the form:
#         {'ABxCD...':{'datasetName1':[u1,u2,...uN], 'dataset2Name':[u1,u2,...uN]...},{}...}

#         and the dictionnary DPs = {'dataset1Name':'dataset1Path', ...}'''
#     # Loop through patterns
#     for pattern, unitsPerDataset in selected_units.items():
#         trgs=[pattern[0:2], pattern[3:5]]
#         print('\nTriggers for pattern {}: {}'.format(pattern, trgs))
#         avPlotPath=DP+'/'+pattern
#         if not os.path.isdir(avPlotPath): os.mkdir(avPlotPath)
#         ttl='Av. IFR across units displaying pattern {}'.format(pattern)
#         for trg in trgs:
#             fig1 = av_ifr_plot_acrossDP(DPs, unitsPerDataset, trg, title=ttl, b=10,
#                                         window=[-750,750], color=seabornColorsDic[0],
#                                         zscore=False, zscoretype='overall', plot_all_units=False,
#                                         zslines=False, convolve=True, error=True, show=False, ylim=None)
#             fig1.savefig(avPlotPath+'/{}_aligned:{}(avgAcrossUnits).pdf'.format(str(trg), pattern))
#         winds = patterns_compWindows[pattern]
#         fig2, statsTable = ifr_barplot_compWind_acrossDP(DPs, unitsPerDataset, trgs, winds, title=ttl,
#                                              b=10, window=[-750,750], color=seabornColorsDic[0],
#                                              zscore=False, zscoretype='overall', convolve=False, show=False)
#         fig2.savefig(avPlotPath+'/{}vs{}@{}:{}.pdf'.format(trgs[0], trgs[1], winds, pattern))
#         statsTable.to_csv(avPlotPath+'/{}vs{}@{}:{}.csv'.format(trgs[0], trgs[1], winds, pattern))


# def ifr_heatmap(dp, units, selected_units, title='', b=5, window=[-1000,1000],
#                 zscoretype='overall', convolve=True, error=True, show=True, ylim=None, PCAsort=1, PCAwindow=[-1000,1000]):
#     sns.set_style('white')

#     fig, axes = plt.subplots(len(triggersnames))
#     for pattern, datasets in selected_units.items():
#         zscore=True
#         ifr_matrix, x = make_av_ifr_matrix(dp, units, trg, b, window, zscore, zscoretype, convolve)
#         ifr_matrixPCA, xPCA = make_av_ifr_matrix(dp, units, trg, b, PCAwindow, zscore, zscoretype, convolve)
#         # Sort units per first principal component coefficient
#         pca = PCA(n_components=5)
#         Xproj = pca.fit_transform(ifr_matrixPCA)
#         coefPC1 = Xproj[:,0]
#         if PCAsort==1:
#             PC1sorted = np.argsort(coefPC1)
#         elif PCAsort==-1:
#             PC1sorted = np.argsort(coefPC1)[::-1]
#         #mean = pca.mean_
#         #comps = pca.components_
#         #exp_var = pca.explained_variance_ratio_
#         ifr_matrix = ifr_matrix[PC1sorted,:]
#         units = np.array(units)[PC1sorted]

#         #cmap = sns.palplot(sns.diverging_palette(12, 255, l=40, n=100, center="dark"))

#         hm = sns.heatmap(ifr_matrix, vmin=-2, vmax=2, cmap="RdBu_r", center=0, cbar_kws={'label': 'Instantaneous Firing Rate (s.d.)'})

#         if window[0]<0:
#             zero=int(len(x)*(-window[0])/(window[1]-window[0]))
#             hm.axes.plot([zero,zero], hm.axes.get_ylim()[::-1], ls="--", c=[0,0,0], lw=1)
#             if alignement_event=='off':
#                 rewt=zero+int(len(x)*400/(window[1]-window[0]))
#                 hm.axes.plot([rewt,rewt], hm.axes.get_ylim()[::-1], ls="--", c=[30/255,144/255,255/255], lw=1)
#         hm.axes.set_yticklabels(['{}'.format(u) for u in units], rotation='horizontal')
#     #    x_hmm = np.zeros((1,8064))
#     #    hm.axes.set_xticklabels([str[i] for i in x], rotation='vertical')
#     #    for i in range(hmm.shape[1]):
#     #        if (i-8064/(window[1]-window[0]))%12!=0: hm.axes.xaxis.get_major_ticks()[i].set_visible(False)
#     #        else: hm.axes.xaxis.get_major_ticks()[i].set_visible(True)

#     fig = plt.gcf()
#     spt = 'Putative Purkinje cells' if region == 'cortex' else 'Cerebellar Nuclear Cells'
#     fig.suptitle(spt)
#     ax = plt.gca()

#     return fig, ax
# #%% Plot correlation matrix of units list
# from elephant.spike_train_generation import SpikeTrain
# from elephant.conversion import BinnedSpikeTrain
# from elephant.spike_train_correlation import covariance, corrcoef
# from quantities import s, ms

# usedUnits = {'MB021':[], 'MB022':[], 'MB023':[]}
# for ptrn, DSs in selected_units.items():
#     for DS, unts in DSs.items():
#         for u in unts:
#             if type(u)==int: usedUnits[DS].append(u)
# for k, v in usedUnits.items():
#     usedUnits[k]=np.unique(v)

# # 62 units total: 13 sure, 49 unsure.
# paperUnits = {'MB021':npa([219, 220, 221, 222, 223, 225, 226, 229, 230, 231, 232, 233, 234,
#                            284, 285, 286, 287, 289, 290, 295, 321]),
#               'MB022':npa([ 144,  156,  186,  205,  216,  266,  277,  280,  319,  349,  763,
#                            783,  864, 872,  874, 1063, 1078]),
#               'MB023':npa([ 83,  91,  93, 160, 166, 168, 179, 195, 199, 204, 206, 207, 209, 211, 249,
#                            284, 286, 337, 379, 525, 550, 551, 552])}
# sureUnits= {'MB021':npa([321, 290, 220, 221]),
#              'MB022':npa([142, 812, 763, 783, 764, 480]),
#              'MB023':npa([268, 258, 337])}

# def plot_cm(dp, units, b=5, cwin=100, cbin=1, corrEvaluator='CCG', vmax=0):
#     '''Plot correlation matrix.
#     dp: datapath
#     units: units list of the same dataset
#     b: bin, in milliseconds'''
#     try:
#         assert corrEvaluator in ['CCG', 'corrcoeff']
#     except:
#         print('WARNING: {} should be in {}. Exiting now.'.format(corrEvaluator, ['CCG', 'corrcoeff']))
#         return
#     # Sort units by depth
#     if os.path.isfile(dp+'/FeaturesTable/FeaturesTable_good.csv'):
#         ft = pd.read_csv(dp+'/FeaturesTable/FeaturesTable_good.csv', sep=',', index_col=0)
#         bestChs=np.array(ft["WVF-MainChannel"])
#         depthIdx = np.argsort(bestChs)[::-1] # From surface (high ch) to DCN (low ch)
#         table_units=np.array(ft.index, dtype=np.int64)[depthIdx]
#         table_channels = bestChs[depthIdx]
#     else:
#         print('You need to export the features tables using phy first!!')
#         return
#     #TODO make all CS clusters 'good'
#     units = table_units[np.isin(table_units, units)]
#     channels = table_channels[np.isin(table_units, units)]
#     # Get correlation matrix
#     cmCCG=np.empty((len(units), len(units)))
#     trnLs = []
#     for i1, u1 in enumerate(units):
#         tb1=trnb(dp, u1, 1) # 1 in ms
#         t1 = SpikeTrain(trn(dp, u1)*1./30*ms, t_stop=len(tb1)*1)
#         trnLs.append(t1)
#         for i2, u2 in enumerate(units):
#             if u1!=u2:
#                 CCG = ccg(dp, [u1, u2], cbin, cwin)[0,1,:]
#                 coeffCCG = CCG[len(CCG)//2+1]
#                 #coeffCCG/=np.sqrt((1000./np.mean(isi(dp1, u1)))*(1000./np.mean(isi(dp1, u2))))
#             else:
#                 coeffCCG=0
#             cmCCG[i1, i2]=coeffCCG

#     if corrEvaluator == 'CCG':
#         cm = cmCCG
#         vmax = 10 if vmax == 0 else vmax
#     elif corrEvaluator == 'corrcoeff':
#         cm = covariance(BinnedSpikeTrain(trnLs, binsize=b*ms))
#         vmax = 0.05 if vmax == 0 else vmax

#     # Plot correlation matrix
#     plt.figure()
#     hm = sns.heatmap(cm, vmin=0, vmax=vmax, cmap='viridis')
#     hm.axes.plot(hm.axes.get_xlim(), hm.axes.get_ylim()[::-1], ls="--", c=[0.5,0.5,0.5], lw=1)
#     hm.axes.set_yticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=0)
#     hm.axes.set_xticklabels(['{}@{}'.format(units[i], channels[i]) for i in range(len(units))], rotation=45, horizontalalignment='right')
#     hm.axes.set_title('Dataset: {}'.format(dp.split('/')[-1]))
#     hm.axes.set_aspect('equal','box-forced')
#     fig = plt.gcf()
#     plt.tight_layout()
#     print(units)
#     return fig

# def get_summaryMFR(dp, units):
#     allMFR = []
#     for u in units:
#         print('UNIT {}'.format(u))
#         isint_s = isi(dp, u)*1./1000
#         mfr = 1./np.mean(isint_s)
#         allMFR.append(mfr)
#         if mfr>=3:
#             print('WARNING {} mfr >=3!'.format(u))
#     return np.array(allMFR)

# def plot_summary_MFR(DPs, paperUnits, ylim=5, jitter=True, show=True, saveDir='/home/ms047/Desktop', saveFig=False, saveData=False):
#     allMFRs = np.array([])
#     for DS, units in paperUnits.items():
#         mfrs = get_summaryMFR(DPs[DS], units)
#         allMFRs = np.append(allMFRs, mfrs)
#     fig, ax = plt.subplots(figsize=(5,2))
#     sns.stripplot(data=allMFRs, size=6, jitter=jitter, dodge=True, color=seabornColorsDic[3], alpha=0.6, marker="D", orient='h', zorder=1)
#     sns.pointplot(data=allMFRs, dodge=.532, join=False, color=(139/256,0,0), markers="D", scale=1, orient='h', ci='sd')
#     ax.set_title('Mean firing rate of all putative Complex Spikes')
#     ax.set_xlim([0, ylim])
#     ax.set_xlabel('Mean Firing Rate (Hz)')
#     ax.set_yticklabels([])
#     print('All MFR mean: {} +/- {} Hz'.format(np.mean(allMFRs), np.std(allMFRs)))
#     if saveFig or saveData:
#         fig_n = 'summaryIFR'
#         Dir = saveDir+'/'+fig_n
#         if not os.path.isdir(Dir): os.mkdir(Dir)
#         if saveData:
#             np.save(Dir+'.npy', allMFRs)
#         if saveFig:
#             fig.savefig(Dir+'.pdf')
#     if not show:
#         plt.close(fig)
#     return fig

# #%% Align spike train on given events from matlab generated alignements

# def get_ifr_trace_old(unit, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports'):
#     arr = get_npy_export(unit, alignement_event, start_format, dp)
#     if start_format=='bst': # else, already ifr
#         bin_=0.001 # in seconds
#         gaussWin=sgnl.gaussian(64, 4)
#         gaussWin/=sum(gaussWin) # normalize !!!! For convolution, if we want to keep the amplitude unchanged!!
#         ifr = np.zeros((arr.shape[0], 8064))
#         for i in range(arr.shape[0]):
#             ifr[i,:] = np.convolve(arr[i,:]/bin_, gaussWin)

#     elif start_format=='ifr':
#         ifr=arr.copy()
#     ifr_mn = np.array([np.mean(ifr, axis=1), ]*ifr.shape[1]).transpose()
#     ifr_sd = np.array([np.std(ifr, axis=1), ]*ifr.shape[1]).transpose()
#     # Set 0 sd to 1 so that dividing does not change anything
#     for i in range(ifr_sd.shape[0]):
#         if np.all(ifr_sd[i,:]==0): ifr_sd[i,:]=1

#     assert ifr.shape == ifr_mn.shape == ifr_sd.shape
#     return ifr, ifr_mn, ifr_sd

# def make_av_ifr_matrix_old(units, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports', window=[-4000, 4000], zscoretype='overall'):

#     assert zscoretype in ['overall', 'trialwise']
#     # Window and bins translation
#     maxWin=4000; minWin=-4000;
#     window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
#     bin_=1 # 1 ms bins
#     convSamples=8064
#     bin_ifr = bin_*(maxWin-minWin+bin_)*1./convSamples # to compensate for the convolution resampling

#     ifr, ifr_mn, ifr_sd = get_ifr_trace_old(units[0], alignement_event, start_format, dp)

#     y = ifr[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]

#     x = np.arange(window[0], window[1]+bin_ifr, bin_ifr)
#     if x.shape[0]>y.shape[1]:
#         x=x[:-1]
#     assert x.shape[0]==y.shape[1]


#     ifr_matrix=np.zeros((len(units), len(x)))

#     for i, u in enumerate(units):
#         print('for unit {}'.format(u))
#         ifr, ifr_mn, ifr_sd = get_ifr_trace(u, alignement_event, start_format, dp)
#         y = ifr[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]

#         if zscoretype=='overall':
#             ifr_fl = ifr.flatten()
#             y_mn = np.mean(ifr_fl)
#             y_sd = np.std(ifr_fl)
#             print('overall mean:{}, sd:{}'.format(y_mn, y_sd))
#         elif zscoretype=='trialwise':
#             y_mn = ifr_mn[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
#             y_sd = ifr_sd[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
#             print('trialwise mean:{}, sd:{}'.format(y_mn[:,0], y_sd[:,0]))

#         y = (y-y_mn)/y_sd
#         y_zs=np.mean(y, axis=0)

#         ifr_matrix[i, :] = y_zs

#     return ifr_matrix, x

# def ifr_trace_old(units, alignement_event, window=[-4000, 4000], start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports',
#               colors=[(0,0,0), (1,0,0), (0,1,0), (0,0,1)], zscore=False, plot_all_traces=False, zslines=False, offset=True, title=None, zscoretype='overall'):
#     '''Window has to be in milliseconds.'''
#     if type(units)!=list: units=[units]
#     if offset:
#         fig, axes = plt.subplots(len(units), 1, figsize=(10,3*len(units)))
#     else:
#         fig, axes = plt.subplots(1, 1)

#     for ui, unit in enumerate(units):
#         ax=axes[ui] if offset else axes
#         color=colors[ui]
#         print(unit, type(unit))
#         if start_format in ['bst','ifr']:
#             ifr, ifr_mn, ifr_sd = get_ifr_trace_old(unit, alignement_event, start_format, dp)
#         else:
#             return
#         # Window and bins translation
#         maxWin=4000; minWin=-4000;
#         window = [max(window[0], minWin), min(window[1], maxWin)] # cannot be further than -4 - 4 seconds
#         bin_=1 # 1 ms bins
#         convSamples=8064
#         bin_ifr = bin_*(maxWin-minWin+bin_)*1./convSamples # to compensate for the convolution resampling
#         x = np.arange(window[0], window[1]+bin_ifr, bin_ifr)
#         y = ifr[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
#         if x.shape[0]>y.shape[1]:
#             x=x[:-1]
#         assert x.shape[0]==y.shape[1]
#         ax.set_title(str(unit)) if offset else ax.set_title(str(units))


#         if zscore:
#             assert zscoretype in ['overall', 'trialwise']
#             if zscoretype=='overall':
#                 y_mn=np.mean(ifr.flatten())
#                 y_sd=np.std(ifr.flatten())
#                 print('overall mean:{}, sd:{}'.format(y_mn, y_sd))
#             if zscoretype=='trialwise':
#                 y_mn = ifr_mn[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
#                 y_sd = ifr_sd[:, int(convSamples/2)+int(window[0]/bin_ifr)-1:int(convSamples/2)+int(window[1]/bin_ifr)+1]
#                 print('trialwise mean:{}, sd:{}'.format(y_mn[:,0], y_sd[:,0]))
#             y = (y-y_mn)/y_sd
#             y_zs=np.mean(y, axis=0)
#             y_zs_sem=stats.sem(y, axis=0)
#             if plot_all_traces:
#                 for i in range(ifr.shape[0]):
#                         ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
#             ax.plot(x, y_zs, lw=1, color=color)
#             ax.fill_between(x, y_zs-y_zs_sem, y_zs+y_zs_sem, facecolor=color, interpolate=True, alpha=0.2)
#             ax.plot(x, y_zs-y_zs_sem, lw=0.5, color=color)
#             ax.plot(x, y_zs+y_zs_sem, lw=0.5, color=color)
#             ax.plot([x[0], x[-1]], [0,0], ls="--", c=(0,0,0), lw=0.5)
#             if zslines:
#                 ax.plot([x[0], x[-1]], [1,1], ls="--", c=[1,0,0], lw=1)
#                 ax.plot([x[0], x[-1]], [2,2], ls="--", c=[1,0,0], lw=1)
#                 ax.plot([x[0], x[-1]], [3,3], ls="--", c=[1,0,0], lw=1)
#                 ax.plot([x[0], x[-1]], [-1,-1], ls="--", c=[0,0,1], lw=1)
#                 ax.plot([x[0], x[-1]], [-2,-2], ls="--", c=[0,0,1], lw=1)
#                 ax.plot([x[0], x[-1]], [-3,-3], ls="--", c=[0,0,1], lw=1)
#             ax.plot([0,0], [-3, 3], ls="--", c=[0,0,0], lw=0.5)
#             if alignement_event=='off':
#                 ax.plot([400,400], [-3, 3], ls="--", c=[30/255,144/255,255/255], lw=0.5)
#             ax.set_ylim([-1.5, 1.5])
#             ax.set_ylabel('Inst.F.R (s.d.)')
#             ax.set_xlim(window[0], window[1])
#             ax.set_xlabel('Time (ms)')

#         elif not zscore:
#             y_mn = np.mean(y, axis=0)
#             y_sem = stats.sem(y, axis=0)
#             if plot_all_traces:
#                 for i in range(ifr.shape[0]):
#                         ax.plot(x, y[i,:], lw=0.3, color=color, alpha=0.2)
#             ax.plot(x, y_mn, lw=1, color=color, alpha=1)
#             ax.fill_between(x, y_mn-y_sem, y_mn+y_sem, facecolor=color, interpolate=True, alpha=0.2)
#             ax.plot(x, y_mn-y_sem, lw=0.5, color=color)
#             ax.plot(x, y_mn+y_sem, lw=0.5, color=color)
#             ax.plot([0,0], ax.get_ylim(), ls="--", c=[0,0,0], lw=0.5)
#             if alignement_event=='off':
#                 ax.plot([400,400], ax.get_ylim(), ls="--", c=[30/255,144/255,255/255], lw=0.5)
#             ax.set_xlim(window[0], window[1])
#             ax.set_xlabel('Time (ms)')
#             ax.set_ylabel('Inst.F.R (Hz)')

#     fig.tight_layout()
#     return fig, axes


# def ifr_heatmap_old(region, alignement_event, start_format='ifr', dp='/home/ms047/Dropbox/Science/PhD/Data_Presentation/SfN 2018/Behavior/mat-npy-exports', window=[-4000,4000], sort_dir=1):
#     assert region in ['cortex', 'nuclei']
#     sns.set_style('white')
#     if region=='cortex':
#         units=[29, 263, 292, 363, 611, 710, 32039, 33145, 34469, 50838, 50839, 75046]
#     elif region=='nuclei':
#         units=[1348, 3725, 7620, 15097, 15110, 15112, 16421, 18206, 18944, 20041, 20610, 26610, 50842, 63368, 79513]
#     ifr_matrix, x = make_av_ifr_matrix(units, alignement_event, start_format, dp, window)
#     if alignement_event=='off':
#         ifr_matrixPCA, xPCA = make_av_ifr_matrix(units, alignement_event, start_format, dp, [400, 600])
#     if alignement_event=='movon':
#         ifr_matrixPCA, xPCA = make_av_ifr_matrix(units, alignement_event, start_format, dp, [0, 1000])
#     # Sort units per first principal component coefficient
#     pca = PCA(n_components=5)
#     Xproj = pca.fit_transform(ifr_matrixPCA)
#     coefPC1 = Xproj[:,0]
#     if sort_dir==1:
#         PC1sorted = np.argsort(coefPC1)
#     elif sort_dir==-1:
#         PC1sorted = np.argsort(coefPC1)[::-1]
#     #mean = pca.mean_
#     #comps = pca.components_
#     #exp_var = pca.explained_variance_ratio_
#     ifr_matrix = ifr_matrix[PC1sorted,:]
#     units = np.array(units)[PC1sorted]

#     #cmap = sns.palplot(sns.diverging_palette(12, 255, l=40, n=100, center="dark"))
#     fig = plt.figure(figsize=(15, 0.3*len(units)))
#     hm = sns.heatmap(ifr_matrix, vmin=-2, vmax=2, cmap="RdBu_r", center=0, cbar_kws={'label': 'Instantaneous Firing Rate (s.d.)'})

#     if window[0]<0:
#         zero=int(len(x)*(-window[0])/(window[1]-window[0]))
#         hm.axes.plot([zero,zero], hm.axes.get_ylim()[::-1], ls="--", c=[0,0,0], lw=1)
#         if alignement_event=='off':
#             rewt=zero+int(len(x)*400/(window[1]-window[0]))
#             hm.axes.plot([rewt,rewt], hm.axes.get_ylim()[::-1], ls="--", c=[30/255,144/255,255/255], lw=1)
#     hm.axes.set_yticklabels(['{}'.format(u) for u in units], rotation='horizontal')
# #    x_hmm = np.zeros((1,8064))
# #    hm.axes.set_xticklabels([str[i] for i in x], rotation='vertical')
# #    for i in range(hmm.shape[1]):
# #        if (i-8064/(window[1]-window[0]))%12!=0: hm.axes.xaxis.get_major_ticks()[i].set_visible(False)
# #        else: hm.axes.xaxis.get_major_ticks()[i].set_visible(True)

#     fig = plt.gcf()
#     spt = 'Putative Purkinje cells' if region == 'cortex' else 'Cerebellar Nuclear Cells'
#     fig.suptitle(spt)
#     ax = plt.gca()

#     return fig, ax
