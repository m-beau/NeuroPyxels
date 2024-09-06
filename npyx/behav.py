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

from numba import njit

from npyx.utils import npa, thresh, thresh_consec, smooth,\
                        sign, assert_int, assert_iterable, npyx_cacher

from npyx.inout import read_metadata, get_npix_sync, paq_read, list_files
from npyx.gl import get_rec_len
from npyx.spk_t import mean_firing_rate, get_common_good_sections
from npyx.corr import crosscorr_cyrille, frac_pop_sync
from npyx.merger import assert_multi, get_ds_table

#%% Generate trials dataframe from either paqIO file or matlab datastructure

def behav_dic(dp, f_behav=None, vid_path=None, again=False, again_align=False, again_rawpaq=False,
                     lick_ili_th=0.075, n_ticks=1024, diam=200, gsd=25,
                     plot=False, drop_raw=True, cam_paqi_to_use=None):
    '''
    Remove artefactual licking, processes rotary encoder and wheel turn trials.

    Arguments:
        - dp: str, path of kilosort dataset.
        - f_behav: str, paqIO behavioural file. If None, assumed to be at ../behavior relative to dp.
        - vid_path: str, path to directory with videos.
        - again: bool, whether to recompute behavioural variables and convert them to Neuropixels temporal reference frame.
        - again_align: bool, whether to re-align paqIO to Neuropixels trigger.
        - again_rawpaw: bool, whether to reload data from paqIO file.
        - lick_ili_th: float, inter lick interval threshold to exclude 'fake licks' (0.075 is the right threshold, across mice!)
        - n_ticks: int, number of ticks of rotary encoder.
        - diam: float, outer diameter of wheel coupled to the rotary encoder.
        - gsd: int, gaussian standard deviation of roatry decoded speed smoothing gaussian kernel.
        - plot: bool, whether to load plots regarding licking preprocessing to ensure all goes well.
        - drop_raw: drop raw paqIO data for digital channels (huge memory saver)
        - cam_paqi_to_use: [int, int], indices of first and last camera frames to consider for urnning analysis
        
    Returns:
        - behavdic: dict, dictionnary holding behavioural variables.
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
    licks_on = paqdic['LICKS_Piezo_ON_npix'].copy()
    paqdic['LICKS_Piezo_ON_npix'] = licks_on[np.diff(np.append([0],licks_on))>lick_ili_th*npix_fs]
    paqdic['licks'] = paqdic['LICKS_Piezo_ON_npix'] # for convenience

    licks_diff  = np.diff(paqdic['licks'])
    m_pre       = np.append([True], licks_diff>0.4*fs).astype(bool)
    m_post      = np.append((licks_diff<0.25*fs)&(licks_diff>0.1*fs), [True]).astype(bool)
    m_noprepost = m_pre&m_post
    paqdic['lick_salve_onsets'] = paqdic['licks'][m_noprepost]

    print(f'\nInter lick interval lower threshold set at {lick_ili_th} seconds.\n')
    if plot:
        hbins  = np.logspace(np.log10(0.005),np.log10(10), 500)
        fig,ax = plt.subplots()
        ax.set_title('Licks distribution before filtering')
        hist   = np.histogram(np.diff(licks_on/npix_fs), bins = hbins)
        ax.hist(np.diff(licks_on/npix_fs), bins=hbins)
        ax.set_xscale('log')
        plt.xlim(0.005,10)
        plt.ylim(0,max(hist[0][hist[1][:-1]>0.05])+10)
        fig,ax = plt.subplots()
        ax.set_title('Licks distribution after filtering')
        hist   = np.histogram(np.diff(paqdic['LICKS_Piezo_ON_npix']/npix_fs), bins = hbins)
        ax.hist(np.diff(paqdic['LICKS_Piezo_ON_npix']/npix_fs), bins = hbins)
        ax.set_xscale('log')
        plt.xlim(0.005,10)
        plt.ylim(0,max(hist[0][hist[1][:-1]>0.05])+10)

    # Process rotary encoder if running dataset
    # Get periods with/without
    if not 'ROT_A' in paqdic.keys():
        print('No rotary encoder data found in paqdic. Assuming no running, no camera.')
    else:
        print('Running dataset detected.')
        v = decode_rotary(paqdic['ROT_A'], paqdic['ROT_B'], paq_fs, n_ticks, diam, gsd, True)
        # detect whether maximum speed is positive or negative
        # if negative, gotta flip the sign of v so that running forward corresponds to positive values
        v *= sign(abs(np.max(v))-abs(np.min(v)))
        paqdic['ROT_SPEED'] = v

        # Process extra camera frames
        if vid_path is None:
            vid_path = dp.parent/'videos'
        videos = list_files(vid_path, 'avi', 1) if vid_path.exists() else []

        if not any(videos):
            print(f'No videos found at {vid_path} - camera triggers not processed.\n')
        else:
            if cam_paqi_to_use is not None:
                assert len(cam_paqi_to_use)==2
                assert cam_paqi_to_use[0]>=0, 'cam_paqi_to_use[0] cannot be negative!'
                assert cam_paqi_to_use[1]<=len(paqdic['CameraFrames'])-1,\
                    f"cam_paqi_to_use[0] cannot be higher than {len(paqdic['CameraFrames'])-1}!"
                if cam_paqi_to_use[1]<0:cam_paqi_to_use[1]=len(paqdic['CameraFrames'])+cam_paqi_to_use[1]
                ON  = paqdic['CameraFrames_ON']
                OFF = paqdic['CameraFrames_OFF']
                mon = (ON>=cam_paqi_to_use[0])&(ON<=cam_paqi_to_use[1])
                mof = (OFF>=cam_paqi_to_use[0])&(OFF<=cam_paqi_to_use[1])
                paqdic['CameraFrames_ON_npix']  = paqdic['CameraFrames_ON_npix'][mon]
                paqdic['CameraFrames_OFF_npix'] = paqdic['CameraFrames_OFF_npix'][mof]
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
                fi += nf
                ii =  fi+n_man_abort
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
            paqdic['CameraFrames_ON_npix'] = frames_npix

        # load swing onsets if already computed
        swings_fn = dp.parent/'behavior'/'leftfrontpaw_swings.npy'
        if swings_fn.exists():
            print('Loaded swing onsets from deeplabcut data.')
            swings_onof_i=np.load(swings_fn)
            paqdic['swing_ON_npix']  = paqdic['CameraFrames_ON_npix'][swings_onof_i[:,0].ravel()]
            paqdic['swing_OFF_npix'] = paqdic['CameraFrames_ON_npix'][swings_onof_i[:,1].ravel()]

        swings_fn = dp.parent/'behavior'/'swing_onsets_phasethreshold.npy'
        if swings_fn.exists():
            print('Loaded swing phase onsets from deeplabcut data.')
            swing_on  = np.load(swings_fn)
            stance_on = np.load(swings_fn.parent/'stance_onsets_phasethreshold.npy')
            paqdic['swing_phase_ON_npix']  = paqdic['CameraFrames_ON_npix'][swing_on]
            paqdic['stance_phase_ON_npix'] = paqdic['CameraFrames_ON_npix'][stance_on]
            paqdic['deeplabcut_df']        = pd.read_csv(swings_fn.parent/'locomotion_processed.csv')


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

    paqdic['random_rewards']  = npa(random_rewards)    
    paqdic['omitted_cues']    = npa(omitted_cues)
    paqdic['rewarded_cues']   = npa(rewarded_cues)
    paqdic['engaged_rewards'] = npa(engaged_rewards)
    paqdic['engaged_cues']    = npa(engaged_cues)

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

                new_row = pd.DataFrame({'npix': [npixk],
                                           'paq': [paqk],
                                           'p': [p],
                                           'len_match': [lenmatch]})
                paq_npix_df = pd.concat([paq_npix_df, new_row], ignore_index=True)
                # paq_npix_df=paq_npix_df.append({'npix':npixk, 'paq':paqk, 'p':p, 'len_match':lenmatch}, ignore_index=True)
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
    len_arr     = npa([[k,len(npix_ons[k])] for k in npix_paq.keys() if k in npix_ons.keys()])
    sync_npix_k = len_arr[np.argmax(len_arr[:,1]),0]
    sync_npix   = npix_ons[sync_npix_k]
    sync_paq    = paqdic[npix_paq[sync_npix_k]]

    # Model drift: npix_sync = a * paq_sync + b
    (a, b) = np.polyfit(sync_paq, sync_npix, 1)
    paqdic['a'] = a
    paqdic['b'] = b
    print(f'Drift (assumed linear) of {round(abs(a*paq_fs/npix_fs-1)*3600*1000,2)}ms/h, \noffset of {round(b/npix_fs,2)}s between ephys and paq files.\n')
    for paqk in list(paqdic.keys()):
        paqv=paqdic[paqk]
        if '_ON' in paqk and len(paqv)>1:
            off_key=f"{paqk.split('_ON')[0]+'_OFF'}"
            if paqk in npix_paq.keys():
                paqdic[f'{paqk}_npix']    = npix_ons[npix_paq[paqk]]
                paqdic[f"{off_key}_npix"] = npix_ofs[npix_paq[paqk]] # same key for onsets and offsets
            else:
                #paqdic[f'{paqk}_npix_old']=align_timeseries([paqv], [sync_paq, sync_npix], [paq_fs, npix_fs]).astype(np.int64)
                #paqdic[f"{off_key}_npix_old"]=align_timeseries([paqdic[off_key]], [sync_paq, sync_npix], [paq_fs, npix_fs]).astype(np.int64)
                paqdic[f'{paqk}_npix']    = np.round(a*paqv+b, 0).astype(np.int64)
                paqdic[f"{off_key}_npix"] = np.round(a*paqdic[off_key]+b, 0).astype(np.int64)

    pickle.dump(paqdic, open(str(fn),"wb"))

    return paqdic

def load_PAQdata(paq_f, variables='all', again=False, unit='seconds', th_frac=0.2):
    '''
    Used to load analog (wheel position...)
    and threshold digital (piezo lick...) variables from paqIO file.
    If variables is not a list, all PackIO variables will be exported.

    Arguments:
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
          'opto_stims':'digital', 'ROT_A':'digital', 'ROT_B':'digital', 'PUFF':'digital'}
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
    Arguments:
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
    wheel_turn_dic                      = {}
    wheel_turn_dic['vr_fs']             = vr_rate
    wheel_turn_dic['wheel_position_mm'] = paqdic['ROTreal']*(np.pi*wheel_diam)/360/wheel_gain
    wheel_turn_dic['wheel_speed_cm/s']  = np.diff(wheel_turn_dic['wheel_position_mm'])*vr_rate/10 # mm to cm/s

    if Path(fn).exists() and not again:
        df = pd.read_csv(fn)
    else:
        ## Make and fill trials dataframe
        df=pd.DataFrame(columns=['trial_type', 'trialnum', 'trialside', 'trial_onset', 'object_onset',
                                'movement_onset', 'movement_offset', 'movement_duration', 'ballistic',
                                'correct', 'trial_offset',
                                'reward_onset', 'cue_onset', 'ghost_onset', 'lick_onsets'])

        # Process wheel trials
        nwheeltrials=len(paqdic['TRIALON_ON'])
        df['trial_type']=['wheel_turn']*nwheeltrials
        df["lick_onsets"]=df["lick_onsets"].astype(object) # to be able to store array
        if include_wheel_data:
            df['object_position']        = np.nan
            df['wheel_position']         = np.nan
            df['trial_onset_relpaq']     = np.nan
            df['movement_onset_relpaq']  = np.nan
            df['movement_offset_relpaq'] = np.nan
            df['trial_offset_relpaq']    = np.nan
            df['lick_trace']             = np.nan
            df["object_position"]        = df["object_position"].astype(object)
            df["wheel_position"]         = df["wheel_position"].astype(object)
            df['lick_trace']             = df["wheel_position"].astype(object)
        pad=4
        assert difficulty in [2,3]
        for tr in df.index:
            if verbose: print(f'  Wheel steering trial {tr}/{len(df.index)}...')
            npixon        = paqdic['TRIALON_ON_npix'][tr]
            npixof        = paqdic['TRIALON_OFF_npix'][tr]
            paqon         = int(paqdic['TRIALON_ON'][tr])
            paqoff        = int(paqdic['TRIALON_OFF'][tr])
            i1,i2         = int(paqon-pad*paq_fs),int(paqoff+pad*paq_fs)
            ob_on_vel     = np.diff(paqdic['ROTreal'][paqon:paqon+500])
            ob_on_vel     = abs(ob_on_vel)/max(abs(ob_on_vel))
            ob_on_paq     = thresh(ob_on_vel, 0.5, 1)[0]+1 # add 1 because velocity is thresholded
            start_side    = sign(paqdic['ROT'][paqon+ob_on_paq+1]) # 1 or -1
            # wheel and object positions are clipped between -4s before trial onset and 4s after trial offset
            opos          = paqdic['ROT'][i1:i2] # wheel pos in degrees
            wpos          = wheel_turn_dic['wheel_position_mm'][i1:i2] # in mm
            wpos_mvt      = paqdic['ROTreal'][paqon+ob_on_paq:paqoff]
            wvel          = np.diff(wpos)
            wvel_mvt      = wvel[int(pad*paq_fs+ob_on_paq):-int(pad*paq_fs)]
            wvel_mvt_norm = wvel_mvt/min(abs({-1:max, 1:min}[start_side](wvel_mvt)), 2)

            # assess trial outcome from wheel kinetics
            wpos_outeval   = paqdic['ROT'][int(paqon+ob_on_paq):int(paqoff)]
            stay_for_rew   = int(rew_frames/vr_rate*paq_fs)
            correct        = 0
            center_crossed = False
            th0            = np.append(thresh(wpos_outeval, 0, 1), thresh(wpos_outeval, 0, -1))
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
            df.loc[tr, 'trialnum']     = tr
            df.loc[tr, 'trial_onset']  = npixon
            df.loc[tr, 'trial_offset'] = npixof
            df.loc[tr, 'object_onset'] = int(npixon+ob_on_paq*npix_fs/paq_fs)
            df.loc[tr, 'trialside']    = start_side
            df.loc[tr, 'correct']      = correct
            lickmask                   = (paqdic['LICKS_Piezo_ON_npix']>npixon)&(paqdic['LICKS_Piezo_ON_npix']<npixof+5*npix_fs)
            df.at[tr, "lick_onsets"]   = paqdic['LICKS_Piezo_ON_npix'][lickmask]
            if correct:
                movonpaq  = ob_on_paq+thresh(wvel_mvt_norm, -start_side*0.5, -start_side)[0]
                movofpaq  = ob_on_paq+np.append(thresh(wpos_mvt, rew_zone, -1), thresh(wpos_mvt, -rew_zone, 1)).min()
                mov_dur   = (movofpaq-movonpaq)*1000/paq_fs
                ballistic = (mov_dur<ballistic_thresh) # in ms
                df.loc[tr, 'movement_onset']    = int(npixon+(movonpaq)*npix_fs/paq_fs)
                df.loc[tr, 'movement_offset']   = int(npixon+(movofpaq)*npix_fs/paq_fs)
                df.loc[tr, 'movement_duration'] = mov_dur
                df.loc[tr, 'ballistic']         = ballistic
                if plot:
                    plt.figure()
                    plt.plot(wvel[int(pad*paq_fs+500):-int(pad*paq_fs)], c='grey')
                    plt.plot(wpos[int(pad*paq_fs+500):-int(pad*paq_fs)], c='k')
                    ls = '--' if ballistic else '-'
                    plt.plot([movonpaq-500, movonpaq-500], [min(wpos), max(wpos)], c='g', ls=ls)
                    plt.plot([movofpaq-500, movofpaq-500], [min(wpos), max(wpos)], c='r', ls=ls)
                    plt.title(f'trial {tr}\nduration:{mov_dur}ms')
            if include_wheel_data:
                df.at[tr, 'object_position']=opos
                df.at[tr, 'wheel_position']=wpos
                df.loc[tr, 'trial_onset_relpaq']=int(pad*paq_fs)
                if correct:
                    df.loc[tr, 'movement_onset_relpaq']=int(pad*paq_fs)+movonpaq
                    df.loc[tr, 'movement_offset_relpaq']=int(pad*paq_fs)+movofpaq
                df.loc[tr, 'trial_offset_relpaq']=len(wpos)-int(pad*paq_fs)
                df.at[tr, 'lick_trace']=paqdic['LICKS_Piezo'][i1:i2]

        # Append trial rewards onsets ##TODO
        for tr in df.index:
            of=df.loc[tr, 'trial_offset']
            rew_mask=(paqdic['REW_ON_npix']>of)&(paqdic['REW_ON_npix']<(of+1*npix_fs)) # happens about 400ms after trial offset
            if np.any(rew_mask):
                if df.loc[tr, 'correct']   != 1: print(f'WARNING seems like a reward onset was found after trial {tr} offset, although incorrect...\
                                                Probably beause of 360deg jump of ROT channel not followed by ROTreal.')
                df.loc[tr, 'reward_onset'] = paqdic['REW_ON_npix'][rew_mask][0]
            else:
                if df.loc[tr, 'correct']   == 1: print(f'WARNING no reward was found after trial {tr} offset, although correct!!! Figure out the problem ++!!')

        # Now find random rewards and respective cues if any
        random_rewards=paqdic['REW_ON_npix'][~np.isin(paqdic['REW_ON_npix'], df['reward_onset'])]
        for r in random_rewards:
            cue_mask=(paqdic['CUE_ON_npix']>r-1*npix_fs)&(paqdic['CUE_ON_npix']<r) # cues happen about 500ms before reward
            i=df.index[-1]+1
            if np.any(cue_mask):
                c=paqdic['CUE_ON_npix'][cue_mask][0]
                df.loc[i, 'trial_type']   = 'cued_reward'
                df.loc[i, 'reward_onset'] = r
                df.loc[i, 'cue_onset']    = c
                lickmask=(paqdic['LICKS_Piezo_ON_npix']>c)&(paqdic['LICKS_Piezo_ON_npix']<r+5*npix_fs) # from cue to 5s after reward
            else:
                df.loc[i, 'trial_type']   = 'random_reward'
                df.loc[i, 'reward_onset'] = r
                lickmask=(paqdic['LICKS_Piezo_ON_npix']>r)&(paqdic['LICKS_Piezo_ON_npix']<r+5*npix_fs) # from reward to 5s after reward
            df.at[i, "lick_onsets"]=paqdic['LICKS_Piezo_ON_npix'][lickmask]

        # Finally, fill in the cues alone (omitted rewards)
        cues_alone=paqdic['CUE_ON_npix'][~np.isin(paqdic['CUE_ON_npix'], df['cue_onset'])]
        for c in cues_alone:
            i=df.index[-1]+1
            df.loc[i, 'trial_type'] = 'cue_alone'
            df.loc[i, 'cue_onset']  = c
            lickmask=(paqdic['LICKS_Piezo_ON_npix']>c)&(paqdic['LICKS_Piezo_ON_npix']<c+5*npix_fs) # from cue to 5s after cue
            df.at[i, "lick_onsets"] = paqdic['LICKS_Piezo_ON_npix'][lickmask]

        # Also add spontaneous licks
        if add_spont_licks:
            allocated_licks=npa([list(df.loc[i, "lick_onsets"]) for i in df.index]).flatten()
            spontaneous_licks=paqdic['LICKS_Piezo_ON_npix'][~np.isin(paqdic['LICKS_Piezo_ON_npix'], allocated_licks)]
            i=df.index[-1]+1
            df.loc[i, 'trial_type'] = 'spontaneous_licks'
            df.at[i, "lick_onsets"] = spontaneous_licks

        df.to_csv(fn)

    # finally, convert to dictionnary - more convenient
    mask_left      = (df['trialside']==1)
    mask_right     = (df['trialside']==-1)
    mask_correct   = (df['correct']==1)&(df['ballistic']==1)
    mask_incorrect = (df['correct']==0)
    mask_rr        = (df['trial_type']=='random_reward')
    mask_cr        = (df['trial_type']=='cued_reward')
    wheel_turn_dic={**wheel_turn_dic,
    **{'olc_on':  df.loc[mask_left&mask_correct, 'object_onset'].values.astype(float),
     'oli_on':    df.loc[mask_left&mask_incorrect, 'object_onset'].values.astype(float),
     'orc_on':    df.loc[mask_right&mask_correct, 'object_onset'].values.astype(float),
     'ori_on':    df.loc[mask_right&mask_incorrect, 'object_onset'].values.astype(float),

     'lc_on':     df.loc[mask_left&mask_correct, 'movement_onset'].values.astype(float),
     'rc_on':     df.loc[mask_right&mask_correct, 'movement_onset'].values.astype(float),
     'lc_of':     df.loc[mask_left&mask_correct, 'movement_offset'].values.astype(float),
     'rc_of':     df.loc[mask_right&mask_correct, 'movement_offset'].values.astype(float),

     'c_r':       df.loc[mask_correct, 'reward_onset'].values.astype(float),
     'i_of':      df.loc[mask_incorrect, 'trial_offset'].values.astype(float),

    # random/cued rewards are now handled in get_behav_dic
    'rr':         df.loc[mask_rr, 'reward_onset'].values.astype(float),
    'cr_c':       df.loc[mask_cr, 'cue_onset'].values.astype(float),
    'cr_r':       df.loc[mask_cr, 'reward_onset'].values.astype(float)}}

    return df, wheel_turn_dic


#%% Recording period extraction

# Find periods without motion

@npyx_cacher
def load_baseline_periods(dp = None, behavdic = None, rec_len = None, dataset_with_opto = True, 
                          speed_th_steer = 0.1, speed_th_run = 0.5, light_buffer = 0.5,
                          again = False, again_behav = False, verbose = True, return_all=False,
                          cache_results=True, cache_path=None):
    """
    Function to calculate periods of undisturbed neural activity
    (no (monitored) behaviour or optostims).

    Arguments:
        - dp: str, path to Neuropixels dataset (only provide if no behavdic) 
        - behavdic: dict, output of behav_dic() (only provide if no dp)
        - rec_len: float, recording length (in seconds). Only useful is dp is None.
        - dataset_with_opto: whether the dataset had functioning channel rhodopsin and optostims were delivered
        - speed_th_steer: threhsold to consider a 'steering' period (cm/s)
        - speed_th_run: threshold to consider a 'running' period (mm/s)
        - light_buffer: buffer to add before light onsets/after light offsets (s)
        - verbose: bool, whether to print details.
        - return_all: bool, whether to return all two or three arrays - baselin_periods, nomove_periods, noopto_periods\
        - again: bool, whether to recompute results rather than loading them from cache.
        - cache_results: bool, whether to cache results at local_cache_memory.
        - cache_path: None|str, where to cache results.
                        If None, dp/.NeuroPyxels will be used.
    
    Returns:
        - baseline_periods: 2D array of shape (n_periods, 2), [[t1,t2], ...] in seconds.
    """

    assert dp is None or behavdic is None, "Only one of dp or behavdic should be provided"

    if behavdic is None:
        behavdic = behav_dic(dp, f_behav=None, vid_path=None,
        again=again_behav, again_align=again_behav, again_rawpaq=again_behav,
                     lick_ili_th=0.075, n_ticks=1024, diam=200, gsd=25,
                     plot=False, drop_raw=False, cam_paqi_to_use=None)

    if rec_len is None:
        assert dp is not None, "You must provide the recording length in seconds!"
        rec_len = get_rec_len(dp, unit='seconds')

    if 'wheel_position_mm' in behavdic.keys():
        behaviour = "steering"
    elif 'ROT_SPEED' in behavdic.keys():
        behaviour = "running"
    else:
        raise ValueError("Neither 'wheel_position_mm' nor 'ROT_SPEED' was found in behavdic!")

    if dp is not None:
        dp = Path(dp)
        th_str = speed_th_steer if behaviour == "steering" else speed_th_run
        fn_move = f"periods_{th_str}_{behaviour}.npy"
        fn_nomove = fn_move.replace(f"{th_str}_", f"{th_str}_no_")
        if dataset_with_opto:
            fn_light = f"periods_{light_buffer}_opto.npy"
            fn_nolight = fn_light.replace(f"_opto.npy", f"_no_opto.npy")
    move_periods = light_periods = None
    if dp is not None and not again:
        if (dp/fn_move).exists():
            move_periods = np.load(dp/fn_move)
            nomove_periods = np.load(dp/fn_nomove)
            if dataset_with_opto:
                if (dp/fn_light).exists():
                    light_periods = np.load(dp/fn_light)
                    no_light_periods = np.load(dp/fn_nolight)

    if move_periods is None:
        if behaviour == "steering":
            wheel_speed = np.abs(behavdic['wheel_speed_cm/s']) 
            span=1 # s
            # because paqIO oversamples at 5000Hz whereas VR runs at 30Hz, 
            # only onsets are meaningful -> cannot use onest to offset as periods
            # instead, take all periods outside of +/-1second around any threshold crossing
            steer_th = thresh(wheel_speed, speed_th_steer, 1)
            steer_spans = np.hstack([(steer_th-span*behavdic['paq_fs'])[:,None], (steer_th+span*behavdic['paq_fs'])[:,None]])
            steer_mask = np.zeros(wheel_speed.shape).astype(np.int8)
            for steer_span in steer_spans:
                steer_mask[steer_span[0]:steer_span[1]]=1
            
            steer_values   = thresh_consec(steer_mask, 0.5, 1)
            move_periods   = np.round(npa([[r[0,0],r[0,-1]] for r in steer_values])*behavdic['a']+behavdic['b'])/behavdic['npix_fs']
            nomove_periods = np.concatenate([[0], move_periods.ravel(), [rec_len]]).reshape((-1,2))
            
            if np.any(move_periods) and verbose:
                print(f'{np.sum(np.diff(move_periods, axis=1))}s of steering in total.')
                print(f'{np.sum(np.diff(nomove_periods, axis=1))}s of no steering in total.')
                
        elif behaviour == "running":
            v=behavdic['ROT_SPEED']
            consec=1 # s
            run_values = thresh_consec(v, speed_th_run, 1, consec*behavdic['paq_fs'], ret_values=True)
            move_periods = np.round(npa([[r[0,0],r[0,-1]] for r in run_values])*behavdic['a']+behavdic['b'])/behavdic['npix_fs']
            nomove_periods = np.concatenate([[0], move_periods.ravel(), [rec_len]]).reshape((-1,2))
            
            if np.any(move_periods) and verbose:
                print(f'{np.sum(np.diff(move_periods, axis=1))}s of running in total.')
                print(f'{np.sum(np.diff(nomove_periods, axis=1))}s of no running in total.')
            
    # find periods without light
    if dataset_with_opto and light_periods is None:
        light_periods = np.array(list(zip(behavdic['opto_stims_ON_npix']/behavdic['npix_fs'], behavdic['opto_stims_OFF_npix']/behavdic['npix_fs'])))
        light_periods_buffered = light_periods.copy()
        light_periods_buffered[0] -= light_buffer
        light_periods_buffered[1] += light_buffer
        no_light_periods = np.concatenate([[0], light_periods_buffered.ravel(), [rec_len]])
        no_light_periods = no_light_periods.reshape(-1, 2)

        print(f'{np.sum(np.diff(light_periods, axis=1))}s of periods with light in total.')
        print(f'{np.sum(np.diff(no_light_periods, axis=1))}s of periods with no light in total (buffer before onset/after offset: {light_buffer}s).')

    # save arrays to disk
    if dp is not None:
        np.save(dp/fn_move, move_periods)
        np.save(dp/fn_nomove, nomove_periods)
        if dataset_with_opto:
            np.save(dp/fn_light, light_periods)
            np.save(dp/fn_nolight, no_light_periods)

    if dataset_with_opto:
        # get intersection of both
        fs = 30_000
        no_light_nor_move_periods = get_common_good_sections([list((no_light_periods*fs).astype(int)),
                                                              list((nomove_periods*fs).astype(int))])
        no_light_nor_move_periods = np.array(no_light_nor_move_periods)/fs
        print(f'{np.sum(np.diff(no_light_nor_move_periods, axis=1))}s of periods with no light nor running in total (buffer before onset/after light offset: {light_buffer}s).')
        baseline_recording_periods = no_light_nor_move_periods
    else:
        baseline_recording_periods = nomove_periods

    if return_all:
        if dataset_with_opto:
            return baseline_recording_periods, nomove_periods, no_light_periods
        else:
            return baseline_recording_periods, nomove_periods

    return baseline_recording_periods


#%% Alignement, binning and processing of time series

def align_variable(events, variable_t, variable, b=2, window=[-1000,1000], remove_empty_trials=False):
    '''
    Arguments:
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

@npyx_cacher
def align_times(times, events, b=2, window=[-1000,1000], remove_empty_trials=False,
                again=False, cache_results=True, cache_path=None):
    '''
    Arguments:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - aligned_t: dictionnaries where each key is an event in absolute time and value the times aligned to this event within window.
        - aligned_tb: a len(events) x window/b matrix where the spikes have been aligned, in counts.
        - again: bool, whether to recompute results rather than loading them from cache.
        - cache_results: bool, whether to cache results at local_cache_memory.
        - cache_path: None|str, where to cache results.
                        If None, ~/.NeuroPyxels will be used (can be changed in npyx.CONFIG).
    '''
    assert np.any(events), 'You provided an empty array of events!'
    t          = np.sort(times)
    aligned_t  = {}
    tbins      = np.arange(window[0], window[1]+b, b)
    aligned_tb = np.zeros((len(events), len(tbins)-1)).astype(float)
    for i, e in enumerate(events):
        ts = t-e # ts: t shifted
        tsc = ts[(ts>=window[0]/1000)&(ts<=window[1]/1000)] # tsc: ts clipped
        if np.any(tsc) or not remove_empty_trials:
            aligned_t[e]=tsc.tolist()
            tscb = np.histogram(tsc*1000, bins=tbins)[0] # tscb: tsc binned
            aligned_tb[i,:] = tscb
        else:
            aligned_tb[i,:] = np.nan
    aligned_tb=aligned_tb[~np.isnan(aligned_tb).any(axis=1)]

    if not np.any(aligned_tb): aligned_tb = np.zeros((len(events), len(tbins)-1))

    return aligned_t, aligned_tb


def align_times_manyevents(times, events, b=2, window=[-1000,1000], fs=30000):
    '''
    Will run faster than align_times if many events are provided (will run in approx 800ms for 10 or for 600000 events,
                                                                  whereas align_times will run in about 1 second every 2000 event
                                                                  so in 5 minutes for 600000 events!)
    Arguments:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - fs: sampling rate in Hz - does not need to be exact, but the times and events arrays multiplied by that should be integers
    Returns:
        - aligned_tb: a 1 x window/b matrix where the spikes have been aligned, in counts.
    '''
    tfs, efs       = np.round(times*fs, 2), np.round(events*fs, 2)
    assert np.all(tfs==tfs.astype(np.int64)), 'WARNING sampling rate must be wrong or provided times are not in seconds!'
    indices        = np.append(0*events, 0*times+1).astype(np.int64)
    times          = np.append(efs, tfs).astype(np.int64)

    sorti          = np.argsort(times)
    indices, times = indices[sorti], times[sorti]

    win_size       = np.diff(window)[0]
    bin_size       = b

    aligned_tb     = crosscorr_cyrille(times, indices, win_size, bin_size, fs=fs, symmetrize=True)[0,1,:]

    return aligned_tb


@njit(cache=True)
def fast_align_times(times, events, b=2, window=[-1000,1000]):
    '''
    Arguments:
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
    t          = np.sort(times)
    aligned_t  = {}
    tbins      = np.arange(window[0], window[1]+b, b)
    aligned_tb = np.zeros((len(events), len(tbins)-1), dtype=np.float64)

    for i, e in enumerate(events):
        ts = t-e # ts: t shifted
        tsc = ts[(ts>=window[0]/1000)&(ts<=window[1]/1000)] # tsc: ts clipped
        aligned_t[e]=tsc
        tscb = np.histogram(tsc*1000, bins=tbins)[0] # tscb: tsc binned
        aligned_tb[i,:] = tscb

    return aligned_t, aligned_tb

def jPSTH(spikes1, spikes2, events, b=2, window=[-1000,1000], convolve=False, method='gaussian', gsd=2):
    '''
    From A. M. H. J. AERTSEN, G. L. GERSTEIN, M. K. HABIB, AND G. PALM, 1989, Journal of Neurophysiology
    Dynamics of neuronal firing correlation: modulation of 'effective connectivity'

    Arguments:
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

def trial_selector(events, trains, mfr_frac=0.1,
                   window=None,
                   min_isi=2, max_isi=None,
                   min_iti=100, fs=30000,
                   verbose=False):
    """
    Given a set of events and trains (in samples),
    returns the events where ALL trains fired at more
    than mfr_frac*mean_firing_rate inside window.

    Also removes 2nd trial when 2 trials are closer than min_iti ms from each other.
    
    Arguments:
        - events: array, timeseries in samples
        - trains: array, timeseries in samples
        - mfr_frac: [0-1], fraction of mean firing rate to exclude events
        - window: [left, right] floats, window of time in milliseconds
                  (by default, wide enough to sample 30 spikes given mean firing rate)
        - fs: int, sampling frequency of recording
        - min_isi: float, ISIs smaller than min_isi ms are ignored to compute mfr
        - max_isi: float, if an ISI larger than max_isi ms is in the trail, the trial is ignored.
        - min_iti: float, 2nd trial gets removed when 2 trials are closer than min_iti form each other.

    Returns:
        - events: subset of events where ALL trains fired at more
                  than mfr_frac*mean_firing_rate inside window.
        - mask: boolean mask of events (1 for selected events)
    
    """

    # formatting and argument compliance test
    
    if window is not None:
        assert window[0] < window[1], "Make sure window is [left, right] where left < right"
    assert npa(trains[0]).ndim >= 1,\
        "Make sure trains is provided as a list of timeseries ([train] if only one train)."
    assert np.any(events), "You provided an empty array of events!"
    assert events[0] == int(events[0]) and trains[0][0] == int(trains[0][0]),\
        "Make sure events and trains are provided in samples (integers)."

    events = npa(events)
    min_isi = min_isi * fs / 1000

    # iterate
    mask = np.ones(events.shape[0], dtype=bool)
    for t in trains:
        
        t = np.sort(t)
        mean_fr = mean_firing_rate(t, exclusion_quantile=0.005, fs=fs)
        if verbose: print(f"Mean fr: {mean_fr} (threshold {mean_fr*mfr_frac})")

        # if max_isi not provided, 50 times the mean firing rate (e.g. 500 for a 100Hz neuron).
        mx_isi = max_isi if max_isi is not None else (50 * 1000/mean_fr)
        mx_isi = mx_isi * fs / 1000

        # if window not provided, fit 30 spikes
        # [-50, 50] ms for 100Hz unit, [-30_000, 30_000] for 1Hz unit.
        win = window if window is not None else (1000 * 30 / npa([-mean_fr*2, mean_fr*2]))
        win = npa(win) * fs / 1000

        trial_rates = np.zeros(events.shape[0], dtype=float)
        for i, e in enumerate(events):
            ts = t - e # t shifted
            tsc = ts[(ts >= win[0]) & (ts <= win[1])] # t shifted clipped
            tscd = np.diff(tsc)
            tscd = tscd[tscd > min_isi]
            if not np.any(tscd) or np.any(tscd > mx_isi): continue
            trial_rates[i] = fs / np.mean(tscd)
        mask = mask & (trial_rates >= mean_fr * mfr_frac)

        # iti selection
        to_remove = np.append([False], (np.diff(events) < (min_iti * fs / 1000)))
        mask[to_remove] = False
        
    if verbose and mask.sum() >= 1:
        print(f"\nMean/min/max trial rates diff: {trial_rates.mean()}/{trial_rates.min()}/{trial_rates.max()}")
        print(f"Selected {mask.sum()}/{mask.shape[0]} events.")
        print(f"Mean/min/max selected trial rates diff: {trial_rates[mask].mean()}/{trial_rates[mask].min()}/{trial_rates[mask].max()}")

    return events[mask], mask


def get_ifr(times,
            events,
            b=2,
            window=[-1000,1000],
            remove_empty_trials=False,
            again=False):
    '''
    Arguments:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - ifr: a len(events) x window/b matrix where the spikes have been aligned, in Hertz.
    '''
    at, atb = align_times(times, events, b, window, remove_empty_trials, again)

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
                      bsl_subtract=False, bsl_window=[-4000, 0], process_y=False,
                      again=False):
    '''
    Returns the "processed" (averaged and/or smoothed and/or z-scored) instantaneous firing rate of a neuron.

    Arguments:
        - times:  list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b:      float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
        - convolve:      boolean, set to True to convolve the aligned binned train with a half-gaussian window to smooth the ifr
        - gsd:           float, gaussian window standard deviation in ms
        - method:        convolution window shape: gaussian, gaussian_causal, gamma | Default: gaussian
        - bsl_substract: whether to baseline substract the trace. Baseline is taken as the average of the baseline window bsl_window
        - bsl_window:    [t1,t2], window on which the baseline is computed, in ms -> used for zscore and for baseline subtraction (i.e. zscoring without dividing by standard deviation)
        - process_y:     whether to also process the raw trials x bins matrix y (returned raw by default)
    
    Returns:
        - x:       (n_bins,) array tiling bins, in milliseconds
        - y:       (n_trials, n_bins,) array, the unprocessed ifr (by default - can be processed if process_y is set to True)
        - y_p:     (n_bins,) array, the processed instantaneous firing rate (averaged and/or smoothed and/or z-scored)
        - y_p_var: (n_bins,) array, the standard deviation across trials of the processed instantaneous firing rate
    '''

    # Window and bins translation
    x = np.arange(window[0], window[1], b)
    y = get_ifr(times, events, b, window, remove_empty_trials=False, again=again)
    y_bsl = get_ifr(times, events, b, bsl_window, remove_empty_trials=False, again=again)
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


#%% Population synchrony analysis

def get_processed_popsync(trains, events, psthb=10, window=[-1000,1000],
                          events_tiling_frac=0.1, sync_win=2, fs=30000, t_end=None,
                          b=1, sd=1000, th=0.02,
                          again=False, dp=None, U=None,
                          zscore=False, zscoretype='within',
                          convolve=False, gsd=1, method='gaussian',
                          bsl_subtract=False, bsl_window=[-4000, 0], process_y=False):
    '''
    Arguments:
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

def decode_rotary(A, B, fs=5000, n_ticks=1024, diam=200, gsd=25, med_filt=True):
    '''Function to decode velocity from rotary encoder channels.

    Arguments:
        - a: np array, analog recording of channel A at sampling frequency fs
        - b: np array, analog recording of channel B at sampling frequency fs
        - fs: float (Hz), sampling frequency
        - n_ticks: int, number of ticks (periods) on rotary encoder (not number of thresholds (4x more))
        - diam: float (mm), outer diameter of wheel coupled to the encoder
        - gsd: float (ms), std of gaussian kernel (mandatory gaussian-causal smoothing)
        - med_filt: bool, whether to median filter on top of mandatory gaussian smoothing
    Returns:
        - speed: np.array, rotary speed in mm/s
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
    import cv2
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
    import cv2
    cap = cv2.VideoCapture(video_path)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert 0<=frame_i<totalFrames, 'Frame index too high!'
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_i)
    ret, frame = cap.read()
    if plot: plt.imshow(frame)
    return frame


#%% polar analysis

def cart2pol(x, y):
    '''
    Arguments:
    - x: float or np array, x coord. of vector end
    - y: float or np array, y coord. of vector end
    Returns: (r, theta) with
        - r: float or np array, vector norm
        - theta: float or np array, vector angle IN RADIANS
    '''
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta)

def pol2cart(r, theta):
    '''
    Arguments:
    - r: float or np array, vector norm
    - theta: float or np array, vector angle IN RADIANS
    Returns: (x, y) with
        - x: float or np array, x coord. of vector end
        - y: float or np array, y coord. of vector end
    '''
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)

def get_polar_vect(R, thetas, return_dsi=False):
    '''
    Arguments:
    - R:     np array, norms of vectors on a polar plot
    - thetas: np array, list of angles IN RADIANS associated with R values
    - return_dsi: bool, whether to normalize vector norm down to a direction selectivy index between [0-1]

    Returns: (vec_r, vec_th, base) with
        - vec_r: vector norm
        - vec_th: vector angle, in radians
        - base: origin of vector, 0 if R only has positive values
    '''

    assert assert_iterable(R)
    assert assert_iterable(thetas)
    assert len(R) == len(thetas)

    x, y          = pol2cart(R, thetas)

    vec_r, vec_th = cart2pol(np.sum(x), np.sum(y))

    amp_norm      = np.sqrt(np.sum(np.abs(x))**2 + np.sum(np.abs(y))**2)
    if not return_dsi: amp_norm = 1

    vec_r         = vec_r/amp_norm
    vec_th        = vec_th%(2*np.pi)

    return (vec_r, vec_th)


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