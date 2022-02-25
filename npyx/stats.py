
# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London

Inferential statistics tools.
"""

#%% Imports
'''https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/'''

import numpy as np
import math
from numpy.random import seed, randn
import scipy.stats as stt

from npyx.utils import npa

'''Reminder: Observations < Samples < Population'''

#%% Laws

def pdf_normal(X, m=0, s=1):
    'Normal probability density function.'
    X=npa([X]).flatten()
    # try:
    #     return np.exp(-1./2*((X-m)/s)**2)/(s*np.sqrt(2*np.pi))
    # except:
    return stt.norm.pdf(X, m, s)

def pdf_poisson(X, l=1):
    'Poisson probability density function.'
    X=npa([X]).flatten()
    # p=npa(zeros=X.shape)
    # try:
    #     for i, x in enumerate(X):
    #         assert x>=0 and round(x)==x, "A Poisson law only accepts natural integers as inputs!"
    #         p[i]=np.exp(-l)*l**x/math.factorial(x)
    #     return p
    # except:
    return stt.poisson.pmf(X, l)

def cdf(X, _pdf, w1, b, *args):
    X=npa([X]).flatten()
    p=npa(zeros=X.shape)
    for i, x in enumerate(X):
        p[i]=np.sum(_pdf(np.arange(w1, x+b, b), *args)*b)
    return p

def cdf_normal(X, m=0, s=1):
    'Normal probability cumulative function.'
    #return cdf(X, pdf_normal, m-4*s, s/100, m, s)
    return stt.norm.cdf(X, m, s)

def cdf_poisson(X, l=1):
    'Poisson probability cumulative function.'
    #return cdf(X, pdf_poisson, 0, 1, l)
    return stt.poisson.cdf(X, l)

def fractile(p, _cdf, w1, w2, b, *args):
    X=np.arange(w1,w2+b,b)
    return np.round(w1+np.nonzero(_cdf(X, *args)>=p)[0][0]*b, 4)

def fractile_normal(p=0.975, m=0, s=1):
    '''Fractile of order p drawn from the normal cumulative probability density function.
    Ex: fractile_normal(0.975) will be 1.98 (the normal cumulative pdf reaches 0.975 for x=1.96.)
    So 95% of samples of a normal distribution will be between mean+/-1.96*std.
    '''
    w1,w2,b=m-4*s, m+4*s, s/100
    return fractile(p, cdf_normal, w1, w2, b, m, s)

def fractile_poisson(p=0.975, l=1):
    '''Same for Poisson distribution.
    '''
    w1,w2,b=0, l+3*l, 1
    return fractile(p, cdf_poisson, w1, w2, b, l)

#%% A] Statistical tests assumptions checkpoints: 1) no outliers, 2) normal distribution
def check_outliers(x, th_sd=2, remove=False):
    '''x: 1D numpy array
       th_sd: threshold in standard devioations out of it we detect outliers
       remove: boolean, if yes returns x without outliers, if false returns x unchanged.
       IF OUTLIERS, check at the raw data, and either
           1) detect the underlying problem and exclude them 
           2) you cannot find the problem, VERY VALUABLE DATA POINTS - FORBIDDEN TO EXCLUDE THEM.'''
    x = np.asarray(x)
    assert x.ndim==1
    
    mn = np.mean(x)
    sd = np.std(x)
    
    outliers = [i for i in x if ((i>mn+th_sd*sd) or (i<mn-th_sd*sd))]
    print('STATCHECK: OUTLIERS - Detected {} outliers (threshold: {}s.d.).'.format(len(outliers), th_sd))
    
    return x[~np.isin(x, outliers)] if remove else len(outliers)

def check_normality(x, test='shapiro', qqplot=True):
    '''ASSUMPTIONS of 4 tests: Observations in each sample are independent and identically distributed (iid).'''
    x = np.asarray(x)
    assert x.ndim==1
    assert test in ['shapiro', 'lillifors', 'agostino', 'anderson', 'kstest']
    
    # Tests of best fit with a given distribution
    if test=='shapiro':
        D, Pval = stt.shapiro(x)
    if test=='lillifors':
        D, Pval = stt.lillifors(x)
    elif test=='ktest':
        D, Pval = stt.kstest(x, 'norm')
    elif test=='anderson':
        D, Crit_vals, Pval = stt.anderson(x)
        
    # Tests based on descriptive statistics of the sample
    elif test=='agostino':
        D, Pval = stt.normaltest(x)

    print('STATCHECK: NORMALITY - statistics={}, Pval={} with {} test.'.format(D, Pval, test))
    if qqplot:
        stt.probplot(x, dist="norm")
    return D, Pval

def check_eqVariances(*args):
    '''Levene test - to do notably before ANOVA.'''
    (W,p) = stt.levene(*args)
    return W, p

#%% B] Correlation tests: check if two samples are related.

#### Pearson's correlation coefficient
# ->> TEST WHETHER TWO SAMPLES HAVE A LINEAR RELATIONSHIP
# <- Assumptions on observations in each sample: 1) iid 2) normal distribution 3) same variance
def corrTest_pearson(data1, data2):
    '''->> TEST WHETHER TWO SAMPLES HAVE A LINEAR RELATIONSHIP
        <- Assumptions on observations in each sample: 1) iid 2) normal distribution 3) same variance'''
    corr, p = stt.pearsonr(data1, data2)
    return corr, p

"""
#### Spearman's rank correlation
# ->> TEST WHETHER TWO SAMPLES HAVE A MONOTONIC RELATIONSHIP
# <- Assumptions on observations in each sample: 1) iid, 2) observations can be ranked in each sample
from scipy.stats import spearmanr
data1, data2 = ...
corr, p = spearmanr(data1, data2)

#### Kendall's rank correlation
# ->> TEST WHETHER TWO SAMPLES HAVE A MONOTONIC RELATIONSHIP
# <- Assumptions on observations in each sample: 1) iid, 2) observations can be ranked in each sample
from scipy.stats import kendalltau
data1, data2 = ...
corr, p = kendalltau(data1, data2)

#### Chi-squared test
# ->> CHECK AT INDEPENDANCE BETWEEN TWO CATEGORICAL VARIABLES
# <- Assumptions on observations in each sample: 1) independance, 2) N>25 in each cell of the contingency table
from scipy.stats import chi2_contingency
table = ...
stat, p, dof, expected = chi2_contingency(table)

#%% C] Parametric statistical hypothesis tests

#### Student’s t-test
# ->> TEST WHETHER TWO SAMPLES HAVE SIGNIFICANTLY DIFFERENT MEANS
# <- Assumptions on observations in each sample: 1) iid, 2) Normal distribution, 3) Same variance
from scipy.stats import ttest_ind
data1, data2 = ...
stat, p = ttest_ind(data1, data2)

#### Paired Student’s t-test
# ->> TEST WHETHER TWO PAIRED SAMPLES HAVE SIGNIFICANTLY DIFFERENT MEANS
# <- Assumptions on observations in each sample: 1) iid, 2) Normal distribution, 3) Same variance, 4) paired
from scipy.stats import ttest_rel
data1, data2 = ...
stat, p = ttest_rel(data1, data2) # equivalent to ttest_1samp(data1-data2, 0) !!

#### Analysis of Variance Test (ANOVA)
# ->> TEST WHETHER TWO OR MORE SAMPLES HAVE SIGNIFICANTLY DIFFERENT MEANS
# <- Assumptions on observations in each sample: 1) iid, 2) Normal distribution, 3) Same variance
from scipy.stats import f_oneway
data1, data2, ... = ...
stat, p = f_oneway(data1, data2, ...)

#### Repeated Measures ANOVA Test
# ->> TEST WHETHER TWO OR MORE PAIRED SAMPLES HAVE SIGNIFICANTLY DIFFERENT MEANS
# <- Assumptions on observations in each sample: 1) iid, 2) Normal distribution, 3) Same variance, 4) paired
    
#%% D] Non parametric statistical hypothesis tests

#### Mann-Whitney U Test
# ->> TEST WHETHER TWO SAMPLES HAVE SIGNIFICANTLY DIFFERENT DISTRIBUTIONS
# <- Assumptions on observations in each sample: 1) iid, 2) can be ranked
from scipy.stats import mannwhitneyu
data1, data2 = ...
stat, p = mannwhitneyu(data1, data2)

#### Wilcoxon Signed-Rank Test
# ->> TEST WHETHER TWO PAIRED SAMPLES HAVE SIGNIFICANTLY DIFFERENT DISTRIBUTIONS
# <- Assumptions on observations in each sample: 1) iid, 2) can be ranked, 3) paired
from scipy.stats import wilcoxon
data1, data2 = ...
stat, p = wilcoxon(data1, data2)

#### Kruskal-Wallis H Test
# ->> TEST WHETHER TWO OR MORE SAMPLES HAVE SIGNIFICANTLY DIFFERENT DISTRIBUTIONS
# <- Assumptions on observations in each sample: 1) iid, 2) can be ranked
from scipy.stats import kruskal
data1, data2, ... = ...
stat, p = kruskal(data1, data2, ...)

#### Friedman Test
# ->> TEST WHETHER TWO OR MORE PAIRED SAMPLES HAVE SIGNIFICANTLY DIFFERENT DISTRIBUTIONS
# <- Assumptions on observations in each sample: 1) iid, 2) can be ranked, 3) paired
from scipy.stats import friedmanchisquare
data1, data2, ... = ...
stat, p = friedmanchisquare(data1, data2, ...)

#%% E] Pval correction for repeated comparisons







#%% MODERN STATISTICS: NO MORE TESTS BUT DATA MODELLING

"""




#%% Extract subsample of distribution
# (developed to extract subsets of interspike intervals for GRC2019)

def get_all_up_to_median(arr, window_a, window_b, hbin):
    med_idx=np.median(arr)
    return arr<(window_a+(med_idx*hbin))

def get_half_centered_on_mode(arr, window_a, window_b, hbin):
    '''
    Splits a distribution in two parts with equal AUC, centered on the mode of the distribution.
    Parameters:
        - arr: array from which distribution will be drawn,
        between window_a and window_b with bin hbin
        - window_a, window_b, hbin: cf. above
    Returns:
        - Boolean array sized as the input array, with True values for indices around the mode.
    '''
    hist=np.histogram(arr, bins=np.arange(window_a, window_b, hbin))
    
    mode_idx=np.nonzero(hist[0]==max(hist[0]))[0][0]
    AUC=sum(hbin*hist[0])
    win=0
    for win in np.arange(0, mode_idx, 1):# win centered on mode
        AUC_win=sum(hbin*hist[0][mode_idx-win:mode_idx+win])
        if (AUC_win/AUC)>=0.5:
            break
    
    th_a=window_a+(mode_idx-win)*hbin
    th_b=window_a+(mode_idx+win)*hbin
    
    return (arr>th_a)&(arr<th_b)

def split_distr_N(arr, N, bin_val, window_a=None, window_b=None, windows=None, equalAUC=False, a_pc=10, b_cp=70):
    '''
    Splits a distribution in N parts with equal bin window (and optionally equal AUC).
    (the AUC of all the chunks is set the the AUC of the chunk with smallest AUC).
    Parameters:
        - arr: array from which distribution will be drawn,
        between window_a and window_b with bin hbin
        - N: integer, number of chunks the distribution will be split in
        - bin_val: value that will be used to compute the histogram later on - used to ensure that windows are multiples of bins
        - window_a, window_b: cf. above. If None, will take 15th and 85th percentiles. | Default: None
    Returns:
        - list of N boolean masks, each corresponding to a distribution chunk.
        - list of windows used to 
    '''
    assert type(N) in [int, np.int]
    if windows is None:
        if window_a is None or window_b is None:
            window_a=np.percentile(arr, a_pc)
            window_b=np.percentile(arr, b_cp)
        window_a=np.round(window_a-window_a%bin_val, 2)
        window_b=np.round(window_b+window_b%bin_val, 2)
        N_wins=npa([[window_a+i*(window_b-window_a)/N, window_a+(i+1)*(window_b-window_a)/N] for i in range(N)])
    else:
        N_wins=windows
    N_masks=npa([(arr>=w1)&(arr<w2) for (w1,w2) in N_wins])
    
    if equalAUC:
        min_1=min([np.count_nonzero(m) for m in N_masks])
        for i,m in enumerate(N_masks):
            m_1=np.nonzero(m)[0]
            N_masks[i]=np.isin(np.arange(len(m)), m_1[np.random.choice(range(len(m_1)), min_1, replace=False)])
            
    return N_masks, N_wins

#%% Extract time stamps with given properties
# (developed to extract subsets of spikes for GRC 2019)

def get_isolated_stamps(t1, isolation_halfwin=120):
    '''
    Returns elements of t1 surrounded by 2 intervals >= isolation_halfwin
    '''
    if isolation_halfwin==0:
        return t1
    t1=npa(t1)
    isi1=np.diff(t1)
    mask_forward=(isi1>=isolation_halfwin)
    mask_backward=(isi1[::-1]>=isolation_halfwin)[::-1]
    t1=t1[1:-1][mask_forward[1:]&mask_backward[:-1]]
    
    return t1

def get_synced_stamps(t1, t2, sync_halfwin=60, isolation_halfwin=0, return_isis=False):
    '''Returns array t1, time stamps of two time X synchronous with Y.
       A stamp of a series is 'synchronous' if it happens within t+/-isolation_halfwin.
       
       t1, t2: time series (numpy arrays, in samples)
       sync_halfwin: integer, tolerance window to consider a stamp as "shared". In samples.
       isolation_halfwin: integer, only spikes from t1 surrounded by 2 intervals >=isolation_halfwin will be picked. Default: 0 (no effect)
       
       return t1_sync, t1_unsync where t1_sync are the sync spikes and t1_unsync a uniform random sampling from isolated t1 spikes not in t1_sync to match the size of t1_sync.
       '''
    # Get isolated spikes from train 1 only
    t1, t2 = npa(t1), npa(t2)
    t1=get_isolated_stamps(t1, isolation_halfwin)
    
    # Merge series1 and 2 and sort them + compute their ISIs
    t_12=np.append(t1, t2)
    i_12=np.append(1*np.ones((len(t1))),2*np.ones((len(t2))))
    t_i_12=np.vstack((t_12, i_12))
    
    # Sort accordingly to time stamps
    t_i_12=t_i_12[:,t_i_12[0,:].argsort()]
    isi_i_12=np.zeros((2, t_i_12.shape[1]-1)) # isi: inter stamp interval
    isi_i_12[0,:]=t_i_12[0,1:]-t_i_12[0,:-1]
    isi_i_12[1,:]=t_i_12[1,:-1]
    
    # Get spikes where the isi with the closest spike 2 is smaller than sync_halfwin
    mask_f=(isi_i_12[1,:-1]==1)&(isi_i_12[1,1:]==2)&(isi_i_12[0,:-1]<=sync_halfwin) # gets stamp of 1 followed by a 2 with isi smaller than sync_halfwin.
    mask_r=(isi_i_12[1,:-1]==2)&(isi_i_12[1,1:]==1)&(isi_i_12[0,:-1]<=sync_halfwin) # gets stamp of 2 followed by a 1 with isi smaller than halfwin - interested by index i+1 (stamps 1)
    mask_r[1:]=mask_r[:-1]; mask_r[0]=False # i are stamp 2, i+1 are stamp 1. i positions take value of i+1 because we want stamp 1.
    
    isi1_sync=np.append(isi_i_12[0,:-1][mask_f], isi_i_12[0,:-2][mask_f[:-1]])
    t1_sync=t_i_12[0,:-2][mask_f|mask_r] # ISIs where you have a consecutive [1,2] as index: spikes of 1 followed by 2
    assert np.all(np.isin(t1_sync, t1))
    
    if np.any(t1[~np.isin(t1, t1_sync)]):
        t1_unsync=np.random.choice(t1[~np.isin(t1, t1_sync)], len(t1_sync))
    else:
        t1_unsync=npa([])
    
    if return_isis:
        return t1_sync, t1_unsync, isi1_sync
    
    return  t1_sync, t1_unsync

def get_CIH(spk1, spk2):
    '''WARNING direction matters - will return CIH of 1 to 2, not 2 to 1.
    
    spk1 and spk2 are 2 time series.
    
    returns (spk_1to2, isi_1to2)
    '''
    # Merge spikes from 1 and 2 and sort them + compute their ISIs
    spk_12=np.append(spk1, spk2)
    i_12=np.append(1*np.ones((len(spk1))),2*np.ones((len(spk2))))
    spk_i_12=np.vstack((spk_12, i_12))
    
    # Sort accordingly to spike times
    spk_i_12[1, :]=spk_i_12[1, :][np.argsort(spk_i_12[0,:])]
    spk_i_12[0, :]=spk_i_12[0, :][np.argsort(spk_i_12[0,:])]
    isi_i_12=np.zeros((2, spk_i_12.shape[1]-1))
    isi_i_12[0,:]=spk_i_12[0,1:]-spk_i_12[0,:-1]
    isi_i_12[1,:]=spk_i_12[1,:-1]
    
    # Get ISIs of consecutive 1->2 spikes
    isi_1to2=isi_i_12[0,:-1][(isi_i_12[1,:-1]==1)&(isi_i_12[1,1:]==2)]
    spk_1to2=spk_i_12[0,:-2][(isi_i_12[1,:-1]==1)&(isi_i_12[1,1:]==2)] # ISIs where you have a consecutive [1,2] as index: spikes of 1 followed by 2
    assert np.all(np.isin(spk_1to2, spk1))
    
    return  spk_1to2, isi_1to2
