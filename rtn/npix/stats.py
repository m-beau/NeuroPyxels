
# -*- coding: utf-8 -*-
"""
2018-07-20
@author: Maxime Beau, Neural Computations Lab, University College London

Inferential statistics tools.
"""

#%% Imports
'''https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/'''

from numpy.random import seed, randn
import sp.stats as stt
import statsmodels as sttmdl

'''Reminder: Observations < Samples < Population'''

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
        D, Pval = lillifors(data)
    elif test=='ktest':
        D, Pval = stt.kstest(x, 'norm')
    elif test=='anderson':
        D, Crit_vals, Pval = stt.anderson(x)
        
    # Tests based on descriptive statistics of the sample
    elif test=='agostino':
        D, Pval = stt.normaltest(x)

    print('STATCHECK: NORMALITY - statistics={}, Pval={} with {} test.'.format(D, Pval, test))
    if QQplot:
        stats.probplot(x, dist="norm", plot=plt)
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

