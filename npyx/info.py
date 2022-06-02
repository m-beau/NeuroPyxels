# Functions regarding information theoretic analysis of neural data

import numpy as np
from pyparsing import CaselessLiteral

from npyx.utils import smooth
from npyx.behav import get_processed_ifr


### average of pairwise synchronies i.e. P(a int b)/P(a)P(b)

def sync_wr_chance_shadmehr(t1, t2, events, binsize=2, window=[-500, 500],
                   remove_empty_trials=False, smooth_sd=4, pre_smooth=False, post_smooth=False,
                   y1=None, y2=None, return_trials = False, continuous=True, return_terms=False):
    """
    y1 and y2 should be T trials x B bins 2D matrices
    containing integer numbers of spikes (in theory only 0 and 1; any integer above 1 will be reduced to 1.)

    Relates to correlation as follows:
    Corr(X,Y) = Cov(X,Y)/std(X)std(Y)
              = Cov(X,Y)/sqrt(Var(X)Var(Y))
    Cov(X,Y) = E(XY) - E(X)E(Y)
             = P(X int Y) - P(X)P(Y)
    Var(X) = E(X**2) - E(X)**2
           = P(X) - P(X)P(X) because X is binary
           = P(X)(1-P(X))

    So Corr(X,Y) = (P(X int Y) - P(X)P(Y)) / sqrt(P(X)(1-P(X))P(Y)(1-P(Y)))
                 = (P(X int Y)/P(X)P(Y) - 1) / (sqrt(P(X)P(Y))*sqrt((1-P(X))(1-P(Y))) / P(X)P(Y))
                 = (sync(X,Y) - 1) / sqrt( (1-P(X))(1-P(Y)) / P(X)P(Y) )
                 = (sync(X,Y) - 1) / sqrt( 1/(1-P(X))(1-P(Y)) - 1 )
    """
    if y1 is None:
        x, y1, y_p1, y_p_var1 = get_processed_ifr(t1, events, b=binsize, window=window, zscore=False, convolve=0)
        y1 = (y1*binsize/1000).astype(np.int64)
    if y2 is None:
        x, y2, y_p2, y_p_var2 = get_processed_ifr(t2, events, b=binsize, window=window, zscore=False, convolve=0)
        y2 = (y2*binsize/1000).astype(np.int64)

    trial_ids = np.arange(y1.shape[0])
    cell_1_active_m = (np.mean(y1, axis=1) > 0)
    cell_2_active_m = (np.mean(y2, axis=1) > 0)
    cells_active_m = cell_1_active_m & cell_2_active_m
    if cells_active_m.sum()<20:
        print("WARNING two neurons share less than 20 trials in common. Cannot estimate synchrony.")
        if return_trials:
            y1.mean(axis=0)*np.nan, y1*np.nan, trial_ids[cells_active_m]
        return y1.mean(axis=0)*np.nan
    if remove_empty_trials:
        y1 = y1[cells_active_m, :]
        y2 = y2[cells_active_m, :]
        trial_ids = trial_ids[cells_active_m]

    if pre_smooth:
        if continuous:
            y1 = smooth(y1, 'gaussian_causal', smooth_sd)
            y2 = smooth(y2, 'gaussian_causal', smooth_sd)
        else:
            print('WARNING no pre-smoothing as arrays must be binary. Set continuous=True if you wish to pre smooth.')

    if continuous:
        # Max = 1
        y1[y1>1]=1
        y2[y2>1]=1

        # P(A,B)
        p_12_trials = y1*y2
        p_12 = np.mean(p_12_trials, 0)

        # P(A), P(B)
        p1 = np.mean(y1, 0)
        p2 = np.mean(y2, 0)

    else:
        # Binarize spike presence
        y1 = y1.astype(bool)
        y2 = y2.astype(bool)

        # P(A,B)
        p_12_trials = (y1 & y2).astype(int)
        p_12 = np.mean(p_12_trials, 0)

        # P(A), P(B)
        p1 = np.mean(y1.astype(int), 0)
        p2 = np.mean(y2.astype(int), 0)

    # P(A,B) / (P(A) x P(B))
    sync = p_12/(p1*p2)

    if post_smooth:
        sync = smooth(sync, 'gaussian_causal', smooth_sd)

    sync[np.isnan(sync)]=1

    if return_trials:
        if return_terms:
            ret = (sync, p_12_trials/(p1*p2), p_12_trials, p1, p2, trial_ids)
        else:
            ret = (sync, p_12_trials/(p1*p2), trial_ids)
    else:
        if return_terms:
            ret = (sync, p_12_trials, p1, p2)
        else:
            ret = sync

    return ret
    

def compute_sync_matrix(signal):
    """Compute the P(A int B)/P(A)P(B) for each neuron A and B,
    output is a N x N matrix.

    Works for continuous values [0-1] (does compute A&B, rather A*B: 11->1, 01->0, but .5 1 -> .5)
    """
    T, N = signal.shape
    neuron_fire_rate = signal.mean(axis=0)
    normalised_signal = signal / neuron_fire_rate
    sync_matrix_with_diag = normalised_signal.T.dot(normalised_signal) / T
    # simply get rid of diagonal (neurons with themselves)
    sync_matrix = sync_matrix_with_diag - np.diag(np.diag(sync_matrix_with_diag))
    return sync_matrix

def l2_synchrony(signal):
    T, N = signal.shape
    sync_matrix = compute_sync_matrix(signal)
    return np.sqrt((sync_matrix**2).sum()/(N*(N-1)))

def avg_synchrony(signal):
    T, N = signal.shape
    sync_matrix = compute_sync_matrix(signal)
    return sync_matrix.sum()/(N*(N-1))

# def two_synchrony(signal):
#     nb_neurons_active = signal.sum(axis=-1)
#     two_neurons_active = nb_neurons_active > 2
#     sync_gross = two_neurons_active.mean(axis=-1)
#     neuron_rate = signal.mean(axis=-2)
#     my_binary = int_to_binary(array, N)

def total_synchrony(signal, normalize=False, return_trials=False):
    """
    Signal: B x T x N tensor
    Computes ratio of N_neurons_active/N_neurons_active_if_were_independant.

    close to P Latham suggestion (sum across neurons).

    Limited between 0 and 1 if normalized, 0 and N if not.
    """
    N = signal.shape[-1]
    # Z = sum_i(Xi) with i=[1-N] neurons: that's the number of activated neurons = synchrony
    # total synchrony = ratio between Z**2 and Z**2 if neurons were independant
    # (we use Z**2 because it exaggerates differences in number of activated neurons)

    # CalcuLate E(Z**2)
    #     Var(Z) = E(Z**2) - (E(Z))**2 (definition of variance)
    # <=> E(Z**2) = Var(Z) + (E(Z))**2
    #             = Var(sum_i(Xi)) + E(sum_i(Xi))**2
    Z2_E = (signal.sum(axis=-1)**2).mean(axis=-1)
    # in theory same as Z2 = signal.sum(axis=-1).var(axis=-1) + signal.mean(axis=-2).sum(axis=-1)**2
    
    # If Xi are independant, Var(sum_i(Xi)) = sum_i(Var(Xi))
    # so E(Z**2) = Var(sum_i(Xi)) + E(sum_i(Xi))**2
    #            = sum_i(Var(Xi)) + E(sum_i(Xi))**2
    Z2_E_indep = signal.var(axis=-2).sum(axis=-1) + signal.mean(axis=-2).sum(axis=-1)**2

    # E(Z**2)/E_indep(Z**2)
    tot_sync = Z2_E/Z2_E_indep
    if normalize: tot_sync /= N

    if return_trials:
        # directly calculate Z**2, not E(Z**2) across trials
        Z2_trials = (signal.sum(axis=-1)**2)
        tot_sync_trials = Z2_trials/Z2_E_indep
        if normalize: tot_sync_trials /= N
        return tot_sync, tot_sync_trials

    return tot_sync

def total_var_synchrony(signal, normalize=True):
    """
    Signal: B x T x N tensor
    Same approach as total_synchrony (ratio between 2 value with numerator = true data and denominator = indep data),
    but using variances instead of means of squared data.
    Limited between 0 and 1 if normalized, 0 and N if not.

    Source: www.scholarpedia.org/article/Neuronal_synchrony_measures
    (written by David Golomb, citable publication is Golomb abd Rinzel 1993, Coherence metric; Ginzburg and Sompolinsky 1994.)
    but replace Vi(t) voltage of neuron i at time t by Pi(b) probability of spike of neuron i in bin b
    """

    # Z = sum_i(Xi) with i=[1-N] neurons: that's the number of activated neurons = synchrony

    Z_V = signal.sum(axis=-1).var(axis=-1)
    # same as Z_V = (signal.sum(axis=-1)**2).mean(axis=-1) - signal.sum(axis=-1).mean(axis=-1)**2
    
    # If Xi are independant, Var(sum_i(Xi)) = sum_i(Var(Xi))
    # so E(Z**2) = Var(sum_i(Xi)) + E(sum_i(Xi))**2
    #            = sum_i(Var(Xi)) + E(sum_i(Xi))**2
    Z_V_indep = signal.var(axis=-2).sum(axis=-1)

    # E(Z**2)/E_indep(Z**2)
    tot_var_sync = Z_V/Z_V_indep

    tot_var_sync = Z_V/Z_V_indep
    if normalize:
        N = signal.shape[-1]
        tot_var_sync /= N

    return np.sqrt(tot_var_sync)

def mgf_synchrony(signal, lam=2, normalized=True):
    """
    Signal: B x T x N matrix
    Computes moment generating function synchrony across N neurons.
    """
    assert lam>1, "WARNING lambda must be strictly superior to 1."
    # variance across trials (T) of the sum across neurons (N)
    mgf_synchrony = (lam**(signal.sum(axis=-1))).mean(axis=-1)
    if normalized:
        mgf_synchrony /= (lam**signal).mean(axis=-2).prod(axis=-1)

    return mgf_synchrony


### crosscorrelation between neuron population activation
### and target neuron

def lagged_synchrony_analysis(signal, target, lags, res=1):
    synchronic_signal = more_than_n_neurons_active(signal, res)
    return lagged_correlations(synchronic_signal, target, lags)

def more_than_n_neurons_active(signal, res=1):
    nb_neurons = signal.shape[-1]
    nb_active = signal.sum(axis=-1)
    L = []
    for n in np.arange(0,nb_neurons,res):
        L.append(nb_active > n)
    return np.stack(L, axis=-1)

def lagged_correlations(signal, target, lags, axis=1):
    L = [lagged_correlation(signal, target, lag, axis=axis) for lag in lags]
    return np.stack(L, axis=1)

def lagged_correlation(signal, target, lag, axis=1):
    target = np.expand_dims(target, axis=-1)
    target_lagged = np.roll(target, -lag, axis=1)
    cor = correlation(signal, target_lagged, axis=axis)
    return cor

def correlation(x, y, axis=1):
    cov = (x * y).mean(axis=axis) - x.mean(axis=axis) * y.mean(axis=axis)
    sdx = x.std(axis=axis)
    #sdx = np.nan_to_num(sdx, nan=1)
    sdy = y.std(axis=axis)
    #sdy =np.nan_to_num(sdy, nan=1)
    cor = cov / (sdx * sdy)

    return cor



### mutual information between population of PCs and NC
def multivariate_mutual_information(signal_x, signal_y):
    """
    signal_x is a B x T x N matrix
    signal_y is a B x T matrix
    """
    p_joint_x = compute_p_joint(signal_x).mean(axis=-1).T
    H_joint_x = entropy(p_joint_x)

    p_y = signal_y.mean(axis=-1)
    H_y = - p_y * cut_log(p_y) - (1-p_y) * cut_log(1-p_y)

    signal_xy = np.concatenate([signal_x, signal_y[:,:,None]], axis=-1)
    p_joint_xy = compute_p_joint(signal_xy).mean(axis=-1).T
    H_joint_xy = entropy(p_joint_xy)

    return (H_joint_x + H_y - H_joint_xy)/(H_joint_x + H_y)


### Total correlation i.e. generalization of mutual information
### to find consistent patterns of codependance across N neurons

def total_correlation(signal, normalized=True):
    """
    Generalization to N variables of mutual information (for 2 variables, same thing).

    i.e. KL divergence from the joint distribution P(X1,...,Xn)
    to the product (=independant) distribution P(X1)*...*P(Xn).

    If there are only 2 variables, this corresponds to the mutual information.

    This reduces to the simpler difference of entropies,
    sum_over_i(H(Xi)) - H(X1,...,Xn)
    
    Returns the total_correlation normalised between 0 and 1
    for a B x T x N or T x N signal of probabilities between 0 and 1.
    
    If normalize=True, normalized between 0 and 1."""
    # get joint entropy
    # that is across neurons
    p_joint = compute_p_joint(signal)
    p_joint_mean = p_joint.mean(axis=-1).T
    joint_entropy = entropy(p_joint_mean)

    neuron_fire_rate = signal.mean(axis=-2)
    neuron_entropies = - neuron_fire_rate * cut_log(neuron_fire_rate) - (1-neuron_fire_rate) * cut_log(1-neuron_fire_rate)

    # this is equivalent to kullback_leibler(p_joint, p_product)
    C = neuron_entropies.sum(axis=-1) - joint_entropy

    if normalized:
        C_max = neuron_entropies.sum(axis=-1) - neuron_entropies.max(axis=-1)
        C /= C_max
    
    return C

def mutual_information(signal):
    """KL divergence from the joint distribution P(X1,...,Xn)
    to the product (=independant) distribution P(X1)*...*P(Xn)
    
    returns the mutual information for a B x T x N or T x N signal of probabilities between 0 and 1"""
    p_joint = compute_p_joint(signal)
    p_prod = compute_p_prod(signal)
    KL = kullback_leibler(p_joint.mean(axis=-1).T, p_prod.mean(axis=-1).T)
    return KL

def multivariate_copula(signal, normalized=True):
    """Compute the proba Q such that P(X1, ..., XN) ~= Q(X1, ..., XN)P(X1)...P(XN).
    
    If normalize=True,
    return H(Q)/H(Uniform) with Q the copula of the signal
    (divergence between Q and the uniform distribution).
    Value is 0 if independant, 1 if all neurons always fire the same.
    """
    p_joint = compute_p_joint(signal).mean(axis=-1).T
    p_prod = compute_p_prod2(signal)
    q = p_joint / p_prod
    q = q / q.sum(axis=-1, keepdims=True)

    if normalized:
        # divergence between copula and the uniform distribution
        # i.e. ratio of their entropies
        N = signal.shape[-1]
        q = 1 - (entropy(q) / entropy(np.ones(2**N)/2**N))

    return q



### basic information theoretic functions ###

def compute_p_joint(signal):
    """returns the product proba of the configuration: p_joint(config),
    output an array of size 2^N.

    p_joint is of shape (B, 2^N)
    where 2^N corresponds to all combinations of neurons: 0000, 0001, 0010, 0011...
    so p_joint[:,-1] corresponds to the joint probability taking all neurons into consideration
    (P(X1 & X1 & ... & Xn))
    """

    # illustrative example:
    # np.all(p_joint[-1,:,:] == np.logical_and.reduce(M, axis=-1))

    binaries = array_of_all_binaries(signal.shape)
    p_joint = equivalence_measure(signal, binaries)
    return p_joint

def compute_p_prod(signal):
    """returns the proba of the configuration: p_product(config),
    output an array of size 2^N.
    """
    binaries = array_of_all_binaries(signal.shape)
    neuron_fire_rate = signal.mean(axis=-2, keepdims=True)
    p_prod = equivalence_measure(neuron_fire_rate, binaries)
    return p_prod

def compute_p_prod2(signal):
    """returns the proba of the configuration: p_joint(config),
    output an array of size (B,2^N)
    (2^N is the number of combinations of N neurons, for instance 32 for 5 neurons).
    """
    N = signal.shape[-1]
    integers = np.arange(2**N).reshape((2**N, 1))
    binaries = int_to_binary(integers, N)
    neuron_fire_rate = signal.mean(axis=-2, keepdims=True)
    p_prod = equivalence_measure(neuron_fire_rate, binaries)
    return p_prod

def kullback_leibler(p, q):
    """Compute the KL divergence of two probability measures, given by arrays of the same size that sum to one."""
    log_ratio = np.log(p / q, out=np.zeros(p.shape), where=(p != 0))
    KL = (p * log_ratio).sum(axis=-1)
    return KL

def entropy(p, axis=-1):
    return -(p * cut_log(p)).sum(axis=axis)



### utility functions ###

def array_of_all_binaries(signal_shape):
    """returns an array with all binaries in {0, 1}^N, 
    with shape (2^N, 1, ..., 1, N) with the same nb of ones as the dimensions of the signal tensor minus one.
    """
    N = signal_shape[-1]
    shape = broadcastable_shape(2**N, len(signal_shape))
    integers = np.arange(2**N).reshape(shape)
    binaries = int_to_binary(integers, N)
    return binaries

def broadcastable_shape(m, n):
    return (m, *(n*[1]))

def int_to_binary(array, N):
    """A dimension is added to a tensor of integers, such that the last dimension gives the binary decomposition,
    output a tensor with one more dimension of size N.
    """
    indexes = np.arange(N)
    res = (array // 2**indexes) % 2
    return res

def equivalence_measure(p, q):
    """Give the probability that the two measures on {0,1}^N are equal,
    with discrete binaries, it is equivalent to p == q,
    output an tensor of the size broadcasted from the two, minus the last.
    """
    return (p*q + (1-p)*(1-q)).prod(axis=-1)

def cut_log(p):
    """returns log p if p > 0, else 0."""
    return np.log(p, out=np.zeros(p.shape), where=(p > 0))


#####################

def residual_cv2(t, bin, win, events):
    '''

    Returns:
        - cv2_residuals: nevents x time matrix np array
    '''

    

    # Generate events matrix


    # Subtract mean from events matrix

    return

def Paintb_PaPb(y1, y2):
    """
    y1 and y2 should be T x B matrices (T trials and B bins)
    """

    # Binarize spike presence
    y1_bool = y1.astype(bool)
    y2_bool = y2.astype(bool)

    # P(A,B)
    p_12 = np.mean((y1_bool & y2_bool).astype(int), 0)

    # P(A), P(B)
    p1 = np.mean((y1_bool).astype(int), 0)
    p2 = np.mean((y2_bool).astype(int), 0)

    # P(A,B) / (P(A) x P(B))
    sync = p_12/(p1*p2)

    return sync