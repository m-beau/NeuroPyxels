## Code borrowed from Allen institute spikemetrics (jan 2022):
## https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/quality_metrics

import hashlib
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def quality_metrics(dp, units=None, params=None, save=True, again=False):
    """
    Wrapper of calculate_quality_metrics to easily run on a kilosort directory.
    
    """
    # preprocess parameters
    dp = Path(dp)
    if params is None:
        params = {
            "isi_threshold" : 0.0015,
            "min_isi" : 0.000166,
            "num_channels_to_compare" : 4,
            "max_spikes_for_unit" : 500,
            "max_spikes_for_nn" : 10000,
            "n_neighbors" : 4,
            'n_silhouette' : 10000,
            "drift_metrics_interval_s" : 51,
            "drift_metrics_min_spikes_per_interval" : 10,
            "include_pc_metrics" : True,
            }

    params_keys = ['isi_threshold', 'min_isi', 'num_channels_to_compare',
                   'max_spikes_for_unit', 'max_spikes_for_nn', 'n_neighbors',
                   'n_silhouette']
    params_str = str([params[k] for k in params_keys]).replace(' ', '')

    spike_clusters = np.load(dp / 'spike_clusters.npy').squeeze()
    if units is not None:
        units = list(np.sort(units).astype(int))
        units_hash = hashlib.sha1(str(units).encode()).hexdigest()[:20]
    else:
        units_hash = hashlib.sha1(str(np.unique(spike_clusters)).encode()).hexdigest()[:20]

    fn = f"metrics_{params_str}_{units_hash}.csv"

    # Load metrics if found
    if (dp/fn).exists() and not again:
        return pd.read_csv(dp/fn, index_col=0)

    # Load kilosort data
    spike_times = np.load(dp / 'spike_times.npy').squeeze()
    spike_templates = np.load(dp / 'spike_templates.npy').squeeze()
    amplitudes = np.load(dp / 'amplitudes.npy').squeeze()
    channel_map = np.load(dp / 'channel_map.npy').squeeze()
    pc_features = np.load(dp / 'pc_features.npy').squeeze()
    pc_feature_ind = np.load(dp / 'pc_feature_ind.npy').squeeze()

    # Filter unwanted units
    if units is not None:
        print("WARNING if you only subselect some units, \
            relative quality metrics (such as isolation distances) cannot be interpreted.")
        assert np.all(np.isin(units, spike_clusters))
        units_mask = np.isin(spike_clusters, units)
        spike_times = spike_times[units_mask]
        spike_clusters = spike_clusters[units_mask]
        spike_templates = spike_templates[units_mask]
        amplitudes = amplitudes[units_mask]
        pc_features = pc_features[units_mask,:,:]
        pc_feature_ind = pc_feature_ind[units_mask,:]

    # calculate quality metrics
    metrics =  calculate_quality_metrics(
                      spike_times,
                      spike_clusters,
                      spike_templates,
                      amplitudes,
                      channel_map,
                      pc_features,
                      pc_feature_ind,
                      fs = 30_000,
                      periods = None,
                      randseed = None,
                      **params)

    if save:
        metrics.to_csv(dp/fn)

    return metrics



def calculate_quality_metrics(spike_times,
                      spike_clusters,
                      spike_templates,
                      amplitudes,
                      channel_map,
                      pc_features,
                      pc_feature_ind,
                      fs = 30_000,
                      periods = None,
                      randseed = None,
                      **params):

    """ Calculate metrics for all units on one probe

    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in samples
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike
    spike_templates : numpy.ndarray (num_spikes x 0)
        Original template IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    channel_map : numpy.ndarray (num_units x 0)
        Original data channel for pc_feature_ind array
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
        If 'None', PC-based metrics will not be computed
    pc_feature_ind : numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    periods : list of recording periods in seconds [[t1, t2],...]

    randseed : int to seed random number generator, optional
    
    params : dict of parameters
        - "isi_threshold" : minimum time for isi violations (default 0.0015)
        - "min_isi" : (default 0.000166)
        - "num_channels_to_compare" : (default 7)
        - "max_spikes_for_unit" : (default 500)
        - "max_spikes_for_nn" : (default 10000)
        - "n_neighbors" : (default 4)
        - 'n_silhouette' : (default 10000)
        - "drift_metrics_interval_s" : (default 51)
        - "drift_metrics_min_spikes_per_interval" : (default 10)
        - "include_pc_metrics" : (default True)

    Outputs:
    --------
    metrics : pandas.DataFrame
        one column for each metric
        one row per unit per epoch

    """

    # preprocess parameters
    if randseed is not None:
        np.random.seed(randseed)
    spike_times = spike_times/fs

    metrics = pd.DataFrame()

    cluster_ids = np.unique(spike_clusters)
    total_units = len(cluster_ids)

    if periods is None: periods = [[0,spike_times[-1]]]

    for period in periods:

        period_mask = (spike_times > period[0])&(spike_times <= period[1])

        print("Calculating isi violations...")
        isi_viol = calculate_isi_violations(spike_times[period_mask], spike_clusters[period_mask], total_units, params['isi_threshold'], params['min_isi'])

        print("Calculating presence ratio...")
        presence_ratio = calculate_presence_ratio(spike_times[period_mask], spike_clusters[period_mask], total_units)

        print("Calculating firing rate...")
        firing_rate = calculate_firing_rate(spike_times[period_mask], spike_clusters[period_mask], total_units)

        print("Calculating amplitude cutoff...")
        amplitude_cutoff = calculate_amplitude_cutoff(spike_clusters[period_mask], amplitudes[period_mask], total_units)

        if pc_features is not None:

            print("Calculating PC-based metrics...")
            isolation_distance, l_ratio, d_prime, nn_hit_rate, nn_miss_rate = calculate_pc_metrics(spike_clusters[period_mask],
                                                                                                    spike_templates[period_mask],
                                                                                                    total_units,
                                                                                                    pc_features[period_mask,:,:],
                                                                                                    pc_feature_ind,
                                                                                                    params['num_channels_to_compare'],
                                                                                                    params['max_spikes_for_unit'],
                                                                                                    params['max_spikes_for_nn'],
                                                                                                    params['n_neighbors'])



            print("Calculating silhouette score")
            the_silhouette_score = calculate_silhouette_score(spike_clusters[period_mask],
                                                        spike_templates[period_mask],
                                                       total_units,
                                                       pc_features[period_mask,:,:],
                                                       pc_feature_ind,
                                                       params['n_silhouette'])

            # print("Calculating drift metrics")
            # max_drift, cumulative_drift = calculate_drift_metrics(spike_times[period_mask],
            #                                            spike_clusters[period_mask],
            #                                            spike_templates[period_mask],
            #                                            total_units,
            #                                            pc_features[period_mask,:,:],
            #                                            pc_feature_ind,
            #                                            params['drift_metrics_interval_s'],
            #                                            params['drift_metrics_min_spikes_per_interval'])

            max_drift, cumulative_drift = np.nan*the_silhouette_score, np.nan*the_silhouette_score

        if pc_features is not None:
            metrics = pd.concat((metrics, pd.DataFrame(data= OrderedDict((('cluster_id', cluster_ids),
                                ('firing_rate' , firing_rate),
                                ('presence_ratio' , presence_ratio),
                                ('isi_viol' , isi_viol),
                                ('amplitude_cutoff' , amplitude_cutoff),
                                ('isolation_distance' , isolation_distance),
                                ('l_ratio' , l_ratio),
                                ('d_prime' , d_prime),
                                ('nn_hit_rate' , nn_hit_rate),
                                ('nn_miss_rate' , nn_miss_rate),
                                ('silhouette_score', the_silhouette_score),
                                ('max_drift', max_drift),
                                ('cumulative_drift', cumulative_drift),
                                ('period' , [period] * total_units),
                                )))))

        else:
            metrics = pd.concat((metrics,
                        pd.DataFrame(data = OrderedDict((
                                ('cluster_id', cluster_ids),
                                ('firing_rate' , firing_rate),
                                ('presence_ratio' , presence_ratio),
                                ('isi_viol' , isi_viol),
                                ('amplitude_cutoff' , amplitude_cutoff),
                                ('epoch_name' , [period] * total_units),
                                ))
                                )
                                ))

    return metrics

# ===============================================================

# HELPER FUNCTIONS TO LOOP THROUGH CLUSTERS:

# ===============================================================

def calculate_isi_violations(spike_times, spike_clusters, total_units, isi_threshold, min_isi):

    cluster_ids = np.unique(spike_clusters)

    viol_rates = np.zeros((total_units,))

    for idx, cluster_id in enumerate(tqdm(cluster_ids)):

        for_this_cluster = (spike_clusters == cluster_id)
        viol_rates[idx], num_violations = isi_violations(spike_times[for_this_cluster],
                                                       min_time = np.min(spike_times),
                                                       max_time = np.max(spike_times),
                                                       isi_threshold=isi_threshold,
                                                       min_isi = min_isi)

    return viol_rates

def calculate_presence_ratio(spike_times, spike_clusters, total_units):

    cluster_ids = np.unique(spike_clusters)

    ratios = np.zeros((total_units,))

    for idx, cluster_id in enumerate(tqdm(cluster_ids)):

        for_this_cluster = (spike_clusters == cluster_id)
        ratios[idx] = presence_ratio(spike_times[for_this_cluster],
                                                       min_time = np.min(spike_times),
                                                       max_time = np.max(spike_times))

    return ratios



def calculate_firing_rate(spike_times, spike_clusters, total_units):

    cluster_ids = np.unique(spike_clusters)

    firing_rates = np.zeros((total_units,))

    min_time = np.min(spike_times)
    max_time = np.max(spike_times)

    for idx, cluster_id in enumerate(tqdm(cluster_ids)):

        for_this_cluster = (spike_clusters == cluster_id)
        firing_rates[idx] = firing_rate(spike_times[for_this_cluster],
                                        min_time = np.min(spike_times),
                                        max_time = np.max(spike_times))

    return firing_rates


def calculate_amplitude_cutoff(spike_clusters, amplitudes, total_units):

    cluster_ids = np.unique(spike_clusters)

    amplitude_cutoffs = np.zeros((total_units,))

    for idx, cluster_id in enumerate(tqdm(cluster_ids)):

        for_this_cluster = (spike_clusters == cluster_id)
        amplitude_cutoffs[idx] = amplitude_cutoff(amplitudes[for_this_cluster])

    return amplitude_cutoffs


def calculate_pc_metrics_one_cluster(cluster_peak_channels, idx, cluster_id,cluster_ids,
                                         half_spread, pc_features, pc_feature_ind,
                                         spike_clusters, spike_templates,
                                         max_spikes_for_cluster, max_spikes_for_nn, n_neighbors):

    peak_channel = cluster_peak_channels[idx]
    num_spikes_in_cluster = np.sum(spike_clusters == cluster_id)

    half_spread_down = peak_channel \
        if peak_channel < half_spread \
        else half_spread

    half_spread_up = np.max(pc_feature_ind) - peak_channel \
        if peak_channel + half_spread > np.max(pc_feature_ind) \
        else half_spread

    channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)
    units_in_range = cluster_ids[np.isin(cluster_peak_channels, channels_to_use)]

    spike_counts = np.zeros(units_in_range.shape)

    for idx2, cluster_id2 in enumerate(units_in_range):
        spike_counts[idx2] = np.sum(spike_clusters == cluster_id2)

    if num_spikes_in_cluster > max_spikes_for_cluster:
        relative_counts = spike_counts / num_spikes_in_cluster * max_spikes_for_cluster
    else:
        relative_counts = spike_counts

    all_pcs = np.zeros((0, pc_features.shape[1], channels_to_use.size))
    all_labels = np.zeros((0,))

    for idx2, cluster_id2 in enumerate(units_in_range):

        subsample = int(relative_counts[idx2])

        pcs = get_unit_pcs(cluster_id2, spike_clusters, spike_templates,
                           pc_feature_ind, pc_features, channels_to_use,
                           subsample)

        if pcs is not None and len(pcs.shape) == 3:

            labels = np.ones((pcs.shape[0],)) * cluster_id2

            all_pcs = np.concatenate((all_pcs, pcs),0)
            all_labels = np.concatenate((all_labels, labels),0)

    all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], pc_features.shape[1]*channels_to_use.size))
    if ((all_pcs.shape[0] > 10)
            and not (all_labels == cluster_id).all()  # Not all labels are this cluster
            and (sum(all_labels == cluster_id) > 20)  # No fewer than 20 spikes in this cluster
            and (len(channels_to_use) > 0)):
        isolation_distance, l_ratio = mahalanobis_metrics(all_pcs, all_labels, cluster_id)

        d_prime = lda_metrics(all_pcs, all_labels, cluster_id)

        nn_hit_rate, nn_miss_rate = nearest_neighbors_metrics(all_pcs, all_labels,
                                                                             cluster_id,
                                                                             max_spikes_for_nn,
                                                                             n_neighbors)
    else:  # Too few spikes or cluster doesnt exist
        isolation_distance = np.nan
        d_prime = np.nan
        nn_miss_rate = np.nan
        nn_hit_rate = np.nan
        l_ratio = np.nan
    return isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio


def calculate_pc_metrics(spike_clusters,
                         spike_templates,
                         total_units,
                         pc_features,
                         pc_feature_ind,
                         num_channels_to_compare,
                         max_spikes_for_cluster,
                         max_spikes_for_nn,
                         n_neighbors,
                         do_parallel=True):
    """

    :param spike_clusters:
    :param total_units:
    :param pc_features:
    :param pc_feature_ind:
    :param num_channels_to_compare:
    :param max_spikes_for_cluster:
    :param max_spikes_for_nn:
    :param n_neighbors:
    :return:
    """

    assert (num_channels_to_compare % 2 == 1)
    half_spread = int((num_channels_to_compare - 1) / 2)

    cluster_ids = np.unique(spike_clusters)
    template_ids = np.unique(spike_templates)

    template_peak_channels = np.zeros((len(template_ids),), dtype='uint16')
    cluster_peak_channels = np.zeros((len(cluster_ids),), dtype='uint16')

    for idx, template_id in enumerate(template_ids):
        for_template = np.squeeze(spike_templates == template_id)
        pc_max = np.argmax(np.mean(pc_features[for_template, 0, :], 0))
        template_peak_channels[idx] = pc_feature_ind[template_id, pc_max]

    for idx, cluster_id in enumerate(cluster_ids):
        for_unit = np.squeeze(spike_clusters == cluster_id)
        templates_for_unit = np.unique(spike_templates[for_unit])
        template_positions = np.where(np.isin(template_ids, templates_for_unit))[0]
        cluster_peak_channels[idx] = np.median(template_peak_channels[template_positions])

    # Loop over clusters:
    if do_parallel:
        from joblib import Parallel, delayed
        meas = Parallel(n_jobs=10, verbose=3)(  # -1 means use all cores
            delayed(calculate_pc_metrics_one_cluster)
            (cluster_peak_channels, idx, cluster_id, cluster_ids,
             half_spread, pc_features, pc_feature_ind,
             spike_clusters, spike_templates,
             max_spikes_for_cluster, max_spikes_for_nn, n_neighbors
             )
            for idx, cluster_id in enumerate(tqdm(cluster_ids)))
    else:
        meas = []
        for idx, cluster_id in enumerate(tqdm(cluster_ids, total=cluster_ids.max(), desc='PC metrics')):
            meas.append(calculate_pc_metrics_one_cluster(
                cluster_peak_channels, idx, cluster_id, cluster_ids,
                half_spread, pc_features, pc_feature_ind,
                spike_clusters, spike_templates,
                max_spikes_for_cluster, max_spikes_for_nn, n_neighbors))

    # Unpack:
    isolation_distances = []
    l_ratios = []
    d_primes = []
    nn_hit_rates = []
    nn_miss_rates = []
    for mea in meas:
        isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio = mea
        isolation_distances.append(isolation_distance)
        d_primes.append(d_prime)
        nn_miss_rates.append(nn_miss_rate)
        nn_hit_rates.append(nn_hit_rate)
        l_ratios.append(l_ratio)

    return (np.array(isolation_distances), np.array(l_ratios), np.array(d_primes),
            np.array(nn_hit_rates), np.array(nn_miss_rates))


def calculate_silhouette_score(spike_clusters,
                               spike_templates,
                               total_units,
                               pc_features,
                               pc_feature_ind,
                               total_spikes,
                               do_parallel=True):

    def score_inner_loop(i, cluster_ids):
        """
        Helper to loop over cluster_ids in one dimension. We dont want to loop over both dimensions in parallel-
        that will create too much worker overhead
        Args:
            i: index of first dimension
            cluster_ids: iterable of cluster ids

        Returns: scores for dimension j

        """
        scores_1d = []
        for j in cluster_ids:
            if j > i:
                inds = np.in1d(cluster_labels, np.array([i, j]))
                X = all_pcs[inds, :]
                labels = cluster_labels[inds]

                # len(np.unique(labels))=1 Can happen if total_spikes is low:
                if (len(labels) > 2) and (len(np.unique(labels)) > 1):
                    scores_1d.append(silhouette_score(X, labels))
                else:
                    scores_1d.append(np.nan)
            else:
                scores_1d.append(np.nan)
        return scores_1d

    cluster_ids = np.unique(spike_clusters)

    random_spike_inds = np.random.permutation(spike_clusters.size)
    random_spike_inds = random_spike_inds[:total_spikes]
    num_pc_features = pc_features.shape[1]
    num_channels = np.max(pc_feature_ind) + 1

    all_pcs = np.zeros((total_spikes, num_channels * num_pc_features))

    for idx, i in enumerate(random_spike_inds):

        unit_id = spike_templates[i]
        channels = pc_feature_ind[unit_id,:]

        for j in range(0,num_pc_features):
            all_pcs[idx, channels + num_channels * j] = pc_features[i,j,:]

    cluster_labels = np.squeeze(spike_clusters[random_spike_inds])

    SS = np.empty((total_units, total_units))
    SS[:] = np.nan


    # Build lists
    if do_parallel:
        from joblib import Parallel, delayed
        scores = Parallel(n_jobs=10, verbose=2)(delayed(score_inner_loop)(i, cluster_ids) for i in tqdm(cluster_ids))
    else:
        scores = [score_inner_loop(i, cluster_ids) for i in tqdm(cluster_ids)]

    # Fill the 2d array
    for i, col_score in enumerate(scores):
        for j, one_score in enumerate(col_score):
            SS[i, j] = one_score

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      a = np.nanmin(SS, 0)
      b = np.nanmin(SS, 1)

    return np.array([np.nanmin([a,b]) for a, b in zip(a,b)])


def calculate_drift_metrics(spike_times,
                            spike_clusters,
                            spike_templates,
                            total_units,
                            pc_features,
                            pc_feature_ind,
                            interval_length,
                            min_spikes_per_interval,
                            do_parallel=True):
    def calc_one_cluster(cluster_id):
        """
        Helper to calculate drift for one cluster
        Args:
            cluster_id:

        Returns:
            max_drift, cumulative_drift
        """
        in_cluster = spike_clusters == cluster_id
        times_for_cluster = spike_times[in_cluster]
        depths_for_cluster = depths[in_cluster]

        median_depths = []

        for t1, t2 in zip(interval_starts, interval_ends):

            in_range = (times_for_cluster > t1) * (times_for_cluster < t2)

            if np.sum(in_range) >= min_spikes_per_interval:
                median_depths.append(np.median(depths_for_cluster[in_range]))
            else:
                median_depths.append(np.nan)

        median_depths = np.array(median_depths)
        max_drift = np.around(np.nanmax(median_depths) - np.nanmin(median_depths), 2)
        cumulative_drift = np.around(np.nansum(np.abs(np.diff(median_depths))), 2)
        return max_drift, cumulative_drift


    max_drifts = []
    cumulative_drifts = []

    depths = get_spike_depths(spike_templates, pc_features, pc_feature_ind)

    interval_starts = np.arange(np.min(spike_times), np.max(spike_times), interval_length)
    interval_ends = interval_starts + interval_length

    cluster_ids = np.unique(spike_clusters)

    if do_parallel:
        from joblib import Parallel, delayed
        meas = Parallel(n_jobs=10, verbose=2)(delayed(calc_one_cluster)(cluster_id)
                                              for cluster_id in tqdm(cluster_ids))
    else:
        meas = [calc_one_cluster(cluster_id) for cluster_id in tqdm(cluster_ids)]

    for max_drift, cumulative_drift in meas:
        max_drifts.append(max_drift)
        cumulative_drifts.append(cumulative_drift)
    return np.array(max_drifts), np.array(cumulative_drifts)



# ==========================================================

# IMPLEMENTATION OF ACTUAL METRICS:

# ==========================================================


def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz

    Inputs:
    -------
    spike_train : array of spike times, in seconds
    min_time : minimum time for potential spikes, in seconds
    max_time : maximum time for potential spikes, in seconds
    isi_threshold : threshold for isi violation, in seconds
    min_isi : threshold for duplicate spikes, in seconds

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations

    """
    spike_train_chunk = spike_train[(spike_train > min_time) & (spike_train < max_time)]
    duplicate_spikes = np.where(np.diff(spike_train_chunk) <= min_isi)[0]
    spike_train_chunk = np.delete(spike_train_chunk, duplicate_spikes + 1)
    if len(spike_train_chunk) < 2 :
        return 0, 0

    isis = np.diff(spike_train_chunk)

    num_spikes = len(spike_train_chunk)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2*num_spikes*(isi_threshold - min_isi)
    total_rate = mean_firing_rate(spike_train, exclusion_quantile=0.005, fs=1) #firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations/violation_time
    fpRate = violation_rate/total_rate

    return fpRate, num_violations



def presence_ratio(spike_train, min_time, max_time, num_bins=100):
    """Calculate fraction of time the unit is present within an epoch.

    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes

    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking

    """

    h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))

    return np.sum(h > 0) / num_bins


def firing_rate(spike_train, min_time = None, max_time = None):
    """Calculate firing rate for a spike train.

    If no temporal bounds are specified, the first and last spike time are used.

    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)

    Outputs:
    --------
    fr : float
        Firing rate in Hz

    """

    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)

    fr = spike_train.size / duration

    return fr


def amplitude_cutoff(amplitudes, num_histogram_bins = 500, histogram_smoothing_value = 3):

    """ Calculate approximate fraction of spikes missing from a distribution of amplitudes

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)

    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible

    """


    h,b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter1d(h,histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:])*bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):

    """ Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)

    Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    isolation_distance : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    """

    pcs_for_this_unit = all_pcs[all_labels == this_unit_id,:]
    pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]

    mean_value = np.expand_dims(np.mean(pcs_for_this_unit,0),0)

    try:
        VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
    except np.linalg.linalg.LinAlgError: # case of singular matrix
        return np.nan, np.nan

    mahalanobis_other = np.sort(cdist(mean_value,
                       pcs_for_other_units,
                       'mahalanobis', VI = VI)[0])

    mahalanobis_self = np.sort(cdist(mean_value,
                             pcs_for_this_unit,
                             'mahalanobis', VI = VI)[0])

    n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]]) # number of spikes

    if n >= 2:

        dof = pcs_for_this_unit.shape[1] # number of features

        l_ratio = np.sum(1 - chi2.cdf(pow(mahalanobis_other,2), dof)) / \
                mahalanobis_self.shape[0] # normalize by size of cluster, not number of other spikes
        isolation_distance = pow(mahalanobis_other[n-1],2)

    else:
        l_ratio = np.nan
        isolation_distance = np.nan

    return isolation_distance, l_ratio




def lda_metrics(all_pcs, all_labels, this_unit_id):

    """ Calculates d-prime based on Linear Discriminant Analysis

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated

    Outputs:
    --------
    d_prime : float
        Isolation distance of this unit
    l_ratio : float
        L-ratio for this unit

    """

    X = all_pcs

    y = np.zeros((X.shape[0],),dtype='bool')
    y[all_labels == this_unit_id] = True

    lda = LDA(n_components=1)

    X_flda = lda.fit_transform(X, y)

    flda_this_cluster  = X_flda[np.where(y)[0]]
    flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

    d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster))/np.sqrt(0.5*(np.std(flda_this_cluster)**2+np.std(flda_other_cluster)**2))

    return d_prime



def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors):

    """ Calculates unit contamination based on NearestNeighbors search in PCA space

    Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394

    Inputs:
    -------
    all_pcs : numpy.ndarray (num_spikes x PCs)
        2D array of PCs for all spikes
    all_labels : numpy.ndarray (num_spikes x 0)
        1D array of cluster labels for all spikes
    this_unit_id : Int
        number corresponding to unit for which these metrics will be calculated
    max_spikes_for_nn : Int
        number of spikes to use (calculation can be very slow when this number is >20000)
    n_neighbors : Int
        number of neighbors to use

    Outputs:
    --------
    hit_rate : float
        Fraction of neighbors for target cluster that are also in target cluster
    miss_rate : float
        Fraction of neighbors outside target cluster that are in target cluster

    """

    total_spikes = all_pcs.shape[0]
    ratio = max_spikes_for_nn / total_spikes
    this_unit = all_labels == this_unit_id

    X = np.concatenate((all_pcs[this_unit,:], all_pcs[np.invert(this_unit),:]),0)

    n = np.sum(this_unit)

    if ratio < 1:
        inds = np.arange(0,X.shape[0]-1,1/ratio).astype('int')
        X = X[inds,:]
        n = int(n * ratio)


    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    this_cluster_inds = np.arange(n)

    this_cluster_nearest = indices[:n,1:].flatten()
    other_cluster_nearest = indices[n:,1:].flatten()

    hit_rate = np.mean(this_cluster_nearest < n)
    miss_rate = np.mean(other_cluster_nearest < n)

    return hit_rate, miss_rate

# ==========================================================

# HELPER FUNCTIONS:

# ==========================================================

def features_intersect(pc_feature_ind, these_channels):
    """
    # Take only the channels that have calculated features out of the ones we are interested in:
    # This should reduce the occurence of 'except IndexError' below

    Args:
        pc_feature_ind
        these_channels: channels_to_use or units_for_channel

    Returns:
        channels_to_use: intersect of what's available in PCs and what's needed
    """
    intersect = set(pc_feature_ind[these_channels[0], :])  # Initialize
    for cluster_id2 in these_channels:
        # Make a running intersect of what is available and what is needed
        intersect = intersect & set(pc_feature_ind[cluster_id2, :])
    return np.array(list(intersect))


def get_unit_pcs(unit_id,
                 spike_clusters,
                 spike_templates,
                 pc_feature_ind,
                 pc_features,
                 channels_to_use,
                 subsample):

    """ Return PC features for one unit

    Inputs:
    -------
    unit_id : Int
        ID for this unit
    spike_clusters : np.ndarray
        Cluster labels for each spike
    spike_templates : np.ndarry
        Template labels for each spike
    pc_feature_ind : np.ndarray
        Channels used for PC calculation for each unit
    pc_features : np.ndarray
        Array of all PC features
    channels_to_use : np.ndarray
        Channels to use for calculating metrics
    subsample : Int
        maximum number of spikes to return

    Output:
    -------
    unit_PCs : numpy.ndarray (float)
        PCs for one unit (num_spikes x num_PCs x num_channels)

    """


    inds_for_unit = np.where(spike_clusters == unit_id)[0]

    spikes_to_use = np.random.permutation(inds_for_unit)[:subsample]

    unique_template_ids = np.unique(spike_templates[spikes_to_use])

    unit_PCs = []

    for template_id in unique_template_ids:

        index_mask = spikes_to_use[np.squeeze(spike_templates[spikes_to_use]) == template_id]
        these_inds = pc_feature_ind[template_id, :]

        pc_array = []

        for i in channels_to_use:

            if np.isin(i, these_inds):
                channel_index = np.argwhere(these_inds == i)[0][0]
                pc_array.append(pc_features[index_mask, :, channel_index])
            else:
                return None

        unit_PCs.append(np.stack(pc_array, axis=-1))

    if len(unit_PCs) > 0:

        return np.concatenate(unit_PCs)
    else:
        return None


# ===============================================================

# UTILITIES

# ===============================================================

def get_spike_depths(spike_templates, pc_features, pc_feature_ind):

    """
    Calculates the distance (in microns) of individual spikes from the probe tip
    This implementation is based on Matlab code from github.com/cortex-lab/spikes
    Input:
    -----
    spike_templates : numpy.ndarray (N x 0)
        Template IDs for N spikes
    pc_features : numpy.ndarray (N x channels x num_PCs)
        PC features for each spike
    pc_feature_ind  : numpy.ndarray (M x channels)
        Channels used for PC calculation for each unit
    Output:
    ------
    spike_depths : numpy.ndarray (N x 0)
        Distance (in microns) from each spike waveform from the probe tip
    """

    pc_features_copy = np.copy(pc_features)
    pc_features_copy = np.squeeze(pc_features_copy[:,0,:])
    pc_features_copy[pc_features_copy < 0] = 0
    pc_power = pow(pc_features_copy, 2)

    spike_feat_ind = pc_feature_ind[spike_templates, :]
    spike_depths = np.sum(spike_feat_ind * pc_power, 1) / np.sum(pc_power,1)

    return spike_depths * 10

from npyx.spk_t import mean_firing_rate
