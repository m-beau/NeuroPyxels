import contextlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

with contextlib.suppress(ImportError):
    import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut
from tqdm.auto import tqdm

from npyx.utils import npyx_cacher

from npyx.plot_utils import get_ncolors_cmap
from npyx.plot import DistinctColors20

import matplotlib.pyplot as plt


def set_seed(seed=None, seed_torch=False):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
    seed : Integer
            A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
            If `True` sets the random seed for pytorch tensors, so pytorch module
            must be imported. Default is `True`.

    Returns:
    seed : Integer corresponding to the random state.
    """
    if seed is None:
        seed = np.random.choice(2**16)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f"Random seed {seed} has been set.")
    return seed


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_runs: int,
    best_params: dict,
    oversampler=None,
    kfold=None,
    model_class=None,
    get_importance=False,
    scaler: TransformerMixin = None,
    **model_kwargs,
):
    """
    It runs a sklearn model with the best parameters found in the hyperparameter tuning step, and
    then runs a leave-one-out cross validation on the model

    Args:
      X (pd.DataFrame): pd.DataFrame,
      y (pd.Series): the target variable
      n_runs (int): number of cross-validation runs
      best_params (dict): dictionary containing the best parameters for the model
      oversampler: the oversampling method to use. If None, then we use RandomOverSampler.
      kfold: the cross-validation method to use.
      model: the model to use. If None, defaults to RandomForestClassifier
    """

    if kfold is None:
        kfold = LeaveOneOut()

    all_runs_f1_scores = []
    all_runs_targets = []
    all_runs_predictions = []
    all_runs_probabilities = []
    folds_stddev = []

    if get_importance:
        importances_list = []

    for _ in tqdm(range(n_runs), position=0, leave=True, desc="Classifier runs"):
        run_train_accuracies = []
        run_true_targets = []
        run_model_pred = []
        run_probabilites = []
        folds_f1 = []

        seed = np.random.choice(2**32)
        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(X, y)),
            leave=False,
            position=1,
            desc="Cross-validating",
            total=kfold.get_n_splits(X),
        ):
            X_train = X.iloc[train_idx].copy().to_numpy()
            y_train = y.iloc[train_idx].copy().values.astype(int)
            X_test = X.iloc[val_idx].to_numpy()
            y_test = y.iloc[val_idx].values.astype(int)

            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if oversampler is None:
                oversample = RandomOverSampler(random_state=seed)

            X_big, y_big = oversample.fit_resample(X_train, y_train)

            if model_class is None:
                model_class = RandomForestClassifier

            model = model_class(**best_params, **model_kwargs, random_state=seed)

            # fit the model on the data
            model.fit(X_big, y_big)

            # predict
            pred = model.predict(X_test)

            # score
            run_train_accuracies.append(model.score(X_train, y_train))

            # Append results
            run_true_targets.append(y_test)
            run_model_pred.append(pred)
            run_probabilites.append(model.predict_proba(X_test))

            fold_f1 = f1_score(y_test, pred, average="macro")
            folds_f1.append(fold_f1)

            if get_importance:
                importances_list.append(model.feature_importances_)

        run_true_targets = np.concatenate(run_true_targets).squeeze()
        run_model_pred = np.concatenate(run_model_pred).squeeze()

        run_f1 = f1_score(run_true_targets, run_model_pred, average="macro")
        all_runs_f1_scores.append(run_f1)
        all_runs_targets.append(np.array(run_true_targets))
        all_runs_predictions.append(np.array(run_model_pred))
        all_runs_probabilities.append(np.concatenate(run_probabilites, axis=0))
        folds_stddev.append(np.array(folds_f1).std())

    mean_train = np.array(run_train_accuracies).mean()
    mean_validation = (np.array(run_true_targets) == np.array(run_model_pred)).mean()
    print(f"Mean train accuracy is {mean_train:.3f} while cross-validation accuracy is {mean_validation:.3f}")
    print(
        f"Mean cross-validation F1 score across {n_runs} runs is {np.array(all_runs_f1_scores).mean():.3f}, with std {np.array(all_runs_f1_scores).std():.3f}"
    )
    print(f"Average standard deviation in F1 score across folds is {np.array(folds_stddev).mean():.3f}")

    all_targets = np.concatenate(all_runs_targets).squeeze()
    all_probabilities = np.concatenate(all_runs_probabilities).squeeze()

    results_dict = {
        "f1_scores": all_runs_f1_scores,
        "train_accuracies": run_train_accuracies,
        "true_targets": all_targets,
        "predicted_probability": all_probabilities,
        "folds_stddev": np.array(folds_stddev),
    }

    if get_importance:
        results_dict["feature_importance_list"] = importances_list

    return results_dict


# Dimentionality reduction utilities

@npyx_cacher
def umap_cached(X, n_neighbors=5, min_dist=0.01,
                again=False, cache_results=True, cache_path=None):
    """
    A simple way to save UMAP results when running it many times on the same data.
    - again: bool, whether to recompute results rather than loading them from cache.
    - cache_results: bool, whether to cache results at local_cache_memory.
    - cache_path: None|str, where to cache results.
                    If None, ~/.NeuroPyxels will be used (can be changed in npyx.CONFIG).
    """
    import umap
    fit    = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=5)
    return fit.fit_transform(X)

def get_cluster_colors(n_clusters, alpha=0.5, alpha_outliers=0.05):
    # cluster_colors = get_ncolors_cmap(10, 'tab10')
    # cluster_colors = cluster_colors + [c for i, c in enumerate(get_ncolors_cmap(20, 'tab20')) if i%2==1]
    cluster_colors = DistinctColors20[1:]
    cluster_colors = [c+[alpha] for c in cluster_colors]
    outlier_color = [0,0,0,alpha_outliers]
    return [cluster_colors[i%19] for i in range(n_clusters)] + [outlier_color]

def labels_to_rgb_colors(labels, order=None, colormap='tab10'):
    "order: unique list of labels, to choose order of colors from colormap"
    if order is None:
        order = np.unique(labels)
    assert np.all(np.isin(labels, order)), "Missing labels from order."

    palette = DistinctColors20[1:]#get_ncolors_cmap(10, colormap)
    index_dict = {u: i for i, u in enumerate(order)}
    return [palette[index_dict[u]] for u in labels]

def red_dim_plot(X, dims_to_plot = [0,2,1], labels = None,
                 title = None,
                 xlim = None, ylim = None, zlim = None,
                 xlabel = 'Dim 1', ylabel = 'Dim 3', zlabel = 'Dim 2',
                 alpha = 1, alpha_outliers = 0.05):

    elevation = 20

    assert len(dims_to_plot) == 3, "Must provide 3 dimensions to plot."
    assert X.ndim >= np.max(dims_to_plot), "X doesn't have the dimensions passed in dims_to_plot."

    fig, axes = plt.subplots(1,3, subplot_kw=dict(projection='3d'), figsize=(25,8))
    for i, azimuth in enumerate([200, 270, 330]):
        ax = axes[i]

        if labels is not None:
            cluster_ids = np.unique(labels).astype(int)
            n_clusters = len(cluster_ids)
            cluster_colors = get_cluster_colors(n_clusters, alpha, alpha_outliers)
            for cluster_id in cluster_ids:#[-1]+list(cluster_ids):
                cluster_m = labels == cluster_id
                ax.scatter(X[cluster_m,dims_to_plot[0]],
                           X[cluster_m,dims_to_plot[1]],
                           X[cluster_m, dims_to_plot[2]],
                           color=cluster_colors[cluster_id],
                           label=f"Cluster {cluster_id} ({int(cluster_m.sum())})",
                           s=40, lw=0)
            if i == 0: ax.legend()
        else:
            ax.scatter(X[:,dims_to_plot[0]], X[:,dims_to_plot[1]], X[:, dims_to_plot[2]],
                           s=40, lw=0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if zlim is not None: ax.set_zlim(zlim)
        
        ax.view_init(elevation, azimuth)

    fig.suptitle(title, va='top', fontsize=20)
    fig.patch.set_facecolor('white')

    return fig