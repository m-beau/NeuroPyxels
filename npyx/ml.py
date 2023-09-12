import os
import random

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    print(
        (
            "\nPyTorch could not be imported - "
            "some functions from the submodule npyx.ml will not work.\n"
            "To install PyTorch, follow the instructions at http://pytorch.org"
        )
    )

from imblearn.over_sampling import RandomOverSampler
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut
from tqdm.auto import tqdm


def set_seed(seed=None, seed_torch=True):
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
    print(
        f"Mean train accuracy is {mean_train:.3f} while cross-validation accuracy is {mean_validation:.3f}"
    )
    print(
        f"Mean cross-validation F1 score across {n_runs} runs is {np.array(all_runs_f1_scores).mean():.3f}, with std {np.array(all_runs_f1_scores).std():.3f}"
    )
    print(
        f"Average standard deviation in F1 score across folds is {np.array(folds_stddev).mean():.3f}"
    )

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
