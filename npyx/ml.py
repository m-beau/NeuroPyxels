import os
import random

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
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
    scaler=None,
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

    f1_scores = []

    hyper_true_targets = []
    hyper_preds = []
    hyper_probabilities = []

    if get_importance:
        importances_list = []

    for _ in tqdm(range(n_runs), position=0, leave=True, desc="Random Forest runs"):
        train_accuracies = []
        true_targets = []
        model_pred = []
        probabilities = []

        seed = np.random.choice(2**32)
        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(X, y)),
            leave=False,
            position=1,
            desc="Cross-validating",
            total=len(X),
        ):
            X_train = X.iloc[train_idx].copy()
            y_train = y.iloc[train_idx].copy().values
            X_test = X.iloc[val_idx]
            y_test = y.iloc[val_idx].values

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
            train_accuracies.append(model.score(X_train, y_train))

            # Append results
            true_targets.append(y_test)
            model_pred.append(pred)
            probabilities.append(model.predict_proba(X_test))

            if get_importance:
                importances_list.append(model.feature_importances_)

        f1 = f1_score(true_targets, model_pred, average="macro")
        f1_scores.append(f1)
        hyper_true_targets.append(np.array(true_targets))
        hyper_preds.append(np.array(model_pred))
        hyper_probabilities.append(np.array(probabilities))

    mean_train = np.array(train_accuracies).mean()
    mean_validation = (np.array(true_targets) == np.array(model_pred)).mean()
    print(
        f"Mean train accuracy is {mean_train:.3f} while LOO accuracy is {mean_validation:.3f}"
    )
    print(
        f"Mean LOO f1 score across random forests is {np.array(f1_scores).mean():.3f}"
    )

    all_targets = np.concatenate(hyper_true_targets).squeeze()
    all_probabilities = np.concatenate(hyper_probabilities).squeeze()

    results_dict = {
        "f1_scores": f1_scores,
        "train_accuracies": train_accuracies,
        "true_targets": all_targets,
        "predicted_probability": all_probabilities,
    }

    if get_importance:
        results_dict["feature_importance_list"] = importances_list

    return results_dict
