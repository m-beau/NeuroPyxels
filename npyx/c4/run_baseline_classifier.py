import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

import npyx.datasets as datasets
import npyx.plot as npyx_plot
from npyx.ml import run_cross_validation

from . import plots_functions as pf
from .dataset_init import ArgsNamespace, save_results

matplotlib.rcParams["pdf.fonttype"] = 42  # necessary to make the text editable
matplotlib.rcParams["ps.fonttype"] = 42

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

LABELLING = datasets.LABELLING

# To do the inverse
CORRESPONDENCE = datasets.CORRESPONDENCE

CROSS_VAL_REPEATS = 10

SEED = 2023


def train_predict_test(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    params: Dict[str, Union[int, float, str]],
    oversampler: Optional[RandomOverSampler] = None,
    model_class: type = RandomForestClassifier,
    scaler: Optional[StandardScaler] = None,
    n_runs: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Trains a classifier on the training data, predicts on the test data, and returns the f1 score and predicted probabilities.

    Args:
        train_features: A numpy array of shape (n_samples, n_features) containing the training features.
        train_targets: A numpy array of shape (n_samples,) containing the training targets.
        test_features: A numpy array of shape (n_samples, n_features) containing the test features.
        test_targets: A numpy array of shape (n_samples,) containing the test targets.
        params: A dictionary containing the hyperparameters for the model.
        oversampler: An optional oversampler object to use for resampling the training data.
        model_class: The class of the model to use.
        scaler: An optional scaler object to use for scaling the data.
        n_runs: The number of times to run the model.

    Returns:
        A dictionary containing the f1 scores, true targets, and predicted probabilities.
    """
    f1_scores = []
    predicted_probabilities = []
    for run in tqdm(range(n_runs), desc="Training and testing model."):
        seed = np.random.choice(2**32) if run != 0 else SEED

        if scaler is not None:
            X = scaler.fit_transform(train_features)
            test_X = scaler.transform(test_features)
        else:
            X = train_features
            test_X = test_features

        if oversampler is None:
            oversampler = RandomOverSampler(random_state=seed)

        X_big, y_big = oversampler.fit_resample(X, train_targets)

        model = model_class(**params, random_state=seed)

        # fit the model on the data
        model.fit(X_big, y_big)

        predictions = model.predict(test_X)
        predicted_probability = model.predict_proba(test_X)
        predicted_probabilities.append(predicted_probability)

        f1_scr = f1_score(test_targets, predictions, average="macro")
        f1_scores.append(f1_scr)

    return {
        "f1_scores": np.array(f1_scores),
        "true_targets": np.concatenate([test_targets] * n_runs, axis=0),
        "predicted_probability": np.concatenate(predicted_probabilities, axis=0),
    }


def plot_feature_importance(
    feature_importance_list: list,
    features_dataframe: pd.DataFrame,
    save_folder: str = None,
):
    imp = np.stack(feature_importance_list, axis=0)
    mean_imp = np.mean(imp, axis=0)
    std_imp = np.std(imp, axis=0)

    sort = np.argsort(mean_imp, axis=0)[::-1]
    names = features_dataframe.columns.values[sort]
    mean_imp = mean_imp[sort]
    std_imp = std_imp[sort]
    forest_importances = pd.Series(mean_imp, index=names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std_imp, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    if save_folder is not None:
        npyx_plot.save_mpl_fig(fig, "feature_importance", save_folder, "pdf")
        plt.close()


def filter_out_granule_cells(features, targets, return_dicts=False):
    # sourcery skip: move-assign
    global LABELLING
    global CORRESPONDENCE

    is_pandas = isinstance(features, pd.DataFrame)

    if not is_pandas:
        targets = pd.DataFrame(targets, columns=["label"])
        features = pd.DataFrame(features)
    granule_mask = targets != "GrC"
    granule_mask = granule_mask.values.ravel()

    features = features.iloc[granule_mask]
    targets = targets.iloc[granule_mask]

    if not is_pandas:
        features = features.values
        targets = targets.values

    LABELLING = {"PkC_cs": 4, "PkC_ss": 3, "MFB": 2, "MLI": 1, "GoC": 0}

    # To do the inverse
    CORRESPONDENCE = {4: "PkC_cs", 3: "PkC_ss", 2: "MFB", 1: "MLI", 0: "GoC"}
    if return_dicts:
        return features, targets, LABELLING, CORRESPONDENCE

    return features, targets


def save_predictions_df(
    args,
    dataset_info,
    results_dict,
    save_path,
    repeats,
    probability_type=None,
    labelling=None,
    correspondence=None,
):
    if labelling is None:
        labelling = LABELLING
    if correspondence is None:
        correspondence = CORRESPONDENCE

    if probability_type is None:
        probability_type = "predicted_probability"

    predictions_df = pd.DataFrame(
        data=results_dict[probability_type], columns=list(labelling.keys())[::-1]
    )
    predictions_df["true_targets"] = [
        correspondence[true] for true in results_dict["true_targets"]
    ]
    predictions_df["predicted_targets"] = [
        correspondence[pred]
        for pred in np.argmax(results_dict[probability_type], axis=1)
    ]

    engineered_features = "engineered" in args.features_folder

    if engineered_features:
        dataset_info["included"] = (
            dataset_info["features_ok"] * dataset_info["included"]
        )
    included = dataset_info["included"].values.ravel().astype(bool)

    if not args.use_granule:
        included = included & (dataset_info["label"] != "GrC").values.ravel()

    masked_df = (
        dataset_info.iloc[included]
        .copy()
        .drop(columns=["label", "included", "features_ok", "quality_check"])
    ).reset_index(drop=True)

    repeated_df = pd.concat([masked_df] * repeats, axis=0).reset_index(drop=True)

    preds_df = pd.concat([repeated_df, predictions_df], axis=1, ignore_index=False)

    preds_df.to_csv(os.path.join(save_path, "predictions.csv"), index=False)


def get_model_class(args):
    if args.model == "random_forest":
        model_class = RandomForestClassifier
        default_params = {"bootstrap": True}

    elif args.model == "gaussian_process":
        model_class = GaussianProcessClassifier
        if args.kernel == "rbf":
            kernel = 1.0 * RBF(1.0, (1e-5, 1e6))
        elif args.kernel == "matern":
            kernel = 1.0 * Matern(1.0, (1e-5, 1e6))
        elif args.kernel == "dot_product":
            kernel = DotProduct(1, (1e-5, 1e6)) + WhiteKernel(1, (1e-5, 1e6))
        else:
            raise NotImplementedError(f"Kernel {args.kernel} not implemented.")

        default_params = {
            "max_iter_predict": 500,
            "n_restarts_optimizer": 2,
            "kernel": kernel,
        }

    elif args.model == "logistic_regression":
        model_class = LogisticRegression
        default_params = {"penalty": "none", "max_iter": 1000}

    else:
        raise NotImplementedError(f"Model {args.model} is not supported.")

    return model_class, default_params


def main(
    features_folder: str = ".",
    use_granule: bool = False,
    importance: bool = False,
    train_monkey: bool = False,
    model: str = "logistic_regression",
    kernel: str = "rbf",
    loo: bool = False,
):
    args = ArgsNamespace(
        features_folder=features_folder,
        use_granule=use_granule,
        importance=importance,
        train_monkey=train_monkey,
        model=model,
        kernel=kernel,
        loo=loo,
    )

    features_name = str(Path(args.features_folder)).split("/")[-1]
    features_name = " ".join((features_name).split("_"))
    features_name = features_name.title()

    # Load the data
    mouse_targets = pd.read_csv(os.path.join(args.features_folder, "labels.csv"))
    mouse_features = pd.read_csv(os.path.join(args.features_folder, "features.csv"))
    mouse_info_path = Path(args.features_folder).parent.joinpath("dataset_info.csv")

    monkey_features = pd.read_csv(
        os.path.join(args.features_folder, "monkey_features.csv")
    )
    monkey_targets = pd.read_csv(
        os.path.join(args.features_folder, "monkey_labels.csv")
    )
    monkey_info_path = Path(args.features_folder).parent.joinpath(
        "monkey_dataset_info.csv"
    )

    model_class, default_params = get_model_class(args)

    features = monkey_features if args.train_monkey else mouse_features
    targets = monkey_targets if args.train_monkey else mouse_targets
    info_path = monkey_info_path if args.train_monkey else mouse_info_path
    dataset_info = pd.read_csv(info_path)

    # Run cross validation with the chosen training data

    if not args.use_granule:
        features, targets = filter_out_granule_cells(features, targets)

    y = targets.replace(to_replace=LABELLING).squeeze()
    prefix = "monkey_trained_" if args.train_monkey else ""
    plots_prefix = "monkey trained " if args.train_monkey else ""
    results_dict = run_cross_validation(
        features,
        y,
        CROSS_VAL_REPEATS,
        default_params,
        oversampler=None,
        kfold=LeaveOneOut() if args.loo else StratifiedKFold(n_splits=5, shuffle=True),
        model_class=model_class,
        get_importance=args.importance,
        scaler=StandardScaler(),
    )

    suffix = f"_{args.kernel}" if args.model == "gaussian_process" else ""
    model = args.model + suffix
    save_folder = os.path.join(args.features_folder, model, f"{prefix}results")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_results(results_dict, save_folder)
    save_predictions_df(
        args,
        dataset_info,
        results_dict,
        save_folder,
        repeats=CROSS_VAL_REPEATS,
    )

    for threshold in tqdm(
        list(np.arange(0.5, 1, 0.1)) + [0.0], desc="Saving results figures"
    ):
        threshold = round(threshold, 2)
        fig = pf.plot_results_from_threshold(
            results_dict["true_targets"],
            results_dict["predicted_probability"],
            CORRESPONDENCE,
            threshold,
            f"{' '.join(model.split('_')).title()} {plots_prefix}({features_name})",
            collapse_classes=False,
            _shuffle_matrix=[3, 4, 1, 0, 2],
            f1_scores=results_dict["f1_scores"]
            if "f1_scores" in results_dict
            else None,
            _folds_stddev=results_dict["folds_stddev"]
            if "folds_stddev" in results_dict
            else None,
        )
        npyx_plot.save_mpl_fig(
            fig, f"{prefix}{model}_at_threshold_{threshold}", save_folder, "pdf"
        )
        plt.close()

    if "feature_importance_list" in results_dict.keys():
        plot_feature_importance(
            results_dict["feature_importance_list"], features, save_folder
        )

    # Now test on the other type of data

    if args.train_monkey:
        if not args.use_granule:
            mouse_features, mouse_targets = filter_out_granule_cells(
                mouse_features, mouse_targets
            )
        mouse_y = mouse_targets.replace(to_replace=LABELLING).squeeze()

        prefix = "mouse_tested_"
        plots_prefix = "monkey trained, predicting mouse data"
        results_dict = train_predict_test(
            features,
            y,
            mouse_features,
            mouse_y,
            default_params,
            oversampler=None,
            model_class=model_class,
            scaler=StandardScaler(),
            n_runs=CROSS_VAL_REPEATS,
        )
        dataset_info = pd.read_csv(mouse_info_path)
    else:
        monkey_y = monkey_targets.replace(to_replace=LABELLING).squeeze()

        prefix = "monkey_"
        plots_prefix = "predicting monkey data"
        results_dict = train_predict_test(
            features,
            y,
            monkey_features,
            monkey_y,
            default_params,
            oversampler=None,
            model_class=model_class,
            scaler=StandardScaler(),
            n_runs=CROSS_VAL_REPEATS,
        )
        dataset_info = pd.read_csv(monkey_info_path)

    save_folder = os.path.join(args.features_folder, model, f"{prefix}results")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_results(results_dict, save_folder)
    save_predictions_df(
        args,
        dataset_info,
        results_dict,
        save_folder,
        repeats=CROSS_VAL_REPEATS,
    )

    for threshold in tqdm(
        list(np.arange(0.5, 1, 0.1)) + [0.0], desc="Saving results figures"
    ):
        threshold = round(threshold, 2)
        fig = pf.plot_results_from_threshold(
            results_dict["true_targets"],
            results_dict["predicted_probability"],
            CORRESPONDENCE,
            threshold,
            f"{' '.join(model.split('_')).title()} {plots_prefix}({features_name})",
            collapse_classes=False,
            _shuffle_matrix=[3, 4, 1, 0, 2],
            f1_scores=results_dict["f1_scores"]
            if "f1_scores" in results_dict
            else None,
            _folds_stddev=results_dict["folds_stddev"]
            if "folds_stddev" in results_dict
            else None,
        )
        npyx_plot.save_mpl_fig(
            fig, f"{prefix}{model}_at_threshold_{threshold}", save_folder, "pdf"
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a baseline model on the given features."
    )

    parser.add_argument(
        "-f",
        "--features-folder",
        type=str,
        default=".",
        help="Path to the folder containing the features dataframe.",
    )

    parser.add_argument(
        "-g",
        "--use-granule",
        type=bool,
        default=False,
        help="Whether to use GrCs or not.",
    )

    parser.add_argument("--importance", action="store_true")
    parser.set_defaults(importance=False)

    parser.add_argument(
        "--train-monkey",
        action="store_true",
        help="Trains the model on monkey data instead of mouse.",
    )
    parser.set_defaults(train_monkey=False)

    parser.add_argument(
        "--model",
        choices=[
            "random_forest",
            "logistic_regression",
            "gaussian_process",
        ],
        default="logistic_regression",
        help="The sklearn model class that will be used to train.",
    )

    parser.add_argument(
        "--kernel",
        choices=["dot_product", "rbf", "matern"],
        default="rbf",
        help="The kernel to be used in GP classification",
    )

    parser.add_argument(
        "--loo",
        action="store_true",
        help="Whether to use leave one out cross validation.",
    )
    parser.set_defaults(loo=False)

    args = parser.parse_args()

    main(**vars(args))
