import argparse
import contextlib
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

if __name__ == "__main__":
    __package__ = "npyx.c4"

import dill
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

with contextlib.suppress(ImportError):
    import torch
    import torch.utils.data as data

from tqdm.auto import tqdm

import npyx.corr as corr
import npyx.datasets as datasets
from npyx.gl import get_units, load_units_qualities
from npyx.spk_t import trn, trn_filtered
from npyx.spk_wvf import wvf_dsmatch

from .dataset_init import ArgsNamespace, download_file, extract_and_check
from .plots_functions import (
    C4_COLORS,
    plot_features_1cell_vertical,
    plot_survival_confidence,
)
from .run_deep_classifier import (
    CustomDataset,
    encode_layer_info,
    ensemble_predict,
    load_ensemble,
    prepare_classification_dataset,
)

MODELS_URL_DICT = {
    "base": "https://figshare.com/ndownloader/files/42117042?private_link=2530fd0da03e18296d51",
    "mli_clustering": "https://figshare.com/ndownloader/files/42129447?private_link=d508ebc51d544ed8cd4c",
    "layer_information": "https://figshare.com/ndownloader/files/42130083?private_link=6531855c261b7bad032d",
    "layer_information_mli_clustering": "https://figshare.com/ndownloader/files/42130119?private_link=3a81e48aff77d844a402",
}

HESSIANS_URL_DICT = {
    "base": "https://figshare.com/ndownloader/files/42117033?private_link=2530fd0da03e18296d51",
    "mli_clustering": "https://figshare.com/ndownloader/files/42129435?private_link=d508ebc51d544ed8cd4c",
    "layer_information": "https://figshare.com/ndownloader/files/42130065?private_link=6531855c261b7bad032d",
    "layer_information_mli_clustering": "https://figshare.com/ndownloader/files/42130095?private_link=3a81e48aff77d844a402",
}


def get_n_cores(num_cores):
    max_num_cores=60
    max_num_cores = min(multiprocessing.cpu_count(), max_num_cores)
    num_cores = min(num_cores, max_num_cores)
    return num_cores


@contextlib.contextmanager
def redirect_stdout_fd(file):
    stdout_fd = sys.stdout.fileno()
    stdout_fd_dup = os.dup(stdout_fd)
    os.dup2(file.fileno(), stdout_fd)
    file.close()
    try:
        yield
    finally:
        os.dup2(stdout_fd_dup, stdout_fd)
        os.close(stdout_fd_dup)


@contextlib.contextmanager
def handle_outdated_model(exc_type, model_type):
    try:
        yield
    except exc_type:
        models_folder = os.path.join(
            Path.home(), ".npyx_c4_resources", "models", model_type
        )
        if os.path.exists(models_folder):
            print(
                "An error occurred while loading the models, likely due to a change in version. The models folder was removed to force a re-download."
            )
            print("\nPlease restart the program.")
            shutil.rmtree(models_folder)
            sys.exit()


def directory_checks(data_path):
    assert os.path.exists(data_path), "Data folder does not exist."
    if os.path.isfile(data_path) and data_path.endswith(".h5"):
        print(
            "You are using an h5 file as input. Make sure it is formatted correctly accoring to the C4 collaboration pipeline."
        )
        return

    assert os.path.isdir(data_path), "Data folder is not a directory."
    assert os.path.exists(
        os.path.join(data_path, "params.py")
    ), "Make sure that the current working directory contains the output of a spike sorter compatible with phy (in particular the params.py file)."
    if os.path.exists(os.path.join(data_path, "cluster_predicted_cell_type.tsv")):
        while True:
            prompt = input(
                "\nA cluster_predicted_cell_type.tsv file already exists. Are you sure you want to overwrite previous results? If you wish to compare different classifier parameters move the previous results to a different folder before running. (y/n) : "
            )
            if prompt.lower() == "y":
                break
            elif prompt.lower() == "n":
                sys.exit()
            else:
                print("\n  >> Please answer y or n!")
    # Remove potential conflicts with previous versions of the tool
    if os.path.exists(os.path.join(data_path, "cluster_cell_types.tsv")):
        os.remove(os.path.join(data_path, "cluster_cell_types.tsv"))


def prepare_dataset_from_binary(dp, units):
    waveforms = []
    acgs_3d = []
    bad_units = []
    for u in tqdm(
        units,
        desc="Preparing waveforms and ACGs for classification",
        position=0,
        leave=False,
    ):
        t = trn(dp, u)
        if len(t) < 100:
            bad_units.append(u)
            continue
        # We set period_m to None to use the whole recording
        try:
            t, _ = trn_filtered(dp, u, period_m=None)
        except (IndexError, pd.errors.EmptyDataError, ValueError):
            t, _ = trn_filtered(dp, u, period_m=None, again=True, enforced_rp=-1)
        if len(t) < 10:
            bad_units.append(u)
            continue

        try:
            wvf, _, _, _ = wvf_dsmatch(dp, u, t_waveforms=120)
        except (IndexError, pd.errors.EmptyDataError, ValueError):
            wvf, _, _, _ = wvf_dsmatch(dp, u, t_waveforms=120, again=True)
        waveforms.append(datasets.preprocess_template(wvf))

        _, acg = corr.crosscorr_vs_firing_rate(t, t, 2000, 1)
        acg, _ = corr.convert_acg_log(acg, 1, 2000)
        acgs_3d.append(acg.ravel() * 10)

    if bad_units:
        print(
            f"Units {str(bad_units)[1:-1]} were skipped because they had too few good spikes."
        )
    acgs_3d = np.array(acgs_3d)
    waveforms = np.array(waveforms)

    if len(acgs_3d) == 0:
        raise ValueError(
            "No units were found with the provided parameter choices after quality checks."
        )

    return np.concatenate((acgs_3d, waveforms), axis=1), bad_units


def get_layer_information(args, good_units):
    if os.path.isfile(args.data_path) and args.data_path.endswith(".h5"):
        layer = []
        neuron_ids, _ = datasets.get_h5_absolute_ids(args.data_path)
        for neuron_n in np.sort(neuron_ids):
            layer_neuron_n = datasets.get_neuron_attr(
                args.data_path, neuron_n, "phyllum_layer"
            )
            layer_neuron_n = datasets.decode_string(layer_neuron_n)
            layer.append(layer_neuron_n)
        layer = np.array(layer)
    else:
        layer_path = os.path.join(args.data_path, "cluster_layer.tsv")
        assert os.path.exists(
            layer_path
        ), "Layer information not found. Make sure to have a cluster_layer.tsv file in the data folder if you want to use it."
        layer_df = pd.read_csv(
            layer_path, sep="\t"
        )  # layer will be in a cluster_layer.tsv file

        layer_dict = {}
        for index in layer_df.index:
            cluster_id = layer_df["cluster_id"][index]
            if cluster_id in good_units:
                layer_dict[cluster_id] = layer_df["layer"][index]
        layer = np.array([layer_dict[cluster_id] for cluster_id in good_units])

    if np.all(layer == 0):
        print(
            "Warning: all units are assigned to layer 0 (unknown). Make sure that the layer information is correct."
        )
        print("\nFalling back to no layer information.")
        args.use_layer = False
        one_hot_layer = None
    else:
        one_hot_layer = encode_layer_info(layer)

    return one_hot_layer, args


def prepare_dataset_from_h5(data_path):
    _, dataset_class = extract_and_check(
        data_path,
        save=False,
        labelled=False,
        _labels_only=False,
        n_channels=4,
        _extract_layer=False,
    )

    dataset, _ = prepare_classification_dataset(
        dataset_class,
        normalise_acgs=False,
        win_size=2000,
        bin_size=1,
        multi_chan_wave=False,
        _acgs_path=None,
        _acg_mask=None,
        _acg_multi_factor=10,
    )

    return dataset, dataset_class.h5_ids.tolist()


def aux_prepare_dataset(dp, u):
    t = trn(dp, u)
    if len(t) < 100:
        # Bad units
        return [True, [], []]

    # We set period_m to None to use the whole recording
    try:
        t, _ = trn_filtered(dp, u, period_m=None)
    except (IndexError, pd.errors.EmptyDataError, ValueError):
        t, _ = trn_filtered(dp, u, period_m=None, again=True, enforced_rp=-1)
    if len(t) < 10:
        # Bad units
        return [True, [], []]

    try:
        wvf, _, _, _ = wvf_dsmatch(dp, u, t_waveforms=120)
    except (IndexError, pd.errors.EmptyDataError, ValueError):
        wvf, _, _, _ = wvf_dsmatch(dp, u, t_waveforms=120, again=True)
    waveforms = datasets.preprocess_template(wvf)

    _, acg = corr.crosscorr_vs_firing_rate(t, t, 2000, 1)
    acg, _ = corr.convert_acg_log(acg, 1, 2000)
    acgs_3d = acg.ravel() * 10

    return [False, waveforms, acgs_3d]


def prepare_dataset_from_binary_parallel(dp, units):
    waveforms = []
    acgs_3d = []
    bad_units = []

    num_cores = get_n_cores(len(units))

    with redirect_stdout_fd(open(os.devnull, "w")):
        dataset_results = Parallel(n_jobs=num_cores, prefer="processes")(
            delayed(aux_prepare_dataset)(dp, u)
            for u in tqdm(units, desc="Preparing waveforms and ACGs for classification")
        )

    for i in range(len(units)):
        if dataset_results[i][0] is True:
            bad_units.append(units[i])
        else:
            waveforms.append(dataset_results[i][1])
            acgs_3d.append(dataset_results[i][2])

    if bad_units:
        print(
            f"Units {str(bad_units)[1:-1]} were skipped because they had too few good spikes."
        )
    acgs_3d = np.array(acgs_3d)
    waveforms = np.array(waveforms)

    if len(acgs_3d) == 0:
        raise ValueError(
            "No units were found with the provided parameter choices after quality checks."
        )

    return np.concatenate((acgs_3d, waveforms), axis=1), bad_units


def prepare_dataset(args: ArgsNamespace) -> tuple:
    """
    Prepare the dataset for classification.

    Args:
        args (ArgsNamespace): The arguments namespace.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - dataset (numpy.ndarray): A 2D numpy array of shape (n_obs, n_features) containing the preprocessed ACGs
            and waveforms for each observation.
            - good_units (list): A list of unit ids that were included in the dataset.
    """
    # Check if we are dealing with an h5 file
    if os.path.isfile(args.data_path) and args.data_path.endswith(".h5"):
        prediction_dataset, good_units = prepare_dataset_from_h5(args.data_path)

    # Otherwise we are dealing with a phy output folder
    else:
        # First extract the units from the phy output folder
        if args.units is not None:
            units = args.units
        else:
            units = get_units(args.data_path, args.quality)

        if args.parallel:
            prediction_dataset, bad_units = prepare_dataset_from_binary_parallel(
                args.data_path, units
            )
        else:
            prediction_dataset, bad_units = prepare_dataset_from_binary(
                args.data_path, units
            )

        good_units = [u for u in units if u not in bad_units]

    return prediction_dataset, good_units


def format_predictions(predictions_matrix: np.ndarray):
    """
    Formats the predictions matrix by computing the mean predictions, prediction confidences, delta mean confidences,
    and number of votes.

    Args:
        predictions_matrix (numpy.ndarray): A 3D numpy array of shape (n_obs, n_classes, n_models) containing the
        predictions for each observation, class, and model.

    Returns:
        tuple: A tuple containing four numpy arrays:
            - predictions: A 1D numpy array containing the predicted class for each observation.
            - mean_top_pred_confidence: A 1D numpy array containing the mean confidence of the top predicted class for
            each observation.
            - delta_mean_confidences: A 1D numpy array containing the difference between the mean confidence of the top
            predicted class and the second top predicted class for each observation.
            - n_votes: A 1D numpy array containing the number of models that predicted the top predicted class for each
            observation.
    """

    mean_predictions = predictions_matrix.mean(axis=2)

    # compute predictions
    predictions = mean_predictions.argmax(axis=1)
    # compute prediction confidences
    mean_top_pred_confidence = mean_predictions.max(1)
    delta_mean_confidences = np.diff(np.sort(mean_predictions, axis=1), axis=1)[:, -1]
    n_votes = np.array(
        [
            (predictions_matrix[i, :, :].argmax(0) == pred).sum()
            for i, pred in enumerate(predictions)
        ]
    )

    return predictions, mean_top_pred_confidence, delta_mean_confidences, n_votes  #


def save_serialised(la, filepath):
    with open(filepath, "wb") as outpt:
        dill.dump(la, outpt, recurse=True)


def load_serialised(filepath):
    with open(filepath, "rb") as inpt:
        la = dill.load(inpt)
    return la


def load_precalibrated_ensemble(
    models_directory,
    fast=False,
):
    ensemble_paths = sorted(
        os.listdir(models_directory),
        key=lambda x: int(x.split("calibrated_model_")[1].split(".")[0]),
    )

    if fast and len(ensemble_paths) > 100:
        ensemble_paths = np.random.choice(ensemble_paths, 100, replace=False).tolist()

    # Load each model from the nested folder
    models = []
    for model_file in tqdm(ensemble_paths, desc="Loading models"):
        model_path = os.path.join(models_directory, model_file)
        model = load_serialised(model_path)
        models.append(model)

    # Workaround to avoid unknown bug when loading the models
    del models[0]
    models.append(
        load_serialised(os.path.join(models_directory, "calibrated_model_0.pkl"))
    )

    return models


def save_calibrated_ensemble(calibrated_models, save_directory):
    os.makedirs(save_directory, exist_ok=True)

    # Save each model in the temporary directory
    for i, cal_model in tqdm(
        enumerate(calibrated_models),
        desc="Saving calibration for later use",
        total=len(calibrated_models),
    ):
        save_serialised(
            cal_model, os.path.join(save_directory, f"calibrated_model_{i}.pkl")
        )


def main(
    data_path: str = ".",
    quality: str = "good",
    units: Optional[list] = None,
    mli_clustering: bool = False,
    layer: bool = False,
    threshold: float = 0.5,
    parallel: bool = True,
) -> None:
    """
    Predicts the cell types of units in a given dataset using a pre-trained ensemble of classifiers.

    Args:
        data_path (str, optional): Path to the ephys dataset folder. Defaults to ".".
        quality (str, optional): Quality of the units to use. Must be either "all" or "good". Defaults to "good".
        units (list, optional): List of unit IDs to use. If None, all units of "quality" will be used. Defaults to None.
        mli_clustering (bool, optional): Whether to use MLI clustering. Defaults to False.
        layer (bool, optional): Whether to use layer information from phyllum (or other sources). Defaults to False.
        threshold (float, optional): Confidence threshold for cell type predictions. Defaults to 0.5.
        parallel (bool, optional): Whether to use parallel processing. Defaults to True.

    Returns:
        None, but saves classifier results in files in the data folder.
    """
    start_time = time.perf_counter()
    args = ArgsNamespace(
        data_path=data_path,
        quality=quality,
        units=units,
        mli_clustering=mli_clustering,
        use_layer=layer,
        threshold=threshold,
        parallel=parallel,
    )

    assert args.quality in [
        "all",
        "good",
    ], "Invalid value for 'quality'. Must be either 'all' or 'good'."

    # Perform some checks on the data folder
    directory_checks(args.data_path)

	#This function checks the content of cluster_group.tsv file and regenerate this one if it is required.
    load_units_qualities(args.data_path, again=True)

    # Determine the model type that we should use
    if args.mli_clustering and not args.use_layer:
        model_type = "mli_clustering"
    elif args.use_layer and not args.mli_clustering:
        model_type = "layer_information"
    elif args.use_layer:
        model_type = "layer_information_mli_clustering"
    else:
        model_type = "base"

    # Set the labelling and correspondence based on whether we are using mli_clustering or not
    if args.mli_clustering:
        labelling = datasets.LABELLING_MLI_CLUSTER_NO_GRC
        correspondence = datasets.CORRESPONDENCE_MLI_CLUSTER_NO_GRC
    else:
        labelling = datasets.LABELLING_NO_GRC
        correspondence = datasets.CORRESPONDENCE_NO_GRC

    # Determine the URL from which we should download the models
    models_url = MODELS_URL_DICT[model_type]
    hessians_url = HESSIANS_URL_DICT[model_type]

    # First download the models if they are not already downloaded
    models_folder = os.path.join(
        Path.home(), ".npyx_c4_resources", "models", model_type
    )
    models_archive = os.path.join(models_folder, "trained_models.tar.gz")
    hessians_archive = os.path.join(models_folder, "hessians.pt")
    serialised_ensemble = os.path.join(models_folder, "calibrated_models")

    if not os.path.exists(serialised_ensemble):
        if not os.path.exists(models_archive):
            os.makedirs(models_folder, exist_ok=True)
            download_file(models_url, models_archive, description="Downloading models")
        if not os.path.exists(hessians_archive):
            os.makedirs(models_folder, exist_ok=True)
            download_file(
                hessians_url, hessians_archive, description="Downloading hessians"
            )

    # Prepare the data for prediction
    prediction_dataset, good_units = prepare_dataset(args)

    if args.use_layer:
        one_hot_layer, args = get_layer_information(args, good_units)
    else:
        one_hot_layer = None

    prediction_iterator = data.DataLoader(
        CustomDataset(
            prediction_dataset,
            np.zeros(len(prediction_dataset)),
            spikes_list=None,
            layer=one_hot_layer,
        ),
        batch_size=len(prediction_dataset),
    )

    # Check if this is the first time the ensemble is loaded on this machine
    precalibrated_ensemble_present = os.path.exists(serialised_ensemble)

    if not precalibrated_ensemble_present:
        with handle_outdated_model(RuntimeError, model_type):
            ensemble = load_ensemble(
                models_archive,
                device=torch.device("cpu"),
                n_classes=6 if args.use_layer else 5,
                use_layer=args.use_layer,
                fast=False,
                laplace=True,
            )

        # Serialize the ensemble for future use
        save_calibrated_ensemble(ensemble, serialised_ensemble)

        # Remove the models archive to save space
        os.remove(models_archive)
        os.remove(hessians_archive)
    else:
        ensemble = load_precalibrated_ensemble(serialised_ensemble, fast=False)

    raw_probabilities = ensemble_predict(
        ensemble,
        prediction_iterator,
        device=torch.device("cpu"),
        method="raw",
        enforce_layer=False,
        labelling=labelling,
    )

    predictions, mean_top_pred_confidence, _, n_votes = format_predictions(
        raw_probabilities
    )
    predictions_str = [correspondence[int(prediction)] for prediction in predictions]

    predictions_df = pd.DataFrame(
        {
            "cluster_id": good_units,
            "predicted_cell_type": predictions_str,
            "confidence": mean_top_pred_confidence,
            "model_votes": [f"{n}/{len(ensemble)}" for n in n_votes],
        }
    )

    confidence_mask = predictions_df["confidence"] >= args.threshold
    predictions_df = predictions_df[confidence_mask]

    # If running on an .h5 file we need to create a save directory
    if os.path.isfile(args.data_path) and args.data_path.endswith(".h5"):
        save_path = os.path.join(Path(args.data_path).parent, Path(args.data_path).stem)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = args.data_path

    # Save the predictions to a file that can be read by phy
    predictions_df[["cluster_id", "predicted_cell_type"]].to_csv(
        os.path.join(save_path, "cluster_predicted_cell_type.tsv"), sep="\t", index=False
    )
    predictions_df[["cluster_id", "confidence"]].to_csv(
        os.path.join(save_path, "cluster_confidence.tsv"), sep="\t", index=False
    )
    predictions_df[["cluster_id", "model_votes"]].to_csv(
        os.path.join(save_path, "cluster_model_votes.tsv"), sep="\t", index=False
    )

    # Save the raw probabilities and the label correspondence for power users
    plots_folder = os.path.join(save_path, "cell_type_classification")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    np.save(
        os.path.join(plots_folder, "ensemble_predictions_ncells_nclasses_nmodels.npy"),
        raw_probabilities,
    )
    pd.DataFrame(correspondence, index=[0]).to_csv(
        os.path.join(plots_folder, "label_correspondence.tsv"), sep="\t", index=False
    )

    # Finally make summary plots of the classifier output
    confidence_passing = np.array(good_units)[confidence_mask]

    # Define a function to plot the features of a single unit
    def aux_plot_features(i, unit, labelmap):
        if unit in confidence_passing:
            plot_features_1cell_vertical(
                i,
                prediction_dataset[:, :2010].reshape(-1, 10, 201) * 100,
                prediction_dataset[:, 2010:],
                predictions=raw_probabilities,
                saveDir=plots_folder,
                fig_name=f"unit_{unit}_cell_type_predictions",
                plot=False,
                cbin=1,
                cwin=2000,
                figsize=(10, 4),
                LABELMAP=labelmap,
                C4_COLORS=C4_COLORS,
                fs=30000,
                unit_id=unit,
            )

    num_cores = get_n_cores(len(good_units))
    with redirect_stdout_fd(open(os.devnull, "w")):
        Parallel(n_jobs=num_cores, prefer="processes")(
            delayed(aux_plot_features)(i, unit, correspondence)
            for i, unit in enumerate(good_units)
        )

    plot_survival_confidence(
        raw_probabilities,
        correspondence,
        ignore_below_confidence=args.threshold,
        saveDir=plots_folder,
    )
    # Save the raw probabilities and the label correspondence for power users
    np.save(
        os.path.join(plots_folder, "ensemble_predictions_ncells_nclasses_nmodels.npy"),
        raw_probabilities,
    )
    pd.DataFrame(correspondence, index=[0]).to_csv(
        os.path.join(plots_folder, "label_correspondence.tsv"), sep="\t", index=False
    )

    end_time = time.perf_counter()
    print("Cell type classfication execution time: ", end_time - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dp",
        "--data-path",
        type=str,
        default=".",
        help="Path to the folder containing the dataset.",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=str,
        choices=["all", "good"],
        default="good",
        help="Units of which quality we should classify.",
    )

    parser.add_argument(
        "--units",
        nargs="+",
        type=int,
        default=None,
        help="Which units to classify. If not specified, falls back to all units of 'quality' (all good units by default).",
    )
    parser.add_argument(
        "--mli_clustering", action="store_true", help="Divide MLI into two clusters."
    )
    parser.set_defaults(mli_clustering=False)

    parser.add_argument(
        "--layer",
        action="store_true",
        help="Use layer information (if available).",
    )
    parser.set_defaults(layer=False)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to keep model predictions.",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing.",
    )
    parser.add_argument("--serial", dest="parallel", action="store_false")
    parser.set_defaults(parallel=True)

    args = parser.parse_args()
    main(**vars(args))
