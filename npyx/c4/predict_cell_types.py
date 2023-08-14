import os

if __name__ == "__main__":
    __package__ = "npyx.c4"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import torch
    import torch.utils.data as data
except ImportError:
    pass

from tqdm.auto import tqdm

import npyx.corr as corr
import npyx.datasets as datasets
from npyx.gl import get_units
from npyx.spk_t import trn, trn_filtered
from npyx.spk_wvf import wvf_dsmatch

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
)

MODELS_URL_DICT = {
    "base": "https://figshare.com/ndownloader/files/41706915?private_link=2530fd0da03e18296d51"
}
HESSIANS_URL_DICT = {
    "base": "https://figshare.com/ndownloader/files/41706540?private_link=2530fd0da03e18296d51",
}


def download_file(url, output_path, description=None):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=description)

    with open(output_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))

    progress_bar.close()


def directory_checks(data_path):
    assert os.path.exists(data_path), "Data folder does not exist."
    assert os.path.isdir(data_path), "Data folder is not a directory."
    assert os.path.exists(
        os.path.join(data_path, "params.py")
    ), "Make sure that the data folder contains the params.py file used by phy."


def prepare_dataset(dp, units):
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
        t, _ = trn_filtered(dp, u, period_m=None)
        if len(t) < 10:
            bad_units.append(u)
            continue

        wvf, _, _, _ = wvf_dsmatch(dp, u, t_waveforms=120)
        waveforms.append(datasets.preprocess_template(wvf))

        _, acg = corr.crosscorr_vs_firing_rate(t, t, 2000, 1)
        acg, _ = corr.convert_acg_log(acg, 1, 2000)
        acgs_3d.append(acg.ravel() * 10)

    if len(bad_units) > 0:
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

    predictions_matrix = predictions_matrix.round(2)
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

    return predictions, mean_top_pred_confidence, delta_mean_confidences, n_votes


def main():
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
        "--soft_layer",
        action="store_true",
        help="Use 'soft' layer information (if available).",
    )
    parser.set_defaults(soft_layer=False)

    parser.add_argument(
        "--hard_layer",
        action="store_true",
        help="Use 'hard' layer information (if available).",
    )
    parser.set_defaults(hard_layer=False)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to keep model predictions.",
    )

    args = parser.parse_args()

    # Perform some checks on the data folder
    directory_checks(args.data_path)

    args.use_layer = args.soft_layer or args.hard_layer

    if args.use_layer:
        layer_path = os.path.join(args.data_path, "cluster_layer.tsv")
        assert os.path.exists(
            layer_path
        ), "Layer information not found. Make sure to have a cluster_layer.tsv file in the data folder if you want to use it."
        layer_df = pd.read_csv(
            layer_path, sep="\t"
        )  # layer will be in a cluster_layer.tsv file
        layer = layer_df["layer"].values

        if np.all(layer == 0):
            print(
                "Warning: all units are assigned to layer 0 (unknown). Make sure that the layer information is correct."
            )
            print("Falling back to no layer information.")
            args.use_layer = False
            one_hot_layer = None
        else:
            one_hot_layer = encode_layer_info(layer)
    else:
        one_hot_layer = None

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
        correspondence = datasets.LABELLING_MLI_CLUSTER_NO_GRC
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

    if not os.path.exists(models_archive):
        os.makedirs(models_folder, exist_ok=True)
        download_file(models_url, models_archive, description="Downloading models")
    if not os.path.exists(hessians_archive):
        os.makedirs(models_folder, exist_ok=True)
        download_file(
            hessians_url, hessians_archive, description="Downloading hessians"
        )

    # Prepare the data for prediction
    if args.units is not None:
        units = args.units
    else:
        units = get_units(args.data_path, args.quality)

    prediction_dataset, bad_units = prepare_dataset(args.data_path, units)

    good_units = [u for u in units if u not in bad_units]

    prediction_iterator = data.DataLoader(
        CustomDataset(
            prediction_dataset,
            np.zeros(len(prediction_dataset)),
            spikes_list=None,
            layer=one_hot_layer,
        ),
        batch_size=len(prediction_dataset),
    )

    ensemble = load_ensemble(
        models_archive,
        device=torch.device("cpu"),
        n_classes=6 if args.use_layer else 5,
        use_layer=args.use_layer,
        fast=False,
        laplace=True,
    )

    raw_probabilities = ensemble_predict(
        ensemble,
        prediction_iterator,
        device=torch.device("cpu"),
        method="raw",
        enforce_layer=args.hard_layer,
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

    predictions_df.to_csv(
        os.path.join(args.data_path, "cluster_cell_types.tsv"), sep="\t", index=False
    )

    # Finally make summary plots of the classifier output
    plots_folder = os.path.join(args.data_path, "cell_type_classification")

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    confidence_passing = np.array(good_units)[confidence_mask]

    for i, unit in enumerate(good_units):
        if unit not in confidence_passing:
            continue
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
            LABELMAP=datasets.CORRESPONDENCE_NO_GRC,
            C4_COLORS=C4_COLORS,
            fs=30000,
            unit_id=unit,
        )
    plot_survival_confidence(
        raw_probabilities[confidence_mask, :].mean(2),
        correspondence,
        saveDir=plots_folder,
    )


if __name__ == "__main__":
    main()
