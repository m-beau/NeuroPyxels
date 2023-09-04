import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import npyx

from . import plots_functions as pf
from .dataset_init import *


def get_lisberger_dataset(data_path):
    files = os.listdir(data_path)
    lisberger_dataset = [
        dataset
        for dataset in files
        if dataset.endswith(".h5") and "lisberger" in dataset
    ][0]
    return os.path.join(data_path, lisberger_dataset)


MONKEY_CENTRAL_RANGE = int(WAVEFORM_SAMPLES * 40_000 / 30_000)


def main(data_folder=".", plot=True, name="feature_spaces", save_raw=False):
    # Parse the arguments into a class to preserve compatibility
    args = ArgsNamespace(
        data_folder=data_folder, plot=plot, name=name, save_raw=save_raw
    )

    datasets_abs = get_lisberger_dataset(args.data_folder)

    # Extract and check the datasets, saving a dataframe with the results
    dataset_df, dataset_class = extract_and_check(
        datasets_abs,
        save_folder=args.data_folder,
        lisberger=True,
        _label="expert_label",
        n_channels=10,
        central_range=MONKEY_CENTRAL_RANGE,
        _use_amplitudes=False,
        _lisberger=True,
        _id_type="neuron_id",
        _labels_only=True,
        _extract_mli_clusters=False,
    )

    _, mli_dataset_class = extract_and_check(
        datasets_abs,
        save_folder=args.data_folder,
        save=False,
        lisberger=True,
        _label="expert_label",
        n_channels=10,
        central_range=MONKEY_CENTRAL_RANGE,
        _use_amplitudes=False,
        _lisberger=True,
        _id_type="neuron_id",
        _labels_only=True,
        _extract_mli_clusters=True,
    )

    #! Remove MLI_Bs from the dataset
    mli_b = np.array(mli_dataset_class.labels_list) == "MLI_B"
    dataset_class._apply_mask(~mli_b)

    if dataset_class._sampling_rate != 30_000:
        resampled_dataset = npyx.datasets.resample_waveforms(dataset_class, 30_000)

    if args.plot:
        plots_folder = summary_plots(
            args, resampled_dataset, by_line=False, raw=False, monkey=True
        )

    dataset_inner_path = os.path.join(args.data_folder, args.name)
    if not os.path.exists(dataset_inner_path):
        os.mkdir(dataset_inner_path)

    feat_df = npyx.feat.h5_feature_extraction(
        resampled_dataset,
        save_path=os.path.join(dataset_inner_path, "monkey_engineered_features.csv"),
        fix_chanmap=False,
        quality_check=False,
        ignore_exceptions=True,
        _n_channels=resampled_dataset._n_channels,
        _central_range=resampled_dataset._central_range,
        _label="expert_label",
        _sampling_rate=dataset_class._sampling_rate,
        _use_chanmap=False,
        _wvf_type="flipped",
    )

    unusable_features_idx = npyx.feat.get_unusable_features(feat_df)

    if len(unusable_features_idx) > 0:
        dataset_df = report_unusable_features(
            feat_df, dataset_df, unusable_features_idx, args, lisberger=True
        )

    #! Insert here any other mask to remove neurons from the dataset in feature creation

    # Print to a text file called readme the cell counts in dataset.info
    with open(os.path.join(args.data_folder, args.name, "monkey_readme.txt"), "w") as f:
        f.write("Cell counts in dataset.info:\n")
        f.write(
            dataset_df.groupby(["label", "genetic_line"])["included"]
            .sum()
            .to_frame()
            .to_markdown(tablefmt="grid")
        )
        f.write("\n")

    # Save quality checks and feature extraction inclusion results
    plots_folder = make_plots_folder(args)
    if args.plot:
        save_quality_plots(dataset_df, save_folder=plots_folder, lisberger=True)

    # Divide the features dataframe into three main folders, one for waveform features,
    # one for temporal features and one for the combined
    temporal_features = npyx.feat.FEATURES[:20]
    waveform_features = npyx.feat.FEATURES[:3] + npyx.feat.FEATURES[20:]

    save_features(
        feat_df,
        "engineered_waveform_features",
        args,
        bad_idx=unusable_features_idx,
        drop_cols=temporal_features + ["relevant_channel", "any_somatic", "max_peaks"],
        monkey=True,
    )
    save_features(
        feat_df,
        "engineered_temporal_features",
        args,
        bad_idx=unusable_features_idx,
        drop_cols=waveform_features,
        monkey=True,
    )
    save_features(
        feat_df,
        "engineered_combined_features",
        args,
        bad_idx=unusable_features_idx,
        monkey=True,
    )

    ### Generating raw features dataframes
    if args.save_raw:
        common_preprocessing = dataset_class.conformed_waveforms
        labels = dataset_class.labels_list

        lab_df = pd.DataFrame({"label": labels})
        raw_acgs_df = pd.DataFrame(
            dataset_class.acg,
            columns=[f"raw_acg_{i}" for i in range(dataset_class.acg.shape[1])],
        )
        raw_wvf_single_common_preprocessing_df = pd.DataFrame(
            common_preprocessing,
            columns=[f"raw_wvf_{i}" for i in range(common_preprocessing.shape[1])],
        )

        save_features(
            pd.concat(
                [
                    lab_df,
                    raw_acgs_df,
                    raw_wvf_single_common_preprocessing_df,
                ],
                axis=1,
            ),
            "raw_2d_acg_peak_wvf",
            args,
            bad_idx=None,
            drop_cols=["label"],
            monkey=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialise a datasets folder with diagnostic plots for further modelling."
    )

    parser.add_argument(
        "-dp",
        "--data-folder",
        type=str,
        default=".",
        help="Path to the folder containing the dataset.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="feature_spaces",
        help="Name assigned to the dataset.",
    )

    parser.add_argument(
        "--save-raw",
        default="store_true",
        help="Save a dataframe with the raw peak waveform and autocorrelogram in a dataframe to train and use custom scikit-learn style models.",
    )
    parser.set_defaults(save_raw=False)

    parser.set_defaults(plot=True)

    args = parser.parse_args()

    main(**vars(args))
