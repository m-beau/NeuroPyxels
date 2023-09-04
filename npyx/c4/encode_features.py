import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    __package__ = "npyx.c4"

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from npyx.datasets import NeuronsDataset

from .dataset_init import (
    BIN_SIZE,
    WIN_SIZE,
    ArgsNamespace,
    combine_features,
    extract_and_check,
    get_paths_from_dir,
    prepare_classification_dataset,
    save_features,
)
from .dl_utils import load_acg_vae, load_waveform_encoder, load_waveform_vae
from .run_deep_classifier import download_vaes

matplotlib.rcParams["pdf.fonttype"] = 42  # necessary to make the text editable
matplotlib.rcParams["ps.fonttype"] = 42

SEED = 1234

CENTRAL_RANGE = 90

N_CHANNELS = 4


WVF_VAE_PATH_SINGLE = os.path.join(
    Path.home(),
    ".npyx_c4_resources",
    "vaes",
    "wvf_singlechannel_encoder.pt",
)

ACG_ENCODER_PATH = os.path.join(
    Path.home(),
    ".npyx_c4_resources",
    "vaes",
    "3DACG_logscale_encoder.pt",
)

WVF_VAE_PATH_MULTI = os.path.join(
    Path.home(),
    ".npyx_c4_resources",
    "vaes",
    "wvf_multichannel_encoder.pt",
)

WVF_ENCODER_ARGS_SINGLE = {
    "beta": 5,
    "d_latent": 10,
    "dropout_l0": 0.1,
    "dropout_l1": 0.1,
    "lr": 5e-5,
    "n_layers": 2,
    "n_units_l0": 600,
    "n_units_l1": 300,
    "optimizer": "Adam",
    "batch_size": 128,
}

WVF_ENCODER_ARGS_MULTI = {
    "n_channels": 4,
    "central_range": 90,
    "d_latent": 10,
    "device": "cpu",
}


INIT_WEIGHTS = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def encode_acgs(encoder_path, acgs_3d, dataset: NeuronsDataset):
    vae = load_acg_vae(encoder_path, WIN_SIZE // 2, BIN_SIZE, pool="avg")

    encoded_acgs = vae.encode_numpy(acgs_3d).squeeze()

    encoded_db = pd.DataFrame(
        {
            "label": dataset.labels_list,
            "dataset": [dataset.info[i].split("/")[0] for i in range(len(dataset))],
            "unit": [dataset.info[i].split("/")[1] for i in range(len(dataset))],
            **{f"acg_latent_{i}": encoded_acgs[:, i] for i in range(10)},
        }
    )

    return encoded_db.infer_objects()


def encode_waveforms(
    encoder_path, waveforms, dataset: NeuronsDataset, args: ArgsNamespace
):
    if args.multi_channel:
        wvf_vae = load_waveform_vae(
            WVF_ENCODER_ARGS_MULTI,
            WVF_VAE_PATH_MULTI,
        )

        encoded_waveforms = wvf_vae.encode_numpy(waveforms).squeeze()
    else:
        enc = load_waveform_encoder(
            WVF_ENCODER_ARGS_SINGLE,
            encoder_path,
            in_features=CENTRAL_RANGE,
            device=DEVICE,
        )
        encoded_waveforms = (
            enc(torch.Tensor(waveforms).reshape(-1, 1, 90).to(DEVICE))
            .mean.detach()
            .cpu()
            .numpy()
            .squeeze()
        )

    encoded_db = pd.DataFrame(
        {
            "label": dataset.labels_list,
            "dataset": [dataset.info[i].split("/")[0] for i in range(len(dataset))],
            "unit": [dataset.info[i].split("/")[1] for i in range(len(dataset))],
            **{f"wvf_latent_{i}": encoded_waveforms[:, i] for i in range(10)},
        }
    )

    return encoded_db.infer_objects()


def main(
    data_folder=".",
    multi_channel=False,
    labelled=True,
):
    args = ArgsNamespace(
        data_folder=data_folder,
        multi_channel=multi_channel,
        labelled=labelled,
        name="feature_spaces",
    )

    datasets_abs = get_paths_from_dir(args.data_folder)

    # Download the VAEs if they are not present
    download_vaes()

    # Extract and check the datasets, saving a dataframe with the results
    _, dataset_class = extract_and_check(
        *datasets_abs,
        save=False,
        _labels_only=args.labelled,
        labelled=args.labelled,
        normalise_wvf=False,
        n_channels=N_CHANNELS,
        central_range=120,
    )

    # Apply quality checks and filter out granule cells
    checked_dataset = dataset_class.apply_quality_checks()
    LABELLING, CORRESPONDENCE, granule_mask = checked_dataset.filter_out_granule_cells(
        return_mask=True
    )

    dataset, _ = prepare_classification_dataset(
        checked_dataset,
        normalise_acgs=False,
        multi_chan_wave=args.multi_channel,
        process_multi_channel=args.multi_channel,
        _acgs_path=os.path.join(
            args.data_folder, "acgs_vs_firing_rate", "acgs_3d_logscale.npy"
        ),
        _acg_mask=(~granule_mask),
        _acg_multi_factor=10,
        _n_channels=N_CHANNELS,
    )

    acgs = dataset[:, :2010].reshape(-1, 10, 201)[:, :, 100:].reshape(-1, 10 * 101)
    waveforms = dataset[:, 2010:]

    wvf_encoder_path = WVF_VAE_PATH_MULTI if args.multi_channel else WVF_VAE_PATH_SINGLE

    encoded_waveforms_df = encode_waveforms(
        wvf_encoder_path, waveforms, checked_dataset, args
    )

    encoded_acgs_df = encode_acgs(ACG_ENCODER_PATH, acgs, checked_dataset)

    combined_encoding = combine_features(encoded_waveforms_df, encoded_acgs_df)

    suffix = "_multi_channel" if args.multi_channel else ""

    save_features(
        encoded_waveforms_df,
        f"encoded_wvf{suffix}",
        args,
        bad_idx=None,
        drop_cols=["label", "dataset", "unit"],
    )

    save_features(
        encoded_acgs_df,
        "encoded_acg",
        args,
        bad_idx=None,
        drop_cols=["label", "dataset", "unit"],
    )

    save_features(
        combined_encoding,
        f"encoded_acg_wvf{suffix}",
        args,
        bad_idx=None,
        drop_cols=["label", "dataset", "unit"],
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
        help="Path to the folder containing the .h5 dataset.",
    )

    parser.add_argument(
        "--multi-channel",
        action="store_true",
        help="Use multi-channel waveforms encoder.",
    )
    parser.set_defaults(multi_channel=False)

    parser.add_argument("--labelled", action="store_true")
    parser.add_argument("--unlabelled", dest="labelled", action="store_false")
    parser.set_defaults(labelled=True)

    args = parser.parse_args()
    main(**vars(args))
