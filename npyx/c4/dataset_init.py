import os

if __name__ == "__main__":
    __package__ = "npyx.c4"

import argparse
import datetime
import pickle
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import OptimizeWarning
from tqdm.auto import tqdm

import npyx.corr as corr
import npyx.datasets as datasets
import npyx.feat as feat
import npyx.plot as plot

warnings.filterwarnings("ignore", category=OptimizeWarning)
matplotlib.rcParams["pdf.fonttype"] = 42  # necessary to make the text editable
matplotlib.rcParams["ps.fonttype"] = 42

SEED = 1234

# Path to the current hdf5 dataset
HOME = str(Path.home())

# To convert text labels to numbers
LABELLING = {
    "PkC_cs": 5,
    "PkC_ss": 4,
    "MFB": 3,
    "MLI": 2,
    "GoC": 1,
    "GrC": 0,
    "unlabelled": -1,
}

# To do the inverse
CORRESPONDENCE = {
    5: "PkC_cs",
    4: "PkC_ss",
    3: "MFB",
    2: "MLI",
    1: "GoC",
    0: "GrC",
    -1: "unlabelled",
}

WAVEFORM_SAMPLES = 120

N_CHANNELS = 10

BIN_SIZE = 1

WIN_SIZE = 200

COLORS_DICT = {
    "PkC_ss": [28, 120, 181],
    "PkC_cs": [0, 0, 0],
    "MLI": [224, 85, 159],
    "MFB": [214, 37, 41],
    "GrC": [42, 161, 73],
    "GoC": [143, 103, 169],
    "laser": [96, 201, 223],
    "drug": [239, 126, 34],
    "background": [244, 242, 241],
}
HULL_LINES_DICT = {
    "PkC_ss": "unknown",
    "PkC_cs": "unknown",
    "MFB": "Thy1",
    "MLI": "C-Kit",
    "GoC": "C-Kit",
    "GrC": "unkonwn",
}


def assert_presence(datasets_list, name):
    return any(name in dataset for dataset in datasets_list)


def save_results(results_dict, save_path):
    today = datetime.datetime.now().strftime("%d_%b")
    file_name = os.path.join(save_path, f"results_{today}.pkl")

    with open(file_name, "wb") as fobj:
        pickle.dump(results_dict, fobj)


def get_paths_from_dir(
    path_to_dir,
    include_lisberger=False,
    include_medina=False,
    _include_hull_unlab=False,
):
    # Check if the directory structure is correct
    files = os.listdir(path_to_dir)

    if include_lisberger:
        datasets = [
            dataset
            for dataset in files
            if dataset.endswith(".h5") and "medina" not in dataset
        ]
    elif include_medina:
        datasets = [
            dataset
            for dataset in files
            if dataset.endswith(".h5") and "lisberger" not in dataset
        ]
    elif include_lisberger and include_medina:
        datasets = [dataset for dataset in files if dataset.endswith(".h5")]
    else:
        datasets = [
            dataset
            for dataset in files
            if dataset.endswith(".h5")
            and "lisberger" not in dataset
            and "medina" not in dataset
        ]
    datasets_abs = [os.path.join(path_to_dir, dataset) for dataset in datasets]

    if _include_hull_unlab:
        path_hull_unlab = os.listdir(os.path.join(path_to_dir, "unlabelled"))[0]
        datasets_abs.append(os.path.join(path_to_dir, "unlabelled", path_hull_unlab))

    if include_lisberger:
        assert len(datasets) < 4, "More than three unique datasets found"
    elif include_medina:
        assert len(datasets) < 4, "More than three unique datasets found"
    elif include_lisberger and include_medina:
        assert len(datasets) < 5, "More than four unique datasets found"
    elif _include_hull_unlab:
        assert len(datasets) < 6, "More than five unique datasets found"
    else:
        assert len(datasets) < 3, "More than two unique mouse datasets found"

    assert assert_presence(datasets, "hull"), "No hull dataset found"
    assert assert_presence(datasets, "hausser"), "No hausser dataset found"
    if include_lisberger:
        assert assert_presence(datasets, "lisberger"), "No lisberger dataset found"
    if include_medina:
        assert assert_presence(datasets, "medina"), "No medina dataset found"

    return sorted(datasets_abs, key=os.path.getsize)


def combine_features(df1, df2):
    common_columns = df1.columns.intersection(df2.columns).tolist()

    # Ensure common columns have the same data type in both dataframes
    for col in common_columns:
        if df1[col].dtype != df2[col].dtype:
            if df1[col].dtype == "object":
                df1[col] = df1[col].astype(df2[col].dtype)
            elif df2[col].dtype == "object":
                df2[col] = df2[col].astype(df1[col].dtype)
            else:
                df1[col] = df1[col].astype(str)
                df2[col] = df2[col].astype(str)

    # Merge the dataframes on the common columns, using a left join
    combined_df = df1.merge(df2, on=common_columns, how="left")
    return combined_df


def extract_and_merge_datasets(
    *dataset_paths,
    quality_check=False,
    normalise_wvf=False,
    _use_amplitudes=True,
    n_channels=N_CHANNELS,
    central_range=WAVEFORM_SAMPLES,
    _bin_size=BIN_SIZE,
    _win_size=WIN_SIZE,
    labelled=True,
    **kwargs,
) -> datasets.NeuronsDataset:
    # Create a joint dataset class from the paths to the datasets
    dataset_list = []

    for dataset_path in dataset_paths:
        print(f"Loading dataset {dataset_path}")
        dataset_class = datasets.NeuronsDataset(
            dataset_path,
            quality_check=quality_check,
            normalise_wvf=normalise_wvf,
            _use_amplitudes=_use_amplitudes,
            n_channels=n_channels,
            central_range=central_range,
            _bin_size=_bin_size,
            _win_size=_win_size,
            **kwargs,
        )
        if labelled:
            dataset_class.make_labels_only()

        dataset_list.append(dataset_class)

    return datasets.merge_h5_datasets(*dataset_list)


def make_dataset_df(dataset: datasets.NeuronsDataset):
    columns = ["h5_unit_id", "label", "dataset", "unit", "genetic_line", "included"]
    rows = []

    include = 1

    for i in tqdm(
        range(len(dataset)), desc="Generating dataframe", leave=False, position=0
    ):
        dp = "/".join(dataset.info[i].split("/")[:-1])
        unit = int(dataset.info[i].split("/")[-1])
        label = dataset.labels_list[i]
        line = dataset.genetic_line_list[i]
        curr_feat = [i, label, dp, unit, line, include]
        rows.append(dict(zip(columns, curr_feat)))

    return pd.DataFrame(rows, columns=columns)


def extract_and_check(
    *dataset_paths, save=True, save_folder="./", lisberger=False, **kwargs
):
    unchecked_dataset = extract_and_merge_datasets(
        *dataset_paths, quality_check=False, **kwargs
    )

    for i in range(len(unchecked_dataset)):
        if unchecked_dataset.genetic_line_list[i] == "unknown":
            unchecked_dataset.genetic_line_list[i] = HULL_LINES_DICT[
                unchecked_dataset.labels_list[i]
            ]

    # Create a dataframe with the features of the dataset
    dataset_df = make_dataset_df(unchecked_dataset)

    quality_mask = unchecked_dataset.quality_checks_mask.astype(int)

    dataset_df["quality_check"] = quality_mask

    dataset_df["included"] = dataset_df["included"] * dataset_df["quality_check"]

    prefix = "monkey_" if lisberger else ""

    if save:
        save_path = os.path.join(save_folder, f"{prefix}dataset_info.csv")
        dataset_df.to_csv(save_path, index=False)

    return dataset_df, unchecked_dataset


def prepare_classification_dataset(
    dataset: datasets.NeuronsDataset,
    normalise_acgs=True,
    win_size=WIN_SIZE,
    bin_size=BIN_SIZE,
    multi_chan_wave=False,
    _acgs_path=None,
    _acg_mask=None,
    _acg_multi_factor=1,
):
    acgs = [] if _acgs_path is None else np.load(_acgs_path)
    waveforms = []
    for i, spikes in tqdm(
        enumerate(dataset.spikes_list),
        total=len(dataset),
        desc="Preparing waveforms and 3D acgs",
    ):
        if _acgs_path is None:
            _, acg_3d = corr.crosscorr_vs_firing_rate(
                spikes, spikes, win_size=win_size, bin_size=bin_size
            )
            acg_3d, _ = corr.convert_acg_log(acg_3d, bin_size, win_size)
            acgs.append(acg_3d.ravel())
        if not multi_chan_wave:
            processed_wave = dataset.conformed_waveforms[i]
        else:
            processed_wave = dataset.wf[i]

        waveforms.append(processed_wave)
    acgs = (
        np.stack(acgs, axis=0)
        if _acgs_path is None
        else acgs[_acg_mask if _acg_mask is not None else slice(None)]
    )
    acgs *= _acg_multi_factor
    acgs_max = np.max(acgs, axis=1)
    waveforms = np.stack(waveforms, axis=0)

    if normalise_acgs:
        acgs = np.nan_to_num(acgs / acgs_max[:, None])

    classification_dataset = np.concatenate((acgs, waveforms), axis=1)

    return classification_dataset, acgs_max


def plot_quality_checks(dataframe, lab, fig_title):
    plot = sns.barplot(
        x="label",
        y="count",
        hue="situation",
        data=dataframe.loc[dataframe["lab"] == lab],
        palette="Blues",
    )
    plot.legend(title="")
    result = int(plot.get_ylim()[1] * 1.2)
    return plot.mplp(
        figsize=(10, 5),
        title=fig_title,
        xlabel="Cell type",
        ylabel="Number of neurons",
        yticks=np.arange(0, result, result // 6),
    )


def save_quality_plots(
    dataframe: pd.DataFrame, save_folder="./", unlabelled=False, lisberger=False
):
    lab = []
    for i, row in dataframe.iterrows():
        if (
            ("YC" in row["dataset"])
            or ("DK" in row["dataset"])
            or ("MO" in row["dataset"])
            or ("HS" in row["dataset"])
        ):
            lab.append("hausser")
        else:
            lab.append("hull")
    dataframe["lab"] = lab

    checked_df = dataframe.groupby(["lab", "label"])["quality_check"].sum()
    checked_df = checked_df.reset_index()
    checked_df.rename(columns={"quality_check": "count"}, inplace=True)
    checked_df["situation"] = "After quality checks"

    after_feat_df = dataframe.groupby(["lab", "label"])["included"].sum()
    after_feat_df = after_feat_df.reset_index()
    after_feat_df.rename(columns={"included": "count"}, inplace=True)
    after_feat_df["situation"] = "After eng. features extraction"

    grouped_df = dataframe.groupby(["lab", "label"])["quality_check"].count()
    grouped_df = grouped_df.reset_index()
    grouped_df.rename(columns={"quality_check": "count"}, inplace=True)
    grouped_df["situation"] = "Before quality checks"

    merged_df = pd.merge(grouped_df, checked_df, how="outer")
    merged_df = pd.merge(merged_df, after_feat_df, how="outer")

    # order by higher count
    merged_df.sort_values(by=["count", "situation"], ascending=False, inplace=True)

    postfix = "_unlabelled" if unlabelled else ""

    if lisberger:
        save_and_close(
            merged_df,
            "hull",
            "Lisberger dataset",
            os.path.join(save_folder, f"lisberger_quality_check{postfix}.pdf"),
        )
    else:
        save_and_close(
            merged_df,
            "hull",
            "Hull dataset",
            os.path.join(save_folder, f"hull_quality_check{postfix}.pdf"),
        )
        save_and_close(
            merged_df,
            "hausser",
            "Hausser dataset",
            os.path.join(save_folder, f"hausser_quality_check{postfix}.pdf"),
        )


def save_and_close(merged_df, lab, fig_title, file_name):
    fig = plot_quality_checks(merged_df, lab, fig_title)
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_acg(spike_train, unit_n, save_name=None):
    if save_name is None:
        raise NotImplementedError("Please specify a save name")

    # Get the spike train in the spontaneous period that meets false positive and false negative criteria.

    if len(spike_train.ravel()) > 1:
        plt.figure()
        plot.plot_acg(".npyx_placeholder", unit_n, train=spike_train, figsize=(5, 4.5))

        plt.savefig(f"{save_name}-acg.pdf", format="pdf", bbox_inches="tight")
    else:
        quality_failed_plot(save_name)

    plt.close()


def quality_failed_plot(save_name):
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    ax.axis([0, 10, 0, 10])
    ax.text(
        3,
        5,
        "No usable train \n after quality checks",
        style="italic",
        bbox={"facecolor": "red", "alpha": 0.4, "pad": 10},
    )
    plt.savefig(f"{save_name}-acg.pdf", format="pdf", bbox_inches="tight")


def save_wvf(waveform, save_name=None):
    if save_name is None:
        raise NotImplementedError("Please specify a save name")

    plt.figure()
    plot.plt_wvf(waveform, figh_inch=8, figw_inch=5, title=str(save_name))

    plt.savefig(f"{save_name}-wvf.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def make_summary_plots_wvf(
    dataset: datasets.NeuronsDataset, save_folder=".", monkey=False
):
    prefix = "monkey_" if monkey else ""
    for lab in np.unique(dataset.targets):
        lab_mask = dataset.targets == lab
        lab_wvf = dataset.wf[lab_mask]
        col = plot.to_hex(COLORS_DICT[CORRESPONDENCE[lab]])
        for wf in lab_wvf:
            # Normalise before plotting
            peak_wf = wf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)[N_CHANNELS // 2]
            peak_wf = peak_wf / np.max(np.abs(peak_wf))
            plt.plot(
                peak_wf,
                color=col,
                alpha=0.4,
            )
        fig = plot.mplp(
            title=f"{CORRESPONDENCE[lab]} (n = {len(lab_wvf)})",
            xlabel="Time (ms)",
            ylabel="Amplitude (a.u.)",
            figsize=(6, 5),
            xtickslabels=np.arange(
                -(WAVEFORM_SAMPLES // 2) / 30,
                ((WAVEFORM_SAMPLES // 2) + 1) / 30,
                10 / 30,
            ).round(1),
            xticks=np.arange(0, WAVEFORM_SAMPLES + 1, 10),
        )
        plot.save_mpl_fig(
            fig[0],
            f"{prefix}summary_peak_wvf_{CORRESPONDENCE[lab]}",
            save_folder,
            "pdf",
        )
        plt.close()


def get_relevant_waveform(wf, n_channels=N_CHANNELS, central_range=WAVEFORM_SAMPLES):
    if wf.ndim == 1:
        wf = wf.reshape(n_channels, central_range)
    high_amp_channels = feat.filter_out_waves(wf, n_channels // 2)
    (
        candidate_channel_somatic,
        candidate_channel_non_somatic,
        somatic_mask,
        _,
    ) = feat.detect_peaks_2d(wf, high_amp_channels)

    # First find working waveform
    relevant_waveform, _, _ = feat.find_relevant_waveform(
        wf,
        candidate_channel_somatic,
        candidate_channel_non_somatic,
        somatic_mask,
    )

    return relevant_waveform


def make_summary_plots_preprocessed_wvf(
    dataset: datasets.NeuronsDataset, save_folder=".", monkey=False
):
    prefix = "monkey_" if monkey else ""
    for lab in np.unique(dataset.targets):
        lab_mask = dataset.targets == lab
        lab_wvf = dataset.wf[lab_mask]
        col = plot.to_hex(COLORS_DICT[CORRESPONDENCE[lab]])
        for wf in lab_wvf:
            relevant_waveform = datasets.preprocess_template(
                wf,
                clip_size=[
                    WAVEFORM_SAMPLES / (4 * 30_000),
                    WAVEFORM_SAMPLES / (2 * 30_000),
                ],
            )

            plt.plot(
                relevant_waveform,
                color=col,
                alpha=0.4,
            )

        fig = plot.mplp(
            title=f"{CORRESPONDENCE[lab]}",
            xlabel="Time (ms)",
            ylabel="Amplitude (a.u.)",
            figsize=(6, 5),
            xtickslabels=np.arange(
                -(WAVEFORM_SAMPLES // 4) / 30,
                ((WAVEFORM_SAMPLES // 2) + 1) / 30,
                10 / 30,
            ).round(1),
            xticks=np.arange(0, int(WAVEFORM_SAMPLES * 3 / 4 + 1), 10),
        )
        plot.save_mpl_fig(
            fig[0],
            f"{prefix}summary_preprocessed_wvf_{CORRESPONDENCE[lab]}",
            save_folder,
            "pdf",
        )
        plt.close()


def make_summary_plots_relevant_wvf(
    dataset: datasets.NeuronsDataset, save_folder=".", monkey=False
):
    prefix = "monkey_" if monkey else ""
    for lab in np.unique(dataset.targets):
        lab_mask = dataset.targets == lab
        lab_wvf = dataset.wf[lab_mask]
        col = plot.to_hex(COLORS_DICT[CORRESPONDENCE[lab]])
        for wf in lab_wvf:
            if monkey:
                relevant_waveform = wf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)[
                    N_CHANNELS // 2, :
                ]
                if np.abs(np.min(relevant_waveform)) < np.max(relevant_waveform):
                    relevant_waveform = -relevant_waveform

            else:
                relevant_waveform = get_relevant_waveform(wf)
                if relevant_waveform is None:
                    continue
            # Normalise before plotting
            relevant_waveform = relevant_waveform / np.max(np.abs(relevant_waveform))

            plt.plot(
                relevant_waveform,
                color=col,
                alpha=0.4,
            )

        fig = plot.mplp(
            title=f"{CORRESPONDENCE[lab]}",
            xlabel="Time (ms)",
            ylabel="Amplitude (a.u.)",
            figsize=(6, 5),
            xtickslabels=np.arange(
                -(WAVEFORM_SAMPLES // 2) / 30,
                ((WAVEFORM_SAMPLES // 2) + 1) / 30,
                10 / 30,
            ).round(1),
            xticks=np.arange(0, WAVEFORM_SAMPLES + 1, 10),
        )
        plot.save_mpl_fig(
            fig[0],
            f"{prefix}summary_somatic_wvf_{CORRESPONDENCE[lab]}",
            save_folder,
            "pdf",
        )
        plt.close()


def make_summary_plots_wvf_by_line(dataset: datasets.NeuronsDataset, save_folder="."):
    for lab in np.unique(dataset.targets):
        if lab in [LABELLING["PkC_ss"], LABELLING["PkC_cs"]]:
            continue
        lab_mask = dataset.targets == lab

        for line in np.unique(np.array(dataset.genetic_line_list)[lab_mask]):
            line_mask = np.array(dataset.genetic_line_list) == line
            wf_mask = lab_mask & line_mask

            lab_wvf = dataset.wf[wf_mask]
            col = plot.to_hex(COLORS_DICT[CORRESPONDENCE[lab]])

            if len(lab_wvf) == 0:
                continue

            for wf in lab_wvf:
                relevant_waveform = get_relevant_waveform(wf)
                # Normalise before plotting

                relevant_waveform = relevant_waveform / np.max(
                    np.abs(relevant_waveform)
                )

                if relevant_waveform is None:
                    continue

                # Normalise before plotting
                plt.plot(
                    relevant_waveform,
                    color=col,
                    alpha=0.4,
                )
            fig = plot.mplp(
                title=f"{CORRESPONDENCE[lab]}, {line} line (n = {len(lab_wvf)})",
                xlabel="Time (ms)",
                ylabel="Amplitude (a.u.)",
                figsize=(6, 5),
                xtickslabels=np.arange(
                    -(WAVEFORM_SAMPLES // 2) / 30,
                    ((WAVEFORM_SAMPLES // 2) + 1) / 30,
                    10 / 30,
                ).round(1),
                xticks=np.arange(0, WAVEFORM_SAMPLES + 1, 10),
            )
            plot.save_mpl_fig(
                fig[0],
                f"summary_peak_wvf_{CORRESPONDENCE[lab]}_{line}",
                save_folder,
                "pdf",
            )
            plt.close()


def make_summary_plots_acg_by_line(dataset: datasets.NeuronsDataset, save_folder="."):
    for lab in np.unique(dataset.targets):
        lab_mask = dataset.targets == lab

        if lab in [LABELLING["PkC_ss"], LABELLING["PkC_cs"]]:
            continue
        lab_mask = dataset.targets == lab

        for line in np.unique(np.array(dataset.genetic_line_list)[lab_mask]):
            line_mask = np.array(dataset.genetic_line_list) == line
            acg_mask = lab_mask & line_mask

            lab_acg = np.array(dataset.acg_list)[acg_mask]
            col = plot.to_hex(COLORS_DICT[CORRESPONDENCE[lab]])

            if len(lab_acg) == 0:
                continue

            for acg in lab_acg:
                plt.plot(acg, color=col, alpha=0.4)
            fig = plot.mplp(
                title=f"{CORRESPONDENCE[lab]}, {line} (n = {len(lab_acg)})",
                xlabel="Time (ms)",
                ylabel="Autocorrelation (Hz)",
                figsize=(6, 5),
                xtickslabels=np.arange(
                    -(WIN_SIZE // 2), ((WIN_SIZE // 2) + 1), WIN_SIZE // 8
                ).round(1),
                xticks=np.arange(
                    0, int(WIN_SIZE / BIN_SIZE) + 1, int(WIN_SIZE / BIN_SIZE) // 8
                ),
            )
            plot.save_mpl_fig(
                fig[0], f"summary_acg_{CORRESPONDENCE[lab]}_{line}", save_folder, "pdf"
            )
            plt.close()


def make_summary_plots_acg(
    dataset: datasets.NeuronsDataset, save_folder=".", monkey=False
):
    prefix = "monkey_" if monkey else ""
    for lab in np.unique(dataset.targets):
        lab_mask = dataset.targets == lab
        lab_acg = np.array(dataset.acg_list)[lab_mask]
        col = plot.to_hex(COLORS_DICT[CORRESPONDENCE[lab]])
        for acg in lab_acg:
            plt.plot(acg, color=col, alpha=0.4)
        fig = plot.mplp(
            title=f"{CORRESPONDENCE[lab]} (n = {len(lab_acg)})",
            xlabel="Time (ms)",
            ylabel="Autocorrelation (Hz)",
            figsize=(6, 5),
            xtickslabels=np.arange(
                -(WIN_SIZE // 2), ((WIN_SIZE // 2) + 1), WIN_SIZE // 8
            ).round(1),
            xticks=np.arange(
                0, int(WIN_SIZE / BIN_SIZE) + 1, int(WIN_SIZE / BIN_SIZE) // 8
            ),
        )
        plot.save_mpl_fig(
            fig[0], f"{prefix}summary_acg_{CORRESPONDENCE[lab]}", save_folder, "pdf"
        )
        plt.close()


def make_raw_plots(
    dataset: datasets.NeuronsDataset, path_to_dir=".", folder="raw_plots"
):
    save_folder = os.path.join(path_to_dir, folder)

    # If the save folder does not exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_folder_wvf = os.path.join(save_folder, "wvf")
    save_folder_acg = os.path.join(save_folder, "acg")

    if not os.path.exists(save_folder_wvf):
        os.makedirs(save_folder_wvf)
    if not os.path.exists(save_folder_acg):
        os.makedirs(save_folder_acg)

    for plotting_id in tqdm(
        range(len(dataset)), desc="Plotting raw plots", leave=False
    ):
        label = dataset.labels_list[plotting_id]
        line = dataset.genetic_line_list[plotting_id]
        absolute_info = "_".join(dataset.info[plotting_id].split("/"))

        fname = "_".join(
            [str(label), "neuron", str(plotting_id), str(line), absolute_info]
        )

        save_id_wvf = os.path.join(save_folder_wvf, fname)
        save_id_acg = os.path.join(save_folder_acg, fname)

        spike_train = dataset.spikes_list[plotting_id]
        waveform = dataset.wf[plotting_id].reshape(N_CHANNELS, WAVEFORM_SAMPLES)
        save_acg(spike_train, plotting_id, save_name=f"{str(save_id_acg)}")
        save_wvf(waveform.T, save_name=f"{str(save_id_wvf)}")

    return


def find_unusable_index(feat_df, dataset_df, unusable_features_idx):
    selected_columns = ["label", "dataset", "unit"]

    unusable_df = feat_df.iloc[unusable_features_idx]
    unusable_df = unusable_df[selected_columns]
    dataset_df_cut = dataset_df[selected_columns]
    mask = dataset_df_cut.isin(unusable_df.to_dict("list")).all(axis=1)

    return mask[mask].index


def report_unusable_features(
    feat_df, dataset_df, unusable_features_idx, args, save=True, lisberger=False
):
    absolute_unusable_idx = find_unusable_index(
        feat_df, dataset_df, unusable_features_idx
    )
    prefix = "monkey_" if lisberger else ""
    dataset_df_path = os.path.join(args.data_folder, f"{prefix}dataset_info.csv")
    dataset_df = pd.read_csv(dataset_df_path)
    features_ok = np.ones(len(dataset_df), dtype=bool)
    features_ok[absolute_unusable_idx] = False
    dataset_df["features_ok"] = features_ok.astype(int)
    # dataset_df["included"] = dataset_df["features_ok"] * dataset_df["included"]
    new_dataset_df_path = os.path.join(
        args.data_folder, args.name, f"{prefix}dataset_info.csv"
    )
    if save:
        dataset_df.to_csv(new_dataset_df_path, index=False)

    print(
        f"Some neurons failed feature extraction: \n {feat_df.iloc[unusable_features_idx]}"
    )
    # print(f"Corresponding IDs in dataset_df: {absolute_unusable_idx}")
    return dataset_df


def save_features(
    feat_df, features_name, args, bad_idx=None, drop_cols=None, monkey=False
):
    if hasattr(args, "name"):
        features_path = os.path.join(args.data_folder, args.name, features_name)
    else:
        features_path = os.path.join(args.data_folder, features_name)
    if not os.path.exists(features_path):
        os.mkdir(features_path)

    features, labels = feat.prepare_classification(
        feat_df, bad_idx=bad_idx, drop_cols=drop_cols
    )

    prefix = "monkey_" if monkey else ""

    features.to_csv(os.path.join(features_path, f"{prefix}features.csv"), index=False)
    labels.to_csv(os.path.join(features_path, f"{prefix}labels.csv"), index=False)


def make_plots_folder(args):
    plots_folder = os.path.join(args.data_folder, "plots")
    if not os.path.exists(plots_folder):
        os.mkdir(plots_folder)
    return plots_folder


def summary_plots(args, dataset_class, by_line=True, raw=True, monkey=False):
    plots_folder = make_plots_folder(args)
    # Make summary plots
    make_summary_plots_acg(dataset_class, save_folder=plots_folder, monkey=monkey)
    make_summary_plots_wvf(dataset_class, save_folder=plots_folder, monkey=monkey)
    make_summary_plots_relevant_wvf(
        dataset_class, save_folder=plots_folder, monkey=monkey
    )
    make_summary_plots_preprocessed_wvf(
        dataset_class, save_folder=plots_folder, monkey=monkey
    )

    if by_line:
        by_line_folder = os.path.join(plots_folder, "by_line")
        if not os.path.exists(by_line_folder):
            os.mkdir(by_line_folder)
        make_summary_plots_wvf_by_line(dataset_class, save_folder=by_line_folder)
        make_summary_plots_acg_by_line(dataset_class, save_folder=by_line_folder)

    # Make raw plots
    if raw:
        make_raw_plots(dataset_class, path_to_dir=plots_folder)

    return plots_folder


def calc_snr(wvf, noise_samples=15, return_db=False):
    # Assume the first 10 samples as the noise bed
    noise_samples = wvf[:noise_samples]

    # Calculate the noise power (RMS squared)
    RMS_noise = np.sqrt(np.mean(noise_samples**2))

    # Find the absolute peak (either positive or negative) as the signal
    V_signal = np.max(np.abs(wvf))

    # Calculate the signal power
    P_signal = V_signal**2

    # Calculate the SNR in linear scale and in decibels
    SNR_linear = P_signal / (RMS_noise**2)
    return 10 * np.log10(SNR_linear) if return_db else SNR_linear


def main():
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

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="dataset_1",
        help="Name assigned to the dataset.",
    )

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=True)

    parser.add_argument("--WM", action="store_true")
    parser.set_defaults(WM=False)

    args = parser.parse_args()

    datasets_abs = get_paths_from_dir(args.data_folder)

    # Extract and check the datasets, saving a dataframe with the results
    dataset_df, dataset_class = extract_and_check(
        *datasets_abs, save_folder=args.data_folder
    )

    if args.plot:
        plots_folder = summary_plots(args, dataset_class)
    # Do feature extraction and keep track of neurons lost in the process
    quality_checked_dataset = dataset_class.apply_quality_checks()

    dataset_inner_path = os.path.join(args.data_folder, args.name)
    if not os.path.exists(dataset_inner_path):
        os.mkdir(dataset_inner_path)

    feat_df = feat.h5_feature_extraction(
        quality_checked_dataset,
        save_path=os.path.join(dataset_inner_path, "engineered_features.csv"),
        _wvf_type="flipped",
    )

    unusable_features_idx = feat.get_unusable_features(feat_df)

    if len(unusable_features_idx) > 0:
        dataset_df = report_unusable_features(
            feat_df, dataset_df, unusable_features_idx, args
        )

    # Print to a text file called readme the cell counts in dataset.info
    with open(os.path.join(args.data_folder, args.name, "readme.txt"), "w") as f:
        f.write("Cell counts in dataset.info:\n")
        f.write(
            dataset_df.groupby(["label", "genetic_line"])["included"]
            .sum()
            .to_frame()
            .to_markdown(tablefmt="grid")
        )
        f.write("\n")

    # Save quality checks and feature extraction inclusion results
    if args.plot:
        save_quality_plots(dataset_df, save_folder=plots_folder)

    # Divide the features dataframe into three main folders, one for waveform features,
    # one for temporal features and one for the combined
    temporal_features = feat.FEATURES[:20]
    waveform_features = feat.FEATURES[:3] + feat.FEATURES[20:]

    save_features(
        feat_df,
        "engineered_waveform_features",
        args,
        bad_idx=unusable_features_idx,
        drop_cols=temporal_features + ["relevant_channel", "any_somatic", "max_peaks"],
    )
    save_features(
        feat_df,
        "engineered_temporal_features",
        args,
        bad_idx=unusable_features_idx,
        drop_cols=waveform_features,
    )
    save_features(
        feat_df, "engineered_combined_features", args, bad_idx=unusable_features_idx
    )

    ### Generating raw features dataframes

    raw_waveforms_multi_channel = quality_checked_dataset.wf

    relevant_waveforms = []
    common_preprocessing = []
    common_preprocessing_unnormalised = []

    for wf in quality_checked_dataset.wf:
        relevant_waveform = get_relevant_waveform(wf)
        if relevant_waveform is None:
            relevant_waveform = wf.reshape(N_CHANNELS, WAVEFORM_SAMPLES)[
                N_CHANNELS // 2, :
            ]

        relevant_waveforms.append(relevant_waveform)
        common_preprocessing.append(
            datasets.preprocess_template(wf, clip_size=[1e-3, 2e-3])
        )
        common_preprocessing_unnormalised.append(
            datasets.preprocess_template(wf, clip_size=[1e-3, 2e-3], normalize=False)
        )

    relevant_waveforms = np.array(relevant_waveforms)
    common_preprocessing = np.array(common_preprocessing)
    common_preprocessing_unnormalised = np.array(common_preprocessing_unnormalised)

    labels = quality_checked_dataset.labels_list

    raw_wvf_multi_df = pd.DataFrame(raw_waveforms_multi_channel)
    lab_df = pd.DataFrame({"label": labels})
    raw_wvf_single_df = pd.DataFrame(relevant_waveforms)
    raw_wvf_single_norm_df = pd.DataFrame(
        relevant_waveforms / np.max(np.abs(relevant_waveforms), axis=1)[:, None]
    )
    raw_wvf_single_common_preprocessing_df = pd.DataFrame(common_preprocessing)
    raw_wvf_single_common_preprocessing_unnormalised_df = pd.DataFrame(
        common_preprocessing_unnormalised
    )
    raw_acgs_df = pd.DataFrame(quality_checked_dataset.acg)

    save_features(
        pd.concat([lab_df, raw_wvf_multi_df], axis=1),
        "raw_waveforms_multi_channel",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat([lab_df, raw_wvf_single_df], axis=1),
        "raw_waveforms_single_channel",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat([lab_df, raw_wvf_single_norm_df], axis=1),
        "raw_waveforms_single_channel_normalised",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat([lab_df, raw_wvf_single_common_preprocessing_df], axis=1),
        "raw_waveforms_single_channel_common_preprocessing_normalised",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat(
            [lab_df, raw_wvf_single_common_preprocessing_unnormalised_df], axis=1
        ),
        "raw_waveforms_single_channel_common_preprocessing_unnormalised",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat([lab_df, raw_acgs_df], axis=1),
        "raw_acgs",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat([lab_df, raw_wvf_multi_df, raw_acgs_df], axis=1),
        "raw_combined_features_multi_channel",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat([lab_df, raw_wvf_single_df, raw_acgs_df], axis=1),
        "raw_combined_features_single_channel",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    save_features(
        pd.concat(
            [lab_df, raw_wvf_single_common_preprocessing_df, raw_acgs_df], axis=1
        ),
        "raw_combined_features_single_channel_common_preprocessing",
        args,
        bad_idx=None,
        drop_cols=["label"],
    )

    ############################
    if args.WM:
        single_channel_preprocessed = []
        paths = []
        units = []
        labels = []
        snrs = []

        for i, wave in tqdm(
            enumerate(quality_checked_dataset.wf),
            total=len(quality_checked_dataset.wf),
            desc="Preparing WM...",
        ):
            dp = quality_checked_dataset.info[i].split("/")[0]
            unit = quality_checked_dataset.info[i].split("/")[1]
            waveform_2d = wave.reshape(N_CHANNELS, -1)
            for chan, wf in enumerate(waveform_2d):
                snr = calc_snr(wf, return_db=True)

                # Skip waveforms with SNR < 20, but keep them if they are on the peak channel.
                if snr <= 20 and chan != N_CHANNELS // 2:
                    continue
                single_channel_preprocessed.append(
                    datasets.preprocess_template(wf, clip_size=[1e-3, 2e-3])
                )
                paths.append(dp)
                units.append(unit)
                labels.append(quality_checked_dataset.labels_list[i])
                snrs.append(snr)
        metadata_df = pd.DataFrame(
            {"label": labels, "dataset": paths, "unit": units, "snr": snrs}
        )
        preprocessed_df = pd.DataFrame(single_channel_preprocessed)

        singchan_preprocessed_df = pd.concat([metadata_df, preprocessed_df], axis=1)

        save_features(
            singchan_preprocessed_df,
            "raw_waveforms_single_channel_common_preprocessing_normalised_for_WM",
            args,
            bad_idx=None,
            drop_cols=["label"],
        )


if __name__ == "__main__":
    main()
