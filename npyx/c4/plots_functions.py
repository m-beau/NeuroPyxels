from datetime import date

import matplotlib.pyplot as plt
import matplotlib
# try:
#     matplotlib.use('TkAgg')
# except:
#     pass

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import npyx.feat as feat
import npyx.plot as npyx_plot
from npyx.datasets import CORRESPONDENCE_NO_GRC
from npyx.plot import mplp
from npyx.utils import npa

from .dataset_init import BIN_SIZE, N_CHANNELS, WAVEFORM_SAMPLES, WIN_SIZE

C4_COLORS = {
    "PkC_ss": [28, 120, 181],
    "PkC_cs": [0, 0, 0],
    "MLI": [224, 85, 159],
    "MFB": [214, 37, 41],
    "GrC": [42, 161, 73],
    "GoC": [56, 174, 62],
    "laser": [96, 201, 223],
    "drug": [239, 126, 34],
    "background": [244, 242, 241],
    "MLI_A": [224, 85, 150],
    "MLI_B": [220, 80, 160],
}


def make_plotting_df(
    df: pd.DataFrame, save: bool = True, save_path: str = None
) -> pd.DataFrame:
    dataframe = df.copy()

    colors_dict = C4_COLORS
    # Add an 'absolute id' like column for plotting purposes, so that each neuron has an unique identifier
    abs_id = np.arange(len(dataframe))

    dataframe.insert(loc=0, column="plotting_id", value=abs_id)

    new_df = pd.DataFrame(
        columns=[
            "label",
            "feature",
            "normalised_value",
            "raw_value",
            "dp",
            "unit",
            "color",
            "plotting_id",
        ]
    )

    # Why the hard-coded 4? The first 4 columns in the df are metadata (dp, unit, plotting_id, label)
    for i, column in enumerate(dataframe.columns[4:]):
        mean_data = (dataframe[column].to_numpy()).mean()
        std_data = (dataframe[column].to_numpy()).std()
        norm_data = (dataframe[column].to_numpy() - mean_data) / std_data
        raw_data = dataframe[column].to_numpy()
        labels = dataframe["label"].to_numpy()
        feature = [dataframe.columns[i + 4]] * len(norm_data)
        dp = dataframe["dp"].to_numpy()
        unit = dataframe["unit"].to_numpy()
        color = [colors_dict[label] for label in labels]
        plotting_id = dataframe["plotting_id"].to_numpy()

        feat_df = pd.DataFrame(
            {
                "label": labels,
                "feature": feature,
                "normalised_value": norm_data,
                "raw_value": raw_data,
                "dp": dp,
                "unit": unit,
                "color": color,
                "plotting_id": plotting_id,
            }
        )
        new_df = pd.concat([new_df, feat_df], ignore_index=True)

    # Finally assign color grey to neurons with invalid temporal features
    zero_temporal = new_df[new_df["raw_value"] == 0.0]["unit"].value_counts()
    zero_units = (zero_temporal[zero_temporal > 5]).index.to_numpy()
    new_df.loc[new_df["unit"].isin(zero_units), "color"] = "gray"

    # Save if the option is requested
    if save:
        today = date.today().strftime("%b-%d-%Y")
        if save_path is None:
            new_df.to_csv(f"{today}_dashboard_features.csv")
        else:
            new_df.to_csv(f"{save_path}/{today}_dashboard_features.csv")

    return new_df


def save_acg(spike_train, unit_n, save_name=None):
    if save_name is None:
        raise NotImplementedError("Please specify a save name")

    # Get the spike train in the spontaneous period that meets false positive and false negative criteria.

    if len(spike_train.ravel()) > 1:
        plt.figure()
        npyx_plot.plot_acg(None, unit_n, train=spike_train, figsize=(5, 4.5))

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
    npyx_plot.plt_wvf(waveform, figh_inch=6, figw_inch=5)

    plt.savefig(f"{save_name}-wvf.pdf", format="pdf", bbox_inches="tight")
    plt.close()


def save_amplitudes(times, amplitudes, dpi=300, save_name=None):
    if save_name is None:
        raise NotImplementedError("Please specify a save name")

    if len(times) > len(amplitudes):
        times = times[: len(amplitudes)]
    if len(amplitudes) > len(times):
        amplitudes = amplitudes[: len(times)]

    fig, ax = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
    )
    ax[0].plot(
        times / (30000 * 60), amplitudes, marker="o", markersize=2, linestyle="None"
    )
    ax[0].set_xlabel("Time (min)")
    ax[0].set_ylabel(r"Amplitude ($\mu$V)")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    ax[1].hist(amplitudes, bins=200, orientation="horizontal")
    ax[1].set_xlabel("Count")
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"{save_name}-amplitudes.png", format="png", dpi=dpi, bbox_inches="tight"
    )
    plt.close()


def plot_confusion_from_proba(
    true_targets: np.ndarray,
    predicted_proba: np.ndarray,
    correspondence: dict,
    threshold: float = 0.0,
    model_name: str = "model",
    save: bool = False,
    savename: str = None,
    axis: plt.Axes = None,
    _shuffle_matrix: list = None,
) -> plt.Axes:
    """
    Plot confusion matrix from model predictions and true targets.
    The plot will also include the percentage of unclassified samples given the threshold.
    """
    if _shuffle_matrix:
        sorted_correspondence = dict(sorted(correspondence.items()))
        correspondence = dict(
            zip(
                sorted_correspondence.keys(),
                np.array(list(sorted_correspondence.values()))[_shuffle_matrix],
            )
        )
        correspondence = dict(sorted(correspondence.items()))
        predicted_proba = predicted_proba[:, _shuffle_matrix]
        true_targets = np.array([_shuffle_matrix.index(x) for x in true_targets])

    predictions = np.argmax(predicted_proba, axis=1)
    pred_values = predicted_proba[
        np.arange(len(predicted_proba)), predictions
    ].squeeze()

    sup_threshold = pred_values > threshold

    actual_predictions = predictions[sup_threshold]
    unclassified = predictions[~sup_threshold]
    label_unclass, count_unclass = np.unique(unclassified, return_counts=True)

    if len(count_unclass) == 0:
        count_unclass = np.zeros(len(correspondence.keys()))
    elif len(count_unclass) < len(correspondence.keys()):
        unclass_per_lab = dict(zip(label_unclass, count_unclass))
        for i in correspondence:
            if i not in unclass_per_lab.keys():
                unclass_per_lab[i] = 0

        sorted_dict = dict(sorted(unclass_per_lab.items()))
        count_unclass = np.array(list(sorted_dict.values()))

    confusion = confusion_matrix(
        true_targets[sup_threshold],
        actual_predictions,
        labels=list(range(len(correspondence.keys()))),
    )

    unclass_confusion = np.concatenate(
        (confusion, count_unclass.reshape(-1, 1)), axis=1
    )

    unclass_confusion_last = (
        (unclass_confusion / unclass_confusion.sum(axis=1)[:, None]) * 100
    )[:, -1]

    mean_confusion = (
        unclass_confusion / unclass_confusion[:, :-1].sum(axis=1)[:, None]
    ) * 100

    mean_confusion[:, -1] = unclass_confusion_last
    mean_confusion = np.nan_to_num(mean_confusion, nan=0.0)

    for i, _ in enumerate(mean_confusion):
        if i not in true_targets:
            mean_confusion[i, :] = np.nan

    if axis is None:
        ax = plt.gca()
        ax.figure.set_size_inches(10, 9)

    ax = sns.heatmap(
        mean_confusion,
        annot=mean_confusion,
        cmap="viridis",
        cbar=True,
        linewidths=1,
        linecolor="black",
        fmt=".3g",
        square=True,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )

    # Add annotations for NaN values
    for i in range(mean_confusion.shape[0]):
        for j in range(mean_confusion.shape[1]):
            if np.isnan(mean_confusion[i, j]):
                ax.text(
                    j + 0.5, i + 0.5, "nan", ha="center", va="center", color="black"
                )

    x_labels = [
        int(ax.get_xticklabels()[i].get_text())
        for i in range(len((ax.get_xticklabels())))
    ]
    y_labels = [
        int(ax.get_yticklabels()[i].get_text())
        for i in range(len(ax.get_yticklabels()))
    ]

    correspondence_unlab = dict(correspondence)
    correspondence_unlab[max(correspondence.keys()) + 1] = "unclassified"

    ax.set_xticklabels(
        pd.Series(x_labels).replace(to_replace=correspondence_unlab).to_numpy(),
        fontsize=12,
    )
    ax.set_yticklabels(
        pd.Series(y_labels).replace(to_replace=correspondence).to_numpy(),
        fontsize=12,
    )
    yl = ax.get_ylim()
    ax.plot([len(correspondence), len(correspondence)], yl, color="white", lw=4)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)

    if axis is None:
        ax.set_title(
            f"Mean confusion matrix - {model_name}.\nThreshold: {threshold}",
            fontsize=15,
            fontweight="bold",
        )

    if save:
        if savename is not None:
            plt.savefig(savename, bbox_inches="tight")
        else:
            plt.savefig(f"confusion_matrix_{model_name}.pdf", bbox_inches="tight")
    return ax


def plot_results_from_threshold(
    true_targets: np.ndarray,
    predicted_proba: np.ndarray,
    correspondence: dict,
    threshold: float = 0.0,
    model_name: str = "model",
    kde_bandwidth: float = 0.02,
    collapse_classes: bool = False,
    f1_scores: np.ndarray = None,
    save: bool = False,
    savename: str = None,
    _shuffle_matrix: list = None,
    _folds_stddev: np.ndarray = None,
):
    fig, ax = plt.subplots(
        1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    colors_dict = C4_COLORS

    if collapse_classes:
        all_true_positives = []
        all_false_positives = []

    for label in range(len(correspondence.keys())):
        cell_type = correspondence[label]
        col = npyx_plot.to_hex(colors_dict[cell_type])
        predictions = np.argmax(predicted_proba, axis=1)

        predicted_label_mask = predictions == label
        true_label_mask = true_targets == label

        true_positive_p = predicted_proba[predicted_label_mask & true_label_mask, label]
        false_positive_p = predicted_proba[
            predicted_label_mask & (~true_label_mask), label
        ]

        if collapse_classes:
            all_true_positives.append(true_positive_p)
            all_false_positives.append(false_positive_p)
            continue

        skip_tp, skip_fp = False, False
        try:
            density_correct_factor_tp = (
                len(true_positive_p) + len(false_positive_p)
            ) / len(true_positive_p)
        except ZeroDivisionError:
            skip_tp = True

        try:
            density_correct_factor_fp = (
                len(true_positive_p) + len(false_positive_p)
            ) / len(false_positive_p)
        except ZeroDivisionError:
            skip_fp = True

        if not skip_tp:
            kde = sm.nonparametric.KDEUnivariate(true_positive_p)
            kde.fit(bw=kde_bandwidth)  # Estimate the densities
            ax[0].fill_between(
                kde.support,
                kde.density * 0,
                kde.density / 100 / density_correct_factor_tp,
                label=f"Pr(f(x)={cell_type}|{cell_type})",
                facecolor=col,
                lw=2,
                alpha=0.5,
            )

        if not skip_fp:
            kde = sm.nonparametric.KDEUnivariate(false_positive_p)
            kde.fit(bw=kde_bandwidth)  # Estimate the densities
            ax[0].plot(
                kde.support,
                kde.density / 100 / density_correct_factor_fp,
                label=f"Pr(f(x)={cell_type}|¬{cell_type})",
                color=col,
                lw=2,
                alpha=0.8,
            )

    if collapse_classes:
        plot_collapsed_densities(
            all_true_positives, all_false_positives, kde_bandwidth, ax
        )
    ax[0].set_xlim([0.2, 1])
    ax[0].set_ylim([0, 0.1])
    yl = ax[0].get_ylim()
    ax[0].plot([threshold, threshold], yl, color="red", lw=3, ls="-")
    ax[0].legend(loc="upper left", fontsize=12)
    ax[0].set_xlabel("Predicted probability", fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Density", fontsize=14, fontweight="bold")

    ax[1] = plot_confusion_from_proba(
        true_targets,
        predicted_proba,
        correspondence,
        threshold,
        model_name,
        axis=ax[1],
        _shuffle_matrix=_shuffle_matrix,
    )

    # ax[1] = plot_confusion_matrix(
    #     predicted_proba,
    #     true_targets,
    #     correspondence,
    #     confidence_threshold=threshold,
    #     label_order=_shuffle_matrix,
    #     normalize=True,
    #     axis=ax[1],
    #     saveDir=None,
    #     saveFig=False,
    # )

    if f1_scores is not None:
        if type(f1_scores) not in [np.ndarray, list]:
            f1_string = f"F1 score: {f1_scores:.3f}"
        else:
            f1_string = f"Mean F1 score across {len(f1_scores)} runs: {np.mean(f1_scores):.3f}, std: {np.std(f1_scores):.3f}"
        if _folds_stddev is not None:
            f1_string += f"\n Stdev across folds: {_folds_stddev.mean():.3f}"
    else:
        f1_string = ""

    ax[1].set_xlabel("Predicted label", fontsize=14, fontweight="bold")
    ax[1].set_ylabel("True label", fontsize=14, fontweight="bold")
    fig.suptitle(
        f"\nResults for {model_name}, threshold = {threshold}\n" + f1_string,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_collapsed_densities(
    all_true_positives, all_false_positives, kde_bandwidth, ax
):
    true_positive_p = np.concatenate(all_true_positives)
    false_positive_p = np.concatenate(all_false_positives)
    kde = sm.nonparametric.KDEUnivariate(true_positive_p)
    kde.fit(bw=kde_bandwidth)  # Estimate the densities
    ax[0].fill_between(
        kde.support,
        kde.density * 0,
        kde.density / 100,
        label="Pr(f(x) = y|y)",
        facecolor="grey",
        edgecolor="black",
        lw=2,
        alpha=0.5,
    )
    ax[0].plot(kde.support, kde.density / 100, color="k")

    # ## False positives
    # if show_hist:
    #     fp_hist, fp_hist_bins = np.histogram(
    #         false_positive_p, bin_edges, density=True
    #     )
    #     ax[0].step(
    #         fp_hist_bins,
    #         np.append(fp_hist / 100, [0]),
    #         where="post",
    #         color="grey",
    #         zorder=-1,
    #     )
    kde = sm.nonparametric.KDEUnivariate(false_positive_p)
    kde.fit(bw=kde_bandwidth)  # Estimate the densities
    ax[0].fill_between(
        kde.support,
        kde.density * 0,
        kde.density / 100,
        label="Pr(f(x) = y|¬y)",
        facecolor="grey",
        lw=0,
        alpha=0.5,
    )


def plot_cosine_similarity(
    features_matrix,
    labels,
    correspondence,
    cells_name=None,
    labels_name=None,
    threshold=None,
    lines=False,
    **plotkwargs,
):
    """
    It takes a matrix of features, a vector of labels, and a dictionary that maps labels to cell types,
    and it plots a heatmap of the cosine similarity between cells, ordered by label

    Args:
      features_matrix: a matrix of shape (n_cells, n_features)
      labels: the predicted labels for each cell
      correspondence: a dictionary mapping the predicted class to the expert class
    """
    colors_dict = C4_COLORS

    features_matrix = StandardScaler().fit_transform(features_matrix)

    sorting = np.argsort(labels)
    sorted_preds = features_matrix[sorting]

    sorted_similarity_matrix = cosine_similarity(sorted_preds)

    delimiters = np.unique(labels, return_counts=True)[1].cumsum()[:-1]
    delimiters = [0] + list(delimiters) + [len(labels)]

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        sorted_similarity_matrix,
        cmap="viridis",
        ax=plt.gca(),
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"fontsize": 12, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )

    cells_name = cells_name if cells_name is not None else "unlabelled"
    labels_name = labels_name if labels_name is not None else "predicted"

    threshold_text = f"Threshold: {threshold}" if threshold is not None else ""
    ax.set_title(
        f"Cosine similarity between {cells_name} cells features. \n Ordered by {labels_name} label. {threshold_text}"
    )

    if lines:
        ax.hlines(delimiters, *ax.get_xlim(), color="white", linewidth=1)
        ax.vlines(delimiters, *ax.get_ylim(), color="white", linewidth=1)

    ax.set_xlabel("Cell index")
    ax.set_ylabel("Cell index")
    ax.set_xlim(-20, ax.get_xlim()[-1])
    ax.set_yticks([(i + j) / 2 for i, j in zip(delimiters[:-1], delimiters[1:])])
    ax.set_xticks(delimiters)
    ax.set_xticklabels(delimiters)
    ax.set_yticklabels([correspondence[i] for i in range(len(correspondence))])
    ax.set_ylabel("Class")
    for i, n in enumerate(delimiters[:-1]):
        cell_type = correspondence[i]
        ax.vlines(
            -10,
            n,
            delimiters[i + 1],
            color=npyx_plot.to_hex(colors_dict[cell_type]),
            linewidth=10,
        )
    # npyx.mplp(fig, **plotkwargs)
    plt.show()


def threshold_predictions(features, predicted_probabilties, threshold):
    predicted_values = predicted_probabilties.argmax(axis=1)

    top_pred_prob = predicted_probabilties[
        np.arange(len(predicted_probabilties)), predicted_values
    ].squeeze()

    sup_threshold = top_pred_prob > threshold

    thresholded_predictions = predicted_values[sup_threshold]
    if isinstance(features, np.ndarray):
        thresholded_features = features[sup_threshold]
    if isinstance(features, pd.DataFrame):
        thresholded_features = features.iloc[sup_threshold]

    return thresholded_features, thresholded_predictions


def plot_waveforms(waveforms, title, col=None, central_range=WAVEFORM_SAMPLES, ax=None):
    if ax is None:
        ax = plt.gca()

    for wf in waveforms:
        if len(wf) == central_range:
            ax.plot(
                wf,
                color=col,
                alpha=0.4,
            )
        else:
            relevant_waveform = get_relevant_waveform(wf)
            # Normalise before plotting

            relevant_waveform = relevant_waveform / np.max(np.abs(relevant_waveform))

            ax.plot(
                relevant_waveform,
                color=col,
                alpha=0.4,
            )

    fig, ax = npyx_plot.mplp(
        ax=ax,
        title=title,
        xlabel="Time (ms)",
        ylabel="Amplitude (a.u.)",
        figsize=(6, 5) if ax is None else None,
        xtickslabels=np.arange(
            -(central_range // 2) / 30, ((central_range // 2) + 1) / 30, 10 / 30
        ).round(1),
        xticks=np.arange(0, central_range + 1, 10),
    )
    return ax


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


def plot_acgs(acgs, title, col=None, win_size=WIN_SIZE, bin_size=BIN_SIZE, ax=None):
    if ax is None:
        ax = plt.gca()

    for acg in acgs:
        ax.plot(acg, color=col, alpha=0.4)
    fig, ax = npyx_plot.mplp(
        ax=ax,
        title=title,
        xlabel="Time (ms)",
        ylabel="Autocorrelation (Hz)",
        figsize=(6, 5) if ax is None else None,
        xtickslabels=np.arange(
            -(win_size // 2), ((win_size // 2) + 1), win_size // 8
        ).round(1),
        xticks=np.arange(
            0, int(win_size / bin_size) + 1, int(win_size / bin_size) // 8
        ),
    )

    return ax


def plot_features_1cell_vertical(
    i,
    acg_3ds,
    waveforms,
    predictions=None,
    saveDir=None,
    fig_name=None,
    plot=True,
    cbin=1,
    cwin=2000,
    figsize=(10, 4),
    LABELMAP=CORRESPONDENCE_NO_GRC,
    C4_COLORS=C4_COLORS,
    fs=30000,
    unit_id=None,
):
    # parameters
    ticklab_s = 20
    log_ticks = [10, 1000]

    if -1 in LABELMAP.keys():
        del LABELMAP[-1]

    if predictions is not None:
        n_obs, n_classes, n_models = predictions.shape
        mean_predictions = predictions.mean(2)
        pred_dict = {
            v: np.round(mean_predictions[i, k], 2) for k, v in LABELMAP.items()
        }
        pred_str = str(pred_dict).replace("{", "").replace("}", "").replace("'", "")
        pred = np.argmax(mean_predictions[i])
        confidence = mean_predictions[i, pred]
        delta_conf = np.diff(np.sort(mean_predictions[i]))[-1]
        n_votes = np.sum(np.argmax(predictions[i, :, :], axis=0) == pred)
        ct = LABELMAP[pred]
        color = [c / 255 for c in C4_COLORS[ct]]
        ttl = (
            f"Cell {i if unit_id is None else unit_id} | Prediction: {ct}\n"
            f"Confidence: \u03BC = {confidence:.2f}, \u0394\u03BC = {delta_conf:.2f}, votes = {n_votes}/{n_models}\n"
            f" {pred_str}"
        )
    else:
        color = "k"
        ttl = f"Cell {i if unit_id is None else unit_id}"
        n_classes = 5

    fig = plt.figure()
    n_rows = len(LABELMAP)
    grid = plt.GridSpec(n_rows, 6, wspace=0.1, hspace=0.8)

    # confidence
    if predictions is not None:
        row_width = n_rows // n_classes
        for cti in range(n_classes):
            ct = LABELMAP[cti]
            color_ = [c / 255 for c in C4_COLORS[ct]]
            neuron_predictions = predictions[i, cti, :]
            axx = fig.add_subplot(
                grid[cti * row_width : cti * row_width + row_width, -1]
            )
            npyx_plot.hist_MB(
                neuron_predictions, 0, 1, 0.05, ax=axx, color=color_, lw=2
            )
            mean_color = "k" if sum(color_) != 0 else "grey"
            axx.axvline(np.mean(neuron_predictions), color=mean_color, ls="--", lw=1.5)
            xtickslabels = (
                ["0"] + [""] * 4 + ["1"] if (cti == n_classes - 1) else [""] * 6
            )
            xlabel = "Confidence" if (cti == n_classes - 1) else ""
            mplp(
                fig,
                axx,
                xticks=np.arange(0, 1.2, 0.2),
                title=ct,
                title_s=10,
                yticks=[0],
                ytickslabels=[""],
                xtickslabels=xtickslabels,
                ticklab_s=14,
                axlab_s=16,
                ylabel="",
                xlabel=xlabel,
            )

    # ACG
    log_bins = np.logspace(np.log10(cbin), np.log10(cwin // 2), acg_3ds.shape[2] // 2)
    t_log = np.concatenate((-log_bins[::-1], [0], log_bins))
    log_ticks = npa(log_ticks)
    log_ticks = np.concatenate((-log_ticks[::-1], [0], log_ticks))

    acg_3d = acg_3ds[i]
    ax0 = fig.add_subplot(grid[0:2, 0:2])
    ax0.plot(t_log, acg_3d.mean(0), color=color)
    plt.xscale("symlog")
    ax0.xaxis.set_ticklabels([])
    mplp(
        fig,
        ax0,
        xticks=log_ticks,
        ticklab_s=ticklab_s,
        ylabel="Autocorr. (sp/s)",
        xlim=[t_log[0], t_log[-1]],
    )

    ax1 = fig.add_subplot(grid[2:, 0:2])
    vmax = np.max(acg_3d) * 1.1
    vmax = int(np.ceil(vmax / 10) * 10)
    plt.xscale("symlog")
    npyx_plot.imshow_cbar(
        acg_3d,
        ax=ax1,
        function="pcolor",
        xvalues=t_log,
        origin="bottom",
        xticks=log_ticks,
        xticklabels=log_ticks,
        cmapstr="viridis",
        vmin=0,
        vmax=vmax,
        cticks=[0, vmax],
    )

    mplp(fig, ax1, ticklab_s=ticklab_s, xlabel="Time (ms)", ytickslabels=[""] * 5)

    # WVF
    n_samples = waveforms.shape[1]
    t = np.arange(n_samples) / (fs / 1000)

    ax2 = fig.add_subplot(grid[1:4, 2:5])
    ax2.plot(t, waveforms[i], color=color)

    # add scalebar
    # conveniently, the axis are already in ms and microvolts
    npyx_plot.plot_scalebar(
        ax2,
        xscalebar=1,
        yscalebar=None,
        x_unit="ms",
        y_unit="\u03BCV",
        scalepad=0.025,
        fontsize=14,
        lw=3,
        loc="right",
        offset_x=0,
        offset_y=0.1,
    )

    mplp(fig, ax2, figsize, ticklab_s=ticklab_s, hide_axis=True)

    fig.suptitle(ttl, fontsize=16, va="bottom")

    if saveDir is not None:
        fig_name = f"cell {i}" if fig_name is None else fig_name
        npyx_plot.save_mpl_fig(fig, fig_name, saveDir, "pdf")

    if not plot:
        plt.close(fig)


def plot_survival_confidence(
    confidence_matrix,
    correspondence,
    ignore_below_confidence=None,
    saveDir=None,
    correspondence_colors=C4_COLORS,
):
    assert np.all(
        np.isin(np.arange(confidence_matrix.shape[1]), list(correspondence.keys()))
    ), "Keys in correspondence must be in range of confidence_matrix.shape[1]"

    # compute confidences and threshold them
    mean_confidences = confidence_matrix.mean(2).max(1)
    labels = [correspondence[i] for i in confidence_matrix.mean(2).argmax(1)]
    unique_labels = np.unique(labels)
    assert np.all(
        np.isin(unique_labels, list(correspondence_colors.keys()))
    ), "Cell types missing from correspondence_colors!"

    if ignore_below_confidence is not None:
        ignore_m = mean_confidences < ignore_below_confidence
        mean_confidences = mean_confidences[~ignore_m]
        labels = np.array(labels)[~ignore_m]

    conf_b = 0.01
    conf_thresholds = np.arange(0, 1, conf_b)

    plot_arr = np.zeros((len(unique_labels), 2, len(conf_thresholds)))
    for thi, conf_threshold in enumerate(conf_thresholds):
        for i, ct in enumerate(unique_labels):
            n_classified = ((mean_confidences > conf_threshold) & (labels == ct)).sum()
            perc_classified = n_classified / (labels == ct).sum()

            plot_arr[i, 0, thi] = n_classified
            plot_arr[i, 1, thi] = perc_classified

    # plot
    plt.figure()
    for i, ct in enumerate(unique_labels):
        color = [c / 255 for c in correspondence_colors[ct]]
        plt.plot(
            conf_thresholds,
            plot_arr[i, 0, :],
            color=color,
            lw=2,
            label=ct,
        )
    fig1, ax = mplp(
        xlabel="Confidence", ylabel="N classified", xlim=[0, 1], show_legend=True
    )

    plt.figure()
    for i, ct in enumerate(unique_labels):
        color = [c / 255 for c in correspondence_colors[ct]]
        cti = {v: k for k, v in correspondence.items()}[ct]
        plt.plot(
            conf_thresholds,
            plot_arr[i, 1, :],
            color=color,
            lw=2,
            label=ct,
        )
    fig2, ax = mplp(
        xlabel="Confidence", ylabel="% classified", xlim=[0, 1], show_legend=True
    )

    if saveDir is not None:
        npyx_plot.save_mpl_fig(fig1, "survival_plot_Ncells", saveDir, "pdf")
        npyx_plot.save_mpl_fig(fig2, "survival_plot_%cells", saveDir, "pdf")

    return fig1, fig2
