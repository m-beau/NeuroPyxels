import contextlib
import os

if __name__ == "__main__":
    __package__ = "npyx.c4"

import argparse
import gc
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import numpy as np
import pandas as pd

with contextlib.suppress(ImportError):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data as data

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from torchvision import transforms

try:
    from laplace import BaseLaplace, Laplace
    from laplace.utils import KronDecomposed
except ImportError:
    KronDecomposed = None
    BaseLaplace = None
    print(
        (
            "\nlaplace could not be imported - "
            "some functions from the submodule npyx.c4 will not work.\n"
            "To install laplace, see https://pypi.org/project/laplace-torch/."
        )
    )

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    print(
        (
            "\nimblearn could not be imported - "
            "some functions from the submodule npyx.c4 will not work.\n"
            "To install imblearn, see https://pypi.org/project/imblearn/."
        )
    )


from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

import npyx.datasets as datasets
import npyx.plot as npyx_plot
from npyx.ml import set_seed

from . import acg_augmentations
from . import dl_transforms as custom_transforms
from . import plots_functions as pf
from . import waveform_augmentations
from .dataset_init import (
    BIN_SIZE,
    WAVEFORM_SAMPLES,
    WIN_SIZE,
    ArgsNamespace,
    download_file,
    extract_and_check,
    get_paths_from_dir,
    prepare_classification_dataset,
    save_results,
)
from .dl_utils import (
    ConvolutionalEncoder,
    Encoder,
    load_acg_vae,
    load_waveform_encoder,
    load_waveform_vae,
)

SEED = 42

VAES_URL = (
    "https://figshare.com/ndownloader/files/42144024?private_link=93152fd04f501c7760c5"
)

WVF_VAE_PATH_SINGLE = os.path.join(
    Path.home(),
    ".npyx_c4_resources",
    "vaes",
    "wvf_singlechannel_encoder.pt",
)
ACG_VAE_PATH = os.path.join(
    Path.home(),
    ".npyx_c4_resources",
    "vaes",
    "3DACG_logscale_encoder.pt",
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

WVF_VAE_PATH_MULTI = os.path.join(
    Path.home(),
    ".npyx_c4_resources",
    "vaes",
    "wvf_multichannel_encoder.pt",
)

N_CHANNELS = 10


def download_vaes():
    models_folder = os.path.join(Path.home(), ".npyx_c4_resources", "vaes")
    if not os.path.exists(models_folder) or len(os.listdir(models_folder)) < 5:
        os.makedirs(models_folder, exist_ok=True)
        print("VAE checkpoints were not found, downloading...")

        vaes_archive = os.path.join(models_folder, "vaes.tar")
        download_file(
            VAES_URL, vaes_archive, description="Downloading VAEs checkpoints"
        )

        with tarfile.open(vaes_archive, "r:") as tar:
            for file in tar:
                tar.extract(file, models_folder)
        os.remove(vaes_archive)


class CustomCompose:
    def __init__(self, spike_transforms, sample_transforms):
        self.spike_transforms = spike_transforms
        self.sample_transforms = sample_transforms

    def __call__(self, spikes, sample):
        for t in self.spike_transforms:
            spikes, sample = t(spikes, sample)

        for t in self.sample_transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        for t in self.spike_transforms + self.sample_transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class CustomDataset(data.Dataset):
    """Dataset of waveforms and 3D acgs. Every batch will have shape:
    (batch_size, WAVEFORM_SAMPLES * ACG_3D_BINS * ACG_3D_LEN))"""

    def __init__(
        self,
        data,
        targets,
        spikes_list=None,
        spikes_transform=None,
        wave_transform=None,
        layer=None,
        multi_chan_wave=False,
    ):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            targets (string): Array of labels for the provided data
            raw_spikes (ndarray): Array of raw spikes for the provided data, used in acg augmentations.
        """
        self.data = data
        self.targets = targets
        if spikes_list is not None:
            self.spikes_list = np.array(spikes_list, dtype=object)
        else:
            self.spikes_list = None
        self.spikes_transform = spikes_transform
        self.wave_transform = wave_transform
        self.layer = layer
        if self.layer is not None:
            assert len(layer) == len(
                data
            ), f"Layer and data must have same length, got {len(layer)} and {len(data)}"
        self.multi_chan_wave = multi_chan_wave

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_point = self.data[idx, :].astype("float32")
        target = self.targets[idx].astype("int")
        if self.layer is not None:
            layer = self.layer[idx, :].astype("float32")
        if self.spikes_list is not None:
            spikes = self.spikes_list[idx].astype("int")

        if self.spikes_transform is not None:
            acg = data_point[:2010]
            acg = self.spikes_transform(spikes, acg)
        else:
            acg = data_point[:2010].reshape(10, 201)[:, 100:].ravel()

        waveform = data_point[2010:]
        if self.wave_transform is not None:
            waveform = self.wave_transform(waveform).squeeze()
        if self.layer is not None:
            data_point = np.concatenate((acg.ravel(), waveform, layer)).astype(
                "float32"
            )
        else:
            data_point = np.concatenate((acg.ravel(), waveform)).astype("float32")
        return data_point, target


def plot_training_curves(train_losses, f1_train, epochs, save_folder=None):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    axes[0].plot(train_losses.mean(0), label="Mean training loss")
    axes[0].fill_between(
        range(epochs),
        train_losses.mean(0) + train_losses.std(0),
        train_losses.mean(0) - train_losses.std(0),
        facecolor="blue",
        alpha=0.2,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper left")

    axes[1].plot(f1_train.mean(0), label="Mean training F1")
    axes[1].fill_between(
        range(epochs),
        f1_train.mean(0) + f1_train.std(0),
        f1_train.mean(0) - f1_train.std(0),
        facecolor="blue",
        alpha=0.2,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].legend(loc="upper left")

    fig.savefig(os.path.join(save_folder, "training_curves.png"))
    plt.close('all')


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0]


def train(
    model,
    iterator,
    optimizer,
    criterion,
    device,
):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()

    for batch in iterator:
        x = batch[0].to(device)
        y = batch[1].to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        # measure the loss
        loss = criterion(y_pred, y)

        # calculate the backward pass
        loss.backward()

        # updates the weights based on the backward pass
        optimizer.step()

        # compute performance
        with torch.no_grad():
            f1 = f1_score(
                y.cpu(),
                y_pred.cpu().argmax(1),
                labels=np.unique(y.cpu().numpy()),
                average="macro",
                zero_division=1,
            )

            acc = calculate_accuracy(y_pred, y)

        # store performance for this minibatch
        epoch_loss += loss.item()
        epoch_f1 += f1.item()
        epoch_acc += acc.item()

    return (
        epoch_loss / len(iterator),
        epoch_f1 / len(iterator),
        epoch_acc / len(iterator),
    )


def layer_correction(
    probabilities: Union[np.ndarray, torch.Tensor],
    layer_info: Union[np.ndarray, torch.Tensor],
    labelling: dict,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies corrections to the predicted probabilities based on the layer information.
    In essence, enforces that anatomically impossible predictions are not made given the layer information.
    Can work with both numpy arrays and PyTorch tensors.

    Args:
    - probabilities (Union[np.ndarray, torch.Tensor]): The predicted probabilities.
    - layer_info (Union[np.ndarray, torch.Tensor]): The layer information.
    - labelling (dict): The labelling dictionary.

    Returns:
    - new_probs (Union[np.ndarray, torch.Tensor]): Corrected probabilities with layer information enforced.
    """
    if isinstance(probabilities, np.ndarray):
        probabilities = torch.from_numpy(probabilities)
        return_numpy = True
    else:
        return_numpy = False

    if isinstance(layer_info, np.ndarray):
        layer_info = torch.from_numpy(layer_info)

    layer_argmax = torch.argmax(layer_info, dim=1)
    mask = torch.ones_like(probabilities)

    # Apply the proper corrections
    if "MLI_A" in labelling:
        mask[layer_argmax == 1, labelling["MLI_A"]] = 0
        mask[layer_argmax == 1, labelling["MLI_B"]] = 0
    else:
        mask[layer_argmax == 1, labelling["MLI"]] = 0
    mask[layer_argmax == 3, labelling["GoC"]] = 0
    mask[layer_argmax == 3, labelling["MFB"]] = 0

    new_probs = probabilities * mask
    new_probs = new_probs / new_probs.sum(dim=1, keepdims=True)

    if return_numpy:
        new_probs = new_probs.numpy()

    return new_probs


def get_kronecker_hessian_attributes(*kronecker_hessians: KronDecomposed):
    hessians = []
    for h in kronecker_hessians:
        hess_dict = {
            "eigenvalues": h.eigenvalues,
            "eigenvectors": h.eigenvectors,
            "deltas": h.deltas,
            "damping": h.damping,
        }
        hessians.append(hess_dict)
    return hessians


def predict_unlabelled(
    model: Union[BaseLaplace, torch.nn.Module],
    test_loader: data.DataLoader,
    device: torch.device = torch.device("cpu"),
    enforce_layer: bool = False,
    labelling: Optional[dict] = None,
) -> torch.Tensor:
    """
    Predicts the probabilities of the test set using a Laplace model.

    Args:
        model (Union[Laplace, torch.nn.Module]): The model to use for prediction.
        test_loader (data.DataLoader): The data loader for the test set.
        device (torch.device, optional): The device to use for prediction. Defaults to torch.device("cpu").
        enforce_layer (bool, optional): Whether to enforce layer correction. Defaults to False.
        labelling (dict, optional): The labelling dictionary to use for layer correction. Defaults to None.

    Returns:
        torch.Tensor: The predicted probabilities of the test set.
    """

    if enforce_layer:
        assert labelling is not None, "Labelling must be provided if enforcing layer"

    # Finally get adjusted probabilities
    probabilities = []
    with torch.no_grad():
        for x, _ in test_loader:
            model_probabilities = model(x.float().to(device))
            if not isinstance(model, BaseLaplace):
                model_probabilities = torch.softmax(model_probabilities, dim=-1)
            if enforce_layer:
                model_probabilities = layer_correction(
                    model_probabilities, x[:, -4:], labelling
                )
            probabilities.append(model_probabilities)

    return torch.cat(probabilities).cpu()


def get_model_probabilities(
    model: torch.nn.Module,
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    device: torch.device = torch.device("cpu"),
    laplace: bool = True,
    enforce_layer: bool = False,
    labelling: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[BaseLaplace]]:
    """
    Computes the probabilities of a given model for a test dataset, with or without Laplace approximation calibration.

    Args:
    - model: a PyTorch model.
    - train_loader: a PyTorch DataLoader for the training dataset.
    - test_loader: a PyTorch DataLoader for the test dataset.
    - device: a PyTorch device to run the computations on.
    - laplace: a boolean indicating whether to use Laplace approximation calibration or not. Default is True.
    - enforce_layer: a boolean indicating whether to enforce layer correction or not. Default is False.
    - labelling: a dictionary containing the labels for each cell type. Required if enforce_layer is True. Default is None.

    Returns:
    - probs_normal: a PyTorch tensor containing the uncalibrated probabilities for the test dataset.
    - probs_laplace: a PyTorch tensor containing the calibrated probabilities for the test dataset, if laplace is True. Otherwise, it is the same as probs_normal.
    - la: a Laplace object containing the fitted Laplace approximation, if laplace is True. Otherwise, it is None.
    """

    if enforce_layer:
        assert labelling is not None, "Labelling must be provided if enforcing layer"

    model.eval().to(device)
    # First get uncalibrated probabilities
    probs_normal = []
    with torch.no_grad():
        for x, _ in test_loader:
            model_uncalibrated_probabilities = torch.softmax(
                model(x.float().to(device)), dim=-1
            )
            if enforce_layer:
                model_uncalibrated_probabilities = layer_correction(
                    model_uncalibrated_probabilities, x[:, -4:], labelling
                )
            probs_normal.append(model_uncalibrated_probabilities)
    if not laplace:
        return torch.cat(probs_normal).cpu(), torch.cat(probs_normal).cpu(), None

    # Then fit Laplace approximation
    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")

    # Finally get adjusted probabilities
    probs_laplace = []
    with torch.no_grad():
        for x, _ in test_loader:
            model_calibrated_probabilities = la(x.float().to(device))
            if enforce_layer:
                model_calibrated_probabilities = layer_correction(
                    model_calibrated_probabilities, x[:, -4:], labelling
                )
            probs_laplace.append(model_calibrated_probabilities)

    return torch.cat(probs_normal).cpu(), torch.cat(probs_laplace).cpu(), la


class CNNCerebellum(nn.Module):
    """
    A convolutional neural network for classifying cerebellar data.

    Args:
        acg_head (ConvolutionalEncoder): The encoder for the ACG data.
        waveform_head (Encoder): The encoder for the waveform data.
        n_classes (int): The number of classes to classify.
        freeze_vae_weights (bool): Whether to freeze the weights of the VAE.
        use_layer (bool): Whether to use the layer data.
        multi_chan_wave (bool): Whether to use multiple channels for the waveform data.

    Attributes:
        acg_head (ConvolutionalEncoder): The encoder for the ACG data.
        wvf_head (Encoder): The encoder for the waveform data.
        use_layer (bool): Whether to use the layer data.
        multi_chan_wave (bool): Whether to use multiple channels for the waveform data.
        fc1 (nn.LazyLinear): The first fully connected layer.
        fc2 (nn.LazyLinear): The second fully connected layer.
        batch_norm (nn.BatchNorm1d): The batch normalization layer.

    Methods:
        forward(x: torch.Tensor, layer: Optional[torch.Tensor] = None) -> torch.Tensor:
            Performs a forward pass through the network.

    """

    def __init__(
        self,
        acg_head: ConvolutionalEncoder,
        waveform_head: Encoder,
        n_classes: int = 5,
        freeze_vae_weights: bool = False,
        use_layer: bool = False,
        multi_chan_wave: bool = False,
    ) -> None:
        super(CNNCerebellum, self).__init__()
        self.acg_head = acg_head
        self.wvf_head = waveform_head
        self.use_layer = use_layer
        self.multi_chan_wave = multi_chan_wave

        if freeze_vae_weights:
            for param in self.acg_head.parameters():
                param.requires_grad = False
            for param in self.wvf_head.parameters():
                param.requires_grad = False
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.LazyLinear(n_classes)
        self.batch_norm = nn.BatchNorm1d(24) if self.use_layer else nn.BatchNorm1d(20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acg = x[:, :1010]

        if self.use_layer:
            wvf = x[:, 1010:-4]
            layer = x[:, -4:]
        else:
            wvf = x[:, 1010:]

        if self.multi_chan_wave:
            wvf = wvf.reshape(-1, 1, 4, 90)

        acg = self.acg_head.forward(acg.reshape(-1, 1, 10, 101))
        wvf = self.wvf_head.forward(wvf)

        acg_tensor = acg.mean
        wvf_tensor = wvf.mean

        if self.use_layer:
            x = torch.cat((acg_tensor, wvf_tensor, layer), dim=1)
        else:
            x = torch.cat((acg_tensor, wvf_tensor), dim=1)

        x = self.batch_norm(x)

        x = F.relu(self.fc1(x))

        x = self.fc2(torch.cat((x, layer), dim=1)) if self.use_layer else self.fc2(x)

        return x


def define_transformations(
    norm_acg,
    log_acg=True,
    transform_acg=True,
    transform_wave=True,
    multi_chan=False,
):
    acg_transformations = (
        CustomCompose(
            [
                acg_augmentations.SubselectPeriod(p=0.3),
                acg_augmentations.DownsampleSpikes(n=20_000, p=0.5),
                acg_augmentations.DeleteSpikes(p=0.2, deletion_prob=0.2),
                acg_augmentations.RandomJitter(p=0.3, max_shift=50),
                acg_augmentations.AddSpikes(p=0.2, max_addition=0.1),
                acg_augmentations.Make3DACG(
                    bin_size=1,
                    window_size=2000,
                    cut=True,
                    normalise=norm_acg,
                    log_acg=log_acg,
                ),
            ],
            [],
        )
        if transform_acg
        else None
    )
    if transform_wave and not multi_chan:
        wave_transformations = transforms.Compose(
            [
                waveform_augmentations.SelectWave(),
                waveform_augmentations.GaussianNoise(p=0.3, std=0.1),
            ]
        )
    elif transform_wave:
        wave_transformations = transforms.Compose(
            [
                custom_transforms.SwapChannels(p=0.5),
                custom_transforms.VerticalReflection(p=0.5),
                custom_transforms.PermuteChannels(p=0.1, n_channels=2),
            ],
        )
    else:
        wave_transformations = None

    return acg_transformations, wave_transformations


def save_ensemble(models_states, file_path):
    # Create a temporary directory to store the models_states
    temp_dir = "models"
    os.makedirs(temp_dir, exist_ok=True)

    # Save each model in the temporary directory
    for i, state_dict in enumerate(models_states):
        torch.save(state_dict, os.path.join(temp_dir, f"model_{i}.pt"))

    # Create a tar archive containing the models_states
    with tarfile.open(file_path, "w:gz") as tar:
        tar.add(temp_dir, arcname=os.path.basename(temp_dir))

    # Remove the temporary directory
    shutil.rmtree(temp_dir)


def load_ensemble(
    file_path,
    device=None,
    pool_type="avg",
    n_classes=5,
    use_layer=False,
    fast=False,
    laplace=False,
    multi_chan_wave=False,
    **model_kwargs,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a temporary directory to extract the models
    temp_dir = "models"
    os.makedirs(temp_dir, exist_ok=True)

    # Extract the tar archive containing the models
    with tarfile.open(file_path, "r:gz") as tar:
        file_count = len(tar.getmembers())
        progress_bar = tqdm(
            total=file_count, desc="Extracting models files", unit="file"
        )

        for file in tar:
            tar.extract(file, temp_dir)
            progress_bar.update(1)

        progress_bar.close()

    try:
        # Find the nested folder containing the models
        nested_folder = os.path.join(temp_dir, os.listdir(temp_dir)[0])

        ensemble_paths = sorted(
            os.listdir(nested_folder),
            key=lambda x: int(x.split("model_")[1].split(".")[0]),
        )

        if fast and len(ensemble_paths) > 100:
            models_mask = np.random.choice(np.arange(len(ensemble_paths)), 100)
            ensemble_paths = np.array(ensemble_paths)[models_mask].tolist()

        # Load each model from the nested folder
        models = []
        for model_file in tqdm(ensemble_paths, desc="Loading models"):
            model_path = os.path.join(nested_folder, model_file)
            if os.path.isfile(model_path):
                acg_vae = load_acg_vae(
                    None,
                    WIN_SIZE // 2,
                    BIN_SIZE,
                    initialise=False,
                    pool=pool_type,
                )
                acg_head = acg_vae.encoder.to(device)

                if multi_chan_wave:
                    wvf_vae = load_waveform_vae(
                        WVF_ENCODER_ARGS_MULTI,
                        WVF_VAE_PATH_MULTI,
                    )
                    wvf_head = wvf_vae.encoder
                else:
                    wvf_vae = load_waveform_encoder(
                        WVF_ENCODER_ARGS_SINGLE,
                        None,
                        in_features=90,
                        initialise=False,
                    )
                    wvf_head = Encoder(wvf_vae.encoder, 10).to(device)

                model = CNNCerebellum(
                    acg_head,
                    wvf_head,
                    n_classes,
                    use_layer=use_layer,
                    multi_chan_wave=multi_chan_wave,
                    **model_kwargs,
                ).to(device)

                model.load_state_dict(
                    torch.load(model_path, map_location=device), strict=True
                )
                model.eval()
                models.append(model)
            else:
                print(f"Skipping: {model_path}")
    finally:
        # Remove the temporary directory
        shutil.rmtree(temp_dir)

    if laplace:
        hessians = torch.load(
            file_path.replace("trained_models.tar.gz", "hessians.pt"),
            map_location=device,
        )

        if fast and len(ensemble_paths) > 100:
            hessians = (
                hessians[:, :, models_mask]
                if type(hessians) == torch.Tensor
                else np.array(hessians, dtype=object)[models_mask].tolist()
            )
        models = load_calibrated_ensemble(models, hessians)

    return models


def load_calibrated_ensemble(models, hessians):
    calibrated_models = []

    for model, hessian in tqdm(
        zip(models, hessians),
        total=len(models),
        desc="Applying uncertainty calibration",
    ):
        # Create new laplace instance and then load the pre-fitted Hessian
        calibrated_model = Laplace(
            model,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure="kron",
            last_layer_name="fc2",
        )
        hessian = (
            hessian if isinstance(hessian, torch.Tensor) else KronDecomposed(**hessian)
        )
        setattr(
            calibrated_model,
            "mean",
            torch.nn.utils.parameters_to_vector(
                calibrated_model.model.last_layer.parameters()
            ).detach(),
        )
        setattr(calibrated_model, "H", hessian)
        calibrated_model.optimize_prior_precision(method="marglik")
        calibrated_models.append(calibrated_model)

    return calibrated_models


def ensemble_predict(
    ensemble,
    test_iterator,
    device=torch.device("cpu"),
    method="average",
    enforce_layer=False,
    labelling=None,
):
    predicted_probabilities = []

    for model in tqdm(
        ensemble, leave=True, position=0, desc="Predicting with ensemble"
    ):
        if not isinstance(model, BaseLaplace):
            model.eval()
        probabilities = predict_unlabelled(
            model,
            test_iterator,
            device,
            enforce_layer,
            labelling,
        )
        torch.cuda.empty_cache()
        predicted_probabilities.append(probabilities.detach().numpy())

    predicted_probabilities = np.stack(predicted_probabilities, axis=2)

    if method == "average":
        return predicted_probabilities.mean(axis=2)
    elif method == "majority_voting":
        raise NotImplementedError

    elif method == "raw":
        return predicted_probabilities
    else:
        raise ValueError(
            "Invalid method. Choose either 'average', 'majority_voting' or 'raw'."
        )


def cross_validate(
    dataset,
    targets,
    spikes,
    acg_vae_path,
    args,
    layer_info=None,
    epochs=30,
    batch_size=64,
    loo=False,
    n_runs=10,
    VAE_random_init=False,
    freeze_vae_weights=False,
    save_folder=None,
    save_models=False,
    enforce_layer=False,
    labelling=None,
):
    n_splits = len(dataset) if loo else 5
    n_classes = len(np.unique(targets))

    acg_transformations, wave_transformations = define_transformations(
        norm_acg=False,
        transform_acg=args.augment_acg,
        transform_wave=args.augment_wvf,
        multi_chan=args.multi_chan_wave,
    )

    train_losses = np.zeros((n_splits * n_runs, epochs))
    f1_train = np.zeros((n_splits * n_runs, epochs))
    acc_train = np.zeros((n_splits * n_runs, epochs))

    all_runs_f1_scores = []
    all_runs_targets = []
    all_runs_predictions = []
    all_runs_probabilities = []
    folds_stddev = []
    unit_idxes = []
    if save_models:
        models_states = []
        hessians = []
    total_runs = 0

    set_seed(SEED, seed_torch=True)

    for _ in tqdm(range(n_runs), desc="Cross-validation run", position=0, leave=True):
        run_train_accuracies = []
        run_true_targets = []
        run_model_pred = []
        run_probabilities = []
        run_unit_idxes = []
        folds_f1 = []

        cross_seed = SEED + np.random.randint(0, 100)
        kfold = (
            LeaveOneOut()
            if loo
            else StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=cross_seed
            )
        )

        for fold, (train_idx, val_idx) in tqdm(
            enumerate(kfold.split(dataset, targets)),
            leave=False,
            position=1,
            desc="Cross-validating",
            total=kfold.get_n_splits(dataset),
        ):
            dataset_train = dataset[train_idx]
            y_train = targets[train_idx]
            spikes_train = np.array(spikes, dtype="object")[train_idx]
            if layer_info is not None:
                layer_train = layer_info[train_idx]
                layer_val = layer_info[val_idx]
            else:
                layer_train = None
                layer_val = None

            dataset_val = dataset[val_idx]
            y_val = targets[val_idx]
            spikes_val = np.array(spikes, dtype="object")[val_idx].tolist()

            oversample = RandomOverSampler(random_state=cross_seed)
            resample_idx, _ = oversample.fit_resample(
                np.arange(len(dataset_train)).reshape(-1, 1), y_train
            )
            resample_idx = resample_idx.ravel()
            dataset_train = dataset_train[resample_idx]
            y_train = y_train[resample_idx]
            spikes_train = spikes_train[resample_idx].tolist()
            if layer_info is not None:
                layer_train = layer_train[resample_idx]

            train_iterator = data.DataLoader(
                CustomDataset(
                    dataset_train,
                    y_train,
                    spikes_list=spikes_train,
                    spikes_transform=acg_transformations,
                    wave_transform=wave_transformations,
                    layer=layer_train,
                    multi_chan_wave=args.multi_chan_wave,
                ),
                shuffle=True,
                batch_size=batch_size,
                num_workers=4,
            )

            val_iterator = data.DataLoader(
                CustomDataset(
                    dataset_val,
                    y_val,
                    spikes_val,
                    layer=layer_val,
                    multi_chan_wave=args.multi_chan_wave,
                ),
                batch_size=len(dataset_val),
            )

            # Define model
            acg_vae = load_acg_vae(
                acg_vae_path,
                WIN_SIZE // 2,
                BIN_SIZE,
                initialise=not VAE_random_init,
                pool=args.pool_type,
            )
            acg_head = acg_vae.encoder

            if args.multi_chan_wave:
                wvf_vae = load_waveform_vae(
                    WVF_ENCODER_ARGS_MULTI,
                    WVF_VAE_PATH_MULTI,
                )
                wvf_head = wvf_vae.encoder
            else:
                wvf_vae = load_waveform_encoder(
                    WVF_ENCODER_ARGS_SINGLE,
                    WVF_VAE_PATH_SINGLE,
                    in_features=90,
                    initialise=not VAE_random_init,
                )
                wvf_head = Encoder(wvf_vae.encoder, 10)

            model = CNNCerebellum(
                acg_head,
                wvf_head,
                n_classes,
                freeze_vae_weights=freeze_vae_weights,
                use_layer=args.use_layer,
                multi_chan_wave=args.multi_chan_wave,
            ).to(DEVICE)

            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, epochs, 1, last_epoch=-1
            )

            criterion = nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = model.to(device)
            criterion = criterion.to(device)

            for epoch in tqdm(range(epochs), position=2, leave=False, desc="Epochs"):
                train_loss, train_f1, train_acc = train(
                    model,
                    train_iterator,
                    optimizer,
                    criterion,
                    device,
                )

                train_losses[total_runs, epoch] = train_loss
                acc_train[total_runs, epoch] = train_acc
                f1_train[total_runs, epoch] = train_f1
                scheduler.step()

            total_runs += 1

            # Append results
            _, prob_calibrated, model_calibrated = get_model_probabilities(
                model,
                train_iterator,
                val_iterator,
                torch.device("cpu"),
                laplace=True,
                enforce_layer=enforce_layer,
                labelling=labelling,
            )
            run_true_targets.append(y_val)
            run_model_pred.append(prob_calibrated.argmax(1))
            run_probabilities.append(prob_calibrated)

            fold_f1 = f1_score(y_val, prob_calibrated.argmax(1), average="macro")
            folds_f1.append(fold_f1)
            unit_idxes.append(val_idx)
            run_unit_idxes.append(val_idx)

            if save_models:
                models_states.append(model.cpu().eval().state_dict())
                hessians.append(model_calibrated.H)

            del model
            del train_iterator
            del val_iterator
            del model_calibrated
            torch.cuda.empty_cache()
            gc.collect()

        run_unit_idxes = np.concatenate(run_unit_idxes).squeeze()

        # sort arrays using run_unit_idxes to restore original order
        sort_idx = np.argsort(run_unit_idxes)

        run_model_pred = np.concatenate(run_model_pred).squeeze()[sort_idx]
        run_true_targets = np.concatenate(run_true_targets).squeeze()[sort_idx]

        run_f1 = f1_score(run_true_targets, run_model_pred, average="macro")
        all_runs_f1_scores.append(run_f1)
        all_runs_targets.append(np.array(run_true_targets))
        all_runs_predictions.append(np.array(run_model_pred))
        all_runs_probabilities.append(
            np.concatenate(run_probabilities, axis=0)[sort_idx]
        )
        folds_stddev.append(np.array(folds_f1).std())

    plot_training_curves(train_losses, f1_train, epochs, save_folder=save_folder)

    if save_models:
        save_ensemble(models_states, os.path.join(save_folder, "trained_models.tar.gz"))
        if type(hessians[0]) == torch.Tensor:
            hessians = torch.stack(hessians, dim=2)
        elif type(hessians[0]) == KronDecomposed:
            hessians = get_kronecker_hessian_attributes(*hessians)
        torch.save(hessians, os.path.join(save_folder, "hessians.pt"))

    all_targets = np.concatenate(all_runs_targets).squeeze()
    raw_probabilities = np.stack(all_runs_probabilities, axis=2)

    if save_folder is not None:
        np.save(
            os.path.join(
                save_folder, "ensemble_predictions_ncells_nclasses_nmodels.npy"
            ),
            raw_probabilities,
        )

    all_probabilities = np.concatenate(all_runs_probabilities).squeeze()

    return {
        "f1_scores": all_runs_f1_scores,
        "train_accuracies": run_train_accuracies,
        "true_targets": all_targets,
        "predicted_probability": all_probabilities,
        "folds_stddev": np.array(folds_stddev),
        "indexes": np.concatenate(unit_idxes).squeeze(),
    }


def plot_confusion_matrices(
    results_dict,
    save_folder,
    model_name,
    labelling,
    correspondence,
    plots_prefix="",
    loo=False,
):
    if -1 in correspondence.keys():
        del correspondence[-1]
    features_name = "3D ACGs and waveforms"
    prefix = ""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_results(results_dict, save_folder)

    # if loo:
    n_models = len(results_dict["f1_scores"])
    n_classes = results_dict["predicted_probability"].shape[1]
    n_observations = results_dict["predicted_probability"].shape[0] // n_models
    predictions_matrix = results_dict["predicted_probability"].reshape(
        (n_models, n_observations, n_classes)
    )

    predictions_matrix = predictions_matrix.transpose(1, 2, 0)
    predicted_probabilities = predictions_matrix.mean(axis=2)
    true_labels = results_dict["true_targets"][:n_observations]
    # else:
    #     true_labels = results_dict["true_targets"]
    #     predicted_probabilities = results_dict["predicted_probability"]

    for threshold in tqdm(
        list(np.arange(0.4, 1, 0.1)) + [0.0], desc="Saving results figures"
    ):
        threshold = round(threshold, 2)
        fig = pf.plot_results_from_threshold(
            true_labels,
            predicted_probabilities,
            correspondence,
            threshold,
            f"{' '.join(model_name.split('_')).title()} {plots_prefix}({features_name})",
            collapse_classes=False,
            _shuffle_matrix=[4, 5, 3, 1, 2, 0]
            if "MLI_A" in labelling.keys()
            else [3, 4, 1, 0, 2],
            f1_scores=results_dict["f1_scores"]
            if "f1_scores" in results_dict
            else None,
            _folds_stddev=results_dict["folds_stddev"]
            if "folds_stddev" in results_dict
            else None,
        )
        npyx_plot.save_mpl_fig(
            fig, f"{prefix}{model_name}_at_threshold_{threshold}", save_folder, "pdf"
        )
        plt.close()


def ensemble_inference(
    dataset_test: np.ndarray,
    targets_test: np.ndarray,
    args: argparse.Namespace,
    ensemble_path: str,
    n_classes: int,
    layer_test: Optional[np.ndarray] = None,
    save_folder: Optional[str] = None,
    enforce_layer: bool = False,
    labelling: Optional[Dict[int, str]] = None,
    laplace: bool = True,
    fast: bool = False,
) -> Dict[str, Union[np.ndarray, None]]:
    """
    Performs inference on a test dataset using an ensemble of models.

    Args:
        dataset_test (np.ndarray): The test dataset.
        targets_test (np.ndarray): The test targets.
        args (argparse.Namespace): The command line arguments.
        ensemble_path (str): The path to the ensemble model.
        n_classes (int): The number of classes in the dataset.
        layer_test (Optional[np.ndarray], optional): The layer of test neurons. Defaults to None.
        save_folder (Optional[str], optional): The folder to save results to. Defaults to None.
        enforce_layer (bool, optional): Whether to enforce the use of the specified layer. Defaults to False.
        labelling (Optional[Dict[int, str]], optional): The class labels needed when enforcing layer. Defaults to None.
        laplace (bool, optional): Whether to use Laplace calibration. Defaults to True.
        fast (bool, optional): Whether to use fast inference using a subset of 100 random models. Defaults to False.

    Returns:
        Dict[str, Union[np.ndarray, None]]: A dictionary containing the results of the inference.
    """
    set_seed(SEED, seed_torch=True)

    test_iterator = data.DataLoader(
        CustomDataset(
            dataset_test,
            targets_test,
            spikes_list=None,
            layer=layer_test,
            multi_chan_wave=args.multi_chan_wave,
        ),
        batch_size=len(dataset_test),
    )

    ensemble = load_ensemble(
        ensemble_path,
        device=torch.device("cpu"),
        pool_type=args.pool_type,
        n_classes=n_classes,
        use_layer=args.use_layer,
        laplace=laplace,
        fast=fast,
        multi_chan_wave=args.multi_chan_wave,
    )

    # Calculate predictions and append results
    raw_probabilities = ensemble_predict(
        ensemble,
        test_iterator,
        device=torch.device("cpu"),
        method="raw",
        enforce_layer=enforce_layer,
        labelling=labelling,
    )
    mean_probabilities = raw_probabilities.mean(axis=2)

    f1_tests = [
        f1_score(
            targets_test,
            mean_probabilities.argmax(1),
            average="macro",
        )
    ]

    if save_folder is not None:
        np.save(
            os.path.join(
                save_folder, "ensemble_predictions_ncells_nclasses_nmodels.npy"
            ),
            raw_probabilities,
        )

    return {
        "f1_scores": np.array(f1_tests),
        "train_accuracies": np.array([]),
        "true_targets": targets_test,
        "predicted_probability": mean_probabilities,
        "folds_stddev": None,
    }


def post_hoc_layer_correction(
    results_dict,
    one_hot_layer,
    labelling,
    repeats=1,
):
    probabilities = results_dict["predicted_probability"]
    one_hot_layer = np.tile(one_hot_layer, reps=(repeats, 1))
    corrected_probabilities = layer_correction(probabilities, one_hot_layer, labelling)
    corrected_f1_scores = [
        f1_score(true, probas.argmax(1), average="macro")
        for true, probas in zip(
            np.split(results_dict["true_targets"], repeats),
            np.split(corrected_probabilities, repeats),
        )
    ]
    new_results_dict = results_dict.copy()

    new_results_dict["predicted_probability"] = corrected_probabilities
    new_results_dict["f1_scores"] = np.array(corrected_f1_scores)

    return new_results_dict


def encode_layer_info_original(layer_information):
    layer_info = pd.Series(layer_information).replace(
        to_replace=datasets.LAYERS_CORRESPONDENCE
    )

    preprocessor = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), [-1])]
    )

    return preprocessor.fit_transform(layer_info.to_frame()).toarray()


def encode_layer_info(layer_information):
    N_values = len(datasets.LAYERS_CORRESPONDENCE.keys())
    for value in datasets.LAYERS_CORRESPONDENCE.keys():
        layer_information = np.append(layer_information, value)

    layer_info = pd.Series(layer_information).replace(
        to_replace=datasets.LAYERS_CORRESPONDENCE
    )

    preprocessor = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), [-1])]
    )

    result = preprocessor.fit_transform(layer_info.to_frame()).toarray()
    return result[0:-N_values]


def main(
    data_folder: str,
    freeze_vae_weights: bool = False,
    VAE_random_init: bool = False,
    augment_acg: bool = False,
    augment_wvf: bool = False,
    mli_clustering: bool = False,
    use_layer: bool = False,
    loo: bool = False,
    multi_chan_wave: bool = False,
) -> None:
    """
    Runs a deep semi-supervised classifier on the C4 ground-truth datasets,
    training it on mouse opto-tagged data and testing it on expert-labelled monkey neurons.

    Args:
        data_folder: The path to the folder containing the datasets.
        freeze_vae_weights: Whether to freeze the pretrained weights of the VAE.
        VAE_random_init: Whether to randomly initialize the VAE weights that were pretrained.
        augment_acg: Whether to augment the ACGs.
        augment_wvf: Whether to augment the waveforms.
        mli_clustering: Whether to cluster the MLI cells.
        use_layer: Whether to use layer information.
        loo: Whether to use leave-one-out cross-validation.
        multi_chan_wave: Whether to use multi-channel waveforms.

    Returns:
        None
    """
    args = ArgsNamespace(
        data_folder=data_folder,
        freeze_vae_weights=freeze_vae_weights,
        VAE_random_init=VAE_random_init,
        augment_acg=augment_acg,
        augment_wvf=augment_wvf,
        mli_clustering=mli_clustering,
        use_layer=use_layer,
        loo=loo,
        multi_chan_wave=multi_chan_wave,
        pool_type="avg",
    )

    global N_CHANNELS
    N_CHANNELS = 4 if args.multi_chan_wave else 10
    assert (
        np.array([args.freeze_vae_weights, args.VAE_random_init]).sum() <= 1
    ), "Only one of the two can be True"

    datasets_abs = get_paths_from_dir(args.data_folder)

    # Extract and check the datasets, saving a dataframe with the results
    _, dataset_class = extract_and_check(
        *datasets_abs,
        save=False,
        _labels_only=True,
        normalise_wvf=False,
        n_channels=N_CHANNELS,
        _extract_mli_clusters=args.mli_clustering,
        _extract_layer=args.use_layer,
    )

    # Apply quality checks and filter out granule cells
    checked_dataset = dataset_class.apply_quality_checks()
    LABELLING, CORRESPONDENCE, granule_mask = checked_dataset.filter_out_granule_cells(
        return_mask=True
    )

    dataset, _ = prepare_classification_dataset(
        checked_dataset,
        normalise_acgs=False,
        multi_chan_wave=args.multi_chan_wave or args.augment_wvf,
        process_multi_channel=args.multi_chan_wave,
        _acgs_path=os.path.join(
            args.data_folder, "acgs_vs_firing_rate", "acgs_3d_logscale.npy"
        ),
        _acg_mask=(~granule_mask),
        _acg_multi_factor=10,
        _n_channels=N_CHANNELS,
    )

    targets = checked_dataset.targets
    spikes = checked_dataset.spikes_list

    if args.use_layer:
        one_hot_layer = encode_layer_info(checked_dataset.layer_list)

    # Prepare model name to save results
    suffix = ""
    features_suffix = ""
    cv_string = "_loo_cv" if args.loo else "_5fold_cv"
    if args.freeze_vae_weights:
        suffix = "_frozen_heads"
    if args.multi_chan_wave:
        features_suffix += "_multi_channel"
    if args.VAE_random_init:
        suffix = "_VAE_random_init"
    if args.mli_clustering:
        features_suffix += "_mli_clustering"
    features_suffix += "_layer" if args.use_layer else ""
    augmented = ("_augmented_acgs" if args.augment_acg else "") + (
        "_augmented_waveforms" if args.augment_wvf else ""
    )
    model_name = f"deep_semisup{suffix}{augmented}"

    save_folder = os.path.join(
        args.data_folder,
        "feature_spaces",
        f"raw_log_3d_acg_peak_wvf{features_suffix}",
        model_name,
        f"mouse_results{cv_string}",
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    download_vaes()

    results_dict = cross_validate(
        dataset,
        targets,
        spikes,
        ACG_VAE_PATH,
        args,
        layer_info=one_hot_layer if args.use_layer else None,
        epochs=20,
        batch_size=64,
        loo=args.loo,
        n_runs=10,
        VAE_random_init=args.VAE_random_init,
        freeze_vae_weights=args.freeze_vae_weights,
        save_folder=save_folder,
        save_models=True,
        enforce_layer=False,
        labelling=LABELLING,
    )

    plot_confusion_matrices(
        results_dict,
        save_folder,
        model_name,
        labelling=LABELLING,
        correspondence=CORRESPONDENCE,
    )

    #
    # Predict monkey data
    #

    from .monkey_dataset_init import get_lisberger_dataset

    MONKEY_WAVEFORM_SAMPLES = int(WAVEFORM_SAMPLES * 40_000 / 30_000)

    datasets_abs = get_lisberger_dataset(args.data_folder)

    # Extract and check the dataset
    _, monkey_dataset_class = extract_and_check(
        datasets_abs,
        save_folder="",
        save=False,
        normalise_wvf=False,
        lisberger=True,
        _label="expert_label",
        n_channels=N_CHANNELS,
        central_range=MONKEY_WAVEFORM_SAMPLES,
        _use_amplitudes=False,
        _lisberger=True,
        _id_type="neuron_id",
        _labels_only=True,
        _extract_mli_clusters=True,
        _extract_layer=args.use_layer,
    )

    # Remove MLI_Bs from the dataset
    mli_b = np.array(monkey_dataset_class.labels_list) == "MLI_B"
    monkey_dataset_class._apply_mask(~mli_b)
    monkey_dataset_class = datasets.resample_waveforms(monkey_dataset_class)

    monkey_dataset, _ = prepare_classification_dataset(
        monkey_dataset_class,
        normalise_acgs=False,
        multi_chan_wave=args.multi_chan_wave,
        process_multi_channel=args.multi_chan_wave,
        _acgs_path=os.path.join(
            args.data_folder, "acgs_vs_firing_rate", "monkey_acgs_3d_logscale.npy"
        ),
        _acg_mask=~mli_b,
        _acg_multi_factor=10,
        _n_channels=N_CHANNELS,
    )

    if args.mli_clustering:
        monkey_targets = (
            pd.Series(monkey_dataset_class.labels_list)
            .replace(to_replace=LABELLING)
            .squeeze()
            .copy()
            .to_numpy()
        )
    else:
        monkey_targets = (
            pd.Series(monkey_dataset_class.labels_list)
            .replace({"MLI_A": "MLI"})
            .replace(to_replace=LABELLING)
            .squeeze()
            .copy()
            .to_numpy()
        )

    if args.use_layer:
        monkey_one_hot_layer = encode_layer_info(monkey_dataset_class.layer_list)

    # Defining training and test sets
    y_training = targets
    dataset_test = monkey_dataset
    y_test = monkey_targets

    save_folder_monkey = os.path.join(
        args.data_folder,
        "feature_spaces",
        f"raw_log_3d_acg_peak_wvf{features_suffix}",
        model_name,
        "monkey_results",
    )
    if not os.path.exists(save_folder_monkey):
        os.makedirs(save_folder_monkey)

    results_dict_monkey = ensemble_inference(
        dataset_test,
        y_test,
        args,
        ensemble_path=os.path.join(save_folder, "trained_models.tar.gz"),
        n_classes=len(np.unique(y_training)),
        layer_test=monkey_one_hot_layer if args.use_layer else None,
        save_folder=save_folder_monkey,
        enforce_layer=False,
        labelling=LABELLING,
        laplace=True,
        fast=False,
    )

    plot_confusion_matrices(
        results_dict_monkey,
        save_folder_monkey,
        model_name,
        labelling=LABELLING,
        correspondence=CORRESPONDENCE,
        plots_prefix="predicting monkey data",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run deep semi-superivsed classifier on neural data"
    )

    parser.add_argument(
        "-dp", "--data-folder", type=str, help="Path to the folder containing the data"
    )

    parser.add_argument(
        "--freeze_vae_weights",
        action="store_true",
        help="Freeze the weights of the VAE half of the model (the pretrained encoder) during training",
    )
    parser.set_defaults(freeze_vae_weights=False)

    parser.add_argument(
        "--VAE_random_init",
        action="store_true",
        help="Randomly initialize the weights of the VAE half of the model (overwrites the pretrained encoder)",
    )
    parser.set_defaults(VAE_random_init=False)

    parser.add_argument(
        "--augment_acg",
        action="store_true",
        help="Augment the ACG data during training",
    )
    parser.set_defaults(augment_acg=False)

    parser.add_argument(
        "--augment_wvf",
        action="store_true",
        help="Augment the waveform data during training",
    )
    parser.set_defaults(augment_wvf=False)

    parser.add_argument(
        "--mli_clustering",
        action="store_true",
        help="Use MLI clustering during training",
    )
    parser.set_defaults(mli_clustering=False)

    parser.add_argument(
        "--use_layer", action="store_true", help="Use layer information during training"
    )
    parser.set_defaults(use_layer=False)

    parser.add_argument(
        "--loo", action="store_true", help="Use leave-one-out cross-validation"
    )
    parser.set_defaults(loo=False)

    parser.add_argument(
        "--multi-chan-wave",
        action="store_true",
        help="Use multi-channel waveform data during training",
    )
    parser.set_defaults(multi_chan_wave=False)

    # Parse arguments and set global variables

    args = parser.parse_args()
    main(**vars(args))
