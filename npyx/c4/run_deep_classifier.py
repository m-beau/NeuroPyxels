import os

if __name__ == "__main__":
    __package__ = "npyx.c4"

import argparse
import gc
import re
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.utils.data as data
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from torchvision import transforms

except ImportError:
    pass

try:
    from laplace import BaseLaplace, Laplace
    from laplace.utils import FeatureExtractor, KronDecomposed
except ImportError:
    KronDecomposed = None
    BaseLaplace = None
    print(("\nlaplace could not be imported - "
    "some functions from the submodule npyx.c4 will not work.\n"
    "To install laplace, see https://pypi.org/project/laplace-torch/."))

try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    print(("\nimblearn could not be imported - "
    "some functions from the submodule npyx.c4 will not work.\n"
    "To install imblearn, see https://pypi.org/project/imblearn/."))


from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

import npyx.datasets as datasets
import npyx.plot as npyx_plot
from npyx.ml import set_seed

from . import acg_augmentations
from . import plots_functions as pf
from . import waveform_augmentations
from .dataset_init import (
    BIN_SIZE,
    N_CHANNELS,
    WAVEFORM_SAMPLES,
    WIN_SIZE,
    extract_and_check,
    get_paths_from_dir,
    prepare_classification_dataset,
    save_results,
)
from .dl_models import (
    ConvolutionalEncoder,
    Encoder,
    load_acg_vae,
    load_waveform_encoder,
)

SEED = 42

WVF_VAE_PATH_SINGLE = os.path.join(
    Path.home(),
    "Dropbox",
    "celltypes-classification",
    "data_format",
    "final_vaes",
    "VAE_encoder_singchan_Jun-01-2023_b5_GELU.pt",
)
ACG_VAE_PATH = os.path.join(
    Path.home(),
    "Dropbox",
    "celltypes-classification",
    "deep_models",
    "Jun-09-2023-raw_3DACG_encoder_b5_avgpool_logscale.pt",
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
    "n_channels": 10,
    "central_range": 60,
    "d_latent": 10,
    "device": "cpu",
}


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
    ):
        """
        Args:
            data (ndarray): Array of data points, with wvf and acg concatenated
            targets (string): Array of labels for the provided data
            raw_spikes (ndarray): Array of raw spikes for the provided data
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

        if self.wave_transform is not None:
            waveform = data_point[2010:]
            waveform = self.wave_transform(waveform).squeeze()
        elif len(data_point[2010:]) == 90:
            # leave it as is
            waveform = data_point[2010:]
        else:
            waveform = datasets.preprocess_template(
                data_point[2010:].reshape(N_CHANNELS, WAVEFORM_SAMPLES)[
                    N_CHANNELS // 2, :
                ]
            )

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

    plt.savefig(os.path.join(save_folder, "training_curves.png"))
    plt.close()


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

        loss = criterion(y_pred, y)

        with torch.no_grad():
            f1 = f1_score(
                y.cpu(),
                y_pred.cpu().argmax(1),
                labels=np.unique(y.cpu().numpy()),
                average="macro",
                zero_division=1,
            )

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

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


def save_laplace(la, filepath):
    with open(filepath, "wb") as outpt:
        dill.dump(la, outpt)


def load_laplace(filepath):
    with open(filepath, "rb") as inpt:
        la = dill.load(inpt)
    assert isinstance(
        la, BaseLaplace
    ), "Attempting to load a model that is not of class Laplace"
    return la





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
    def __init__(
        self,
        acg_head: ConvolutionalEncoder,
        waveform_head: Encoder,
        n_classes=5,
        freeze_heads=False,
        use_layer=False,
    ):
        super(CNNCerebellum, self).__init__()
        self.acg_head = acg_head
        self.wvf_head = waveform_head
        self.use_layer = use_layer

        if freeze_heads:
            for param in self.acg_head.parameters():
                param.requires_grad = False
            for param in self.wvf_head.parameters():
                param.requires_grad = False
        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.LazyLinear(n_classes)

    def forward(self, x):
        acg = x[:, :1010]

        if self.use_layer:
            wvf = x[:, 1010:-4]
            layer = x[:, -4:]
        else:
            wvf = x[:, 1010:]

        acg = self.acg_head.forward(acg.reshape(-1, 1, 10, 101))
        wvf = self.wvf_head.forward(wvf)

        acg_tensor = acg.mean
        wvf_tensor = wvf.mean

        if self.use_layer:
            x = torch.cat((acg_tensor, wvf_tensor, layer), dim=1)
        else:
            x = torch.cat((acg_tensor, wvf_tensor), dim=1)

        x = F.relu(self.fc1(x))

        x = self.fc2(torch.cat((x, layer), dim=1)) if self.use_layer else self.fc2(x)

        return x


def define_transformations(
    norm_acg,
    log_acg=True,
    transform_acg=True,
    transform_wave=True,
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
    if transform_wave:
        wave_transformations = transforms.Compose(
            [
                waveform_augmentations.SelectWave(),
                waveform_augmentations.GaussianNoise(p=0.3, std=0.1),
            ]
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


def save_calibrated_ensemble(calibrated_models, file_path):
    assert isinstance(
        calibrated_models[0], BaseLaplace
    ), "Calibrated models must be a Laplace object"
    # Create a temporary directory to store the models_states
    temp_dir = "calibrated_models"
    os.makedirs(temp_dir, exist_ok=True)

    # Save each model in the temporary directory
    for i, cal_model in enumerate(calibrated_models):
        save_laplace(cal_model, os.path.join(temp_dir, f"calibrated_model_{i}.pkl"))

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
    **model_kwargs,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create a temporary directory to extract the models
    temp_dir = "models"
    os.makedirs(temp_dir, exist_ok=True)

    # # Extract the tar archive containing the models
    # with tarfile.open(file_path, "r:gz") as tar:
    #     tar.extractall(temp_dir)

    # Extract the tar archive containing the models
    with tarfile.open(file_path, "r:gz") as tar:
        file_count = len(tar.getmembers())
        progress_bar = tqdm(total=file_count, desc="Extracting files", unit="file")

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


def join_calibrated_models(path):
    """
    Unpacks the tar archives of calibrated models in `path` and joins all of the models inside each of the archives
    into one archive only.
    """
    # Find all the calibrated models archives in the given path
    archives = [
        f
        for f in os.listdir(path)
        if f.startswith("calibrated_models_") and f.endswith(".tar.gz")
    ]
    archives.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

    # Create a temporary directory to extract the models
    tmp_dir = os.path.join(path, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Extract the models from each archive and save them in the temporary directory
    model_count = 0
    for archive in tqdm(archives, desc="Extracting archives", position=0):
        archive_path = os.path.join(path, archive)
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tqdm(tar.getmembers(), desc="Extracting models", position=1):
                if member.isfile() and "calibrated_model_" in member.name:
                    # Extract the model and rename it with an incremental index
                    model_name = f"calibrated_model_{model_count}.pkl"
                    member.name = model_name
                    tar.extract(member, tmp_dir)
                    model_count += 1

    # Join all the models in the temporary directory into a single archive
    joined_models_path = os.path.join(path, "calibrated_models.tar.gz")
    with tarfile.open(joined_models_path, "w:gz") as tar:
        for root, dirs, files in tqdm(
            os.walk(tmp_dir), desc="Joining models", position=2
        ):
            for file in tqdm(files, desc="Adding files to archive", position=3):
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=file)

    # Remove the temporary directory
    for archive in archives:
        archive_path = os.path.join(path, archive)
        os.remove(archive_path)
    shutil.rmtree(tmp_dir)


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
    random_init=False,
    freeze_heads=False,
    save_folder=None,
    save_models=False,
    enforce_layer=False,
    labelling=None,
):
    LOO = loo
    N_SPLITS = len(dataset) if loo else 5
    N_RUNS = n_runs
    EPOCHS = epochs
    BATCH_SIZE = batch_size
    RANDOM_INIT = random_init
    FREEZE_HEADS = freeze_heads
    N_CLASSES = len(np.unique(targets))

    acg_transformations, wave_transformations = define_transformations(
        norm_acg=False,
        transform_acg=args.augment_acg,
        transform_wave=args.augment_wvf,
    )

    train_losses = np.zeros((N_SPLITS * N_RUNS, EPOCHS))
    f1_train = np.zeros((N_SPLITS * N_RUNS, EPOCHS))
    acc_train = np.zeros((N_SPLITS * N_RUNS, EPOCHS))

    all_runs_f1_scores = []
    all_runs_targets = []
    all_runs_predictions = []
    all_runs_probabilities = []
    folds_variance = []
    unit_idxes = []
    if save_models:
        models_states = []
        hessians = []
    total_runs = 0

    set_seed(SEED, seed_torch=True)

    for run_i in tqdm(
        range(N_RUNS), desc="Cross-validation run", position=0, leave=True
    ):
        run_train_accuracies = []
        run_true_targets = []
        run_model_pred = []
        run_probabilites = []
        folds_f1 = []

        cross_seed = SEED + np.random.randint(0, 100)
        kfold = (
            LeaveOneOut()
            if LOO
            else StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=cross_seed
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
                ),
                shuffle=True,
                batch_size=BATCH_SIZE,
                num_workers=4,
            )

            val_iterator = data.DataLoader(
                CustomDataset(
                    dataset_val,
                    y_val,
                    spikes_val,
                    layer=layer_val,
                ),
                batch_size=len(dataset_val),
            )

            # Define model
            acg_vae = load_acg_vae(
                acg_vae_path,
                WIN_SIZE // 2,
                BIN_SIZE,
                initialise=not RANDOM_INIT,
                pool=args.pool_type,
            )
            acg_head = acg_vae.encoder

            wvf_vae = load_waveform_encoder(
                WVF_ENCODER_ARGS_SINGLE,
                WVF_VAE_PATH_SINGLE,
                in_features=90,
                initialise=not RANDOM_INIT,
            )
            wvf_head = Encoder(wvf_vae.encoder, 10)

            model = CNNCerebellum(
                acg_head,
                wvf_head,
                N_CLASSES,
                freeze_heads=FREEZE_HEADS,
                use_layer=args.use_layer,
            ).to(DEVICE)

            optimizer = optim.AdamW(model.parameters(), lr=1e-3)

            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, EPOCHS, 1, last_epoch=-1
            )

            criterion = nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = model.to(device)
            criterion = criterion.to(device)

            for epoch in tqdm(range(EPOCHS), position=2, leave=False, desc="Epochs"):
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
            run_probabilites.append(prob_calibrated)

            fold_f1 = f1_score(y_val, prob_calibrated.argmax(1), average="macro")
            folds_f1.append(fold_f1)
            unit_idxes.append(val_idx)

            if save_models:
                models_states.append(model.cpu().eval().state_dict())
                hessians.append(model_calibrated.H)

            del model
            del train_iterator
            del val_iterator
            del model_calibrated
            torch.cuda.empty_cache()
            gc.collect()

        run_model_pred = np.concatenate(run_model_pred).squeeze()
        run_true_targets = np.concatenate(run_true_targets).squeeze()

        run_f1 = f1_score(run_true_targets, run_model_pred, average="macro")
        all_runs_f1_scores.append(run_f1)
        all_runs_targets.append(np.array(run_true_targets))
        all_runs_predictions.append(np.array(run_model_pred))
        all_runs_probabilities.append(np.concatenate(run_probabilites, axis=0))
        folds_variance.append(np.array(folds_f1).std())

    plot_training_curves(train_losses, f1_train, EPOCHS, save_folder=save_folder)

    if save_models:
        save_ensemble(models_states, os.path.join(save_folder, "trained_models.tar.gz"))
        if type(hessians[0]) == torch.Tensor:
            hessians = torch.stack(hessians, dim=2)
        elif type(hessians[0]) == KronDecomposed:
            hessians = get_kronecker_hessian_attributes(*hessians)
        torch.save(hessians, os.path.join(save_folder, "hessians.pt"))

    all_targets = np.concatenate(all_runs_targets).squeeze()
    all_probabilities = np.concatenate(all_runs_probabilities).squeeze()
    return {
        "f1_scores": all_runs_f1_scores,
        "train_accuracies": run_train_accuracies,
        "true_targets": all_targets,
        "predicted_probability": all_probabilities,
        "folds_variance": np.array(folds_variance),
        "indexes": np.concatenate(unit_idxes).squeeze(),
    }


def plot_confusion_matrices(
    results_dict, save_folder, model_name, labelling, correspondence, plots_prefix=""
):
    if -1 in correspondence.keys():
        del correspondence[-1]
    features_name = "3D ACGs and waveforms"
    prefix = ""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_results(results_dict, save_folder)

    n_models = len(results_dict["f1_scores"])
    n_classes = results_dict["predicted_probability"].shape[1]
    n_observations = results_dict["predicted_probability"].shape[0] // n_models
    predictions_matrix = results_dict["predicted_probability"].reshape(
        (n_models, n_observations, n_classes)
    )

    predictions_matrix = predictions_matrix.transpose(1, 2, 0)
    predicted_probabilities = predictions_matrix.mean(axis=2)
    true_labels = results_dict["true_targets"][:n_observations]

    for threshold in tqdm(
        list(np.arange(0.5, 1, 0.1)) + [0.0], desc="Saving results figures"
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
            _folds_variance=results_dict["folds_variance"]
            if "folds_variance" in results_dict
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
            os.path.join(save_folder, "all_runs_raw_prob_calibrated.npy"),
            raw_probabilities,
        )

    return {
        "f1_scores": np.array(f1_tests),
        "train_accuracies": np.array([]),
        "true_targets": targets_test,
        "predicted_probability": mean_probabilities,
        "folds_variance": None,
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


def encode_layer_info(layer_information):
    layer_info = pd.Series(layer_information).replace(
        to_replace=datasets.LAYERS_CORRESPONDENCE
    )
    preprocessor = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), [-1])]
    )
    return preprocessor.fit_transform(layer_info.to_frame()).toarray()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dp", "--data_folder", type=str)

    parser.add_argument("--freeze", action="store_true")
    parser.set_defaults(freeze=False)

    parser.add_argument("--random_init", action="store_true")
    parser.set_defaults(random_init=False)

    parser.add_argument("--augment_acg", action="store_true")
    parser.set_defaults(augment_acg=False)

    parser.add_argument("--augment_wvf", action="store_true")
    parser.set_defaults(augment_wvf=False)

    parser.add_argument("--mli_clustering", action="store_true")
    parser.set_defaults(mli_clustering=False)

    parser.add_argument("--use_layer", action="store_true")
    parser.set_defaults(use_layer=False)

    parser.add_argument("--loo", action="store_true")
    parser.set_defaults(loo=False)

    # Parse arguments and set global variables

    args = parser.parse_args()

    args.pool_type = "avg"

    assert (
        np.array([args.freeze, args.random_init]).sum() <= 1
    ), "Only one of the two can be True"

    datasets_abs = get_paths_from_dir(args.data_folder)

    # Extract and check the datasets, saving a dataframe with the results
    _, dataset_class = extract_and_check(
        *datasets_abs,
        save=False,
        _labels_only=True,
        normalise_wvf=True,
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
        multi_chan_wave=True,
        _acgs_path=os.path.join(
            args.data_folder, "acgs_vs_firing_rate", "acgs_3d_logscale.npy"
        ),
        _acg_mask=(~granule_mask),
        _acg_multi_factor=10,
    )

    targets = checked_dataset.targets
    spikes = checked_dataset.spikes_list

    if args.use_layer:
        one_hot_layer = encode_layer_info(checked_dataset.layer_list)

    # Prepare model name to save results
    suffix = ""
    features_suffix = ""
    cv_string = "_loo_cv" if args.loo else "_5fold_cv"
    if args.freeze:
        suffix = "_frozen_heads"
    if args.random_init:
        suffix = "_random_init"
    if args.mli_clustering:
        features_suffix += "_mli_clustering"
    features_suffix += "_soft_layer" if args.use_layer else ""
    augmented = ("_augmented_acgs" if args.augment_acg else "") + (
        "_augmented_waveforms" if args.augment_wvf else ""
    )
    model_name = f"deep_semisup{suffix}{augmented}"

    save_folder = os.path.join(
        args.data_folder,
        "dataset_1",
        f"encoded_acg_wvf{features_suffix}",
        model_name,
        f"mouse_results{cv_string}",
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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
        random_init=args.random_init,
        freeze_heads=args.freeze,
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

    # Apply hard layer correction as well if user requested to use layer
    if args.use_layer:
        new_save_folder = os.path.join(
            args.data_folder,
            "dataset_1",
            f"encoded_acg_wvf{features_suffix.split('_soft_layer')[0]}",
            model_name,
            f"mouse_results{cv_string}",
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        corrected_results_dict = post_hoc_layer_correction(
            results_dict, one_hot_layer, LABELLING, repeats=10
        )
        plot_confusion_matrices(
            corrected_results_dict,
            new_save_folder,
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
        normalise_wvf=True,
        lisberger=True,
        _label="expert_label",
        n_channels=10,
        central_range=MONKEY_WAVEFORM_SAMPLES,
        _use_amplitudes=False,
        _lisberger=True,
        _id_type="neuron_id",
        _labels_only=True,
        _extract_mli_clusters=True,
        _extract_layer=args.use_layer,
    )

    #! Remove MLI_Bs from the dataset
    mli_b = np.array(monkey_dataset_class.labels_list) == "MLI_B"
    monkey_dataset_class._apply_mask(~mli_b)
    monkey_dataset_class = datasets.resample_waveforms(monkey_dataset_class)

    monkey_dataset, _ = prepare_classification_dataset(
        monkey_dataset_class,
        normalise_acgs=False,
        multi_chan_wave=False,
        _acgs_path=os.path.join(
            args.data_folder, "acgs_vs_firing_rate", "monkey_acgs_3d_logscale.npy"
        ),
        _acg_mask=~mli_b,
        _acg_multi_factor=10,
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
        "dataset_1",
        f"encoded_acg_wvf{features_suffix}",
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

    # Apply hard layer correction as well if user requested to use layer
    if args.use_layer:
        new_save_folder_monkey = os.path.join(
            args.data_folder,
            "dataset_1",
            f"encoded_acg_wvf{features_suffix.split('_soft_layer')[0]}",
            model_name,
            "monkey_results",
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        corrected_results_dict_monkey = post_hoc_layer_correction(
            results_dict_monkey, monkey_one_hot_layer, LABELLING
        )
        plot_confusion_matrices(
            corrected_results_dict_monkey,
            new_save_folder_monkey,
            model_name,
            labelling=LABELLING,
            correspondence=CORRESPONDENCE,
        )


if __name__ == "__main__":
    main()
