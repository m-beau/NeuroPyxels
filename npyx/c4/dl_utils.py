import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    import torch
    import torch.distributions as dist
    import torch.nn as nn
    import torch.nn.functional as F

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    from torchvision import transforms
except ImportError:
    print(
        (
            "\ntorchvision could not be imported - "
            "some functions from the submodule npyx.c4 will not work.\n"
            "To install torchvision, see https://pypi.org/project/torchvision/."
        )
    )


class ConvEncoderResize(nn.Module):
    def __init__(
        self,
        d_latent,
        image_size=100,
        channels=8,
        pool_window=3,
        kernel_size=6,
        flattened_size=16 * 8 * 8,
        initialise=True,
    ):
        super().__init__()
        if type(pool_window) == int:
            pool_window = (pool_window, pool_window)
        self.resize = transforms.Resize((image_size, image_size))
        self.conv1 = nn.Conv2d(1, channels, kernel_size)
        self.maxpool1 = nn.MaxPool2d(pool_window[0])
        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size)
        self.maxpool2 = nn.MaxPool2d(pool_window[1])
        self.batchnorm2 = nn.BatchNorm2d(channels * 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, d_latent * 2)
        self.d_latent = d_latent
        self.dropout = nn.Dropout(0.2)

        if initialise:
            self.conv1.weight.data.normal_(0, 0.001)
            self.conv1.bias.data.normal_(0, 0.001)

            self.conv2.weight.data.normal_(0, 0.001)
            self.conv2.bias.data.normal_(0, 0.001)

            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

    def forward(self, x) -> dist.Normal:
        x = self.resize(x)
        x = self.conv1(x)
        x = F.gelu(self.maxpool1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.gelu(self.maxpool2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        h = self.dropout(self.fc1(x))
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        return dist.Normal(mu, torch.clip(torch.exp(log_var), 1e-5, 1e5))


class ConvEncoderWVF(nn.Module):
    def __init__(
        self,
        d_latent,
        channels=4,
        pool_window=((1, 2), (1, 2)),
        kernel_size_1=(1, 8),
        kernel_size_2=(3, 1),
        flattened_size=8 * 2 * 20,
        initialise=True,
    ):
        super().__init__()
        if type(pool_window) == int:
            pool_window = (pool_window, pool_window)
        self.conv1 = nn.Conv2d(1, channels, kernel_size_1)
        self.maxpool1 = nn.AvgPool2d(pool_window[0])
        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size_2)
        self.maxpool2 = nn.AvgPool2d(pool_window[1])
        self.batchnorm2 = nn.BatchNorm2d(channels * 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, d_latent * 2)
        self.d_latent = d_latent
        self.dropout = nn.Dropout(0.2)

        if initialise:
            self.conv1.weight.data.normal_(0, 0.001)
            self.conv1.bias.data.normal_(0, 0.001)

            self.conv2.weight.data.normal_(0, 0.001)
            self.conv2.bias.data.normal_(0, 0.001)

            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

    def forward(self, x) -> dist.Normal:
        x = self.conv1(x)
        x = F.gelu(self.maxpool1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.gelu(self.maxpool2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        h = self.dropout(self.fc1(x))
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        return dist.Normal(mu, torch.clip(torch.exp(log_var), 1e-5, 1e5))


class ForwardDecoder(nn.Module):
    def __init__(
        self, d_latent, central_range, n_channels, hidden_units=None, initialise=True
    ):
        super().__init__()
        self.central_range = central_range
        self.n_channels = n_channels
        self.d_latent = d_latent
        if hidden_units is None:
            hidden_units = [100, 200]
        self.fc1 = nn.Linear(d_latent, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], int(n_channels * central_range))

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        if initialise:
            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

            self.fc2.weight.data.normal_(0, 0.001)
            self.fc2.bias.data.normal_(0, 0.001)

            self.fc3.weight.data.normal_(0, 0.001)
            self.fc3.bias.data.normal_(0, 0.001)

    def forward(self, z):
        # flatten the latent vector
        z = z.view(z.shape[0], -1)
        h = self.dropout1(F.relu(self.fc1(z)))
        h = self.dropout2(F.relu(self.fc2(h)))
        X_reconstructed = self.fc3(h)

        return X_reconstructed.reshape(-1, 1, self.n_channels, self.central_range)


class BaseVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder
        self.decoder = decoder

    def reconstruct(self, x):
        return self.decoder(self.encoder(x).mean)

    def encode(self, x: torch.Tensor, augment=False) -> torch.Tensor:
        return self.encoder(x).sample() if augment else self.encoder(x).mean

    def load_weights(self, encoder_path, decoder_path):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))

    def encode_numpy(self, x):
        return NotImplementedError


class ConvolutionalEncoder(nn.Module):
    def __init__(self, d_latent, initialise=False, pool="max"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 10))
        self.maxpool1 = (
            nn.MaxPool2d(kernel_size=(2, 2))
            if pool == "max"
            else nn.AvgPool2d(kernel_size=(2, 2))
        )
        self.batchnorm1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, (5, 1))
        self.maxpool2 = (
            nn.MaxPool2d(kernel_size=(1, 2))
            if pool == "max"
            else nn.AvgPool2d(kernel_size=(1, 2))
        )
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 1 * 23, 200)
        self.fc2 = nn.Linear(200, d_latent * 2)
        self.d_latent = d_latent
        self.dropout = nn.Dropout(0.2)

        if initialise:
            self.conv1.weight.data.normal_(0, 0.001)
            self.conv1.bias.data.normal_(0, 0.001)

            self.conv2.weight.data.normal_(0, 0.001)
            self.conv2.bias.data.normal_(0, 0.001)

            self.fc1.weight.data.normal_(0, 0.001)
            self.fc1.bias.data.normal_(0, 0.001)

            self.fc2.weight.data.normal_(0, 0.001)
            self.fc2.bias.data.normal_(0, 0.001)

    def forward(self, x, return_mu=False) -> dist.Normal:
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = F.relu(self.maxpool2(x))
        x = self.batchnorm2(x)
        x = self.flatten(x)
        h = self.dropout(F.relu(self.fc1(x)))
        h = self.fc2(h)
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        return (
            mu
            if return_mu
            else dist.Normal(mu, torch.exp(log_var), validate_args=False)
        )


class ACG3DVAE(BaseVAE):
    def __init__(self, d_latent, acg_bins, acg_width, device=None, pool="max"):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = ConvolutionalEncoder(d_latent, pool=pool).to(self.device)
        self.decoder = ForwardDecoder(
            d_latent, acg_width, acg_bins, hidden_units=[250, 500]
        ).to(self.device)
        self.acg_bins = acg_bins
        self.acg_width = acg_width

    def encode_numpy(self, x):
        with torch.no_grad():
            self.encoder.eval()
            x_tensor = (
                torch.tensor(x)
                .to(self.device)
                .float()
                .reshape(-1, 1, self.acg_bins, self.acg_width)
            )
            return self.encode(x_tensor).detach().cpu().numpy()


class WFConvVAE(BaseVAE):
    def __init__(
        self, d_latent, central_range, n_channels, initialise=True, device=None
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_channels = n_channels
        self.central_range = central_range
        self.encoder = ConvEncoderWVF(
            d_latent,
            initialise=initialise,
        ).to(self.device)
        self.decoder = ForwardDecoder(
            d_latent,
            central_range,
            n_channels,
            hidden_units=(200, 400),
            initialise=initialise,
        ).to(self.device)

    def encode_numpy(self, x, augment=False):
        with torch.no_grad():
            self.encoder.eval()
            x_tensor = (
                torch.tensor(x)
                .to(self.device)
                .float()
                .reshape(-1, 1, self.n_channels, self.central_range)
            )
            return self.encode(x_tensor, augment).detach().cpu().numpy()


class ACGForwardVAE(nn.Module):
    def __init__(self, encoder, decoder, in_features, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.encoder = encoder
        self.decoder = decoder
        self.in_features = in_features

    def encode_numpy(self, x):
        with torch.no_grad():
            self.encoder.eval()
            x_tensor = (
                torch.tensor(x).to(self.device).float().reshape(-1, 1, self.in_features)
            )
            return self.encode(x_tensor).detach().cpu().numpy()


class Decoder(nn.Module):
    def __init__(self, decoder, d_latent, in_features):
        super().__init__()
        self.decoder = decoder.float()
        self.d_latent = d_latent
        self.in_features = in_features

    def forward(self, z):
        # flatten the latent vector
        z = z.view(z.shape[0], -1)
        # forward pass through decoder network
        h = self.decoder(z)

        return h.reshape(-1, 1, self.in_features)


def define_forward_vae(in_features, init_weights=True, params=None, device=None):
    if params is None:
        best_params = {
            "batch_size": 32,
            "optimizer": "Adam",
            "lr": 1e-3,
            "n_layers": 2,
            "d_latent": 10,
            "n_units_l0": 200,
            "dropout_l0": 0.1,
            "n_units_l1": 100,
            "dropout_l1": 0.1,
        }
    else:
        best_params = params

    n_layers = best_params["n_layers"]
    d_latent = best_params["d_latent"]

    initial_in_features = in_features
    encoder_layers = []
    decoder_layers = []
    first_units = None

    for i in range(n_layers):
        out_features = best_params[f"n_units_l{i}"]
        p = best_params[f"dropout_l{i}"]
        if i == 0:
            first_units = out_features

        # Create and properly init encoder layer
        cur_enc_layer = nn.Linear(in_features, out_features)
        if init_weights:
            cur_enc_layer.weight.data.normal_(0, 0.001)
            cur_enc_layer.bias.data.normal_(0, 0.001)

        # Create and properly init decoder layer
        cur_dec_layer = nn.Linear(out_features, in_features)
        if init_weights:
            cur_dec_layer.weight.data.normal_(0, 0.001)
            cur_dec_layer.bias.data.normal_(0, 0.001)

        encoder_layers.append(cur_enc_layer)
        decoder_layers.append(cur_dec_layer)

        encoder_layers.append(nn.GELU())
        decoder_layers.append(nn.Dropout(p))

        encoder_layers.append(nn.Dropout(p))
        decoder_layers.append(nn.GELU())

        in_features = out_features
    encoder_layers.append(nn.Linear(in_features, d_latent))
    decoder_layers.append(nn.Linear(d_latent, in_features))

    encoder = nn.Sequential(*encoder_layers[:-1], nn.Linear(in_features, 2 * d_latent))
    decoder = nn.Sequential(
        *decoder_layers[:0:-1], nn.Linear(first_units, (initial_in_features))
    )

    encoder = Encoder(encoder, d_latent)
    decoder = Decoder(decoder, d_latent, initial_in_features)

    return encoder.to(device), decoder.to(device)


class CNNCerebellum(nn.Module):
    def __init__(self, acg_head, waveform_head, n_classes=5):
        super(CNNCerebellum, self).__init__()
        self.acg_head = acg_head
        self.wvf_head = waveform_head

        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x):
        acg = x[:, :1010]
        wvf = x[:, 1010:]
        acg = self.acg_head(acg.reshape(-1, 1, 10, 101))
        wvf = self.wvf_head(wvf)
        x = torch.cat((acg.mean, wvf.mean), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_acg_vae(
    encoder_path,
    win_size,
    bin_size,
    d_latent=10,
    initialise=True,
    device=None,
    pool="max",
):
    # Initialise and apply the encoder to the waveforms

    vae = ACG3DVAE(
        d_latent,
        acg_bins=10,
        acg_width=int(win_size // bin_size + 1),
        device=device,
        pool=pool,
    )

    if initialise:
        decoder_path = encoder_path.replace("encoder", "decoder")

        vae.load_weights(encoder_path, decoder_path)

    return vae


def load_waveform_encoder(
    encoder_args, encoder_path, in_features, initialise=True, device=None
):
    enc, _ = define_forward_vae(in_features, params=encoder_args, device=device)

    if initialise:
        enc.load_state_dict(torch.load(encoder_path, map_location=device))

    return enc


def load_waveform_vae(encoder_args, encoder_path, device=None):
    if "device" not in encoder_args.keys():
        encoder_args["device"] = device
    vae = WFConvVAE(**encoder_args)

    decoder_path = encoder_path.replace("encoder", "decoder")

    vae.load_weights(encoder_path, decoder_path)

    return vae


class Encoder(nn.Module):
    def __init__(self, encoder, d_latent):
        super().__init__()
        self.encoder = encoder.float()
        self.d_latent = d_latent

    def forward(self, x, return_mu=False) -> dist.Normal:
        # flatten the image
        x = x.view(x.shape[0], -1)
        # forward pass through encoder network
        h = self.encoder(x)
        # split the output into mu and log_var
        mu = h[:, : self.d_latent]
        log_var = h[:, self.d_latent :]
        # return mu and log_var
        if return_mu:
            return mu
        return dist.Normal(mu, torch.exp(log_var), validate_args=False)


def ELBO_VAE(enc, dec, X, dataset_size, device, beta=1, n_samples=10):
    """
    Computes the Evidence Lower Bound (ELBO) for a Variational Autoencoder (VAE).

    Args:
        enc (nn.Module): The encoder neural network.
        dec (nn.Module): The decoder neural network.
        X (torch.Tensor): The input data tensor.
        dataset_size (int): The size of the dataset.
        device (str): The device to use for computations.
        beta (float, optional): The weight of the KL divergence term in the ELBO. Defaults to 1.
        n_samples (int, optional): The number of samples to use for Monte Carlo estimation. Defaults to 10.

    Returns:
        torch.Tensor: The ELBO value.
    """
    batch_size = X.shape[0]
    ELBO = torch.zeros(batch_size).to(device)
    for _ in range(n_samples):
        q_z = enc.forward(X)  # q(Z | X)
        z = (
            q_z.rsample()
        )  # Samples from the encoder posterior q(Z | X) using the reparameterization trick

        reconstruction = dec.forward(z)  # distribution p(x | z)

        prior = dist.Normal(
            torch.zeros_like(q_z.loc).to(device), torch.ones_like(q_z.scale).to(device)
        )

        MSE = F.mse_loss(reconstruction, X, reduction="none").sum(dim=(1, 2, 3))

        KLD = dist.kl_divergence(q_z, prior).sum(dim=1)

        ELBO += MSE + beta * (batch_size / dataset_size) * KLD

    return (ELBO / n_samples).mean()


def generate_kl_weight(epochs, beta=1):
    """
    Generate an array of weights to be used for the KL divergence loss in a VAE model.
    This function is used to anneal the KL divergence loss over the course of training,
    a simple trick that helps with convergence of VAEs as shown in:
    - Bowman et al., 2015 (https://arxiv.org/abs/1602.02282)
    - Sonderby et al., 2016 (https://arxiv.org/abs/1511.06349)

    Parameters:
    epochs (int): The number of epochs to train the VAE model for.
    beta (float): The scaling factor for the KL divergence loss.

    Returns:
    numpy.ndarray: An array of weights to be used for the KL divergence loss in a VAE model.
    """

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    weight = np.logspace(5, -20, epochs)
    weight = sigmoid(-np.log10(weight)) * beta

    return weight
