"""Experimental modules and functions."""

import torch
import math
from torch import nn
from torch import Tensor
import numpy as np
import scipy as sp
from sklearn.metrics import roc_auc_score


########################################
###  Generative adversarial network
########################################


class AnoGenAdvNet(nn.Module):
    """Generative adversarial network for anomaly detection."""

    def __init__(
        self,
        generator: nn.Module,
        extractor: nn.Module,
        discriminator: nn.Module,
        encoder: nn.Module,
        dim_latent: int,
        ratio=1
    ):
        """
        Initialize with the generator, discriminator (along with its feature
        extractor), and encoder.

        Pre-conditions
        --------------
        For a given batch size `sz`:
        1. The input size of `generator` and the output size of `encoder` must
        match with (`sz`, `dim_latent`).
        2. The input size of `extractor` and `encoder` must match the output
        size of `generator`.
        3. The input size of `discriminator` must mathc the output size of
        `extractor`.
        4. The output size of `discriminator` must match with (`sz`,).

        Note
        ----
        See for reference Schlegl, T., Seeböck, P., Waldstein, S. M., Langs, G.,
        & Schmidt-Erfurth, U. (2019). f-AnoGAN: Fast unsupervised anomaly
        detection with generative adversarial networks. Medical image analysis,
        54, 30-44.
        """
        super().__init__()
        self.gen = generator
        self.feat = extractor
        self.disc = discriminator
        self.enc = encoder
        self.r = ratio
        self.dim = dim_latent
    
    def forward(self, data: Tensor):
        """Return the anomaly score."""
        recon = self.reconstruct(data)
        score = self.reconstruction_err(data, recon)
        score += self.feature_err(data, recon) * self.r
        return score

    def reconstruction_err(self, real: Tensor, fake: Tensor):
        """Return the reconstruction error of the encoder and generator."""
        sz = real.size(0)
        real, fake = real.view(sz, -1), fake.view(sz, -1)
        return (real - fake).pow(2).mean(dim=1)

    def feature_err(self, real: Tensor, fake: Tensor):
        """Return the feature matching error in the discriminator."""
        sz = real.size(0)
        real, fake = self.feat(real).view(sz, -1), self.feat(fake).view(sz, -1)
        return (real - fake).pow(2).mean(dim=1)

    def reconstruct(self, data: Tensor):
        """Return the reconstruction through the encoder and generator."""
        return self.gen(self.enc(data))

    def generate(self, batch_size, device=None):
        """Return generated data from the generator."""
        # Generate data from prior samples
        samples = torch.randn(batch_size, self.dim, device=device)
        return self.gen(samples)
    
    def discriminate(self, data: Tensor):
        """Return the discrimination score."""
        return self.disc(self.feat(data))

    def freeze_generator_discriminator(self):
        """Un-track gradient of the generator and the discriminator."""
        self.gen.requires_grad_(False)
        self.feat.requires_grad_(False)
        self.disc.requires_grad_(False)

    def residual(self, data: Tensor):
        """Return the residual between the data and its reconstruction."""
        return (data - self.reconstruct(data)).abs()


####################################
###  Loss functions
####################################


class GenWassersteinLoss(nn.Module):
    """
    Wasserstein distance loss with respect to generator.

    Note
    ----
    See for reference Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., &
    Courville, A. (2017). Improved training of wasserstein gans. arXiv preprint
    arXiv:1704.00028.
    """

    def __init__(self, model: AnoGenAdvNet):
        super().__init__()
        self.model = model
    
    def forward(self, data: Tensor):
        fake = self.model.generate(len(data), device=data.device)
        return -1 * self.model.discriminate(fake).mean()


class DiscWassersteinGradPenLoss(nn.Module):
    """
    Wasserstein distance loss with gradient penalty for discriminator.

    Note
    ----
    See for reference Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., &
    Courville, A. (2017). Improved training of wasserstein gans. arXiv preprint
    arXiv:1704.00028.
    """

    def __init__(self, model: AnoGenAdvNet, coef: float = 10.0):
        super().__init__()
        self.model = model
        self.coef = coef

    def forward(self, data: Tensor):
        fake = self.model.generate(len(data), device=data.device)
        dist = (self.model.discriminate(fake) - self.model.discriminate(data)).mean()
        return dist + self.coef * self.gradient_penalty(data, fake)
    
    def gradient_penalty(self, real: Tensor, fake: Tensor):
        """
        Return gradient penalty for the discriminator to be 1-Lipschitz.
        
        This implementation is adopted from the repository
        https://github.com/eriklindernoren/PyTorch-GAN.
        """
        # Create interpolated variables with random ratios
        r_inter = torch.rand(
            (len(real),) + (1,) * (len(real.size()) - 1),
            device=real.device
        )
        inter = (r_inter * real + (1 - r_inter) * fake).requires_grad_(True)
        # Compute the gradient of discriminator output w.r.t the interpolates
        disc = self.model.discriminate(inter)
        gradient, = torch.autograd.grad(
            outputs=disc,
            inputs=inter,
            grad_outputs=torch.ones_like(disc),
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )
        gradient = gradient.view(gradient.size(0), -1)  # flatten gradients
        return (gradient.norm(p=2, dim=1) - 1).pow(2).mean()


####################################
###  Training
####################################


def train_generator_discriminator(
    dataloader,
    loss_fn_disc,
    loss_fn_gen,
    optimizer_gen,
    optimizer_disc,
    n_disc=1
):
    """Train the generator and discriminator in alternations."""
    for batch, data in enumerate(dataloader):
        # Update the discriminator
        loss_disc = loss_fn_disc(data)
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()
        # Update the generator every `n_disc` batches
        if (batch + 1) % n_disc == 0:
            loss_gen = loss_fn_gen(data)
            optimizer_gen.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()


def train_encoder(dataloader, model: AnoGenAdvNet, optimizer):
    """Train the encoder of `model` to minimize the output anomaly score."""
    for batch, data in enumerate(dataloader):
        # Update the model
        loss = model(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


####################################
###  Evaluation
####################################


def frechet_inception_distance(real: Tensor, fake: Tensor, feature: nn.Module):
    """
    Return the Frechet incepction distance between features of the real and
    the generated data.

    Pre-condition
    -------------
    `real` and `fake` must have the same shape.

    Notes
    -----
    1. See for reference Heusel, M., Ramsauer, H., Unterthiner, T., Nessler,
       B., & Hochreiter, S. (2017). GANs trained by a two time-scale update
       rule converge to a local nash equilibrium. Advances in neural
       information processing systems, 30.
    2. The formula approximates the data with multivariate Gaussian
       distributions.
    """
    # Flatten the feature vector for individual samples
    real = feature(real).view(real.size(0), -1)
    fake = feature(fake).view(fake.size(0), -1)
    # Compute the Frechet inception distance
    dist = np.linalg.norm(real.mean(dim=0) - fake.mean(dim=0)) ** 2
    cov_r, cov_f = np.cov(real.T), np.cov(fake.T)  # covariance matrices
    dist += np.trace(cov_r + cov_f - 2 * sp.linalg.sqrtm(cov_r.dot(cov_f)))
    return dist


def generator_discriminator_evaluation(
    dataloader,
    model: AnoGenAdvNet,
    feature: nn.Module
):
    """
    Return the Frechet inception distance between real and generated data
    under a given feature extraction.

    Pre-condition
    -------------
    `dataloader` should iterate over only one batch.
    """
    with torch.no_grad():
        real, _ = next(iter(dataloader))
        fake = model.generate(len(real), device=real.device)
    return frechet_inception_distance(real, fake, feature)


def model_evaluation(dataloader, model: AnoGenAdvNet):
    """
    Return the area under the ROC curve of `model`.

    Pre-condition
    -------------
    `dataloader` should iterate over only one batch.
    """
    with torch.no_grad():
        data, labels = next(iter(dataloader))
        scores = model(data)  # anomaly score from `model`
    return roc_auc_score(labels, scores)


##################################################
###  Attention mechanism in frequency domain
##################################################


class FreqAttnAnoGenAdvNet(AnoGenAdvNet):
    """Anomaly detection GAN with attention mechanism in frequency domain."""

    def __init__(
        self,
        dim_latent: int,
        num_vars: int,
        num_steps: int,
        kwargs_gen: dict = {},
        kwargs_disc: dict = {},
        kwargs_enc: dict = {}
    ):
        """
        Arguments
        ---------
            dim_latent: Dimension of the latent space
            num_vars: Number of variables in the time series data
            num_steps: Number of timesteps in the time series data
            kwargs_gen: Keyword arguments passed to the generator
            kwargs_disc: Keyword arguments passed to the discriminator
            kwargs_enc: Keyword arguments passed to the encoder
        """
        args = (dim_latent, num_vars, num_steps)
        generator = FreqAttnDecoder(*args, **kwargs_gen)
        extractor = FreqAttnEncoder(*args, **kwargs_disc)
        discriminator = nn.Linear(dim_latent, 1)
        encoder = FreqAttnEncoder(*args, **kwargs_enc)
        super().__init__(
            generator,
            extractor,
            discriminator,
            encoder,
            dim_latent
        )
        self.dim_latent = dim_latent
        self.num_vars = num_vars
        self.num_feats = num_steps


class FreqAttnEncoder(nn.Module):
    """
    Encoder from time series data to a latent space with attention mechanism
    between frequency components.
    """

    def __init__(
        self,
        dim_latent: int,
        num_vars: int,
        num_steps: int,
        num_blocks: int = 4,
        num_attnlayers: int = 3,
        num_attnheads: int = 8,
        attn_actv: str = 'relu'
    ):
        """
        Arguments
        ---------
            dim_latent: Dimension of the latent space
            num_vars: Number of variables in the time series data
            num_steps: Number of timesteps in the time series data
            num_blocks: Number of structurally similar computation blocks
            num_attnlayers: Number of attention layers per block
            num_attnheads: Number of attention heads
            attn_actv: Activation function in attention modules
        """
        super().__init__()
        self.dim_latent = dim_latent
        self.num_vars = num_vars
        self.num_feats = num_steps
        self.num_blocks = num_blocks
        self.num_attnlayers = num_attnlayers
        self.num_attnheads = num_attnheads
        self.attn_actv = attn_actv
        # Pre-compute input/output size of each block
        sizes_vars = np.logspace(
            np.log10(self.num_vars),
            np.log10(self.dim_latent),
            self.num_blocks+1,
            dtype=int
        )
        sizes_feats = np.logspace(
            np.log10(self.num_feats),
            0,
            self.num_blocks+1,
            dtype=int
        )
        sizes_kernel = sizes_feats[:-1] - sizes_feats[1:] + 1
        # Initialize modules in each block
        self.pos_encs = nn.ModuleList(
            [
                PositionalEncoding(dim, max_len=_len)
                for dim, _len in zip(sizes_vars[:-1], sizes_feats[:-1])
            ]
        )
        self.attns = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        dim,
                        self.num_attnheads,
                        activation=self.attn_actv,
                        batch_first=True
                    ),
                    self.num_attnlayers
                )
                for dim in sizes_vars[:-1]
            ]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(sz_in, sz_out, sz_k)
                for sz_in, sz_out, sz_k in zip(
                    sizes_vars[:-1],
                    sizes_vars[1:],
                    sizes_kernel
                )
            ]
        )
    
    def forward(self, data: Tensor):
        """
        Notes
        -----
        1. The inpupt tensor is organized as [batch, variable, timestamp].
        2. The input tensor must have shape [*, self.num_vars, self.num_feats].
        """
        latent = self.to_frequency_domain(data)
        for pos_enc, attn, conv in zip(self.pos_encs, self.attns, self.convs):
            latent = pos_enc(latent.transpose(-1, -2))
            latent = attn(latent)
            latent = conv(latent.transpose(-1, -2))
        latent = latent.squeeze()  # flatten the latent feature vector
        return latent

    def to_frequency_domain(self, data: Tensor):
        """
        Return the frequency components of time series data (as a vector of
        the same degree of freedom).
        """
        # Convert time series to frequency components
        latent = torch.fft.rfft(data, norm="ortho")
        latent = torch.view_as_real(latent)
        # Remove zeros constrained by frrequency components of real values
        mask = latent.new_ones(latent.size()[-2:], dtype=torch.bool)
        if data.size(-1) % 2 == 0:
            mask[(0, -1), 1] = False
        else:
            mask[0, 1] = False
        latent = latent[..., mask]
        return latent


class FreqAttnDecoder(nn.Module):
    """
    Decoder from a latent space to time series data with attention mechanism
    between frequency components.
    """

    def __init__(
        self,
        dim_latent: int,
        num_vars: int,
        num_steps: int,
        num_blocks: int = 4,
        num_attnlayers: int = 3,
        num_attnheads:int = 8,
        attn_actv: str = 'relu'
    ):
        """
        Arguments
        ---------
            dim_latent: Dimension of the latent space
            num_vars: Number of variables in the time series data
            num_steps: Number of timesteps in the time series data
            num_blocks: Number of structurally similar computation blocks
            num_attnlayers: Number of attention layers per block
            num_attnheads: Number of attention heads
            attn_actv: Activation function in attention modules
        """
        super().__init__()
        self.dim_latent = dim_latent
        self.num_vars = num_vars
        self.num_feats = num_steps
        self.num_blocks = num_blocks
        self.num_attnlayers = num_attnlayers
        self.num_attnheads = num_attnheads
        self.attn_actv = attn_actv
        # Pre-compute input/output size of each block
        sizes_vars = np.logspace(
            np.log10(self.dim_latent),
            np.log10(self.num_vars),
            self.num_blocks+1,
            dtype=int
        )
        sizes_feats = np.logspace(
            0,
            np.log10(self.num_feats),
            self.num_blocks+1,
            dtype=int
        )
        sizes_kernel = sizes_feats[1:] - sizes_feats[:-1] + 1
        # Initialize modules in each block
        self.pos_encs = nn.ModuleList(
            [
                PositionalEncoding(dim, max_len=_len)
                for dim, _len in zip(sizes_vars[:-1], sizes_feats[:-1])
            ]
        )
        self.attns = nn.ModuleList(
            [
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        dim,
                        self.num_attnheads,
                        activation=self.attn_actv,
                        batch_first=True
                    ),
                    self.num_attnlayers
                )
                for dim in sizes_vars[:-1]
            ]
        )
        self.convs = nn.ModuleList(
            [
                nn.ConvTranspose1d(sz_in, sz_out, sz_k)
                for sz_in, sz_out, sz_k in zip(
                    sizes_vars[:-1],
                    sizes_vars[1:],
                    sizes_kernel
                )
            ]
        )
    
    def forward(self, latent: Tensor):
        """
        Notes
        -----
        1. The input tensor is organized as [batch, feature].
        2. The input tensor must have shape [*, self.dim_latent].
        """
        latent = latent.unsqueeze(dim=-1)  # expand the latent feature vector
        for pos_enc, attn, conv in zip(self.pos_encs, self.attns, self.convs):
            latent = pos_enc(latent.transpose(-1, -2))
            latent = attn(latent)
            latent = conv(latent.transpose(-1, -2))
        data = self.from_frequency_domain(latent)
        return data

    def from_frequency_domain(self, latent: Tensor):
        """
        Return the time series data from frequency components (as a vector of
        the same degree of freedom).
        """
        # Add zeros to ensure frequency components of real-valued time series
        _len = latent.size(-1)
        zeros = torch.zeros_like(latent[..., :1])
        if _len % 2 == 0:
            latent = torch.cat(
                (latent[..., :1], zeros, latent[..., 1:], zeros),
                dim=-1
            )
        else:
            latent = torch.cat(
                (latent[..., :1], zeros, latent[..., 1:]),
                dim=-1
            )
        latent = latent.view(*latent.size()[:-1], -1, 2)
        # Convert frequency components to time series
        latent = torch.view_as_complex(latent)
        data = torch.fft.irfft(latent, n=_len, norm="ortho")
        return data


class PositionalEncoding(nn.Module):
    """
    Position encoding layer for transformer-based modules.

    This implementation is adopted from the Pytorch tutorial
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(0)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, data: Tensor) -> Tensor:
        """
        Note
        ----
        The input tensor must have shape [batch_size, seq_len, embedding_dim].
        """
        data = data + self.pe[:, :data.size(1)]
        return self.dropout(data)


####################################
###  End matters
####################################


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()