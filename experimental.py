"""Experimental modules and functions."""

from typing import Optional, Callable
import torch
import math
from torch import nn
from torch import Tensor
import pytorch_lightning as pl
import numpy as np
import scipy as sp
import pandas as pd
from pytorch_lightning.utilities.cli import instantiate_class
from sklearn.metrics import roc_curve, auc
# import sktime
from sktime.transformations.panel.rocket import MiniRocketMultivariate


########################################
###  Generative adversarial network
########################################


class AnoGenAdvNet(pl.LightningModule):
    """Generative adversarial network for anomaly detection."""

    def __init__(
        self,
        generator: nn.Module,
        extractor: nn.Module,
        discriminator: nn.Module,
        encoder: nn.Module,
        dim_latent: int,
        training_phase: str = 'encoder',
        ratio: float = 1.0,
        penalty_coef: float = 10.0,
        fid_feat_init: Optional[dict] = None,
        gen_opt_init: Optional[dict] = None,
        disc_opt_init: Optional[dict] = None,
        enc_opt_init: Optional[dict] = None
    ):
        """
        Arguments
        ---------
        generator: Generator module
        extractor: Feature extractor module during discrimination
        discriminator: Discriminator module following feature extractor
        encoder: Encoder module
        dim_latent: Dimension of latent space
        training_phase: Indicator of `'adversarial'` or `'encoder'` training
        ratio: Weight of feature matching error in anomaly score
        penalty_coef: Coefficient of gradient penalty in adversarial training
        fid_feat_init: Class and arguments to instantiate FID feature module
        gen_opt_init: Class and arguments to instantiate optimizer of generator
        disc_opt_init: Class and arguments to instantiate optimizer of discriminator
        enc_opt_init: Class and arguments to instantiate optimizer of encoder

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
        # Submodules
        self.gen = generator
        self.feat = extractor
        self.disc = discriminator
        self.enc = encoder
        # Model parameters
        self.dim = dim_latent
        self.r = ratio
        self.freeze_by_phase(training_phase)
        self.phase = training_phase
        # Loss functions
        self.lossfn_gen = GenWassersteinLoss(self)
        self.lossfn_disc = DiscWassersteinGradPenLoss(self, penalty_coef)
        self.lossfn_enc = self.anomaly_score
        # Validation function
        if fid_feat_init is None:
            fid_feat_init = {
                'class_path': MiniRocket,
                'init_args': {}
            }
        self.fidfn = FrechetInceptionDistance(
            instantiate_class(fid_feat_init)
        )
        # Optimizer parameters
        if gen_opt_init is None:
            gen_opt_init = {
                'class_path': torch.optim.AdamW,
                'init_args': {'lr': 4e-3}
            }
        self.gen_opt_init = gen_opt_init
        if disc_opt_init is None:
            disc_opt_init = {
                'class_path': torch.optim.AdamW,
                'init_args': {'lr': 1e-3}
            }
        self.disc_opt_init = disc_opt_init
        if enc_opt_init is None:
            enc_opt_init = {
                'class_path': torch.optim.AdamW,
                'init_args': {'lr': 1e-3}
            }
        self.enc_opt_init = enc_opt_init
    
    def freeze_by_phase(self, phase):
        """Freeze part of the model according to training phase."""
        if phase == 'adversarial':
            self.gen.requires_grad_(True)
            self.feat.requires_grad_(True)
            self.disc.requires_grad_(True)
            self.enc.requires_grad_(False)
        elif phase == 'encoder':
            self.gen.requires_grad_(False)
            self.feat.requires_grad_(False)
            self.disc.requires_grad_(False)
            self.enc.requires_grad_(True)
        else:
            raise ValueError(f'invalid training phase {phase}')

    def generate(self, batch_size, device=None):
        """Return generated data from the generator."""
        # Generate data from prior samples
        samples = torch.randn(batch_size, self.dim, device=device)
        return self.gen(samples)
    
    def discriminate(self, data: Tensor):
        """Return the discrimination score."""
        return self.disc(self.feat(data))

    def reconstruct(self, data: Tensor):
        """Return the reconstruction through the encoder and generator."""
        return self.gen(self.enc(data))

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
    
    def anomaly_score(self, data: Tensor):
        """Return the anomaly score."""
        recon = self.reconstruct(data)
        score = self.reconstruction_err(data, recon)
        score += self.feature_err(data, recon) * self.r
        return score

    def residual(self, data: Tensor):
        """Return the residual between the data and its reconstruction."""
        return (data - self.reconstruct(data)).abs()
    
    def forward(self, data: Tensor):
        return self.anomaly_score(data)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        data, _ = batch
        if self.phase == 'adversarial':
            if optimizer_idx == 0:  # train generator
                loss = self.lossfn_gen(data)
                self.log('Loss/Generator', loss)
            if optimizer_idx == 1:  # train discriminator
                loss = self.lossfn_disc(data)
                self.log('Loss/Discriminator', loss)
        else:  # train encoder
            loss = self.lossfn_enc(data)
            self.log('Loss/Encoder', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        if self.phase == 'adversarial':  # validate adversarial training
            fake = self.generate(len(data), device=data.device)
            fid = self.fidfn(data, fake)
            self.log('FID', fid)
        else:  # validate encoder training
            scores = self.anomaly_score(data)
            roc_auc, acc = roc_performance(labels, scores)
            self.log('ROC AUC', roc_auc)
            self.log('Accuracy', acc)

    def configure_optimizers(self):
        if self.phase == 'adversarial':
            gen_optimizer = instantiate_class(
                self.gen.parameters(),
                self.gen_opt_init
            )
            disc_optimizer = instantiate_class(
                nn.ModuleList([self.feat, self.disc]).parameters(),
                self.disc_opt_init
            )
            return (
                {'optimizer': gen_optimizer},
                {'optimizer': disc_optimizer},
            )
        else:
            enc_optimizer = instantiate_class(
                self.enc.parameters(),
                self.enc_opt_init
            )
            return {'optimizer': enc_optimizer}


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
###  Validation
####################################


class FrechetInceptionDistance(nn.Module):
    """"
    Frechet inception distance module.

    Notes
    -----
    1. See for reference Heusel, M., Ramsauer, H., Unterthiner, T., Nessler,
       B., & Hochreiter, S. (2017). GANs trained by a two time-scale update
       rule converge to a local nash equilibrium. Advances in neural
       information processing systems, 30.
    2. The formula approximates the data with multivariate Gaussian
       distributions.
    """

    def __init__(self, feature: nn.Module):
        """
        Arguments
        ---------
        feature: Feature extractor acted upon real/generated data
        """
        super().__init__()
        self.feat = feature
    
    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        """
        Return the Frechet incepction distance between features of the real
        and the generated data.

        Pre-condition
        -------------
        `real` and `fake` must have the same shape.
        """
        # Flatten the feature vector for individual samples
        real = self.feat(real).view(real.size(0), -1)
        fake = self.feat(fake).view(fake.size(0), -1)
        # Compute the Frechet inception distance
        dist = np.linalg.norm(real.mean(dim=0) - fake.mean(dim=0)) ** 2
        cov_r, cov_f = np.cov(real.T), np.cov(fake.T)  # covariance matrices
        dist += np.trace(cov_r + cov_f - 2 * sp.linalg.sqrtm(cov_r.dot(cov_f)))
        return dist


class MiniRocket(nn.Module):
    """
    MiniRocket module to extract features from time series

    Notes
    -----
    1. This adopted implementation supposes that data (e.g., validation set)
       only has a single batch.
    2. See for reference Dempster, A., Schmidt, D. F., & Webb, G. I. (2021).
       Minirocket: A very fast (almost) deterministic transform for time series
       classification. In Proceedings of the 27th ACM SIGKDD Conference on
       Knowledge Discovery & Data Mining (pp. 248-257).
    """

    def __init__(self, **kwargs):
        """
        Arguments
        ---------
        kwargs: Keyword arguments passed to MiniRocketMultivariate
        """
        super().__init__()
        self.minirocket = MiniRocketMultivariate(**kwargs)

    def forward(self, data: torch.Tensor):
        """Return the transformed features from time series data."""
        data = pd.DataFrame({  # convert time series to sktime format
            f'dim_{var_idx}': [
                pd.Series(data[ts_idx, var_idx])
                for ts_idx in range(data.size(0))
            ]
            for var_idx in range(data.size(1))
        })
        if not self.minirocket.check_is_fitted(): self.minirocket.fit(data)
        feats = self.minirocket.transform(data)
        feats = data.new_tensor(feats.values)  # convert features to tensor
        return feats


def roc_performance(labels: torch.Tensor, scores: torch.Tensor):
    """
    Return the area under ROC curve and the accuracy at the cut-off point that
    maximizes the sum of sensitivity and specificity.

    Arguments
    ---------
    labels: Binary annotation of data that takes value of either 0 or 1
    scores: Predicted probabilities/scores for classification

    Pre-condition
    -------------
    `labels` and `scores` must have the same shape.
    """
    fpr, tpr, threshold = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    cutoff = threshold[np.argmax(tpr - fpr)]
    acc = ((scores >= cutoff) == labels).float().mean().item()
    return roc_auc, acc


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
        gen_init_args: Optional[dict] = None,
        disc_init_args: Optional[dict] = None,
        enc_init_args: Optional[dict] = None,
        **kwargs
    ):
        """
        Arguments
        ---------
        dim_latent: Dimension of the latent space
        num_vars: Number of variables in the time series data
        num_steps: Number of timesteps in the time series data
        gen_init_args: Keyword arguments passed to the generator
        disc_init_args: Keyword arguments passed to the discriminator
        enc_init_args: Keyword arguments passed to the encoder
        kwargs: Keyword arguments passed to the AnoGenAdvNet superclass
        """
        args = (dim_latent, num_vars, num_steps)
        if gen_init_args is None: gen_init_args = {}
        generator = FreqAttnDecoder(*args, **gen_init_args)
        if disc_init_args is None: disc_init_args = {}
        extractor = FreqAttnEncoder(*args, **disc_init_args)
        discriminator = nn.Linear(dim_latent, 1)
        if enc_init_args is None: enc_init_args = {}
        encoder = FreqAttnEncoder(*args, **enc_init_args)
        super().__init__(
            generator,
            extractor,
            discriminator,
            encoder,
            dim_latent,
            **kwargs
        )


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