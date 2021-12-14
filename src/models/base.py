"""General anomaly detection GAN model framework."""


from typing import Optional
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import instantiate_class
from ..utils import MiniRocket, FrechetInceptionDistance, roc_performance


class BaseAnoGenAdvNet(pl.LightningModule):
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
    
    def discriminate(self, data: torch.Tensor):
        """Return the discrimination score."""
        return self.disc(self.feat(data))

    def reconstruct(self, data: torch.Tensor):
        """Return the reconstruction through the encoder and generator."""
        return self.gen(self.enc(data))

    def reconstruction_err(self, real: torch.Tensor, fake: torch.Tensor):
        """Return the reconstruction error of the encoder and generator."""
        sz = real.size(0)
        real, fake = real.view(sz, -1), fake.view(sz, -1)
        return (real - fake).pow(2).mean(dim=1)

    def feature_err(self, real: torch.Tensor, fake: torch.Tensor):
        """Return the feature matching error in the discriminator."""
        sz = real.size(0)
        real, fake = self.feat(real).view(sz, -1), self.feat(fake).view(sz, -1)
        return (real - fake).pow(2).mean(dim=1)
    
    def anomaly_score(self, data: torch.Tensor):
        """Return the anomaly score."""
        recon = self.reconstruct(data)
        score = self.reconstruction_err(data, recon)
        score += self.feature_err(data, recon) * self.r
        return score

    def residual(self, data: torch.Tensor):
        """Return the residual between the data and its reconstruction."""
        return (data - self.reconstruct(data)).abs()
    
    def forward(self, data: torch.Tensor):
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


class GenWassersteinLoss(nn.Module):
    """
    Wasserstein distance loss with respect to generator.

    Note
    ----
    See for reference Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., &
    Courville, A. (2017). Improved training of wasserstein gans. arXiv preprint
    arXiv:1704.00028.
    """

    def __init__(self, model: BaseAnoGenAdvNet):
        super().__init__()
        self.model = model
    
    def forward(self, data: torch.Tensor):
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

    def __init__(self, model: BaseAnoGenAdvNet, coef: float = 10.0):
        super().__init__()
        self.model = model
        self.coef = coef

    def forward(self, data: torch.Tensor):
        fake = self.model.generate(len(data), device=data.device)
        dist = (self.model.discriminate(fake) - self.model.discriminate(data)).mean()
        return dist + self.coef * self.gradient_penalty(data, fake)
    
    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor):
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


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()