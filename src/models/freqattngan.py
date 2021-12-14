"""Implementation of anomaly detection GAN with attention in frequency domain."""


from typing import Optional
import torch
from torch import nn
import numpy as np
from .base import BaseAnoGenAdvNet
from ..utils import PositionalEncoding


class FreqAttnAnoGenAdvNet(BaseAnoGenAdvNet):
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
    
    def forward(self, data: torch.Tensor):
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

    def to_frequency_domain(self, data: torch.Tensor):
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
    
    def forward(self, latent: torch.Tensor):
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

    def from_frequency_domain(self, latent: torch.Tensor):
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


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()