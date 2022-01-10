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
        Args:
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
        Args:
            dim_latent: Dimension of the latent space
            num_vars: Number of variables in the time series data
            num_steps: Number of timesteps in the time series data
            num_blocks: Number of structurally similar computation blocks
            num_attnlayers: Number of attention layers per block
            num_attnheads: Number of attention heads
            attn_actv: Activation function in attention modules
        """
        super().__init__()
        # Pre-compute input/output size of each block
        sizes_vars = np.around(np.logspace(
            np.log10(num_vars),
            np.log10(dim_latent),
            num_blocks+1
        )).astype(int)
        sizes_vars, nums_heads = divisible_blocksizes_numattnheads(
            sizes_vars,
            num_attnheads
        )
        sizes_feats = np.around(np.logspace(
            np.log10(num_steps),
            0,
            num_blocks+1
        )).astype(int)
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
                        nhead,
                        activation=attn_actv,
                        batch_first=True
                    ),
                    num_attnlayers
                )
                for dim, nhead in zip(sizes_vars[:-1], nums_heads)
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
        Args:
            dim_latent: Dimension of the latent space
            num_vars: Number of variables in the time series data
            num_steps: Number of timesteps in the time series data
            num_blocks: Number of structurally similar computation blocks
            num_attnlayers: Number of attention layers per block
            num_attnheads: Number of attention heads
            attn_actv: Activation function in attention modules
        """
        super().__init__()
        # Pre-compute input/output size of each block
        sizes_vars = np.around(np.logspace(
            np.log10(dim_latent),
            np.log10(num_vars),
            num_blocks+1
        )).astype(int)
        sizes_vars, nums_heads = divisible_blocksizes_numattnheads(
            sizes_vars,
            num_attnheads
        )
        sizes_feats = np.around(np.logspace(
            0,
            np.log10(num_steps),
            num_blocks+1
        )).astype(int)
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
                        nhead,
                        activation=attn_actv,
                        batch_first=True
                    ),
                    num_attnlayers
                )
                for dim, nhead in zip(sizes_vars[:-1], nums_heads)
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


def divisible_blocksizes_numattnheads(sizes, num_attnheads):
    """
    Return block sizes and numbers of attention heads where the former is
    divisible by the latter.
    """
    sizes = smaller_or_divisible(num_attnheads)(sizes)
    nums = np.where(sizes[:-1] > num_attnheads, num_attnheads, 1)
    return sizes, nums


def smaller_or_divisible(num):
    """
    Return a vectorized function that modifies an array whose elements are
    either smaller or divisible by a number.
    """
    return np.vectorize(lambda x: x if x < num else num * round(x / num))


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()