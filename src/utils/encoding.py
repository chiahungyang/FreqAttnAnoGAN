"""Utility module for positional encoding."""


import torch
from torch import nn
import math


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

    def forward(self, data: torch.Tensor):
        """
        Note
        ----
        The input tensor must have shape [batch_size, seq_len, embedding_dim].
        """
        data = data + self.pe[:, :data.size(1)]
        return self.dropout(data)


def main():
    """Empty main."""
    return


if __name__ == '__main__':
    main()