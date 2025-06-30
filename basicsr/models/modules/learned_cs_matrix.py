import torch
import torch.nn as nn
from typing import Tuple

class LearnableCompressionMatrix(nn.Module):
    """
    A learnable channel compression layer for PA signals.

    Compresses along the input channel (transducer) axis only.
    """

    def __init__(self, c: int, n: int):
        """
        Parameters
        ----------
        c : int
            Compression ratio (e.g., 8 means compress from 128 to 16 channels).
        n : int
            Number of input channels (e.g., 128 transducers).
        """
        super().__init__()

        if not isinstance(c, int) or c <= 0:
            raise ValueError("Compression ratio c must be a positive integer.")
        if n % c != 0:
            raise ValueError(f"Compression ratio c={c} must divide n={n} evenly.")

        self.n = n
        self.c = c
        self.m = n // c
        self.A = nn.Parameter(torch.randn(self.m, self.n) * 0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, C, T) or (B, C, T).

        Returns
        -------
        torch.Tensor
            Compressed output of shape (B, m, T)
        torch.Tensor
            The matrix A used for compression
        """
        if x.ndim == 4:
            x = x.squeeze(1)  # from (B, 1, C, T) to (B, C, T)

        B, C, T = x.shape
        if C != self.n:
            raise ValueError(f"Expected {self.n} input channels but got {C}.")

        x_perm = x.permute(0, 2, 1)           # (B, T, C)
        Ax = torch.matmul(x_perm, self.A.t()) # (B, T, m)
        Ax = Ax.permute(0, 2, 1)              # (B, m, T)
        return Ax, self.A