import torch
import torch.nn as nn
from typing import Tuple

class BlockLearnableCompressionMatrix(nn.Module):
    """
    A block-wise learnable channel compression layer for PA signals.

    Each of the `m = n // c` rows of the compression matrix learns a 
    distinct `1 × c` block of weights applied to a disjoint subset of `c` channels.
    These blocks are placed along the diagonal in a sparse (m × n) matrix.
    """

    def __init__(self, c: int, n: int, noise_std: float = 1e-1):
        super().__init__()

        if not isinstance(c, int) or c <= 0:
            raise ValueError("Compression block size c must be a positive integer.")
        if n % c != 0:
            raise ValueError(f"Block size c={c} must divide n={n} evenly.")

        self.n = n
        self.c = c
        self.m = n // c

        # Precompute the column‐indices for each block
        block_idx = (torch.arange(self.m).unsqueeze(1) * c
                     + torch.arange(c).unsqueeze(0))  # shape (m, c)
        self.register_buffer('block_idx', block_idx)

        # Learnable block weights (m × c), init to “block-sum + noise”
        W_init = torch.ones(self.m, self.c)
        W_init += torch.randn_like(W_init) * noise_std
        self.W = nn.Parameter(W_init)

    @property
    def A(self) -> torch.Tensor:
        """
        Returns the full (m × n) compression matrix A, where each row `i`
        has its learned `c` values (from `W[i]`) inserted in columns `[i * c : (i + 1) * c]`.
        """

        # build it on the current device / dtype
        device = self.W.device
        dtype = self.W.dtype
        A_full = torch.zeros(self.m, self.n, device=device, dtype=dtype)
        return A_full.scatter(1, self.block_idx, self.W)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 1, C, T) or (B, C, T)
        Returns (Ax, A_full)
        """
        if x.ndim == 4:
            x = x.squeeze(1)  # (B, C, T)

        B, C, T = x.shape
        if C != self.n:
            raise ValueError(f"Expected {self.n} channels but got {C}.")

        # use the property A to build the full matrix
        A_full = self.A  # (m, n)

        # apply: x (B, C, T) → (B, T, C) @ (C, m) → (B, T, m) → (B, m, T)
        x_t = x.permute(0, 2, 1)
        Ax = x_t.matmul(A_full.t()).permute(0, 2, 1)

        return Ax, A_full