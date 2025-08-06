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

    def __init__(self, c: int, n: int, noise_std: float = 1.0):
        super().__init__()

        if not isinstance(c, int) or c <= 0:
            raise ValueError("Compression block size c must be a positive integer.")
        if n % c != 0:
            raise ValueError(f"Block size c={c} must divide n={n} evenly.")

        self.n = n
        self.c = c
        self.m = n // c

        # Create each block as its own nn.Parameter
        for i in range(self.m):
            block = 5 * torch.ones(c) + torch.randn(c) * noise_std  # shape (c,)
            param = nn.Parameter(block)
            setattr(self, f'block{i+1}', param)  # block1, block2, ..., blockm

    @property
    def A(self) -> torch.Tensor:
        """
        Returns the full (m × n) compression matrix A, where each row `i`
        has its learned `c` values (from block{i+1}) inserted in columns `[i * c : (i + 1) * c]`,
        with values constrained to [-1, 1] using tanh.
        """
        # Gather all blocks, apply tanh, and stack into a list of (1, c) tensors
        blocks = [torch.tanh(getattr(self, f'block{i+1}').unsqueeze(0)) for i in range(self.m)]
        A_full = torch.block_diag(*blocks)  # shape (m, n)

        return A_full.to(blocks[0].device, blocks[0].dtype)


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

        A_full = self.A  # (m, n)

        # apply: x (B, C, T) → (B, T, C) @ (C, m) → (B, T, m) → (B, m, T)
        x_t = x.permute(0, 2, 1)
        Ax = x_t.matmul(A_full.t()).permute(0, 2, 1)

        return Ax, A_full