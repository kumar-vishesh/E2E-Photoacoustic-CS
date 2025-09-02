# ------------------------------------------------------------------------
# VK (2025)
# Frontend for compressed sensing (A) + upsampling (pinv or learned)
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from basicsr.models.modules.Learned_Upsampler import Learned_Upsampler


class CSFrontend(nn.Module):
    """
    Compression with matrix A (learned or fixed), followed by upsampling
    via pinv(A) or a learned upsampler.

    Supported combos:
      1) fixed A  + pinv   → no grads to A
      2) fixed A  + learned→ grads only to upsampler
      3) learned A+ pinv   → grads flow through pinv to A
      4) learned A+ learned→ grads to both A and upsampler
    """
    def __init__(self, config):
        super().__init__()
        self.compression_factor = config.get('compression_factor', 8)
        self.input_size = config.get('input_size', 128)
        self.output_size = self.input_size // self.compression_factor

        self.matrix_learned = config.get('matrix_learned', 'learned').lower()  # 'fixed' | 'learned'
        self.matrix_init    = config.get('matrix_init', 'blocksum').lower()
        self.upsampler_type = config.get('upsampler', 'learned').lower()       # 'pinv' | 'learned'

        self._init_compression_matrix()  # sets self.compression_matrix (Parameter or buffer)
        self._init_upsampler()           # sets self.upsample (Module)

    # ---------------- A (compression matrix) ----------------
    def _init_compression_matrix(self):
        matrix_type   = self.matrix_init
        num_blocks    = self.output_size                 # rows of A
        num_channels  = self.input_size                  # cols of A
        k             = self.compression_factor

        if matrix_type == "blocksum":
            group = torch.ones(k)                        # shape fixes left as you had them
            mat = torch.kron(torch.eye(num_blocks), group)
        elif matrix_type == "blockwise_random":
            mat = torch.zeros((num_blocks, num_channels))
            for i in range(num_blocks):
                s, e = i * k, (i + 1) * k
                block = (torch.randint(0, 2, (k,)) * 2 - 1).float()
                mat[i, s:e] = block
        elif matrix_type == "fully_random":
            mat = torch.zeros((num_blocks, num_channels))
            perm = torch.randperm(num_channels)
            for i in range(num_blocks):
                idx = perm[i * k:(i + 1) * k]
                mat[i, idx] = (torch.randint(0, 2, (k,)) * 2 - 1).float()
        elif matrix_type == "naive":
            mat = torch.zeros((num_blocks, num_channels))
            for i in range(num_blocks):
                mat[i, i * k] = 1.0
        else:
            raise ValueError(f"Unknown matrix type: {self.matrix_init}")

        mat = mat.float()

        if self.matrix_learned == 'learned':
            # Cases (3) and (4): trainable A
            self.compression_matrix = nn.Parameter(mat)
        elif self.matrix_learned == 'fixed':
            # Cases (1) and (2): non-trainable A
            self.register_buffer('compression_matrix', mat)
        else:
            raise ValueError("matrix_learned must be 'fixed' or 'learned'")

    # ---------------- Upsampler ----------------
    def _init_upsampler(self):
        if self.upsampler_type == 'learned':
            # Cases (2) and (4): learnable upsampler
            self.upsample = Learned_Upsampler(compression_factor=self.compression_factor)

        elif self.upsampler_type == 'pinv':
            if self.matrix_learned == 'fixed':
                # Case (1): fixed A + pinv  → compute once (no grad to A)
                pinv_A = torch.linalg.pinv(self.compression_matrix)
                self.register_buffer('pinv_matrix', pinv_A)

                class FixedPinv(nn.Module):
                    def __init__(self, pinv):
                        super().__init__()
                        self.register_buffer('pinv', pinv)
                    def forward(self, x):
                        # x: (B,C,rows,W) ; pinv: (cols,rows) ; output: (B,C,cols,W)
                        return (x.permute(0,1,3,2) @ self.pinv.t()).permute(0,1,3,2)

                self.upsample = FixedPinv(self.pinv_matrix)

            else:
                # Case (3): learned A + pinv → recompute each forward (grad flows to A)
                class PinvUpsampler(nn.Module):
                    def __init__(self, getA):
                        super().__init__()
                        self._getA = getA   # callable; avoids parent<->child cycle
                    def forward(self, x):
                        A = self._getA()                 # (rows, cols)
                        pinv_A = torch.linalg.pinv(A)    # differentiable wrt A
                        return (x.permute(0,1,3,2) @ pinv_A.t()).permute(0,1,3,2)

                self.upsample = PinvUpsampler(self.get_matrix)
        else:
            raise ValueError("upsampler must be 'pinv' or 'learned'")

    # ---------------- Forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        A = self.get_matrix()                                # (rows=H/k, cols=H)
        x_cs = (x.permute(0,1,3,2) @ A.t()).permute(0,1,3,2) # (B,C,rows,W)

        if self.upsampler_type == 'learned' and x_cs.shape[1] != 1:
            raise ValueError(f"Learned_Upsampler expects C==1, got {x_cs.shape[1]}")

        return self.upsample(x_cs)

    # ---------------- Accessors ----------------
    def get_matrix(self) -> torch.Tensor:
        return self.compression_matrix

    def set_matrix(self, new_matrix: torch.Tensor):
        with torch.no_grad():
            self.compression_matrix.copy_(new_matrix)

    # (optional helpers)
    def get_compressed(self, x: torch.Tensor) -> torch.Tensor:
        A = self.get_matrix()
        return (x.permute(0,1,3,2) @ A.t()).permute(0,1,3,2)

    def get_upsampled(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)