# ------------------------------------------------------------------------
# Vishesh Kumar, 2025 
# ------------------------------------------------------------------------
# This is the front end compressed sensing model for photoacoustic imaging.
# ------------------------------------------------------------------------
# NOTE THIS FILE IS NOT IN USE YET, IT IS A WORK IN PROGRESS.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from basicsr.models.modules.Learned_Upsampler import LearnedUpsampler

class CSFrontend(nn.Module):
    """
    Modular front end for compressed sensing: compression + upsampling.
    Compression is done by a matrix (learnable or fixed), upsampling by pinv, bilinear, or learned upsampler.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.compression_factor = config.get('compression_factor', 8)
        self.input_size = config.get('input_size', 128)
        self.output_size = self.input_size // self.compression_factor

        self.matrix_learned = config.get('matrix_learned', 'learned')
        self.matrix_init = config.get('matrix_init', 'BlockSum')
        self.upsampler_type = config.get('upsampler', 'learned')

        self._init_compression_matrix()
        self._init_upsampler()

    def _init_compression_matrix(self):
        # Initialize compression matrix shape: (output_size, input_size)
        if self.matrix_learned == 'learned':
            if self.matrix_init == 'BlockSum':
                mat = torch.randn(self.output_size, self.input_size)
            elif self.matrix_init == 'Naive':
                mat = torch.randn(self.output_size, self.input_size)
            else:
                raise ValueError(f"Invalid matrix initialization: {self.matrix_init}")
            self.compression_matrix = nn.Parameter(mat)
        elif self.matrix_learned == 'fixed':
            if self.matrix_init == 'BlockSum':
                mat = torch.randn(self.output_size, self.input_size)
            elif self.matrix_init == 'Naive':
                mat = torch.randn(self.output_size, self.input_size)
            else:
                raise ValueError(f"Invalid matrix initialization: {self.matrix_init}")
            self.register_buffer('compression_matrix', mat)
        else:
            raise ValueError(f"Invalid matrix_learned flag: {self.matrix_learned}")

    def _init_upsampler(self):
        if self.upsampler_type == 'learned':
            self.upsampler = LearnedUpsampler(compression_factor=self.compression_factor)
        elif self.upsampler_type == 'bilinear':
            self.upsampler = nn.Upsample(scale_factor=self.compression_factor, mode='bilinear', align_corners=False)
        elif self.upsampler_type == 'pinv':
            # pinv matrix: (input_size, output_size)
            pinv_mat = torch.pinverse(self.compression_matrix if isinstance(self.compression_matrix, torch.Tensor) else self.compression_matrix.data)
            self.register_buffer('pinv_matrix', pinv_mat)
            self.upsampler = None
        else:
            raise ValueError(f"Invalid upsampler type: {self.upsampler_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) or (B, 1, H, W)
        B, C, H, W = x.shape
        # Flatten spatial dims for compression: (B, C, H*W)
        x_flat = x.view(B, C, -1)
        # Apply compression matrix: (output_size, input_size)
        if isinstance(self.compression_matrix, nn.Parameter) or self.matrix_learned == 'learned':
            x_cs = torch.matmul(x_flat, self.compression_matrix.t())  # (B, C, output_size)
        else:
            x_cs = torch.matmul(x_flat, self.compression_matrix.t())

        # Upsampling
        if self.upsampler_type == 'pinv':
            # Apply pseudo-inverse: (B, C, input_size)
            x_up = torch.matmul(x_cs, self.pinv_matrix.t())
            # Reshape back to (B, C, H, W)
            x_up = x_up.view(B, C, H, W)
        else:
            # Reshape to (B, C, output_H, output_W) for upsampler
            # Assume output_H, output_W = H // compression_factor, W // compression_factor
            output_H = H // self.compression_factor
            output_W = W // self.compression_factor
            x_cs_img = x_cs.view(B, C, output_H, output_W)
            x_up = self.upsampler(x_cs_img)
        return x_up

    def set_matrix(self, new_matrix: torch.Tensor):
        """Update compression matrix."""
        if self.matrix_learned == 'learned':
            self.compression_matrix.data.copy_(new_matrix)
        else:
            self.compression_matrix.copy_(new_matrix)

    def get_matrix(self) -> torch.Tensor:
        return self.compression_matrix

    def get_upsampler(self):
        if self.upsampler_type == 'pinv':
            return self.pinv_matrix
        return self.upsampler
        y = self.upsampler(x_compressed)

        return x