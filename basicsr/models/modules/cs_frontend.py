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
from basicsr.models.modules.learned_upsampler import LearnedUpsampler



class CSFrontend(nn.Module):
    ''' Front end compressed sensing model for photoacoustic imaging.
        This class takes care of all of the flags associated with the compressed sensing model.
        In particular it can configure the following:
        - Whether the compression matrix is learned or not.
            - The matrix size will always be fixed, this determines wether gradients are passed through it.
            - The options for the fixed matrix are BlockSum, Naive
                - If the matrix is learned the matrix choice will be the initialization.

        - Whether the upsampler is learned or not.
            - The fixed upsampler options are pinv and bicubic upsampling.
            - The learned upsampler will always be Learned_Upsampler() from basicsr.models.modules.Learned_Upsampler.

        - Possible future options:
            - Variable lr for the compression matrix and upsampler.
            - Different compression matrix densities.
                - For 8x compression: Blocks of 16 in 2 out, 32 in 4 out, etc.
            - Different upsampling methods.
            - Loss term for the output of the upsampler.
            - Quantization of the compression matrix.

        The input to this class is a tensor of shape (B, C, H, W) where B is the batch size,
        C is the number of channels, H is the height and W is the width. For PA data C is 1.
        The output is a tensor of shape (B, C, H, W) where B is the batch size.

        The compression matrix is of shape (input_size // compression_factor, input_size).

    '''
    def __init__(self, config):
        self.config = config
        self.matrix_learned = config.get('matrix_learned', None)
        self.matrix_init = config.get('matrix_init', None)
        self.upsampler = config.get('upsampler', None)

        if self.matrix_learned is None:
            raise ValueError("Matrix learned flag must be specified in the config, options are: 'learned', 'fixed'.")
        if self.matrix_init is None:
            raise ValueError("Matrix initialization must be specified in the config, options are: 'BlockSum', 'Naive'.")
        if self.upsampler is None:
            raise ValueError("Upsampler must be specified in the config, options are: 'learned', 'bilinear', 'pinv'.")
        
        # initialize the compression matrix
        if self.matrix_learned == 'learned':
            if self.matrix_init == 'BlockSum':
                self.compression_matrix = nn.Parameter(torch.randn(128 , 128))
            elif self.matrix_init == 'Naive':
                self.compression_matrix = nn.Parameter(torch.randn(128, 128))
            else:
                raise ValueError(f"Invalid matrix initialization: {self.matrix_init}. Options are: 'BlockSum', 'Naive'.")
        elif self.matrix_learned == 'fixed':
            if self.matrix_init == 'BlockSum':
                self.compression_matrix = torch.randn(128, 128)
            elif self.matrix_init == 'Naive':
                self.compression_matrix = torch.randn(128, 128)
            else:
                raise ValueError(f"Invalid matrix initialization: {self.matrix_init}. Options are: 'BlockSum', 'Naive'.")
        else:
            raise ValueError(f"Invalid matrix learned flag: {self.matrix_learned}. Options are: 'learned', 'fixed'.")
        
        # initialize the upsampler
        if self.upsampler == 'learned':
            self.upsampler = LearnedUpsampler(compression_factor=config.get('compression_factor', 8))
        elif self.upsampler == 'bilinear':
            self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif self.upsampler == 'pinv':
            self.upsampler = torch.pinverse(self.compression_matrix)
        else:
            raise ValueError(f"Invalid upsampler: {self.upsampler}. Options are: 'learned', 'bilinear', 'pinv'.")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the compression matrix
        if self.matrix_learned == 'learned':
            x = torch.matmul(x, self.compression_matrix)
        else:
            x = torch.matmul(x, self.compression_matrix)

        # Apply the upsampler
        if isinstance(self.upsampler, LearnedUpsampler):
            x = self.upsampler(x)
        else:
            x = self.upsampler(x)

        return x
