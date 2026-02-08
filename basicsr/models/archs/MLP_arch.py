# Fully Connected MLP Architecture for Image Reconstruction
# This is a simple MLP architecture that takes a multiplexed image as input and learns to reconstruct the original image.
# It is not designed for high performance but serves as a baseline for comparison.

import torch
import torch.nn as nn

clas MLP_arch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):