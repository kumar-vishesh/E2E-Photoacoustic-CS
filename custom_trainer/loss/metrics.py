import torch
import torch.nn as nn
import torch.nn.functional as F

# L1 Loss
class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(input - target))

# MSE Loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)

# Time Gain Compensation (TGC) Metric
class TGCMetric(nn.Module):
    def __init__(self, eps=1e-8):
        super(TGCMetric, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        """
        Compute Time Gain Compensation (TGC) metric. An inverse Beer-Lambert law along the time axis(assumed to be the last dimension).
        
        Args:
            input (Tensor): Predicted tensor of shape (N, C, H, W).
            target (Tensor): Ground truth tensor of shape (N, C, H, W).
        
        Returns:
            Tensor: Computed TGC metric.
        """
        return NotImplementedError
        N, C, H, W = input.shape
        device = input.device

        # Create exponential weights along the time axis (W)
        weights = torch.exp(-torch.linspace(0, 1, steps=W, device=device))
        weights = weights / (weights.sum() + self.eps)  # Normalize weights

        # Compute weighted L2 loss
        diff = (input - target) ** 2  # Squared differences
        weighted_diff = diff * weights.view(1, 1, 1, W)  # Apply weights
        tgc_metric = torch.sum(weighted_diff) / (N * C * H + self.eps)  # Average over N, C, H

        return tgc_metric


