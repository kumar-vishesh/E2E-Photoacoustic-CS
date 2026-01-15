# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target, reduction='none', beta=1e-5)

# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss."""

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * smooth_l1_loss(
            pred, target, weight, reduction=self.reduction)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TGCMetric(nn.Module):
    """
    Time Gain Compensation (TGC) Metric Loss.
    Applies inverse Beer-Lambert (absorption) AND linear geometric spreading 
    compensation along the time axis.
    """

    def __init__(self, loss_weight, tgc_weight=0.5, size=1024, resolution=0.0375, reduction='mean'):
        super(TGCMetric, self).__init__()
        
        self.loss_weight = loss_weight
        self.size = size
        self.resolution = resolution
        self.reduction = reduction

        # 1. Create the distance vector z (in mm)
        # We add 'resolution' to avoid starting at 0 (geometric spreading at z=0 is undefined)
        z = torch.arange(size).float() * resolution
        z = z + resolution 

        # 2. Absorption Component (Linear-in-dB)
        # dB_factor = 10^(dB/20) which is equivalent to e^(dB * ln(10) / 20)
        absorption_gain = torch.exp(tgc_weight * z / 20 * np.log(10))

        # 3. Geometric Spreading Component (Linear-in-Time)
        # Compensation for 1/z decay is simply multiplying by z
        geometric_gain = z

        # 4. Combined Weights
        # Total Gain = z * e^(alpha * z)
        self.tgc_weights = absorption_gain * geometric_gain

    def forward(self, pred, target):
        assert pred.shape == target.shape, "Predicted and target tensors must have the same shape."
        
        device = pred.device
        # Ensure weights are on the correct device and reshaped for (N, C, H, W)
        # Assuming the 'time/depth' axis is the last dimension (W)
        tgc_weights = self.tgc_weights.to(device).view(1, 1, 1, -1)

        # Apply TGC weights to both tensors
        # This penalizes errors in the deep field more heavily to counteract signal decay
        pred_tgc = pred * tgc_weights
        target_tgc = target * tgc_weights

        loss = F.l1_loss(pred_tgc, target_tgc, reduction=self.reduction)

        return self.loss_weight * loss
        
# if __name__ == '__main__':
#     # Example usage
#     tgc_loss = TGCMetric(loss_weight=1.0, tgc_weight=0.2, size=1024, resolution=0.0375, reduction='mean')
#     mse = MSELoss()
#     pred = torch.ones(8, 1, 128, 1024)
#     target = torch.zeros(8, 1, 128, 1024)
#     print('TGC Metric Loss:', tgc_loss(pred, target).item())
#     print('MSE Loss:', mse(pred, target).item())