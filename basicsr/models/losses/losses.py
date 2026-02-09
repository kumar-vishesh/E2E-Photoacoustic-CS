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
from basicsr.models.losses.beamformers import DifferentiableStolt
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
    
class TGCMetric(nn.Module):
    """
    Time Gain Compensation (TGC) Metric Loss.
    Applies inverse Beer-Lambert (absorption) AND linear geometric spreading 
    compensation along the time axis.
    """

    def __init__(self, loss_weight, tgc_weight=0.3, size=1024, resolution=0.0375, reduction='mean'):
        super(TGCMetric, self).__init__()
        
        self.loss_weight = loss_weight
        self.size = size
        self.resolution = resolution
        self.reduction = reduction

        # 1. Create the distance vector z (in mm)
        # We add 'resolution' to avoid starting at 0 (geometric spreading at z=0 is undefined)
        z = torch.arange(size).float() * resolution
        z = z + resolution 

        # 2. Absorption Component (Linear-in-dB-in-Time)
        # dB_factor = 10^(dB/20) which is equivalent to e^(dB * ln(10) / 20)
        absorption_gain = torch.exp(tgc_weight * z / 20 * np.log(10))

        # 3. Geometric Spreading Component (Linear-in-Time)
        # Compensation for 1/z decay is simply multiplying by z
        geometric_gain = z # techinically units of mm, but relative scaling is what matters

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

class BeamformedL1Loss(nn.Module):
    """
    Beamformed L1 Loss.
    
    This loss function takes raw sensor data (prediction), beamforms it using 
    differentiable Stolt migration, and computes the L1 loss against a 
    pre-beamformed ground truth image (or beamforms the GT raw data on the fly).
    
    Args:
        loss_weight (float): Weight of this loss term. Default: 1.0.
        reduction (str): Reduction mode: 'none', 'mean', 'sum'. Default: 'mean'.
        beamformer_params (dict): Dictionary of parameters for the StoltBeamformer.
        target_is_raw (bool): If True, assumes 'target' is raw sensor data and needs 
                              beamforming. If False, assumes 'target' is already a 
                              beamformed image. Default: False.
    """

    def __init__(self, 
                 loss_weight=1.0, 
                 reduction='mean', 
                 beamformer_params=None, 
                 target_is_raw=True):
        super(BeamformedL1Loss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.target_is_raw = target_is_raw

        # Default Beamformer Parameters
        default_params = {
            'F': 40e6,
            'pitch': 3.125e-4,
            'c': 1500.0,
            'samplingX': 8,
            'coeffT': 5,
            'zeroX': True,
            'zeroT': True
        }
        
        # Update defaults with provided params
        if beamformer_params is not None:
            default_params.update(beamformer_params)
            
        # Instantiate the Differentiable Beamformer
        self.beamformer = DifferentiableStolt(**default_params)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): Predicted raw sensor data. 
                           Shape: (N, 1, Time, Channels) or (N, Time, Channels).
            target (Tensor): Ground truth. 
                             If target_is_raw=True: Shape matches pred.
                             If target_is_raw=False: Shape is (N, Time, Channels) (Beamformed Image).
            weight (Tensor, optional): Element-wise weights. Default: None.
        """
        
        # 1. Beamform the prediction (Sensor Data -> Image Domain)
        # Note: If input is (N, 1, T, C), beamformer handles the squeeze internally
        pred_img = self.beamformer(pred)

        # 2. Prepare the target
        if self.target_is_raw:
            # If target is also raw data, beamform it to compare in image domain
            with torch.no_grad():
                target_img = self.beamformer(target)
        else:
            # Target is already an image
            target_img = target

        # Ensure shapes match after beamforming
        # pred_img shape will be (Batch, Time, Channels)
        # If target was (Batch, 1, Time, Channels) image, squeeze it
        if target_img.ndim == 4 and target_img.shape[1] == 1:
            target_img = target_img.squeeze(1)

        # 3. Compute L1 Loss
        loss = F.l1_loss(pred_img, target_img, reduction='none')
        
        # Apply weights if provided
        if weight is not None:
            # Ensure weight shape aligns
            if weight.ndim == 4 and weight.shape[1] == 1:
                weight = weight.squeeze(1)
            loss = loss * weight

        # 4. Reduction
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)

        return self.loss_weight * loss

class ConsistencyLoss(nn.Module):
    """
    Consistency Loss that allows gradients to flow back into the Compression Frontend.
    """
    def __init__(self, loss_weight=1.0, compression_frontend=None, reduction='mean'):
        super(ConsistencyLoss, self).__init__()
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')

        self.loss_weight = loss_weight
        self.reduction = reduction
        
        if compression_frontend is None:
            raise ValueError("A compression frontend must be provided for ConsistencyLoss.")
        
        # 1. Store the module reference directly. 
        # PyTorch sees this as a submodule. When you call loss.to(device), 
        # it ensures the frontend is on the device (safe even if done twice).
        self.compression_frontend = compression_frontend

    def forward(self, pred, target, weight=None, **kwargs):
        # 2. Dynamically access the matrix. 
        # This returns the live nn.Parameter (or buffer) attached to the graph.
        # This is CRITICAL for gradients to flow back to A.
        A = self.compression_frontend.get_matrix()
        
        # Apply compression (same logic as your frontend)
        # Assuming pred shape is (N, 1, T, C) and we compress T (dim 2)
        pred_cs = (pred.permute(0,1,3,2) @ A.t()).permute(0,1,3,2)
        target_cs = (target.permute(0,1,3,2) @ A.t()).permute(0,1,3,2)
        
        loss = F.l1_loss(pred_cs, target_cs, reduction='none') 
        
        if weight is not None:
            if weight.ndim == 4 and weight.shape[1] == 1:
                weight = weight.squeeze(1)
            loss = loss * weight
            
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction == 'mean':
            loss = torch.mean(loss)
            
        return self.loss_weight * loss

        
if __name__ == '__main__':
    # Example usage
    tgc_loss = TGCMetric(loss_weight=1.0, tgc_weight=0.5, size=1024, resolution=0.0375, reduction='mean')
    mse = MSELoss()
    pred = torch.tensor(np.load('datasets/Experimental/visualization/back_hand_corrected.npy')).unsqueeze(0).unsqueeze(0).float()
    target = torch.zeros(1, 1, 128, 1024)
    loss, corrected_target = tgc_loss(pred, target)
    print('TGC Metric Loss:', loss.item())
    print('MSE Loss:', mse(pred, target).item())
    import matplotlib.pyplot as plt
    plt.imsave('original_pred.png', pred.squeeze().cpu().numpy(), cmap='gray')
    plt.imsave('corrected_target.png', corrected_target.squeeze().cpu().numpy(), cmap='gray')
    print('Corrected target saved as corrected_target.png and original prediction as original_pred.png')