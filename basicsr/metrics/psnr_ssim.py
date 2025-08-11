# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# modified from https://github.com/mayorx/matlab_ssim_pytorch_implementation/blob/main/calc_ssim.py
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
# Modified by: VK (2025)
# ------------------------------------------------------------------------
import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from skimage.metrics import structural_similarity
import torch

def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        img2 (ndarray/tensor): Images with range [0, 255]/[0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    def _psnr(img1, img2):
        if test_y_channel:
            img1 = to_y_channel(img1)
            img2 = to_y_channel(img2)

        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        max_value = 1. if img1.max() <= 1 else 255.

        return 20. * np.log10(max_value / np.sqrt(mse))

    if img1.ndim == 3 and img1.shape[2] == 6:
        l1, r1 = img1[:,:,:3], img1[:,:,3:]
        l2, r2 = img2[:,:,:3], img2[:,:,3:]
        return (_psnr(l1, l2) + _psnr(r1, r2))/2
    else:
        return _psnr(img1, img2)

def calculate_psnr_left(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    assert input_order == 'HWC'
    assert crop_border == 0

    img1 = img1[:,64:,:3]
    img2 = img2[:,64:,:3]
    return calculate_psnr(img1=img1, img2=img2, crop_border=0, input_order=input_order, test_y_channel=test_y_channel)

def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """ 
    Calculate ssim for two objects with float support. This is actually done by scikit in the back end already based on data type.
    For all float data this is assumed to be (-1,1)
    Args:
        img1 (ndarray/tensor): Image with float support
        img2 (ndarray/tensor): Image with float support.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """
    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Invalid input_order: {input_order}')
    # Convert tensors to numpy
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Reduce to single channel
    if img1.ndim == 3:
       img1 = img1.squeeze(2)
    if img2.ndim == 3:
       img2 = img2.squeeze(2)
    
    # Cast to high precision float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    return structural_similarity(img1, img2, data_range = 2)



def calculate_mse(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):

    """Calculate MSE (Mean Squared Error)."""

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Invalid input_order: {input_order}')

    # Convert tensors to numpy
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    # Reorder to HWC RGB
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # Crop borders
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Y channel if needed
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return float(np.mean((img1 - img2) ** 2))

def calculate_l1(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate L1 loss (Mean Absolute Error)."""

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Invalid input_order: {input_order}')

    # Convert tensors to numpy
    if isinstance(img1, torch.Tensor):
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    # Reorder to HWC RGB
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # Crop borders
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Y channel if needed
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    return float(np.mean(np.abs(img1 - img2)))