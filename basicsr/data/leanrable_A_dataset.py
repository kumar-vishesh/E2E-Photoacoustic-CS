import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

class UncompressedNpyDataset(Dataset):
    """
    A PyTorch dataset for loading uncompressed photoacoustic (PA) signals from .npy files.

    Each sample is a 2D tensor of shape (C, T), where:
      - C is the number of transducers (channels)
      - T is the number of time steps

    The dataset supports optional:
      - additive Gaussian noise
      - normalization
      - reshaping to image-like shape (for compatibility with CNNs)
      - saving debug visualizations and raw tensors
    """

    def __init__(self, opt, normalize=True, noise_std=0.0, reshape_to_image=True, target_size=None):
        """
        Initialize the dataset.

        Parameters
        ----------
        opt : dict
            Configuration dictionary with keys such as 'target_dir', 'phase', etc.
        normalize : bool, optional
            Whether to max-abs normalize each sample (default: True).
        noise_std : float, optional
            Standard deviation of Gaussian noise to add to the input signal (default: 0.0).
        reshape_to_image : bool, optional
            Whether to add a channel dimension (1, C, T) to each tensor (default: True).
        target_size : tuple or None
            Optional override for output shape. If None, inferred from data (default: None).
        """
        self.opt = opt
        self.target_dir = opt['target_dir']
        self.target_paths = sorted([
            str(os.path.join(self.target_dir, f))
            for f in os.listdir(self.target_dir)
            if f.endswith('.npy')
        ])

        self.phase = opt['phase']
        self.normalize = normalize
        self.noise_std = float(opt.get('noise_std', noise_std))
        self.reshape_to_image = reshape_to_image

        self.save_debug_data = opt.get('save_debug_data', False)
        self.debug_save_dir = opt.get('debug_save_dir', 'debug_pairs')
        os.makedirs(self.debug_save_dir, exist_ok=True)
        self.debug_counter = 0
        self.debug_max_to_save = opt.get('debug_max_to_save', 100)

        # Infer target size from the first sample if not explicitly set
        if self.reshape_to_image:
            sample = np.load(self.target_paths[0])
            if sample.ndim == 2:
                self.target_size = (1, *sample.shape)
            elif sample.ndim == 3:
                self.target_size = tuple(sample.shape)
            else:
                raise ValueError(f"Unsupported input shape: {sample.shape}")
        else:
            self.target_size = target_size

    def __len__(self):
        """
        Return the total number of samples.

        Returns
        -------
        int
            Number of .npy files in the target directory.
        """
        return len(self.target_paths)

    def normalize_data(self, data_tensor):
        """
        Normalize a tensor to [-1, 1] based on its max absolute value.

        Parameters
        ----------
        data_tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        max_val = torch.max(torch.abs(data_tensor))
        return data_tensor / max_val if max_val > 0 else data_tensor

    def __getitem__(self, idx):
        """
        Load a single (noisy, normalized) PA signal and its corresponding ground truth.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary with:
            - 'lq': noisy input tensor (B, C, T) or (C, T)
            - 'gt': clean ground truth tensor
            - 'lq_path': path to input file
            - 'gt_path': path to GT file (same as input)
        """
        path = self.target_paths[idx]
        gt_np = np.load(path).astype(np.float32)
        gt_tensor = torch.tensor(gt_np)

        if self.normalize:
            gt_tensor = self.normalize_data(gt_tensor)

        noisy_tensor = gt_tensor.clone()
        if self.noise_std > 0:
            noise = torch.randn_like(noisy_tensor) * self.noise_std
            noisy_tensor += noise

        lq = noisy_tensor
        gt = gt_tensor

        if self.reshape_to_image:
            lq = lq.unsqueeze(0)
            gt = gt.unsqueeze(0)

        if self.save_debug_data and self.debug_counter < self.debug_max_to_save:
            basename = os.path.splitext(os.path.basename(path))[0]
            self.save_data(lq, gt, basename)
            self.debug_counter += 1

        return {
            'lq': lq,
            'gt': gt,
            'lq_path': path,
            'gt_path': path
        }

    def save_data(self, lq, gt, basename):
        """
        Save a debug pair (noisy input and ground truth) as PNGs and .npy files.

        Parameters
        ----------
        lq : torch.Tensor
            Noisy input tensor.
        gt : torch.Tensor
            Ground truth tensor.
        basename : str
            Base filename (no extension) to use when saving.
        """
        def normalize_for_image(tensor):
            tensor = tensor.detach().cpu()
            min_val = tensor.min()
            max_val = tensor.max()
            return (tensor - min_val) / (max_val - min_val + 1e-5)

        lq_img = normalize_for_image(lq)
        gt_img = normalize_for_image(gt)

        torchvision.utils.save_image(lq_img, os.path.join(self.debug_save_dir, f'{basename}_lq.png'))
        torchvision.utils.save_image(gt_img, os.path.join(self.debug_save_dir, f'{basename}_gt.png'))

        np.save(os.path.join(self.debug_save_dir, f'{basename}_lq.npy'), lq.detach().cpu().numpy())
        np.save(os.path.join(self.debug_save_dir, f'{basename}_gt.npy'), gt.detach().cpu().numpy())