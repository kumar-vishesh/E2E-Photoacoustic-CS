import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class CompressedNpyDataset(Dataset):
    def __init__(self, opt, upsample_type='pinv', normalize=True, noise_std=0.0,
                 compression_factor=None, reshape_to_image=True, target_size=None):
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
        self.upsample_type = upsample_type
        self.reshape_to_image = reshape_to_image
        self.save_debug_data = opt.get('save_debug_data', False)
        self.debug_save_dir = opt.get('debug_save_dir', 'debug_pairs')
        os.makedirs(self.debug_save_dir, exist_ok=True)
        self.debug_counter = 0
        self.debug_max_to_save = opt.get('debug_max_to_save', 100)

        # Load compression matrix A and its pseudoinverse
        A_path = opt.get('A_path', None)
        if A_path is None or not A_path.endswith('.npy') or not os.path.isfile(A_path):
            raise ValueError("A_path must be a valid path to a .npy file containing the compression matrix A.")

        A_np = np.load(A_path)
        self.A = torch.tensor(A_np).float()
        self.A_pinv = torch.linalg.pinv(self.A)

        # Compression factor for bicubic
        if self.upsample_type == 'bicubic':
            if compression_factor is None:
                raise ValueError("compression_factor must be provided when using 'bicubic' upsampling.")
            self.compression_factor = compression_factor
        else:
            self.compression_factor = None

        # Infer target size
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
        return len(self.target_paths)

    def compress_and_reconstruct_pinv(self, noisy_tensor):
        y = self.A @ noisy_tensor
        recon = self.A_pinv @ y
        return recon

    def compress_and_reconstruct_bicubic(self, noisy_tensor):
        # torch-only bicubic would require 4D input for F.interpolate, so using cv2 for now
        C, T = noisy_tensor.shape
        kept = noisy_tensor[::self.compression_factor, :]
        compressed_image = kept.T.cpu().numpy()
        recon_image = cv2.resize(compressed_image, dsize=(C, T), interpolation=cv2.INTER_CUBIC)
        return torch.tensor(recon_image.T).float()

    def normalize_data(self, data_tensor):
        max_val = torch.max(torch.abs(data_tensor))
        return data_tensor / max_val if max_val > 0 else data_tensor

    def __getitem__(self, idx):
        # Load GT as tensor
        path = self.target_paths[idx]
        gt_np = np.load(path).astype(np.float32)
        gt_tensor = torch.tensor(gt_np)

        # Normalize GT to [-1, 1] if needed
        if self.normalize:
            gt_tensor = self.normalize_data(gt_tensor)

        # Add noise to GT (after normalization)
        noisy_tensor = gt_tensor.clone()
        if self.noise_std > 0:
            noise = torch.randn_like(noisy_tensor) * self.noise_std
            noisy_tensor += noise

        # Compress + reconstruct
        if self.upsample_type == 'pinv':
            noisy_tensor_sq = noisy_tensor.squeeze(0) if noisy_tensor.ndim == 3 and noisy_tensor.shape[0] == 1 else noisy_tensor
            if self.A.shape[1] != noisy_tensor_sq.shape[0]:
                raise ValueError(f"A shape {self.A.shape} incompatible with input {noisy_tensor_sq.shape}")
            recon_tensor = self.compress_and_reconstruct_pinv(noisy_tensor_sq)

        elif self.upsample_type == 'bicubic':
            noisy_tensor_sq = noisy_tensor.squeeze(0) if noisy_tensor.ndim == 3 and noisy_tensor.shape[0] == 1 else noisy_tensor
            recon_tensor = self.compress_and_reconstruct_bicubic(noisy_tensor_sq)

        else:
            raise ValueError(f"Unknown upsample_type: {self.upsample_type}")

        # Max-abs normalize recon_tensor
        recon_tensor = self.normalize_data(recon_tensor)

        # Reshape if needed
        if self.reshape_to_image:
            lq = recon_tensor.unsqueeze(0)
            gt = gt_tensor.unsqueeze(0)
        else:
            lq = recon_tensor
            gt = gt_tensor

        # Save debug data if enabled
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
        import torchvision

        def normalize_for_image(tensor):
            tensor = tensor.detach().cpu()
            min_val = tensor.min()
            max_val = tensor.max()
            return (tensor - min_val) / (max_val - min_val + 1e-5)

        # Save PNGs (normalized for visualization)
        lq_img = normalize_for_image(lq)
        gt_img = normalize_for_image(gt)

        torchvision.utils.save_image(lq_img, os.path.join(self.debug_save_dir, f'{basename}_lq.png'))
        torchvision.utils.save_image(gt_img, os.path.join(self.debug_save_dir, f'{basename}_gt.png'))

        # Save .npy files (raw values)
        np.save(os.path.join(self.debug_save_dir, f'{basename}_lq.npy'), lq.detach().cpu().numpy())
        np.save(os.path.join(self.debug_save_dir, f'{basename}_gt.npy'), gt.detach().cpu().numpy())
