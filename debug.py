import os
import torch
import numpy as np
import argparse
import torch.nn.functional as F

from basicsr.data import create_dataset, create_dataloader
from basicsr.models import create_model
from basicsr.utils.options import parse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Folder with .npy ground truth files')
    parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--yaml_path', required=True, help='Path to model config (.yml)')
    parser.add_argument('--compression_ratio', type=int, required=True)
    parser.add_argument('--num_channels', type=int, required=True)
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    return parser.parse_args()

def compute_mse(model, dataloader, device='cuda'):
    model.net_g.to(device)
    model.net_g.eval()

    total_mse = 0.0
    total_pixels = 0

    with torch.no_grad():
        for data in dataloader:
            lq = data['lq'].to(device)  # Noisy input
            gt = data['gt'].to(device)  # Clean ground truth

            # Apply model compression to the noisy input
            x_recon, A, Ax = model.net_g.apply_compression(lq)
            output = model.net_g(x_recon.unsqueeze(1))

            mse = F.mse_loss(output, gt, reduction='sum').item()
            total_mse += mse
            total_pixels += gt.numel()

    mean_mse = total_mse / total_pixels
    return mean_mse



def main():
    args = parse_args()

    # Load and patch YAML options
    opt = parse(args.yaml_path, is_train=False)
    opt['dist'] = False
    opt['rank'] = 0

    # Override model parameters
    opt['network_g']['compression_ratio'] = args.compression_ratio
    opt['network_g']['num_input_channels'] = args.num_channels

    # Override pretrain path
    opt['path']['pretrain_network_g'] = args.ckpt_path
    opt['path']['strict_load_g'] = True

    # Override dataset path dynamically
    opt['datasets'] = {

    'val': {
            'name': 'eval_npy',
            'type': 'UncompressedNpyDataset',
            'target_dir': args.data_path,
            'noise_std': 0.01,
            'normalize': True,
            'reshape_to_image': True,
            'upsample_type': 'pinv',
            'io_backend': {'type': 'disk'},
            'batch_size_per_gpu': 1,
            'num_worker_per_gpu': 1,
            'use_shuffle': False,
            'phase': 'val',
            'save_debug_pairs': True
        }
    }

    # Create dataset and dataloader via BasicSR factory
    val_dataset = create_dataset(opt['datasets']['val'])
    val_loader = create_dataloader(
        val_dataset,
        opt['datasets']['val'],
        num_gpu=1,
        dist=False,
        sampler=None
    )

    # Create model using BasicSR
    model = create_model(opt)

    # Evaluate MSE
    mse = compute_mse(model, val_loader, device=args.device)
    print(f"\nMean Pixelwise MSE: {mse:.6e}")


if __name__ == '__main__':
    main()