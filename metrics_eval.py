import os
import time
from os import path as osp
from copy import deepcopy

import torch
from tqdm import tqdm
import numpy as np

# basicsr imports
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import get_root_logger
from basicsr.utils.options import dict2str
from basicsr.utils.img_util import tensor2img
from basicsr.metrics import calculate_psnr, calculate_ssim

# ============================
# User Config (Edit as needed)
# ============================
CHECKPOINT_PATH = "experiments/Blocksum_8x_L1_025_DirectionalResponse_TGC_0.5/models/best_models/best_model_epoch_108.pth"  # <-- .pth
DATASET_DIR     = "/rhf/vk38/25x_Dataset/Dataset/val"


def build_opt(checkpoint_path: str, dataset_dir: str) -> dict:
    """Minimal opt dict derived from your configuration."""
    return {
        'name': 'eval_psnr_ssim',
        'model_type': 'E2ECompressedSensing',
        'scale': 1,
        'num_gpu': 1,
        'mixed_precision': True,
        'is_train': False,

        'datasets': {
            'test': {
                'name': 'experimental-test',
                'type': 'UncompressedNpyDataset',
                'target_dir': dataset_dir,
                'normalize': True,
                'noise_std': 0.00,
                'reshape_to_image': True,
                'num_worker_per_gpu': 1,
                'batch_size_per_gpu': 1,  # Usually 1 for accurate metric eval per image
                'use_shuffle': False,
                'phase': 'test',
                'save_debug_data': True,
            }
        },

        'network_g': {
            'type': 'NAFNet',
            'img_channel': 1,
            'width': 32,
            'enc_blk_nums': [1, 1, 1, 28],
            'middle_blk_num': 1,
            'dec_blk_nums': [1, 1, 1, 1]
        },

        'compression': {
            'matrix_init': 'blocksum',
            'matrix_learned': 'fixed',
            'compression_factor': 8,
            'input_size': 128,
            'upsampler': 'pinv',
        },

        'path': {
            'root': os.getcwd(),
            'experiments_root': osp.join(os.getcwd(), 'experiments'),
            'results_root': osp.join(os.getcwd(), 'results'),
            'pretrain_network_g': checkpoint_path,
            'strict_load_g': True,
            'resume_state': None,
            'log': osp.join(os.getcwd(), 'logs'),
            'visualization': osp.join(os.getcwd(), 'vis'),
        },

        'logger': {
            'print_freq': 200,
            'save_checkpoint_freq': 1e9,
            'use_tb_logger': False,
            'use_wandb': False,
        },

        'val': {
            'save_img': False,
            'use_image': True,
            'rgb2bgr': True,
            'metrics': {
                'psnr': {'type': 'calculate_psnr', 'crop_border': 0, 'test_y_channel': False},
                'ssim': {'type': 'calculate_ssim', 'crop_border': 0, 'test_y_channel': False},
            },
        },

        'dist': False,
        'rank': 0,
        'world_size': 1,
        'manual_seed': 42,
    }


@torch.no_grad()
def evaluate_dataset(model, data_loader):
    """
    Iterates over the dataset, computes PSNR/SSIM for 'result' vs 'gt',
    and returns the average metrics.
    """
    model.net_g.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    # Progress bar
    pbar = tqdm(data_loader, desc='[Evaluating]', unit='img')

    for data in pbar:
        model.feed_data(data, is_val=True)
        model.test()
        visuals = model.get_current_visuals()

        # visuals['result'] is the prediction
        # visuals['gt'] is the ground truth
        
        # We handle batch size explicitly, though usually batch=1 for testing
        pred_batch = visuals['result']
        gt_batch = visuals['gt']
        
        batch_size = pred_batch.size(0)
        
        for i in range(batch_size):
            # tensor2img converts tensor to uint8 numpy array [0, 255]
            # shape (H, W) for grayscale or (H, W, 3) for RGB
            img_sr = tensor2img(pred_batch[i])
            img_gt = tensor2img(gt_batch[i])

            # Calculate PSNR and SSIM
            # crop_border=0 because usually we evaluate the whole image in this context
            # input_order='HWC' is default for basicsr images
            cur_psnr = calculate_psnr(img_sr, img_gt, crop_border=0, input_order='HWC', test_y_channel=False)
            cur_ssim = calculate_ssim(img_sr, img_gt, crop_border=0, input_order='HWC', test_y_channel=False)

            total_psnr += cur_psnr
            total_ssim += cur_ssim
            count += 1
        
        # Update progress bar description with current average
        if count > 0:
            pbar.set_postfix({'PSNR': f"{total_psnr/count:.2f}", 'SSIM': f"{total_ssim/count:.4f}"})

    avg_psnr = total_psnr / count if count > 0 else 0.0
    avg_ssim = total_ssim / count if count > 0 else 0.0
    
    return avg_psnr, avg_ssim, count


def main():
    # 1. Build options
    opt = build_opt(CHECKPOINT_PATH, DATASET_DIR)

    # 2. Logger setup
    os.makedirs(opt['path']['log'], exist_ok=True)
    logger = get_root_logger('basicsr', log_level='INFO')
    logger.info("Options:\n" + dict2str(opt))

    # 3. Create DataLoaders
    test_opt = deepcopy(opt['datasets']['test'])
    # Force batch size to 1 for precise metric calculation if desired, 
    # though the loop handles larger batches fine.
    test_opt['batch_size_per_gpu'] = 100 
    
    test_set = create_dataset(test_opt)
    test_loader = create_dataloader(
        test_set,
        test_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'],
    )
    logger.info(f"Test dataset size: {len(test_set)}")

    # 4. Create Model
    model = create_model(opt)

    # 5. Run Evaluation
    logger.info("Starting evaluation (PSNR/SSIM only)...")
    psnr, ssim, n_samples = evaluate_dataset(model, test_loader)

    # 6. Print Results
    result_str = (
        f"\n{'='*40}\n"
        f" Evaluation Complete\n"
        f" Samples: {n_samples}\n"
        f" Average PSNR: {psnr:.4f} dB\n"
        f" Average SSIM: {ssim:.4f}\n"
        f"{'='*40}\n"
    )
    
    print(result_str)
    logger.info(result_str)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()