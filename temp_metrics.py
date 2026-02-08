# Necassary imports
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

# PSNR function
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(img1)
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

# SSIM function
def ssim(img1, img2):
    ssim_value = ssim_metric(img1, img2, data_range=img1.max() - img1.min())
    return ssim_value

if __name__ == "__main__":
    # Example usage
    gt = np.load("tmp/new_dataset/fix_matrix_fix_up/000002_gt_das.npy")
    pred_baseline = np.load("tmp/new_dataset/fix_matrix_fix_up/000002_pred_das.npy")
    pred_tgc_01 = np.load("tmp/new_dataset/fix_matrix_TGC_0.1/000002_pred_das.npy")
    pred_tgc_03 = np.load("tmp/new_dataset/fix_matrix_TGC_0.3/000002_pred_das.npy")
    pred_tgc_05 = np.load("tmp/new_dataset/fix_matrix_TGC_0.5/000002_pred_das.npy")
    pred_image_space = np.load("tmp/new_dataset/fix_matrix_BeamformedLoss_Bernoulli/000002_pred_das.npy")
    pred_transformer = np.load("tmp/new_dataset/fix_matrix_BeamformedLoss_Bernoulli_Restormer/000002_pred_das.npy")
    
    # Full PSNR calculations
    psnr_baseline = psnr(gt, pred_baseline)
    psnr_tgc_01 = psnr(gt, pred_tgc_01)
    psnr_tgc_03 = psnr(gt, pred_tgc_03)
    psnr_tgc_05 = psnr(gt, pred_tgc_05)
    psnr_image_space = psnr(gt, pred_image_space)
    psnr_transformer = psnr(gt, pred_transformer)

    # SSIM calculations
    ssim_baseline = ssim(gt, pred_baseline)
    ssim_tgc_01 = ssim(gt, pred_tgc_01)
    ssim_tgc_03 = ssim(gt, pred_tgc_03)
    ssim_tgc_05 = ssim(gt, pred_tgc_05)
    ssim_image_space = ssim(gt, pred_image_space)
    ssim_transformer = ssim(gt, pred_transformer)

    # Bottom 1/2 PSNR calculations
    height = gt.shape[0]
    gt_bottom = gt[1*height//2:, :]
    pred_baseline_bottom = pred_baseline[1*height//2:, :]
    pred_tgc_01_bottom = pred_tgc_01[1*height//2:, :]
    pred_tgc_03_bottom = pred_tgc_03[1*height//2:, :]
    pred_tgc_05_bottom = pred_tgc_05[1*height//2:, :]
    pred_image_space_bottom = pred_image_space[1*height//2:, :]
    pred_transformer_bottom = pred_transformer[1*height//2:, :]

    psnr_baseline_bottom = psnr(gt_bottom, pred_baseline_bottom)
    psnr_tgc_01_bottom = psnr(gt_bottom, pred_tgc_01_bottom)
    psnr_tgc_03_bottom = psnr(gt_bottom, pred_tgc_03_bottom)
    psnr_tgc_05_bottom = psnr(gt_bottom, pred_tgc_05_bottom)
    psnr_image_space_bottom = psnr(gt_bottom, pred_image_space_bottom)
    psnr_transformer_bottom = psnr(gt_bottom, pred_transformer_bottom)

    ssim_baseline_bottom = ssim(gt_bottom, pred_baseline_bottom)
    ssim_tgc_01_bottom = ssim(gt_bottom, pred_tgc_01_bottom)
    ssim_tgc_03_bottom = ssim(gt_bottom, pred_tgc_03_bottom)
    ssim_tgc_05_bottom = ssim(gt_bottom, pred_tgc_05_bottom)
    ssim_image_space_bottom = ssim(gt_bottom, pred_image_space_bottom)  
    ssim_transformer_bottom = ssim(gt_bottom, pred_transformer_bottom)

    # Print results
    print(f"PSNR baseline:{psnr_baseline}")
    print(f"PSNR TGC 0.1:{psnr_tgc_01}")
    print(f"PSNR TGC 0.3:{psnr_tgc_03}")
    print(f"PSNR TGC 0.5:{psnr_tgc_05}")
    print(f"PSNR Image Space:{psnr_image_space}")
    print(f"PSNR Transformer:{psnr_transformer}")
    print("\n")
    print(f"SSIM baseline:{ssim_baseline}")
    print(f"SSIM TGC 0.1:{ssim_tgc_01}")
    print(f"SSIM TGC 0.3:{ssim_tgc_03}")
    print(f"SSIM TGC 0.5:{ssim_tgc_05}")
    print(f"SSIM Image Space:{ssim_image_space}")
    print(f"SSIM Transformer:{ssim_transformer}")
    print("\n")
    print(f"PSNR baseline bottom 1/4:{psnr_baseline_bottom}")
    print(f"PSNR TGC 0.1 bottom 1/4:{psnr_tgc_01_bottom}")
    print(f"PSNR TGC 0.3 bottom 1/4:{psnr_tgc_03_bottom}")
    print(f"PSNR TGC 0.5 bottom 1/4:{psnr_tgc_05_bottom}")
    print(f"PSNR Image Space bottom 1/4:{psnr_image_space_bottom}")
    print(f"PSNR Transformer bottom 1/4:{psnr_transformer_bottom}")
    print("\n")
    print(f"SSIM baseline bottom 1/4:{ssim_baseline_bottom}")
    print(f"SSIM TGC 0.1 bottom 1/4:{ssim_tgc_01_bottom}")
    print(f"SSIM TGC 0.3 bottom 1/4:{ssim_tgc_03_bottom}")
    print(f"SSIM TGC 0.5 bottom 1/4:{ssim_tgc_05_bottom}")
    print(f"SSIM Image Space bottom 1/4:{ssim_image_space_bottom}")
    print(f"SSIM Transformer bottom 1/4:{ssim_transformer_bottom}")
    print("Done")