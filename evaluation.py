#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dump input/output pairs (.npy) for your E2ECompressedSensing + NAFNet model.
Optional: also compute metrics using model.epoch_summary (same as your training flow).

Corrections implemented:
- Treat 128×1024 as (elements, time) and transpose before DAS.
- Use raw 2D arrays (no normalization or channel shuffling) for DAS.
- Keep tensor2img for .png convenience saves, but NOT for DAS inputs.
- Clearer rekon_OA_freqdom docstring and expectations.
"""

# ============================
# #### FILL IN PATHS BELOW ####
# ============================
CHECKPOINT_PATH = "experiments/LearnedA_8x_L1_025_learned_upsampler/models/best_models/best_model_epoch_171.pth"  # <-- .pth
DATASET_DIR     = "/home/vk38/E2E-Photoacoustic-CS/datasets/Experimental/visualization"  # <-- folder for the test set
OUTPUT_DIR      = "tmp/learn_matrix_learn_up"  # where .npy files will be written

# If your repo isn’t already on PYTHONPATH, uncomment:
# import sys; sys.path.append("/home/vk38/E2E-Photoacoustic-CS")

# ============================
# No changes needed below
# ============================

# ============================
# #### DAS PARAMS (edit if needed) ####
# ============================
DAS_FREQUENCY_MHZ = 40.0      # F in MHz (as in your example call)
DAS_PITCH_MM      = 0.3125    # element pitch (mm)
DAS_SOUND_MM_US   = 1.5       # speed of sound c (mm/us)
DAS_DELAY_US      = 0.0       # acquisition delay
DAS_ZERO_X        = 1         # zero-pad laterally
DAS_ZERO_T        = 1         # zero-pad in time
DAS_COEFF_T       = 5         # temporal Fourier coeffs
# samplingX will be adapted to current #channels (see _run_das)

import os, json, time, math
from os import path as osp
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import make_exp_dirs, get_root_logger
from basicsr.utils.options import dict2str
from basicsr.utils.img_util import tensor2img

try:
    from scipy.signal import hilbert
except Exception:
    hilbert = None  # safe fallback


def build_opt(checkpoint_path: str, dataset_dir: str) -> dict:
    """Minimal opt dict derived from your YAML (test branch only)."""
    return {
        'name': 'dump_npy',
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
                'upsample_type': 'learned',
                'normalize': True,
                'noise_std': 0.00,
                'reshape_to_image': True,
                'num_worker_per_gpu': 1,
                'batch_size_per_gpu': 100,
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
            'matrix_learned': 'learned',
            'compression_factor': 8,
            'input_size': 128,
            'upsampler': 'learned',
        },

        'path': {
            'root': os.getcwd(),
            'experiments_root': osp.join(os.getcwd(), 'experiments'),
            'results_root': osp.join(os.getcwd(), 'results'),
            'pretrain_network_g': checkpoint_path,  # <-- load weights from .pth
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

         # these mirror your val/test metric settings (used if we call epoch_summary)
        'val': {
            'save_img': False,
            'use_image': True,
            'rgb2bgr': True,
            'metrics': {
                'l1':   {'type': 'calculate_l1',  'crop_border': 0, 'test_y_channel': False},
                'mse':  {'type': 'calculate_mse', 'crop_border': 0, 'test_y_channel': False},
                'ssim': {'type': 'calculate_ssim','crop_border': 0, 'test_y_channel': False},
            },
        },

        'dist': False,
        'rank': 0,
        'world_size': 1,
        'manual_seed': 1234,
    }


def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize numpy array to [0,1]."""
    arr = arr.astype(np.float32)
    minv, maxv = arr.min(), arr.max()
    if maxv > minv:
        arr = (arr - minv) / (maxv - minv)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


# ============================
# DAS (Fourier-domain OA reconstruction)
# ============================
def _centered_axis(n: int, extent: float) -> np.ndarray:
    left = int(np.ceil((n - 1) / 2.0))
    right = int(np.floor((n - 1) / 2.0))
    return np.arange(-left, right + 1, dtype=float) / float(extent)

def _lateral_conditioner(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    width = int(2 * win + 1)
    pad = width // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((width, 1)) / width
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel[:, 0], mode="valid"), 0, padded)
    return out

def rekon_OA_freqdom(sig: np.ndarray, F: float, pitch: float, c: float,
                     delay: float, zeroX: int, zeroT: int, coeffT: int, samplingX: int):
    """
    Parameters
    ----------
    sig : (T, Nc) float32
        RF-like time-series: time (rows) × elements (cols).
        Internally we transpose to (X elements, Z time) for processing.
    """
    # Internally operate on (X, Z)
    sig = sig.T  # to (X elements, Z time)

    X, Z = sig.shape
    T = Z

    Xextent = X * pitch
    Zextent = Z * c / F
    Textent = T * c / F

    if samplingX > 1:
        sig2 = np.zeros((samplingX * (X - 1) + 1, Z), dtype=sig.dtype)
        sig2[np.arange(X) * samplingX, :] = sig
        pitch = pitch / samplingX
        X = samplingX * (X - 1) + 1
        Xextent = X * pitch
        sig = sig2

    if zeroT:
        sig = np.hstack((sig, np.zeros_like(sig)))
        Z *= 2
        Zextent *= 2
        T *= 2
        Textent *= 2

    deltaX = 0
    deltaXextent = 0.0
    if zeroX:
        deltaX = int(np.round(X / 2.0))
        deltaXextent = Xextent / 2.0
        sig = np.vstack((np.zeros((deltaX, T), dtype=sig.dtype), sig, np.zeros((deltaX, T), dtype=sig.dtype)))
        X = X + 2 * deltaX
        Xextent = Xextent + 2 * deltaXextent

    kx_axis = _centered_axis(X, Xextent)
    kz_axis = _centered_axis(Z, Zextent)
    kx, kz = np.meshgrid(kx_axis, kz_axis, indexing='ij')
    kt = kz.copy()
    kt2 = -np.sqrt(kz**2 + kx**2)

    origin = (kz == 0) & (kx == 0)
    kt2_safe = kt2.copy()
    kt2_safe[origin] = 1.0
    jakobiante = kz / kt2_safe
    jakobiante[origin] = 1.0
    kt2[origin] = 0.0

    samplfreq = T / Textent
    kt2 = (kt2 + samplfreq / 2.0) % samplfreq - samplfreq / 2.0

    sigtrans = np.fft.fftshift(np.fft.fft2(sig))
    sigtrans[kt > 0] = 0

    ptrans = np.zeros((X, Z), dtype=complex)
    nTup = int(np.ceil((coeffT - 1) / 2.0))
    nTdo = int(np.floor((coeffT - 1) / 2.0))
    ktrange = np.arange(-nTdo, nTup + 1, dtype=int)
    halfT = int(np.ceil((T - 1) / 2.0))

    for xind in range(X):
        base = np.round(kt2[xind, :] * Textent).astype(int) + halfT
        ktind = base[:, None] + ktrange[None, :]
        ktind %= T

        row_sig = sigtrans[xind, :]
        V = row_sig[ktind]
        row_kt = kt[xind, :]
        Kt = row_kt[ktind]

        deltakt = kt2[xind, :][:, None] - Kt
        coeff = np.ones_like(deltakt, dtype=complex)
        nz = (deltakt != 0)
        if np.any(nz):
            coeff[nz] = (1.0 - np.exp(-2j * np.pi * deltakt[nz] * Textent)) / (2j * np.pi * deltakt[nz] * Textent)

        ptrans[xind, :] = (np.sum(V * coeff, axis=1)) * jakobiante[xind, :]

    ptrans[kt > 0] = 0
    ptrans *= np.exp(-2j * np.pi * kt2 * DAS_DELAY_US * DAS_SOUND_MM_US)
    ptrans *= np.exp( 2j * np.pi * kz  * DAS_DELAY_US * DAS_SOUND_MM_US)

    p = np.real(np.fft.ifft2(np.fft.ifftshift(ptrans)))
    puncut = p.copy()

    if zeroT:
        Z //= 2
        p = p[:, :Z]
    if zeroX:
        X_orig = X - 2 * deltaX
        p = p[deltaX:deltaX + X_orig, :]

    if samplingX > 1:
        p = _lateral_conditioner(p, win=int(np.round(samplingX / 2.0)))
        puncut = _lateral_conditioner(puncut, win=0)

    rekon = p.T  # (Z x X)
    if hilbert is not None:
        rekon = np.abs(hilbert(rekon, axis=0))
    else:
        rekon = np.abs(rekon)
    return rekon  # (Z, X)


def _run_das(arr2d: np.ndarray) -> np.ndarray:
    """
    Unambiguous orientation for your fixed shape case:
    - (128,1024) means (elements, time) -> transpose to (time, elements)
    - (1024,128) already (time, elements)
    """
    arr = np.asarray(arr2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"DAS expects 2D array, got shape {arr.shape}")

    if arr.shape == (128, 1024):
        time_elems = arr.T  # (1024, 128)
    elif arr.shape == (1024, 128):
        time_elems = arr    # already (time, elements)
    else:
        raise ValueError(f"Unexpected shape {arr.shape}; expected (128,1024) or (1024,128).")

    Nc = time_elems.shape[1]  # elements (should be 128)
    samplingX = max(1, int(round(8.0 * 128.0 / float(Nc))))  # -> 8 when Nc=128

    rekon = rekon_OA_freqdom(
        sig=time_elems,
        F=DAS_FREQUENCY_MHZ,
        pitch=DAS_PITCH_MM,
        c=DAS_SOUND_MM_US,
        delay=DAS_DELAY_US,
        zeroX=DAS_ZERO_X,
        zeroT=DAS_ZERO_T,
        coeffT=DAS_COEFF_T,
        samplingX=samplingX
    )
    return rekon  # (Z, X)


@torch.no_grad()
def dump_pairs(model, data_loader, out_dir, rgb2bgr=True, float_range=(-1, 1)):
    """
    feed_data -> test -> get_current_visuals()
    Save per-sample: lq, pred, gt (if present), Ax/x_upsample (if present), A once per batch.
    Each saved as .npy and .png (normalized to [0,1]).
    Additionally save DAS recon for pred and gt: *_pred_das.*, *_gt_das.* (using RAW 2D arrays).
    """
    os.makedirs(out_dir, exist_ok=True)
    model.net_g.eval()

    saved = 0
    for data in tqdm(data_loader, desc='[Dump]', unit='batch'):
        model.feed_data(data, is_val=True)
        model.test()
        visuals = model.get_current_visuals()

        # batch size from any batched tensor
        batch = 0
        for k in ('result', 'lq', 'gt', 'Ax', 'x_upsample'):
            if k in visuals and isinstance(visuals[k], torch.Tensor) and visuals[k].dim() >= 4:
                batch = visuals[k].size(0)
                break
        if batch == 0:
            continue

        for i in range(batch):
            base = f"{saved:06d}"

            # For PNGs and easy viewing (OK to normalize here)
                    # === Helpers: RAW for .npy; normalized only for PNGs ===

            def tensor_to_numpy_raw(key):
                """Return raw float array (C,H,W) or (H,W) with NO normalization."""
                if key not in visuals:
                    return None
                t = visuals[key]
                if not isinstance(t, torch.Tensor):
                    # If a numpy-like object slipped in, coerce to float32
                    return np.asarray(t, dtype=np.float32)
                one = t[i] if t.dim() >= 4 else t  # take ith sample if batched
                # Detach/copy to be extra safe; no squeezing of channel dim
                a = one.detach().cpu().to(torch.float32).contiguous().numpy().copy()
                return a

            def make_png_view(arr_raw, rgb=True):
                """
                Convert RAW array to a PNG-ready array:
                - (H,W) stays (H,W),
                - (C,H,W) -> (H,W,C) for C in {1,3}; if C>3, use first channel,
                - normalize to [0,1] ONLY for PNG display,
                - honor rgb2bgr if rgb=True and 3 channels (for cv2-style viewers).
                """
                a = arr_raw
                if a is None:
                    return None

                # Put channels last if needed
                if a.ndim == 3 and a.shape[0] in (1, 3):
                    a = np.transpose(a, (1, 2, 0))  # (H,W,C)

                # If more than 3 channels, pick the first for visualization
                if a.ndim == 3 and a.shape[2] not in (1, 3):
                    a = a[:, :, :1]

                # Optional BGR flip for 3-channel "rgb" images (matplotlib expects RGB;
                # keep as RGB unless you explicitly want BGR-looking output)
                if rgb and a.ndim == 3 and a.shape[2] == 3 and rgb2bgr:
                    a = a[..., ::-1]  # RGB->BGR

                # Normalize ONLY for PNG
                a01 = _normalize_to_01(a)
                return a01

            def save_pair(arr_raw: np.ndarray, stem: str, rgb=False):
                """Save RAW .npy + normalized .png."""
                # RAW .npy (no scaling)
                np.save(osp.join(out_dir, f"{base}_{stem}.npy"), arr_raw.astype(np.float32))

                # PNG: normalized view
                png_arr = make_png_view(arr_raw, rgb=rgb)

                # Choose colormap: use "hot" only for DAS PNGs and only if single-channel
                use_hot = ("das" in stem)
                if png_arr.ndim == 2 or (png_arr.ndim == 3 and png_arr.shape[2] == 1):
                    cmap = "hot" if use_hot else "gray"
                    plt.imsave(osp.join(out_dir, f"{base}_{stem}.png"),
                            np.squeeze(png_arr), cmap=cmap)
                else:
                    # 3-channel image: no cmap
                    plt.imsave(osp.join(out_dir, f"{base}_{stem}.png"), png_arr)

            # --- core I/O for viewer-friendly saves (RAW .npy + normalized PNG) ---
            lq_raw   = tensor_to_numpy_raw('lq')          # grayscale-like
            pred_raw = tensor_to_numpy_raw('result')
            gt_raw   = tensor_to_numpy_raw('gt')
            Ax_raw   = tensor_to_numpy_raw('Ax')
            xup_raw  = tensor_to_numpy_raw('x_upsample')

            if lq_raw   is not None: save_pair(lq_raw,   'lq',          rgb=False)
            if pred_raw is not None: save_pair(pred_raw, 'pred',        rgb=False)
            if gt_raw   is not None: save_pair(gt_raw,   'gt',          rgb=False)
            if xup_raw  is not None: save_pair(xup_raw,  'x_upsample',  rgb=False)
            if Ax_raw   is not None: save_pair(Ax_raw,   'Ax',          rgb=False)

            # Save A once (first in batch)
            if 'A' in visuals and i == 0:
                A_raw = tensor_to_numpy_raw('A')
                if A_raw is not None:
                    save_pair(A_raw, 'A', rgb=False)

            # === DAS on RAW pred / gt (still use raw 2D arrays) ===
            def to_numpy_raw2d(key):
                t = visuals.get(key, None)
                if not isinstance(t, torch.Tensor):
                    return None
                x = t[i] if t.dim() >= 4 else t
                a = x.detach().cpu().to(torch.float32).squeeze().contiguous().numpy().copy()
                if a.ndim != 2:
                    raise ValueError(f"{key} not 2D after squeeze: {a.shape}")
                return a

            pred_raw2d = to_numpy_raw2d('result')
            gt_raw2d   = to_numpy_raw2d('gt')

            if pred_raw2d is not None:
                try:
                    pred_das = _run_das(pred_raw2d)
                    save_pair(pred_das, 'pred_das', rgb=False)  # PNG uses "hot"
                except Exception:
                    np.save(osp.join(out_dir, f"{base}_pred_shape.npy"),
                            np.array(pred_raw2d.shape, dtype=np.int32))

            if gt_raw2d is not None:
                try:
                    gt_das = _run_das(gt_raw2d)
                    save_pair(gt_das, 'gt_das', rgb=False)
                except Exception:
                    np.save(osp.join(out_dir, f"{base}_gt_shape.npy"),
                            np.array(gt_raw2d.shape, dtype=np.int32))

            saved += 1

    return saved


def main():
    # build opts
    opt = build_opt(CHECKPOINT_PATH, DATASET_DIR)

    # make_exp_dirs(opt)
    os.makedirs(opt['path']['log'], exist_ok=True)
    logger = get_root_logger('basicsr', log_level='INFO')
    logger.info("Options:\n" + dict2str(opt))

    # dataloader
    test_opt = deepcopy(opt['datasets']['test'])
    test_set = create_dataset(test_opt)
    test_loader = create_dataloader(
        test_set,
        test_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'],
    )
    logger.info(f"Test size: {len(test_set)}")

    # model
    model = create_model(opt)

    # dump npy+png (+ DAS for pred/gt)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n = dump_pairs(
        model=model,
        data_loader=test_loader,
        out_dir=OUTPUT_DIR,
        rgb2bgr=True,
        float_range=(-1, 1),
    )

    manifest = {
        "checkpoint": osp.abspath(CHECKPOINT_PATH),
        "dataset_dir": osp.abspath(DATASET_DIR),
        "output_dir": osp.abspath(OUTPUT_DIR),
        "num_items_saved": int(n),
        "das_params": {
            "F_MHz": DAS_FREQUENCY_MHZ,
            "pitch_mm": DAS_PITCH_MM,
            "c_mm_per_us": DAS_SOUND_MM_US,
            "delay_us": DAS_DELAY_US,
            "zeroX": DAS_ZERO_X,
            "zeroT": DAS_ZERO_T,
            "coeffT": DAS_COEFF_T,
            "samplingX_rule": "round(8 * 128 / Nc)",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(osp.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Done. Saved {n} samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()