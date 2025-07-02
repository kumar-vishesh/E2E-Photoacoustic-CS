import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# --- Limit threads for deterministic performance ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# --- BasicSR framework imports ---
from basicsr.models import create_model
from basicsr.utils.options import parse


class CompressedNpyDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset to load .npy files representing 2D tensors.

    Parameters
    ----------
    opt : dict
        Configuration dictionary containing:
        - 'target_dir': path to directory with .npy files
        - 'normalize': whether to normalize input
        - 'reshape_to_image': whether to reshape to (1, C, T)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'lq': low-quality (input)
        - 'gt': ground truth (same as input)
        - 'lq_path': original path
    """
    def __init__(self, opt):
        self.opt = opt
        self.file_list = sorted([
            os.path.join(opt['target_dir'], f) for f in os.listdir(opt['target_dir'])
            if f.endswith('.npy')
        ])
        self.phase = opt.get('phase', 'val')
        self.normalize = opt.get('normalize', True)
        self.reshape_to_image = opt.get('reshape_to_image', True)

    def __len__(self):
        return len(self.file_list)

    def normalize_data(self, tensor):
        max_val = torch.max(torch.abs(tensor))
        return tensor / max_val if max_val > 0 else tensor

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        arr = np.load(file_path).astype(np.float32)
        tensor = torch.from_numpy(arr)  # expected shape: (128, T)

        if self.normalize:
            tensor = self.normalize_data(tensor)

        if self.reshape_to_image:
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 3 and tensor.shape[0] == 1:
                pass
            else:
                raise ValueError(f"Unexpected shape in reshape_to_image: {tensor.shape}")

        return {
            'lq': tensor,
            'gt': tensor,
            'lq_path': [file_path]
        }


def save_matrix_heatmap(A, save_path, cmap='viridis'):
    """
    Save a heatmap of the compression matrix A.

    Parameters
    ----------
    A : ndarray
        Compression matrix of shape (m, n)
    save_path : str
        Output path for PNG
    cmap : str, optional
        Matplotlib colormap name
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(A, aspect='auto', cmap=cmap)
    plt.colorbar(im, shrink=0.8)
    plt.title("Compression Matrix A")
    plt.xlabel("Original Channels")
    plt.ylabel("Compressed Channels")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_gray_tensor(img_tensor, save_path):
    """
    Save a single-channel tensor as a normalized grayscale image.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Input tensor of shape (1, H, W) or (H, W)
    save_path : str
        Output path
    """
    img = img_tensor.squeeze(0)
    min_val = img.min()
    max_val = img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)
    save_image(img.unsqueeze(0), save_path, normalize=False)


def load_model_from_opt(opt_path, ckpt_path):
    """
    Load a model and weights from YAML config and checkpoint.

    Parameters
    ----------
    opt_path : str
        Path to YAML config
    ckpt_path : str
        Path to .pth checkpoint

    Returns
    -------
    model : BasicSR model object
        Fully initialized and loaded model
    opt : dict
        Parsed configuration dictionary
    """
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    opt['path']['pretrain_network_g'] = ckpt_path
    model = create_model(opt)

    # Print compression matrix stats if it exists
    if hasattr(model.net_g, 'cs_matrix'):
        A_weights = model.net_g.cs_matrix.A.detach().cpu().numpy()
        print("Loaded compression matrix A stats:")
        print(f" - Shape: {A_weights.shape}")
        print(f" - Mean:  {A_weights.mean():.4f}")
        print(f" - Std:   {A_weights.std():.4f}")

    model.net_g.eval()
    return model, opt


def run_inference(model, dataloader, save_dir, opt, device='cuda'):
    """
    Run inference on a dataset and save compressed/intermediate outputs.

    Parameters
    ----------
    model : BasicSR model object
        Trained model with .net_g containing NAFNet
    dataloader : DataLoader
        Dataloader wrapping test dataset
    save_dir : str
        Output directory for saving results
    opt : dict
        Parsed YAML configuration
    device : str, optional
        Device for inference
    """
    os.makedirs(save_dir, exist_ok=True)
    model.device = device
    model.net_g = model.net_g.to(device)

    # Save compression matrix A once if available
    if hasattr(model.net_g, 'cs_matrix'):
        model.net_g.cs_matrix = model.net_g.cs_matrix.to(device)
        A_matrix = model.net_g.cs_matrix.A.detach().cpu().numpy()
        np.save(os.path.join(save_dir, "compression_matrix_A.npy"), A_matrix)
        save_matrix_heatmap(A_matrix, os.path.join(save_dir, "compression_matrix_A.png"))

    for i, batch in enumerate(tqdm(dataloader, desc="Running Inference")):
        model.feed_data(batch, is_val=True)

        # Extract and reshape input
        x = batch['lq'].to(device)  # (1, 128, T)
        while x.dim() > 3:
            x = x.squeeze(0)
        B, C, T = x.shape

        # Apply compression and reconstruction
        Ax, A = model.net_g.cs_matrix(x)  # (1, m, T), (m, n)
        A_pinv = torch.linalg.pinv(A)     # (n, m)
        x_recon = torch.matmul(A_pinv.unsqueeze(0), Ax)  # (1, n, T)

        # Reshape for NAFNet input
        x_recon = x_recon.unsqueeze(1)  # (1, 1, 128, T)

        with torch.no_grad():
            output = model.net_g(x_recon)

        # Detach all tensors to CPU for saving
        output = output.detach().cpu()
        x = x.detach().cpu()
        Ax = Ax.squeeze(0).detach().cpu()
        x_recon = x_recon.detach().cpu()
        gt_img = batch['gt'].cpu()

        # Extract filename
        img_path = batch['lq_path']
        while isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # --- Compute and print MSE between x_recon and x ---
        mse_val = torch.mean((x_recon.squeeze(0) - x.squeeze(0)) ** 2).item()
        print(f"[{base_name}] MSE (recon vs GT): {mse_val:.6f}")

        # Save numpy arrays
        np.save(os.path.join(save_dir, f"{base_name}_x.npy"), x.squeeze(0).numpy())
        np.save(os.path.join(save_dir, f"{base_name}_Ax.npy"), Ax.numpy())
        np.save(os.path.join(save_dir, f"{base_name}_xrecon.npy"), x_recon.squeeze(0).numpy())
        np.save(os.path.join(save_dir, f"{base_name}_output.npy"), output.squeeze(0).numpy())

        # Save grayscale PNGs
        save_gray_tensor(x, os.path.join(save_dir, f"{base_name}_x.png"))
        save_gray_tensor(x_recon, os.path.join(save_dir, f"{base_name}_xrecon.png"))
        save_gray_tensor(output, os.path.join(save_dir, f"{base_name}_output.png"))


def main():
    """
    Command-line interface for running inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to test .yml config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint .pth')
    parser.add_argument('--input_dir', type=str, required=True, help='Folder of input .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    print(f"\nLoading model from: {args.ckpt}")
    model, opt = load_model_from_opt(args.opt, args.ckpt)

    print(f"Loading test data from: {args.input_dir}")
    dataset_opt = {
        'target_dir': args.input_dir,
        'phase': 'val'
    }
    test_set = CompressedNpyDataset(dataset_opt)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    print(f"Running inference and saving to: {args.output_dir}")
    run_inference(model, dataloader, args.output_dir, opt, device=args.device)


if __name__ == '__main__':
    main()