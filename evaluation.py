import os
import re
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
import argparse
import time
import yaml

# ------------------------
# Environment Config
# ------------------------
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ------------------------
# Custom Imports
# ------------------------
from basicsr.models.archs.NAFNet_arch import NAFNet

# ------------------------
# Config Path
# ------------------------
CONFIG_PATH = '/home/vk38/E2E-Photoacoustic-CS/config/test/temp.yml'

# ------------------------
# Argument Parsing
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to data root containing data_split/train and val')
    parser.add_argument('--model_root', type=str, required=True,
                        help='Path to experiment dir containing "models" subfolder')
    parser.add_argument('--results_root', type=str, required=True,
                        help='Output directory for saving PNGs')
    parser.add_argument('--compression_ratio', type=int, required=True,
                        help='Compression ratio used during training (e.g. 4, 8, 16)')
    parser.add_argument('--num_channels', type=int, required=True,
                        help='Number of input channels (typically 128)')
    return parser.parse_args()

# ------------------------
# Dataset for Full GT Images
# ------------------------
class GroundTruthDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith('.npy')
        ])

    def __getitem__(self, idx):
        img = np.load(self.img_paths[idx])
        img = torch.from_numpy(img).float().unsqueeze(0)
        return {
            'gt': img,
            'name': os.path.splitext(os.path.basename(self.img_paths[idx]))[0]
        }

    def __len__(self):
        return len(self.img_paths)

# ------------------------
# Utilities
# ------------------------
def get_random_subset(dataset, num_samples=5, seed=42):
    random.seed(seed)
    indices = random.sample(range(len(dataset)), num_samples)

    print(f"\nSelected {dataset.__class__.__name__} samples:")
    for idx in indices:
        sample = dataset[idx]
        print(f" - {sample['name']}")
    return Subset(dataset, indices)

def load_model_from_yaml_and_ckpt(yaml_path, ckpt_path, compression_ratio, num_channels):
    with open(yaml_path, 'r') as f:
        opt = yaml.safe_load(f)

    model_args = dict(opt['network_g'])
    model_args.pop('type', None)

    # Override values from CLI
    model_args['compression_ratio'] = compression_ratio
    model_args['num_input_channels'] = num_channels

    model = NAFNet(**model_args)

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['params'] if 'params' in state_dict else state_dict, strict=True)
    model.eval()
    return model

def run_and_save(model, dataloader, save_dir, tag='val', device='cuda', compression_ratio=1):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {tag} samples")):
        time.sleep(0.2)
        gt_img = batch['gt'].to(device)
        name = batch['name']
        name = name if isinstance(name, str) else name[0]

        with torch.no_grad():
            B, C, H, W = gt_img.shape

            # Flatten image spatial dims → (B, C, T)
            x_unrolled = gt_img

            # Apply compression and reconstruct: x_recon has shape (B, 128, H*W)
            x_recon, A, Ax = model.apply_compression(x_unrolled)

            # Feed reconstructed input into model
            output = model(x_recon.unsqueeze(1))  # Add channel dim: (B, 1, C, T)
        
        # Squeeze output to match gt_img shape
        output = output.squeeze(1)  # (B, C, H, W)
        x_recon = x_recon.squeeze(1)  # (B, C, T)
        gt_img = gt_img.squeeze()  # (B, C, T)
        Ax = Ax.squeeze(1)  # (B, m, T)
        # Save PNGs

        save_image(gt_img, os.path.join(save_dir, f'{name}_gt.png'), normalize=True)
        save_image(x_recon, os.path.join(save_dir, f'{name}_input.png'), normalize=True)
        save_image(output, os.path.join(save_dir, f'{name}_output.png'), normalize=True)
        save_image(A, os.path.join(save_dir, f'{name}_A.png'), normalize=True)
        save_image(Ax, os.path.join(save_dir, f'{name}_Ax.png'), normalize=True)

        # Optional: Save .npy if needed
        np.save(os.path.join(save_dir, f'{name}_gt.npy'), gt_img.cpu().numpy())
        np.save(os.path.join(save_dir, f'{name}_input.npy'), x_recon.cpu().numpy())
        np.save(os.path.join(save_dir, f'{name}_output.npy'), output.cpu().numpy())

def evaluate_checkpoints(ckpt_files, ckpt_dir, dataloaders, result_root, device, compression_ratio, num_channels):
    for ckpt_file in ckpt_files:
        iter_num = re.findall(r'\d+', ckpt_file)[0]
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        print(f"\nLoading checkpoint: {ckpt_path}")

        model = load_model_from_yaml_and_ckpt(CONFIG_PATH, ckpt_path, compression_ratio, num_channels)

        tag = f'iter_{iter_num}'
        for phase, loader in dataloaders.items():
            out_dir = os.path.join(result_root, tag, phase)
            print(f"Saving to: {out_dir}")
            run_and_save(model, loader, out_dir, tag=phase, device=device, compression_ratio=compression_ratio)

# ------------------------
# Main Logic
# ------------------------
def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dir = os.path.join(args.data_root, 'data_split/train')
    val_dir = os.path.join(args.data_root, 'data_split/val')

    train_set = GroundTruthDataset(train_dir)
    val_set = GroundTruthDataset(val_dir)

    train_subset = get_random_subset(train_set, seed=42)
    val_subset = get_random_subset(val_set, seed=42)

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)

    ckpt_dir = os.path.join(args.model_root, 'models')
    ckpt_files = sorted(
        [f for f in os.listdir(ckpt_dir) if re.match(r'net_g_\d+\.pth', f)],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

    evaluate_checkpoints(
        ckpt_files=ckpt_files,
        ckpt_dir=ckpt_dir,
        dataloaders={'train': train_loader, 'val': val_loader},
        result_root=args.results_root,
        device=device,
        compression_ratio=args.compression_ratio,
        num_channels=args.num_channels
    )

if __name__ == '__main__':
    main()