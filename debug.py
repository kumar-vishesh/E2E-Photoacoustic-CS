import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from basicsr.models.modules.blockwise_matrix import BlockLearnableCompressionMatrix
from PIL import Image

# ------------------------ Configuration ------------------------
DATA_ROOT = '/home/vk38/E2E-Photoacoustic-CS/datasets/25x_Averaged/Original/data_split/train'
NUM_EPOCHS = 300
BATCH_SIZE = 8
LR = 1e-4
COMPRESSION_RATIO = 8
NUM_CHANNELS = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NOISE_STD = 1.0
GIF_OUTPUT_PATH = 'tmp/block_sim_hist.gif'

# ------------------------ Dataset Loader ------------------------
class FullPADataset(Dataset):
    def __init__(self, root_dir):
        self.img_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
            if f.endswith('.npy')
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = np.load(self.img_paths[idx])  # shape (C, T)
        x = torch.from_numpy(x).float()
        return x

train_dataset = FullPADataset(DATA_ROOT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# ------------------------ Compression Module ------------------------
A_module = BlockLearnableCompressionMatrix(
    c=COMPRESSION_RATIO,
    n=NUM_CHANNELS,
    noise_std=NOISE_STD
).to(DEVICE)

optimizer = torch.optim.Adam(A_module.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ------------------------ Training Loop ------------------------
all_sims = []
hist_images = []

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    A_module.train()

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        x = batch.to(DEVICE)  # (B, C, T)
        Ax, A = A_module(x)
        A_pinv = torch.linalg.pinv(A)
        x_recon = torch.matmul(A_pinv.unsqueeze(0), Ax)

        loss = loss_fn(x_recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # ------------------ Block Similarity Diagnostic ------------------
    with torch.no_grad():
        blocks = [getattr(A_module, f'block{i+1}').detach().cpu() for i in range(A_module.m)]
        sims = [
            F.cosine_similarity(b1, b2, dim=0).item()
            for i, b1 in enumerate(blocks)
            for j, b2 in enumerate(blocks) if i < j
        ]
        all_sims.append(sims)

        # Plot and save histogram frame
        plt.figure(figsize=(6, 4))
        plt.hist(sims, bins=20, edgecolor='black')
        plt.title(f'Block Cosine Similarity - Epoch {epoch+1}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        frame_path = f'tmp/sim_hist_epoch_{epoch+1:03d}.png'
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Load with PIL instead of imageio
        hist_images.append(Image.open(frame_path).convert('RGB'))

    print(f"[Epoch {epoch+1:03d}] Loss: {avg_loss:.6f} | Block sim (mean/min/max): {np.mean(sims):.4f} / {np.min(sims):.4f} / {np.max(sims):.4f}")

# Convert to RGB and ensure consistent sizing
hist_images_pil = hist_images

# Save animated GIF using PIL
if hist_images_pil:
    hist_images_pil[0].save(
        GIF_OUTPUT_PATH,
        save_all=True,
        append_images=hist_images_pil[1:],
        duration=100,  # ms per frame = 1 fps
        loop=0
    )
    print(f"GIF saved to: {GIF_OUTPUT_PATH}")