import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def fista(A, b, alpha, L, n_iter=200):
    n = A.shape[1] # Original channels (128)
    m = b.shape[1] # Time steps (1024)

    x_prev = torch.zeros((n, m)).to(b.device)
    y = x_prev.clone()
    t = torch.tensor(1.0).to(b.device)

    At = A.T
    AtA = At @ A
    Atb = At @ b

    for i in range(n_iter):
        # Gradient of 0.5 * ||Ax - b||^2 is A^T(Ax - b)
        grad = AtA @ y - Atb
        
        # Proximal operator (Soft Thresholding)
        # Assuming sparsity in Identity domain
        step = y - grad / L
        x = torch.sign(step) * torch.clamp(torch.abs(step) - alpha / L, min=0.0)
        
        t_next = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_prev)
        x_prev = x
        t = t_next
        
    return x

# 1. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load 1 Data Point
data_files = sorted(glob("datasets/Experimental/forearm/*.npy"))
raw_data = np.load(data_files[0])[:, 200:] # Shape: (128, 1024)
x_true = torch.from_numpy(raw_data).float().to(device)

# 3. Create 4x Bernoulli Sensing Matrix (128 channels -> 32 measurements)
N = 128
M = 32
# Bernoulli: entries are 1/sqrt(M) or -1/sqrt(M)
A = torch.where(torch.rand(M, N) > 0.5, 1.0, -1.0).to(device)
A = A / np.sqrt(M)

# 4. Perform Compression (Multiplexing along the sensor dimension)
# b = A @ x_true -> (32, 128) @ (128, 1024) = (32, 1024)
b = A @ x_true

# 5. Parameters for Reconstruction
# L should be >= max eigenvalue of A.T @ A
L = torch.linalg.eigvalsh(A.T @ A).max().item()
alpha = 5e-6 # Regularization parameter - tune this!

# 6. Run FISTA
x_recon = fista(A, b, alpha, L, n_iter=5000)
pinv_recon = torch.pinverse(A) @ b

# 7. Visualization
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.title("Original Data (128 Channels)")
plt.imshow(x_true.cpu(), aspect='auto', cmap='magma')
plt.colorbar()

plt.subplot(4, 1, 2)
plt.title(f"Compressed Measurements ({M} Channels)")
plt.imshow(b.cpu(), aspect='auto', cmap='magma')
plt.colorbar()

plt.subplot(4, 1, 3)
plt.title("Reconstructed Data (FISTA)")
plt.imshow(x_recon.cpu(), aspect='auto', cmap='magma')
plt.colorbar()

plt.subplot(4, 1, 4)
plt.title("Reconstructed Data (Pseudo-inverse)")
plt.imshow(pinv_recon.cpu(), aspect='auto', cmap='magma')
plt.colorbar()

plt.tight_layout()
plt.savefig("fista_results/fista_reconstruction.png")

# Calculate Error
mse_fista = torch.mean((x_true - x_recon)**2)
print(f"Reconstruction MSE (FISTA): {mse_fista.item():.6f}")

mse_pinv = torch.mean((x_true - pinv_recon)**2)
print(f"Reconstruction MSE (Pseudo-inverse): {mse_pinv.item():.6f}")
