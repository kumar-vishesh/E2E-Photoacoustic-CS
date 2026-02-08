# import torch
# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# from glob import glob

# def get_wavelet_basis(n, wavelet='db4', level=3):
#     """
#     Generates the Synthesis Matrix (Psi) for a 1D Wavelet transform.
#     This matrix maps Sparse Coefficients -> Signal (x = Psi @ s).
#     """
#     # Build coefficient bookkeeping using a zero signal
#     dummy = np.zeros(n)
#     coeffs = pywt.wavedec(dummy, wavelet, level=level)
#     coeffs_array, coeff_slices = pywt.coeffs_to_array(coeffs)

#     total_coeffs = coeffs_array.size
#     psi_list = []

#     # Iterate over each coefficient position in the flattened coefficient array
#     for i in range(total_coeffs):
#         arr = np.zeros_like(coeffs_array)
#         arr.flat[i] = 1.0

#         # Convert flat array back to coeff list expected by waverec
#         s_coeffs = pywt.array_to_coeffs(arr, coeff_slices, output_format='wavedec')

#         # Inverse transform to get basis column (trim/pad to length n if needed)
#         col = pywt.waverec(s_coeffs, wavelet)
#         if len(col) > n:
#             col = col[:n]
#         elif len(col) < n:
#             col = np.pad(col, (0, n - len(col)))

#         psi_list.append(col)

#     Psi = np.stack(psi_list, axis=1)  # Shape (n, total_coeffs)
#     return Psi

# def fista_synthesis(Theta, b, alpha, L, Psi, n_iter=200):
#     """
#     Solves min ||Theta @ s - b||_2 + alpha ||s||_1
#     Then returns x = Psi @ s
#     """
#     m_meas, n_coeffs = Theta.shape # 32 x 128
#     n_timesteps = b.shape[1]       # 1024
    
#     # Initialize s (sparse coefficients)
#     s_prev = torch.zeros((n_coeffs, n_timesteps)).to(b.device)
#     y = s_prev.clone()
#     t = torch.tensor(1.0).to(b.device)
    
#     # Pre-compute Gram matrix for gradient steps
#     Tt = Theta.T
#     TtT = Tt @ Theta
#     Ttb = Tt @ b
    
#     for i in range(n_iter):
#         # Gradient descent on the effective matrix Theta
#         grad = TtT @ y - Ttb
        
#         # Soft Thresholding (Proximal Operator)
#         step = y - grad / L
#         s = torch.sign(step) * torch.clamp(torch.abs(step) - alpha / L, min=0.0)
        
#         # FISTA Momentum
#         t_next = (1 + torch.sqrt(1 + 4 * t**2)) / 2
#         y = s + ((t - 1) / t_next) * (s - s_prev)
#         s_prev = s
#         t = t_next
        
#     # Transform sparse s back to signal x
#     x_recon = Psi @ s
#     return x_recon

# # --- Main Execution ---

# # 1. Setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# N_channels = 128
# M_measurements = 32 # 4x Compression

# # 2. Load Data
# data_files = sorted(glob("datasets/Experimental/forearm/*.npy"))
# raw_data = np.load(data_files[0])[:, 200:] # Shape: (128, 1024)
# x_true = torch.from_numpy(raw_data).float().to(device)

# # 3. Create Wavelet Synthesis Matrix (Psi)
# # We do this on CPU once, then move to GPU
# Psi_np = get_wavelet_basis(N_channels, wavelet='db4', level=3)
# Psi = torch.from_numpy(Psi_np).float().to(device)

# # 4. Create Sensing Matrix (A) and Effective Matrix (Theta)
# # A is Bernoulli (random +1/-1)
# A = torch.where(torch.rand(M_measurements, N_channels) > 0.5, 1.0, -1.0).to(device)
# A = A / np.sqrt(M_measurements)

# # Theta = A @ Psi. This maps s -> y directly.
# Theta = A @ Psi 

# # 5. Compress
# b = A @ x_true # Real measurements happen in spatial domain (A @ x)

# # 6. Reconstruct
# # L must be >= max eigenvalue of Theta.T @ Theta
# L = torch.linalg.eigvalsh(Theta.T @ Theta).max().item()
# alpha = 1e-6 # Sparsity penalty

# x_recon = fista_synthesis(Theta, b, alpha, L, Psi, n_iter=5000)
# x_recon_pinv = torch.pinverse(A) @ b

# # 7. Visualize and Compare
# mse_wavelet = torch.mean((x_true - x_recon)**2).item()
# mse_pinv = torch.mean((x_true - x_recon_pinv)**2).item()

# plt.figure(figsize=(12, 10))

# plt.subplot(4, 1, 1)
# plt.title("Ground Truth")
# plt.imshow(x_true.cpu(), aspect='auto', cmap='magma')
# plt.colorbar()

# plt.subplot(4, 1, 2)
# plt.title(f"Reconstruction (Wavelet Domain, MSE: {mse_wavelet:.6f})")
# plt.imshow(x_recon.cpu(), aspect='auto', cmap='magma')
# plt.colorbar()

# plt.subplot(4, 1, 3)
# plt.title("Error Map (True - Recon)")
# plt.imshow(torch.abs(x_true - x_recon).cpu(), aspect='auto', cmap='inferno')
# plt.colorbar()

# plt.subplot(4, 1, 4)
# plt.title(f"Reconstruction (Pseudo-inverse, MSE: {mse_pinv:.6f})")
# plt.imshow(x_recon_pinv.cpu(), aspect='auto', cmap='magma')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig("fista_results/fista_wavelet_reconstruction.png")

# print(f"Wavelet Domain MSE: {mse_wavelet:.6f}")
# print(f"Pseudo-inverse MSE: {mse_pinv:.6f}")

import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
from glob import glob

# --- Helper: 2D Wavelet Proximal Operator ---
def prox_wavelet_2d(x_tensor, alpha, wavelet='db4', level=3):
    """
    Applies Soft Thresholding in the 2D Wavelet Domain.
    Moves data to CPU for pywt, then back to GPU.
    """
    device = x_tensor.device
    x_np = x_tensor.detach().cpu().numpy()
    
    # 1. Decompose
    coeffs = pywt.wavedec2(x_np, wavelet, level=level)
    
    # 2. Soft Thresholding Loop
    # coeffs[0] is approximation (LL), usually we DON'T threshold this heavily
    # but for sparse signals, we often do. Let's threshold everything.
    coeffs_thresh = []
    
    # Handle Approximation Coeffs (LL)
    coeffs_thresh.append(pywt.threshold(coeffs[0], alpha, mode='soft'))
    
    # Handle Detail Coeffs (tuples of LH, HL, HH)
    for detail_tuple in coeffs[1:]:
        thresh_tuple = tuple(pywt.threshold(d, alpha, mode='soft') for d in detail_tuple)
        coeffs_thresh.append(thresh_tuple)
        
    # 3. Reconstruct
    x_recon = pywt.waverec2(coeffs_thresh, wavelet)
    
    # 4. Crop to original size (padding issues can occur)
    if x_recon.shape != x_np.shape:
        x_recon = x_recon[:x_np.shape[0], :x_np.shape[1]]
        
    return torch.from_numpy(x_recon).float().to(device)

def fista_2d(A, b, alpha, L, n_iter=200):
    """
    FISTA with 2D Wavelet Regularization.
    """
    n = A.shape[1] # 128
    m = b.shape[1] # 1024

    x_prev = torch.zeros((n, m)).to(b.device)
    y = x_prev.clone()
    t = torch.tensor(1.0).to(b.device)

    At = A.T
    AtA = At @ A
    Atb = At @ b

    for i in range(n_iter):
        # 1. Gradient Descent Step (Unchanged - Linear Physics)
        # grad = A^T(Ax - b)
        grad = AtA @ y - Atb
        z = y - grad / L
        
        # 2. Proximal Step (Changed - 2D Wavelet Denoising)
        # Instead of pixel-wise thresholding, we threshold 2D coefficients
        # Scaling alpha by 1/L is standard FISTA
        x = prox_wavelet_2d(z, alpha / L, wavelet='db4', level=3)
        
        # 3. Momentum Step
        t_next = (1 + torch.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_next) * (x - x_prev)
        x_prev = x
        t = t_next
        
        if i % 50 == 0:
             pass # Optional: print loss
             
    return x

# --- Main Execution ---

# 1. Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load Data (Focus on a window with signal)
data_files = sorted(glob("datasets/Experimental/forearm/*.npy"))
raw_data = np.load(data_files[0])[:, 200:] # Load full
x_true = torch.from_numpy(raw_data).float().to(device)

# Normalize for stable FISTA parameters
scale_factor = torch.max(torch.abs(x_true))
x_true = x_true / scale_factor

# 3. Create 4x Bernoulli Sensing Matrix (128 -> 32)
N_sensors = 128
M_meas = 32
A = torch.where(torch.rand(M_meas, N_sensors) > 0.5, 1.0, -1.0).to(device)
A = A / np.sqrt(M_meas)

# 4. Compress
b = A @ x_true

# 5. Parameters
# Lipschitz constant L for AtA
L = torch.linalg.eigvalsh(A.T @ A).max().item()

# Alpha tuning is critical. 
# Too high = erases signal. Too low = noisy.
# Since we normalized x_true to range ~[-1, 1], alpha around 0.01-0.1 is usually good.
alpha = 1e-2

# 6. Run 2D FISTA
print("Running 2D FISTA (Wavelet Domain)...")
x_recon = fista_2d(A, b, alpha, L, n_iter=200)
x_recon_pinv = torch.pinverse(A) @ b

# Un-normalize for display
x_recon = x_recon * scale_factor
x_true = x_true * scale_factor

# 7. Visualization
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.title("Ground Truth (128 Channels)")
plt.imshow(x_true.cpu(), aspect='auto', cmap='magma', vmin=-scale_factor*0.5, vmax=scale_factor*0.5)
plt.colorbar()

plt.subplot(4, 1, 2)
plt.title(f"2D FISTA Reconstruction (Compressed {N_sensors}->{M_meas})")
plt.imshow(x_recon.cpu(), aspect='auto', cmap='magma', vmin=-scale_factor*0.5, vmax=scale_factor*0.5)
plt.colorbar()

plt.subplot(4, 1, 3)
plt.title("Error Map (Difference)")
plt.imshow(np.log((x_true - x_recon).abs().cpu()), aspect='auto', cmap='inferno')
plt.colorbar()

plt.subplot(4, 1, 4)
plt.title("Pseudo-inverse Reconstruction")
plt.imshow(x_recon_pinv.cpu(), aspect='auto', cmap='magma', vmin=-scale_factor*0.5, vmax=scale_factor*0.5)
plt.colorbar()

plt.tight_layout()
plt.savefig("fista_results/fista_wavelet_reconstruction.png")

mse = torch.mean((x_true - x_recon)**2)
mse_pinv = torch.mean((x_true - x_recon_pinv)**2)
print(f"Reconstruction MSE (2D FISTA): {mse.item():.6e}")
print(f"Reconstruction MSE (Pseudo-inverse): {mse_pinv.item():.6e}")