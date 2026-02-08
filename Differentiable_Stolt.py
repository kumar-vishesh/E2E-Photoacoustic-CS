import torch
import torch.nn as nn
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------
# 1. Functional Implementation (Batched & Differentiable)
# ---------------------------------------------------------
def differentiable_stolt(sig_batch: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Applies Stolt Migration to a batch of sensor data.
    
    Parameters
    ----------
    sig_batch : torch.Tensor
        Input signals. Shape: (Batch, T, Nc) or (Batch, 1, T, Nc).
    params : dict
        Dictionary containing OA parameters ('F', 'pitch', 'c', etc.).
        
    Returns
    -------
    images : torch.Tensor
        Beamformed images. Shape: (Batch, Z, X).
    """
    # 1. Handle Input Shapes (Batch, 1, T, Nc) -> (Batch, T, Nc)
    if sig_batch.ndim == 4:
        sig_batch = sig_batch.squeeze(1)
        
    batch_size = sig_batch.shape[0]
    device = sig_batch.device
    
    # List to collect processed images from the batch
    processed_images = []

    # 2. Iterate over batch
    # (The core Stolt logic is complex to vectorize efficiently across batches 
    # due to specific indexing/grid generation, so looping is safer and standard here)
    for i in range(batch_size):
        single_sig = sig_batch[i]
        
        # Call the core reconstruction logic (defined below)
        img = _core_stolt_logic(single_sig, params, device)
        processed_images.append(img)
    
    # 3. Stack into a single tensor (Preserves gradients)
    return torch.stack(processed_images, dim=0)


def _core_stolt_logic(sig_tensor: torch.Tensor, params: dict, device: str) -> torch.Tensor:
    """
    Internal function: Single-sample Differentiable Stolt Migration.
    Expects sig_tensor shape: (T, Nc)
    """
    # Extract params
    F = params['F']
    pitch = params['pitch']
    c = params['c']
    delay = params['delay']
    zeroX = params['zeroX']
    zeroT = params['zeroT']
    coeffT = params['coeffT']
    samplingX = params['samplingX']

    # Transpose to (X, Z) logic as per original implementation
    sig_tensor = sig_tensor.T 

    X, Z = sig_tensor.shape
    T = Z
    Xextent = X * pitch
    Textent = T * c / F

    # --- Upsampling & Padding ---
    if samplingX > 1:
        new_X = samplingX * (X - 1) + 1
        sig2 = torch.zeros((new_X, Z), dtype=sig_tensor.dtype, device=device)
        indices = torch.arange(X, device=device) * samplingX
        sig2[indices, :] = sig_tensor
        sig_tensor = sig2
        pitch = pitch / samplingX
        X = new_X
        Xextent = X * pitch

    if zeroT:
        sig_tensor = torch.cat((sig_tensor, torch.zeros_like(sig_tensor)), dim=1)
        Z *= 2
        T *= 2
        Textent *= 2

    deltaX = 0
    if zeroX:
        deltaX = int(np.round(X / 2.0))
        zeros_side = torch.zeros((deltaX, T), dtype=sig_tensor.dtype, device=device)
        sig_tensor = torch.cat((zeros_side, sig_tensor, zeros_side), dim=0)
        X = sig_tensor.shape[0]
        Xextent = X * pitch

    # --- Grid Generation (No Grad) ---
    Zextent = Z * c / F
    
    with torch.no_grad():
        left_x = int(np.ceil((X - 1) / 2.0))
        right_x = int(np.floor((X - 1) / 2.0))
        kx_axis = torch.arange(-left_x, right_x + 1, device=device, dtype=torch.float32) / Xextent

        left_z = int(np.ceil((Z - 1) / 2.0))
        right_z = int(np.floor((Z - 1) / 2.0))
        kz_axis = torch.arange(-left_z, right_z + 1, device=device, dtype=torch.float32) / Zextent

        kx, kz = torch.meshgrid(kx_axis, kz_axis, indexing='ij')

        kt = kz.clone()
        kt2 = -torch.sqrt(kz**2 + kx**2)
        
        origin_mask = (kz == 0) & (kx == 0)
        kt2_safe = kt2.clone()
        kt2_safe[origin_mask] = 1.0
        
        jakobiante = kz / kt2_safe
        jakobiante[origin_mask] = 1.0
        kt2[origin_mask] = 0.0

        samplfreq = T / Textent
        kt2 = (kt2 + samplfreq / 2.0) % samplfreq - samplfreq / 2.0

    # --- FFT & Processing ---
    sigtrans = torch.fft.fftshift(torch.fft.fft2(sig_tensor))
    mask = (kt <= 0).float()
    sigtrans = sigtrans * mask 

    nTup = int(np.ceil((coeffT - 1) / 2.0))
    nTdo = int(np.floor((coeffT - 1) / 2.0))
    ktrange = torch.arange(-nTdo, nTup + 1, device=device)
    halfT = int(np.ceil((T - 1) / 2.0))

    base = torch.round(kt2 * Textent).long() + halfT
    ktind = base.unsqueeze(-1) + ktrange.unsqueeze(0).unsqueeze(0)
    ktind = ktind % T

    grid_x = torch.arange(X, device=device).view(X, 1, 1).expand(-1, Z, coeffT)
    V = sigtrans[grid_x, ktind]
    
    kt_1d = kz_axis
    Kt = kt_1d[ktind] 
    deltakt = kt2.unsqueeze(-1) - Kt
    
    nz_mask = (deltakt != 0)
    coeff = torch.ones_like(deltakt, dtype=torch.complex64)
    arg = 2 * np.pi * deltakt[nz_mask] * Textent
    
    # Safe division for sinc
    coeff[nz_mask] = (1.0 - torch.exp(-1j * arg)) / (1j * arg)

    ptrans = torch.sum(V * coeff, dim=2) * jakobiante
    ptrans = ptrans * mask
    
    phase_term = -2j * np.pi * kt2 * delay * c + 2j * np.pi * kz * delay * c
    ptrans = ptrans * torch.exp(phase_term)

    p = torch.real(torch.fft.ifft2(torch.fft.ifftshift(ptrans)))

    # Crop
    if zeroT:
        Z //= 2
        p = p[:, :Z]
    if zeroX:
        X_orig = X - 2 * deltaX
        p = p[deltaX:deltaX + X_orig, :]

    # Lateral Conditioning
    if samplingX > 1:
        win = int(np.round(samplingX / 2.0))
        if win > 1:
            width = int(2 * win + 1)
            kernel = torch.ones((1, 1, width), device=device) / width
            p_in = p.T.unsqueeze(1)
            pad_size = width // 2
            p_padded = torch.nn.functional.pad(p_in, (pad_size, pad_size), mode='replicate')
            p_out = torch.nn.functional.conv1d(p_padded, kernel)
            p = p_out.squeeze(1).T 

    # Hilbert Transform (Envelope Detection)
    rekon = p.T
    n_fft = rekon.shape[0]
    
    with torch.no_grad():
        freqs = torch.fft.fftfreq(n_fft, device=device)
        h_mask = torch.zeros_like(freqs)
        h_mask[0] = 1.0
        h_mask[freqs > 0] = 2.0
    
    h_f = torch.fft.fft(rekon, dim=0)
    h_f = h_f * h_mask.unsqueeze(1)
    analytic = torch.fft.ifft(h_f, dim=0)
    
    return torch.abs(analytic)


# -----------------------------------------------------------------------------
# 2. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Ensure output directory exists
    os.makedirs("output_images_batch", exist_ok=True)

    # Set torch device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- A. Load All Data into a Batch ---
    data_path = "datasets/Experimental/visualization/*.npy"
    file_list = glob.glob(data_path)
    
    if not file_list:
        print(f"No .npy files found in {data_path}")
        exit()
    
    # Sort files to ensure deterministic order
    file_list.sort()
    
    print(f"Found {len(file_list)} files. Loading...")
    
    sig_data_list = []
    file_names = []

    for i, fpath in enumerate(file_list):
        # Load and Transpose to (T, Nc) as per original logic
        arr = np.load(fpath).T 
        sig_data_list.append(arr)
        # Use filename for clearer mapping
        fname = f"datapoint{i}"
        file_names.append(fname)
        print(f"Loaded {fname} with shape {arr.shape}")

    # Stack into (Batch, T, Nc)
    sig_batch_np = np.stack(sig_data_list, axis=0) 
    print(f"Batch Tensor Shape: {sig_batch_np.shape}")

    # --- B. Parameters ---
    params = {
        'F': 40e6,         # Sampling Freq (Hz)
        'pitch': 3.125e-4, # Element Pitch (m)
        'c': 1500.0,       # Speed of Sound (m/s)
        'delay': 0.0,
        'zeroX': 1,
        'zeroT': 1,
        'coeffT': 5,
        'samplingX': 8,
        'device': device
    }

    # --- C. Prepare Targets (Ground Truth Images) ---
    print("Generating Target Images from GT Data...")
    gt_sensor_tensor = torch.tensor(sig_batch_np, dtype=torch.float32, device=device)
    
    # We create the "perfect" targets by beamforming the original data once
    with torch.no_grad():
        target_images = differentiable_stolt(gt_sensor_tensor, params)
    print(f"Target Images Shape: {target_images.shape}")

    # --- D. Initialize Compressed / Degraded Signal ---
    print("Compressing Signals for Initialization...")
    
    # Compress via block sum matrix
    compression_factor = 8
    Nc = sig_batch_np.shape[2]
    
    # Create Matrix (Nc, Nc/compression_factor)
    compression_matrix = np.kron(np.eye(Nc // compression_factor), np.ones((compression_factor, 1)))
    pinv_compression_matrix = np.linalg.pinv(compression_matrix)
    
    # Vectorized Matmul: (B, T, Nc) @ (Nc, Nc_comp) -> (B, T, Nc_comp)
    compressed_sig_batch = sig_batch_np @ compression_matrix 
    
    # Re-expand: (B, T, Nc_comp) @ (Nc_comp, Nc) -> (B, T, Nc)
    init_batch_np = compressed_sig_batch @ pinv_compression_matrix 
    
    # Create the Optimization Variable
    init_tensor = torch.tensor(init_batch_np, dtype=torch.float32, device=device)
    train_tensor = init_tensor.clone().detach().requires_grad_(True)

    # Save initialized (degraded) beamformed images for reference
    with torch.no_grad():
        init_beamformed = differentiable_stolt(init_tensor, params)

    # Ensure base output directory exists
    base_out = "output_images_batch"
    
    # For each datapoint, create a subfolder and save
    print("Saving Initial Visualizations...")
    for i, fname in enumerate(file_names):
        folder = os.path.join(base_out, fname)
        os.makedirs(folder, exist_ok=True)

        # Original sensor signal
        orig_sig = gt_sensor_tensor[i].cpu().numpy()
        plt.imsave(os.path.join(folder, "original_signal.png"), orig_sig, cmap='bwr', vmin=np.min(orig_sig), vmax=np.max(orig_sig))

        # Original beamformed image (target)
        gt_img = target_images[i].cpu().numpy()
        plt.imsave(os.path.join(folder, "original_beamformed.png"), gt_img, cmap='hot', vmin=np.min(gt_img), vmax=np.max(gt_img))

        # Compressed sensor signal
        comp_sig = init_tensor[i].cpu().numpy()
        plt.imsave(os.path.join(folder, "compressed_signal.png"), comp_sig, cmap='bwr', vmin=np.min(comp_sig), vmax=np.max(comp_sig))

        # Beamformed image from re-expanded compressed (init)
        init_img = init_beamformed[i].cpu().numpy()
        plt.imsave(os.path.join(folder, "compressed_beamformed.png"), init_img, cmap='hot', vmin=np.min(init_img), vmax=np.max(init_img))
        
    print("Initialization complete. Ready for optimization.")

    # --- E. Optimization Loop ---
    optimizer = torch.optim.Adam([train_tensor], lr=1e0)
    loss_fn = torch.nn.L1Loss()
    
    num_iters = 1000
    pbar = tqdm(range(num_iters), desc="Optimizing Batch")
    
    for it in pbar:
        optimizer.zero_grad()
        
        # 1. Batched Forward Pass (Stolt)
        output_opt_batch = differentiable_stolt(train_tensor, params)

        # 2. Batched Loss
        loss_val = loss_fn(output_opt_batch, target_images)
        
        # 3. Backward
        loss_val.backward()
        optimizer.step()
        
        if (it + 1) % 10 == 0:
            pbar.set_postfix(loss=f"{loss_val.item():.6e}")

    # --- F. Save Results ---
    print("Optimization Complete. Saving results...")
    
    optimized_sensor_data = train_tensor.detach().cpu().numpy()
    final_beamformed_batch = differentiable_stolt(train_tensor, params).detach().cpu().numpy()

    # Iterate through batch to save images into each datapoint subfolder
    for i, fname in enumerate(file_names):
        folder = os.path.join(base_out, fname)
        os.makedirs(folder, exist_ok=True)

        # Optimized Sensor Data (Input domain)
        opt_sig = optimized_sensor_data[i]
        plt.imsave(os.path.join(folder, "recovered_signal.png"), opt_sig, cmap='bwr', vmin=np.min(opt_sig), vmax=np.max(opt_sig))
        
        # Optimized Image (Output domain)
        opt_img = final_beamformed_batch[i]
        plt.imsave(os.path.join(folder, "recovered_beamformed.png"), opt_img, cmap='hot', vmin=np.min(opt_img), vmax=np.max(opt_img))

    # Global MSE
    global_mse = np.mean((optimized_sensor_data - sig_batch_np)**2)
    print(f"Global Batch Sensor Space MSE: {global_mse:.6e}")