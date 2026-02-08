import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import time
import numba as nb
import torch
import glob
import os
import glob

# Helper functions
def _lateral_conditioner(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    width = int(2 * win + 1)
    pad = width // 2
    padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((width, 1)) / width
    out = np.apply_along_axis(lambda v: np.convolve(v, kernel[:, 0], mode="valid"), 0, padded)
    return out

def _centered_axis(n: int, extent: float) -> np.ndarray:
    left = int(np.ceil((n - 1) / 2.0))
    right = int(np.floor((n - 1) / 2.0))
    return np.arange(-left, right + 1, dtype=float) / float(extent)

# Stolt Migration Reconstruction
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

# Numba-accelerated Stolt Interpolation
@nb.njit(parallel=True, fastmath=False) 
def _stolt_interpolation_numba(sigtrans, kt2, kt, jakobiante, Textent, T, coeffT):
    X, Z = sigtrans.shape
    ptrans = np.zeros((X, Z), dtype=np.complex128)
    
    nTup = int(np.ceil((coeffT - 1) / 2.0))
    nTdo = int(np.floor((coeffT - 1) / 2.0))
    halfT = int(np.ceil((T - 1) / 2.0))
    pi = np.pi
    
    for xind in nb.prange(X):
        for zind in range(Z):
            base_val = kt2[xind, zind] * Textent
            base = int(round(base_val)) + halfT
            
            val_sum = 0.0 + 0.0j
            
            for k in range(-nTdo, nTup + 1):
                ktind = (base + k) % T
                
                V = sigtrans[xind, ktind]
                Kt_val = kt[xind, ktind]
                deltakt = kt2[xind, zind] - Kt_val
                
                if deltakt == 0.0:
                    coeff = 1.0 + 0.0j
                else:
                    arg = 2 * pi * deltakt * Textent
                    term = np.exp(-1j * arg)
                    coeff = (1.0 - term) / (1j * arg)
                
                val_sum += V * coeff
            
            ptrans[xind, zind] = val_sum * jakobiante[xind, zind]
            
    return ptrans

# Numba-accelerated Stolt Migration Reconstruction
def rekon_OA_freqdom_numba(sig: np.ndarray, F: float, pitch: float, c: float,
                           delay: float, zeroX: int, zeroT: int, coeffT: int, samplingX: int):
    sig = sig.T
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
    if zeroX:
        deltaX = int(np.round(X / 2.0))
        sig = np.vstack((np.zeros((deltaX, T), dtype=sig.dtype), sig, np.zeros((deltaX, T), dtype=sig.dtype)))
        X = X + 2 * deltaX
        Xextent = Xextent + (Xextent / (X - 2 * deltaX)) * 2 * deltaX

    kx_axis = np.arange(-(int(np.ceil((X-1)/2))), int(np.floor((X-1)/2)) + 1) / Xextent
    kz_axis = np.arange(-(int(np.ceil((Z-1)/2))), int(np.floor((Z-1)/2)) + 1) / Zextent
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

    # --- EXECUTE KERNEL ---
    ptrans = _stolt_interpolation_numba(sigtrans, kt2, kt, jakobiante, Textent, T, coeffT)

    # --- POST PROCESSING ---
    ptrans[kt > 0] = 0
    ptrans *= np.exp(-2j * np.pi * kt2 * delay * c)
    ptrans *= np.exp( 2j * np.pi * kz  * delay * c)

    p = np.real(np.fft.ifft2(np.fft.ifftshift(ptrans)))
    
    if zeroT:
        Z //= 2
        p = p[:, :Z]
    if zeroX:
        X_orig = X - 2 * deltaX
        p = p[deltaX:deltaX + X_orig, :]

    if samplingX > 1:
        p = _lateral_conditioner(p, win=int(np.round(samplingX / 2.0)))

    rekon = p.T
    if hilbert is not None:
        rekon = np.abs(hilbert(rekon, axis=0))
    else:
        rekon = np.abs(rekon)
        
    return rekon

# Vectorized Stolt Migration Reconstruction
def rekon_OA_freqdom_parallel(sig: np.ndarray, F: float, pitch: float, c: float,
                              delay: float, zeroX: int, zeroT: int, coeffT: int, samplingX: int):
    """
    Parallelized (Vectorized) Fourier-domain Reconstruction (Stolt Migration).
    """
    # 1. Setup and Dimensions
    # -----------------------
    sig = sig.T  # Transpose to (X elements, Z time)
    X, Z = sig.shape
    T = Z

    Xextent = X * pitch
    Zextent = Z * c / F
    Textent = T * c / F

    # 2. Upsampling and Padding
    # -------------------------
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

    # 3. K-Space Generation
    # ---------------------
    kx_axis = _centered_axis(X, Xextent)
    kz_axis = _centered_axis(Z, Zextent)
    kx, kz = np.meshgrid(kx_axis, kz_axis, indexing='ij')
    
    # Dispersion Relation: map spatial freqs to temporal freqs
    kt = kz.copy()
    kt2 = -np.sqrt(kz**2 + kx**2)

    # Jacobian Calculation (Derivative Filter)
    origin = (kz == 0) & (kx == 0)
    kt2_safe = kt2.copy()
    kt2_safe[origin] = 1.0
    jakobiante = kz / kt2_safe
    jakobiante[origin] = 1.0
    kt2[origin] = 0.0

    # Wrap frequencies to sampling range
    samplfreq = T / Textent
    kt2 = (kt2 + samplfreq / 2.0) % samplfreq - samplfreq / 2.0

    # FFT to transform (x,t) -> (kx, kt)
    sigtrans = np.fft.fftshift(np.fft.fft2(sig))
    sigtrans[kt > 0] = 0

    # 4. Vectorized Stolt Interpolation
    # ---------------------------------
    # Instead of looping X times, we broadcast operations to (X, Z, coeffT)
    
    nTup = int(np.ceil((coeffT - 1) / 2.0))
    nTdo = int(np.floor((coeffT - 1) / 2.0))
    ktrange = np.arange(-nTdo, nTup + 1, dtype=int)
    halfT = int(np.ceil((T - 1) / 2.0))

    # A. Calculate Base Indices for the whole grid
    # base shape: (X, Z)
    base = np.round(kt2 * Textent).astype(int) + halfT

    # B. Expand to 3D for coefficients
    # ktind shape: (X, Z, coeffT)
    # We add axes to broadcast: (X, Z, 1) + (1, 1, coeffT)
    ktind = base[:, :, None] + ktrange[None, None, :]
    ktind %= T

    # C. Advanced Indexing to gather values
    # We need an index grid for X to pair with ktind
    x_grid = np.arange(X)[:, None, None]  # Shape (X, 1, 1)
    
    # V shape: (X, Z, coeffT) - The spectral values at the neighbor indices
    V = sigtrans[x_grid, ktind]
    
    # Kt shape: (X, Z, coeffT) - The actual k-values at those indices
    Kt = kt[x_grid, ktind]

    # D. Calculate Sinc Interpolation Coefficients
    # deltakt shape: (X, Z, coeffT)
    deltakt = kt2[:, :, None] - Kt
    
    coeff = np.ones_like(deltakt, dtype=complex)
    nz = (deltakt != 0)
    
    # Vectorized computation of the sinc-like kernel
    # (1 - e^(-i theta)) / (i theta)
    term = 2j * np.pi * deltakt[nz] * Textent
    coeff[nz] = (1.0 - np.exp(-term)) / term

    # E. Weighted Summation
    # Collapse the coefficient dimension (axis 2) and apply Jacobian
    ptrans = np.sum(V * coeff, axis=2) * jakobiante

    # 5. Post-Processing
    # ------------------
    # Apply delay corrections (Phase Shift)
    ptrans[kt > 0] = 0
    ptrans *= np.exp(-2j * np.pi * kt2 * delay * c)  # Using 'c' variable for consistency
    ptrans *= np.exp( 2j * np.pi * kz  * delay * c)

    # Inverse FFT to get Image
    p = np.real(np.fft.ifft2(np.fft.ifftshift(ptrans)))
    puncut = p.copy()

    # Un-padding
    if zeroT:
        Z //= 2
        p = p[:, :Z]
    if zeroX:
        X_orig = X - 2 * deltaX
        p = p[deltaX:deltaX + X_orig, :]

    # Lateral Conditioning (Smoothing)
    if samplingX > 1:
        p = _lateral_conditioner(p, win=int(np.round(samplingX / 2.0)))
        puncut = _lateral_conditioner(puncut, win=0)

    # 6. Final Envelope Detection
    # ---------------------------
    rekon = p.T  # Transpose back to (Z, X)
    rekon = np.abs(hilbert(rekon, axis=0))
    
    return rekon

# GPU-accelerated Stolt Migration Reconstruction
def rekon_OA_freqdom_torch(sig: np.ndarray, F: float, pitch: float, c: float,
                           delay: float, zeroX: int, zeroT: int, coeffT: int, samplingX: int,
                           device='cuda'):
    """
    PyTorch (GPU) implementation of Fourier-domain OA reconstruction.
    
    Parameters
    ----------
    sig : (T, Nc) float32
        Input signal.
    device : str
        'cuda' or 'cpu'.
    """
    # Move to GPU immediately
    # Transpose to (X, Z) to match logic
    sig_tensor = torch.tensor(sig.T, dtype=torch.float32, device=device)
    
    X, Z = sig_tensor.shape
    T = Z

    Xextent = X * pitch
    Zextent = Z * c / F
    Textent = T * c / F

    # --- 1. Upsampling & Padding (done on GPU for speed) ---
    if samplingX > 1:
        new_X = samplingX * (X - 1) + 1
        sig2 = torch.zeros((new_X, Z), dtype=sig_tensor.dtype, device=device)
        # Scatter rows
        indices = torch.arange(X, device=device) * samplingX
        sig2[indices, :] = sig_tensor
        
        sig_tensor = sig2
        pitch = pitch / samplingX
        X = new_X
        Xextent = X * pitch

    if zeroT:
        sig_tensor = torch.cat((sig_tensor, torch.zeros_like(sig_tensor)), dim=1)
        Z *= 2
        Zextent *= 2
        T *= 2
        Textent *= 2

    deltaX = 0
    if zeroX:
        deltaX = int(np.round(X / 2.0))
        zeros_side = torch.zeros((deltaX, T), dtype=sig_tensor.dtype, device=device)
        sig_tensor = torch.cat((zeros_side, sig_tensor, zeros_side), dim=0)
        
        # Recalculate dimensions
        X = sig_tensor.shape[0]
        # Xextent updates based on physical pitch * new count
        Xextent = X * pitch

    # --- 2. Grid Generation ---
    # We generate axes using standard torch functions
    kx_idx = torch.arange(-(X-1)//2, (X-1)//2 + 1, device=device) if X % 2 != 0 else \
             torch.arange(-X//2, X//2, device=device)
    # Note: Logic must match the specific "ceil/floor" centering of the original exactly
    # to avoid phase artifacts. Implementing the original _centered_axis logic:
    
    left_x = int(np.ceil((X - 1) / 2.0))
    right_x = int(np.floor((X - 1) / 2.0))
    kx_axis = torch.arange(-left_x, right_x + 1, device=device, dtype=torch.float32) / Xextent

    left_z = int(np.ceil((Z - 1) / 2.0))
    right_z = int(np.floor((Z - 1) / 2.0))
    kz_axis = torch.arange(-left_z, right_z + 1, device=device, dtype=torch.float32) / Zextent

    kx, kz = torch.meshgrid(kx_axis, kz_axis, indexing='ij')

    # Dispersion Relation
    kt = kz.clone()
    kt2 = -torch.sqrt(kz**2 + kx**2)

    # Jacobian
    origin_mask = (kz == 0) & (kx == 0)
    kt2_safe = kt2.clone()
    kt2_safe[origin_mask] = 1.0
    
    jakobiante = kz / kt2_safe
    jakobiante[origin_mask] = 1.0
    kt2[origin_mask] = 0.0

    # Wrapping
    samplfreq = T / Textent
    kt2 = (kt2 + samplfreq / 2.0) % samplfreq - samplfreq / 2.0

    # --- 3. FFT ---
    # torch.fft.fft2 computes unshifted output, so we shift it
    sigtrans = torch.fft.fftshift(torch.fft.fft2(sig_tensor))
    sigtrans[kt > 0] = 0

    # --- 4. Vectorized Interpolation (Stolt) ---
    # This part is memory heavy. On GPU it is fast, but VRAM can be a limit.
    # If OOM occurs, this specific block needs to be chunked.
    
    nTup = int(np.ceil((coeffT - 1) / 2.0))
    nTdo = int(np.floor((coeffT - 1) / 2.0))
    ktrange = torch.arange(-nTdo, nTup + 1, device=device)
    halfT = int(np.ceil((T - 1) / 2.0))

    # Base indices (X, Z)
    base = torch.round(kt2 * Textent).long() + halfT
    
    # Broadcast to (X, Z, coeffT)
    # shape: (X, Z, coeffT)
    ktind = base.unsqueeze(-1) + ktrange.unsqueeze(0).unsqueeze(0)
    ktind = ktind % T

    # Gather Values (Advanced Indexing)
    # We expand sigtrans to (X, Z) -> gather on axis 1 requires flat indexing or gather
    # Simplest torch way: Index using expanded grids
    
    grid_x = torch.arange(X, device=device).view(X, 1, 1).expand(-1, Z, coeffT)
    # V shape: (X, Z, coeffT)
    V = sigtrans[grid_x, ktind]
    
    # Kt values at those indices
    # kt is (X, Z), need to gather from it as well? 
    # Actually kt is constant along X for a specific ktind? No, kt varies by X and Z.
    # We need the kt value at the specific temporal frequency index we just calculated.
    # However, 'kt' array is (X, Z) in the standard grid. 
    # The 'kt' value corresponding to index 'ktind' is simply:
    # row_kt[ktind] where row_kt is the kt axis.
    # But wait, 'kt' variable earlier was 2D meshgrid.
    # Actually, the temporal frequency axis is invariant to X. It is just the Z-axis of the FFT.
    # So we can just look up the 1D axis.
    
    # Extract 1D kt axis from the 2D meshgrid (it's the same for all X)
    kt_1d = kz_axis # In the original code: kt = kz.copy()
    Kt = kt_1d[ktind] # Shape (X, Z, coeffT) via broadcasting if kt_1d is (Z,) -> mapped by ktind values

    # Deltas
    deltakt = kt2.unsqueeze(-1) - Kt
    
    # Sinc Weights
    # (1 - exp(-i 2pi delta Textent)) / (i 2pi delta Textent)
    nz_mask = (deltakt != 0)
    coeff = torch.ones_like(deltakt, dtype=torch.complex64)
    
    arg = 2 * np.pi * deltakt[nz_mask] * Textent
    # We use complex64 to save memory, or complex128 for precision
    denom = 1j * arg
    num = 1.0 - torch.exp(-1j * arg)
    coeff[nz_mask] = num / denom

    # Summation
    ptrans = torch.sum(V * coeff, dim=2) * jakobiante

    # --- 5. Inverse FFT & Cleanup ---
    ptrans[kt > 0] = 0
    # Phase shifts
    phase_term = -2j * np.pi * kt2 * delay * c + 2j * np.pi * kz * delay * c
    ptrans *= torch.exp(phase_term)

    p = torch.real(torch.fft.ifft2(torch.fft.ifftshift(ptrans)))

    # Crop
    if zeroT:
        Z //= 2
        p = p[:, :Z]
    if zeroX:
        X_orig = X - 2 * deltaX
        p = p[deltaX:deltaX + X_orig, :]

    # Lateral Conditioning
    # Implementing the 1D convolution on GPU
    if samplingX > 1:
        win = int(np.round(samplingX / 2.0))
        if win > 1:
            width = int(2 * win + 1)
            kernel = torch.ones((1, 1, width), device=device) / width
            # Conv1d requires (N, C, L). We treat X columns as Channels or Batches?
            # We want to smooth along X (lateral)? No, the original code applies along axis 0 (X).
            # "np.apply_along_axis(..., 0, padded)" -> Smoothing along the columns (vertical in X,Z notation?)
            # Wait, standard OA lateral resolution is along the element direction.
            # In (X, Z) shape: X is elements, Z is time. 
            # The original code smooths along axis 0 (X). 
            
            # Reshape for Conv1d: (Batch=Z, Channel=1, Length=X)
            # We want to filter the X dimension.
            p_in = p.T.unsqueeze(1) # (Z, 1, X)
            
            # Padding: "edge" mode in numpy. Torch has replication padding.
            pad_size = width // 2
            # Pad the last dimension (X)
            p_padded = torch.nn.functional.pad(p_in, (pad_size, pad_size), mode='replicate')
            
            p_out = torch.nn.functional.conv1d(p_padded, kernel)
            p = p_out.squeeze(1).T # Back to (X, Z)

    # Hilbert (Envelope Detection)
    # Scipy's hilbert is not in torch. We do it manually via FFT.
    # H(x) = FFT -> zero negative freqs -> x2 -> IFFT
    rekon = p.T # (Z, X)
    
    # Hilbert along axis 0 (Z, time)
    n_fft = rekon.shape[0]
    freqs = torch.fft.fftfreq(n_fft, device=device)
    h_f = torch.fft.fft(rekon, dim=0)
    
    # Hilbert mask: 1 at 0, 2 at positive, 0 at negative
    h_mask = torch.zeros_like(freqs)
    h_mask[0] = 1.0
    h_mask[freqs > 0] = 2.0
    # Nyquist (if even) is kept as is or 0? Standard analytic signal usually zeros negative.
    # Scipy implementation: 2*pos, 0*neg.
    
    h_f *= h_mask.unsqueeze(1) # Broadcast across X
    analytic = torch.fft.ifft(h_f, dim=0)
    
    rekon_final = torch.abs(analytic)
    
    return rekon_final.cpu().numpy()

# PSNR Calculation
def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(img1)
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

    
if __name__ == "__main__":

    DAS_FREQUENCY_MHZ = 40.0      # F in MHz
    DAS_PITCH_MM      = 0.3125    # element pitch (mm)
    DAS_SOUND_MM_US   = 1.5       # speed of sound c (mm/us)
    DAS_DELAY_US      = 0.0       # acquisition delay
    DAS_ZERO_X        = 1         # zero-pad laterally
    DAS_ZERO_T        = 1         # zero-pad in time
    DAS_COEFF_T       = 5         # temporal Fourier coeffs to use
    DAS_SAMPLING_X   = 8        # lateral upsampling factor

    dataset_paths = glob.glob("datasets/Experimental/visualization/*.npy")
    dataset = {path.split("/")[-1].split(".npy")[0]: np.load(path) for path in dataset_paths}
    print(f"Loaded {len(dataset)} datapoints for testing.")
    # Accumulators
    base_times = []
    parallel_times = []
    numba_times = []
    torch_times = []

    max_errs_parallel = []
    max_errs_numba = []
    max_errs_torch = []

    psnrs_parallel = []
    psnrs_numba = []
    psnrs_torch = []
    psnrs_base = []

    for name, raw in dataset.items():
        datapoint = raw.T
        out_dir = os.path.join("tmp", name)
        os.makedirs(out_dir, exist_ok=True)

        # Base (freq-domain)
        t0 = time.time()
        image_freqdom = rekon_OA_freqdom(datapoint,
                                F=DAS_FREQUENCY_MHZ,
                                pitch=DAS_PITCH_MM,
                                c=DAS_SOUND_MM_US,
                                delay=DAS_DELAY_US,
                                zeroX=DAS_ZERO_X,
                                zeroT=DAS_ZERO_T,
                                coeffT=DAS_COEFF_T,
                                samplingX=DAS_SAMPLING_X)
        t1 = time.time()
        base_times.append(t1 - t0)

        # Parallel
        t0 = time.time()
        image_freqdom_parallel = rekon_OA_freqdom_parallel(datapoint,
                                F=DAS_FREQUENCY_MHZ,
                                pitch=DAS_PITCH_MM,
                                c=DAS_SOUND_MM_US,
                                delay=DAS_DELAY_US,
                                zeroX=DAS_ZERO_X,
                                zeroT=DAS_ZERO_T,
                                coeffT=DAS_COEFF_T,
                                samplingX=DAS_SAMPLING_X)
        t1 = time.time()
        parallel_times.append(t1 - t0)
        abs_error_parallel = np.abs(image_freqdom - image_freqdom_parallel)
        max_errs_parallel.append(float(np.max(abs_error_parallel)))

        # NUMBA
        t0 = time.time()
        images_freqdom_numba = rekon_OA_freqdom_numba(datapoint,
                                F=DAS_FREQUENCY_MHZ,
                                pitch=DAS_PITCH_MM,
                                c=DAS_SOUND_MM_US,
                                delay=DAS_DELAY_US,
                                zeroX=DAS_ZERO_X,
                                zeroT=DAS_ZERO_T,
                                coeffT=DAS_COEFF_T,
                                samplingX=DAS_SAMPLING_X)
        t1 = time.time()
        numba_times.append(t1 - t0)
        abs_error_numba = np.abs(image_freqdom - images_freqdom_numba)
        max_errs_numba.append(float(np.max(abs_error_numba)))

        # Torch
        t0 = time.time()
        image_torch = rekon_OA_freqdom_torch(datapoint,
                                F=DAS_FREQUENCY_MHZ,
                                pitch=DAS_PITCH_MM,
                                c=DAS_SOUND_MM_US,
                                delay=DAS_DELAY_US,
                                zeroX=DAS_ZERO_X,
                                zeroT=DAS_ZERO_T,
                                coeffT=DAS_COEFF_T,
                                samplingX=DAS_SAMPLING_X,
                                device='cuda')
        t1 = time.time()
        torch_times.append(t1 - t0)
        abs_error_torch = np.abs(image_freqdom - image_torch)
        max_errs_torch.append(float(np.max(abs_error_torch)))

        # PSNRs
        psnrs_parallel.append(psnr(image_freqdom, image_freqdom_parallel))
        psnrs_numba.append(psnr(image_freqdom, images_freqdom_numba))
        psnrs_torch.append(psnr(image_freqdom, image_torch))
        psnrs_base.append(psnr(image_freqdom, image_freqdom))

        # Save per-datapoint outputs
        plt.imsave(os.path.join(out_dir, "Raw_signal.png"), datapoint, cmap='gray')
        plt.imsave(os.path.join(out_dir, "FreqDom_reconstruction.png"), image_freqdom, cmap='hot')
        plt.imsave(os.path.join(out_dir, "Parallel_FreqDom_reconstruction.png"), image_freqdom_parallel, cmap='hot')
        plt.imsave(os.path.join(out_dir, "NUMBA_FreqDom_reconstruction.png"), images_freqdom_numba, cmap='hot')
        plt.imsave(os.path.join(out_dir, "Torch_FreqDom_reconstruction.png"), image_torch, cmap='hot')

        plt.imsave(os.path.join(out_dir, "Abs_Error_FreqDom_vs_Parallel.png"), abs_error_parallel, cmap='hot', vmin=0, vmax=np.max(abs_error_parallel))
        plt.imsave(os.path.join(out_dir, "Abs_Error_FreqDom_vs_NUMBA.png"), abs_error_numba, cmap='hot', vmin=0, vmax=np.max(abs_error_numba))
        plt.imsave(os.path.join(out_dir, "Abs_Error_FreqDom_vs_Torch.png"), abs_error_torch, cmap='hot', vmin=0, vmax=np.max(abs_error_torch))

    # Compute mean metrics across dataset
    base_timer = float(np.mean(base_times)) if base_times else 0.0
    parallel_timer = float(np.mean(parallel_times)) if parallel_times else 0.0
    numba_timer = float(np.mean(numba_times)) if numba_times else 0.0
    torch_timer = float(np.mean(torch_times)) if torch_times else 0.0

    speed_parallel = base_timer / parallel_timer if parallel_timer > 0 else float('inf')
    speed_numba = base_timer / numba_timer if numba_timer > 0 else float('inf')
    speed_torch = base_timer / torch_timer if torch_timer > 0 else float('inf')

    max_err_parallel = float(np.mean(max_errs_parallel)) if max_errs_parallel else 0.0
    max_err_numba = float(np.mean(max_errs_numba)) if max_errs_numba else 0.0
    max_err_torch = float(np.mean(max_errs_torch)) if max_errs_torch else 0.0

    psnr_parallel = float(np.mean(psnrs_parallel)) if psnrs_parallel else float('nan')
    psnr_numba = float(np.mean(psnrs_numba)) if psnrs_numba else float('nan')
    psnr_torch = float(np.mean(psnrs_torch)) if psnrs_torch else float('nan')
    psnr_base = float(np.mean(psnrs_base)) if psnrs_base else float('nan')

    col1_w, col2_w, col3_w, col4_w, col5_w = 30, 12, 12, 16, 10
    total_w = col1_w + col2_w + col3_w + col4_w + col5_w
    sep = "-" * total_w

    print()
    print("=" * total_w)
    print(f"{'Method (mean over dataset)':<{col1_w}}{'Time (s)':>{col2_w}}{'Speedup':>{col3_w}}{'Mean Max Err':>{col4_w}}{'PSNR (dB)':>{col5_w}}")
    print(sep)
    print(f"{'Base (freq-domain)':<{col1_w}}{base_timer:>{col2_w}.4f}{'1.00x':>{col3_w}}{0.0:>{col4_w}.4e}{psnr_base:>{col5_w}.2f}")
    print(f"{'Parallel (vectorized)':<{col1_w}}{parallel_timer:>{col2_w}.4f}{speed_parallel:>{col3_w}.2f}x{max_err_parallel:>{col4_w}.4e}{psnr_parallel:>{col5_w}.2f}")
    print(f"{'NUMBA (jit)':<{col1_w}}{numba_timer:>{col2_w}.4f}{speed_numba:>{col3_w}.2f}x{max_err_numba:>{col4_w}.4e}{psnr_numba:>{col5_w}.2f}")
    print(f"{'Torch (GPU)':<{col1_w}}{torch_timer:>{col2_w}.4f}{speed_torch:>{col3_w}.2f}x{max_err_torch:>{col4_w}.4e}{psnr_torch:>{col5_w}.2f}")
    print("=" * total_w)
    print()