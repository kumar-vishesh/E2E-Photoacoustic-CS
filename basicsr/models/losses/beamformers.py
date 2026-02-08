import torch
import torch.nn as nn
import numpy as np

class DifferentiableStolt(nn.Module):
    """
    Differentiable Stolt Migration Beamformer.
    
    This module implements a fully differentiable, vectorized Stolt migration 
    algorithm for ultrasound beamforming. It includes lateral upsampling,
    time/lateral padding, FFT-based migration in the k-space, and Hilbert 
    transform envelope detection.

    Key Features:
    - Batch-safe operations using grouped convolutions for lateral smoothing.
    - Fully differentiable end-to-end.
    - Configurable physical parameters (speed of sound, pitch, etc.).
    """

    def __init__(self, 
                 F: float = 40e6, 
                 pitch: float = 3.125e-4, 
                 c: float = 1500.0, 
                 samplingX: int = 8,
                 coeffT: int = 5,
                 zeroX: bool = True,
                 zeroT: bool = True,
                 delay: float = 0.0):
        """
        Args:
            F (float): Sampling frequency (Hz).
            pitch (float): Element pitch (m).
            c (float): Speed of sound (m/s).
            samplingX (int): Lateral upsampling factor.
            coeffT (int): Number of coefficients for Stolt interpolation kernel.
            zeroX (bool): Enable lateral zero padding (prevents wrap-around artifacts).
            zeroT (bool): Enable axial zero padding.
            delay (float): Global delay adjustment (s).
        """
        super(DifferentiableStolt, self).__init__()
        self.F = F
        self.pitch = pitch
        self.c = c
        self.samplingX = samplingX
        self.coeffT = coeffT
        self.zeroX = zeroX
        self.zeroT = zeroT
        self.delay = delay

    def forward(self, sig_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sig_batch (torch.Tensor): Input RF data of shape (Batch, 1, Time, Channels) 
                                      or (Batch, Time, Channels).
        
        Returns:
            torch.Tensor: Beamformed magnitude image of shape (Batch, Time, Channels).
        """
        # 1. Standardize Input: (Batch, 1, T, Nc) -> (Batch, T, Nc)
        if sig_batch.ndim == 4:
            sig_batch = sig_batch.squeeze(1)
            
        device = sig_batch.device
        
        # Local copy of mutable parameters
        pitch_val = self.pitch
        
        # 2. Permute to internal Stolt orientation: (Batch, X, Z)
        # Input (Batch, Time, Channels) -> (Batch, Channels, Time)
        sig_tensor = sig_batch

        B, X, Z = sig_tensor.shape
        T = Z 
        
        # 3. Upsampling (Lateral)
        if self.samplingX > 1:
            new_X = self.samplingX * (X - 1) + 1
            sig2 = torch.zeros((B, new_X, Z), dtype=sig_tensor.dtype, device=device)
            indices = torch.arange(X, device=device) * self.samplingX
            # Broadcasting handles batch dim correctly
            sig2[:, indices, :] = sig_tensor
            sig_tensor = sig2
            pitch_val = pitch_val / self.samplingX
            X = new_X

        # 4. Zero Padding (Time)
        if self.zeroT:
            sig_tensor = torch.cat((sig_tensor, torch.zeros_like(sig_tensor)), dim=2)
            Z *= 2
            T *= 2

        # 5. Zero Padding (Lateral)
        deltaX = 0
        if self.zeroX:
            deltaX = int(np.round(X / 2.0))
            zeros_side = torch.zeros((B, deltaX, T), dtype=sig_tensor.dtype, device=device)
            sig_tensor = torch.cat((zeros_side, sig_tensor, zeros_side), dim=1)
            X = sig_tensor.shape[1]

        Xextent = X * pitch_val
        Zextent = Z * self.c / self.F
        Textent = T * self.c / self.F

        # --- GEOMETRY GENERATION ---
        # Note: We compute this on the fly to support dynamic batch sizes or changing devices.
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

        # --- FFT ---
        # FFT on last two dims (X, Z). Batch dim 0 preserved.
        sigtrans = torch.fft.fftshift(torch.fft.fft2(sig_tensor, dim=(-2, -1)))
        mask = (kt <= 0).float()
        sigtrans = sigtrans * mask 

        # --- STOLT INTERPOLATION ---
        nTup = int(np.ceil((self.coeffT - 1) / 2.0))
        nTdo = int(np.floor((self.coeffT - 1) / 2.0))
        ktrange = torch.arange(-nTdo, nTup + 1, device=device)
        halfT = int(np.ceil((T - 1) / 2.0))

        base = torch.round(kt2 * Textent).long() + halfT
        ktind = base.unsqueeze(-1) + ktrange.unsqueeze(0).unsqueeze(0) 
        ktind = ktind % T

        grid_x = torch.arange(X, device=device).view(X, 1, 1).expand(-1, Z, self.coeffT)

        # Advanced Indexing: (B, X, Z, coeffT)
        V = sigtrans[:, grid_x, ktind]

        kt_1d = kz_axis
        Kt = kt_1d[ktind] 
        deltakt = kt2.unsqueeze(-1) - Kt
        
        nz_mask = (deltakt != 0)
        coeff = torch.ones_like(deltakt, dtype=torch.complex64)
        arg = 2 * np.pi * deltakt[nz_mask] * Textent
        coeff[nz_mask] = (1.0 - torch.exp(-1j * arg)) / (1j * arg)

        ptrans = torch.sum(V * coeff, dim=3) * jakobiante
        ptrans = ptrans * mask 
        
        phase_term = -2j * np.pi * kt2 * self.delay * self.c + 2j * np.pi * kz * self.delay * self.c
        ptrans = ptrans * torch.exp(phase_term)

        # --- IFFT ---
        p = torch.real(torch.fft.ifft2(torch.fft.ifftshift(ptrans, dim=(-2, -1)), dim=(-2, -1)))

        # --- CROP ---
        if self.zeroT:
            Z //= 2
            p = p[..., :Z]
        if self.zeroX:
            X_orig = X - 2 * deltaX
            p = p[:, deltaX:deltaX + X_orig, :]

        # --- LATERAL CONDITIONING (GROUPED CONVOLUTION FIX) ---
        if self.samplingX > 1:
            win = int(np.round(self.samplingX / 2.0))
            if win > 1:
                width = int(2 * win + 1)
                base_kernel = torch.ones((1, 1, width), device=device) / width
                
                # Permute to (Batch, Z, X) and treat Z as Channels
                p_in = p.permute(0, 2, 1) # (Batch, Channels=Z, Length=X)
                
                curr_Z = p_in.shape[1]
                
                # Repeat kernel for every channel (Time slice)
                kernel = base_kernel.repeat(curr_Z, 1, 1)
                
                pad_size = width // 2
                p_padded = torch.nn.functional.pad(p_in, (pad_size, pad_size), mode='replicate')
                
                # Grouped Convolution: groups=Z maintains batch isolation
                p_out = torch.nn.functional.conv1d(p_padded, kernel, groups=curr_Z)
                
                # Permute back to (B, X, Z)
                p = p_out.permute(0, 2, 1)

        # --- HILBERT ---
        rekon = p.permute(0, 2, 1) 
        n_fft = rekon.shape[1]
        
        freqs = torch.fft.fftfreq(n_fft, device=device)
        h_mask = torch.zeros_like(freqs)
        h_mask[0] = 1.0
        h_mask[freqs > 0] = 2.0
        
        h_f = torch.fft.fft(rekon, dim=1)
        h_f = h_f * h_mask.view(1, -1, 1)
        analytic = torch.fft.ifft(h_f, dim=1)
        
        # Returns (Batch, Time, Channels)
        return torch.abs(analytic)