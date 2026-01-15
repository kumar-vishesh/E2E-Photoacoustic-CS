import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import time
import numba as nb
# TODO: Add numba acceleration to the heavy loops
# TODO: compare vs torch implementation of vectorized Stolt

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

def rekon_OA_DAS(sig: np.ndarray, 
                 pitch: float, 
                 fs: float, 
                 c: float, 
                 pixel_size: float, 
                 roi_width_mm: float, 
                 roi_depth_mm: float,
                 delay_us: float = 0.0):
    """
    Standard Time-Domain Delay-and-Sum (DAS) for Photoacoustics.
    
    Parameters
    ----------
    sig : (T, Nc) np.ndarray
        Raw RF data: Time steps (rows) x Channels (cols).
    pitch : float (mm)
        Distance between transducer elements.
    fs : float (MHz)
        Sampling frequency.
    c : float (mm/us)
        Speed of sound.
    pixel_size : float (mm)
        Desired resolution of the reconstructed image.
    roi_width_mm : float
        Width of the imaging area.
    roi_depth_mm : float
        Depth of the imaging area.
    """

    
if __name__ == "__main__":

    DAS_FREQUENCY_MHZ = 40.0      # F in MHz
    DAS_PITCH_MM      = 0.3125    # element pitch (mm)
    DAS_SOUND_MM_US   = 1.5       # speed of sound c (mm/us)
    DAS_DELAY_US      = 0.0       # acquisition delay
    DAS_ZERO_X        = 1         # zero-pad laterally
    DAS_ZERO_T        = 1         # zero-pad in time
    DAS_COEFF_T       = 5         # temporal Fourier coeffs to use
    DAS_SAMPLING_X   = 8        # lateral upsampling factor

    datapoint = np.load("datasets/Experimental/visualization/back_hand.npy").T
    timer1 = time.time()
    image_freqdom = rekon_OA_freqdom(datapoint,
                            F=DAS_FREQUENCY_MHZ,
                            pitch=DAS_PITCH_MM,
                            c=DAS_SOUND_MM_US,
                            delay=DAS_DELAY_US,
                            zeroX=DAS_ZERO_X,
                            zeroT=DAS_ZERO_T,
                            coeffT=DAS_COEFF_T,
                            samplingX=DAS_SAMPLING_X)
    timer2 = time.time()
    print(f"Frequency Domain Reconstruction Time: {timer2 - timer1:.4f} seconds")
    
    timer3 = time.time()
    image_freqdom_parallel = rekon_OA_freqdom_parallel(datapoint,
                            F=DAS_FREQUENCY_MHZ,
                            pitch=DAS_PITCH_MM,
                            c=DAS_SOUND_MM_US,
                            delay=DAS_DELAY_US,
                            zeroX=DAS_ZERO_X,
                            zeroT=DAS_ZERO_T,
                            coeffT=DAS_COEFF_T,
                            samplingX=DAS_SAMPLING_X)
    timer4 = time.time()
    print(f"Parallel Frequency Domain Reconstruction Time: {timer4 - timer3:.4f} seconds")
    print(f"Speedup: {(timer2 - timer1)/(timer4 - timer3):.2f}x")
    # # 1. Calculate derived DAS parameters
    # Nc = 128   # Number of channels
    # T = 1024   # Time steps

    # # Lateral resolution based on upsampling factor
    # pixel_size = DAS_PITCH_MM / DAS_SAMPLING_X 

    # # ROI Extent (matching your Fourier padding logic)
    # # Width: Physical aperture + padding
    # roi_width_mm = (Nc * DAS_PITCH_MM)
    # # Depth: Time extent * speed of sound 
    # roi_depth_mm = (T / DAS_FREQUENCY_MHZ) * DAS_SOUND_MM_US

    # image_DAS = rekon_OA_DAS(
    #                         sig=datapoint, 
    #                         pitch=DAS_PITCH_MM, 
    #                         fs=DAS_FREQUENCY_MHZ, 
    #                         c=DAS_SOUND_MM_US, 
    #                         pixel_size=pixel_size, 
    #                         roi_width_mm=roi_width_mm, 
    #                         roi_depth_mm=roi_depth_mm,
    #                         delay_us=DAS_DELAY_US)
    
    
    plt.imsave("tmp/Raw_signal.png", datapoint, cmap='gray')
    plt.imsave("tmp/FreqDom_reconstruction.png", image_freqdom, cmap='hot')
    plt.imsave("tmp/Parallel_FreqDom_reconstruction.png", image_freqdom_parallel, cmap='hot') 
    abs_error = np.abs(image_freqdom - image_freqdom_parallel)
    print("Max Absolute Error between implementations:", np.max(abs_error))
    plt.imsave("tmp/Abs_Error_FreqDom_vs_Parallel.png", abs_error, cmap='hot')
