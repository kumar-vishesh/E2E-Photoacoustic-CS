import numpy as np
import scipy.fftpack as fft
import pywt
from sklearn.decomposition import TruncatedSVD, MiniBatchDictionaryLearning
from glob import glob

def get_metrics(name, coef):
    """Compute Gini and Hoyer sparsity metrics."""
    # Flatten to treat all coefficients as a single distribution
    x = np.abs(coef.flatten())
    # Small epsilon to prevent division by zero in empty arrays
    if np.sum(x) == 0: return 0.0, 0.0
    
    x_sorted = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    
    g = (np.sum((2 * index - n - 1) * x_sorted)) / (n * np.sum(x_sorted))
    h = (np.sqrt(n) - (np.sum(x) / np.sqrt(np.sum(x**2) + 1e-12))) / (np.sqrt(n) - 1)
    return g, h

# 1. Load Data
# Using slightly fewer frames to keep Dictionary Learning fast for this demo
data_files = sorted(glob("datasets/Experimental/forearm/*.npy"))
frames = [np.load(f)[:, 200:] for f in data_files[:100]] 
all_data = np.stack(frames, axis=0) # Shape: (100, 128, 1024)

# Reshape: (128 Channels, All Time Steps)
# This treats the 128-sensor array as the "vector" we want to compress
spatial_matrix = all_data.transpose(1, 0, 2).reshape(128, -1)
print(f"Data Matrix Shape: {spatial_matrix.shape}")

# 2. Prepare Transforms
results = {}

# --- Domain: Identity ---
results["Identity"] = spatial_matrix

# --- Domain: DCT-1D ---
results["DCT-1D"] = fft.dct(spatial_matrix, axis=0, norm='ortho')

# --- Domain: Wavelet-1D ---
coeffs = pywt.wavedec(spatial_matrix, 'db4', axis=0, level=3)
results["Wavelet-1D"] = np.vstack(coeffs)

# --- Domain: SVD (Coefficients) ---
# We calculate the actual coefficients U.T @ X to be comparable to other domains
U, S, Vh = np.linalg.svd(spatial_matrix, full_matrices=False)
# Project data onto the SVD basis
svd_coeffs = U.T @ spatial_matrix
results["SVD (Coefficients)"] = svd_coeffs

# --- Domain: Learned Dictionary ---
# We want to learn atoms of length 128.
# sklearn expects (n_samples, n_features), so we transpose our matrix.
X_train = spatial_matrix.T # Shape: (N_samples, 128)

print("Training Learned Dictionary (this may take a minute)...")
n_atoms = 64
dico = MiniBatchDictionaryLearning(n_components=n_atoms, alpha=1, max_iter=500, random_state=42, verbose=1)

# Fit the dictionary (find the atoms)
dico.fit(X_train)

# Transform (find the sparse codes for our data)
# This solves the Lasso/OMP problem: X ~ code @ dictionary
dl_coeffs = dico.transform(X_train)

# Transpose back to (n_atoms, n_samples) for consistent metric calculation
results["Learned Dict"] = dl_coeffs.T

# 3. Benchmark
print("\n" + "="*60)
print(f"{'Domain':<25} | {'Gini Index (0-1)':<18} | {'Hoyer (0-1)':<12}")
print("-" * 60)

for name, coef in results.items():
    g, h = get_metrics(name, coef)
    print(f"{name:<25} | {g:.4f}             | {h:.4f}")
print("="*60)