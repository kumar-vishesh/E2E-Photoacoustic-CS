import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import MiniBatchDictionaryLearning
from glob import glob

# 1. Load Data (Load at least 5 frames)
data_files = sorted(glob("datasets/Experimental/forearm/*.npy"))
# Load first 100 frames for training, we will test on 5 of them
train_frames = [np.load(f)[:, 200:] for f in data_files[:100]] 
X_train_raw = np.stack(train_frames, axis=0) # (100, 128, 824)

# Reshape for Dictionary Learning: (Samples, Features) -> (100*824, 128)
# We treat every time-step as a "sample" vector of 128 sensors
X_train = X_train_raw.transpose(0, 2, 1).reshape(-1, 128)

# Normalize (Important for Dictionary Learning convergence)
std_val = np.std(X_train)
X_train /= std_val

print(f"Training Dictionary on {X_train.shape[0]} vectors of size 128...")

# 2. Train the Dictionary (Physics Discovery)
n_atoms = 64 # Keep it tight to force it to learn fundamental shapes
dico = MiniBatchDictionaryLearning(
    n_components=n_atoms,
    alpha=1, 
    max_iter=500,
    batch_size=2048,
    random_state=42,
    verbose=1
)

dico.fit(X_train)
print("Dictionary Learned.")

# 3. Test & Visualize on 5 Full Frames
test_indices = [0, 1, 2, 3, 4] # Test the first 5 frames
fig, axes = plt.subplots(5, 2, figsize=(10, 15))

total_mse = 0

print("\nReconstructing Test Frames...")
for i, idx in enumerate(test_indices):
    # Get one full image (128, 1024)
    original_img = X_train_raw[idx] / std_val
    
    # Reshape to (1024, 128) for the transformer
    img_flat = original_img.T 
    
    # A. Encode (Compress to Sparse Code)
    code = dico.transform(img_flat)
    
    # B. Decode (Reconstruct from Code)
    recon_flat = code @ dico.components_
    
    # Reshape back to image (128, 1024)
    recon_img = recon_flat.T
    
    # C. Calculate Error
    mse = np.mean((original_img - recon_img)**2)
    total_mse += mse
    
    # D. Plot
    # Original
    ax_orig = axes[i, 0]
    im1 = ax_orig.imshow(original_img, aspect='auto', cmap='magma', vmin=-1, vmax=1)
    ax_orig.set_title(f"Frame {idx} Original")
    ax_orig.axis('off')
    
    # Reconstructed
    ax_recon = axes[i, 1]
    im2 = ax_recon.imshow(recon_img, aspect='auto', cmap='magma', vmin=-1, vmax=1)
    ax_recon.set_title(f"Reconstructed (MSE: {mse:.5f})")
    ax_recon.axis('off')

plt.tight_layout()
plt.savefig("dictionary_learning_reconstruction.png")

print(f"Average MSE over 5 frames: {total_mse/5:.6f}")

# Optional: Visualize the Atoms
plt.figure(figsize=(12, 4))
plt.title("First 10 Learned Atoms (The 'Building Blocks')")
plt.imshow(dico.components_[:10], cmap='seismic', aspect='auto')
plt.ylabel("Atom Index")
plt.xlabel("Sensor Index (0-127)")
plt.colorbar()
plt.savefig("learned_atoms.png")