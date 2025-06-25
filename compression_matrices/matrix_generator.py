# Necessary Imports
import numpy as np
from pathlib import Path
import re
import argparse

# --------------------------
# Argument parser
# --------------------------
parser = argparse.ArgumentParser(description="Generate compression matrix and save it with versioning.")

parser.add_argument("--compression_factor", type=int, required=True, help="Compression factor (must divide num_channels)")
parser.add_argument("--num_channels", type=int, required=True, help="Total number of input channels")
parser.add_argument("--matrix_type", type=str, required=True,
                    choices=["blocksum", "blockwise_random", "fully_random", "naive"],
                    help="Type of compression matrix to generate")
parser.add_argument("--save_dir", type=str, default="matrices", help="Directory to save matrices")

args = parser.parse_args()

# --------------------------
# Sanity check
# --------------------------
assert args.num_channels % args.compression_factor == 0, \
    f"Compression factor must divide num_channels exactly! ({args.num_channels} % {args.compression_factor} != 0)"

# --------------------------
# Matrix generation function
# --------------------------
def generate_matrix(matrix_type, compression_factor, num_channels):
    num_blocks = num_channels // compression_factor
    num_rows = num_blocks

    if matrix_type == "blocksum":
        group = np.ones(compression_factor)
        A = np.kron(np.eye(num_blocks), group)

    elif matrix_type == "blockwise_random":
        A = np.zeros((num_rows, num_channels))
        for i in range(num_blocks):
            start_idx = i * compression_factor
            end_idx = start_idx + compression_factor

            block_weights = np.random.choice([-1, 1], size=compression_factor)
            A[i, start_idx:end_idx] = block_weights

    elif matrix_type == "fully_random":
        A = np.zeros((num_rows, num_channels))
        permuted_indices = np.random.permutation(num_channels)

        for i in range(num_rows):
            channels_for_row = permuted_indices[i * compression_factor : (i + 1) * compression_factor]
            A[i, channels_for_row] = np.random.choice([-1, 1], size=compression_factor)

    elif matrix_type == "naive":
        A = np.zeros((num_rows, num_channels))
        for i in range(num_rows):
            channel_idx = i * compression_factor
            A[i, channel_idx] = 1.0

    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

    return A

# --------------------------
# Auto-versioned save logic
# --------------------------
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    rf"A_{args.matrix_type}_{args.compression_factor}x_{args.num_channels}ch_v(\d+)\.npy"
)

existing_versions = [
    int(m.group(1)) for f in save_dir.glob(f"A_{args.matrix_type}_{args.compression_factor}x_{args.num_channels}ch_v*.npy")
    if (m := pattern.match(f.name))
]

next_version = max(existing_versions, default=0) + 1
filename = f"A_{args.matrix_type}_{args.compression_factor}x_{args.num_channels}ch_v{next_version}.npy"
save_path = save_dir / filename

# --------------------------
# Save generated matrix
# --------------------------
A = generate_matrix(args.matrix_type, args.compression_factor, args.num_channels)
np.save(save_path, A)

# --------------------------
# Output info
# --------------------------
print(f"Matrix saved to: {save_path}")
print(f"Shape: {A.shape}")
print(f"Matrix type: {args.matrix_type}")
print(f"Saved version: v{next_version}")
