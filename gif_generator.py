# gif_generator.py

"""
Generates two GIFs:
1. Compression matrix heatmap per checkpoint (net_g_*.pth).
2. Validation MSE curve over training iterations (log scale).

Author: vk38
Modified: 2025-07-02
"""

import os
import glob
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re


def extract_A_from_checkpoint(pth_file):
    """
    Load a .pth file and extract the compression matrix A.

    Parameters
    ----------
    pth_file : str
        Path to .pth checkpoint file.

    Returns
    -------
    numpy.ndarray
        Compression matrix A of shape (m, n)
    """
    ckpt = torch.load(pth_file, map_location='cpu')
    if 'params' in ckpt and 'cs_matrix.A' in ckpt['params']:
        A_tensor = ckpt['params']['cs_matrix.A']
    elif 'model' in ckpt and 'cs_matrix.A' in ckpt['model']:
        A_tensor = ckpt['model']['cs_matrix.A']
    else:
        raise KeyError(f"No compression matrix 'cs_matrix.A' found in {pth_file}")
    return A_tensor.cpu().numpy()


def extract_checkpoint_number(filename):
    """
    Extract numeric suffix from a filename like net_g_123.pth

    Parameters
    ----------
    filename : str
        Filename containing iteration number

    Returns
    -------
    int
        Extracted iteration number or -1 if not found
    """
    match = re.search(r'net_g_(\d+)\.pth$', filename)
    return int(match.group(1)) if match else -1


def plot_heatmap(A, title=None):
    """
    Generate a heatmap image from matrix A and return as a PIL Image.

    Parameters
    ----------
    A : ndarray
        Compression matrix
    title : str, optional
        Title to display on the plot

    Returns
    -------
    PIL.Image.Image
        Rendered image of the heatmap
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title)
    plt.tight_layout()

    # Save plot to memory as PIL Image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(image)


def parse_log_for_mse(log_path):
    """
    Parse the log file to extract validation MSE per checkpoint iteration.

    Parameters
    ----------
    log_path : str
        Path to the .log file

    Returns
    -------
    list of tuples
        Each tuple contains (iteration_number, mse_value)
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()

    pattern = re.compile(r"iter:\s*([\d,]+).*?m_mse:\s*([\d\.e\-\+]+)")
    mse_data = []

    for line in lines:
        match = pattern.search(line)
        if match:
            iter_num = int(match.group(1).replace(',', ''))  # remove commas
            mse_val = float(match.group(2))
            mse_data.append((iter_num, mse_val))

    return sorted(mse_data, key=lambda x: x[0])


def plot_mse_curve(iters, mses, current_index, title=None):
    """
    Plot MSE curve up to a certain point and return as PIL Image.

    Parameters
    ----------
    iters : list of int
        Iteration numbers
    mses : list of float
        MSE values
    current_index : int
        Index up to which to plot
    title : str, optional
        Title for the frame

    Returns
    -------
    PIL.Image.Image
        Line plot image
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(iters[:current_index+1], mses[:current_index+1], marker='o')
    ax.set_yscale("log")  # Set y-axis to log scale
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE (log scale)")  # Updated label
    if title:
        ax.set_title(title)
    plt.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return Image.fromarray(image)


def make_dual_gifs(model_dir, matrix_gif_path, mse_gif_path, fps=50):
    """
    Create two GIFs:
    1. Compression matrix heatmaps.
    2. MSE curve evolution.

    Parameters
    ----------
    model_dir : str
        Directory containing net_g_*.pth and a .log file
    matrix_gif_path : str
        Path to save the matrix evolution GIF
    mse_gif_path : str
        Path to save the MSE evolution GIF
    fps : int
        Frames per second for both GIFs
    """
    # Find and sort all valid checkpoint files
    all_pth_files = glob.glob(os.path.join(model_dir, "net_g_*.pth"))
    pth_files = sorted(
        [f for f in all_pth_files if extract_checkpoint_number(os.path.basename(f)) != -1],
        key=lambda f: extract_checkpoint_number(os.path.basename(f))
    )

    if not pth_files:
        print("No valid net_g_*.pth files found.")
        return

    print(f"Found {len(pth_files)} valid checkpoint files.")

    # Locate corresponding .log file
    experiment_root = os.path.dirname(model_dir)
    log_files = glob.glob(os.path.join(experiment_root, "*.log"))
    if not log_files:
        print("No log file found for MSE parsing.")
        return
    log_path = log_files[0]

    # Parse log file for MSE values
    mse_data = parse_log_for_mse(log_path)
    iter2mse = dict(mse_data)

    print(f"Parsed {len(iter2mse)} MSE entries from log file.")
    print(f"First 5 MSE entries: {list(iter2mse.items())[:5]}")
    
    matrix_images = []
    mse_images = []
    iter_list = []
    mse_list = []

    for pth_file in pth_files:
        iter_num = extract_checkpoint_number(os.path.basename(pth_file))

        if iter_num not in iter2mse:
            print(f"Skipping {pth_file}: no matching MSE entry in log.")
            continue

        try:
            # --- Generate matrix heatmap frame ---
            A = extract_A_from_checkpoint(pth_file)
            matrix_img = plot_heatmap(A, title=f"Iter {iter_num}")
            matrix_images.append(matrix_img)

            # --- Generate MSE curve frame ---
            iter_list.append(iter_num)
            mse_list.append(iter2mse[iter_num])
            mse_img = plot_mse_curve(iter_list, mse_list, len(iter_list)-1, title=f"MSE up to Iter {iter_num}")
            mse_images.append(mse_img)

        except Exception as e:
            print(f"Skipping {pth_file}: {e}")

    duration = 1000 / fps  # ms per frame

    if matrix_images:
        matrix_images[0].save(
            matrix_gif_path,
            save_all=True,
            append_images=matrix_images[1:],
            duration=duration,
            loop=0
        )
        print(f"Matrix GIF saved to {matrix_gif_path}")

    if mse_images:
        mse_images[0].save(
            mse_gif_path,
            save_all=True,
            append_images=mse_images[1:],
            duration=duration,
            loop=0
        )
        print(f"MSE GIF saved to {mse_gif_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to folder containing net_g_*.pth files')
    parser.add_argument('--matrix_gif', type=str, required=True,
                        help='Output path for matrix evolution GIF')
    parser.add_argument('--mse_gif', type=str, required=True,
                        help='Output path for validation MSE GIF')
    parser.add_argument('--fps', type=int, default=50,
                        help='Frames per second for GIFs')
    args = parser.parse_args()

    make_dual_gifs(args.model_dir, args.matrix_gif, args.mse_gif, fps=args.fps)


if __name__ == '__main__':
    main()