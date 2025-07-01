import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import imageio

# --- Config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PCA_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "initial", "pca", "pca_data_removed_errors.json")
SUMMARY_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "initial", "visualizations", "server_patterns", "server_cluster_summary.csv")

# --- Load Data ---
with open(PCA_JSON_PATH, "r") as f:
    pca_data = json.load(f)
rows = []
for k, v in pca_data.items():
    row = {
        "id": k,
        "server": v["server"],
        "gender": v["gender"],
        "PC1": v["PC1"],
        "PC2": v["PC2"],
        "PC3": v["PC3"],
        "Speed_MPH": v.get("Speed_MPH", np.nan)
    }
    rows.append(row)
df = pd.DataFrame(rows)

# --- Load summary for all servers (for summary chart) ---
if os.path.exists(SUMMARY_CSV_PATH):
    summary_df = pd.read_csv(SUMMARY_CSV_PATH)
else:
    summary_df = None

def animate_server_comparison(server1, server2, save_path=None):
    """
    Animates serves for two servers, with a GIF (left_shoulder_average.gif) on the left, and PCA plots for each server in the center and right.
    """
    # --- Load the GIF frames ---
    gif_path = os.path.join(PROJECT_ROOT, "data", "initial", "visualizations", "server_comparison", "left_shoulder_average.gif")
    gif_frames = imageio.mimread(gif_path)
    n_gif_frames = len(gif_frames)

    # --- Prepare data ---
    sub1 = df[df['server'] == server1].reset_index(drop=True)
    sub2 = df[df['server'] == server2].reset_index(drop=True)
    n1, n2 = len(sub1), len(sub2)
    n_frames = max(n1, n2, n_gif_frames)

    # --- Layout: 1 row, 3 columns ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    ax_gif, ax1, ax2 = axes
    ax_gif.axis('off')
    ax1.set_title(f"{server1} Serves in PCA Space")
    ax2.set_title(f"{server2} Serves in PCA Space")
    for ax in [ax1, ax2]:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(df['PC1'].min()-10, df['PC1'].max()+10)
        ax.set_ylim(df['PC2'].min()-10, df['PC2'].max()+10)
    scat1 = ax1.scatter([], [], c='blue', alpha=0.8, label=server1)
    scat2 = ax2.scatter([], [], c='red', alpha=0.8, label=server2)
    ax1.legend()
    ax2.legend()
    gif_im = ax_gif.imshow(gif_frames[0])

    # --- Animation function ---
    def update(frame):
        # GIF frame
        gif_idx = min(frame, n_gif_frames-1)
        gif_im.set_data(gif_frames[gif_idx])
        # Reveal up to 'frame' serves for each server
        scat1.set_offsets(sub1[['PC1', 'PC2']].iloc[:min(frame+1, n1)].values)
        scat2.set_offsets(sub2[['PC1', 'PC2']].iloc[:min(frame+1, n2)].values)
        return gif_im, scat1, scat2

    anim = FuncAnimation(fig, update, frames=n_frames, interval=150, blit=False, repeat=False)
    plt.tight_layout()
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=5)
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Animate comparison of two servers.")
    parser.add_argument("server1", type=str, help="Name of first server")
    parser.add_argument("server2", type=str, help="Name of second server")
    parser.add_argument("--save", type=str, default=None, help="Path to save animation (mp4). If not set, will show interactively.")
    args = parser.parse_args()
    animate_server_comparison(args.server1, args.server2, args.save)
