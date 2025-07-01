import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path to the precomputed angles file
ANGLES_SAVE_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/angles/joint_angles_gender.npy"

# Angle names (should match the order in the data)
ANGLE_NAMES = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]

with open(ANGLES_SAVE_PATH, "rb") as f:
    all_angles_gender = pickle.load(f)

# Separate serves by gender
men_serves = []
men_names = []
women_serves = []
women_names = []

for match_name, v in all_angles_gender.items():
    angles = v['angles']
    gender = v['gender']
    if np.any(np.isnan(angles)):
        continue
    if gender.lower().startswith('m'):
        men_serves.append(angles)
        men_names.append(match_name)
    elif gender.lower().startswith('w'):
        women_serves.append(angles)
        women_names.append(match_name)

men_serves = np.array(men_serves)  # (N_men, frames, angles)
women_serves = np.array(women_serves)

# Compute average trajectory for each gender
avg_men = np.mean(men_serves, axis=0)
avg_women = np.mean(women_serves, axis=0)

# Find the serve closest to the average for each gender
men_flat = men_serves.reshape(len(men_serves), -1)
women_flat = women_serves.reshape(len(women_serves), -1)
avg_men_flat = avg_men.flatten()
avg_women_flat = avg_women.flatten()
men_dists = np.linalg.norm(men_flat - avg_men_flat, axis=1)
women_dists = np.linalg.norm(women_flat - avg_women_flat, axis=1)
best_men_idx = np.argmin(men_dists)
best_women_idx = np.argmin(women_dists)

# Plotting function
def plot_serve_group(serves, avg, best_idx, names, gender_label, color, save_dir):
    n_angles = serves.shape[2]
    n_frames = serves.shape[1]
    fig, axes = plt.subplots(n_angles, 1, figsize=(12, 2.5 * n_angles), sharex=True)
    if n_angles == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        # Plot all serves (light gray)
        for s in serves:
            ax.plot(s[:, i], color='lightgray', alpha=0.3, linewidth=0.8)
        # Plot average (bold)
        ax.plot(avg[:, i], color=color, label=f"Average {gender_label}", linewidth=2.5)
        # Plot best-matching serve (bold, dashed)
        ax.plot(serves[best_idx][:, i], color='black', linestyle='--', label=f"Best-Matching Example\n({names[best_idx]})", linewidth=2)
        ax.set_ylabel(f"{ANGLE_NAMES[i]} (deg)")
        ax.legend()
    axes[-1].set_xlabel("Frame")
    fig.suptitle(f"{gender_label.capitalize()} Serves: Average vs. Best-Matching Example")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # Save plot
    fname = f"best_vs_average_{gender_label}.png"
    out_path = os.path.join(save_dir, fname)
    plt.savefig(out_path)
    plt.close(fig)

# Save directory for plots
SAVE_DIR = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/average_trajectories/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Plot for men
plot_serve_group(men_serves, avg_men, best_men_idx, men_names, 'men', 'blue', SAVE_DIR)
# Plot for women
plot_serve_group(women_serves, avg_women, best_women_idx, women_names, 'women', 'red', SAVE_DIR)
