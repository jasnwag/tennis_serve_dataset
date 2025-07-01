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
women_serves = []
for match_name, v in all_angles_gender.items():
    angles = v['angles']  # (frames, angles)
    gender = v['gender']
    if np.any(np.isnan(angles)):
        continue  # skip if any NaNs
    if gender.lower().startswith('m'):
        men_serves.append(angles)
    elif gender.lower().startswith('w'):
        women_serves.append(angles)

men_serves = np.array(men_serves)  # (N_men, frames, angles)
women_serves = np.array(women_serves)

# Compute average trajectory for each gender
avg_men = np.mean(men_serves, axis=0)  # (frames, angles)
avg_women = np.mean(women_serves, axis=0)

# Plot all average trajectories for each joint angle, with ±1 std shading and consistent backgrounds
n_angles = avg_men.shape[1]
n_frames = avg_men.shape[0]
fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, facecolor='white')  # 8 total, but only 7 used
axes = axes.flatten()

# Calculate std for shading
std_men = np.std(men_serves, axis=0)  # (frames, angles)
std_women = np.std(women_serves, axis=0)

for i, angle_name in enumerate(ANGLE_NAMES):
    ax = axes[i]
    ax.set_facecolor('white')  # Explicitly set background
    # Men avg and std
    ax.plot(avg_men[:, i], label="Men Avg", color="blue", linewidth=2.5)
    ax.fill_between(
        np.arange(n_frames),
        avg_men[:, i] - std_men[:, i],
        avg_men[:, i] + std_men[:, i],
        color="blue", alpha=0.15, label="Men ±1 SD"
    )
    # Women avg and std
    ax.plot(avg_women[:, i], label="Women Avg", color="red", linewidth=2.5)
    ax.fill_between(
        np.arange(n_frames),
        avg_women[:, i] - std_women[:, i],
        avg_women[:, i] + std_women[:, i],
        color="red", alpha=0.15, label="Women ±1 SD"
    )
    ax.set_title(angle_name.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel("Angle (deg)", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
# Only hide the 8th (unused) subplot, not the right knee
if len(ANGLE_NAMES) < len(axes):
    axes[len(ANGLE_NAMES)].axis('off')
# Set x-label for bottom row
for idx in range(4, 4 + min(4, len(ANGLE_NAMES)-4)):
    axes[idx].set_xlabel("Frame", fontsize=12)
fig.suptitle("Average Joint Angle Trajectories by Gender (±1 SD)", fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.96])

save_dir = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/average_trajectories"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "all_average_trajectories.png"), dpi=200, facecolor=fig.get_facecolor())
plt.close()
print("Saved all average trajectories to all_average_trajectories.png (with ±1 SD)")
