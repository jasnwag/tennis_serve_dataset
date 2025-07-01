import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Paths ---
ANGLES_SAVE_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/angles/joint_angles_gender.npy"
GIF_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/server_comparison/left_shoulder_average.gif"

# --- Angle index for left shoulder ---
ANGLE_NAMES = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]
LEFT_SHOULDER_IDX = 2

# --- Load data ---
with open(ANGLES_SAVE_PATH, "rb") as f:
    all_angles_gender = pickle.load(f)

men_serves = []
women_serves = []
for match_name, v in all_angles_gender.items():
    angles = v['angles']
    gender = v['gender']
    if np.any(np.isnan(angles)):
        continue
    if gender.lower().startswith('m'):
        men_serves.append(angles)
    elif gender.lower().startswith('w'):
        women_serves.append(angles)
men_serves = np.array(men_serves)
women_serves = np.array(women_serves)

# --- Compute average trajectory for left shoulder ---
avg_men_left = np.mean(men_serves, axis=0)[:, LEFT_SHOULDER_IDX]
avg_women_left = np.mean(women_serves, axis=0)[:, LEFT_SHOULDER_IDX]
n_frames = avg_men_left.shape[0]

# --- Plot and animate ---
fig, ax = plt.subplots(figsize=(20, 5))  # Wider plot
ax.plot(avg_men_left, label="Men Avg Left Shoulder", color="blue", linewidth=2)
ax.plot(avg_women_left, label="Women Avg Left Shoulder", color="red", linewidth=2)
ax.set_xlabel("Frame")
ax.set_ylabel("Angle (deg)")
ax.set_title("Average Left Shoulder Angle Trajectory")
ax.legend()
vertical_line = ax.axvline(0, color="black", linestyle="--", linewidth=2)

# --- Animation function ---
def animate(frame):
    vertical_line.set_xdata([frame, frame])
    return vertical_line,

ani = FuncAnimation(fig, animate, frames=n_frames, interval=20, blit=True)  # Faster animation
ani.save(GIF_PATH, writer=PillowWriter(fps=40))  # Higher fps for smoother, faster GIF
print(f"Animated GIF saved to {GIF_PATH}")
