import os
import pickle
import numpy as np

# Path to the precomputed angles file
ANGLES_SAVE_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/angles/joint_angles_gender.npy"

# Load the data
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
        continue  # skip if any NaNs
    flat = angles.flatten()
    if gender.lower().startswith('m'):
        men_serves.append(flat)
        men_names.append(match_name)
    elif gender.lower().startswith('w'):
        women_serves.append(flat)
        women_names.append(match_name)

men_serves = np.array(men_serves)
women_serves = np.array(women_serves)

# Compute average trajectory for each gender
avg_men = np.mean(men_serves, axis=0)
avg_women = np.mean(women_serves, axis=0)

# Find the serve closest to the average for each gender
men_dists = np.linalg.norm(men_serves - avg_men, axis=1)
women_dists = np.linalg.norm(women_serves - avg_women, axis=1)

best_men_idx = np.argmin(men_dists)
best_women_idx = np.argmin(women_dists)

print("Best matching men's serve:", men_names[best_men_idx])
print("Best matching women's serve:", women_names[best_women_idx])
