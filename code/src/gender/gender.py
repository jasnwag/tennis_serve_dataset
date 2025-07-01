import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- Configuration ---
DATA_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/raw/all_matches_data_gender.npy"
ANGLE_NAMES = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]
ANGLES = {
    'left_elbow': [11, 12, 13],    # Left shoulder, elbow, wrist
    'right_elbow': [14, 15, 16],   # Right shoulder, elbow, wrist  
    'left_shoulder': [12, 11, 4],   # Left elbow, shoulder, hip
    'right_shoulder': [15, 14, 1],  # Right elbow, shoulder, hip
    'left_hip': [11, 4, 5],        # Left shoulder, hip, knee
    'right_hip': [14, 1, 2],       # Right shoulder, hip, knee
    'left_knee': [4, 5, 6],        # Left hip, knee, ankle
    'right_knee': [1, 2, 3]        # Right hip, knee, ankle
}

# --- Functions ---
def calculate_angle(A, B, C):
    AB = B - A
    CB = B - C
    dot_product = np.dot(AB, CB)
    mag_AB = np.linalg.norm(AB)
    mag_CB = np.linalg.norm(CB)
    if mag_AB == 0 or mag_CB == 0:
        return np.nan
    cos_theta = np.clip(dot_product / (mag_AB * mag_CB), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    degrees = np.degrees(theta)
    return 180 - degrees

def extract_angles_from_keypoints(keypoints, angles_dict):
    n_frames = keypoints.shape[0]
    n_angles = len(angles_dict)
    angle_array = np.full((n_frames, n_angles), np.nan)
    for frame_idx in range(n_frames):
        frame = keypoints[frame_idx]
        for angle_idx, indices in enumerate(angles_dict.values()):
            A, B, C = frame[indices[0]], frame[indices[1]], frame[indices[2]]
            if np.any(A) and np.any(B) and np.any(C):
                angle_array[frame_idx, angle_idx] = calculate_angle(A, B, C)
    return angle_array

def smooth_angles(angle_array, window_length=11, polyorder=3):
    smoothed = np.copy(angle_array)
    for i in range(angle_array.shape[1]):
        if not np.any(np.isnan(angle_array[:, i])):
            smoothed[:, i] = savgol_filter(angle_array[:, i], window_length, polyorder)
        else:
            smoothed[:, i] = np.nan
    return smoothed

# --- Main Processing ---
import os
import pickle

ANGLES_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(DATA_PATH)),
    "angles", "joint_angles_gender.npy"
)
os.makedirs(os.path.dirname(ANGLES_SAVE_PATH), exist_ok=True)

# Try to load precomputed angles, else compute and save
if os.path.exists(ANGLES_SAVE_PATH):
    print(f"Loading precomputed joint angles from {ANGLES_SAVE_PATH}")
    with open(ANGLES_SAVE_PATH, "rb") as f:
        all_angles_gender = pickle.load(f)
else:
    print("Computing joint angles for all matches...")
    data = np.load(DATA_PATH, allow_pickle=True).item()
    all_angles_gender = {}
    for match_name, match_data in tqdm(data.items()):
        keypoints = match_data['keypoints']  # (90, 17, 3)
        gender = match_data['gender']
        angles = extract_angles_from_keypoints(keypoints, ANGLES)
        smoothed = smooth_angles(angles)
        all_angles_gender[match_name] = {
            'angles': smoothed,
            'gender': gender
        }
    # Save for next time
    with open(ANGLES_SAVE_PATH, "wb") as f:
        pickle.dump(all_angles_gender, f)
    print(f"Saved joint angles to {ANGLES_SAVE_PATH}")

# Prepare serve_instances and gender_labels from loaded data
serve_instances = []
gender_labels = []
for match_name, v in all_angles_gender.items():
    smoothed = v['angles']
    gender = v['gender']
    if not np.any(np.isnan(smoothed)):
        serve_instances.append(smoothed.flatten())
        gender_labels.append(gender)

# --- DataFrame for easy matching ---
df = pd.DataFrame(serve_instances)
df['gender'] = gender_labels

# --- PCA & Plotting ---
X = df.drop(columns=['gender']).values
y = df['gender'].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# plt.figure(figsize=(20, 10))
# for gender, color in zip(np.unique(y), ['blue', 'red']):
#     mask = y == gender
#     plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=gender, color=color, alpha=0.1, s=10)

# plt.title('PCA of Full Serve Sequences by Gender')
# plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
# plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)')

# plt.legend()
# plt.grid(True)
# plt.show()