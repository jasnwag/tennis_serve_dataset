import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
ANGLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "initial", "angles", "joint_angles_gender.npy"
)
with open(ANGLES_PATH, "rb") as f:
    all_angles_gender = pickle.load(f)

serve_instances = []
gender_labels = []
for match_name, v in all_angles_gender.items():
    smoothed = v['angles']
    gender = v['gender']
    serve_flat = smoothed.flatten()
    if (not np.any(np.isnan(serve_flat))) and (gender is not None) and (gender != '') and (gender != 'NA'):
        serve_instances.append(serve_flat)
        gender_labels.append(gender)

df = pd.DataFrame(serve_instances)
df['gender'] = gender_labels

# --- Encode gender as binary ---
gender_map = {g: i for i, g in enumerate(sorted(df['gender'].unique()))}
df['gender_bin'] = df['gender'].map(gender_map)

# --- Train/Test Split ---
X = df.drop(columns=['gender', 'gender_bin']).values
y = df['gender_bin'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- Classifier: Random Forest ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1] if clf.n_classes_ == 2 else None

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=[k for k,v in sorted(gender_map.items(), key=lambda x:x[1])])
roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

print("=== Random Forest Classifier Results ===")
print(f"Accuracy: {acc:.3f}")
if roc_auc is not None:
    print(f"ROC-AUC: {roc_auc:.3f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)

# --- Feature Importance ---
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# Show top 10 most important features (which angle, which frame)
print("Top 10 Important Features (by index):")
for rank in range(10):
    idx = indices[rank]
    angle_idx = idx % 8
    frame_idx = idx // 8
    print(f"Rank {rank+1}: Angle {angle_idx} (frame {frame_idx}), Importance: {importances[idx]:.4f}")

# --- Visualize Feature Importances (heatmap) ---
importances_reshaped = importances.reshape(-1, 8)  # frames x angles
plt.figure(figsize=(12, 6))
# --- Save all visualizations to the visualizations/ folder ---
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'initial', 'visualizations')
FI_DIR = os.path.join(VIS_DIR, 'feature_importance')
AT_DIR = os.path.join(VIS_DIR, 'average_trajectories')
LR_DIR = os.path.join(VIS_DIR, 'logistic_regression')
for d in [FI_DIR, AT_DIR, LR_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Feature Importance Heatmap ---
plt.figure(figsize=(12, 6))
sns.heatmap(importances_reshaped.T, cmap="viridis", cbar_kws={'label': 'Importance'})
plt.yticks(np.arange(8)+0.5, [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
], rotation=0)
plt.xlabel('Frame')
plt.ylabel('Angle')
plt.title('Random Forest Feature Importances (Frame x Angle)')
plt.tight_layout()
plt.savefig(os.path.join(FI_DIR, 'random_forest_feature_importance.png'))
plt.close()

# --- Visualize Average Angle Trajectories for Each Gender ---
angle_names = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]
frames = np.arange(importances_reshaped.shape[0])
for angle_idx, angle_name in enumerate(angle_names):
    plt.figure(figsize=(10, 5))
    for gender_val in df['gender'].unique():
        mask = df['gender'] == gender_val
        # Only use numeric columns for angle data
        angle_series = df.loc[mask].drop(columns=['gender', 'gender_bin']).iloc[:, angle_idx::8].values
        mean_curve = np.mean(angle_series, axis=0)
        std_curve = np.std(angle_series, axis=0)
        plt.plot(frames, mean_curve, label=f"{gender_val} (mean)")
        plt.fill_between(frames, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)
    plt.title(f"Average {angle_name} Angle Trajectory by Gender")
    plt.xlabel("Frame")
    plt.ylabel("Angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(AT_DIR, f"average_trajectory_{angle_name}.png"))
    plt.close()

# --- Logistic Regression Coefficients (Optional) ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
coefs = logreg.coef_.reshape(-1, 8)  # frames x angles
plt.figure(figsize=(12, 6))
sns.heatmap(coefs.T, center=0, cmap="coolwarm", cbar_kws={'label': 'Coefficient'})
plt.yticks(np.arange(8)+0.5, angle_names, rotation=0)
plt.xlabel('Frame')
plt.ylabel('Angle')
plt.title('Logistic Regression Coefficients (Frame x Angle)')
plt.tight_layout()
plt.savefig(os.path.join(LR_DIR, 'logistic_regression_coefficients.png'))
plt.close()

