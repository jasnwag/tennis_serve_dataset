import os
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

# --- Configuration ---
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "initial", "angles", "joint_angles_gender.npy"
)

# --- Load precomputed joint angles and gender ---
with open(DATA_PATH, "rb") as f:
    all_angles_gender = pickle.load(f)

serve_instances = []
gender_labels = []
for match_name, v in all_angles_gender.items():
    smoothed = v['angles']
    gender = v['gender']
    serve_flat = smoothed.flatten()
    if not np.any(np.isnan(serve_flat)):
        serve_instances.append(serve_flat)
        gender_labels.append(gender)

df = pd.DataFrame(serve_instances)
df['gender'] = gender_labels

# Drop any rows with NA in gender or in the features
clean_df = df.dropna(axis=0)
clean_df = clean_df[clean_df['gender'].notna() & (clean_df['gender'] != '') & (clean_df['gender'] != 'NA') & (clean_df['gender'] != None)]

# --- PCA (3 components) ---
X = clean_df.drop(columns=['gender']).values
y = clean_df['gender'].values

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# --- Plotly 3D scatter plot ---
fig = px.scatter_3d(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    z=X_pca[:, 2],
    color=y,
    labels={
        'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
        'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
        'z': f'PC3 ({pca.explained_variance_ratio_[2]:.2%})',
        'color': 'Gender'
    },
    title='3D PCA of Full Serve Sequences by Gender',
    opacity=0.7,
)
fig.update_traces(marker=dict(size=5))
fig.update_layout(
    legend_title_text='Gender',
    scene = dict(
        xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
        yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
        zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)',
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
fig.show()
