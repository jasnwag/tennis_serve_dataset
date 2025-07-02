# Analysis Examples

This document provides practical examples and code snippets for analyzing the tennis serve dataset.

## Basic Data Loading and Exploration

### Load and Explore Dataset
```python
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Total serves: {len(df)}")
print(f"Unique players: {df['server_name'].nunique()}")
```

### Basic Statistics
```python
# Gender distribution
gender_stats = df['server_gender'].value_counts()
print("Gender distribution:")
print(gender_stats)

# Frame count statistics
frame_stats = df['n_frames'].describe()
print("\nFrame count statistics:")
print(frame_stats)

# Top players by serve count
top_players = df['server_name'].value_counts().head(10)
print("\nTop 10 players by serve count:")
print(top_players)
```

## Keypoint Analysis

### Load Keypoints for Analysis
```python
def load_keypoints_from_row(row):
    """Load keypoints from a dataset row."""
    keypoints_str = row['keypoints_clean']
    scores_str = row['keypoint_scores_clean']
    
    keypoints = np.array(json.loads(keypoints_str))
    scores = np.array(json.loads(scores_str))
    
    return keypoints, scores

# Example: Load keypoints for first serve
first_serve = df.iloc[0]
keypoints, scores = load_keypoints_from_row(first_serve)

print(f"Keypoints shape: {keypoints.shape}")
print(f"Scores shape: {scores.shape}")
```

### Analyze Joint Trajectories
```python
def analyze_joint_trajectory(keypoints, joint_index, joint_name):
    """Analyze trajectory of a specific joint."""
    joint_positions = keypoints[:, joint_index, :]
    
    # Calculate movement metrics
    total_distance = np.sum(np.linalg.norm(np.diff(joint_positions, axis=0), axis=1))
    max_velocity = np.max(np.linalg.norm(np.diff(joint_positions, axis=0), axis=1))
    
    print(f"{joint_name} Analysis:")
    print(f"  Total distance: {total_distance:.3f}")
    print(f"  Max velocity: {max_velocity:.3f}")
    
    return joint_positions

# Analyze right wrist trajectory (key for serve)
right_wrist_traj = analyze_joint_trajectory(keypoints, 10, "Right Wrist")
```

### Calculate Joint Angles
```python
def calculate_joint_angle(p1, p2, p3):
    """Calculate angle between three points (p1-p2-p3)."""
    v1 = p1 - p2
    v2 = p3 - p2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return np.degrees(angle)

def analyze_serve_angles(keypoints):
    """Analyze key angles during serve motion."""
    angles = {}
    
    for frame in range(len(keypoints)):
        # Right elbow angle (shoulder-elbow-wrist)
        shoulder = keypoints[frame, 6, :]  # Right shoulder
        elbow = keypoints[frame, 8, :]     # Right elbow
        wrist = keypoints[frame, 10, :]    # Right wrist
        
        elbow_angle = calculate_joint_angle(shoulder, elbow, wrist)
        
        if frame == 0:
            angles['elbow'] = []
        angles['elbow'].append(elbow_angle)
    
    return angles

# Analyze angles for a serve
serve_angles = analyze_serve_angles(keypoints)
print(f"Elbow angle range: {min(serve_angles['elbow']):.1f}° - {max(serve_angles['elbow']):.1f}°")
```

## Gender Analysis

### Compare Serve Characteristics by Gender
```python
def compare_gender_characteristics(df):
    """Compare serve characteristics between genders."""
    
    # Group by gender and calculate statistics
    gender_stats = df.groupby('server_gender').agg({
        'n_frames': ['mean', 'std', 'count'],
        'server_name': 'nunique'
    }).round(2)
    
    print("Serve characteristics by gender:")
    print(gender_stats)
    
    # Visualize frame count distribution
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    df.boxplot(column='n_frames', by='server_gender')
    plt.title('Frame Count by Gender')
    plt.suptitle('')
    
    plt.subplot(1, 2, 2)
    df['n_frames'].hist(by=df['server_gender'], bins=20, alpha=0.7)
    plt.title('Frame Count Distribution')
    
    plt.tight_layout()
    plt.show()

compare_gender_characteristics(df)
```

### Gender Classification from Motion
```python
def extract_motion_features(keypoints):
    """Extract motion features for gender classification."""
    features = {}
    
    # Shoulder width (distance between shoulders)
    shoulder_width = np.linalg.norm(keypoints[:, 5, :] - keypoints[:, 6, :], axis=1)
    features['avg_shoulder_width'] = np.mean(shoulder_width)
    features['max_shoulder_width'] = np.max(shoulder_width)
    
    # Hip width (distance between hips)
    hip_width = np.linalg.norm(keypoints[:, 11, :] - keypoints[:, 12, :], axis=1)
    features['avg_hip_width'] = np.mean(hip_width)
    features['max_hip_width'] = np.max(hip_width)
    
    # Arm span (distance from shoulder to wrist)
    left_arm_span = np.linalg.norm(keypoints[:, 5, :] - keypoints[:, 9, :], axis=1)
    right_arm_span = np.linalg.norm(keypoints[:, 6, :] - keypoints[:, 10, :], axis=1)
    features['avg_arm_span'] = np.mean((left_arm_span + right_arm_span) / 2)
    
    return features

# Extract features for all serves
motion_features = []
for idx, row in df.head(100).iterrows():  # Sample first 100 serves
    try:
        keypoints, _ = load_keypoints_from_row(row)
        features = extract_motion_features(keypoints)
        features['gender'] = row['server_gender']
        motion_features.append(features)
    except:
        continue

features_df = pd.DataFrame(motion_features)
print("Motion features by gender:")
print(features_df.groupby('gender').mean())
```

## Player Analysis

### Individual Player Analysis
```python
def analyze_player_serves(df, player_name):
    """Analyze serves for a specific player."""
    player_serves = df[df['server_name'] == player_name]
    
    print(f"Analysis for {player_name}:")
    print(f"  Total serves: {len(player_serves)}")
    print(f"  Average frames per serve: {player_serves['n_frames'].mean():.1f}")
    print(f"  Frame count range: {player_serves['n_frames'].min()} - {player_serves['n_frames'].max()}")
    
    # Analyze keypoint quality
    quality_scores = []
    for idx, row in player_serves.iterrows():
        try:
            _, scores = load_keypoints_from_row(row)
            quality_scores.append(np.mean(scores))
        except:
            continue
    
    if quality_scores:
        print(f"  Average keypoint quality: {np.mean(quality_scores):.3f}")
    
    return player_serves

# Analyze top players
top_players_list = df['server_name'].value_counts().head(5).index
for player in top_players_list:
    analyze_player_serves(df, player)
    print()
```

### Player Comparison
```python
def compare_players(df, players):
    """Compare serve characteristics between players."""
    comparison_data = []
    
    for player in players:
        player_serves = df[df['server_name'] == player]
        if len(player_serves) > 0:
            comparison_data.append({
                'player': player,
                'serve_count': len(player_serves),
                'avg_frames': player_serves['n_frames'].mean(),
                'gender': player_serves['server_gender'].iloc[0]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("Player comparison:")
    print(comparison_df)
    
    return comparison_df

# Compare top male and female players
top_male = df[df['server_gender'] == 'M']['server_name'].value_counts().head(3).index
top_female = df[df['server_gender'] == 'F']['server_name'].value_counts().head(3).index

print("Top male players:")
compare_players(df, top_male)
print("\nTop female players:")
compare_players(df, top_female)
```

## Visualization Examples

### Serve Motion Visualization
```python
def visualize_serve_motion(keypoints, title="Serve Motion"):
    """Create 3D visualization of serve motion."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot key joint trajectories
    joints_to_plot = [6, 8, 10]  # Right shoulder, elbow, wrist
    joint_names = ['Shoulder', 'Elbow', 'Wrist']
    colors = ['red', 'blue', 'green']
    
    for joint_idx, joint_name, color in zip(joints_to_plot, joint_names, colors):
        joint_pos = keypoints[:, joint_idx, :]
        ax.plot(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2], 
                color=color, label=joint_name, linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    plt.show()

# Visualize a sample serve
sample_serve = df.iloc[0]
keypoints, _ = load_keypoints_from_row(sample_serve)
visualize_serve_motion(keypoints, f"Serve by {sample_serve['server_name']}")
```

### Statistical Visualizations
```python
# Create comprehensive visualization dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Frame count distribution by gender
df.boxplot(column='n_frames', by='server_gender', ax=axes[0,0])
axes[0,0].set_title('Frame Count by Gender')

# 2. Player serve counts
top_10_players = df['server_name'].value_counts().head(10)
top_10_players.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Top 10 Players by Serve Count')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Gender distribution
df['server_gender'].value_counts().plot(kind='pie', ax=axes[0,2])
axes[0,2].set_title('Gender Distribution')

# 4. Frame count histogram
df['n_frames'].hist(bins=30, ax=axes[1,0])
axes[1,0].set_title('Frame Count Distribution')
axes[1,0].set_xlabel('Number of Frames')

# 5. Serves per player (top 20)
df['server_name'].value_counts().head(20).plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Serves per Player (Top 20)')
axes[1,1].tick_params(axis='x', rotation=45)

# 6. Frame count by gender (violin plot)
sns.violinplot(data=df, x='server_gender', y='n_frames', ax=axes[1,2])
axes[1,2].set_title('Frame Count Distribution by Gender')

plt.tight_layout()
plt.show()
```

## Advanced Analysis

### Serve Clustering
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_serves_by_motion(df, n_clusters=3):
    """Cluster serves based on motion characteristics."""
    motion_features = []
    
    for idx, row in df.head(500).iterrows():  # Sample for clustering
        try:
            keypoints, _ = load_keypoints_from_row(row)
            features = extract_motion_features(keypoints)
            features['serve_id'] = idx
            motion_features.append(features)
        except:
            continue
    
    features_df = pd.DataFrame(motion_features)
    
    # Prepare features for clustering
    feature_cols = ['avg_shoulder_width', 'avg_hip_width', 'avg_arm_span']
    X = features_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    features_df['cluster'] = clusters
    
    print("Serve clusters:")
    for cluster in range(n_clusters):
        cluster_data = features_df[features_df['cluster'] == cluster]
        print(f"Cluster {cluster}: {len(cluster_data)} serves")
        print(f"  Average features: {cluster_data[feature_cols].mean().round(3).to_dict()}")
    
    return features_df

# Perform clustering
clustered_serves = cluster_serves_by_motion(df)
```

These examples demonstrate various ways to analyze the tennis serve dataset, from basic exploration to advanced machine learning applications. 