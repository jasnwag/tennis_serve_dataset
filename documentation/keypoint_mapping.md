# Keypoint Mapping

This document describes the 17 keypoint joints used in the tennis serve analysis dataset.

## Joint Definitions

The dataset uses 17 keypoints to track the full body motion during tennis serves. Each keypoint has 3D coordinates (x, y, z) and a confidence score.

### Keypoint Index Mapping

| Index | Joint Name | Description | Body Region |
|-------|------------|-------------|-------------|
| 0 | Nose | Tip of the nose | Head |
| 1 | Left Eye | Left eye center | Head |
| 2 | Right Eye | Right eye center | Head |
| 3 | Left Ear | Left ear position | Head |
| 4 | Right Ear | Right ear position | Head |
| 5 | Left Shoulder | Left shoulder joint | Upper Body |
| 6 | Right Shoulder | Right shoulder joint | Upper Body |
| 7 | Left Elbow | Left elbow joint | Arms |
| 8 | Right Elbow | Right elbow joint | Arms |
| 9 | Left Wrist | Left wrist joint | Arms |
| 10 | Right Wrist | Right wrist joint | Arms |
| 11 | Left Hip | Left hip joint | Lower Body |
| 12 | Right Hip | Right hip joint | Lower Body |
| 13 | Left Knee | Left knee joint | Legs |
| 14 | Right Knee | Right knee joint | Legs |
| 15 | Left Ankle | Left ankle joint | Legs |
| 16 | Right Ankle | Right ankle joint | Legs |

## Coordinate System

### 3D Coordinates
- **X-axis**: Horizontal position (left to right)
- **Y-axis**: Vertical position (top to bottom)
- **Z-axis**: Depth position (front to back)

### Normalization
- All coordinates are normalized to the video frame dimensions
- X and Y coordinates range from 0.0 to 1.0
- Z coordinates represent relative depth (negative = closer to camera)

## Data Structure

### Keypoints Array Format
```python
keypoints = [
    [x0, y0, z0],  # Nose
    [x1, y1, z1],  # Left Eye
    [x2, y2, z2],  # Right Eye
    # ... 17 keypoints total
]
```

### Confidence Scores
- Range: 0.0 to 1.0
- 0.0 = Low confidence (keypoint may be occluded or unclear)
- 1.0 = High confidence (keypoint clearly visible)

## Usage Examples

### Loading Keypoints from CSV
```python
import pandas as pd
import numpy as np
import json

# Load dataset
df = pd.read_csv('data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv')

# Get keypoints for first serve
keypoints_str = df.iloc[0]['keypoints_clean']
keypoints = json.loads(keypoints_str)
keypoints_array = np.array(keypoints)

print(f"Keypoints shape: {keypoints_array.shape}")
# Output: (n_frames, 17, 3)
```

### Accessing Specific Joints
```python
# Get right wrist positions across all frames
right_wrist = keypoints_array[:, 10, :]  # Index 10 = Right Wrist

# Get left shoulder in first frame
left_shoulder_frame0 = keypoints_array[0, 5, :]  # Index 5 = Left Shoulder
```

### Calculating Joint Angles
```python
def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1, 1))

# Example: Calculate right elbow angle
shoulder = keypoints_array[0, 6, :]  # Right shoulder
elbow = keypoints_array[0, 8, :]     # Right elbow
wrist = keypoints_array[0, 10, :]    # Right wrist

elbow_angle = calculate_angle(shoulder, elbow, wrist)
```

## Quality Considerations

### High-Quality Keypoints
- **Head**: Nose, eyes, ears (indices 0-4)
- **Shoulders**: Left and right shoulder joints (indices 5-6)
- **Hips**: Left and right hip joints (indices 11-12)

### Potentially Noisy Keypoints
- **Extremities**: Wrists and ankles (indices 9-10, 15-16)
- **Occluded joints**: May have lower confidence scores

### Confidence Thresholds
```python
# Filter keypoints by confidence
confidence_threshold = 0.5
high_confidence_keypoints = keypoints_array[confidence_scores > confidence_threshold]
```

## Biomechanical Analysis

### Key Joints for Serve Analysis
1. **Shoulders** (5-6): Core rotation and power generation
2. **Elbows** (7-8): Arm extension and ball contact
3. **Wrists** (9-10): Racket control and follow-through
4. **Hips** (11-12): Lower body rotation and stability
5. **Knees** (13-14): Leg drive and power transfer

### Common Measurements
- **Shoulder rotation**: Angle between shoulders and hips
- **Elbow extension**: Angle at elbow joint
- **Wrist position**: Ball contact point relative to body
- **Hip rotation**: Lower body contribution to serve power 