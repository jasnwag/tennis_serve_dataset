# Tennis Serve Analysis Dataset

![Tennis Serve Analysis Demo](assets/bounding_grid_8x3.gif)

### Comparative Skeletal Motion Analysis
<div align="center">
  <img src="assets/skeleton.gif" alt="Side-by-side comparison of skeletal motion between two tennis players during serve execution" width="600"/>
  <p><em>Skeletal motion comparison between two players showcasing biomechanical differences in serve technique</em></p>
</div>

A comprehensive dataset of tennis serves from the 2024 US Open, featuring 3D keypoint tracking, serve analysis, and gender classification data.

## 🎬 Additional Dimensionaltiy Reduction Visualizations 


### Gender-Based Motion Clustering
<div align="center">
  <img src="assets/gender.gif" alt="3D visualization showing distinct clustering patterns between male and female tennis players in motion space" width="600"/>
  <p><em>3D motion space visualization revealing distinct separation patterns between male and female serve biomechanics</em></p>
</div>

### Player-Specific Serve Signatures  
<div align="center">
  <img src="assets/server.gif" alt="3D clustering visualization showing how different tennis players form distinct clusters in motion space" width="600"/>
  <p><em>Individual player clustering in 3D motion space - each player develops unique biomechanical signatures</em></p>
</div>

## 🎾 Dataset Overview

This dataset contains **6,370 tennis serves** from the 2024 US Open tournament, featuring:

- **3D Keypoint Tracking**: 17 joints per frame with full body motion capture
- **Player Demographics**: 118 unique players (60 male, 58 female)
- **Serve Analytics**: Speed, direction, outcome, and match context
- **High-Quality Data**: Professional tournament footage with expert annotation

## 📊 Quick Stats

- **Total Serves**: 6,370
- **Unique Players**: 118 (60 male, 58 female)
- **Matches**: 118 different tennis matches
- **Frame Range**: 60-120 frames per serve (mean: 81.2 frames)
- **Data Format**: CSV with embedded JSON keypoint arrays
- **Dataset Size**: ~849MB (full dataset available via Google Drive)

## 🗂️ Repository Structure

```
tennis_serve_dataset/
├── assets/                     # Visualization GIFs and images
│   ├── bounding_grid_8x3.gif
│   ├── skeleton.gif
│   ├── gender.gif
│   └── server.gif
├── data/
│   └── us_open_data/           # Dataset files
│       └── 2024-usopen-final-filtered-cleaned-lite.csv
├── code/                       # Analysis code and experiments
└── README.md                   # This file
```

## 🚀 Getting Started

### Download the Dataset

**Option 1: Full Dataset (Recommended)**
📁 **[Download Complete Dataset from Google Drive](https://drive.google.com/your-link-here)** (~849MB)

The full dataset includes all biomechanical features:
- 3D keypoints for each serve
- Joint angles and angular velocities
- Complete statistical data

**Option 2: Lite Version (GitHub)**
The repository contains a cleaned version with essential data:
```
data/us_open_data/2024-usopen-final-filtered-cleaned-lite.csv
```
*Note: This version has empty columns removed but still contains all keypoint data*

### Quick Start Example

```python
import pandas as pd
import numpy as np
import json

# Load the dataset
df = pd.read_csv('data/us_open_data/2024-usopen-final-filtered-cleaned-lite.csv')

# Basic statistics
print(f"Total serves: {len(df)}")
print(f"Unique players: {df['server'].nunique()}")
print(f"Player examples: {df['server'].value_counts().head()}")

# Load keypoints for analysis (if available)
if 'keypoints' in df.columns:
    keypoints = json.loads(df.iloc[0]['keypoints'])
    print(f"Keypoints shape: {np.array(keypoints).shape}")
    # Output: (n_frames, 17, 3) - frames × joints × coordinates
```

## 📈 Key Features

### 🎯 Serve Data
- **Player Information**: Server name, gender, match details
- **Serve Metrics**: Speed, direction, outcome
- **Match Context**: Tournament round, court, date
- **Point Details**: Score, game state, rally length

### 🦴 3D Keypoint Tracking
- **17 Joints**: Full body tracking including arms, legs, torso
- **3D Coordinates**: X, Y, Z positions for each joint
- **Confidence Scores**: Reliability metrics for each keypoint
- **Frame-by-Frame**: Complete serve motion capture

### 👥 Player Demographics
- **Gender Distribution**: 54.7% Male, 45.3% Female
- **Player Diversity**: 118 unique players
- **Top Players**: Sinner (224 serves), Tiafoe (221 serves), Sabalenka (194 serves)

## 🔬 Research Applications

### Biomechanics
- Serve motion analysis and optimization
- Joint angle calculations and biomechanical modeling
- Performance comparison across players and genders

### Machine Learning
- Gender classification from motion patterns
- Serve outcome prediction
- Player identification and verification
- Motion synthesis and generation

### Sports Analytics
- Serve effectiveness analysis
- Player comparison studies
- Performance benchmarking and ranking

## 📋 Data Dictionary

### Core Columns
- `match_id`: Unique match identifier
- `server`: Name of the serving player
- `player1`, `player2`: Match participants
- `PointServer`: Server identifier (1 or 2)
- `n_frames`: Number of frames in the serve sequence
- `Speed_KMH`: Serve speed in kilometers per hour

### Keypoint Data
- `keypoints`: 3D coordinates array (n_frames × 17 × 3)
- Joint angle data: Left/right angles for elbows, shoulders, hips, knees
- Angular velocity and acceleration data for all joints

### Match Context
- `ElapsedTime`: Time in match
- `SetNo`, `GameNo`, `PointNumber`: Match progression
- `P1Score`, `P2Score`: Point scores
- `match_num`: Match number in tournament

*Note: Empty columns (court info, player IDs, etc.) have been removed from the lite version*

## 📊 Sample Analysis Results

### Serve Length Distribution
- **Mean**: 81.2 frames per serve
- **Range**: 60-120 frames
- **Most Common**: 90 frames (468 serves)

### Gender Analysis
- **Male Players**: 60 players, 3,659 serves (54.7%)
- **Female Players**: 58 players, 3,035 serves (45.3%)

### Top Players by Serve Count
1. **Jannik Sinner**: 224 serves
2. **Frances Tiafoe**: 221 serves  
3. **Taylor Fritz**: 214 serves
4. **Aryna Sabalenka**: 194 serves
5. **Jessica Pegula**: 169 serves

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{tennis_serve_analysis_2024,
  title={Tennis Serve Analysis Dataset: 3D Keypoint Tracking from US Open 2024},
  author={Tennis Analytics Research Team},
  year={2024},
  url={https://github.com/jasonwang/tennis_serve_dataset}
}
```

## 📄 License

This dataset is licensed under the [MIT License](LICENSE).

## 🤝 Contributing

We welcome contributions to improve the dataset documentation and analysis! Please feel free to:

- Report data quality issues
- Suggest additional documentation
- Share analysis results
- Propose dataset improvements

## 📞 Contact

For questions about the dataset or collaboration opportunities, please open an issue on GitHub.

---

**Dataset Version**: 1.1  
**Last Updated**: September 2025  
**Repository Size**: ~37MB (assets) + ~849MB (lite dataset)  
**Full Dataset Size**: ~849MB (available via Google Drive) 