# Tennis Serve Analysis Dataset

A comprehensive dataset of tennis serves from the 2024 US Open, featuring 3D keypoint tracking, serve analysis, and gender classification data.

## 📊 Dataset Overview

- **Total Serves**: 6,694 serves
- **Players**: 118 unique players (60 male, 58 female)
- **Matches**: 118 different tennis matches
- **Data Sources**: US Open 2024 tournament
- **Keypoint Tracking**: 17 joints per frame, 3D coordinates (x, y, z)
- **Frame Range**: 60-120 frames per serve (mean: 81.2 frames)

## 🗂️ Dataset Structure

```
tennis-dataset/
├── data/
│   ├── full/
│   │   ├── all_jsons/                           # Original keypoint JSON files (10,727 files)
│   │   ├── restructured_jsons/                  # Clean keypoint format (10,727 files)
│   │   └── usopen_points_clean_keypoints_cleaned_with_server_gender.csv  # Main dataset
│   ├── scorebug/                                # Scorebug detection data
│   └── initial/                                 # Initial processing data
├── code/
│   ├── src/
│   │   ├── json_investigation/                  # Data processing scripts
│   │   ├── coaching/                           # Coaching analysis tools
│   │   ├── gender/                             # Gender classification analysis
│   │   └── logistics/                          # Data integration tools
│   └── notebooks/                              # Jupyter notebooks for analysis
├── documentation/
│   ├── data_dictionary.md                      # Column descriptions
│   ├── keypoint_mapping.md                     # 17 joint definitions
│   └── analysis_examples.md                    # Usage examples
└── README.md                                   # This file
```

## 📈 Key Features

### 🎾 Serve Data
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

## 🔧 Data Processing Pipeline

1. **Raw Data Collection**: Video analysis and keypoint extraction
2. **JSON Restructuring**: Clean format conversion (n_frames × 17 joints × 3 coordinates)
3. **CSV Integration**: Match data + keypoints + metadata
4. **Data Cleaning**: Remove empty columns, add derived features
5. **Quality Control**: Validation and verification

## 📋 Data Dictionary

### Core Columns
- `video_name`: Original video filename
- `server_name`: Name of the serving player
- `server_gender`: Gender of server (M/F)
- `player1`, `player2`: Match participants
- `PointServer`: Server identifier (1 or 2)
- `n_frames`: Number of frames in the serve sequence

### Keypoint Data
- `keypoints_clean`: 3D coordinates array (n_frames × 17 × 3)
- `keypoint_scores_clean`: Confidence scores array (n_frames × 17)

### Match Context
- `tournament`: Tournament name
- `round`: Match round
- `court`: Court information
- `date`: Match date

## 🚀 Getting Started

### Quick Start
```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv')

# Basic statistics
print(f"Total serves: {len(df)}")
print(f"Unique players: {df['server_name'].nunique()}")
print(f"Gender distribution:\n{df['server_gender'].value_counts()}")

# Load keypoints for a specific serve
import json
keypoints = json.loads(df.iloc[0]['keypoints_clean'])
print(f"Keypoints shape: {np.array(keypoints).shape}")
```

### Analysis Examples
```python
# Analyze serves by gender
gender_stats = df.groupby('server_gender').agg({
    'n_frames': ['mean', 'std'],
    'server_name': 'count'
}).round(2)

# Top servers by serve count
top_servers = df['server_name'].value_counts().head(10)

# Frame distribution analysis
frame_dist = df['n_frames'].describe()
```

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

## 🔬 Research Applications

### Biomechanics
- Serve motion analysis
- Joint angle calculations
- Performance optimization

### Machine Learning
- Gender classification from motion
- Serve outcome prediction
- Player identification

### Sports Analytics
- Serve effectiveness analysis
- Player comparison studies
- Performance benchmarking

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{tennis_serve_analysis_2024,
  title={Tennis Serve Analysis Dataset: 3D Keypoint Tracking from US Open 2024},
  author={Tennis Analytics Research Team},
  year={2024},
  url={https://github.com/yourusername/tennis-dataset}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This dataset is licensed under [MIT License](LICENSE).

## 📞 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Last Updated**: July 2024  
**Dataset Version**: 1.0  
**Total Size**: ~1.3GB 