# Tennis Serve Analysis: Dataset & Toolkit

![Tennis Serve Analysis Demo](bounding_grid_8x3.gif)

A comprehensive repository containing both a tennis serve dataset from the 2024 US Open and a complete Python toolkit for analyzing tennis serves using 3D keypoint tracking, biomechanical analysis, and machine learning techniques.

## 🎾 Overview

This repository provides:
- **📊 Complete Dataset**: 6,694 tennis serves with 3D keypoint tracking data
- **🔧 Analysis Toolkit**: Python codebase for biomechanical analysis, gender classification, and data processing
- **📓 Jupyter Notebooks**: Ready-to-use analysis examples and exploratory data analysis
- **🤖 OpenAI Integration**: Advanced analysis tools using OpenAI APIs

## 🗂️ Repository Structure

```
tennis/
├── 📊 DATA & DOCUMENTATION
│   ├── data/                                    # Complete tennis serve dataset
│   │   ├── full/                                # Main dataset files
│   │   │   ├── all_jsons/                       # Original keypoint JSON files (10,727 files)
│   │   │   ├── restructured_jsons/              # Clean keypoint format (10,727 files)
│   │   │   └── usopen_points_clean_keypoints_cleaned_with_server_gender.csv
│   │   ├── scorebug/                            # Scorebug detection data
│   │   └── initial/                             # Initial processing data
│   ├── documentation/                           # Dataset documentation
│   │   ├── data_dictionary.md                   # Column descriptions
│   │   ├── keypoint_mapping.md                  # 17 joint definitions
│   │   └── analysis_examples.md                 # Usage examples
│   ├── DATASET_README.md                        # Dataset-specific documentation
│   ├── setup_dataset.py                         # Dataset setup script
│   └── requirements_dataset.txt                 # Dataset-specific dependencies
├── 🔧 ANALYSIS TOOLKIT
│   ├── code/                                    # Legacy code structure
│   │   ├── src/                                 # Core analysis modules
│   │   ├── notebooks/                           # Jupyter notebooks for analysis
│   │   └── openai/                              # OpenAI API integration
│   ├── src/                                     # Main analysis modules
│   │   ├── coaching/                            # Biomechanical analysis
│   │   ├── gender/                              # Gender classification
│   │   ├── json_investigation/                  # Data processing
│   │   ├── logistics/                           # Data integration
│   │   ├── server/                              # Server analysis
│   │   └── speed/                               # Speed analysis
│   ├── requirements.txt                         # Python dependencies
│   ├── setup.py                                 # Package installation
│   └── __init__.py                              # Package initialization
├── 📋 PROJECT FILES
│   ├── README.md                                # This file
│   ├── LICENSE                                  # MIT License
│   ├── .gitignore                               # Git ignore rules
│   └── GITHUB_SETUP.md                          # GitHub setup instructions
```

## 📊 Dataset Overview

- **Total Serves**: 6,694 serves
- **Players**: 118 unique players (60 male, 58 female)
- **Matches**: 118 different tennis matches
- **Data Sources**: US Open 2024 tournament
- **Keypoint Tracking**: 17 joints per frame, 3D coordinates (x, y, z)
- **Frame Range**: 60-120 frames per serve (mean: 81.2 frames)

### Key Features
- **Player Information**: Server name, gender, match details
- **Serve Metrics**: Speed, direction, outcome
- **3D Keypoint Tracking**: 17 joints with confidence scores
- **Match Context**: Tournament round, court, date

## 🚀 Quick Start

### Using the Dataset
```python
import pandas as pd
import numpy as np

# Load the main dataset
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

### Using the Analysis Toolkit

#### Installation
```bash
# Install toolkit dependencies
pip install -r requirements.txt

# Install as package (for development)
pip install -e .
```

#### Basic Usage
```python
# Import analysis modules
from src.coaching import calculate_angles
from src.gender import gender_classification
from src.logistics import merge_data

# Calculate joint angles
angles = calculate_angles(keypoints_data)

# Perform gender classification
gender_pred = gender_classification(serve_features)

# Process and integrate data
processed_data = merge_data(raw_data, metadata)
```

## 🔧 Analysis Toolkit Modules

### 🎯 Coaching Analysis (`src/coaching/`)
- **Angle Calculations**: Compute joint angles throughout serve motion
- **3D Visualization**: Generate 3D plots and animations
- **Biomechanical Analysis**: Analyze serve mechanics and form
- **Performance Metrics**: Calculate serve efficiency and consistency

### 👥 Gender Classification (`src/gender/`)
- **Machine Learning Models**: Gender classification from serve motion
- **Feature Engineering**: Extract relevant features from 3D keypoints
- **Comparative Analysis**: Compare male vs female serve characteristics
- **Visualization**: Plot gender-specific serve patterns

### 📊 Data Processing (`src/json_investigation/`)
- **JSON Restructuring**: Clean and format keypoint data
- **Data Integration**: Merge multiple data sources
- **Quality Control**: Validate and verify data integrity
- **Metadata Addition**: Add server information and match details

### 🔧 Logistics (`src/logistics/`)
- **Data Integration**: Merge different data sources
- **Sequence Timing**: Analyze serve timing patterns
- **Data Validation**: Ensure data quality and consistency

### 🎾 Server Analysis (`src/server/`)
- **Player Comparison**: Compare different players' serve styles
- **Unsupervised Analysis**: Discover serve patterns and clusters
- **Performance Metrics**: Analyze serve effectiveness

### ⚡ Speed Analysis (`src/speed/`)
- **Serve Speed Calculation**: Compute serve velocities
- **Speed Pattern Analysis**: Analyze speed variations
- **Performance Correlation**: Correlate speed with other metrics

## 📓 Jupyter Notebooks

Explore the `code/notebooks/` directory for:
- **EDA**: Exploratory data analysis
- **Annotation Matching**: Data quality analysis
- **Visualization Examples**: Ready-to-use plotting code

## 🤖 OpenAI Integration

Advanced analysis tools in `code/openai/`:
- **Batch Processing**: Automated analysis workflows
- **Image Processing**: Scorebug detection and analysis
- **Data Management**: Large-scale data processing utilities

## 📊 Sample Analysis Results

### Dataset Statistics
- **Mean Serve Length**: 81.2 frames per serve
- **Frame Range**: 60-120 frames
- **Gender Distribution**: 54.7% Male, 45.3% Female

### Top Players by Serve Count
1. **Jannik Sinner**: 224 serves
2. **Frances Tiafoe**: 221 serves  
3. **Taylor Fritz**: 214 serves
4. **Aryna Sabalenka**: 194 serves
5. **Jessica Pegula**: 169 serves

## 🔬 Research Applications

### Dataset Uses
- **Biomechanics Research**: Serve motion analysis
- **Machine Learning**: Gender classification, serve prediction
- **Sports Analytics**: Performance benchmarking
- **Computer Vision**: 3D pose estimation validation

### Toolkit Applications
- **Coaching**: Biomechanical analysis and feedback
- **Research**: Automated analysis pipelines
- **Development**: Extensible framework for new analyses
- **Education**: Teaching sports analytics and biomechanics

## 📋 Requirements

### Dataset Requirements
```bash
pip install -r requirements_dataset.txt
```

### Toolkit Requirements
```bash
pip install -r requirements.txt
```

## 🛠️ Development

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/tennis-serve-analysis.git
cd tennis-serve-analysis

# Install dataset dependencies
pip install -r requirements_dataset.txt

# Install toolkit dependencies
pip install -r requirements.txt

# Install as development package
pip install -e .
```

### Running Analysis
```bash
# Set up dataset
python setup_dataset.py

# Run example analysis
python -m src.coaching.calculate_angles
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this dataset or toolkit in your research, please cite:

```bibtex
@dataset{tennis_serve_analysis_2024,
  title={Tennis Serve Analysis: Dataset and Toolkit for 3D Keypoint Tracking},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/tennis-serve-analysis}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, collaboration opportunities, or support:
- Open an issue on GitHub
- Email: [your-email@domain.com]

---

**Last Updated**: July 2024  
**Version**: 1.0  
**Dataset Size**: ~1.3GB  
**Python Version**: 3.8+ 