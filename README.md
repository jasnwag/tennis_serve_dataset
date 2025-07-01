# Tennis Analysis Project

A comprehensive analysis project for tennis match data, including serve analysis, gender classification, coaching insights, and scorebug detection.

## Project Structure

```
tennis/
├── code/
│   ├── notebooks/           # Jupyter notebooks for analysis
│   │   ├── annotation_matching/
│   │   ├── EDA/
│   │   └── filename_counter.ipynb
│   ├── openai/             # OpenAI API integration scripts
│   │   ├── batch/          # Batch processing scripts
│   │   ├── downscale/      # Image downscaling utilities
│   │   ├── file_logistics/ # File management utilities
│   │   ├── split/          # Image splitting utilities
│   │   └── test_and_visualize/ # Testing and visualization scripts
│   └── src/                # Main source code
│       ├── coaching/       # Coaching analysis tools
│       ├── gender/         # Gender classification analysis
│       ├── logistics/      # Data logistics and integration
│       ├── server/         # Server analysis tools
│       └── speed/          # Speed analysis tools
├── data/                   # Data directory (not included in repo)
│   ├── initial/           # Initial processed data
│   ├── scorebug/          # Scorebug detection data
│   ├── USTA/              # USTA-specific data
│   └── visualizations/    # Generated visualizations
└── requirements.txt       # Python dependencies
```

## Features

### 1. Serve Analysis
- Joint angle calculations and analysis
- Serve motion comparison tools
- Coaching insights and recommendations

### 2. Gender Classification
- Machine learning models for gender classification
- Analysis of serve patterns by gender
- Visualization of gender-specific serve characteristics

### 3. Scorebug Detection
- Automated scorebug detection in tennis videos
- Batch processing capabilities
- Image preprocessing and downscaling utilities

### 4. Data Processing
- Comprehensive data integration tools
- Sequence timing analysis
- File logistics and management utilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tennis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Analysis Scripts

Navigate to the specific analysis directory and run the scripts:

```bash
# For coaching analysis
cd code/src/coaching
python calculate_angles.py

# For gender classification
cd code/src/gender
python gender_classification_analysis.py

# For server analysis
cd code/src/server
python server_comparison.py
```

### Jupyter Notebooks

Launch Jupyter and explore the notebooks in `code/notebooks/`:

```bash
jupyter lab
```

## Data Management

**Important**: The `data/` directory is not included in this repository due to size constraints and privacy considerations.

### Setting Up Data on Another Computer

1. **Create the data directory structure**:
```bash
mkdir -p data/{initial,scorebug,USTA,visualizations}
```

2. **Transfer data files** using one of these methods:
   - **Cloud Storage**: Upload to Google Drive, Dropbox, or OneDrive
   - **External Drive**: Copy to USB/external hard drive
   - **Network Transfer**: Use SCP, rsync, or similar tools
   - **Git LFS**: For smaller datasets, consider Git Large File Storage

3. **Recommended data organization**:
```
data/
├── initial/
│   ├── angles/
│   ├── pca/
│   ├── raw/
│   └── visualizations/
├── scorebug/
│   ├── batch_1/
│   ├── batch_2/
│   ├── positive_instances/
│   └── positive_scorebugs/
├── USTA/
│   ├── angle_analysis/
│   ├── predictions/
│   └── visualizations/
└── visualizations/
```

### Data Access Scripts

The project includes utilities in `code/openai/file_logistics/` for managing data files and ensuring proper file paths across different systems.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Contact

[Add your contact information here] 