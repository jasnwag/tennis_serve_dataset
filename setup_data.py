#!/usr/bin/env python3
"""
Data Setup Script for Tennis Analysis Project

This script helps set up the data directory structure and provides utilities
for managing data across different computers.
"""

import os
import shutil
import argparse
from pathlib import Path

def create_data_structure():
    """Create the standard data directory structure."""
    data_dirs = [
        "data/initial/angles",
        "data/initial/pca", 
        "data/initial/raw",
        "data/initial/visualizations",
        "data/scorebug/batch_1",
        "data/scorebug/batch_2",
        "data/scorebug/combined_batches",
        "data/scorebug/positive_instances",
        "data/scorebug/positive_scorebugs",
        "data/scorebug/positive_scorebug_chunks",
        "data/scorebug/scorebug_test",
        "data/USTA/angle_analysis",
        "data/USTA/predictions",
        "data/USTA/visualizations",
        "data/visualizations"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_data_info_file():
    """Create a data info file with instructions."""
    info_content = """# Data Directory Information

This directory contains the data files for the tennis analysis project.

## Directory Structure:
- initial/: Initial processed data including angles, PCA results, and raw data
- scorebug/: Scorebug detection data and results
- USTA/: USTA-specific analysis data
- visualizations/: Generated visualizations and plots

## Data Transfer Instructions:

### Option 1: Cloud Storage
1. Upload data to Google Drive, Dropbox, or OneDrive
2. Download on the target computer
3. Extract to this data/ directory

### Option 2: External Drive
1. Copy data/ directory to external drive
2. Transfer to target computer
3. Copy to project root

### Option 3: Network Transfer
Use rsync or scp:
```bash
rsync -avz /path/to/source/data/ user@target:/path/to/project/data/
```

### Option 4: Git LFS (for smaller datasets)
1. Add data files to Git LFS
2. Push to repository
3. Pull on target computer

## File Types:
- .mp4: Video files
- .npy: NumPy arrays
- .json: JSON data files
- .csv: CSV data files
- .pkl: Pickle files
- .jpg/.png: Image files

## Expected Data Files:
- data/initial/angles/joint_angles_gender.npy
- data/initial/pca/pca_data_removed_errors.json
- data/initial/raw/all_matches_data.npy
- data/scorebug/positive_instances/ (video files)
- data/scorebug/positive_scorebugs/ (image files)
- data/USTA/ (USTA-specific data)

## Notes:
- Large video files are stored in scorebug/ directories
- Processed data is stored in initial/ directories
- Visualizations are generated and stored in visualizations/ directories
"""
    
    with open("data/README.md", "w") as f:
        f.write(info_content)
    print("Created data/README.md with transfer instructions")

def check_data_integrity():
    """Check if expected data files exist."""
    expected_files = [
        "data/initial/angles/joint_angles_gender.npy",
        "data/initial/pca/pca_data_removed_errors.json", 
        "data/initial/raw/all_matches_data.npy"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing expected data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease transfer the data files to continue with analysis.")
    else:
        print("All expected data files found!")

def main():
    parser = argparse.ArgumentParser(description="Setup data directory structure for tennis analysis project")
    parser.add_argument("--create-structure", action="store_true", 
                       help="Create the data directory structure")
    parser.add_argument("--check-integrity", action="store_true",
                       help="Check if expected data files exist")
    parser.add_argument("--all", action="store_true",
                       help="Run all setup tasks")
    
    args = parser.parse_args()
    
    if args.all or args.create_structure:
        print("Creating data directory structure...")
        create_data_structure()
        create_data_info_file()
        print("\nData directory structure created successfully!")
    
    if args.all or args.check_integrity:
        print("\nChecking data integrity...")
        check_data_integrity()
    
    if not any([args.create_structure, args.check_integrity, args.all]):
        print("No action specified. Use --help for options.")
        print("Recommended: python setup_data.py --all")

if __name__ == "__main__":
    main() 