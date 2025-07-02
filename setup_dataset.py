#!/usr/bin/env python3
"""
Setup script for Tennis Serve Analysis Dataset.

This script helps users set up the dataset repository and verify their installation.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'scikit-learn', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - NOT INSTALLED")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ All required packages are installed!")
    return True

def verify_dataset():
    """Verify that the main dataset file exists and is accessible."""
    dataset_path = "data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure the dataset file is in the correct location.")
        return False
    
    try:
        # Try to load a small sample
        df_sample = pd.read_csv(dataset_path, nrows=5)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Shape: {df_sample.shape}")
        print(f"   Columns: {len(df_sample.columns)}")
        
        # Check for key columns
        key_columns = ['server_name', 'server_gender', 'n_frames', 'keypoints_clean']
        missing_columns = [col for col in key_columns if col not in df_sample.columns]
        
        if missing_columns:
            print(f"⚠️  Missing key columns: {missing_columns}")
            return False
        
        print(f"✅ All key columns present!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_keypoint_loading():
    """Test loading keypoints from the dataset."""
    dataset_path = "data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv"
    
    try:
        # Load first row
        df = pd.read_csv(dataset_path, nrows=1)
        row = df.iloc[0]
        
        # Load keypoints
        keypoints_str = row['keypoints_clean']
        scores_str = row['keypoint_scores_clean']
        
        keypoints = np.array(json.loads(keypoints_str))
        scores = np.array(json.loads(scores_str))
        
        print(f"✅ Keypoints loaded successfully!")
        print(f"   Keypoints shape: {keypoints.shape}")
        print(f"   Scores shape: {scores.shape}")
        print(f"   Number of frames: {row['n_frames']}")
        print(f"   Server: {row['server_name']} ({row['server_gender']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading keypoints: {e}")
        return False

def show_dataset_stats():
    """Show basic statistics about the dataset."""
    dataset_path = "data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        
        print("\n📊 DATASET STATISTICS")
        print("=" * 50)
        print(f"Total serves: {len(df):,}")
        print(f"Unique players: {df['server_name'].nunique()}")
        print(f"Unique matches: {df['video_name'].nunique()}")
        
        # Gender distribution
        gender_dist = df['server_gender'].value_counts()
        print(f"\nGender distribution:")
        for gender, count in gender_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {gender}: {count:,} serves ({percentage:.1f}%)")
        
        # Frame statistics
        frame_stats = df['n_frames'].describe()
        print(f"\nFrame count statistics:")
        print(f"  Mean: {frame_stats['mean']:.1f}")
        print(f"  Std: {frame_stats['std']:.1f}")
        print(f"  Min: {frame_stats['min']:.0f}")
        print(f"  Max: {frame_stats['max']:.0f}")
        
        # Top players
        top_players = df['server_name'].value_counts().head(5)
        print(f"\nTop 5 players by serve count:")
        for i, (player, count) in enumerate(top_players.items(), 1):
            print(f"  {i}. {player}: {count} serves")
        
        return True
        
    except Exception as e:
        print(f"❌ Error showing statistics: {e}")
        return False

def create_sample_analysis():
    """Create a sample analysis to demonstrate usage."""
    print("\n🔬 SAMPLE ANALYSIS")
    print("=" * 50)
    
    try:
        # Load dataset
        df = pd.read_csv("data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv")
        
        # Basic gender comparison
        gender_comparison = df.groupby('server_gender')['n_frames'].agg(['mean', 'std', 'count'])
        print("Frame count comparison by gender:")
        print(gender_comparison.round(2))
        
        # Load keypoints for first serve
        first_serve = df.iloc[0]
        keypoints = np.array(json.loads(first_serve['keypoints_clean']))
        
        print(f"\nSample serve analysis:")
        print(f"  Player: {first_serve['server_name']}")
        print(f"  Gender: {first_serve['server_gender']}")
        print(f"  Frames: {first_serve['n_frames']}")
        print(f"  Keypoints shape: {keypoints.shape}")
        
        # Calculate shoulder width (example biomechanical measure)
        shoulder_width = np.linalg.norm(keypoints[:, 5, :] - keypoints[:, 6, :], axis=1)
        avg_shoulder_width = np.mean(shoulder_width)
        print(f"  Average shoulder width: {avg_shoulder_width:.3f}")
        
        print("\n✅ Sample analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in sample analysis: {e}")
        return False

def main():
    """Main setup function."""
    print("🎾 TENNIS SERVE ANALYSIS DATASET SETUP")
    print("=" * 60)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    # Verify dataset
    print("\n2. Verifying dataset...")
    dataset_ok = verify_dataset()
    
    # Test keypoint loading
    print("\n3. Testing keypoint loading...")
    keypoints_ok = test_keypoint_loading()
    
    # Show statistics
    print("\n4. Loading dataset statistics...")
    stats_ok = show_dataset_stats()
    
    # Sample analysis
    print("\n5. Running sample analysis...")
    analysis_ok = create_sample_analysis()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    checks = [
        ("Dependencies", deps_ok),
        ("Dataset file", dataset_ok),
        ("Keypoint loading", keypoints_ok),
        ("Statistics", stats_ok),
        ("Sample analysis", analysis_ok)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 SETUP COMPLETE!")
        print("You're ready to start analyzing the tennis serve dataset!")
        print("\nNext steps:")
        print("  1. Read the documentation in the 'documentation/' folder")
        print("  2. Try the analysis examples in 'documentation/analysis_examples.md'")
        print("  3. Explore the Jupyter notebooks in 'code/notebooks/'")
        print("  4. Check out the processing scripts in 'code/src/'")
    else:
        print("\n⚠️  SETUP INCOMPLETE")
        print("Please address the failed checks above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 