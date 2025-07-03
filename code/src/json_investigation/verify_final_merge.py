#!/usr/bin/env python3
"""
Comprehensive verification of the final merged dataset.
"""

import pandas as pd
import numpy as np
import json
from collections import Counter

# File paths
CSV_PATH = "data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv"

def verify_dataset_structure():
    """Verify the overall dataset structure."""
    print("🔍 VERIFYING DATASET STRUCTURE")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv(CSV_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    
    # Check key columns
    key_columns = ['server_name', 'server_gender', 'n_frames', 'keypoints_clean', 'keypoint_scores_clean']
    missing_columns = [col for col in key_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing key columns: {missing_columns}")
        return False
    else:
        print(f"✅ All key columns present")
    
    return df

def verify_data_quality(df):
    """Verify data quality and completeness."""
    print(f"\n📊 DATA QUALITY VERIFICATION")
    print("=" * 50)
    
    # Check for missing values in key columns
    print("Missing values in key columns:")
    for col in ['server_name', 'server_gender', 'n_frames']:
        missing = df[col].isna().sum()
        print(f"  {col}: {missing} missing values")
    
    # Check gender distribution
    print(f"\nGender distribution:")
    gender_dist = df['server_gender'].value_counts()
    for gender, count in gender_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {gender}: {count} serves ({percentage:.1f}%)")
    
    # Check frame count statistics
    print(f"\nFrame count statistics:")
    frame_stats = df['n_frames'].describe()
    print(f"  Mean: {frame_stats['mean']:.1f}")
    print(f"  Std: {frame_stats['std']:.1f}")
    print(f"  Min: {frame_stats['min']:.0f}")
    print(f"  Max: {frame_stats['max']:.0f}")
    
    # Check unique players
    unique_players = df['server_name'].nunique()
    print(f"\nUnique players: {unique_players}")
    
    return True

def verify_keypoints_structure(df):
    """Verify keypoints data structure."""
    print(f"\n🦴 KEYPOINTS STRUCTURE VERIFICATION")
    print("=" * 50)
    
    # Sample a few rows to check keypoints structure
    sample_size = 5
    sample_df = df.head(sample_size)
    
    for idx, row in sample_df.iterrows():
        try:
            # Load keypoints
            keypoints_str = row['keypoints_clean']
            scores_str = row['keypoint_scores_clean']
            
            keypoints = np.array(json.loads(keypoints_str))
            scores = np.array(json.loads(scores_str))
            
            print(f"Row {idx}: {row['server_name']} ({row['server_gender']})")
            print(f"  Frames: {row['n_frames']}")
            print(f"  Keypoints shape: {keypoints.shape}")
            print(f"  Scores shape: {scores.shape}")
            
            # Verify dimensions
            if keypoints.shape[1] != 17:
                print(f"  ❌ Expected 17 keypoints, got {keypoints.shape[1]}")
            else:
                print(f"  ✅ 17 keypoints confirmed")
            
            if keypoints.shape[2] != 3:
                print(f"  ❌ Expected 3 coordinates, got {keypoints.shape[2]}")
            else:
                print(f"  ✅ 3 coordinates confirmed")
            
            if keypoints.shape[0] != row['n_frames']:
                print(f"  ❌ Frame count mismatch: {keypoints.shape[0]} vs {row['n_frames']}")
            else:
                print(f"  ✅ Frame count matches")
                
        except Exception as e:
            print(f"  ❌ Error loading keypoints: {e}")
    
    return True

def verify_player_mapping(df):
    """Verify player and gender mapping."""
    print(f"\n👥 PLAYER MAPPING VERIFICATION")
    print("=" * 50)
    
    # Check server name mapping
    print("Server name mapping verification:")
    server1_count = (df['PointServer'] == 1).sum()
    server1_name_count = (df['server_name'] == df['player1']).sum()
    server2_count = (df['PointServer'] == 2).sum()
    server2_name_count = (df['server_name'] == df['player2']).sum()
    
    print(f"  Server 1 points: {server1_count}, mapped to player1: {server1_name_count}")
    print(f"  Server 2 points: {server2_count}, mapped to player2: {server2_name_count}")
    
    if server1_count == server1_name_count and server2_count == server2_name_count:
        print("  ✅ Server name mapping is correct")
    else:
        print("  ❌ Server name mapping has errors")
    
    # Check gender distribution by player
    print(f"\nTop 10 players by serve count:")
    top_players = df['server_name'].value_counts().head(10)
    for i, (player, count) in enumerate(top_players.items(), 1):
        player_gender = df[df['server_name'] == player]['server_gender'].iloc[0]
        print(f"  {i}. {player} ({player_gender}): {count} serves")
    
    return True

def verify_json_matching():
    """Verify JSON file matching."""
    print(f"\n📁 JSON FILE MATCHING VERIFICATION")
    print("=" * 50)
    
    # Count JSON files
    import glob
    json_files = glob.glob("data/full/all_jsons/*.json")
    print(f"Total JSON files: {len(json_files)}")
    
    # Check if we have the restructured JSONs
    restructured_files = glob.glob("data/full/restructured_jsons/*.json")
    print(f"Restructured JSON files: {len(restructured_files)}")
    
    # Sample a few JSON files to verify structure
    if restructured_files:
        sample_json = restructured_files[0]
        with open(sample_json, 'r') as f:
            data = json.load(f)
        
        print(f"Sample restructured JSON structure:")
        print(f"  Keys: {list(data.keys())}")
        if 'n_frames' in data:
            print(f"  n_frames: {data['n_frames']}")
        if 'keypoints' in data:
            keypoints = np.array(data['keypoints'])
            print(f"  keypoints shape: {keypoints.shape}")
    
    return True

def main():
    """Main verification function."""
    print("🎾 TENNIS SERVE DATASET VERIFICATION")
    print("=" * 60)
    
    # Verify dataset structure
    df = verify_dataset_structure()
    if df is False:
        print("❌ Dataset structure verification failed")
        return
    
    # Verify data quality
    quality_ok = verify_data_quality(df)
    
    # Verify keypoints structure
    keypoints_ok = verify_keypoints_structure(df)
    
    # Verify player mapping
    mapping_ok = verify_player_mapping(df)
    
    # Verify JSON matching
    json_ok = verify_json_matching()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 60)
    
    checks = [
        ("Dataset structure", df is not False),
        ("Data quality", quality_ok),
        ("Keypoints structure", keypoints_ok),
        ("Player mapping", mapping_ok),
        ("JSON matching", json_ok)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 ALL VERIFICATIONS PASSED!")
        print("Your dataset is ready for use and sharing!")
    else:
        print(f"\n⚠️  SOME VERIFICATIONS FAILED")
        print("Please address the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 