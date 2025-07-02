#!/usr/bin/env python3
"""
Restructure keypoints data to clean format.

This script:
1. Restructures JSON files from nested format to n_frames × 17 joints × 3 coordinates
2. Updates the CSV to have clean keypoints columns
3. Creates both restructured JSON files and updated CSV
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

# File paths
JSON_DIR = "data/full/all_jsons"
CSV_PATH = "data/full/usopen_points_with_keypoints.csv"
OUTPUT_JSON_DIR = "data/full/restructured_jsons"
OUTPUT_CSV_PATH = "data/full/usopen_points_clean_keypoints.csv"

# Create output directory
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

def restructure_json_file(json_path, output_path):
    """
    Restructure a single JSON file from nested format to clean format.
    
    Input format: [{"frame_id": 0, "instances": [{"keypoints": [[x,y,z], ...], "keypoint_scores": [...]}]}, ...]
    Output format: {"n_frames": N, "keypoints": [[[x,y,z], ...], ...], "keypoint_scores": [[...], ...]}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    n_frames = len(data)
    keypoints = []
    keypoint_scores = []
    
    for frame in data:
        # Extract keypoints and scores from the first instance
        if frame['instances'] and len(frame['instances']) > 0:
            instance = frame['instances'][0]
            keypoints.append(instance['keypoints'])
            keypoint_scores.append(instance['keypoint_scores'])
        else:
            # Handle missing instances
            keypoints.append([[0.0, 0.0, 0.0]] * 17)
            keypoint_scores.append([0.0] * 17)
    
    # Convert to numpy arrays for easier handling
    keypoints = np.array(keypoints)  # Shape: (n_frames, 17, 3)
    keypoint_scores = np.array(keypoint_scores)  # Shape: (n_frames, 17)
    
    # Create clean structure
    clean_data = {
        "n_frames": n_frames,
        "keypoints": keypoints.tolist(),
        "keypoint_scores": keypoint_scores.tolist(),
        "original_filename": os.path.basename(json_path)
    }
    
    # Save restructured JSON
    with open(output_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    return clean_data

def restructure_all_jsons():
    """Restructure all JSON files in the directory."""
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, '*.json')))
    print(f"Restructuring {len(json_files)} JSON files...")
    
    restructured_data = {}
    
    for json_path in tqdm(json_files, desc="Restructuring JSONs"):
        # Get base filename without extension for matching
        base_name = os.path.basename(json_path).replace('.json', '')
        
        # Create output path
        output_path = os.path.join(OUTPUT_JSON_DIR, f"{base_name}_restructured.json")
        
        try:
            clean_data = restructure_json_file(json_path, output_path)
            restructured_data[base_name] = clean_data
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
    
    return restructured_data

def update_csv_with_clean_keypoints():
    """Update the CSV to have clean keypoints columns."""
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Create new columns for clean keypoints
    df['n_frames'] = None
    df['keypoints_clean'] = None
    df['keypoint_scores_clean'] = None
    
    print("Updating CSV with clean keypoints...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating CSV"):
        if row['json_file_found']:
            try:
                # Get the restructured JSON data
                json_filename = row['json_file_path']
                # Extract base filename from full path
                base_name = os.path.basename(json_filename).replace('.json', '')
                restructured_path = os.path.join(OUTPUT_JSON_DIR, f"{base_name}_restructured.json")
                
                if os.path.exists(restructured_path):
                    with open(restructured_path, 'r') as f:
                        clean_data = json.load(f)
                    
                    df.at[idx, 'n_frames'] = clean_data['n_frames']
                    df.at[idx, 'keypoints_clean'] = json.dumps(clean_data['keypoints'])
                    df.at[idx, 'keypoint_scores_clean'] = json.dumps(clean_data['keypoint_scores'])
                
            except Exception as e:
                print(f"Error updating row {idx}: {e}")
    
    # Save updated CSV
    print(f"Saving updated CSV to {OUTPUT_CSV_PATH}")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    return df

def create_summary():
    """Create a summary of the restructuring."""
    print("\n=== RESTRUCTURING SUMMARY ===")
    
    # Count restructured JSONs
    restructured_jsons = glob.glob(os.path.join(OUTPUT_JSON_DIR, '*_restructured.json'))
    print(f"Restructured JSON files: {len(restructured_jsons)}")
    
    # Load updated CSV
    if os.path.exists(OUTPUT_CSV_PATH):
        df = pd.read_csv(OUTPUT_CSV_PATH)
        print(f"Updated CSV rows: {len(df)}")
        print(f"Rows with clean keypoints: {df['n_frames'].notna().sum()}")
        
        # Sample statistics
        if df['n_frames'].notna().sum() > 0:
            print(f"Average frames per serve: {df['n_frames'].mean():.1f}")
            print(f"Frame range: {df['n_frames'].min()} - {df['n_frames'].max()}")
    
    print(f"\nOutput files:")
    print(f"- Restructured JSONs: {OUTPUT_JSON_DIR}")
    print(f"- Updated CSV: {OUTPUT_CSV_PATH}")

def main():
    """Main function to run the restructuring."""
    print("Starting keypoints restructuring...")
    
    # Step 1: Restructure all JSON files
    restructured_data = restructure_all_jsons()
    
    # Step 2: Update CSV with clean keypoints
    updated_df = update_csv_with_clean_keypoints()
    
    # Step 3: Create summary
    create_summary()
    
    print("\n✅ Restructuring complete!")
    print("\nNew data format:")
    print("- JSON: {'n_frames': N, 'keypoints': [[[x,y,z], ...], ...], 'keypoint_scores': [[...], ...]}")
    print("- CSV: Added 'n_frames', 'keypoints_clean', 'keypoint_scores_clean' columns")

if __name__ == "__main__":
    main() 