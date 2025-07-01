#!/usr/bin/env python3
"""
Merge JSON keypoints data with CSV match data.

This script:
1. Reads the CSV file with match data
2. For each row, finds the corresponding JSON file (by matching video_name without extension)
3. Extracts keypoints from the JSON file
4. Adds the keypoints data to the CSV row
5. Saves the merged data to data/full/
"""

import os
import json
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm

# File paths
CSV_PATH = "data/scorebug/usopen_points_with_scorebug.csv"
JSON_DIR = "data/full/all_jsons"
OUTPUT_DIR = "data/full"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "usopen_points_with_keypoints.csv")

def get_json_filename_from_video_name(video_name):
    """Convert video_name from CSV to expected JSON filename."""
    # Remove .jpg extension
    if video_name.endswith('.jpg'):
        base_name = video_name[:-4]
    else:
        base_name = video_name
    
    # JSON files have format: {base_name}_{base_name}_trimmed.json
    json_filename = f"{base_name}_{base_name}_trimmed.json"
    return json_filename

def load_json_keypoints(json_path):
    """Load keypoints from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract keypoints from all frames
        keypoints_data = []
        for frame in data:
            frame_id = frame.get('frame_id', 0)
            instances = frame.get('instances', [])
            
            if instances:
                # Take the first instance (assuming one person per frame)
                instance = instances[0]
                keypoints = instance.get('keypoints', [])
                keypoint_scores = instance.get('keypoint_scores', [])
                
                keypoints_data.append({
                    'frame_id': frame_id,
                    'keypoints': keypoints,
                    'keypoint_scores': keypoint_scores
                })
        
        return keypoints_data
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load CSV data
    print("Loading CSV data...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from CSV")
    
    # Get list of all JSON files
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    json_filenames = {os.path.basename(f): f for f in json_files}
    print(f"Found {len(json_files)} JSON files")
    
    # Add new columns for keypoints data
    df['keypoints_data'] = None
    df['json_file_found'] = False
    df['json_file_path'] = None
    
    # Process each row
    print("Processing rows and matching JSON files...")
    matched_count = 0
    not_found_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        video_name = row['video_name']
        json_filename = get_json_filename_from_video_name(video_name)
        
        # Check if JSON file exists
        if json_filename in json_filenames:
            json_path = json_filenames[json_filename]
            keypoints_data = load_json_keypoints(json_path)
            
            if keypoints_data is not None:
                df.at[idx, 'keypoints_data'] = json.dumps(keypoints_data)
                df.at[idx, 'json_file_found'] = True
                df.at[idx, 'json_file_path'] = json_path
                matched_count += 1
            else:
                not_found_count += 1
        else:
            not_found_count += 1
    
    # Save the merged data
    print(f"Saving merged data to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print("\n=== MERGE SUMMARY ===")
    print(f"Total CSV rows: {len(df)}")
    print(f"Successfully matched: {matched_count}")
    print(f"Not found: {not_found_count}")
    print(f"Match rate: {matched_count/len(df)*100:.2f}%")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Show some sample data
    print("\n=== SAMPLE DATA ===")
    sample_matched = df[df['json_file_found'] == True].head(3)
    for idx, row in sample_matched.iterrows():
        print(f"Row {idx}: {row['video_name']}")
        print(f"  JSON file: {row['json_file_path']}")
        keypoints_data = json.loads(row['keypoints_data'])
        print(f"  Frames: {len(keypoints_data)}")
        if keypoints_data:
            print(f"  First frame keypoints: {len(keypoints_data[0]['keypoints'])} points")
        print()

if __name__ == "__main__":
    main() 