#!/usr/bin/env python3
"""
Update CSV with clean keypoints data (JSON restructuring already done).
"""

import os
import json
import pandas as pd
from tqdm import tqdm

# File paths
CSV_PATH = "data/full/usopen_points_with_keypoints.csv"
OUTPUT_JSON_DIR = "data/full/restructured_jsons"
OUTPUT_CSV_PATH = "data/full/usopen_points_clean_keypoints.csv"

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
    import glob
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
    """Main function to run the CSV update."""
    print("Starting CSV update with clean keypoints...")
    
    # Update CSV with clean keypoints
    updated_df = update_csv_with_clean_keypoints()
    
    # Create summary
    create_summary()
    
    print("\n✅ CSV update complete!")

if __name__ == "__main__":
    main() 