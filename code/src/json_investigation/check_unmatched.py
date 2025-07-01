#!/usr/bin/env python3
"""
Check which rows were not matched in the merge.
"""

import pandas as pd

# Load the merged data
df = pd.read_csv("data/full/usopen_points_with_keypoints.csv")

# Find unmatched rows
unmatched = df[df['json_file_found'] == False]

print(f"Total rows: {len(df)}")
print(f"Matched rows: {len(df[df['json_file_found'] == True])}")
print(f"Unmatched rows: {len(unmatched)}")

if len(unmatched) > 0:
    print("\nUnmatched video_names:")
    for idx, row in unmatched.iterrows():
        print(f"Row {idx}: {row['video_name']}")
        
        # Try to find similar files
        import glob
        import os
        
        base_name = row['video_name'].replace('.jpg', '')
        json_files = glob.glob(f"data/full/all_jsons/*{base_name}*")
        
        if json_files:
            print(f"  Similar JSON files found:")
            for f in json_files[:3]:  # Show first 3
                print(f"    {os.path.basename(f)}")
        else:
            print(f"  No similar JSON files found")
        print() 