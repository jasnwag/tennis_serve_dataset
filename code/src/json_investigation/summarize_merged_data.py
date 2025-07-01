#!/usr/bin/env python3
"""
Summarize the merged data with keypoints.
"""

import pandas as pd
import json
import numpy as np

# Load the merged data
print("Loading merged data...")
df = pd.read_csv("data/full/usopen_points_with_keypoints.csv")

print(f"\n=== MERGED DATA SUMMARY ===")
print(f"Total rows: {len(df)}")
print(f"Matched rows: {len(df[df['json_file_found'] == True])}")
print(f"Unmatched rows: {len(df[df['json_file_found'] == False])}")
print(f"Match rate: {len(df[df['json_file_found'] == True])/len(df)*100:.2f}%")

# Analyze keypoints data
matched_df = df[df['json_file_found'] == True]

print(f"\n=== KEYPOINTS ANALYSIS ===")
frame_counts = []
keypoint_counts = []

for idx, row in matched_df.head(100).iterrows():  # Sample first 100 for analysis
    try:
        keypoints_data = json.loads(row['keypoints_data'])
        frame_counts.append(len(keypoints_data))
        
        if keypoints_data:
            # Count keypoints in first frame
            first_frame = keypoints_data[0]
            keypoint_counts.append(len(first_frame['keypoints']))
    except:
        continue

if frame_counts:
    print(f"Frame count statistics (sample of 100):")
    print(f"  Min frames: {min(frame_counts)}")
    print(f"  Max frames: {max(frame_counts)}")
    print(f"  Mean frames: {np.mean(frame_counts):.1f}")
    print(f"  Median frames: {np.median(frame_counts):.1f}")

if keypoint_counts:
    print(f"\nKeypoint count statistics (sample of 100):")
    print(f"  Min keypoints: {min(keypoint_counts)}")
    print(f"  Max keypoints: {max(keypoint_counts)}")
    print(f"  Mean keypoints: {np.mean(keypoint_counts):.1f}")
    print(f"  Most common: {max(set(keypoint_counts), key=keypoint_counts.count)}")

# Show sample of keypoints structure
print(f"\n=== SAMPLE KEYPOINTS STRUCTURE ===")
sample_row = matched_df.iloc[0]
print(f"Video: {sample_row['video_name']}")
print(f"Match: {sample_row['player1']} vs {sample_row['player2']}")
print(f"Point: {sample_row['PointNumber']}")

try:
    keypoints_data = json.loads(sample_row['keypoints_data'])
    print(f"Frames: {len(keypoints_data)}")
    
    if keypoints_data:
        first_frame = keypoints_data[0]
        print(f"First frame ID: {first_frame['frame_id']}")
        print(f"Keypoints: {len(first_frame['keypoints'])} points")
        print(f"Keypoint scores: {len(first_frame['keypoint_scores'])} scores")
        
        # Show first few keypoints
        print(f"\nFirst 3 keypoints (x, y, z):")
        for i in range(min(3, len(first_frame['keypoints']))):
            kp = first_frame['keypoints'][i]
            score = first_frame['keypoint_scores'][i] if i < len(first_frame['keypoint_scores']) else 'N/A'
            print(f"  Point {i}: ({kp[0]:.2f}, {kp[1]:.2f}, {kp[2]:.2f}) - Score: {score}")
            
except Exception as e:
    print(f"Error parsing keypoints: {e}")

# Show unique matches
print(f"\n=== UNIQUE MATCHES ===")
unique_matches = df[['player1', 'player2', 'match_id']].drop_duplicates()
print(f"Total unique matches: {len(unique_matches)}")

print(f"\nFirst 10 matches:")
for idx, row in unique_matches.head(10).iterrows():
    print(f"  {row['match_id']}: {row['player1']} vs {row['player2']}")

print(f"\n=== OUTPUT FILE INFO ===")
print(f"File: data/full/usopen_points_with_keypoints.csv")
print(f"Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}") 