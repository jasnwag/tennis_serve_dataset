#!/usr/bin/env python3
"""
Process MMPose JSON outputs to calculate joint angles.
"""

import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def calculate_angle(A, B, C):
    """
    Calculate the angle between three points.
    
    Args:
        A, B, C: 2D or 3D points (numpy arrays)
        
    Returns:
        angle in degrees (180 - degrees between vectors)
    """
    # Convert to numpy arrays to ensure consistent handling
    A = np.array(A[:2])  # Use only x,y coordinates
    B = np.array(B[:2])
    C = np.array(C[:2])
    
    AB = B - A 
    CB = B - C
    dot_product = np.dot(AB, CB)
    mag_AB = np.linalg.norm(AB)
    mag_CB = np.linalg.norm(CB)
    
    if mag_AB == 0 or mag_CB == 0:
        return None
    
    # Handle potential numerical errors that could make cos_theta out of bounds
    cos_theta = max(-1.0, min(1.0, dot_product / (mag_AB * mag_CB)))
    theta = np.arccos(cos_theta)
    degrees = np.degrees(theta)
    
    return 180 - degrees

def process_json_file(json_path):
    """
    Process MMPose JSON file and calculate joint angles.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        DataFrame with frame_id and calculated angles
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Define joint connections for calculating angles
    # Based on MMPose COCO 17-keypoint format:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    angles_config = {
        'left_elbow': [11, 12, 13],    # Left shoulder, elbow, wrist
        'right_elbow': [14, 15, 16],   # Right shoulder, elbow, wrist  
        'left_shoulder': [12, 11, 4],   # Left elbow, shoulder, hip
        'right_shoulder': [15, 14, 1],  # Right elbow, shoulder, hip
        'left_hip': [11, 4, 5],        # Left shoulder, hip, knee
        'right_hip': [14, 1, 2],       # Right shoulder, hip, knee
        'left_knee': [4, 5, 6],        # Left hip, knee, ankle
        'right_knee': [1, 2, 3]        # Right hip, knee, ankle
    }
    # Initialize results
    results = []
    
    # Process each frame
    for frame in data:
        frame_id = frame['frame_id']
        
        # Skip frames with no instances
        if not frame.get('instances'):
            continue
            
        # Use first person if multiple people detected
        instance = frame['instances'][0]
        keypoints = instance['keypoints']
        
        # Calculate angles for this frame
        frame_angles = {'frame_id': frame_id}
        
        for angle_name, indices in angles_config.items():
            # Get the three points needed for this angle
            points = [keypoints[i] for i in indices]
            
            # Only calculate angle if all points have reasonable confidence
            # (third value in each keypoint is confidence)
            if all(point[2] > 0.3 for point in points):
                angle = calculate_angle(points[0], points[1], points[2])
                frame_angles[angle_name] = angle
            else:
                frame_angles[angle_name] = None
        
        results.append(frame_angles)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    return df

def plot_angles(df, output_dir=None):
    """
    Plot the angles over time.
    
    Args:
        df: DataFrame with frame_id and calculated angles
        output_dir: Directory to save plots (if None, just display)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    angle_columns = [col for col in df.columns if col != 'frame_id']
    
    # Create plots
    for angle_name in angle_columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df['frame_id'], df[angle_name])
        plt.title(f'{angle_name} angle over time')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'{angle_name}.png'))
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process MMPose JSON files to calculate joint angles')
    parser.add_argument('json_path', help='Path to the MMPose JSON file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--plot', '-p', help='Directory to save angle plots')
    
    args = parser.parse_args()
    
    # Process the JSON file
    df = process_json_file(args.json_path)
    
    # Save the results to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    
    # Generate plots if requested
    if args.plot:
        plot_angles(df, args.plot)
        print(f"Plots saved to {args.plot}")
    
    # Print summary statistics
    print("\nSummary statistics for joint angles:")
    for col in df.columns:
        if col != 'frame_id':
            print(f"{col}:")
            print(f"  Mean: {df[col].mean():.2f} degrees")
            print(f"  Min: {df[col].min():.2f} degrees")
            print(f"  Max: {df[col].max():.2f} degrees")
            print()
    
if __name__ == "__main__":
    main() 