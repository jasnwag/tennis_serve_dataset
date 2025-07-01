#!/usr/bin/env python3
"""
Visualize joint angles on video frames.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import os
from tqdm import tqdm
from calculate_angles import calculate_angle

# COCO keypoints connections for visualization
SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Face
    [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
    [5, 6], [5, 11], [6, 12], [11, 12],  # Torso
    [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
]

# Colors for visualization
COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green 
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 165, 0),  # Orange
    (128, 0, 128)   # Purple
]

def draw_skeleton(img, keypoints, thickness=2):
    """
    Draw the skeleton on the image.
    
    Args:
        img: OpenCV image
        keypoints: List of keypoints (x, y, confidence)
        thickness: Line thickness
    """
    h, w = img.shape[:2]
    
    # Scale keypoints from normalized coords to image coords
    scaled_keypoints = []
    for kp in keypoints:
        # Convert from [-0.5, 0.5] to [0, w] for x and [0, h] for y
        x = int((kp[0] + 0.5) * w)
        y = int((kp[1] + 0.5) * h)
        scaled_keypoints.append((x, y, kp[2]))
    
    # Draw skeleton connections
    for i, connection in enumerate(SKELETON):
        idx1, idx2 = connection
        
        # Skip if either keypoint has low confidence
        if keypoints[idx1][2] < 0.3 or keypoints[idx2][2] < 0.3:
            continue
        
        x1, y1, _ = scaled_keypoints[idx1]
        x2, y2, _ = scaled_keypoints[idx2]
        
        # Draw line
        color = COLORS[i % len(COLORS)]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw keypoints
    for x, y, conf in scaled_keypoints:
        if conf > 0.3:  # Only draw keypoints with sufficient confidence
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
    
    return img

def draw_angle(img, keypoints, angle_indices, angle_value, label, color=(255, 255, 255), thickness=2):
    """
    Draw an angle arc and label on the image.
    
    Args:
        img: OpenCV image
        keypoints: List of keypoints (x, y, confidence)
        angle_indices: Indices of the three keypoints forming the angle [a, b, c]
        angle_value: Angle value in degrees
        label: Text label for the angle
        color: Color of the angle arc and label
        thickness: Line thickness
    """
    h, w = img.shape[:2]
    
    # Skip if any keypoint has low confidence
    if any(keypoints[i][2] < 0.3 for i in angle_indices):
        return img
    
    # Scale keypoints from normalized coords to image coords
    a_idx, b_idx, c_idx = angle_indices
    
    ax = int((keypoints[a_idx][0] + 0.5) * w)
    ay = int((keypoints[a_idx][1] + 0.5) * h)
    
    bx = int((keypoints[b_idx][0] + 0.5) * w)
    by = int((keypoints[b_idx][1] + 0.5) * h)
    
    cx = int((keypoints[c_idx][0] + 0.5) * w)
    cy = int((keypoints[c_idx][1] + 0.5) * h)
    
    # Draw angle arc
    vec1 = np.array([ax - bx, ay - by])
    vec2 = np.array([cx - bx, cy - by])
    
    # Normalize vectors to a fixed length for the arc
    radius = 30
    len1 = np.linalg.norm(vec1)
    len2 = np.linalg.norm(vec2)
    
    if len1 > 0 and len2 > 0:
        vec1_normalized = vec1 * (radius / len1)
        vec2_normalized = vec2 * (radius / len2)
        
        pt1 = (int(bx + vec1_normalized[0]), int(by + vec1_normalized[1]))
        pt2 = (bx, by)
        pt3 = (int(bx + vec2_normalized[0]), int(by + vec2_normalized[1]))
        
        # Draw the lines to represent the angle
        cv2.line(img, pt1, pt2, color, thickness)
        cv2.line(img, pt2, pt3, color, thickness)
        
        # Draw arc representing the angle
        angle_rads = np.arccos(np.dot(vec1_normalized, vec2_normalized) / 
                              (np.linalg.norm(vec1_normalized) * np.linalg.norm(vec2_normalized)))
        angle_degrees = np.degrees(angle_rads)
        
        # Calculate start and end angles for the arc
        start_angle = np.arctan2(vec1_normalized[1], vec1_normalized[0])
        end_angle = np.arctan2(vec2_normalized[1], vec2_normalized[0])
        
        # Ensure correct order for drawing arc
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
            
        # Convert to degrees
        start_angle = np.degrees(start_angle)
        end_angle = np.degrees(end_angle)
        
        # Draw the arc
        cv2.ellipse(img, (bx, by), (radius, radius), 0, start_angle, end_angle, color, thickness)
    
    # Draw the angle value text
    if angle_value is not None:
        # Position the text near the angle
        mid_x = (ax + bx + cx) // 3
        mid_y = (ay + by + cy) // 3
        
        # Draw text with value
        text = f"{label}: {angle_value:.1f}°"
        cv2.putText(img, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return img

def visualize_frame(json_data, frame_id, angles_df, output_path=None):
    """
    Visualize a single frame with joint angles.
    
    Args:
        json_data: MMPose JSON data
        frame_id: Frame ID to visualize
        angles_df: DataFrame with angle values
        output_path: If provided, save the visualization to this path
    
    Returns:
        The visualization image
    """
    # Find the frame in the JSON data
    frame_data = next((f for f in json_data if f['frame_id'] == frame_id), None)
    if not frame_data or not frame_data.get('instances'):
        print(f"Frame {frame_id} not found or has no instances")
        return None
    
    # Get the keypoints
    keypoints = frame_data['instances'][0]['keypoints']
    
    # Create a blank image (black background)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw the skeleton
    img = draw_skeleton(img, keypoints)
    
    # Draw angles if available in the DataFrame
    if angles_df is not None:
        angle_row = angles_df[angles_df['frame_id'] == frame_id]
        
        if not angle_row.empty:
            angle_configs = {
                'left_elbow': {'indices': [5, 7, 9], 'color': (255, 0, 0)},     
                'right_elbow': {'indices': [6, 8, 10], 'color': (255, 0, 0)},   
                'left_shoulder': {'indices': [7, 5, 11], 'color': (0, 255, 0)},  
                'right_shoulder': {'indices': [8, 6, 12], 'color': (0, 255, 0)}, 
                'left_hip': {'indices': [5, 11, 13], 'color': (0, 0, 255)},     
                'right_hip': {'indices': [6, 12, 14], 'color': (0, 0, 255)},    
                'left_knee': {'indices': [11, 13, 15], 'color': (255, 255, 0)},  
                'right_knee': {'indices': [12, 14, 16], 'color': (255, 255, 0)}  
            }
            
            for angle_name, config in angle_configs.items():
                if angle_name in angle_row.columns:
                    angle_value = angle_row[angle_name].values[0]
                    if not pd.isna(angle_value):
                        img = draw_angle(img, keypoints, config['indices'], 
                                        angle_value, angle_name, config['color'])
    
    # Add frame number to image
    cv2.putText(img, f"Frame: {frame_id}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save image if output path provided
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img

def create_video(json_path, angles_csv_path, output_path, fps=30, start_frame=0, end_frame=None):
    """
    Create a video visualizing the joint angles.
    
    Args:
        json_path: Path to the MMPose JSON file
        angles_csv_path: Path to the CSV file with angle values
        output_path: Path to save the output video
        fps: Frames per second for the output video
        start_frame: First frame to include
        end_frame: Last frame to include (None for all frames)
    """
    # Load the JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load the angles DataFrame
    angles_df = pd.read_csv(angles_csv_path)
    
    # Determine frame range
    if end_frame is None:
        end_frame = max(f['frame_id'] for f in json_data)
    
    frames = [f for f in json_data if start_frame <= f['frame_id'] <= end_frame]
    if not frames:
        print(f"No frames found in the specified range ({start_frame}-{end_frame})")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Set up video writer
    output_width, output_height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    # Process each frame
    print(f"Creating video with {len(frames)} frames...")
    for frame in tqdm(frames):
        frame_id = frame['frame_id']
        img = visualize_frame(json_data, frame_id, angles_df)
        if img is not None:
            out.write(img)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize joint angles from MMPose JSON output')
    parser.add_argument('json_path', help='Path to the MMPose JSON file')
    parser.add_argument('angles_csv', help='Path to the CSV file with calculated angles')
    parser.add_argument('--output', '-o', help='Output directory for visualizations', required=True)
    parser.add_argument('--video', '-v', action='store_true', help='Create video visualization')
    parser.add_argument('--frames', '-f', action='store_true', help='Create individual frame visualizations')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video')
    parser.add_argument('--start_frame', type=int, default=0, help='First frame to visualize')
    parser.add_argument('--end_frame', type=int, default=None, help='Last frame to visualize')
    parser.add_argument('--step', type=int, default=1, help='Frame step size for individual frame output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON data
    with open(args.json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load the angles DataFrame
    angles_df = pd.read_csv(args.angles_csv)
    
    # Determine frame range
    end_frame = args.end_frame if args.end_frame is not None else max(f['frame_id'] for f in json_data)
    frames = [f for f in json_data if args.start_frame <= f['frame_id'] <= end_frame]
    
    if not frames:
        print(f"No frames found in the specified range ({args.start_frame}-{end_frame})")
        return
    
    # Get the file name without extension
    base_name = Path(args.json_path).stem
    
    # Create video if requested
    if args.video:
        video_path = output_dir / f"{base_name}_angles.mp4"
        create_video(args.json_path, args.angles_csv, str(video_path), 
                     args.fps, args.start_frame, end_frame)
    
    # Create individual frame visualizations if requested
    if args.frames:
        frames_dir = output_dir / f"{base_name}_frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"Creating visualizations for individual frames...")
        for frame in tqdm([f for f in frames if (f['frame_id'] - args.start_frame) % args.step == 0]):
            frame_id = frame['frame_id']
            output_path = frames_dir / f"frame_{frame_id:04d}.png"
            visualize_frame(json_data, frame_id, angles_df, str(output_path))
        
        print(f"Frame visualizations saved to {frames_dir}")

if __name__ == "__main__":
    main() 