#!/usr/bin/env python3
"""
Trim a video to a specific frame range.
"""

import cv2
import argparse
from pathlib import Path
import os
from tqdm import tqdm
import glob

def trim_video(input_path, output_path, start_frame, end_frame):
    """
    Trim a video to a specific frame range.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        start_frame: First frame to include
        end_frame: Last frame to include
    """
    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate frame range
    if start_frame < 0 or end_frame >= total_frames or start_frame > end_frame:
        print(f"Error: Invalid frame range. Video has {total_frames} frames.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print(f"Trimming video from frame {start_frame} to {end_frame}...")
    frame_count = 0
    
    with tqdm(total=end_frame - start_frame + 1) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only write frames in the specified range
            if start_frame <= frame_count <= end_frame:
                out.write(frame)
                pbar.update(1)
            
            frame_count += 1
            
            # Stop if we've reached the end frame
            if frame_count > end_frame:
                break
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Trimmed video saved to {output_path}")

def find_video_file(directory, pattern):
    """
    Find a video file in the directory that matches the pattern.
    
    Args:
        directory: Directory to search in
        pattern: Pattern to match (without extension)
        
    Returns:
        Path to the matching video file or None if not found
    """
    # Try to find the file with different extensions
    for ext in ['.mp4', '.MP4', '.mov', '.MOV']:
        # Use glob to handle special characters in filenames
        matches = glob.glob(os.path.join(directory, f"{pattern}{ext}"))
        if matches:
            return matches[0]
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Trim a video to a specific frame range')
    parser.add_argument('input_path', help='Path to the input video or pattern to match')
    parser.add_argument('--output_path', '-o', help='Path to save the output video')
    parser.add_argument('--start_frame', '-s', type=int, required=True, help='First frame to include')
    parser.add_argument('--end_frame', '-e', type=int, required=True, help='Last frame to include')
    
    args = parser.parse_args()
    
    # Check if the input path exists
    if not os.path.exists(args.input_path):
        # Try to find the file using the pattern
        directory = os.path.dirname(args.input_path)
        pattern = os.path.splitext(os.path.basename(args.input_path))[0]
        input_path = find_video_file(directory, pattern)
        
        if input_path is None:
            print(f"Error: Could not find video file matching pattern: {args.input_path}")
            return
    else:
        input_path = args.input_path
    
    # If output path is not specified, create one based on the input path
    if args.output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_trimmed{input_path_obj.suffix}")
    else:
        output_path = args.output_path
    
    # Trim the video
    trim_video(input_path, output_path, args.start_frame, args.end_frame)

if __name__ == "__main__":
    main() 