#!/usr/bin/env python3
"""
Add a frame counter to a video.
"""

import cv2
import argparse
from pathlib import Path
import os
from tqdm import tqdm

def add_frame_counter(input_path, output_path, counter_position='top_right', font_scale=1.0, thickness=2, color=(255, 255, 255)):
    """
    Add a frame counter to a video.
    
    Args:
        input_path: Path to the input video
        output_path: Path to save the output video
        counter_position: Position of the counter ('top_right', 'top_left', 'bottom_right', 'bottom_left')
        font_scale: Scale of the font
        thickness: Thickness of the text
        color: Color of the text (B, G, R)
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
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print(f"Processing video with {total_frames} frames...")
    frame_count = 0
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add frame counter
            text = f"Frame: {frame_count}"
            
            # Get text size for positioning
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position the text based on the specified position
            if counter_position == 'top_right':
                position = (width - text_width - 10, text_height + 10)
            elif counter_position == 'top_left':
                position = (10, text_height + 10)
            elif counter_position == 'bottom_right':
                position = (width - text_width - 10, height - 10)
            elif counter_position == 'bottom_left':
                position = (10, height - 10)
            else:
                position = (width - text_width - 10, text_height + 10)  # Default to top right
            
            # Add text to frame
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            # Write the frame
            out.write(frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video with frame counter saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Add a frame counter to a video')
    parser.add_argument('input_path', help='Path to the input video')
    parser.add_argument('--output_path', '-o', help='Path to save the output video')
    parser.add_argument('--position', '-p', choices=['top_right', 'top_left', 'bottom_right', 'bottom_left'], 
                        default='top_right', help='Position of the frame counter')
    parser.add_argument('--font_scale', '-f', type=float, default=1.0, help='Scale of the font')
    parser.add_argument('--thickness', '-t', type=int, default=2, help='Thickness of the text')
    parser.add_argument('--color', '-c', nargs=3, type=int, default=[255, 255, 255], 
                        help='Color of the text (B, G, R)')
    
    args = parser.parse_args()
    
    # If output path is not specified, create one based on the input path
    if args.output_path is None:
        input_path = Path(args.input_path)
        output_path = str(input_path.parent / f"{input_path.stem}_with_counter{input_path.suffix}")
    else:
        output_path = args.output_path
    
    # Convert color from list to tuple
    color = tuple(args.color)
    
    # Add frame counter to the video
    add_frame_counter(args.input_path, output_path, args.position, args.font_scale, args.thickness, color)

if __name__ == "__main__":
    main() 