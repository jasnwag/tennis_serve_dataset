#!/usr/bin/env python3
"""
Example script showing how to use the calculate_angles.py module.
"""

import os
import sys
from pathlib import Path
from calculate_angles import process_json_file, plot_angles

def main():
    # Example usage
    json_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/USTA/predictions/5.6 tall junior with 🔥🔥 serve motion #shorts simple serve done right 💪💪💪.json"
    
    # Create output directories
    base_dir = Path("/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/USTA")
    output_dir = base_dir / "angle_analysis"
    plots_dir = output_dir / "plots"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get the filename without extension for output naming
    filename = Path(json_path).stem
    output_csv = output_dir / f"{filename}_angles.csv"
    
    print(f"Processing {json_path}...")
    
    # Process the JSON file
    angles_df = process_json_file(json_path)
    
    # Save to CSV
    angles_df.to_csv(output_csv, index=False)
    print(f"Angles saved to {output_csv}")
    
    # Generate plots
    plot_dir = plots_dir / filename
    os.makedirs(plot_dir, exist_ok=True)
    plot_angles(angles_df, plot_dir)
    print(f"Plots saved to {plot_dir}")
    
    # Print summary statistics
    print("\nSummary statistics for joint angles:")
    for col in angles_df.columns:
        if col != 'frame_id':
            valid_angles = angles_df[col].dropna()
            if len(valid_angles) > 0:
                print(f"{col}:")
                print(f"  Mean: {valid_angles.mean():.2f} degrees")
                print(f"  Min: {valid_angles.min():.2f} degrees")
                print(f"  Max: {valid_angles.max():.2f} degrees")
                print(f"  Valid frames: {len(valid_angles)} / {len(angles_df)}")
                print()

if __name__ == "__main__":
    main() 