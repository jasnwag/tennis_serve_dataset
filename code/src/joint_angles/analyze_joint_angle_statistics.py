#!/usr/bin/env python3
"""
Calculate mean and standard deviation time series for joint angles across all instances.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
from typing import Dict, List, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_joint_angles(joint_angles_str: str) -> Optional[List[Dict]]:
    """Parse the joint_angles string."""
    if pd.isna(joint_angles_str) or joint_angles_str == '':
        return None
    
    try:
        joint_angles_data = ast.literal_eval(joint_angles_str)
        return joint_angles_data
    except (ValueError, SyntaxError):
        try:
            joint_angles_data = json.loads(joint_angles_str)
            return joint_angles_data
        except json.JSONDecodeError:
            return None

def collect_joint_time_series(df: pd.DataFrame, max_length: int = 200) -> Dict[str, List[List[float]]]:
    """
    Collect time series data for each joint across all instances.
    
    Args:
        df: DataFrame with joint_angles column
        max_length: Maximum sequence length to consider
        
    Returns:
        Dictionary with joint names as keys and list of time series as values
    """
    joint_names = [
        'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip', 'left_knee', 'right_knee'
    ]
    
    # Initialize storage for each joint
    joint_data = {joint: [] for joint in joint_names}
    
    print("Collecting joint angle time series data...")
    valid_sequences = 0
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
            
        joint_angles_data = parse_joint_angles(row['joint_angles'])
        
        if joint_angles_data is None or len(joint_angles_data) < 10:
            continue
            
        # Limit sequence length
        sequence_length = min(len(joint_angles_data), max_length)
        joint_angles_data = joint_angles_data[:sequence_length]
        
        # Extract time series for each joint
        joint_sequences = {joint: [] for joint in joint_names}
        
        for frame_data in joint_angles_data:
            for joint in joint_names:
                value = frame_data.get(joint)
                if value is not None and not pd.isna(value):
                    joint_sequences[joint].append(value)
                else:
                    joint_sequences[joint].append(np.nan)
        
        # Only include sequences that have reasonable data
        if all(len(joint_sequences[joint]) >= 10 for joint in joint_names):
            for joint in joint_names:
                joint_data[joint].append(joint_sequences[joint])
            valid_sequences += 1
    
    print(f"Collected {valid_sequences} valid sequences")
    return joint_data

def calculate_statistics(joint_data: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Calculate mean and standard deviation for each joint at each time point.
    
    Args:
        joint_data: Dictionary with joint time series data
        
    Returns:
        Dictionary with statistics for each joint
    """
    joint_stats = {}
    
    for joint_name, time_series_list in joint_data.items():
        if not time_series_list:
            continue
            
        print(f"Calculating statistics for {joint_name} ({len(time_series_list)} sequences)")
        
        # Find the maximum length among all sequences for this joint
        max_length = max(len(ts) for ts in time_series_list)
        
        # Create arrays to store values at each time point
        time_point_values = [[] for _ in range(max_length)]
        
        # Collect values at each time point across all sequences
        for time_series in time_series_list:
            for t, value in enumerate(time_series):
                if t < max_length and not pd.isna(value):
                    time_point_values[t].append(value)
        
        # Calculate statistics at each time point
        means = []
        stds = []
        counts = []
        
        for t in range(max_length):
            values = time_point_values[t]
            if len(values) >= 3:  # Need at least 3 values for meaningful stats
                means.append(np.mean(values))
                stds.append(np.std(values))
                counts.append(len(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                counts.append(len(values))
        
        joint_stats[joint_name] = {
            'mean': means,
            'std': stds,
            'count': counts,
            'upper_bound': [m + s if not pd.isna(m) and not pd.isna(s) else np.nan 
                          for m, s in zip(means, stds)],
            'lower_bound': [m - s if not pd.isna(m) and not pd.isna(s) else np.nan 
                          for m, s in zip(means, stds)]
        }
    
    return joint_stats

def create_joint_statistics_plots(joint_stats: Dict[str, Dict[str, List[float]]]):
    """Create plots showing mean and standard deviation for each joint."""
    joint_names = list(joint_stats.keys())
    
    # Create subplots - 2 rows, 4 columns for 8 joints
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Joint Angle Time Series: Mean ± 1 Standard Deviation', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, joint_name in enumerate(joint_names):
        ax = axes_flat[i]
        stats = joint_stats[joint_name]
        
        time_points = range(len(stats['mean']))
        
        # Plot mean line
        ax.plot(time_points, stats['mean'], 'b-', linewidth=2, label='Mean', alpha=0.8)
        
        # Plot standard deviation bands
        ax.fill_between(time_points, 
                       stats['lower_bound'], 
                       stats['upper_bound'], 
                       alpha=0.3, color='blue', label='±1 Std Dev')
        
        ax.set_title(f'{joint_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Angle (degrees)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        valid_frames = sum(1 for count in stats['count'] if count >= 3)
        max_count = max(stats['count']) if stats['count'] else 0
        ax.text(0.02, 0.98, f'Valid frames: {valid_frames}\nMax instances: {max_count}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def save_statistics_to_csv(joint_stats: Dict[str, Dict[str, List[float]]], output_file: str):
    """Save the statistics to a CSV file."""
    # Prepare data for DataFrame
    data_rows = []
    
    # Find maximum length across all joints
    max_length = max(len(stats['mean']) for stats in joint_stats.values())
    
    for frame_idx in range(max_length):
        row = {'frame': frame_idx}
        
        for joint_name, stats in joint_stats.items():
            if frame_idx < len(stats['mean']):
                row[f'{joint_name}_mean'] = stats['mean'][frame_idx]
                row[f'{joint_name}_std'] = stats['std'][frame_idx]
                row[f'{joint_name}_count'] = stats['count'][frame_idx]
                row[f'{joint_name}_upper'] = stats['upper_bound'][frame_idx]
                row[f'{joint_name}_lower'] = stats['lower_bound'][frame_idx]
            else:
                row[f'{joint_name}_mean'] = np.nan
                row[f'{joint_name}_std'] = np.nan
                row[f'{joint_name}_count'] = 0
                row[f'{joint_name}_upper'] = np.nan
                row[f'{joint_name}_lower'] = np.nan
        
        data_rows.append(row)
    
    df_stats = pd.DataFrame(data_rows)
    df_stats.to_csv(output_file, index=False)
    print(f"Statistics saved to: {output_file}")

def main():
    """Main function to analyze joint angle statistics."""
    input_file = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles.csv'
    
    print("Loading CSV file...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Collect joint time series data
    joint_data = collect_joint_time_series(df, max_length=150)  # Limit to 150 frames for computational efficiency
    
    # Calculate statistics
    print("\nCalculating statistics...")
    joint_stats = calculate_statistics(joint_data)
    
    # Create plots
    print("\nCreating plots...")
    fig = create_joint_statistics_plots(joint_stats)
    
    # Save plot
    plot_output = 'data/joint_angle_mean_std_timeseries.png'
    fig.savefig(plot_output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_output}")
    
    # Save statistics to CSV
    csv_output = 'data/joint_angle_statistics_timeseries.csv'
    save_statistics_to_csv(joint_stats, csv_output)
    
    # Show summary statistics
    print("\nSummary:")
    for joint_name, stats in joint_stats.items():
        valid_frames = sum(1 for count in stats['count'] if count >= 3)
        max_count = max(stats['count']) if stats['count'] else 0
        mean_angle = np.nanmean(stats['mean'])
        mean_std = np.nanmean(stats['std'])
        
        print(f"{joint_name}:")
        print(f"  Valid time points: {valid_frames}")
        print(f"  Max instances at any time point: {max_count}")
        print(f"  Overall mean angle: {mean_angle:.1f}°")
        print(f"  Average std deviation: {mean_std:.1f}°")
    
    plt.show()

if __name__ == "__main__":
    main() 