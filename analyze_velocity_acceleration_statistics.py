#!/usr/bin/env python3
"""
Calculate mean and standard deviation time series for joint velocities and accelerations across all instances.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import ast
from typing import Dict, List, Optional, Tuple
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_joint_data(data_str: str) -> Optional[List[float]]:
    """Parse the joint velocity or acceleration data string."""
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

def collect_velocity_acceleration_data(df: pd.DataFrame, max_length: int = 150) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
    """
    Collect velocity and acceleration time series data for each joint across all instances.
    
    Args:
        df: DataFrame with velocity and acceleration columns
        max_length: Maximum sequence length to consider
        
    Returns:
        Tuple of (velocity_data, acceleration_data) dictionaries
    """
    joint_names = [
        'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip', 'left_knee', 'right_knee'
    ]
    
    # Initialize storage for each joint
    velocity_data = {joint: [] for joint in joint_names}
    acceleration_data = {joint: [] for joint in joint_names}
    
    print("Collecting velocity and acceleration time series data...")
    valid_sequences = 0
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        # Check if this row has velocity/acceleration data
        has_data = True
        joint_velocity_sequences = {}
        joint_acceleration_sequences = {}
        
        for joint in joint_names:
            velocity_col = f'{joint}_velocity'
            acceleration_col = f'{joint}_acceleration'
            
            if velocity_col not in df.columns or acceleration_col not in df.columns:
                has_data = False
                break
            
            velocity_values = parse_joint_data(row[velocity_col])
            acceleration_values = parse_joint_data(row[acceleration_col])
            
            if velocity_values is None or acceleration_values is None:
                has_data = False
                break
            
            if len(velocity_values) < 10 or len(acceleration_values) < 10:
                has_data = False
                break
            
            # Limit sequence length
            velocity_values = velocity_values[:max_length]
            acceleration_values = acceleration_values[:max_length]
            
            joint_velocity_sequences[joint] = velocity_values
            joint_acceleration_sequences[joint] = acceleration_values
        
        if has_data:
            for joint in joint_names:
                velocity_data[joint].append(joint_velocity_sequences[joint])
                acceleration_data[joint].append(joint_acceleration_sequences[joint])
            valid_sequences += 1
    
    print(f"Collected {valid_sequences} valid sequences")
    return velocity_data, acceleration_data

def calculate_statistics(data: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Calculate mean and standard deviation for each joint at each time point.
    
    Args:
        data: Dictionary with joint time series data
        
    Returns:
        Dictionary with statistics for each joint
    """
    stats = {}
    
    for joint_name, time_series_list in data.items():
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
        
        stats[joint_name] = {
            'mean': means,
            'std': stds,
            'count': counts,
            'upper_bound': [m + s if not pd.isna(m) and not pd.isna(s) else np.nan 
                          for m, s in zip(means, stds)],
            'lower_bound': [m - s if not pd.isna(m) and not pd.isna(s) else np.nan 
                          for m, s in zip(means, stds)]
        }
    
    return stats

def create_velocity_plots(velocity_stats: Dict[str, Dict[str, List[float]]]):
    """Create plots showing mean and standard deviation for joint velocities."""
    joint_names = list(velocity_stats.keys())
    
    # Create subplots - 2 rows, 4 columns for 8 joints
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Joint Velocity Time Series: Mean ± 1 Standard Deviation', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, joint_name in enumerate(joint_names):
        ax = axes_flat[i]
        stats = velocity_stats[joint_name]
        
        time_points = range(len(stats['mean']))
        
        # Plot mean line
        ax.plot(time_points, stats['mean'], 'g-', linewidth=2, label='Mean', alpha=0.8)
        
        # Plot standard deviation bands
        ax.fill_between(time_points, 
                       stats['lower_bound'], 
                       stats['upper_bound'], 
                       alpha=0.3, color='green', label='±1 Std Dev')
        
        ax.set_title(f'{joint_name.replace("_", " ").title()} Velocity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Velocity (deg/frame)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        valid_frames = sum(1 for count in stats['count'] if count >= 3)
        max_count = max(stats['count']) if stats['count'] else 0
        ax.text(0.02, 0.98, f'Valid frames: {valid_frames}\nMax instances: {max_count}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    return fig

def create_acceleration_plots(acceleration_stats: Dict[str, Dict[str, List[float]]]):
    """Create plots showing mean and standard deviation for joint accelerations."""
    joint_names = list(acceleration_stats.keys())
    
    # Create subplots - 2 rows, 4 columns for 8 joints
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Joint Acceleration Time Series: Mean ± 1 Standard Deviation', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, joint_name in enumerate(joint_names):
        ax = axes_flat[i]
        stats = acceleration_stats[joint_name]
        
        time_points = range(len(stats['mean']))
        
        # Plot mean line
        ax.plot(time_points, stats['mean'], 'r-', linewidth=2, label='Mean', alpha=0.8)
        
        # Plot standard deviation bands
        ax.fill_between(time_points, 
                       stats['lower_bound'], 
                       stats['upper_bound'], 
                       alpha=0.3, color='red', label='±1 Std Dev')
        
        ax.set_title(f'{joint_name.replace("_", " ").title()} Acceleration', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Acceleration (deg/frame²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        valid_frames = sum(1 for count in stats['count'] if count >= 3)
        max_count = max(stats['count']) if stats['count'] else 0
        ax.text(0.02, 0.98, f'Valid frames: {valid_frames}\nMax instances: {max_count}', 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    return fig

def save_statistics_to_csv(stats: Dict[str, Dict[str, List[float]]], output_file: str, data_type: str):
    """Save the statistics to a CSV file."""
    # Prepare data for DataFrame
    data_rows = []
    
    # Find maximum length across all joints
    max_length = max(len(joint_stats['mean']) for joint_stats in stats.values())
    
    for frame_idx in range(max_length):
        row = {'frame': frame_idx}
        
        for joint_name, joint_stats in stats.items():
            if frame_idx < len(joint_stats['mean']):
                row[f'{joint_name}_{data_type}_mean'] = joint_stats['mean'][frame_idx]
                row[f'{joint_name}_{data_type}_std'] = joint_stats['std'][frame_idx]
                row[f'{joint_name}_{data_type}_count'] = joint_stats['count'][frame_idx]
                row[f'{joint_name}_{data_type}_upper'] = joint_stats['upper_bound'][frame_idx]
                row[f'{joint_name}_{data_type}_lower'] = joint_stats['lower_bound'][frame_idx]
            else:
                row[f'{joint_name}_{data_type}_mean'] = np.nan
                row[f'{joint_name}_{data_type}_std'] = np.nan
                row[f'{joint_name}_{data_type}_count'] = 0
                row[f'{joint_name}_{data_type}_upper'] = np.nan
                row[f'{joint_name}_{data_type}_lower'] = np.nan
        
        data_rows.append(row)
    
    df_stats = pd.DataFrame(data_rows)
    df_stats.to_csv(output_file, index=False)
    print(f"{data_type.title()} statistics saved to: {output_file}")

def print_summary(stats: Dict[str, Dict[str, List[float]]], data_type: str):
    """Print summary statistics."""
    print(f"\n{data_type.title()} Summary:")
    for joint_name, joint_stats in stats.items():
        valid_frames = sum(1 for count in joint_stats['count'] if count >= 3)
        max_count = max(joint_stats['count']) if joint_stats['count'] else 0
        mean_value = np.nanmean(joint_stats['mean'])
        mean_std = np.nanmean(joint_stats['std'])
        
        print(f"{joint_name}:")
        print(f"  Valid time points: {valid_frames}")
        print(f"  Max instances at any time point: {max_count}")
        print(f"  Overall mean {data_type}: {mean_value:.2f}")
        print(f"  Average std deviation: {mean_std:.2f}")

def main():
    """Main function to analyze velocity and acceleration statistics."""
    input_file = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_velocity_acceleration.csv'
    
    print("Loading CSV file with velocity and acceleration data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Collect velocity and acceleration time series data
    velocity_data, acceleration_data = collect_velocity_acceleration_data(df, max_length=120)
    
    # Calculate statistics for velocity
    print("\nCalculating velocity statistics...")
    velocity_stats = calculate_statistics(velocity_data)
    
    # Calculate statistics for acceleration
    print("\nCalculating acceleration statistics...")
    acceleration_stats = calculate_statistics(acceleration_data)
    
    # Create velocity plots
    print("\nCreating velocity plots...")
    velocity_fig = create_velocity_plots(velocity_stats)
    velocity_plot_output = 'data/joint_velocity_mean_std_timeseries.png'
    velocity_fig.savefig(velocity_plot_output, dpi=300, bbox_inches='tight')
    print(f"Velocity plot saved to: {velocity_plot_output}")
    
    # Create acceleration plots
    print("\nCreating acceleration plots...")
    acceleration_fig = create_acceleration_plots(acceleration_stats)
    acceleration_plot_output = 'data/joint_acceleration_mean_std_timeseries.png'
    acceleration_fig.savefig(acceleration_plot_output, dpi=300, bbox_inches='tight')
    print(f"Acceleration plot saved to: {acceleration_plot_output}")
    
    # Save statistics to CSV files
    velocity_csv_output = 'data/joint_velocity_statistics_timeseries.csv'
    acceleration_csv_output = 'data/joint_acceleration_statistics_timeseries.csv'
    
    save_statistics_to_csv(velocity_stats, velocity_csv_output, 'velocity')
    save_statistics_to_csv(acceleration_stats, acceleration_csv_output, 'acceleration')
    
    # Print summaries
    print_summary(velocity_stats, 'velocity')
    print_summary(acceleration_stats, 'acceleration')
    
    plt.show()

if __name__ == "__main__":
    main() 