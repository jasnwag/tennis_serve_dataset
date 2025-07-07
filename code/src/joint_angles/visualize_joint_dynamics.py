#!/usr/bin/env python3
"""
Visualize joint angles, velocity, and acceleration over time with statistical summaries.
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

def parse_joint_data(data_str: str) -> Optional[List[float]]:
    """
    Parse joint data string (JSON format).
    
    Args:
        data_str: String representation of joint data
        
    Returns:
        List of float values, or None if parsing fails
    """
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON first
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

def extract_time_series_data(df: pd.DataFrame, joint_names: List[str]) -> Dict:
    """
    Extract time series data for all joints and metrics.
    
    Args:
        df: DataFrame with joint data
        joint_names: List of joint names
        
    Returns:
        Dictionary with organized time series data
    """
    data = {
        'angles': {joint: [] for joint in joint_names},
        'velocity': {joint: [] for joint in joint_names},
        'acceleration': {joint: [] for joint in joint_names}
    }
    
    print("Extracting time series data...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        # Parse joint angles
        joint_angles_data = parse_joint_data(row['joint_angles'])
        if joint_angles_data is not None:
            angle_sequence = {joint: [] for joint in joint_names}
            for frame_data in joint_angles_data:
                for joint in joint_names:
                    value = frame_data.get(joint)
                    if value is not None and not pd.isna(value):
                        angle_sequence[joint].append(value)
                    else:
                        angle_sequence[joint].append(np.nan)
            
            for joint in joint_names:
                if len(angle_sequence[joint]) > 0:
                    data['angles'][joint].append(angle_sequence[joint])
        
        # Parse velocity and acceleration
        for joint in joint_names:
            # Velocity
            velocity_data = parse_joint_data(row[f'{joint}_velocity'])
            if velocity_data is not None:
                data['velocity'][joint].append(velocity_data)
            
            # Acceleration
            acceleration_data = parse_joint_data(row[f'{joint}_acceleration'])
            if acceleration_data is not None:
                data['acceleration'][joint].append(acceleration_data)
    
    return data

def calculate_statistics(sequences: List[List[float]]) -> tuple:
    """
    Calculate mean and standard deviation across sequences at each time point.
    
    Args:
        sequences: List of sequences (each sequence is a list of values over time)
        
    Returns:
        Tuple of (time_points, mean_values, std_values)
    """
    if not sequences:
        return [], [], []
    
    # Find the maximum length to determine time points
    max_length = max(len(seq) for seq in sequences if seq)
    
    if max_length == 0:
        return [], [], []
    
    time_points = list(range(max_length))
    mean_values = []
    std_values = []
    
    for t in range(max_length):
        values_at_t = []
        for seq in sequences:
            if len(seq) > t and not pd.isna(seq[t]):
                values_at_t.append(seq[t])
        
        if values_at_t:
            mean_values.append(np.mean(values_at_t))
            std_values.append(np.std(values_at_t))
        else:
            mean_values.append(np.nan)
            std_values.append(np.nan)
    
    return time_points, mean_values, std_values

def create_visualization(data: Dict, joint_names: List[str]):
    """
    Create comprehensive visualization of joint dynamics.
    
    Args:
        data: Dictionary with time series data
        joint_names: List of joint names
    """
    metrics = ['angles', 'velocity', 'acceleration']
    metric_labels = ['Joint Angles (degrees)', 'Angular Velocity (degrees/frame)', 'Angular Acceleration (degrees/frame²)']
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(joint_names), len(metrics), figsize=(18, 24))
    fig.suptitle('Joint Dynamics Over Time: Angles, Velocity, and Acceleration', fontsize=16, fontweight='bold')
    
    for i, joint in enumerate(joint_names):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # Calculate statistics
            sequences = data[metric][joint]
            time_points, mean_values, std_values = calculate_statistics(sequences)
            
            if time_points and not all(pd.isna(mean_values)):
                # Convert to numpy arrays for easier manipulation
                time_points = np.array(time_points)
                mean_values = np.array(mean_values)
                std_values = np.array(std_values)
                
                # Remove NaN values
                valid_mask = ~(pd.isna(mean_values) | pd.isna(std_values))
                time_points = time_points[valid_mask]
                mean_values = mean_values[valid_mask]
                std_values = std_values[valid_mask]
                
                if len(time_points) > 0:
                    # Plot mean line
                    ax.plot(time_points, mean_values, color=metric_colors[j], 
                           linewidth=2, label='Mean', alpha=0.8)
                    
                    # Plot standard deviation band
                    ax.fill_between(time_points, 
                                   mean_values - std_values, 
                                   mean_values + std_values,
                                   color=metric_colors[j], alpha=0.3, 
                                   label='±1 Std Dev')
                    
                    # Formatting
                    ax.set_title(f'{joint.replace("_", " ").title()} - {metric_labels[j]}', 
                               fontweight='bold')
                    ax.set_xlabel('Frame Number')
                    ax.set_ylabel(metric_labels[j])
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    # Add statistics text
                    stats_text = f'Samples: {len(sequences)}\nMean: {np.mean(mean_values):.2f}\nStd: {np.mean(std_values):.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
            
            # Set consistent y-axis limits for similar metrics across joints
            if j == 0:  # angles
                ax.set_ylim(0, 180)
            elif j == 1:  # velocity
                ax.set_ylim(-50, 50)
            elif j == 2:  # acceleration
                ax.set_ylim(-30, 30)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create visualizations."""
    input_file = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_velocity_acceleration.csv'
    output_file = 'joint_dynamics_visualization.png'
    
    print("Loading CSV file...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows")
    
    # Define joint names
    joint_names = [
        'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip', 'left_knee', 'right_knee'
    ]
    
    # Extract time series data
    data = extract_time_series_data(df, joint_names)
    
    # Create visualization
    print("Creating visualization...")
    fig = create_visualization(data, joint_names)
    
    # Save the plot
    print(f"Saving visualization to {output_file}...")
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization complete!")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 60)
    for joint in joint_names:
        print(f"\n{joint.replace('_', ' ').title()}:")
        for metric in ['angles', 'velocity', 'acceleration']:
            sequences = data[metric][joint]
            if sequences:
                all_values = [val for seq in sequences for val in seq if not pd.isna(val)]
                if all_values:
                    mean_val = np.mean(all_values)
                    std_val = np.std(all_values)
                    print(f"  {metric.capitalize()}: Mean={mean_val:.2f}, Std={std_val:.2f}, Samples={len(sequences)}")

if __name__ == "__main__":
    main() 