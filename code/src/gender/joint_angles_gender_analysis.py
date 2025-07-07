#!/usr/bin/env python3
"""
Joint Angles Analysis by Gender
Analyzes joint angle patterns comparing male and female tennis players
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(csv_path):
    """Load the CSV data and extract joint angles information."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} rows")
    
    # Extract and process joint angles data
    joint_angles_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['joint_angles']) and row['joint_angles'] != '':
            try:
                # Parse the joint angles JSON string
                angles = json.loads(row['joint_angles'])
                
                # Extract frame data
                frames = len(angles)
                if frames > 1:  # Need at least 2 points for interpolation
                    joint_angles_data.append({
                        'server_gender': row['server_gender'],
                        'angles': angles,
                        'original_frames': frames
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    
    print(f"Found {len(joint_angles_data)} sequences with valid joint angles data")
    return joint_angles_data

def interpolate_sequences(joint_angles_data, target_frames=100):
    """Interpolate all sequences to the same number of frames."""
    print(f"Interpolating sequences to {target_frames} frames...")
    
    # Define the joints to analyze
    joints = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
              'left_hip', 'right_hip', 'left_knee', 'right_knee']
    
    # Interpolate each sequence to target_frames
    interpolated_data = {'M': {joint: [] for joint in joints}, 
                        'F': {joint: [] for joint in joints}}
    
    for sequence in joint_angles_data:
        angles = sequence['angles']
        gender = sequence['server_gender']
        original_frames = len(angles)
        
        # Skip if gender not recognized
        if gender not in ['M', 'F']:
            continue
        
        # Create original frame indices
        original_indices = np.arange(original_frames)
        # Create new frame indices (0 to target_frames-1)
        new_indices = np.linspace(0, original_frames - 1, target_frames)
        
        # Interpolate each joint
        for joint in joints:
            try:
                # Extract joint angle values across frames
                joint_values = [frame_data[joint] for frame_data in angles]
                
                # Interpolate to target_frames
                f = interp1d(original_indices, joint_values, kind='linear')
                interpolated_values = f(new_indices)
                
                interpolated_data[gender][joint].append(interpolated_values)
            except KeyError:
                # Skip if joint not found in this sequence
                continue
    
    # Print counts
    for gender in ['M', 'F']:
        total_sequences = len(interpolated_data[gender]['left_elbow']) if 'left_elbow' in interpolated_data[gender] else 0
        print(f"Gender {gender}: {total_sequences} sequences processed")
    
    return interpolated_data, joints

def calculate_statistics(interpolated_data, joints):
    """Calculate mean and standard deviation for each joint by gender."""
    print("Calculating statistics...")
    
    means_stds = {'M': {}, 'F': {}}
    
    for gender in ['M', 'F']:
        for joint in joints:
            if len(interpolated_data[gender][joint]) > 0:
                joint_array = np.array(interpolated_data[gender][joint])
                means_stds[gender][joint] = {
                    'mean': np.mean(joint_array, axis=0),
                    'std': np.std(joint_array, axis=0),
                    'count': len(interpolated_data[gender][joint])
                }
    
    return means_stds

def create_comprehensive_plot(means_stds, joints, target_frames=100):
    """Create comprehensive plot comparing male and female joint angles."""
    print("Creating plots...")
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    frame_indices = np.arange(target_frames)
    colors = {'M': 'blue', 'F': 'red'}
    gender_labels = {'M': 'Male', 'F': 'Female'}
    
    for i, joint in enumerate(joints):
        ax = axes[i]
        
        for gender in ['M', 'F']:
            if joint in means_stds[gender] and gender in means_stds and means_stds[gender][joint]['count'] > 0:
                mean_vals = means_stds[gender][joint]['mean']
                std_vals = means_stds[gender][joint]['std']
                count = means_stds[gender][joint]['count']
                
                # Plot mean line
                ax.plot(frame_indices, mean_vals, color=colors[gender], 
                       linewidth=2, label=f'{gender_labels[gender]} (n={count})')
                
                # Plot standard deviation band
                ax.fill_between(frame_indices, 
                               mean_vals - std_vals, 
                               mean_vals + std_vals, 
                               color=colors[gender], alpha=0.2)
        
        ax.set_title(f'{joint.replace("_", " ").title()} Angle', fontsize=12)
        ax.set_xlabel(f'Frame (normalized to {target_frames})', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Joint Angles: Mean ± 1 Standard Deviation by Gender\n(All sequences interpolated to {target_frames} frames)', 
                 fontsize=16, y=1.02)
    
    # Save the plot
    plt.savefig('/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/code/src/visualizations/7_7_25/joint_angles_gender_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(means_stds, joints):
    """Print summary statistics for all joints by gender."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for joint in joints:
        print(f"\n{joint.replace('_', ' ').title()}:")
        for gender in ['M', 'F']:
            gender_label = 'Male' if gender == 'M' else 'Female'
            if joint in means_stds[gender]:
                data = means_stds[gender][joint]
                overall_mean = np.mean(data['mean'])
                overall_std_mean = np.mean(data['std'])
                
                print(f"  {gender_label}: n={data['count']}, "
                      f"overall mean={overall_mean:.1f}°, "
                      f"avg std={overall_std_mean:.1f}°")

def main():
    """Main analysis function."""
    # File path
    csv_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_velocity_acceleration.csv'
    target_frames = 100
    
    try:
        # Load and process data
        joint_angles_data = load_and_process_data(csv_path)
        
        if len(joint_angles_data) == 0:
            print("No valid joint angles data found!")
            return
        
        # Interpolate sequences
        interpolated_data, joints = interpolate_sequences(joint_angles_data, target_frames)
        
        # Calculate statistics
        means_stds = calculate_statistics(interpolated_data, joints)
        
        # Create plots
        create_comprehensive_plot(means_stds, joints, target_frames)
        
        # Print summary
        print_summary_statistics(means_stds, joints)
        
        print(f"\nAnalysis complete! Plot saved to visualizations/7_7_25/")
        
    except FileNotFoundError:
        print(f"Error: Could not find the data file at {csv_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 