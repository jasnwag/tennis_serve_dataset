#!/usr/bin/env python3
"""
Process joint angles data with smoothing to calculate velocity and acceleration.
"""

import pandas as pd
import numpy as np
import json
import ast
from typing import Dict, List, Optional
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def parse_joint_angles(joint_angles_str: str) -> Optional[List[Dict]]:
    """
    Parse the joint_angles string (which appears to be a list of dictionaries).
    
    Args:
        joint_angles_str: String representation of joint angles data
        
    Returns:
        List of dictionaries with joint angles data, or None if parsing fails
    """
    if pd.isna(joint_angles_str) or joint_angles_str == '':
        return None
    
    try:
        # Try to parse as literal (using ast.literal_eval)
        joint_angles_data = ast.literal_eval(joint_angles_str)
        return joint_angles_data
    except (ValueError, SyntaxError):
        try:
            # Try to parse as JSON
            joint_angles_data = json.loads(joint_angles_str)
            return joint_angles_data
        except json.JSONDecodeError:
            print(f"Warning: Could not parse joint angles data: {joint_angles_str[:100]}...")
            return None

def smooth_data(values: List[float], method: str = 'savgol', **kwargs) -> List[float]:
    """
    Smooth the data using various methods.
    
    Args:
        values: List of values to smooth
        method: Smoothing method ('savgol', 'gaussian', 'moving_average', 'butterworth')
        **kwargs: Additional parameters for smoothing methods
        
    Returns:
        Smoothed values
    """
    # Remove NaN values for processing
    clean_values = np.array(values)
    nan_mask = np.isnan(clean_values)
    
    if np.all(nan_mask) or len(clean_values) < 3:
        return values
    
    # Interpolate NaN values for smoothing
    if np.any(nan_mask):
        valid_indices = np.where(~nan_mask)[0]
        clean_values = np.interp(np.arange(len(clean_values)), valid_indices, clean_values[valid_indices])
    
    if method == 'savgol':
        # Savitzky-Golay filter - good for preserving peaks and trends
        window_length = kwargs.get('window_length', min(11, len(clean_values) if len(clean_values) % 2 == 1 else len(clean_values) - 1))
        if window_length < 3:
            window_length = 3
        if window_length % 2 == 0:
            window_length += 1
        polyorder = kwargs.get('polyorder', min(3, window_length - 1))
        
        try:
            smoothed = signal.savgol_filter(clean_values, window_length, polyorder)
        except:
            # Fallback to simpler parameters if savgol fails
            smoothed = signal.savgol_filter(clean_values, 3, 1)
            
    elif method == 'gaussian':
        # Gaussian smoothing
        sigma = kwargs.get('sigma', 1.0)
        smoothed = gaussian_filter1d(clean_values, sigma=sigma)
        
    elif method == 'moving_average':
        # Simple moving average
        window = kwargs.get('window', 5)
        smoothed = np.convolve(clean_values, np.ones(window)/window, mode='same')
        
    elif method == 'butterworth':
        # Butterworth low-pass filter
        fs = kwargs.get('fs', 30.0)  # Sampling frequency (frames per second)
        cutoff = kwargs.get('cutoff', 10.0)  # Cutoff frequency
        order = kwargs.get('order', 4)
        
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        smoothed = signal.filtfilt(b, a, clean_values)
    
    else:
        # No smoothing
        smoothed = clean_values
    
    # Restore NaN values at original positions
    result = smoothed.copy()
    result[nan_mask] = np.nan
    
    return result.tolist()

def calculate_derivatives(values: List[float], smooth: bool = True, smooth_method: str = 'savgol') -> tuple[List[float], List[float]]:
    """
    Calculate velocity (first derivative) and acceleration (second derivative) with optional smoothing.
    
    Args:
        values: List of angle values over time
        smooth: Whether to apply smoothing before differentiation
        smooth_method: Smoothing method to use
        
    Returns:
        Tuple of (velocity, acceleration) lists
    """
    if len(values) < 2:
        return [], []
    
    # Apply smoothing if requested
    if smooth:
        smoothed_values = smooth_data(values, method=smooth_method)
    else:
        smoothed_values = values
    
    # Calculate velocity (first derivative)
    velocity = []
    for i in range(len(smoothed_values)):
        if pd.isna(smoothed_values[i]):
            velocity.append(np.nan)
            continue
            
        if i == 0:
            # Forward difference for first point
            if i + 1 < len(smoothed_values) and not pd.isna(smoothed_values[i + 1]):
                velocity.append(smoothed_values[i + 1] - smoothed_values[i])
            else:
                velocity.append(np.nan)
        elif i == len(smoothed_values) - 1:
            # Backward difference for last point
            if not pd.isna(smoothed_values[i - 1]):
                velocity.append(smoothed_values[i] - smoothed_values[i - 1])
            else:
                velocity.append(np.nan)
        else:
            # Central difference for middle points
            if not pd.isna(smoothed_values[i - 1]) and not pd.isna(smoothed_values[i + 1]):
                velocity.append((smoothed_values[i + 1] - smoothed_values[i - 1]) / 2)
            else:
                velocity.append(np.nan)
    
    # Apply additional smoothing to velocity if requested
    if smooth:
        velocity = smooth_data(velocity, method=smooth_method, window_length=7)
    
    # Calculate acceleration (second derivative)
    acceleration = []
    for i in range(len(velocity)):
        if pd.isna(velocity[i]):
            acceleration.append(np.nan)
            continue
            
        if i == 0:
            # Forward difference for first point
            if i + 1 < len(velocity) and not pd.isna(velocity[i + 1]):
                acceleration.append(velocity[i + 1] - velocity[i])
            else:
                acceleration.append(np.nan)
        elif i == len(velocity) - 1:
            # Backward difference for last point
            if not pd.isna(velocity[i - 1]):
                acceleration.append(velocity[i] - velocity[i - 1])
            else:
                acceleration.append(np.nan)
        else:
            # Central difference for middle points
            if not pd.isna(velocity[i - 1]) and not pd.isna(velocity[i + 1]):
                acceleration.append((velocity[i + 1] - velocity[i - 1]) / 2)
            else:
                acceleration.append(np.nan)
    
    # Apply additional smoothing to acceleration if requested
    if smooth:
        acceleration = smooth_data(acceleration, method=smooth_method, window_length=5)
    
    return velocity, acceleration

def process_joint_angles_data(df: pd.DataFrame, smooth: bool = True, smooth_method: str = 'savgol') -> pd.DataFrame:
    """
    Process the joint angles data to add velocity and acceleration columns with smoothing.
    
    Args:
        df: DataFrame with joint_angles column
        smooth: Whether to apply smoothing
        smooth_method: Smoothing method to use
        
    Returns:
        DataFrame with additional velocity and acceleration columns
    """
    # Joint angle names (based on the calculate_angles.py file)
    joint_names = [
        'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip', 'left_knee', 'right_knee'
    ]
    
    # Initialize new columns
    for joint in joint_names:
        if smooth:
            df[f'{joint}_smoothed'] = None
        df[f'{joint}_velocity'] = None
        df[f'{joint}_acceleration'] = None
    
    # Process each row
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing row {idx}/{len(df)}")
            
        joint_angles_data = parse_joint_angles(row['joint_angles'])
        
        if joint_angles_data is None:
            continue
        
        # Extract angle values for each joint over time
        joint_values = {joint: [] for joint in joint_names}
        
        for frame_data in joint_angles_data:
            for joint in joint_names:
                value = frame_data.get(joint)
                if value is not None and not pd.isna(value):
                    joint_values[joint].append(value)
                else:
                    joint_values[joint].append(np.nan)
        
        # Calculate derivatives for each joint
        for joint in joint_names:
            if len(joint_values[joint]) > 1:
                # Apply smoothing and calculate derivatives
                smoothed_values = smooth_data(joint_values[joint], method=smooth_method) if smooth else joint_values[joint]
                velocity, acceleration = calculate_derivatives(joint_values[joint], smooth=smooth, smooth_method=smooth_method)
                
                # Store results as JSON strings to maintain the list format
                if smooth:
                    df.at[idx, f'{joint}_smoothed'] = json.dumps(smoothed_values)
                df.at[idx, f'{joint}_velocity'] = json.dumps(velocity)
                df.at[idx, f'{joint}_acceleration'] = json.dumps(acceleration)
    
    return df

def main():
    """Main function to process the CSV file with smoothing options."""
    input_file = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles.csv'
    output_file = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_smoothed_velocity_acceleration.csv'
    
    print("Loading CSV file...")
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} rows")
    print("Processing joint angles data with smoothing...")
    
    # Process the data with smoothing
    # You can change these parameters:
    # - smooth=True/False: Enable/disable smoothing
    # - smooth_method: 'savgol', 'gaussian', 'moving_average', 'butterworth'
    df_processed = process_joint_angles_data(df, smooth=True, smooth_method='savgol')
    
    print("Saving processed data...")
    df_processed.to_csv(output_file, index=False)
    
    print(f"Processed data saved to: {output_file}")
    
    # Show summary of new columns
    joint_names = [
        'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
        'left_hip', 'right_hip', 'left_knee', 'right_knee'
    ]
    
    print("\nSummary of new columns:")
    for joint in joint_names:
        smoothed_col = f'{joint}_smoothed'
        velocity_col = f'{joint}_velocity'
        acceleration_col = f'{joint}_acceleration'
        
        smoothed_count = df_processed[smoothed_col].notna().sum() if smoothed_col in df_processed.columns else 0
        velocity_count = df_processed[velocity_col].notna().sum()
        acceleration_count = df_processed[acceleration_col].notna().sum()
        
        print(f"{joint}:")
        if smoothed_count > 0:
            print(f"  Smoothed values: {smoothed_count}")
        print(f"  Velocity values: {velocity_count}")
        print(f"  Acceleration values: {acceleration_count}")

if __name__ == "__main__":
    main() 