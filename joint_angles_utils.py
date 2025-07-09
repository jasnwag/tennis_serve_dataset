import pandas as pd
import numpy as np
import ast

def load_joint_angles(csv_path, row_index=None, use_100_frames=False):
    """
    Load joint_angles data from the converted CSV file
    
    Args:
        csv_path: Path to the converted CSV file
        row_index: Optional row index to load specific row
        use_100_frames: If True, load the interpolated 100-frame version
        
    Returns:
        If row_index is provided: numpy array with shape (n_frames, 8) or (100, 8)
        If row_index is None: list of numpy arrays for all rows
    """
    df = pd.read_csv(csv_path)
    
    column_name = 'joint_angles_100' if use_100_frames else 'joint_angles'
    
    if row_index is not None:
        if row_index >= len(df):
            raise ValueError(f"Row index {row_index} out of range. File has {len(df)} rows.")
        
        joint_angles_str = df.iloc[row_index][column_name]
        if pd.isna(joint_angles_str):
            return None
        
        return np.array(ast.literal_eval(joint_angles_str))
    else:
        results = []
        for _, row in df.iterrows():
            joint_angles_str = row[column_name]
            if pd.isna(joint_angles_str):
                results.append(None)
            else:
                results.append(np.array(ast.literal_eval(joint_angles_str)))
        return results

def load_joint_velocities(csv_path, row_index=None, use_100_frames=False):
    """
    Load joint_velocities data from the converted CSV file
    
    Args:
        csv_path: Path to the converted CSV file
        row_index: Optional row index to load specific row
        use_100_frames: If True, load the interpolated 100-frame version
        
    Returns:
        If row_index is provided: numpy array with shape (n_frames, 8) or (100, 8)
        If row_index is None: list of numpy arrays for all rows
    """
    df = pd.read_csv(csv_path)
    
    column_name = 'joint_velocities_100' if use_100_frames else 'joint_velocities'
    
    if row_index is not None:
        if row_index >= len(df):
            raise ValueError(f"Row index {row_index} out of range. File has {len(df)} rows.")
        
        joint_velocities_str = df.iloc[row_index][column_name]
        if pd.isna(joint_velocities_str):
            return None
        
        return np.array(ast.literal_eval(joint_velocities_str))
    else:
        results = []
        for _, row in df.iterrows():
            joint_velocities_str = row[column_name]
            if pd.isna(joint_velocities_str):
                results.append(None)
            else:
                results.append(np.array(ast.literal_eval(joint_velocities_str)))
        return results

def load_joint_accelerations(csv_path, row_index=None, use_100_frames=False):
    """
    Load joint_accelerations data from the converted CSV file
    
    Args:
        csv_path: Path to the converted CSV file
        row_index: Optional row index to load specific row
        use_100_frames: If True, load the interpolated 100-frame version
        
    Returns:
        If row_index is provided: numpy array with shape (n_frames, 8) or (100, 8)
        If row_index is None: list of numpy arrays for all rows
    """
    df = pd.read_csv(csv_path)
    
    column_name = 'joint_accelerations_100' if use_100_frames else 'joint_accelerations'
    
    if row_index is not None:
        if row_index >= len(df):
            raise ValueError(f"Row index {row_index} out of range. File has {len(df)} rows.")
        
        joint_accelerations_str = df.iloc[row_index][column_name]
        if pd.isna(joint_accelerations_str):
            return None
        
        return np.array(ast.literal_eval(joint_accelerations_str))
    else:
        results = []
        for _, row in df.iterrows():
            joint_accelerations_str = row[column_name]
            if pd.isna(joint_accelerations_str):
                results.append(None)
            else:
                results.append(np.array(ast.literal_eval(joint_accelerations_str)))
        return results

def get_joint_names():
    """
    Get the list of joint names in the order they appear in the arrays
    
    Returns:
        List of joint names: ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                             'left_hip', 'right_hip', 'left_knee', 'right_knee']
    """
    return ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
            'left_hip', 'right_hip', 'left_knee', 'right_knee']

def load_all_joint_data(csv_path, row_index=None, use_100_frames=False):
    """
    Load all joint data (angles, velocities, accelerations) for a specific row
    
    Args:
        csv_path: Path to the converted CSV file
        row_index: Row index to load
        use_100_frames: If True, load the interpolated 100-frame versions
        
    Returns:
        Dictionary with 'angles', 'velocities', 'accelerations' keys, each containing numpy arrays
    """
    angles = load_joint_angles(csv_path, row_index, use_100_frames)
    velocities = load_joint_velocities(csv_path, row_index, use_100_frames)
    accelerations = load_joint_accelerations(csv_path, row_index, use_100_frames)
    
    return {
        'angles': angles,
        'velocities': velocities,
        'accelerations': accelerations
    }

def get_data_statistics(csv_path, use_100_frames=False):
    """
    Get statistics about the joint data in the CSV file
    
    Args:
        csv_path: Path to the converted CSV file
        use_100_frames: If True, analyze the interpolated 100-frame versions
        
    Returns:
        Dictionary with statistics about the data
    """
    df = pd.read_csv(csv_path)
    
    # Count non-null values for each joint data column
    angles_col = 'joint_angles_100' if use_100_frames else 'joint_angles'
    velocities_col = 'joint_velocities_100' if use_100_frames else 'joint_velocities'
    accelerations_col = 'joint_accelerations_100' if use_100_frames else 'joint_accelerations'
    
    stats = {
        'total_rows': len(df),
        'angles_available': df[angles_col].notna().sum(),
        'velocities_available': df[velocities_col].notna().sum(),
        'accelerations_available': df[accelerations_col].notna().sum(),
        'use_100_frames': use_100_frames
    }
    
    # Get frame counts for a sample
    sample_angles = load_joint_angles(csv_path, 0, use_100_frames)
    if sample_angles is not None:
        stats['frame_count'] = sample_angles.shape[0]
        stats['joint_count'] = sample_angles.shape[1]
    
    return stats

if __name__ == "__main__":
    # Test the utility functions
    csv_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_velocity_acceleration_100_frames.csv'
    
    print("=== Joint Angles Utils Test ===")
    
    # Test loading original data
    print("\n1. Testing original data loading:")
    angles_orig = load_joint_angles(csv_path, 0, use_100_frames=False)
    velocities_orig = load_joint_velocities(csv_path, 0, use_100_frames=False)
    accelerations_orig = load_joint_accelerations(csv_path, 0, use_100_frames=False)
    
    print(f"Original angles shape: {angles_orig.shape}")
    print(f"Original velocities shape: {velocities_orig.shape}")
    print(f"Original accelerations shape: {accelerations_orig.shape}")
    
    # Test loading 100-frame data
    print("\n2. Testing 100-frame data loading:")
    angles_100 = load_joint_angles(csv_path, 0, use_100_frames=True)
    velocities_100 = load_joint_velocities(csv_path, 0, use_100_frames=True)
    accelerations_100 = load_joint_accelerations(csv_path, 0, use_100_frames=True)
    
    print(f"100-frame angles shape: {angles_100.shape}")
    print(f"100-frame velocities shape: {velocities_100.shape}")
    print(f"100-frame accelerations shape: {accelerations_100.shape}")
    
    # Test loading all data at once
    print("\n3. Testing load_all_joint_data:")
    all_data = load_all_joint_data(csv_path, 0, use_100_frames=True)
    print(f"All data keys: {list(all_data.keys())}")
    print(f"All data shapes: {[(k, v.shape) for k, v in all_data.items()]}")
    
    # Test statistics
    print("\n4. Testing data statistics:")
    stats = get_data_statistics(csv_path, use_100_frames=True)
    print(f"Data statistics: {stats}")
    
    # Test joint names
    print("\n5. Joint names:")
    joint_names = get_joint_names()
    print(f"Joint names: {joint_names}")
    
    print("\n=== All tests completed successfully! ===") 