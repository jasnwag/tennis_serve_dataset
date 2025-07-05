import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast


def calculate_3d_angle(A, B, C):
    """
    Calculate the angle between three points in 3D space.
    
    Args:
        A, B, C: 3D points (numpy arrays with x, y, z coordinates)
        
    Returns:
        angle in degrees (180 - degrees between vectors)
    """
    # Convert to numpy arrays to ensure consistent handling
    A = np.array(A[:3])  # Use x, y, z coordinates
    B = np.array(B[:3])
    C = np.array(C[:3])
    
    AB = B - A 
    CB = B - C
    dot_product = np.dot(AB, CB)
    mag_AB = np.linalg.norm(AB)
    mag_CB = np.linalg.norm(CB)
    
    if mag_AB == 0 or mag_CB == 0:
        return None
    
    # Handle potential numerical errors that could make cos_theta out of bounds
    cos_theta = max(-1.0, min(1.0, dot_product / (mag_AB * mag_CB)))
    theta = np.arccos(cos_theta)
    degrees = np.degrees(theta)
    
    return 180 - degrees


def calculate_joint_angles(csv_path):
    """
    Calculate joint angles for all frames in the dataset.
    
    Args:
        csv_path: Path to the CSV file containing keypoints data
        
    Returns:
        DataFrame with calculated joint angles for each frame
    """
    # Load the final dataset
    df = pd.read_csv(csv_path)
    
    # Parse the keypoints string and convert to list of coordinates
    keypoints_data = ast.literal_eval(df['keypoints_clean'][0])
    
    # Define joint connections for calculating 3D angles
    # Based on MMPose COCO 17-keypoint format:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    angles_config_3d = {
        'left_elbow': [5, 7, 9],     # Left shoulder, elbow, wrist
        'right_elbow': [6, 8, 10],   # Right shoulder, elbow, wrist  
        'left_shoulder': [7, 5, 11], # Left elbow, shoulder, hip
        'right_shoulder': [8, 6, 12], # Right elbow, shoulder, hip
        'left_hip': [5, 11, 13],     # Left shoulder, hip, knee
        'right_hip': [6, 12, 14],    # Right shoulder, hip, knee
        'left_knee': [11, 13, 15],   # Left hip, knee, ankle
        'right_knee': [12, 14, 16]   # Right hip, knee, ankle
    }
    
    # Calculate 3D angles for all frames
    all_frame_angles = []
    
    for frame_idx, frame_keypoints in enumerate(keypoints_data):
        frame_angles_3d = {'frame_id': frame_idx}
        
        for angle_name, indices in angles_config_3d.items():
            # Get the three points needed for this angle
            points = [frame_keypoints[i] for i in indices]
            
            # Only calculate angle if all points have reasonable confidence
            # (third value in each keypoint is confidence)
            if all(point[2] > 0.3 for point in points):
                angle = calculate_3d_angle(points[0], points[1], points[2])
                frame_angles_3d[angle_name] = angle
            else:
                frame_angles_3d[angle_name] = None
        
        all_frame_angles.append(frame_angles_3d)
    
    # Convert to DataFrame for easier analysis
    angles_3d_df = pd.DataFrame(all_frame_angles)
    
    return angles_3d_df


def main():
    """
    Main function to run the joint angle calculation algorithm.
    """
    # Path to the dataset
    csv_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv'
    
    # Calculate joint angles
    angles_3d_df = calculate_joint_angles(csv_path)
    
    # Print results
    print(f"Calculated 3D joint angles for {len(angles_3d_df)} frames")
    print("\nFirst 5 frames:")
    print(angles_3d_df.head())
    
    print(f"\nDataFrame shape: {angles_3d_df.shape}")
    print(f"Columns: {list(angles_3d_df.columns)}")
    
    # Display the full dataframe
    print("\nFull DataFrame:")
    print(angles_3d_df)
    
    return angles_3d_df


if __name__ == "__main__":
    angles_df = main() 