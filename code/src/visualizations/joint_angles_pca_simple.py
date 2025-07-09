import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import warnings
warnings.filterwarnings('ignore')

def parse_joint_angles(joint_angles_str):
    """Parse the joint angles string into a numpy array"""
    try:
        # Convert string representation to actual list/array
        joint_angles = ast.literal_eval(joint_angles_str)
        # Flatten the array if it's nested
        if isinstance(joint_angles[0], list):
            joint_angles = np.array(joint_angles).flatten()
        else:
            joint_angles = np.array(joint_angles)
        return joint_angles
    except:
        return None

def standardize_data(data):
    """Standardize data to have zero mean and unit variance"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    return (data - mean) / std, mean, std

def pca_manual(data, n_components=2):
    """Perform PCA manually using numpy"""
    # Standardize the data
    data_std, mean, std = standardize_data(data)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_std.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the first n_components
    principal_components = eigenvectors[:, :n_components]
    
    # Transform the data
    pca_result = np.dot(data_std, principal_components)
    
    # Calculate explained variance ratio
    explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
    
    return pca_result, explained_variance_ratio, principal_components

def load_and_preprocess_data(file_path):
    """Load CSV data and preprocess joint angles"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Filter out rows where joint_angles_100 or server_name is missing
    df_clean = df.dropna(subset=['joint_angles_100', 'server_name'])
    print(f"After removing missing values: {df_clean.shape}")
    
    # Parse joint angles
    print("Parsing joint angles...")
    joint_angles_list = []
    valid_indices = []
    
    for idx, row in df_clean.iterrows():
        angles = parse_joint_angles(row['joint_angles_100'])
        if angles is not None and len(angles) > 0:
            joint_angles_list.append(angles)
            valid_indices.append(idx)
    
    if not joint_angles_list:
        raise ValueError("No valid joint angles data found")
    
    # Convert to numpy array - make sure all arrays have the same length
    min_length = min(len(angles) for angles in joint_angles_list)
    joint_angles_array = np.array([angles[:min_length] for angles in joint_angles_list])
    
    print(f"Joint angles array shape: {joint_angles_array.shape}")
    print(f"Using {min_length} features per instance")
    
    # Get corresponding server names
    server_names = df_clean.loc[valid_indices, 'server_name'].values
    
    return joint_angles_array, server_names, df_clean.loc[valid_indices]

def create_pca_visualization(pca_result, server_names, explained_variance_ratio, save_path=None):
    """Create PCA visualization colored by server name"""
    print("Creating PCA visualization...")
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'server_name': server_names
    })
    
    # Filter out instances where PC1 > 25
    print(f"Original number of instances: {len(pca_df)}")
    outlier_mask = pca_df['PC1'] > 25
    outliers_count = outlier_mask.sum()
    print(f"Removing {outliers_count} instances with PC1 > 25")
    
    pca_df_filtered = pca_df[~outlier_mask].copy()
    print(f"Remaining instances after filtering: {len(pca_df_filtered)}")
    
    # Get unique server names and create a color palette
    unique_servers = pca_df_filtered['server_name'].unique()
    print(f"Number of unique servers: {len(unique_servers)}")
    print(f"Servers: {unique_servers}")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Use a color palette with enough colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_servers)))
    
    # Create scatter plot for each server
    for i, server in enumerate(unique_servers):
        server_data = pca_df_filtered[pca_df_filtered['server_name'] == server]
        plt.scatter(server_data['PC1'], server_data['PC2'], 
                   c=[colors[i]], label=server, alpha=0.7, s=50)
    
    plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
    plt.title(f'PCA Visualization of Joint Angles by Server Name (PC1 ≤ 25)\n(Total explained variance: {sum(explained_variance_ratio):.1%})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return pca_df_filtered

def create_detailed_analysis(pca_df, pca_result, server_names, explained_variance_ratio):
    """Create additional analysis plots"""
    
    # Use the filtered data from pca_df
    filtered_server_names = pca_df['server_name'].values
    filtered_pc1 = pca_df['PC1'].values
    filtered_pc2 = pca_df['PC2'].values
    
    # Server distribution and PCA components
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Server distribution (filtered data)
    server_counts = pd.Series(filtered_server_names).value_counts()
    ax1.bar(range(len(server_counts)), server_counts.values)
    ax1.set_xlabel('Server')
    ax1.set_ylabel('Number of Instances')
    ax1.set_title('Distribution of Instances by Server (PC1 ≤ 25)')
    ax1.set_xticks(range(len(server_counts)))
    ax1.set_xticklabels(server_counts.index, rotation=45)
    
    # PCA component distribution (filtered data)
    ax2.hist(filtered_pc1, bins=30, alpha=0.7, label='PC1', color='blue')
    ax2.hist(filtered_pc2, bins=30, alpha=0.7, label='PC2', color='red')
    ax2.set_xlabel('Component Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of PCA Components (PC1 ≤ 25)')
    ax2.legend()
    
    # Explained variance
    ax3.bar(range(len(explained_variance_ratio)), explained_variance_ratio)
    ax3.set_xlabel('Principal Component')
    ax3.set_ylabel('Explained Variance Ratio')
    ax3.set_title('Explained Variance by Component')
    ax3.set_xticks(range(len(explained_variance_ratio)))
    ax3.set_xticklabels([f'PC{i+1}' for i in range(len(explained_variance_ratio))])
    
    # Box plot of PC1 by server (filtered data)
    unique_servers = np.unique(filtered_server_names)
    pc1_by_server = [filtered_pc1[filtered_server_names == server] for server in unique_servers]
    ax4.boxplot(pc1_by_server, labels=unique_servers)
    ax4.set_xlabel('Server')
    ax4.set_ylabel('PC1 Value')
    ax4.set_title('PC1 Distribution by Server (PC1 ≤ 25)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n=== Summary Statistics (After Filtering PC1 ≤ 25) ===")
    print(f"Total number of instances: {len(filtered_server_names)}")
    print(f"Number of unique servers: {len(np.unique(filtered_server_names))}")
    print(f"Explained variance ratios: {explained_variance_ratio}")
    print(f"Total explained variance: {sum(explained_variance_ratio):.3f}")
    
    print(f"\nServer distribution:")
    for server in np.unique(filtered_server_names):
        count = np.sum(filtered_server_names == server)
        percentage = (count / len(filtered_server_names)) * 100
        print(f"  {server}: {count} instances ({percentage:.1f}%)")

def main():
    # File path
    file_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv'
    
    try:
        # Load and preprocess data
        joint_angles_array, server_names, df_clean = load_and_preprocess_data(file_path)
        
        # Perform PCA analysis
        print("Performing PCA analysis...")
        pca_result, explained_variance_ratio, principal_components = pca_manual(joint_angles_array, n_components=2)
        
        # Create main PCA visualization
        save_path = 'joint_angles_pca_by_server.png'
        pca_df = create_pca_visualization(pca_result, server_names, explained_variance_ratio, save_path)
        
        # Create additional analysis
        create_detailed_analysis(pca_df, pca_result, server_names, explained_variance_ratio)
        
        # Save the PCA results to CSV for further analysis
        pca_df['original_index'] = df_clean.index
        pca_df.to_csv('joint_angles_pca_results.csv', index=False)
        print("PCA results saved to: joint_angles_pca_results.csv")
        
        return pca_df, principal_components
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pca_df, principal_components = main() 