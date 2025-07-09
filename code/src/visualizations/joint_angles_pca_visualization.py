import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        if angles is not None:
            joint_angles_list.append(angles)
            valid_indices.append(idx)
    
    if not joint_angles_list:
        raise ValueError("No valid joint angles data found")
    
    # Convert to numpy array
    joint_angles_array = np.array(joint_angles_list)
    print(f"Joint angles array shape: {joint_angles_array.shape}")
    
    # Get corresponding server names
    server_names = df_clean.loc[valid_indices, 'server_name'].values
    
    return joint_angles_array, server_names, df_clean.loc[valid_indices]

def perform_pca_analysis(joint_angles_array, n_components=2):
    """Perform PCA analysis on joint angles data"""
    print("Performing PCA analysis...")
    
    # Standardize the data
    scaler = StandardScaler()
    joint_angles_scaled = scaler.fit_transform(joint_angles_array)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(joint_angles_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    return pca_result, pca, scaler

def create_pca_visualization(pca_result, server_names, save_path=None):
    """Create PCA visualization colored by server name"""
    print("Creating PCA visualization...")
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'server_name': server_names
    })
    
    # Get unique server names and create a color palette
    unique_servers = pca_df['server_name'].unique()
    print(f"Number of unique servers: {len(unique_servers)}")
    print(f"Servers: {unique_servers}")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Use a color palette with enough colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_servers)))
    
    # Create scatter plot for each server
    for i, server in enumerate(unique_servers):
        server_data = pca_df[pca_df['server_name'] == server]
        plt.scatter(server_data['PC1'], server_data['PC2'], 
                   c=[colors[i]], label=server, alpha=0.7, s=50)
    
    plt.xlabel(f'First Principal Component (PC1)')
    plt.ylabel(f'Second Principal Component (PC2)')
    plt.title('PCA Visualization of Joint Angles (100 frames) by Server Name')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return pca_df

def create_detailed_analysis(pca_df, pca_result, server_names):
    """Create additional analysis plots"""
    
    # Server distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    server_counts = pd.Series(server_names).value_counts()
    plt.bar(range(len(server_counts)), server_counts.values)
    plt.xlabel('Server Index')
    plt.ylabel('Number of Instances')
    plt.title('Distribution of Instances by Server')
    plt.xticks(range(len(server_counts)), [f'S{i+1}' for i in range(len(server_counts))], rotation=45)
    
    # PCA component distribution
    plt.subplot(1, 2, 2)
    plt.hist(pca_result[:, 0], bins=30, alpha=0.7, label='PC1')
    plt.hist(pca_result[:, 1], bins=30, alpha=0.7, label='PC2')
    plt.xlabel('Component Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of PCA Components')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total number of instances: {len(pca_result)}")
    print(f"Number of unique servers: {len(np.unique(server_names))}")
    print(f"Server names: {np.unique(server_names)}")
    
    for server in np.unique(server_names):
        count = np.sum(server_names == server)
        print(f"  {server}: {count} instances")

def main():
    # File path
    file_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv'
    
    try:
        # Load and preprocess data
        joint_angles_array, server_names, df_clean = load_and_preprocess_data(file_path)
        
        # Perform PCA analysis
        pca_result, pca, scaler = perform_pca_analysis(joint_angles_array)
        
        # Create main PCA visualization
        save_path = 'joint_angles_pca_by_server.png'
        pca_df = create_pca_visualization(pca_result, server_names, save_path)
        
        # Create additional analysis
        create_detailed_analysis(pca_df, pca_result, server_names)
        
        # Save the PCA results to CSV for further analysis
        pca_df['original_index'] = df_clean.index
        pca_df.to_csv('joint_angles_pca_results.csv', index=False)
        print("PCA results saved to: joint_angles_pca_results.csv")
        
        return pca_df, pca, scaler
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pca_df, pca, scaler = main() 