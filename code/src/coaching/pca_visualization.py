#!/usr/bin/env python3
"""
Interactive 3D PCA visualization for tennis serve analysis.
"""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

def load_pca_data(json_path):
    """
    Load and process PCA data from JSON file.
    
    Args:
        json_path: Path to the PCA JSON file
        
    Returns:
        DataFrame with processed PCA data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Add serve ID as a column
    df['serve_id'] = df.index
    
    # Extract player name from serve_id
    df['player'] = df['serve_id'].apply(lambda x: x.split('.')[0])
    
    # Create speed categories
    df['speed_category'] = pd.cut(df['Speed_MPH'], 
                                 bins=[0, 90, 100, 110, 120, 130, 140],
                                 labels=['<90', '90-100', '100-110', '110-120', '120-130', '>130'])
    
    return df

def create_3d_pca_plot(df, output_path=None):
    """
    Create an interactive 3D PCA plot.
    
    Args:
        df: DataFrame with PCA data
        output_path: Path to save the HTML plot (if None, will show in browser)
    """
    # Create the 3D scatter plot
    fig = px.scatter_3d(df, 
                        x='PC1', y='PC2', z='PC3',
                        color='gender',
                        symbol='speed_category',
                        hover_data=['player', 'Speed_MPH', 'serve_id'],
                        title='3D PCA Visualization of Tennis Serves',
                        labels={'PC1': 'Principal Component 1',
                               'PC2': 'Principal Component 2',
                               'PC3': 'Principal Component 3'},
                        color_discrete_map={'M': 'blue', 'W': 'red'})
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title=dict(
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add speed legend
    speed_legend = go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10),
        name='Serve Speed (MPH)',
        showlegend=True
    )
    
    for speed in df['speed_category'].unique():
        speed_legend = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=10, symbol='circle'),
            name=f'Speed: {speed}',
            showlegend=True
        )
        fig.add_trace(speed_legend)
    
    # Save or show the plot
    if output_path:
        fig.write_html(output_path)
        print(f"Plot saved to {output_path}")
    else:
        fig.show()

def main():
    # Path to PCA data
    pca_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/previous/pose_youtube/code/analysis/initial_5000/pca_data_removed_errors.json"
    
    # Create output directory
    output_dir = Path("/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/USTA/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    print("Loading PCA data...")
    df = load_pca_data(pca_path)
    
    # Create visualization
    print("Creating 3D PCA visualization...")
    output_path = output_dir / "pca_visualization.html"
    create_3d_pca_plot(df, output_path)
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total serves: {len(df)}")
    print("\nServes by gender:")
    print(df['gender'].value_counts())
    print("\nSpeed distribution:")
    print(df['speed_category'].value_counts().sort_index())
    
    # Calculate and print speed statistics by gender
    print("\nSpeed statistics by gender:")
    speed_stats = df.groupby('gender')['Speed_MPH'].agg(['mean', 'std', 'min', 'max'])
    print(speed_stats)

if __name__ == "__main__":
    main() 