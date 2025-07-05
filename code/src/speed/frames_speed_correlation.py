#!/usr/bin/env python3
"""
Frames vs Serve Speed Analysis
=============================

This script analyzes the relationship between the number of frames in the video analysis
and the serve speed in mph to understand if there's a correlation between analysis duration
and serve performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import os
import sys

# Set up paths
DATA_PATH = os.path.expanduser(
    "~/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv"
)

def load_and_clean_data():
    """Load and clean the tennis dataset."""
    try:
        print("Loading tennis dataset...")
        # Increase CSV field size limit for large keypoints data
        import csv
        csv.field_size_limit(200000)
        
        data = pd.read_csv(DATA_PATH)
        print(f"✓ Loaded {len(data):,} records with {len(data.columns)} columns")
        
        # Clean data
        print("Cleaning data...")
        
        # Filter for records with both Speed_MPH and n_frames data
        clean_data = data.dropna(subset=['Speed_MPH', 'n_frames'])
        print(f"  - After removing missing values: {len(clean_data):,}")
        
        # Filter for reasonable values
        clean_data = clean_data[(clean_data['Speed_MPH'] >= 60) & (clean_data['Speed_MPH'] <= 160)]
        clean_data = clean_data[(clean_data['n_frames'] >= 10) & (clean_data['n_frames'] <= 300)]
        print(f"  - After filtering reasonable ranges: {len(clean_data):,}")
        
        return clean_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_correlation(data):
    """Analyze correlation between frames and speed."""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(data['n_frames'], data['Speed_MPH'])
    spearman_corr, spearman_p = spearmanr(data['n_frames'], data['Speed_MPH'])
    
    print(f"\nPearson Correlation: {pearson_corr:.4f}")
    print(f"P-value: {pearson_p:.6f}")
    print(f"Significance: {'Significant' if pearson_p < 0.05 else 'Not significant'}")
    
    print(f"\nSpearman Correlation: {spearman_corr:.4f}")
    print(f"P-value: {spearman_p:.6f}")
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['n_frames'], data['Speed_MPH'])
    
    print(f"\nLinear Regression:")
    print(f"  Equation: Speed = {slope:.4f} × Frames + {intercept:.4f}")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    return pearson_corr, pearson_p, slope, intercept, r_value

def create_visualizations(data):
    """Create visualizations for the analysis."""
    print("\nCreating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Frames vs Serve Speed Analysis', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    ax1 = axes[0, 0]
    sns.scatterplot(data=data, x='n_frames', y='Speed_MPH', alpha=0.6, ax=ax1)
    sns.regplot(data=data, x='n_frames', y='Speed_MPH', scatter=False, color='red', ax=ax1)
    ax1.set_title('Scatter Plot: Frames vs Speed')
    ax1.set_xlabel('Number of Frames')
    ax1.set_ylabel('Serve Speed (mph)')
    
    # Add correlation info
    corr, _ = pearsonr(data['n_frames'], data['Speed_MPH'])
    ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Distribution of frames
    ax2 = axes[0, 1]
    data['n_frames'].hist(bins=30, ax=ax2, alpha=0.7, color='skyblue')
    ax2.set_title('Distribution of Frame Counts')
    ax2.set_xlabel('Number of Frames')
    ax2.set_ylabel('Frequency')
    ax2.axvline(data['n_frames'].mean(), color='red', linestyle='--', 
               label=f'Mean: {data["n_frames"].mean():.1f}')
    ax2.legend()
    
    # 3. Distribution of speeds
    ax3 = axes[1, 0]
    data['Speed_MPH'].hist(bins=30, ax=ax3, alpha=0.7, color='lightgreen')
    ax3.set_title('Distribution of Serve Speeds')
    ax3.set_xlabel('Serve Speed (mph)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(data['Speed_MPH'].mean(), color='red', linestyle='--', 
               label=f'Mean: {data["Speed_MPH"].mean():.1f}')
    ax3.legend()
    
    # 4. Gender comparison if available
    ax4 = axes[1, 1]
    if 'server_gender' in data.columns and data['server_gender'].notna().any():
        sns.boxplot(data=data, x='server_gender', y='Speed_MPH', ax=ax4)
        ax4.set_title('Speed Distribution by Gender')
        ax4.set_xlabel('Server Gender')
        ax4.set_ylabel('Serve Speed (mph)')
    else:
        # Hexbin plot for density
        ax4.hexbin(data['n_frames'], data['Speed_MPH'], gridsize=20, cmap='Blues')
        ax4.set_title('Density Plot: Frames vs Speed')
        ax4.set_xlabel('Number of Frames')
        ax4.set_ylabel('Serve Speed (mph)')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.join(os.path.dirname(DATA_PATH), '..', 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'frames_vs_speed_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")
    
    plt.show()
    
    return plot_path

def generate_summary_report(data, pearson_corr, pearson_p, slope, intercept, r_value):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  - Total analyzed serves: {len(data):,}")
    print(f"  - Speed range: {data['Speed_MPH'].min():.1f} - {data['Speed_MPH'].max():.1f} mph")
    print(f"  - Frame range: {data['n_frames'].min():.0f} - {data['n_frames'].max():.0f} frames")
    print(f"  - Average speed: {data['Speed_MPH'].mean():.1f} mph")
    print(f"  - Average frames: {data['n_frames'].mean():.1f} frames")
    
    print(f"\nKey Findings:")
    print(f"  - Pearson correlation: {pearson_corr:.4f}")
    print(f"  - Statistical significance: {'Yes' if pearson_p < 0.05 else 'No'} (p = {pearson_p:.6f})")
    print(f"  - Variance explained: {(pearson_corr**2)*100:.2f}%")
    
    if slope > 0:
        print(f"  - Relationship: Each additional frame associated with {slope:.4f} mph increase")
    else:
        print(f"  - Relationship: Each additional frame associated with {abs(slope):.4f} mph decrease")
    
    # Interpretation
    if abs(pearson_corr) < 0.1:
        strength = "negligible"
    elif abs(pearson_corr) < 0.3:
        strength = "weak"
    elif abs(pearson_corr) < 0.5:
        strength = "moderate"
    elif abs(pearson_corr) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    print(f"\nInterpretation:")
    print(f"  - Correlation strength: {strength}")
    print(f"  - Direction: {'positive' if pearson_corr > 0 else 'negative'}")
    
    if abs(pearson_corr) > 0.3:
        print(f"  - There is a {strength} relationship between analysis duration and serve speed")
        if pearson_corr > 0:
            print(f"  - Longer video analyses tend to capture faster serves")
        else:
            print(f"  - Longer video analyses tend to capture slower serves")
    else:
        print(f"  - The relationship between analysis duration and serve speed is {strength}")
        print(f"  - Video analysis duration appears largely independent of serve speed")

def main():
    """Main function to run the analysis."""
    print("TENNIS SERVE ANALYSIS: FRAMES vs SPEED")
    print("="*60)
    
    # Load and clean data
    data = load_and_clean_data()
    if data is None:
        return 1
    
    # Analyze correlation
    pearson_corr, pearson_p, slope, intercept, r_value = analyze_correlation(data)
    
    # Create visualizations
    try:
        create_visualizations(data)
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
    
    # Generate summary report
    generate_summary_report(data, pearson_corr, pearson_p, slope, intercept, r_value)
    
    print("\n✓ Analysis completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 