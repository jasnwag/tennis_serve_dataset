#!/usr/bin/env python3
"""
Frames vs Serve Speed Analysis
=============================

This script analyzes the relationship between the number of frames in the video analysis
and the serve speed in mph to understand if there's a correlation between analysis duration
and serve performance.

Author: Tennis Analysis Project
Date: 2024
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

class FramesSpeedAnalyzer:
    """Analyzes the relationship between video analysis frames and serve speed."""
    
    def __init__(self, data_path):
        """Initialize the analyzer with data path."""
        self.data_path = data_path
        self.data = None
        self.clean_data = None
        
    def load_data(self):
        """Load the tennis dataset."""
        try:
            print("Loading tennis dataset...")
            # Increase CSV field size limit for large keypoints data
            import csv
            csv.field_size_limit(200000)
            
            self.data = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.data):,} records with {len(self.data.columns)} columns")
            
            # Display basic info
            print(f"  - Date range: {self.data['year'].min()} to {self.data['year'].max()}")
            print(f"  - Unique matches: {self.data['match_id'].nunique():,}")
            print(f"  - Unique players: {pd.concat([self.data['player1'], self.data['player2']]).nunique():,}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def clean_and_filter_data(self):
        """Clean and filter data for analysis."""
        print("\nCleaning and filtering data...")
        
        # Start with original data
        df = self.data.copy()
        
        print(f"  - Original records: {len(df):,}")
        
        # Filter for records with both Speed_MPH and n_frames data
        df = df.dropna(subset=['Speed_MPH', 'n_frames'])
        print(f"  - After removing missing speed/frames: {len(df):,}")
        
        # Filter for reasonable speed values (serve speeds typically 60-160 mph)
        df = df[(df['Speed_MPH'] >= 60) & (df['Speed_MPH'] <= 160)]
        print(f"  - After filtering reasonable speeds (60-160 mph): {len(df):,}")
        
        # Filter for reasonable frame counts (typically 10-300 frames)
        df = df[(df['n_frames'] >= 10) & (df['n_frames'] <= 300)]
        print(f"  - After filtering reasonable frame counts (10-300): {len(df):,}")
        
        # Remove outliers using IQR method
        def remove_outliers(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        
        # Remove outliers for both variables
        df = remove_outliers(df, 'Speed_MPH')
        df = remove_outliers(df, 'n_frames')
        print(f"  - After removing statistical outliers: {len(df):,}")
        
        self.clean_data = df
        
        if len(df) == 0:
            print("✗ No data remaining after filtering!")
            return False
        
        print(f"✓ Final dataset ready with {len(df):,} records")
        return True
    
    def descriptive_statistics(self):
        """Generate descriptive statistics for the variables."""
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)
        
        df = self.clean_data
        
        print("\nSERVE SPEED (MPH):")
        print("-" * 20)
        speed_stats = df['Speed_MPH'].describe()
        print(speed_stats)
        
        print(f"\nAdditional Speed Statistics:")
        print(f"  - Range: {df['Speed_MPH'].min():.1f} - {df['Speed_MPH'].max():.1f} mph")
        print(f"  - Variance: {df['Speed_MPH'].var():.2f}")
        print(f"  - Standard Deviation: {df['Speed_MPH'].std():.2f}")
        
        print("\nNUMBER OF FRAMES:")
        print("-" * 20)
        frames_stats = df['n_frames'].describe()
        print(frames_stats)
        
        print(f"\nAdditional Frames Statistics:")
        print(f"  - Range: {df['n_frames'].min():.0f} - {df['n_frames'].max():.0f} frames")
        print(f"  - Variance: {df['n_frames'].var():.2f}")
        print(f"  - Standard Deviation: {df['n_frames'].std():.2f}")
        
        # Gender breakdown
        print("\nGENDER DISTRIBUTION:")
        print("-" * 20)
        gender_dist = df['server_gender'].value_counts()
        print(gender_dist)
        print(f"Percentage distribution:")
        for gender, count in gender_dist.items():
            print(f"  - {gender}: {count/len(df)*100:.1f}%")
    
    def correlation_analysis(self):
        """Perform correlation analysis between frames and speed."""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        df = self.clean_data
        
        # Calculate correlations
        pearson_corr, pearson_p = pearsonr(df['n_frames'], df['Speed_MPH'])
        spearman_corr, spearman_p = spearmanr(df['n_frames'], df['Speed_MPH'])
        
        print(f"\nPEARSON CORRELATION:")
        print(f"  - Correlation coefficient: {pearson_corr:.4f}")
        print(f"  - P-value: {pearson_p:.6f}")
        print(f"  - Significance: {'Significant' if pearson_p < 0.05 else 'Not significant'} (α = 0.05)")
        
        print(f"\nSPEARMAN CORRELATION:")
        print(f"  - Correlation coefficient: {spearman_corr:.4f}")
        print(f"  - P-value: {spearman_p:.6f}")
        print(f"  - Significance: {'Significant' if spearman_p < 0.05 else 'Not significant'} (α = 0.05)")
        
        # Interpretation
        print(f"\nINTERPRETATION:")
        print(f"  - Correlation strength: {self._interpret_correlation(abs(pearson_corr))}")
        print(f"  - Direction: {'Positive' if pearson_corr > 0 else 'Negative'}")
        
        # R-squared
        r_squared = pearson_corr ** 2
        print(f"  - R-squared: {r_squared:.4f}")
        print(f"  - Variance explained: {r_squared*100:.2f}%")
        
        return pearson_corr, pearson_p, spearman_corr, spearman_p
    
    def _interpret_correlation(self, corr_value):
        """Interpret correlation coefficient magnitude."""
        if corr_value < 0.1:
            return "Negligible"
        elif corr_value < 0.3:
            return "Weak"
        elif corr_value < 0.5:
            return "Moderate"
        elif corr_value < 0.7:
            return "Strong"
        else:
            return "Very Strong"
    
    def regression_analysis(self):
        """Perform regression analysis."""
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS")
        print("="*60)
        
        df = self.clean_data
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(df['n_frames'], df['Speed_MPH'])
        
        print(f"\nLINEAR REGRESSION (Speed = f(Frames)):")
        print(f"  - Equation: Speed = {slope:.4f} × Frames + {intercept:.4f}")
        print(f"  - Slope: {slope:.4f} mph per frame")
        print(f"  - Intercept: {intercept:.4f} mph")
        print(f"  - R-squared: {r_value**2:.4f}")
        print(f"  - P-value: {p_value:.6f}")
        print(f"  - Standard Error: {std_err:.4f}")
        
        # Interpretation
        if slope > 0:
            print(f"\n  → For every additional frame, serve speed increases by {slope:.4f} mph on average")
        else:
            print(f"\n  → For every additional frame, serve speed decreases by {abs(slope):.4f} mph on average")
        
        return slope, intercept, r_value, p_value, std_err
    
    def gender_analysis(self):
        """Analyze the relationship by gender."""
        print("\n" + "="*60)
        print("GENDER-BASED ANALYSIS")
        print("="*60)
        
        df = self.clean_data
        
        for gender in df['server_gender'].unique():
            if pd.isna(gender):
                continue
                
            gender_data = df[df['server_gender'] == gender]
            
            print(f"\n{gender.upper()} PLAYERS:")
            print("-" * 20)
            print(f"  - Sample size: {len(gender_data):,}")
            
            # Correlation for this gender
            if len(gender_data) > 10:  # Minimum sample size
                corr, p_val = pearsonr(gender_data['n_frames'], gender_data['Speed_MPH'])
                print(f"  - Correlation: {corr:.4f}")
                print(f"  - P-value: {p_val:.6f}")
                print(f"  - Significance: {'Significant' if p_val < 0.05 else 'Not significant'}")
                
                # Means
                print(f"  - Average speed: {gender_data['Speed_MPH'].mean():.1f} mph")
                print(f"  - Average frames: {gender_data['n_frames'].mean():.1f}")
            else:
                print(f"  - Sample size too small for reliable analysis")
    
    def create_visualizations(self):
        """Create visualizations for the analysis."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        df = self.clean_data
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Frames vs Serve Speed Analysis', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot with regression line
        ax1 = axes[0, 0]
        sns.scatterplot(data=df, x='n_frames', y='Speed_MPH', alpha=0.6, ax=ax1)
        sns.regplot(data=df, x='n_frames', y='Speed_MPH', scatter=False, color='red', ax=ax1)
        ax1.set_title('Scatter Plot: Frames vs Speed')
        ax1.set_xlabel('Number of Frames')
        ax1.set_ylabel('Serve Speed (mph)')
        
        # Add correlation info
        corr, _ = pearsonr(df['n_frames'], df['Speed_MPH'])
        ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Distribution of frames
        ax2 = axes[0, 1]
        df['n_frames'].hist(bins=30, ax=ax2, alpha=0.7, color='skyblue')
        ax2.set_title('Distribution of Frame Counts')
        ax2.set_xlabel('Number of Frames')
        ax2.set_ylabel('Frequency')
        ax2.axvline(df['n_frames'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["n_frames"].mean():.1f}')
        ax2.legend()
        
        # 3. Distribution of speeds
        ax3 = axes[1, 0]
        df['Speed_MPH'].hist(bins=30, ax=ax3, alpha=0.7, color='lightgreen')
        ax3.set_title('Distribution of Serve Speeds')
        ax3.set_xlabel('Serve Speed (mph)')
        ax3.set_ylabel('Frequency')
        ax3.axvline(df['Speed_MPH'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Speed_MPH"].mean():.1f}')
        ax3.legend()
        
        # 4. Gender comparison
        ax4 = axes[1, 1]
        if 'server_gender' in df.columns and df['server_gender'].notna().any():
            sns.boxplot(data=df, x='server_gender', y='Speed_MPH', ax=ax4)
            ax4.set_title('Speed Distribution by Gender')
            ax4.set_xlabel('Server Gender')
            ax4.set_ylabel('Serve Speed (mph)')
        else:
            ax4.text(0.5, 0.5, 'Gender data not available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Gender Analysis')
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(os.path.dirname(self.data_path), '..', 'analysis_output')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'frames_vs_speed_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {plot_path}")
        
        # Display the plot
        plt.show()
        
        return plot_path
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("="*60)
        
        df = self.clean_data
        
        # Calculate key metrics
        pearson_corr, pearson_p, spearman_corr, spearman_p = self.correlation_analysis()
        slope, intercept, r_value, p_value, std_err = self.regression_analysis()
        
        print(f"\n" + "="*60)
        print("EXECUTIVE SUMMARY")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  - Total analyzed serves: {len(df):,}")
        print(f"  - Speed range: {df['Speed_MPH'].min():.1f} - {df['Speed_MPH'].max():.1f} mph")
        print(f"  - Frame range: {df['n_frames'].min():.0f} - {df['n_frames'].max():.0f} frames")
        print(f"  - Average speed: {df['Speed_MPH'].mean():.1f} mph")
        print(f"  - Average frames: {df['n_frames'].mean():.1f} frames")
        
        print(f"\nKey Findings:")
        print(f"  - Pearson correlation: {pearson_corr:.4f} ({self._interpret_correlation(abs(pearson_corr))})")
        print(f"  - Statistical significance: {'Yes' if pearson_p < 0.05 else 'No'} (p = {pearson_p:.6f})")
        print(f"  - Variance explained: {(pearson_corr**2)*100:.2f}%")
        
        if slope > 0:
            print(f"  - Relationship: Each additional frame associated with {slope:.4f} mph increase")
        else:
            print(f"  - Relationship: Each additional frame associated with {abs(slope):.4f} mph decrease")
        
        print(f"\nPractical Implications:")
        if abs(pearson_corr) > 0.3:
            print(f"  - There is a {self._interpret_correlation(abs(pearson_corr)).lower()} relationship between analysis duration and serve speed")
            if pearson_corr > 0:
                print(f"  - Longer video analyses tend to capture faster serves")
                print(f"  - This might indicate that faster serves require more detailed analysis")
            else:
                print(f"  - Longer video analyses tend to capture slower serves")
                print(f"  - This might indicate that slower serves require more detailed analysis")
        else:
            print(f"  - The relationship between analysis duration and serve speed is negligible")
            print(f"  - Video analysis duration appears largely independent of serve speed")
        
        print(f"\nRecommendations:")
        if abs(pearson_corr) > 0.3:
            print(f"  - Consider the relationship when designing analysis protocols")
            print(f"  - Account for speed-dependent analysis requirements")
        else:
            print(f"  - Analysis duration can be standardized regardless of expected serve speed")
            print(f"  - Focus on other factors for optimizing analysis protocols")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("TENNIS SERVE ANALYSIS: FRAMES vs SPEED")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Clean data
        if not self.clean_and_filter_data():
            return False
        
        # Run analyses
        self.descriptive_statistics()
        self.correlation_analysis()
        self.regression_analysis()
        self.gender_analysis()
        
        # Create visualizations
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        # Generate final report
        self.generate_report()
        
        return True

def main():
    """Main function to run the analysis."""
    print("Starting Frames vs Serve Speed Analysis...")
    
    # Create analyzer instance
    analyzer = FramesSpeedAnalyzer(DATA_PATH)
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✓ Analysis completed successfully!")
    else:
        print("\n✗ Analysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 