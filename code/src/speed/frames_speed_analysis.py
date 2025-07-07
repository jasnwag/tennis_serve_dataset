#!/usr/bin/env python3
"""
Analysis of the relationship between n_frames and Speed_MPH
with groupings by server characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_clean_data(file_path):
    """Load and clean the dataset."""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Remove rows with missing values in key columns
    initial_rows = len(df)
    df = df.dropna(subset=['n_frames', 'Speed_MPH', 'server_name', 'server_gender'])
    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} rows with missing data")
    print(f"Final dataset: {final_rows} rows")
    
    return df

def basic_statistics(df):
    """Generate basic statistics for the dataset."""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print("\nOverall Statistics:")
    print(f"Number of serves: {len(df)}")
    print(f"n_frames - Mean: {df['n_frames'].mean():.1f}, Std: {df['n_frames'].std():.1f}")
    print(f"n_frames - Range: {df['n_frames'].min():.0f} to {df['n_frames'].max():.0f}")
    print(f"Speed_MPH - Mean: {df['Speed_MPH'].mean():.1f}, Std: {df['Speed_MPH'].std():.1f}")
    print(f"Speed_MPH - Range: {df['Speed_MPH'].min():.0f} to {df['Speed_MPH'].max():.0f}")
    
    # Correlation analysis
    correlation_pearson, p_value_pearson = pearsonr(df['n_frames'], df['Speed_MPH'])
    correlation_spearman, p_value_spearman = spearmanr(df['n_frames'], df['Speed_MPH'])
    
    print(f"\nCorrelation Analysis:")
    print(f"Pearson correlation: {correlation_pearson:.4f} (p-value: {p_value_pearson:.6f})")
    print(f"Spearman correlation: {correlation_spearman:.4f} (p-value: {p_value_spearman:.6f})")
    
    return correlation_pearson, correlation_spearman

def analyze_by_server_gender(df):
    """Analyze the relationship by server gender."""
    print("\n" + "="*60)
    print("ANALYSIS BY SERVER GENDER")
    print("="*60)
    
    gender_stats = []
    
    for gender in sorted(df['server_gender'].unique()):
        subset = df[df['server_gender'] == gender]
        corr_p, p_val_p = pearsonr(subset['n_frames'], subset['Speed_MPH'])
        corr_s, p_val_s = spearmanr(subset['n_frames'], subset['Speed_MPH'])
        
        stats_dict = {
            'Gender': gender,
            'Count': len(subset),
            'Frames_Mean': subset['n_frames'].mean(),
            'Frames_Std': subset['n_frames'].std(),
            'Speed_Mean': subset['Speed_MPH'].mean(),
            'Speed_Std': subset['Speed_MPH'].std(),
            'Pearson_Corr': corr_p,
            'Pearson_P': p_val_p,
            'Spearman_Corr': corr_s,
            'Spearman_P': p_val_s
        }
        gender_stats.append(stats_dict)
        
        print(f"\n{gender} Players:")
        print(f"  Count: {len(subset)}")
        print(f"  Frames - Mean: {subset['n_frames'].mean():.1f} ± {subset['n_frames'].std():.1f}")
        print(f"  Speed - Mean: {subset['Speed_MPH'].mean():.1f} ± {subset['Speed_MPH'].std():.1f}")
        print(f"  Pearson correlation: {corr_p:.4f} (p={p_val_p:.6f})")
        print(f"  Spearman correlation: {corr_s:.4f} (p={p_val_s:.6f})")
    
    return pd.DataFrame(gender_stats)

def analyze_by_server_name(df, top_n=10):
    """Analyze the relationship by individual server names."""
    print("\n" + "="*60)
    print(f"ANALYSIS BY SERVER NAME (Top {top_n} by serve count)")
    print("="*60)
    
    # Get top servers by count
    server_counts = df['server_name'].value_counts()
    top_servers = server_counts.head(top_n).index
    
    server_stats = []
    
    for server in top_servers:
        subset = df[df['server_name'] == server]
        if len(subset) < 5:  # Skip servers with too few serves
            continue
            
        corr_p, p_val_p = pearsonr(subset['n_frames'], subset['Speed_MPH'])
        corr_s, p_val_s = spearmanr(subset['n_frames'], subset['Speed_MPH'])
        
        stats_dict = {
            'Server': server,
            'Gender': subset['server_gender'].iloc[0],
            'Count': len(subset),
            'Frames_Mean': subset['n_frames'].mean(),
            'Frames_Std': subset['n_frames'].std(),
            'Speed_Mean': subset['Speed_MPH'].mean(),
            'Speed_Std': subset['Speed_MPH'].std(),
            'Pearson_Corr': corr_p,
            'Pearson_P': p_val_p,
            'Spearman_Corr': corr_s,
            'Spearman_P': p_val_s
        }
        server_stats.append(stats_dict)
        
        print(f"\n{server} ({subset['server_gender'].iloc[0]}):")
        print(f"  Count: {len(subset)}")
        print(f"  Frames - Mean: {subset['n_frames'].mean():.1f} ± {subset['n_frames'].std():.1f}")
        print(f"  Speed - Mean: {subset['Speed_MPH'].mean():.1f} ± {subset['Speed_MPH'].std():.1f}")
        print(f"  Pearson correlation: {corr_p:.4f} (p={p_val_p:.6f})")
    
    return pd.DataFrame(server_stats)

def analyze_by_point_server(df):
    """Analyze the relationship by PointServer (1 or 2)."""
    print("\n" + "="*60)
    print("ANALYSIS BY POINT SERVER")
    print("="*60)
    
    server_stats = []
    
    for server_num in sorted(df['PointServer'].unique()):
        subset = df[df['PointServer'] == server_num]
        corr_p, p_val_p = pearsonr(subset['n_frames'], subset['Speed_MPH'])
        corr_s, p_val_s = spearmanr(subset['n_frames'], subset['Speed_MPH'])
        
        stats_dict = {
            'PointServer': server_num,
            'Count': len(subset),
            'Frames_Mean': subset['n_frames'].mean(),
            'Frames_Std': subset['n_frames'].std(),
            'Speed_Mean': subset['Speed_MPH'].mean(),
            'Speed_Std': subset['Speed_MPH'].std(),
            'Pearson_Corr': corr_p,
            'Pearson_P': p_val_p,
            'Spearman_Corr': corr_s,
            'Spearman_P': p_val_s
        }
        server_stats.append(stats_dict)
        
        print(f"\nPlayer {server_num}:")
        print(f"  Count: {len(subset)}")
        print(f"  Frames - Mean: {subset['n_frames'].mean():.1f} ± {subset['n_frames'].std():.1f}")
        print(f"  Speed - Mean: {subset['Speed_MPH'].mean():.1f} ± {subset['Speed_MPH'].std():.1f}")
        print(f"  Pearson correlation: {corr_p:.4f} (p={p_val_p:.6f})")
        print(f"  Spearman correlation: {corr_s:.4f} (p={p_val_s:.6f})")
    
    return pd.DataFrame(server_stats)

def create_visualizations(df, save_path_prefix):
    """Create comprehensive visualizations."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Overall scatter plot with regression line
    plt.figure(figsize=(12, 8))
    plt.scatter(df['n_frames'], df['Speed_MPH'], alpha=0.6, s=30)
    z = np.polyfit(df['n_frames'], df['Speed_MPH'], 1)
    p = np.poly1d(z)
    plt.plot(df['n_frames'], p(df['n_frames']), "r--", alpha=0.8, linewidth=2)
    plt.xlabel('Number of Frames')
    plt.ylabel('Speed (MPH)')
    plt.title('Relationship between Number of Frames and Serve Speed')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_overall_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. By gender
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plots by gender
    for i, gender in enumerate(sorted(df['server_gender'].unique())):
        subset = df[df['server_gender'] == gender]
        axes[i].scatter(subset['n_frames'], subset['Speed_MPH'], alpha=0.6, s=30)
        z = np.polyfit(subset['n_frames'], subset['Speed_MPH'], 1)
        p = np.poly1d(z)
        axes[i].plot(subset['n_frames'], p(subset['n_frames']), "r--", alpha=0.8, linewidth=2)
        axes[i].set_xlabel('Number of Frames')
        axes[i].set_ylabel('Speed (MPH)')
        axes[i].set_title(f'{gender} Players (n={len(subset)})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_by_gender.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plots comparing distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Frames by gender
    df.boxplot(column='n_frames', by='server_gender', ax=axes[0,0])
    axes[0,0].set_title('Frames Distribution by Gender')
    axes[0,0].set_xlabel('Server Gender')
    axes[0,0].set_ylabel('Number of Frames')
    
    # Speed by gender
    df.boxplot(column='Speed_MPH', by='server_gender', ax=axes[0,1])
    axes[0,1].set_title('Speed Distribution by Gender')
    axes[0,1].set_xlabel('Server Gender')
    axes[0,1].set_ylabel('Speed (MPH)')
    
    # Frames by PointServer
    df.boxplot(column='n_frames', by='PointServer', ax=axes[1,0])
    axes[1,0].set_title('Frames Distribution by Point Server')
    axes[1,0].set_xlabel('Point Server')
    axes[1,0].set_ylabel('Number of Frames')
    
    # Speed by PointServer
    df.boxplot(column='Speed_MPH', by='PointServer', ax=axes[1,1])
    axes[1,1].set_title('Speed Distribution by Point Server')
    axes[1,1].set_xlabel('Point Server')
    axes[1,1].set_ylabel('Speed (MPH)')
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Top servers analysis
    server_counts = df['server_name'].value_counts()
    top_servers = server_counts.head(8).index
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, server in enumerate(top_servers):
        subset = df[df['server_name'] == server]
        axes[i].scatter(subset['n_frames'], subset['Speed_MPH'], alpha=0.7, s=40)
        if len(subset) > 2:
            z = np.polyfit(subset['n_frames'], subset['Speed_MPH'], 1)
            p = np.poly1d(z)
            axes[i].plot(subset['n_frames'], p(subset['n_frames']), "r--", alpha=0.8, linewidth=2)
        axes[i].set_xlabel('Frames')
        axes[i].set_ylabel('Speed (MPH)')
        axes[i].set_title(f'{server} (n={len(subset)})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_top_servers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Correlation heatmap for top servers
    top_servers_corr = []
    top_servers_names = []
    
    for server in server_counts.head(10).index:
        subset = df[df['server_name'] == server]
        if len(subset) >= 5:
            corr, _ = pearsonr(subset['n_frames'], subset['Speed_MPH'])
            top_servers_corr.append(corr)
            top_servers_names.append(f"{server} (n={len(subset)})")
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(top_servers_names))
    colors = ['red' if x < 0 else 'blue' for x in top_servers_corr]
    bars = plt.barh(y_pos, top_servers_corr, color=colors, alpha=0.7)
    plt.yticks(y_pos, top_servers_names)
    plt.xlabel('Pearson Correlation (Frames vs Speed)')
    plt.title('Correlation between Frames and Speed by Top Servers')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, top_servers_corr)):
        plt.text(corr + 0.01 if corr >= 0 else corr - 0.01, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', ha='left' if corr >= 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations saved successfully!")

def save_results(gender_stats, server_stats, point_server_stats, save_path_prefix):
    """Save analysis results to CSV files."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save statistics
    gender_stats.to_csv(f'{save_path_prefix}_gender_analysis.csv', index=False)
    server_stats.to_csv(f'{save_path_prefix}_server_analysis.csv', index=False)
    point_server_stats.to_csv(f'{save_path_prefix}_point_server_analysis.csv', index=False)
    
    print(f"Results saved:")
    print(f"- {save_path_prefix}_gender_analysis.csv")
    print(f"- {save_path_prefix}_server_analysis.csv")
    print(f"- {save_path_prefix}_point_server_analysis.csv")

def main():
    """Main analysis function."""
    # File paths
    data_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_velocity_acceleration_backup.csv'
    save_path_prefix = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/frames_speed_analysis'
    
    # Load data
    df = load_and_clean_data(data_path)
    
    # Basic statistics
    overall_corr_p, overall_corr_s = basic_statistics(df)
    
    # Group analyses
    gender_stats = analyze_by_server_gender(df)
    server_stats = analyze_by_server_name(df, top_n=15)
    point_server_stats = analyze_by_point_server(df)
    
    # Create visualizations
    create_visualizations(df, save_path_prefix)
    
    # Save results
    save_results(gender_stats, server_stats, point_server_stats, save_path_prefix)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overall dataset: {len(df)} serves")
    print(f"Overall Pearson correlation: {overall_corr_p:.4f}")
    print(f"Overall Spearman correlation: {overall_corr_s:.4f}")
    print("\nKey findings:")
    
    # Gender differences
    male_data = gender_stats[gender_stats['Gender'] == 'M'].iloc[0]
    female_data = gender_stats[gender_stats['Gender'] == 'F'].iloc[0] if 'F' in gender_stats['Gender'].values else None
    
    print(f"- Male players: {male_data['Count']} serves, correlation = {male_data['Pearson_Corr']:.4f}")
    if female_data is not None:
        print(f"- Female players: {female_data['Count']} serves, correlation = {female_data['Pearson_Corr']:.4f}")
    
    # Strongest correlations
    strongest_positive = server_stats.loc[server_stats['Pearson_Corr'].idxmax()]
    strongest_negative = server_stats.loc[server_stats['Pearson_Corr'].idxmin()]
    
    print(f"- Strongest positive correlation: {strongest_positive['Server']} ({strongest_positive['Pearson_Corr']:.4f})")
    print(f"- Strongest negative correlation: {strongest_negative['Server']} ({strongest_negative['Pearson_Corr']:.4f})")
    
    print("\nAnalysis complete! Check the generated files for detailed results.")

if __name__ == "__main__":
    main() 