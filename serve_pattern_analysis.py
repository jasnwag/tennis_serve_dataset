import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class ServePatternAnalyzer:
    def __init__(self, csv_path):
        """Initialize the analyzer with tennis serve data."""
        self.df = pd.read_csv(csv_path)
        self.clean_data()
        
    def clean_data(self):
        """Clean and prepare the data."""
        print("Cleaning data...")
        # Remove rows with missing data
        self.df_clean = self.df[
            self.df['joint_angles_100'].notna() & 
            (self.df['joint_angles_100'] != '') & 
            self.df['Speed_MPH'].notna()
        ].copy()
        
        print(f"Original data: {len(self.df)} rows")
        print(f"Clean data: {len(self.df_clean)} rows")
        print(f"Speed range: {self.df_clean['Speed_MPH'].min():.1f} - {self.df_clean['Speed_MPH'].max():.1f} mph")
        
    def parse_joint_angles(self, angles_str):
        """Parse joint angles string to numpy array."""
        try:
            angles = ast.literal_eval(angles_str)
            return np.array(angles)
        except:
            return None
    
    def categorize_serves(self, percentiles=(33, 67)):
        """Categorize serves by speed into low, medium, high."""
        speeds = self.df_clean['Speed_MPH']
        low_thresh = np.percentile(speeds, percentiles[0])
        high_thresh = np.percentile(speeds, percentiles[1])
        
        def categorize(speed):
            if speed <= low_thresh:
                return 'Low'
            elif speed <= high_thresh:
                return 'Medium'
            else:
                return 'High'
        
        self.df_clean['speed_category'] = speeds.apply(categorize)
        
        print(f"\nSpeed Categories:")
        print(f"Low: ≤ {low_thresh:.1f} mph")
        print(f"Medium: {low_thresh:.1f} - {high_thresh:.1f} mph")  
        print(f"High: > {high_thresh:.1f} mph")
        print(f"\nCategory distribution:")
        print(self.df_clean['speed_category'].value_counts())
        
        return low_thresh, high_thresh
    
    def analyze_joint_angle_patterns(self):
        """Analyze joint angle patterns across different speed categories."""
        print("\nAnalyzing joint angle patterns...")
        
        # Parse all joint angles
        joint_angles_data = []
        speeds = []
        categories = []
        
        for _, row in self.df_clean.iterrows():
            angles = self.parse_joint_angles(row['joint_angles_100'])
            if angles is not None:
                # Flatten the angle data
                if angles.ndim == 3:  # (frames, joints, coords)
                    angles_flat = angles.reshape(angles.shape[0], -1)  # (frames, features)
                else:
                    angles_flat = angles.reshape(1, -1)  # Single frame
                
                joint_angles_data.append(angles_flat)
                speeds.append(row['Speed_MPH'])
                categories.append(row['speed_category'])
        
        return joint_angles_data, speeds, categories
    
    def find_key_frames(self, joint_angles_data, speeds, categories):
        """Identify key frames in the serve motion that correlate with speed."""
        print("\nFinding key frames...")
        
        high_speed_serves = []
        low_speed_serves = []
        
        for i, (angles, category) in enumerate(zip(joint_angles_data, categories)):
            if category == 'High':
                high_speed_serves.append(angles)
            elif category == 'Low':
                low_speed_serves.append(angles)
        
        if not high_speed_serves or not low_speed_serves:
            print("Insufficient data for comparison")
            return None
        
        # Calculate average patterns
        high_speed_avg = np.mean(high_speed_serves, axis=0)
        low_speed_avg = np.mean(low_speed_serves, axis=0)
        
        # Find frames with largest differences
        frame_differences = np.abs(high_speed_avg - low_speed_avg)
        avg_frame_diff = np.mean(frame_differences, axis=1)
        
        # Find peaks in differences (key moments)
        peaks, _ = find_peaks(avg_frame_diff, height=np.percentile(avg_frame_diff, 70))
        
        print(f"Key frames identified: {peaks}")
        return peaks, high_speed_avg, low_speed_avg, avg_frame_diff
    
    def cluster_serve_patterns(self, joint_angles_data, n_clusters=5):
        """Cluster serves based on their joint angle patterns."""
        print(f"\nClustering serve patterns into {n_clusters} groups...")
        
        # Flatten all angle data for clustering
        flattened_data = []
        for angles in joint_angles_data:
            flattened_data.append(angles.flatten())
        
        X = np.array(flattened_data)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=50)  # Reduce to 50 components
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        
        return clusters, pca, scaler, kmeans
    
    def analyze_biomechanical_efficiency(self, joint_angles_data, speeds):
        """Analyze biomechanical efficiency patterns."""
        print("\nAnalyzing biomechanical efficiency...")
        
        efficiency_metrics = []
        
        for angles, speed in zip(joint_angles_data, speeds):
            if angles.ndim >= 2:
                # Calculate motion smoothness (less jerk = more efficient)
                velocity = np.diff(angles, axis=0)
                acceleration = np.diff(velocity, axis=0)
                jerk = np.diff(acceleration, axis=0)
                
                # Smoothness metric (lower is better)
                smoothness = np.mean(np.std(jerk, axis=0))
                
                # Range of motion
                rom = np.mean(np.ptp(angles, axis=0))
                
                # Consistency (lower std = more consistent)
                consistency = np.mean(np.std(angles, axis=0))
                
                efficiency_metrics.append({
                    'speed': speed,
                    'smoothness': smoothness,
                    'range_of_motion': rom,
                    'consistency': consistency
                })
        
        return pd.DataFrame(efficiency_metrics)
    
    def plot_speed_distribution(self):
        """Plot speed distribution and categories."""
        plt.figure(figsize=(12, 8))
        
        # Overall distribution
        plt.subplot(2, 2, 1)
        plt.hist(self.df_clean['Speed_MPH'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Serve Speed Distribution')
        plt.xlabel('Speed (mph)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # By category
        plt.subplot(2, 2, 2)
        for category in ['Low', 'Medium', 'High']:
            data = self.df_clean[self.df_clean['speed_category'] == category]['Speed_MPH']
            plt.hist(data, alpha=0.6, label=category, bins=15)
        plt.title('Speed Distribution by Category')
        plt.xlabel('Speed (mph)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.df_clean, x='speed_category', y='Speed_MPH')
        plt.title('Speed by Category')
        plt.grid(True, alpha=0.3)
        
        # Statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""
        Total Serves: {len(self.df_clean)}
        
        Speed Statistics:
        Mean: {self.df_clean['Speed_MPH'].mean():.1f} mph
        Std: {self.df_clean['Speed_MPH'].std():.1f} mph
        Min: {self.df_clean['Speed_MPH'].min():.1f} mph
        Max: {self.df_clean['Speed_MPH'].max():.1f} mph
        
        Category Counts:
        {self.df_clean['speed_category'].value_counts().to_string()}
        """
        plt.text(0.1, 0.1, stats_text, fontsize=10, verticalalignment='bottom')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('serve_speed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_key_frames_analysis(self, peaks, high_speed_avg, low_speed_avg, avg_frame_diff):
        """Plot key frames analysis."""
        plt.figure(figsize=(15, 10))
        
        # Frame differences
        plt.subplot(2, 2, 1)
        plt.plot(avg_frame_diff)
        plt.scatter(peaks, avg_frame_diff[peaks], color='red', s=100, zorder=5)
        plt.title('Average Frame Differences (High vs Low Speed)')
        plt.xlabel('Frame Number')
        plt.ylabel('Average Difference')
        plt.grid(True, alpha=0.3)
        
        # High vs low speed patterns (first few features)
        plt.subplot(2, 2, 2)
        n_features_to_show = min(10, high_speed_avg.shape[1])
        frames = range(high_speed_avg.shape[0])
        
        for i in range(n_features_to_show):
            plt.plot(frames, high_speed_avg[:, i], 'r-', alpha=0.5, linewidth=0.5)
            plt.plot(frames, low_speed_avg[:, i], 'b-', alpha=0.5, linewidth=0.5)
        
        plt.title('Joint Angle Patterns\n(Red: High Speed, Blue: Low Speed)')
        plt.xlabel('Frame Number')
        plt.ylabel('Joint Angle Value')
        plt.grid(True, alpha=0.3)
        
        # Key frames highlighted
        plt.subplot(2, 2, 3)
        plt.plot(avg_frame_diff, 'k-', linewidth=2)
        for peak in peaks:
            plt.axvline(x=peak, color='red', linestyle='--', alpha=0.7)
            plt.text(peak, avg_frame_diff[peak], f'Frame {peak}', 
                    rotation=90, verticalalignment='bottom')
        plt.title('Key Moments in Serve Motion')
        plt.xlabel('Frame Number')
        plt.ylabel('Difference Magnitude')
        plt.grid(True, alpha=0.3)
        
        # Key frame analysis text
        plt.subplot(2, 2, 4)
        analysis_text = f"""
        Key Frames Analysis:
        
        Total frames analyzed: {len(avg_frame_diff)}
        Key frames identified: {len(peaks)}
        
        Key frame positions: {peaks.tolist()}
        
        These frames represent moments where
        high-speed and low-speed serves show
        the greatest differences in joint angles.
        
        Potential interpretations:
        - Early frames: Preparation phase
        - Middle frames: Power generation
        - Late frames: Follow-through
        """
        plt.text(0.1, 0.1, analysis_text, fontsize=10, verticalalignment='bottom')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('key_frames_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cluster_analysis(self, clusters, speeds):
        """Plot cluster analysis results."""
        plt.figure(figsize=(15, 10))
        
        # Cluster distribution
        plt.subplot(2, 3, 1)
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        plt.bar(unique_clusters, counts)
        plt.title('Cluster Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Serves')
        plt.grid(True, alpha=0.3)
        
        # Speed by cluster
        plt.subplot(2, 3, 2)
        cluster_speeds = []
        for cluster_id in unique_clusters:
            cluster_speeds.append([speeds[i] for i in range(len(speeds)) if clusters[i] == cluster_id])
        
        plt.boxplot(cluster_speeds, labels=unique_clusters)
        plt.title('Speed Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Speed (mph)')
        plt.grid(True, alpha=0.3)
        
        # Scatter plot: cluster vs speed
        plt.subplot(2, 3, 3)
        plt.scatter(clusters, speeds, alpha=0.6)
        plt.title('Cluster vs Speed')
        plt.xlabel('Cluster')
        plt.ylabel('Speed (mph)')
        plt.grid(True, alpha=0.3)
        
        # Cluster statistics
        plt.subplot(2, 3, 4)
        cluster_stats = []
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_speed_data = np.array(speeds)[cluster_mask]
            cluster_stats.append({
                'cluster': cluster_id,
                'count': np.sum(cluster_mask),
                'mean_speed': np.mean(cluster_speed_data),
                'std_speed': np.std(cluster_speed_data),
                'max_speed': np.max(cluster_speed_data)
            })
        
        stats_df = pd.DataFrame(cluster_stats)
        plt.table(cellText=stats_df.round(2).values,
                 colLabels=['Cluster', 'Count', 'Mean Speed', 'Std Speed', 'Max Speed'],
                 cellLoc='center',
                 loc='center')
        plt.axis('off')
        plt.title('Cluster Statistics')
        
        # High-speed cluster identification
        plt.subplot(2, 3, 5)
        best_cluster = stats_df.loc[stats_df['mean_speed'].idxmax(), 'cluster']
        worst_cluster = stats_df.loc[stats_df['mean_speed'].idxmin(), 'cluster']
        
        analysis_text = f"""
        Cluster Analysis Results:
        
        Best performing cluster: {best_cluster}
        (Highest average speed: {stats_df.loc[stats_df['cluster'] == best_cluster, 'mean_speed'].iloc[0]:.1f} mph)
        
        Lowest performing cluster: {worst_cluster}
        (Lowest average speed: {stats_df.loc[stats_df['cluster'] == worst_cluster, 'mean_speed'].iloc[0]:.1f} mph)
        
        Speed difference: {stats_df['mean_speed'].max() - stats_df['mean_speed'].min():.1f} mph
        
        This suggests distinct serve patterns
        that correlate with performance.
        """
        plt.text(0.1, 0.1, analysis_text, fontsize=10, verticalalignment='bottom')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return stats_df
    
    def plot_biomechanical_analysis(self, efficiency_df):
        """Plot biomechanical efficiency analysis."""
        plt.figure(figsize=(15, 12))
        
        # Smoothness vs Speed
        plt.subplot(2, 3, 1)
        plt.scatter(efficiency_df['smoothness'], efficiency_df['speed'], alpha=0.6)
        correlation = efficiency_df['smoothness'].corr(efficiency_df['speed'])
        plt.title(f'Smoothness vs Speed\n(Correlation: {correlation:.3f})')
        plt.xlabel('Smoothness (lower = better)')
        plt.ylabel('Speed (mph)')
        plt.grid(True, alpha=0.3)
        
        # Range of Motion vs Speed
        plt.subplot(2, 3, 2)
        plt.scatter(efficiency_df['range_of_motion'], efficiency_df['speed'], alpha=0.6)
        correlation = efficiency_df['range_of_motion'].corr(efficiency_df['speed'])
        plt.title(f'Range of Motion vs Speed\n(Correlation: {correlation:.3f})')
        plt.xlabel('Range of Motion')
        plt.ylabel('Speed (mph)')
        plt.grid(True, alpha=0.3)
        
        # Consistency vs Speed
        plt.subplot(2, 3, 3)
        plt.scatter(efficiency_df['consistency'], efficiency_df['speed'], alpha=0.6)
        correlation = efficiency_df['consistency'].corr(efficiency_df['speed'])
        plt.title(f'Consistency vs Speed\n(Correlation: {correlation:.3f})')
        plt.xlabel('Consistency (lower = better)')
        plt.ylabel('Speed (mph)')
        plt.grid(True, alpha=0.3)
        
        # Speed categories comparison
        high_speed_threshold = np.percentile(efficiency_df['speed'], 75)
        efficiency_df['speed_category'] = efficiency_df['speed'].apply(
            lambda x: 'High' if x >= high_speed_threshold else 'Low'
        )
        
        metrics = ['smoothness', 'range_of_motion', 'consistency']
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, 4 + i)
            high_speed_data = efficiency_df[efficiency_df['speed_category'] == 'High'][metric]
            low_speed_data = efficiency_df[efficiency_df['speed_category'] == 'Low'][metric]
            
            plt.boxplot([low_speed_data, high_speed_data], labels=['Low Speed', 'High Speed'])
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.ylabel(metric.replace("_", " ").title())
            plt.grid(True, alpha=0.3)
            
            # Statistical test
            statistic, p_value = stats.ttest_ind(high_speed_data, low_speed_data)
            plt.text(1.5, plt.ylim()[1] * 0.9, f'p-value: {p_value:.3f}', 
                    horizontalalignment='center')
        
        plt.tight_layout()
        plt.savefig('biomechanical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_recommendations(self, efficiency_df, cluster_stats, key_frames):
        """Generate coaching recommendations based on analysis."""
        print("\n" + "="*60)
        print("COACHING RECOMMENDATIONS FOR HIGH-SPEED SERVES")
        print("="*60)
        
        # Speed insights
        speed_stats = self.df_clean.groupby('speed_category')['Speed_MPH'].agg(['mean', 'std', 'count'])
        print(f"\n1. SPEED CATEGORIES:")
        print(speed_stats)
        
        # Biomechanical insights
        high_speed_threshold = np.percentile(efficiency_df['speed'], 75)
        high_speed_serves = efficiency_df[efficiency_df['speed'] >= high_speed_threshold]
        low_speed_serves = efficiency_df[efficiency_df['speed'] < high_speed_threshold]
        
        print(f"\n2. BIOMECHANICAL PATTERNS:")
        print(f"High-speed serves (≥{high_speed_threshold:.1f} mph):")
        print(f"  - Average smoothness: {high_speed_serves['smoothness'].mean():.3f}")
        print(f"  - Average range of motion: {high_speed_serves['range_of_motion'].mean():.3f}")
        print(f"  - Average consistency: {high_speed_serves['consistency'].mean():.3f}")
        
        print(f"\nLow-speed serves (<{high_speed_threshold:.1f} mph):")
        print(f"  - Average smoothness: {low_speed_serves['smoothness'].mean():.3f}")
        print(f"  - Average range of motion: {low_speed_serves['range_of_motion'].mean():.3f}")
        print(f"  - Average consistency: {low_speed_serves['consistency'].mean():.3f}")
        
        # Key frame insights
        print(f"\n3. KEY MOMENTS IN SERVE MOTION:")
        print(f"Critical frames identified: {key_frames}")
        print("These represent moments where technique differs most between fast and slow serves.")
        
        # Best cluster insights
        best_cluster = cluster_stats.loc[cluster_stats['mean_speed'].idxmax()]
        print(f"\n4. OPTIMAL SERVE PATTERN:")
        print(f"Cluster {best_cluster['cluster']} shows the highest average speed:")
        print(f"  - Average speed: {best_cluster['mean_speed']:.1f} mph")
        print(f"  - Consistency: ±{best_cluster['std_speed']:.1f} mph")
        print(f"  - Number of serves: {best_cluster['count']}")
        
        # Correlations
        correlations = efficiency_df[['smoothness', 'range_of_motion', 'consistency']].corrwith(efficiency_df['speed'])
        print(f"\n5. TECHNIQUE CORRELATIONS WITH SPEED:")
        for metric, corr in correlations.items():
            direction = "positively" if corr > 0 else "negatively"
            strength = "strongly" if abs(corr) > 0.5 else "moderately" if abs(corr) > 0.3 else "weakly"
            print(f"  - {metric.replace('_', ' ').title()}: {strength} {direction} correlated (r={corr:.3f})")
        
        print(f"\n6. ACTIONABLE RECOMMENDATIONS:")
        
        # Smoothness recommendation
        if correlations['smoothness'] < -0.2:
            print("  ✓ Focus on smoother motion - reduce jerkiness in movement")
        
        # Range of motion recommendation
        if correlations['range_of_motion'] > 0.2:
            print("  ✓ Increase range of motion in key joints")
        elif correlations['range_of_motion'] < -0.2:
            print("  ✓ Focus on controlled, efficient movement patterns")
        
        # Consistency recommendation
        if correlations['consistency'] < -0.2:
            print("  ✓ Work on movement consistency and repeatability")
        
        # Key frames recommendation
        if len(key_frames) > 0:
            early_frames = [f for f in key_frames if f < 30]
            mid_frames = [f for f in key_frames if 30 <= f < 70]
            late_frames = [f for f in key_frames if f >= 70]
            
            if early_frames:
                print(f"  ✓ Focus on preparation phase (frames {early_frames})")
            if mid_frames:
                print(f"  ✓ Optimize power generation phase (frames {mid_frames})")
            if late_frames:
                print(f"  ✓ Improve follow-through technique (frames {late_frames})")
        
        print(f"\n7. NEXT STEPS:")
        print("  • Analyze video of serves in the optimal cluster")
        print("  • Focus training on identified key frames")
        print("  • Monitor biomechanical metrics during practice")
        print("  • Compare individual serves to optimal patterns")

def main():
    """Main analysis function."""
    csv_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv'
    
    # Initialize analyzer
    analyzer = ServePatternAnalyzer(csv_path)
    
    # Categorize serves by speed
    low_thresh, high_thresh = analyzer.categorize_serves()
    
    # Plot speed distribution
    analyzer.plot_speed_distribution()
    
    # Analyze joint angle patterns
    joint_angles_data, speeds, categories = analyzer.analyze_joint_angle_patterns()
    
    # Find key frames
    key_frame_results = analyzer.find_key_frames(joint_angles_data, speeds, categories)
    if key_frame_results:
        peaks, high_speed_avg, low_speed_avg, avg_frame_diff = key_frame_results
        analyzer.plot_key_frames_analysis(peaks, high_speed_avg, low_speed_avg, avg_frame_diff)
    
    # Cluster analysis
    clusters, pca, scaler, kmeans = analyzer.cluster_serve_patterns(joint_angles_data)
    cluster_stats = analyzer.plot_cluster_analysis(clusters, speeds)
    
    # Biomechanical analysis
    efficiency_df = analyzer.analyze_biomechanical_efficiency(joint_angles_data, speeds)
    analyzer.plot_biomechanical_analysis(efficiency_df)
    
    # Generate recommendations
    analyzer.generate_recommendations(efficiency_df, cluster_stats, peaks if key_frame_results else [])
    
    print("\nAnalysis complete! Check the generated plots for detailed insights.")
    
    return analyzer, efficiency_df, cluster_stats

if __name__ == "__main__":
    analyzer, efficiency_df, cluster_stats = main() 