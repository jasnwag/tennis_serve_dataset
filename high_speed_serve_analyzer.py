import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class HighSpeedServeAnalyzer:
    def __init__(self, csv_path, high_speed_percentile=80):
        """
        Initialize analyzer focused on high-speed serves.
        
        Args:
            csv_path: Path to the CSV file
            high_speed_percentile: Percentile threshold for "high speed" serves
        """
        self.df = pd.read_csv(csv_path)
        self.high_speed_percentile = high_speed_percentile
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and filter data for analysis."""
        print("Preparing data for high-speed serve analysis...")
        
        # Clean data
        self.df_clean = self.df[
            self.df['joint_angles_100'].notna() & 
            (self.df['joint_angles_100'] != '') & 
            self.df['Speed_MPH'].notna()
        ].copy()
        
        # Define high-speed threshold
        self.high_speed_threshold = np.percentile(self.df_clean['Speed_MPH'], self.high_speed_percentile)
        
        # Create binary high-speed indicator
        self.df_clean['is_high_speed'] = self.df_clean['Speed_MPH'] >= self.high_speed_threshold
        
        print(f"Total serves: {len(self.df_clean)}")
        print(f"High-speed threshold: {self.high_speed_threshold:.1f} mph ({self.high_speed_percentile}th percentile)")
        print(f"High-speed serves: {self.df_clean['is_high_speed'].sum()}")
        print(f"Regular serves: {(~self.df_clean['is_high_speed']).sum()}")
        
    def extract_serve_features(self):
        """Extract meaningful features from pose sequences."""
        print("Extracting serve features...")
        
        features_list = []
        speeds = []
        is_high_speed = []
        
        for _, row in self.df_clean.iterrows():
            try:
                # Parse joint angles
                angles = ast.literal_eval(row['joint_angles_100'])
                angles_array = np.array(angles)
                
                if angles_array.ndim == 3:  # (frames, joints, coords)
                    frames, joints, coords = angles_array.shape
                    
                    # Reshape to (frames, features)
                    angles_flat = angles_array.reshape(frames, -1)
                    
                    # Extract temporal features
                    features = self.calculate_temporal_features(angles_flat)
                    
                    features_list.append(features)
                    speeds.append(row['Speed_MPH'])
                    is_high_speed.append(row['is_high_speed'])
                    
            except Exception as e:
                continue
        
        self.features_df = pd.DataFrame(features_list)
        self.features_df['speed'] = speeds
        self.features_df['is_high_speed'] = is_high_speed
        
        print(f"Features extracted for {len(self.features_df)} serves")
        print(f"Feature dimensions: {self.features_df.shape[1] - 2}")  # -2 for speed and is_high_speed
        
        return self.features_df
    
    def calculate_temporal_features(self, angles_sequence):
        """Calculate temporal features from joint angle sequence."""
        features = {}
        
        # Basic statistics across time
        features['mean_angle'] = np.mean(angles_sequence)
        features['std_angle'] = np.std(angles_sequence)
        features['max_angle'] = np.max(angles_sequence)
        features['min_angle'] = np.min(angles_sequence)
        features['range_angle'] = features['max_angle'] - features['min_angle']
        
        # Velocity features (first derivative)
        velocity = np.diff(angles_sequence, axis=0)
        features['mean_velocity'] = np.mean(np.abs(velocity))
        features['max_velocity'] = np.max(np.abs(velocity))
        features['velocity_std'] = np.std(velocity)
        
        # Acceleration features (second derivative)
        acceleration = np.diff(velocity, axis=0)
        features['mean_acceleration'] = np.mean(np.abs(acceleration))
        features['max_acceleration'] = np.max(np.abs(acceleration))
        features['acceleration_std'] = np.std(acceleration)
        
        # Jerk features (third derivative) - smoothness indicator
        jerk = np.diff(acceleration, axis=0)
        features['mean_jerk'] = np.mean(np.abs(jerk))
        features['jerk_std'] = np.std(jerk)
        features['smoothness'] = -np.log(features['mean_jerk'] + 1e-8)  # Higher = smoother
        
        # Peak detection features
        smoothed = savgol_filter(np.mean(angles_sequence, axis=1), 
                                window_length=min(11, len(angles_sequence)//2*2+1), 
                                polyorder=3)
        
        features['num_peaks'] = len(self.find_peaks_simple(smoothed))
        features['peak_prominence'] = np.std(smoothed)
        
        # Temporal progression features
        early_phase = angles_sequence[:len(angles_sequence)//3]
        mid_phase = angles_sequence[len(angles_sequence)//3:2*len(angles_sequence)//3]
        late_phase = angles_sequence[2*len(angles_sequence)//3:]
        
        features['early_phase_energy'] = np.mean(np.sum(early_phase**2, axis=1))
        features['mid_phase_energy'] = np.mean(np.sum(mid_phase**2, axis=1))
        features['late_phase_energy'] = np.mean(np.sum(late_phase**2, axis=1))
        
        # Energy distribution
        total_energy = features['early_phase_energy'] + features['mid_phase_energy'] + features['late_phase_energy']
        if total_energy > 0:
            features['early_energy_ratio'] = features['early_phase_energy'] / total_energy
            features['mid_energy_ratio'] = features['mid_phase_energy'] / total_energy
            features['late_energy_ratio'] = features['late_phase_energy'] / total_energy
        else:
            features['early_energy_ratio'] = features['mid_energy_ratio'] = features['late_energy_ratio'] = 0
        
        return features
    
    def find_peaks_simple(self, signal, threshold=None):
        """Simple peak detection."""
        if threshold is None:
            threshold = np.std(signal) * 0.5
        
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
                peaks.append(i)
        return peaks
    
    def identify_discriminative_features(self):
        """Identify features that best distinguish high-speed serves."""
        print("Identifying discriminative features...")
        
        # Prepare features for ML
        feature_cols = [col for col in self.features_df.columns if col not in ['speed', 'is_high_speed']]
        X = self.features_df[feature_cols]
        y = self.features_df['is_high_speed']
        
        # Train Random Forest to identify important features
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y.astype(int))
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 discriminative features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def analyze_high_speed_characteristics(self):
        """Analyze what makes high-speed serves different."""
        print("Analyzing high-speed serve characteristics...")
        
        high_speed_serves = self.features_df[self.features_df['is_high_speed']]
        regular_serves = self.features_df[~self.features_df['is_high_speed']]
        
        # Statistical comparison
        feature_cols = [col for col in self.features_df.columns if col not in ['speed', 'is_high_speed']]
        
        comparison_results = []
        for feature in feature_cols:
            high_mean = high_speed_serves[feature].mean()
            regular_mean = regular_serves[feature].mean()
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((high_speed_serves[feature].std()**2 + regular_serves[feature].std()**2) / 2))
            cohens_d = (high_mean - regular_mean) / pooled_std if pooled_std > 0 else 0
            
            comparison_results.append({
                'feature': feature,
                'high_speed_mean': high_mean,
                'regular_mean': regular_mean,
                'difference': high_mean - regular_mean,
                'percent_difference': ((high_mean - regular_mean) / regular_mean * 100) if regular_mean != 0 else 0,
                'cohens_d': cohens_d
            })
        
        comparison_df = pd.DataFrame(comparison_results).sort_values('cohens_d', key=abs, ascending=False)
        
        print("\nFeatures with largest effect sizes (Cohen's d):")
        print(comparison_df.head(10)[['feature', 'percent_difference', 'cohens_d']])
        
        return comparison_df
    
    def plot_high_speed_analysis(self, feature_importance, comparison_df):
        """Create comprehensive plots for high-speed serve analysis."""
        plt.figure(figsize=(20, 15))
        
        # 1. Speed distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.df_clean['Speed_MPH'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(self.high_speed_threshold, color='red', linestyle='--', 
                   label=f'High-speed threshold: {self.high_speed_threshold:.1f} mph')
        plt.title('Serve Speed Distribution')
        plt.xlabel('Speed (mph)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Feature importance
        plt.subplot(3, 4, 2)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 10 Discriminative Features')
        plt.xlabel('Feature Importance')
        plt.grid(True, alpha=0.3)
        
        # 3. Effect sizes
        plt.subplot(3, 4, 3)
        top_effects = comparison_df.head(10)
        colors = ['red' if x > 0 else 'blue' for x in top_effects['cohens_d']]
        plt.barh(range(len(top_effects)), top_effects['cohens_d'], color=colors)
        plt.yticks(range(len(top_effects)), top_effects['feature'])
        plt.title('Effect Sizes (Cohen\'s d)')
        plt.xlabel('Effect Size')
        plt.axvline(0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # 4. Energy distribution comparison
        plt.subplot(3, 4, 4)
        energy_features = ['early_energy_ratio', 'mid_energy_ratio', 'late_energy_ratio']
        high_speed_energy = [self.features_df[self.features_df['is_high_speed']][f].mean() for f in energy_features]
        regular_energy = [self.features_df[~self.features_df['is_high_speed']][f].mean() for f in energy_features]
        
        x = np.arange(len(energy_features))
        width = 0.35
        plt.bar(x - width/2, high_speed_energy, width, label='High Speed', alpha=0.8)
        plt.bar(x + width/2, regular_energy, width, label='Regular', alpha=0.8)
        plt.xlabel('Serve Phase')
        plt.ylabel('Energy Ratio')
        plt.title('Energy Distribution Across Serve Phases')
        plt.xticks(x, ['Early', 'Mid', 'Late'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5-8. Box plots for key features
        key_features = feature_importance.head(4)['feature'].tolist()
        for i, feature in enumerate(key_features):
            plt.subplot(3, 4, 5 + i)
            data = [
                self.features_df[~self.features_df['is_high_speed']][feature],
                self.features_df[self.features_df['is_high_speed']][feature]
            ]
            plt.boxplot(data, labels=['Regular', 'High Speed'])
            plt.title(f'{feature}')
            plt.grid(True, alpha=0.3)
        
        # 9. Smoothness vs Speed scatter
        plt.subplot(3, 4, 9)
        plt.scatter(self.features_df['smoothness'], self.features_df['speed'], 
                   c=self.features_df['is_high_speed'], cmap='coolwarm', alpha=0.6)
        plt.xlabel('Smoothness')
        plt.ylabel('Speed (mph)')
        plt.title('Smoothness vs Speed')
        plt.colorbar(label='High Speed')
        plt.grid(True, alpha=0.3)
        
        # 10. Velocity vs Speed scatter
        plt.subplot(3, 4, 10)
        plt.scatter(self.features_df['max_velocity'], self.features_df['speed'], 
                   c=self.features_df['is_high_speed'], cmap='coolwarm', alpha=0.6)
        plt.xlabel('Max Velocity')
        plt.ylabel('Speed (mph)')
        plt.title('Max Velocity vs Speed')
        plt.colorbar(label='High Speed')
        plt.grid(True, alpha=0.3)
        
        # 11. Summary statistics
        plt.subplot(3, 4, 11)
        summary_text = f"""
        High-Speed Serve Analysis Summary:
        
        Total serves analyzed: {len(self.features_df)}
        High-speed serves: {self.features_df['is_high_speed'].sum()}
        Regular serves: {(~self.features_df['is_high_speed']).sum()}
        
        Speed Statistics:
        High-speed mean: {self.features_df[self.features_df['is_high_speed']]['speed'].mean():.1f} mph
        Regular mean: {self.features_df[~self.features_df['is_high_speed']]['speed'].mean():.1f} mph
        
        Top Discriminative Features:
        1. {feature_importance.iloc[0]['feature']}
        2. {feature_importance.iloc[1]['feature']}
        3. {feature_importance.iloc[2]['feature']}
        """
        plt.text(0.1, 0.1, summary_text, fontsize=9, verticalalignment='bottom')
        plt.axis('off')
        
        # 12. Recommendations
        plt.subplot(3, 4, 12)
        recommendations = self.generate_coaching_insights(comparison_df, feature_importance)
        plt.text(0.1, 0.1, recommendations, fontsize=9, verticalalignment='bottom')
        plt.axis('off')
        plt.title('Coaching Insights')
        
        plt.tight_layout()
        plt.savefig('high_speed_serve_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_coaching_insights(self, comparison_df, feature_importance):
        """Generate actionable coaching insights."""
        top_features = feature_importance.head(5)
        top_differences = comparison_df.head(5)
        
        insights = "Key Insights for High-Speed Serves:\n\n"
        
        # Analyze top discriminative features
        for _, row in top_differences.iterrows():
            feature = row['feature']
            diff = row['percent_difference']
            
            if 'velocity' in feature.lower():
                if diff > 0:
                    insights += f"• Increase {feature.replace('_', ' ')}\n"
                else:
                    insights += f"• Control {feature.replace('_', ' ')}\n"
            elif 'smoothness' in feature.lower():
                if diff > 0:
                    insights += f"• Focus on smoother motion\n"
                else:
                    insights += f"• Allow for more dynamic movement\n"
            elif 'energy' in feature.lower():
                if diff > 0:
                    insights += f"• Emphasize {feature.replace('_', ' ')}\n"
                else:
                    insights += f"• Reduce emphasis on {feature.replace('_', ' ')}\n"
        
        return insights
    
    def find_optimal_serve_examples(self, top_n=5):
        """Find examples of optimal high-speed serves."""
        high_speed_serves = self.features_df[self.features_df['is_high_speed']].copy()
        
        # Score serves based on multiple criteria
        high_speed_serves['composite_score'] = (
            high_speed_serves['speed'] * 0.4 +  # Speed weight
            high_speed_serves['smoothness'] * 0.3 +  # Smoothness weight
            (1 / (high_speed_serves['jerk_std'] + 1e-8)) * 0.3  # Consistency weight
        )
        
        optimal_serves = high_speed_serves.nlargest(top_n, 'composite_score')
        
        print(f"\nTop {top_n} optimal high-speed serves:")
        print("="*50)
        for i, (idx, serve) in enumerate(optimal_serves.iterrows(), 1):
            print(f"{i}. Speed: {serve['speed']:.1f} mph")
            print(f"   Smoothness: {serve['smoothness']:.3f}")
            print(f"   Composite Score: {serve['composite_score']:.3f}")
            print(f"   Index: {idx}")
            print()
        
        return optimal_serves

def main():
    """Main analysis function."""
    csv_path = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final.csv'
    
    # Initialize analyzer
    analyzer = HighSpeedServeAnalyzer(csv_path, high_speed_percentile=80)
    
    # Extract features
    features_df = analyzer.extract_serve_features()
    
    # Identify discriminative features
    feature_importance = analyzer.identify_discriminative_features()
    
    # Analyze characteristics
    comparison_df = analyzer.analyze_high_speed_characteristics()
    
    # Create plots
    analyzer.plot_high_speed_analysis(feature_importance, comparison_df)
    
    # Find optimal examples
    optimal_serves = analyzer.find_optimal_serve_examples()
    
    print("\nHigh-speed serve analysis complete!")
    print("Key takeaways:")
    print("1. Check the generated plot for visual insights")
    print("2. Focus on the top discriminative features for training")
    print("3. Use optimal serve examples as technical models")
    
    return analyzer, features_df, feature_importance, comparison_df

if __name__ == "__main__":
    analyzer, features_df, feature_importance, comparison_df = main() 