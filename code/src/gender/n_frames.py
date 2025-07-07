# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/final_with_angles_velocity_acceleration.csv')

# %%
df.columns

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional appearance
plt.style.use('seaborn-v0_8-whitegrid')

# Create boxplot of n_frames by server_gender with different colors
plt.figure(figsize=(10, 7))

# Define colors for men and women
colors = ['#3498db', '#e74c3c']  # Blue for men, red for women
ax = sns.boxplot(data=df, x='server_gender', y='n_frames', 
                 palette=colors, linewidth=1.5, fliersize=3)

# Customize the plot for professional appearance
plt.title('Distribution of Frame Count by Server Gender', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Server Gender', fontsize=14, fontweight='medium')
plt.ylabel('Number of Frames', fontsize=14, fontweight='medium')

# Improve tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add subtle grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Tight layout for better spacing
plt.tight_layout()
plt.show()


# %%
# Create correlation analysis between n_frames and Speed_MPH by gender and serve number
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Filter out any rows with missing Speed_MPH data
df_clean = df.dropna(subset=['Speed_MPH', 'n_frames', 'ServeNumber'])

# Define colors for consistency
male_color = '#3498db'
female_color = '#e74c3c'

# Create separate datasets for each combination
male_serve1 = df_clean[(df_clean['server_gender'] == 'M') & (df_clean['ServeNumber'] == 1)]
male_serve2 = df_clean[(df_clean['server_gender'] == 'M') & (df_clean['ServeNumber'] == 2)]
female_serve1 = df_clean[(df_clean['server_gender'] == 'F') & (df_clean['ServeNumber'] == 1)]
female_serve2 = df_clean[(df_clean['server_gender'] == 'F') & (df_clean['ServeNumber'] == 2)]

# Row 1: Male servers
# Plot 1: Male First Serves
if len(male_serve1) > 0:
    axes[0,0].scatter(male_serve1['Speed_MPH'], male_serve1['n_frames'], 
                     alpha=0.6, color=male_color, s=20)
    
    # Add trend line
    z = np.polyfit(male_serve1['Speed_MPH'], male_serve1['n_frames'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(male_serve1['Speed_MPH'], p(male_serve1['Speed_MPH']), 
                  color='darkblue', linewidth=2, linestyle='--')
    
    # Calculate correlation
    male_serve1_corr = np.corrcoef(male_serve1['Speed_MPH'], male_serve1['n_frames'])[0,1]
    
    axes[0,0].set_title(f'Male First Serves\nCorrelation: r = {male_serve1_corr:.3f}\nn = {len(male_serve1):,}', 
                       fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Serve Speed (MPH)', fontsize=11)
    axes[0,0].set_ylabel('Number of Frames', fontsize=11)
    axes[0,0].grid(True, alpha=0.3)

# Plot 2: Male Second Serves
if len(male_serve2) > 0:
    axes[0,1].scatter(male_serve2['Speed_MPH'], male_serve2['n_frames'], 
                     alpha=0.6, color=male_color, s=20)
    
    # Add trend line
    z = np.polyfit(male_serve2['Speed_MPH'], male_serve2['n_frames'], 1)
    p = np.poly1d(z)
    axes[0,1].plot(male_serve2['Speed_MPH'], p(male_serve2['Speed_MPH']), 
                  color='darkblue', linewidth=2, linestyle='--')
    
    # Calculate correlation
    male_serve2_corr = np.corrcoef(male_serve2['Speed_MPH'], male_serve2['n_frames'])[0,1]
    
    axes[0,1].set_title(f'Male Second Serves\nCorrelation: r = {male_serve2_corr:.3f}\nn = {len(male_serve2):,}', 
                       fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Serve Speed (MPH)', fontsize=11)
    axes[0,1].set_ylabel('Number of Frames', fontsize=11)
    axes[0,1].grid(True, alpha=0.3)

# Plot 3: Male Combined
axes[0,2].scatter(male_serve1['Speed_MPH'], male_serve1['n_frames'], 
                 alpha=0.6, color='darkblue', s=20, label='First Serve')
axes[0,2].scatter(male_serve2['Speed_MPH'], male_serve2['n_frames'], 
                 alpha=0.6, color='lightblue', s=20, label='Second Serve')

# Add trend lines
if len(male_serve1) > 0:
    z1 = np.polyfit(male_serve1['Speed_MPH'], male_serve1['n_frames'], 1)
    p1 = np.poly1d(z1)
    axes[0,2].plot(male_serve1['Speed_MPH'], p1(male_serve1['Speed_MPH']), 
                  color='darkblue', linewidth=2, linestyle='--', alpha=0.8)

if len(male_serve2) > 0:
    z2 = np.polyfit(male_serve2['Speed_MPH'], male_serve2['n_frames'], 1)
    p2 = np.poly1d(z2)
    axes[0,2].plot(male_serve2['Speed_MPH'], p2(male_serve2['Speed_MPH']), 
                  color='lightblue', linewidth=2, linestyle='--', alpha=0.8)

axes[0,2].set_title('Male Serves Combined', fontsize=12, fontweight='bold')
axes[0,2].set_xlabel('Serve Speed (MPH)', fontsize=11)
axes[0,2].set_ylabel('Number of Frames', fontsize=11)
axes[0,2].legend(fontsize=10)
axes[0,2].grid(True, alpha=0.3)

# Row 2: Female servers
# Plot 4: Female First Serves
if len(female_serve1) > 0:
    axes[1,0].scatter(female_serve1['Speed_MPH'], female_serve1['n_frames'], 
                     alpha=0.6, color=female_color, s=20)
    
    # Add trend line
    z = np.polyfit(female_serve1['Speed_MPH'], female_serve1['n_frames'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(female_serve1['Speed_MPH'], p(female_serve1['Speed_MPH']), 
                  color='darkred', linewidth=2, linestyle='--')
    
    # Calculate correlation
    female_serve1_corr = np.corrcoef(female_serve1['Speed_MPH'], female_serve1['n_frames'])[0,1]
    
    axes[1,0].set_title(f'Female First Serves\nCorrelation: r = {female_serve1_corr:.3f}\nn = {len(female_serve1):,}', 
                       fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Serve Speed (MPH)', fontsize=11)
    axes[1,0].set_ylabel('Number of Frames', fontsize=11)
    axes[1,0].grid(True, alpha=0.3)

# Plot 5: Female Second Serves
if len(female_serve2) > 0:
    axes[1,1].scatter(female_serve2['Speed_MPH'], female_serve2['n_frames'], 
                     alpha=0.6, color=female_color, s=20)
    
    # Add trend line
    z = np.polyfit(female_serve2['Speed_MPH'], female_serve2['n_frames'], 1)
    p = np.poly1d(z)
    axes[1,1].plot(female_serve2['Speed_MPH'], p(female_serve2['Speed_MPH']), 
                  color='darkred', linewidth=2, linestyle='--')
    
    # Calculate correlation
    female_serve2_corr = np.corrcoef(female_serve2['Speed_MPH'], female_serve2['n_frames'])[0,1]
    
    axes[1,1].set_title(f'Female Second Serves\nCorrelation: r = {female_serve2_corr:.3f}\nn = {len(female_serve2):,}', 
                       fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Serve Speed (MPH)', fontsize=11)
    axes[1,1].set_ylabel('Number of Frames', fontsize=11)
    axes[1,1].grid(True, alpha=0.3)

# Plot 6: Female Combined
axes[1,2].scatter(female_serve1['Speed_MPH'], female_serve1['n_frames'], 
                 alpha=0.6, color='darkred', s=20, label='First Serve')
axes[1,2].scatter(female_serve2['Speed_MPH'], female_serve2['n_frames'], 
                 alpha=0.6, color='lightcoral', s=20, label='Second Serve')

# Add trend lines
if len(female_serve1) > 0:
    z1 = np.polyfit(female_serve1['Speed_MPH'], female_serve1['n_frames'], 1)
    p1 = np.poly1d(z1)
    axes[1,2].plot(female_serve1['Speed_MPH'], p1(female_serve1['Speed_MPH']), 
                  color='darkred', linewidth=2, linestyle='--', alpha=0.8)

if len(female_serve2) > 0:
    z2 = np.polyfit(female_serve2['Speed_MPH'], female_serve2['n_frames'], 1)
    p2 = np.poly1d(z2)
    axes[1,2].plot(female_serve2['Speed_MPH'], p2(female_serve2['Speed_MPH']), 
                  color='lightcoral', linewidth=2, linestyle='--', alpha=0.8)

axes[1,2].set_title('Female Serves Combined', fontsize=12, fontweight='bold')
axes[1,2].set_xlabel('Serve Speed (MPH)', fontsize=11)
axes[1,2].set_ylabel('Number of Frames', fontsize=11)
axes[1,2].legend(fontsize=10)
axes[1,2].grid(True, alpha=0.3)

plt.suptitle('Correlation Analysis: Frame Count vs Serve Speed by Gender and Serve Number', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# Print comprehensive summary statistics
from scipy.stats import pearsonr

print("Detailed Correlation Analysis Summary:")
print("=" * 60)

datasets = [
    ("Male First Serves", male_serve1),
    ("Male Second Serves", male_serve2),
    ("Female First Serves", female_serve1),
    ("Female Second Serves", female_serve2)
]

for name, data in datasets:
    if len(data) > 0:
        corr_coef = np.corrcoef(data['Speed_MPH'], data['n_frames'])[0,1]
        corr_coef_scipy, p_value = pearsonr(data['Speed_MPH'], data['n_frames'])
        
        print(f"\n{name}:")
        print(f"  Sample size: {len(data):,}")
        print(f"  Correlation coefficient: {corr_coef:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Mean speed: {data['Speed_MPH'].mean():.1f} MPH")
        print(f"  Mean frames: {data['n_frames'].mean():.1f}")
        print(f"  Speed range: {data['Speed_MPH'].min():.1f} - {data['Speed_MPH'].max():.1f} MPH")
        print(f"  Frame range: {data['n_frames'].min():.0f} - {data['n_frames'].max():.0f}")

# Summary comparison table
print(f"\n{'='*60}")
print("SUMMARY COMPARISON TABLE:")
print(f"{'='*60}")
print(f"{'Category':<20} {'n':<8} {'r':<8} {'p-val':<8} {'Avg Speed':<12} {'Avg Frames':<12}")
print(f"{'-'*60}")

for name, data in datasets:
    if len(data) > 0:
        corr_coef = np.corrcoef(data['Speed_MPH'], data['n_frames'])[0,1]
        _, p_value = pearsonr(data['Speed_MPH'], data['n_frames'])
        print(f"{name:<20} {len(data):<8,} {corr_coef:<8.3f} {p_value:<8.3f} {data['Speed_MPH'].mean():<12.1f} {data['n_frames'].mean():<12.1f}")



# %%
# Overall correlation analysis between serve speed and number of frames
print("\n" + "=" * 50)
print("OVERALL CORRELATION ANALYSIS")
print("=" * 50)

# Filter data for overall analysis
overall_data = df.dropna(subset=['Speed_MPH', 'n_frames'])
overall_data = overall_data[overall_data['Speed_MPH'] > 0]

if len(overall_data) > 0:
    # Calculate overall correlation
    overall_corr_coef, overall_p_value = pearsonr(overall_data['Speed_MPH'], overall_data['n_frames'])
    
    print(f"Overall Statistics:")
    print(f"  Total sample size: {len(overall_data):,}")
    print(f"  Correlation coefficient: {overall_corr_coef:.4f}")
    print(f"  P-value: {overall_p_value:.4f}")
    print(f"  Statistical significance: {'Yes' if overall_p_value < 0.05 else 'No'} (α = 0.05)")
    print(f"  Mean speed: {overall_data['Speed_MPH'].mean():.1f} MPH")
    print(f"  Mean frames: {overall_data['n_frames'].mean():.1f}")
    print(f"  Speed range: {overall_data['Speed_MPH'].min():.1f} - {overall_data['Speed_MPH'].max():.1f} MPH")
    print(f"  Frame range: {overall_data['n_frames'].min():.0f} - {overall_data['n_frames'].max():.0f} frames")

    # Create overall scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(overall_data['Speed_MPH'], overall_data['n_frames'], 
                alpha=0.6, s=30, color='blue')
    
    # Add trend line
    z = np.polyfit(overall_data['Speed_MPH'], overall_data['n_frames'], 1)
    p = np.poly1d(z)
    plt.plot(overall_data['Speed_MPH'], p(overall_data['Speed_MPH']), 
             color='darkblue', linewidth=2, linestyle='--')
    
    plt.title(f'Overall Correlation: Frame Count vs Serve Speed\n(r = {overall_corr_coef:.4f}, p = {overall_p_value:.4f})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Serve Speed (MPH)', fontsize=12)
    plt.ylabel('Number of Frames', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add correlation text
    plt.text(0.05, 0.95, f'Correlation: {overall_corr_coef:.4f}\nP-value: {overall_p_value:.4f}\nSample size: {len(overall_data):,}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Interpretation
    print(f"\nInterpretation:")
    if abs(overall_corr_coef) < 0.1:
        strength = "negligible"
    elif abs(overall_corr_coef) < 0.3:
        strength = "weak"
    elif abs(overall_corr_coef) < 0.5:
        strength = "moderate"
    elif abs(overall_corr_coef) < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if overall_corr_coef > 0 else "negative"
    print(f"  There is a {strength} {direction} correlation between serve speed and frame count.")
    
    if overall_p_value < 0.05:
        print(f"  This correlation is statistically significant at the 0.05 level.")
    else:
        print(f"  This correlation is not statistically significant at the 0.05 level.")

else:
    print("No valid data available for overall correlation analysis.")



# %%
# Boxplot comparing n_frames between first and second serves
print("Analyzing frame count differences between first and second serves...")

# Filter data to only include first and second serves
serve_data = df[df['ServeNumber'].isin([1, 2])].copy()

if len(serve_data) > 0:
    # Create the boxplot
    plt.figure(figsize=(10, 10))
    
    # Create boxplot
    ax = sns.boxplot(data=serve_data, x='ServeNumber', y='n_frames', 
                     palette=['skyblue', 'lightcoral'])
    
    # Customize the plot
    plt.title('Frame Count Distribution: First vs Second Serves', fontsize=14, fontweight='bold')
    plt.xlabel('Serve Number', fontsize=12)
    plt.ylabel('Number of Frames', fontsize=12)
    
    # Update x-axis labels
    ax.set_xticklabels(['First Serve', 'Second Serve'])
    
    # Add statistics
    first_serve_data = serve_data[serve_data['ServeNumber'] == 1]['n_frames'].dropna()
    second_serve_data = serve_data[serve_data['ServeNumber'] == 2]['n_frames'].dropna()
    
    # Calculate summary statistics
    first_median = first_serve_data.median()
    second_median = second_serve_data.median()
    first_mean = first_serve_data.mean()
    second_mean = second_serve_data.mean()
    
    # Add sample sizes to plot
    plt.text(0, plt.ylim()[1] * 0.95, f'n = {len(first_serve_data):,}', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(1, plt.ylim()[1] * 0.95, f'n = {len(second_serve_data):,}', 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"First Serves:")
    print(f"  Count: {len(first_serve_data):,}")
    print(f"  Mean: {first_mean:.2f} frames")
    print(f"  Median: {first_median:.2f} frames")
    print(f"  Std Dev: {first_serve_data.std():.2f} frames")
    
    print(f"\nSecond Serves:")
    print(f"  Count: {len(second_serve_data):,}")
    print(f"  Mean: {second_mean:.2f} frames")
    print(f"  Median: {second_median:.2f} frames")
    print(f"  Std Dev: {second_serve_data.std():.2f} frames")
    
    # Statistical test
    from scipy import stats
    if len(first_serve_data) > 0 and len(second_serve_data) > 0:
        t_stat, p_value = stats.ttest_ind(first_serve_data, second_serve_data)
        print(f"\nStatistical Test (Independent t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  There is a statistically significant difference between first and second serves (p < 0.05)")
        else:
            print(f"  No statistically significant difference between first and second serves (p >= 0.05)")
        
        difference = first_mean - second_mean
        print(f"  Mean difference: {difference:.2f} frames (First - Second)")
else:
    print("No data available for serve number analysis.")


# %%
import json
import ast
from scipy.interpolate import interp1d
import numpy as np

def parse_joint_data(data_str):
    """Parse the joint velocity data string."""
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

def normalize_to_100_timesteps(sequence):
    """Normalize a sequence to 100 timesteps using interpolation."""
    if sequence is None or len(sequence) < 2:
        return None
    
    original_length = len(sequence)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, 100)
    
    # Use linear interpolation
    f = interp1d(original_indices, sequence, kind='linear')
    normalized_sequence = f(target_indices)
    
    return normalized_sequence.tolist()

# Joint names to analyze
joint_names = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]

# Colors for male and female
male_color = '#1f77b4'  # Blue
female_color = '#ff7f0e'  # Orange

# Process each joint
for joint in joint_names:
    velocity_col = f'{joint}_velocity'
    
    if velocity_col not in df.columns:
        print(f"Column {velocity_col} not found, skipping...")
        continue
    
    print(f"\nProcessing {joint} velocity...")
    
    # Collect normalized sequences by gender
    male_sequences = []
    female_sequences = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['server_gender']):
            continue
            
        velocity_data = parse_joint_data(row[velocity_col])
        if velocity_data is None:
            continue
            
        normalized_seq = normalize_to_100_timesteps(velocity_data)
        if normalized_seq is None:
            continue
            
        if row['server_gender'] == 'M':
            male_sequences.append(normalized_seq)
        elif row['server_gender'] == 'F':
            female_sequences.append(normalized_seq)
    
    print(f"  Male sequences: {len(male_sequences)}")
    print(f"  Female sequences: {len(female_sequences)}")
    
    if len(male_sequences) == 0 and len(female_sequences) == 0:
        print(f"  No valid sequences found for {joint}, skipping...")
        continue
    
    # Convert to numpy arrays for statistics
    if len(male_sequences) > 0:
        male_array = np.array(male_sequences)
        male_mean = np.mean(male_array, axis=0)
        male_std = np.std(male_array, axis=0)
    else:
        male_mean = male_std = None
    
    if len(female_sequences) > 0:
        female_array = np.array(female_sequences)
        female_mean = np.mean(female_array, axis=0)
        female_std = np.std(female_array, axis=0)
    else:
        female_mean = female_std = None
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    timesteps = np.arange(100)
    
    # Plot male data
    if male_mean is not None:
        plt.plot(timesteps, male_mean, color=male_color, linewidth=2, label=f'Male (n={len(male_sequences)})')
        plt.fill_between(timesteps, 
                        male_mean - male_std, 
                        male_mean + male_std,
                        color=male_color, alpha=0.3)
    
    # Plot female data
    if female_mean is not None:
        plt.plot(timesteps, female_mean, color=female_color, linewidth=2, label=f'Female (n={len(female_sequences)})')
        plt.fill_between(timesteps, 
                        female_mean - female_std, 
                        female_mean + female_std,
                        color=female_color, alpha=0.3)
    
    plt.title(f'{joint.replace("_", " ").title()} Velocity: Male vs Female\n(Normalized to 100 timesteps)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Timestep', fontsize=12)
    plt.ylabel('Velocity (radians/frame)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
import json
import ast
from scipy.interpolate import interp1d
import numpy as np

def parse_joint_data(data_str):
    """Parse the joint velocity or acceleration data string."""
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

def normalize_to_100_timesteps(sequence):
    """Normalize a sequence to 100 timesteps using interpolation."""
    if sequence is None or len(sequence) < 2:
        return None
    
    original_length = len(sequence)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, 100)
    
    # Use linear interpolation
    f = interp1d(original_indices, sequence, kind='linear')
    normalized_sequence = f(target_indices)
    
    return normalized_sequence.tolist()

# Joint names to analyze
joint_names = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]

# Colors for male and female
male_color = '#1f77b4'  # Blue
female_color = '#ff7f0e'  # Orange

# Create subplots: 4 rows x 4 columns (velocity on left 2 columns, acceleration on right 2 columns)
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle('Joint Velocity and Acceleration: Male vs Female\n(Normalized to 100 timesteps)', 
             fontsize=18, fontweight='bold', y=0.98)

# Process each joint
for joint_idx, joint in enumerate(joint_names):
    # Velocity plots (left 2 columns)
    vel_row = joint_idx // 2
    vel_col = joint_idx % 2
    vel_ax = axes[vel_row, vel_col]
    
    # Acceleration plots (right 2 columns)
    acc_row = joint_idx // 2
    acc_col = (joint_idx % 2) + 2
    acc_ax = axes[acc_row, acc_col]
    
    for metric, ax in [('velocity', vel_ax), ('acceleration', acc_ax)]:
        metric_col = f'{joint}_{metric}'
        
        if metric_col not in df.columns:
            print(f"Column {metric_col} not found, skipping...")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{joint.replace("_", " ").title()}\n{metric.title()}', fontsize=11, fontweight='bold')
            continue
        
        print(f"\nProcessing {joint} {metric}...")
        
        # Collect normalized sequences by gender
        male_sequences = []
        female_sequences = []
        
        for idx, row_data in df.iterrows():
            if pd.isna(row_data['server_gender']):
                continue
                
            metric_data = parse_joint_data(row_data[metric_col])
            if metric_data is None:
                continue
                
            normalized_seq = normalize_to_100_timesteps(metric_data)
            if normalized_seq is None:
                continue
                
            if row_data['server_gender'] == 'M':
                male_sequences.append(normalized_seq)
            elif row_data['server_gender'] == 'F':
                female_sequences.append(normalized_seq)
        
        print(f"  Male sequences: {len(male_sequences)}")
        print(f"  Female sequences: {len(female_sequences)}")
        
        if len(male_sequences) == 0 and len(female_sequences) == 0:
            print(f"  No valid sequences found for {joint} {metric}, skipping...")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{joint.replace("_", " ").title()}\n{metric.title()}', fontsize=11, fontweight='bold')
            continue
        
        # Convert to numpy arrays for statistics
        if len(male_sequences) > 0:
            male_array = np.array(male_sequences)
            male_mean = np.mean(male_array, axis=0)
            male_std = np.std(male_array, axis=0)
        else:
            male_mean = male_std = None
        
        if len(female_sequences) > 0:
            female_array = np.array(female_sequences)
            female_mean = np.mean(female_array, axis=0)
            female_std = np.std(female_array, axis=0)
        else:
            female_mean = female_std = None
        
        timesteps = np.arange(100)
        
        # Plot male data
        if male_mean is not None:
            ax.plot(timesteps, male_mean, color=male_color, linewidth=2, label=f'Male (n={len(male_sequences)})')
            ax.fill_between(timesteps, 
                           male_mean - male_std, 
                           male_mean + male_std,
                           color=male_color, alpha=0.3)
        
        # Plot female data
        if female_mean is not None:
            ax.plot(timesteps, female_mean, color=female_color, linewidth=2, label=f'Female (n={len(female_sequences)})')
            ax.fill_between(timesteps, 
                           female_mean - female_std, 
                           female_mean + female_std,
                           color=female_color, alpha=0.3)
        
        # Set appropriate ylabel based on metric
        ylabel = 'Velocity (rad/frame)' if metric == 'velocity' else 'Acceleration (rad/frame²)'
        
        ax.set_title(f'{joint.replace("_", " ").title()}\n{metric.title()}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Normalized Timestep', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# %%
# Analyze correlation between peak right shoulder velocity and serve speed
def parse_joint_data(data_str):
    """Parse joint velocity data string."""
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

# Filter data with both right shoulder velocity and speed information
analysis_data = []

for idx, row in df.iterrows():
    if pd.notna(row['right_shoulder_velocity']) and pd.notna(row['Speed_MPH']):
        velocity_data = parse_joint_data(row['right_shoulder_velocity'])
        if velocity_data is not None and len(velocity_data) > 0:
            # Get the maximum absolute velocity value in the sequence
            max_velocity = max(abs(v) for v in velocity_data)
            analysis_data.append({
                'max_right_shoulder_velocity': max_velocity,
                'speed_mph': row['Speed_MPH'],
                'server_gender': row['server_gender']
            })

analysis_df = pd.DataFrame(analysis_data)

if len(analysis_df) > 0:
    print(f"Analysis includes {len(analysis_df)} serves with complete data")
    print(f"Gender breakdown: {analysis_df['server_gender'].value_counts()}")
    
    # Calculate correlation
    correlation = analysis_df['max_right_shoulder_velocity'].corr(analysis_df['speed_mph'])
    print(f"\nOverall correlation between peak right shoulder velocity and serve speed: {correlation:.4f}")
    
    # Calculate correlation by gender
    for gender in ['Male', 'Female']:
        gender_data = analysis_df[analysis_df['server_gender'] == gender]
        if len(gender_data) > 10:  # Only if we have enough data
            gender_corr = gender_data['max_right_shoulder_velocity'].corr(gender_data['speed_mph'])
            print(f"{gender} correlation: {gender_corr:.4f} (n={len(gender_data)})")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall scatter plot
    ax1 = axes[0]
    ax1.scatter(analysis_df['max_right_shoulder_velocity'], analysis_df['speed_mph'], 
                alpha=0.6, color='steelblue')
    
    # Add trend line
    z = np.polyfit(analysis_df['max_right_shoulder_velocity'], analysis_df['speed_mph'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(analysis_df['max_right_shoulder_velocity'].min(), 
                         analysis_df['max_right_shoulder_velocity'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Peak Right Shoulder Velocity (rad/frame)', fontsize=12)
    ax1.set_ylabel('Serve Speed (MPH)', fontsize=12)
    ax1.set_title(f'Peak Right Shoulder Velocity vs Serve Speed\nCorrelation: {correlation:.4f}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Gender-separated scatter plot
    ax2 = axes[1]
    colors = {'M': '#2E86AB', 'F': '#A23B72'}
    
    for gender in ['M', 'F']:
        gender_data = analysis_df[analysis_df['server_gender'] == gender]
        if len(gender_data) > 0:
            ax2.scatter(gender_data['max_right_shoulder_velocity'], gender_data['speed_mph'], 
                       alpha=0.6, color=colors[gender], label=f'{gender.title()} (n={len(gender_data)})')
            
            # Add trend line for each gender if enough data
            if len(gender_data) > 10:
                z = np.polyfit(gender_data['max_right_shoulder_velocity'], gender_data['speed_mph'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(gender_data['max_right_shoulder_velocity'].min(), 
                                     gender_data['max_right_shoulder_velocity'].max(), 100)
                ax2.plot(x_trend, p(x_trend), color=colors[gender], linestyle='--', 
                        alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Peak Right Shoulder Velocity (rad/frame)', fontsize=12)
    ax2.set_ylabel('Serve Speed (MPH)', fontsize=12)
    ax2.set_title('Peak Right Shoulder Velocity vs Serve Speed by Gender', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(analysis_df.groupby('server_gender').agg({
        'max_right_shoulder_velocity': ['mean', 'std', 'min', 'max'],
        'speed_mph': ['mean', 'std', 'min', 'max']
    }).round(4))
    
else:
    print("No valid data found for analysis")


# %%
import ast

def calculate_hip_shoulder_separation(keypoints_3d):
    """
    Calculate the rotational angle between the pelvis axis and shoulder axis.
    
    Args:
        keypoints_3d: numpy array of shape (n_frames, 17, 3) with 3D keypoints
        
    Returns:
        numpy array of hip-shoulder separation angles in radians for each frame
    """
    n_frames = keypoints_3d.shape[0]
    separation_angles = np.zeros(n_frames)
    
    for frame in range(n_frames):
        # Extract hip joints (indices 1 and 4 for right and left hip)
        right_hip = keypoints_3d[frame, 1, :]  # Right-Hip
        left_hip = keypoints_3d[frame, 4, :]   # Left-Hip
        
        # Extract shoulder joints (indices 11 and 14 for left and right shoulder)
        left_shoulder = keypoints_3d[frame, 11, :]   # Left-Shoulder
        right_shoulder = keypoints_3d[frame, 14, :]  # Right-Shoulder
        
        # Calculate pelvis axis (vector from left hip to right hip)
        pelvis_axis = right_hip - left_hip
        
        # Calculate shoulder axis (vector from left shoulder to right shoulder)
        shoulder_axis = right_shoulder - left_shoulder
        
        # Project vectors onto horizontal plane (remove Y component for frontal plane analysis)
        pelvis_axis_2d = np.array([pelvis_axis[0], pelvis_axis[1]])
        shoulder_axis_2d = np.array([shoulder_axis[0], shoulder_axis[1]])
        
        # Calculate angle between the two axes
        dot_product = np.dot(pelvis_axis_2d, shoulder_axis_2d)
        norms = np.linalg.norm(pelvis_axis_2d) * np.linalg.norm(shoulder_axis_2d)
        
        if norms > 0:
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle = np.arccos(cos_angle)
        else:
            angle = 0.0
        
        separation_angles[frame] = angle
    
    return separation_angles

# Test with first sample
sample_keypoints = np.array(ast.literal_eval(df['keypoints_clean'][0]))
print(f"Keypoints shape: {sample_keypoints.shape}")

# Calculate hip-shoulder separation for the sample
hip_shoulder_angles = calculate_hip_shoulder_separation(sample_keypoints)
print(f"Hip-shoulder separation angles shape: {hip_shoulder_angles.shape}")
print(f"Mean separation angle: {np.mean(hip_shoulder_angles):.4f} radians ({np.degrees(np.mean(hip_shoulder_angles)):.2f} degrees)")

# Plot hip-shoulder separation over time
plt.figure(figsize=(12, 6))
plt.plot(range(len(hip_shoulder_angles)), np.degrees(hip_shoulder_angles), 'b-', linewidth=2)
plt.xlabel('Frame Number')
plt.ylabel('Hip-Shoulder Separation Angle (degrees)')
plt.title('Hip-Shoulder Separation Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate hip-shoulder separation for all rows in the dataset
print("Calculating hip-shoulder separation angles for all samples...")

# Initialize list to store hip-shoulder separation data
hip_shoulder_separation_data = []

for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"Processing row {idx}/{len(df)}")
    
    # Check if keypoints data exists and is valid
    if pd.isna(row['keypoints_clean']) or row['keypoints_clean'] == '':
        hip_shoulder_separation_data.append(None)
        continue
    
    try:
        # Parse keypoints data
        keypoints = np.array(ast.literal_eval(row['keypoints_clean']))
        
        # Calculate hip-shoulder separation angles
        hip_shoulder_angles = calculate_hip_shoulder_separation(keypoints)
        
        # Store the time series as a list
        hip_shoulder_separation_data.append(hip_shoulder_angles.tolist())
        
    except (ValueError, SyntaxError, TypeError) as e:
        # Handle parsing errors
        hip_shoulder_separation_data.append(None)
        if idx < 10:  # Only print first few errors to avoid spam
            print(f"Error processing row {idx}: {e}")

# Add the hip-shoulder separation data to the dataframe
df['hip_shoulder_separation'] = hip_shoulder_separation_data

# Print summary statistics
valid_separations = [sep for sep in hip_shoulder_separation_data if sep is not None]
print(f"\nProcessed {len(df)} rows")
print(f"Valid hip-shoulder separation calculations: {len(valid_separations)}")
print(f"Invalid/missing data: {len(df) - len(valid_separations)}")

if valid_separations:
    # Calculate some statistics
    all_angles = [angle for sequence in valid_separations for angle in sequence]
    mean_angle = np.mean(all_angles)
    std_angle = np.std(all_angles)
    
    print(f"Overall mean hip-shoulder separation: {mean_angle:.4f} radians ({np.degrees(mean_angle):.2f} degrees)")
    print(f"Overall std hip-shoulder separation: {std_angle:.4f} radians ({np.degrees(std_angle):.2f} degrees)")


# %%
# Create time series analysis of hip-shoulder separation by gender
import matplotlib.pyplot as plt
import seaborn as sns

# Filter data with valid hip-shoulder separation
valid_data = df[df['hip_shoulder_separation'].notna() & 
               (df['hip_shoulder_separation'] != '') &
               df['server_gender'].notna()].copy()

print(f"Valid data for analysis: {len(valid_data)} samples")

# Normalize all sequences to 100 time points and apply smoothing
normalized_sequences = {}
for gender in ['M', 'F']:
    gender_data = valid_data[valid_data['server_gender'] == gender]
    sequences = []
    
    for _, row in gender_data.iterrows():
        seq = row['hip_shoulder_separation']
        if seq and len(seq) > 0:
            normalized_seq = normalize_sequence_to_100(seq)
            if normalized_seq is not None:
                # Apply smoothing to the normalized sequence
                smoothed_seq = smooth_sequence(normalized_seq, sigma=2.0)
                sequences.append(smoothed_seq)
    
    normalized_sequences[gender] = sequences
    print(f"Gender {gender}: {len(sequences)} valid sequences")

# Calculate mean and standard deviation for each gender
time_points = np.arange(100)
gender_stats = {}

for gender in ['M', 'F']:
    if normalized_sequences[gender]:
        sequences_array = np.array(normalized_sequences[gender])
        mean_sequence = np.mean(sequences_array, axis=0)
        std_sequence = np.std(sequences_array, axis=0)
        
        # Apply additional smoothing to the mean and std
        smoothed_mean = smooth_sequence(mean_sequence, sigma=1.5)
        smoothed_std = smooth_sequence(std_sequence, sigma=1.5)
        
        gender_stats[gender] = {
            'mean': smoothed_mean,
            'std': smoothed_std,
            'upper': smoothed_mean + smoothed_std,
            'lower': smoothed_mean - smoothed_std,
            'count': len(sequences_array)
        }

# Create the time series plot
plt.figure(figsize=(14, 8))

colors = {'M': 'blue', 'F': 'red'}
labels = {'M': 'Men', 'F': 'Women'}

for gender in ['M', 'F']:
    if gender in gender_stats:
        stats = gender_stats[gender]
        
        # Plot mean line
        plt.plot(time_points, np.degrees(stats['mean']), 
                color=colors[gender], linewidth=3, 
                label=f"{labels[gender]} Mean (n={stats['count']})")
        
        # Plot standard deviation band
        plt.fill_between(time_points, 
                        np.degrees(stats['lower']), 
                        np.degrees(stats['upper']),
                        color=colors[gender], alpha=0.2, 
                        label=f"{labels[gender]} ±1 SD")

plt.xlabel('Time Point (Normalized to 100 frames)', fontsize=12)
plt.ylabel('Hip-Shoulder Separation (degrees)', fontsize=12)
plt.title('Hip-Shoulder Separation Over Time by Gender (Smoothed)\n(Mean ± 1 Standard Deviation)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add some statistics text
if 'M' in gender_stats and 'F' in gender_stats:
    male_mean = np.mean(np.degrees(gender_stats['M']['mean']))
    female_mean = np.mean(np.degrees(gender_stats['F']['mean']))
    
    plt.text(0.02, 0.98, 
             f"Overall Mean:\nMen: {male_mean:.2f}°\nWomen: {female_mean:.2f}°\nDifference: {male_mean-female_mean:.2f}°",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.show()

# Print summary statistics
print("\nSummary Statistics (Smoothed):")
for gender in ['M', 'F']:
    if gender in gender_stats:
        stats = gender_stats[gender]
        mean_deg = np.degrees(stats['mean'])
        std_deg = np.degrees(stats['std'])
        
        print(f"\n{labels[gender]} (n={stats['count']}):")
        print(f"  Mean separation across time: {np.mean(mean_deg):.2f}° ± {np.mean(std_deg):.2f}°")
        print(f"  Range: {np.min(mean_deg):.2f}° to {np.max(mean_deg):.2f}°")
        print(f"  Peak separation: {np.max(mean_deg):.2f}° at frame {np.argmax(mean_deg)}")
        print(f"  Minimum separation: {np.min(mean_deg):.2f}° at frame {np.argmin(mean_deg)}")



# %%
# Normalize hip-shoulder separation sequences to 100 indices and create time series plot separated by gender
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def normalize_sequence_to_100(sequence):
    """Normalize a sequence to exactly 100 time points using interpolation."""
    if sequence is None or len(sequence) == 0:
        return None
    
    sequence = np.array(sequence)
    if len(sequence) == 1:
        # If only one point, repeat it 100 times
        return np.full(100, sequence[0])
    
    # Create interpolation function
    original_indices = np.linspace(0, 1, len(sequence))
    target_indices = np.linspace(0, 1, 100)
    
    interp_func = interp1d(original_indices, sequence, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    
    return interp_func(target_indices)

def smooth_sequence(sequence, sigma=1.5):
    """Apply Gaussian smoothing to a sequence."""
    if sequence is None or len(sequence) == 0:
        return sequence
    
    # Apply Gaussian filter for smoothing
    smoothed = gaussian_filter1d(sequence, sigma=sigma)
    return smoothed

def calculate_velocity(sequence):
    """Calculate velocity (first derivative) of a sequence."""
    if sequence is None or len(sequence) < 2:
        return None
    
    # Calculate velocity using numpy gradient
    velocity = np.gradient(sequence)
    return velocity

def calculate_acceleration(velocity):
    """Calculate acceleration (second derivative) of a sequence."""
    if velocity is None or len(velocity) < 2:
        return None
    
    # Calculate acceleration using numpy gradient of velocity
    acceleration = np.gradient(velocity)
    return acceleration

# Separate data by gender and normalize sequences
male_sequences = []
female_sequences = []
male_velocities = []
female_velocities = []
male_accelerations = []
female_accelerations = []

for idx, row in df.iterrows():
    normalized = normalize_sequence_to_100(row['hip_shoulder_separation'])
    if normalized is not None:
        # Apply smoothing to the normalized sequence
        smoothed = smooth_sequence(normalized)
        # Convert to degrees
        smoothed_degrees = np.degrees(smoothed)
        
        # Calculate velocity and acceleration
        velocity = calculate_velocity(smoothed_degrees)
        acceleration = calculate_acceleration(velocity) if velocity is not None else None
        
        if row['server_gender'] == 'M':
            male_sequences.append(smoothed_degrees)
            if velocity is not None:
                male_velocities.append(velocity)
            if acceleration is not None:
                male_accelerations.append(acceleration)
        elif row['server_gender'] == 'F':
            female_sequences.append(smoothed_degrees)
            if velocity is not None:
                female_velocities.append(velocity)
            if acceleration is not None:
                female_accelerations.append(acceleration)

print(f"Successfully normalized and smoothed sequences:")
print(f"Male sequences: {len(male_sequences)}")
print(f"Female sequences: {len(female_sequences)}")
print(f"Male velocities: {len(male_velocities)}")
print(f"Female velocities: {len(female_velocities)}")
print(f"Male accelerations: {len(male_accelerations)}")
print(f"Female accelerations: {len(female_accelerations)}")

if len(male_sequences) > 0 or len(female_sequences) > 0:
    # Create time indices (0 to 99 representing normalized time)
    time_indices = np.arange(100)
    
    # Create the plot with 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    colors = {'male': 'blue', 'female': 'red'}
    
    # TOP ROW: VELOCITY PLOTS
    # Plot 1,1: Male velocities
    if len(male_velocities) > 0:
        male_vel_array = np.array(male_velocities)
        male_vel_mean = np.mean(male_vel_array, axis=0)
        male_vel_std = np.std(male_vel_array, axis=0)
        
        # Plot individual velocity sequences with low alpha
        for velocity in male_vel_array[:100]:  # Plot first 100 for visibility
            axes[0,0].plot(time_indices, velocity, color='lightblue', alpha=0.1, linewidth=0.5)
        
        # Plot mean line and std bands
        axes[0,0].plot(time_indices, male_vel_mean, color=colors['male'], linewidth=3, label='Male Velocity Mean')
        axes[0,0].fill_between(time_indices, 
                        male_vel_mean - male_vel_std, 
                        male_vel_mean + male_vel_std,
                        color=colors['male'], alpha=0.2, label='±1 Standard Deviation')
        
        axes[0,0].set_title('Male Hip-Shoulder Separation Velocity', fontsize=14)
        axes[0,0].set_ylabel('Velocity (degrees/frame)', fontsize=12)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 1,2: Female velocities
    if len(female_velocities) > 0:
        female_vel_array = np.array(female_velocities)
        female_vel_mean = np.mean(female_vel_array, axis=0)
        female_vel_std = np.std(female_vel_array, axis=0)
        
        # Plot individual velocity sequences with low alpha
        for velocity in female_vel_array[:100]:  # Plot first 100 for visibility
            axes[0,1].plot(time_indices, velocity, color='lightcoral', alpha=0.1, linewidth=0.5)
        
        # Plot mean line and std bands
        axes[0,1].plot(time_indices, female_vel_mean, color=colors['female'], linewidth=3, label='Female Velocity Mean')
        axes[0,1].fill_between(time_indices, 
                        female_vel_mean - female_vel_std, 
                        female_vel_mean + female_vel_std,
                        color=colors['female'], alpha=0.2, label='±1 Standard Deviation')
        
        axes[0,1].set_title('Female Hip-Shoulder Separation Velocity', fontsize=14)
        axes[0,1].set_ylabel('Velocity (degrees/frame)', fontsize=12)
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 1,3: Velocity comparison
    if len(male_velocities) > 0 and len(female_velocities) > 0:
        axes[0,2].plot(time_indices, male_vel_mean, color=colors['male'], linewidth=3, label='Male Velocity Mean')
        axes[0,2].plot(time_indices, female_vel_mean, color=colors['female'], linewidth=3, label='Female Velocity Mean')
        
        # Add standard deviation bands
        axes[0,2].fill_between(time_indices, 
                        male_vel_mean - male_vel_std, 
                        male_vel_mean + male_vel_std,
                        color=colors['male'], alpha=0.2)
        axes[0,2].fill_between(time_indices, 
                        female_vel_mean - female_vel_std, 
                        female_vel_mean + female_vel_std,
                        color=colors['female'], alpha=0.2)
        
        axes[0,2].set_title('Male vs Female Velocity Comparison', fontsize=14)
        axes[0,2].set_ylabel('Velocity (degrees/frame)', fontsize=12)
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].legend()
        axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # BOTTOM ROW: ACCELERATION PLOTS
    # Plot 2,1: Male accelerations
    if len(male_accelerations) > 0:
        male_acc_array = np.array(male_accelerations)
        male_acc_mean = np.mean(male_acc_array, axis=0)
        male_acc_std = np.std(male_acc_array, axis=0)
        
        # Plot individual acceleration sequences with low alpha
        for acceleration in male_acc_array[:100]:  # Plot first 100 for visibility
            axes[1,0].plot(time_indices, acceleration, color='lightblue', alpha=0.1, linewidth=0.5)
        
        # Plot mean line and std bands
        axes[1,0].plot(time_indices, male_acc_mean, color=colors['male'], linewidth=3, label='Male Acceleration Mean')
        axes[1,0].fill_between(time_indices, 
                        male_acc_mean - male_acc_std, 
                        male_acc_mean + male_acc_std,
                        color=colors['male'], alpha=0.2, label='±1 Standard Deviation')
        
        axes[1,0].set_title('Male Hip-Shoulder Separation Acceleration', fontsize=14)
        axes[1,0].set_ylabel('Acceleration (degrees/frame²)', fontsize=12)
        axes[1,0].set_xlabel('Normalized Time Index (0-99)', fontsize=12)
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2,2: Female accelerations
    if len(female_accelerations) > 0:
        female_acc_array = np.array(female_accelerations)
        female_acc_mean = np.mean(female_acc_array, axis=0)
        female_acc_std = np.std(female_acc_array, axis=0)
        
        # Plot individual acceleration sequences with low alpha
        for acceleration in female_acc_array[:100]:  # Plot first 100 for visibility
            axes[1,1].plot(time_indices, acceleration, color='lightcoral', alpha=0.1, linewidth=0.5)
        
        # Plot mean line and std bands
        axes[1,1].plot(time_indices, female_acc_mean, color=colors['female'], linewidth=3, label='Female Acceleration Mean')
        axes[1,1].fill_between(time_indices, 
                        female_acc_mean - female_acc_std, 
                        female_acc_mean + female_acc_std,
                        color=colors['female'], alpha=0.2, label='±1 Standard Deviation')
        
        axes[1,1].set_title('Female Hip-Shoulder Separation Acceleration', fontsize=14)
        axes[1,1].set_ylabel('Acceleration (degrees/frame²)', fontsize=12)
        axes[1,1].set_xlabel('Normalized Time Index (0-99)', fontsize=12)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2,3: Acceleration comparison
    if len(male_accelerations) > 0 and len(female_accelerations) > 0:
        axes[1,2].plot(time_indices, male_acc_mean, color=colors['male'], linewidth=3, label='Male Acceleration Mean')
        axes[1,2].plot(time_indices, female_acc_mean, color=colors['female'], linewidth=3, label='Female Acceleration Mean')
        
        # Add standard deviation bands
        axes[1,2].fill_between(time_indices, 
                        male_acc_mean - male_acc_std, 
                        male_acc_mean + male_acc_std,
                        color=colors['male'], alpha=0.2)
        axes[1,2].fill_between(time_indices, 
                        female_acc_mean - female_acc_std, 
                        female_acc_mean + female_acc_std,
                        color=colors['female'], alpha=0.2)
        
        axes[1,2].set_title('Male vs Female Acceleration Comparison', fontsize=14)
        axes[1,2].set_ylabel('Acceleration (degrees/frame²)', fontsize=12)
        axes[1,2].set_xlabel('Normalized Time Index (0-99)', fontsize=12)
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].legend()
        axes[1,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics by gender
    print(f"\nTime series statistics by gender (with smoothing, in degrees):")
    
    if len(male_sequences) > 0:
        male_array = np.array(male_sequences)
        male_mean = np.mean(male_array, axis=0)
        male_std = np.std(male_array, axis=0)
        
        print(f"\nMale statistics (n={len(male_sequences)}):")
        print(f"Mean separation at start: {male_mean[0]:.2f}°")
        print(f"Mean separation at middle: {male_mean[49]:.2f}°")
        print(f"Mean separation at end: {male_mean[99]:.2f}°")
        print(f"Overall mean: {np.mean(male_mean):.2f}°")
        print(f"Overall std: {np.mean(male_std):.2f}°")
        
        if len(male_velocities) > 0:
            print(f"Velocity statistics:")
            print(f"Mean velocity at start: {male_vel_mean[0]:.2f}°/frame")
            print(f"Mean velocity at middle: {male_vel_mean[49]:.2f}°/frame")
            print(f"Mean velocity at end: {male_vel_mean[99]:.2f}°/frame")
            print(f"Max velocity: {np.max(male_vel_mean):.2f}°/frame")
            print(f"Min velocity: {np.min(male_vel_mean):.2f}°/frame")
        
        if len(male_accelerations) > 0:
            print(f"Acceleration statistics:")
            print(f"Mean acceleration at start: {male_acc_mean[0]:.2f}°/frame²")
            print(f"Mean acceleration at middle: {male_acc_mean[49]:.2f}°/frame²")
            print(f"Mean acceleration at end: {male_acc_mean[99]:.2f}°/frame²")
            print(f"Max acceleration: {np.max(male_acc_mean):.2f}°/frame²")
            print(f"Min acceleration: {np.min(male_acc_mean):.2f}°/frame²")
    
    if len(female_sequences) > 0:
        female_array = np.array(female_sequences)
        female_mean = np.mean(female_array, axis=0)
        female_std = np.std(female_array, axis=0)
        
        print(f"\nFemale statistics (n={len(female_sequences)}):")
        print(f"Mean separation at start: {female_mean[0]:.2f}°")
        print(f"Mean separation at middle: {female_mean[49]:.2f}°")
        print(f"Mean separation at end: {female_mean[99]:.2f}°")
        print(f"Overall mean: {np.mean(female_mean):.2f}°")
        print(f"Overall std: {np.mean(female_std):.2f}°")
        
        if len(female_velocities) > 0:
            print(f"Velocity statistics:")
            print(f"Mean velocity at start: {female_vel_mean[0]:.2f}°/frame")
            print(f"Mean velocity at middle: {female_vel_mean[49]:.2f}°/frame")
            print(f"Mean velocity at end: {female_vel_mean[99]:.2f}°/frame")
            print(f"Max velocity: {np.max(female_vel_mean):.2f}°/frame")
            print(f"Min velocity: {np.min(female_vel_mean):.2f}°/frame")
        
        if len(female_accelerations) > 0:
            print(f"Acceleration statistics:")
            print(f"Mean acceleration at start: {female_acc_mean[0]:.2f}°/frame²")
            print(f"Mean acceleration at middle: {female_acc_mean[49]:.2f}°/frame²")
            print(f"Mean acceleration at end: {female_acc_mean[99]:.2f}°/frame²")
            print(f"Max acceleration: {np.max(female_acc_mean):.2f}°/frame²")
            print(f"Min acceleration: {np.min(female_acc_mean):.2f}°/frame²")
    
    # Statistical comparison if both genders available
    if len(male_sequences) > 0 and len(female_sequences) > 0:
        from scipy import stats
        
        # Perform t-test on overall means for position
        male_all_values = male_array.flatten()
        female_all_values = female_array.flatten()
        
        t_stat, p_value = stats.ttest_ind(male_all_values, female_all_values)
        
        print(f"\nStatistical comparison (smoothed data, degrees):")
        print(f"Position T-statistic: {t_stat:.4f}")
        print(f"Position P-value: {p_value:.6f}")
        print(f"Position significant difference (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        # T-test for velocities if available
        if len(male_velocities) > 0 and len(female_velocities) > 0:
            male_vel_all = male_vel_array.flatten()
            female_vel_all = female_vel_array.flatten()
            
            vel_t_stat, vel_p_value = stats.ttest_ind(male_vel_all, female_vel_all)
            
            print(f"Velocity T-statistic: {vel_t_stat:.4f}")
            print(f"Velocity P-value: {vel_p_value:.6f}")
            print(f"Velocity significant difference (p<0.05): {'Yes' if vel_p_value < 0.05 else 'No'}")
        
        # T-test for accelerations if available
        if len(male_accelerations) > 0 and len(female_accelerations) > 0:
            male_acc_all = male_acc_array.flatten()
            female_acc_all = female_acc_array.flatten()
            
            acc_t_stat, acc_p_value = stats.ttest_ind(male_acc_all, female_acc_all)
            
            print(f"Acceleration T-statistic: {acc_t_stat:.4f}")
            print(f"Acceleration P-value: {acc_p_value:.6f}")
            print(f"Acceleration significant difference (p<0.05): {'Yes' if acc_p_value < 0.05 else 'No'}")

else:
    print("No valid hip-shoulder separation sequences found for plotting")

# %%
import json
import ast
from scipy.interpolate import interp1d
import numpy as np

def parse_joint_data(data_str):
    """Parse the joint velocity data string."""
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

def normalize_to_100_timesteps(sequence):
    """Normalize a sequence to 100 timesteps using interpolation."""
    if sequence is None or len(sequence) < 2:
        return None
    
    original_length = len(sequence)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, 100)
    
    # Use linear interpolation
    f = interp1d(original_indices, sequence, kind='linear')
    normalized_sequence = f(target_indices)
    
    return normalized_sequence.tolist()

# Joint names to analyze
joint_names = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]

# Dictionary to store standard deviation values for each server
server_std_dict = {}
# Dictionary to store overall server standard deviation (averaged across all joints)
server_overall_std = {}

# Process each joint
for joint in joint_names:
    velocity_col = f'{joint}_velocity'
    
    if velocity_col not in df.columns:
        print(f"Column {velocity_col} not found, skipping...")
        continue
    
    print(f"\nProcessing {joint} velocity...")
    
    # Collect sequences by server
    server_sequences = {}
    
    for idx, row in df.iterrows():
        if pd.isna(row['server_name']):
            continue
            
        velocity_data = parse_joint_data(row[velocity_col])
        if velocity_data is None:
            continue
            
        normalized_seq = normalize_to_100_timesteps(velocity_data)
        if normalized_seq is None:
            continue
            
        server_name = row['server_name']
        if server_name not in server_sequences:
            server_sequences[server_name] = []
        server_sequences[server_name].append(normalized_seq)
    
    print(f"  Found {len(server_sequences)} servers with valid sequences")
    
    # Calculate standard deviation for each server
    joint_std_dict = {}
    for server_name, sequences in server_sequences.items():
        if len(sequences) > 1:  # Need at least 2 sequences to calculate std
            server_array = np.array(sequences)
            # Calculate standard deviation across sequences for each timestep
            server_std_per_timestep = np.std(server_array, axis=0)
            # Average standard deviation across all timesteps
            server_avg_std = np.mean(server_std_per_timestep)
            joint_std_dict[server_name] = server_avg_std
            print(f"    {server_name}: {len(sequences)} sequences, avg_std={server_avg_std:.4f}")
    
    # Store in main dictionary
    server_std_dict[joint] = joint_std_dict
    
    # Colors for male and female (for visualization)
    male_color = '#1f77b4'  # Blue
    female_color = '#ff7f0e'  # Orange
    
    # Collect normalized sequences by gender for plotting
    male_sequences = []
    female_sequences = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['server_gender']):
            continue
            
        velocity_data = parse_joint_data(row[velocity_col])
        if velocity_data is None:
            continue
            
        normalized_seq = normalize_to_100_timesteps(velocity_data)
        if normalized_seq is None:
            continue
            
        if row['server_gender'] == 'M':
            male_sequences.append(normalized_seq)
        elif row['server_gender'] == 'F':
            female_sequences.append(normalized_seq)
    
    # Convert to numpy arrays for statistics
    if len(male_sequences) > 0:
        male_array = np.array(male_sequences)
        male_mean = np.mean(male_array, axis=0)
        male_std = np.std(male_array, axis=0)
        male_avg_std = np.mean(male_std)
    else:
        male_mean = male_std = male_avg_std = None
    
    if len(female_sequences) > 0:
        female_array = np.array(female_sequences)
        female_mean = np.mean(female_array, axis=0)
        female_std = np.std(female_array, axis=0)
        female_avg_std = np.mean(female_std)
    else:
        female_mean = female_std = female_avg_std = None
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    timesteps = np.arange(100)
    
    # Plot male data
    if male_mean is not None:
        plt.plot(timesteps, male_mean, color=male_color, linewidth=2, 
                label=f'Male (n={len(male_sequences)}, avg_std={male_avg_std:.3f})')
        plt.fill_between(timesteps, 
                        male_mean - male_std, 
                        male_mean + male_std,
                        color=male_color, alpha=0.3)
    
    # Plot female data
    if female_mean is not None:
        plt.plot(timesteps, female_mean, color=female_color, linewidth=2, 
                label=f'Female (n={len(female_sequences)}, avg_std={female_avg_std:.3f})')
        plt.fill_between(timesteps, 
                        female_mean - female_std, 
                        female_mean + female_std,
                        color=female_color, alpha=0.3)
    
    plt.title(f'{joint.replace("_", " ").title()} Velocity: Male vs Female\n(Normalized to 100 timesteps)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Timestep', fontsize=12)
    plt.ylabel('Velocity (radians/frame)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Calculate overall standard deviation for each server (averaged across all joints)
print("\n" + "="*80)
print("Calculating overall standard deviation for each server...")
print("="*80)

# Get all unique server names that have data
all_servers = set()
for joint_dict in server_std_dict.values():
    all_servers.update(joint_dict.keys())

# Calculate average std across all joints for each server
for server_name in all_servers:
    server_joint_stds = []
    for joint, joint_dict in server_std_dict.items():
        if server_name in joint_dict:
            server_joint_stds.append(joint_dict[server_name])
    
    if len(server_joint_stds) > 0:
        overall_avg_std = np.mean(server_joint_stds)
        server_overall_std[server_name] = overall_avg_std
        print(f"{server_name}: {overall_avg_std:.4f} (based on {len(server_joint_stds)} joints)")

# Print summary of server standard deviations
print("\n" + "="*80)
print("SUMMARY: Standard Deviation by Server for Each Joint")
print("="*80)

for joint, server_dict in server_std_dict.items():
    if len(server_dict) > 0:
        print(f"\n{joint.replace('_', ' ').title()}:")
        for server_name, std_value in sorted(server_dict.items(), key=lambda x: x[1]):
            print(f"  {server_name}: {std_value:.4f}")

print("\n" + "="*80)
print("OVERALL SERVER STANDARD DEVIATION (Averaged Across All Joints)")
print("="*80)
print("Format: Server Name: Overall Std Dev")

for server_name, overall_std in sorted(server_overall_std.items(), key=lambda x: x[1]):
    print(f"{server_name}: {overall_std:.4f}")

# %%
import json
import ast
from scipy.interpolate import interp1d
import numpy as np

def parse_joint_data(data_str):
    """Parse the joint velocity data string."""
    if pd.isna(data_str) or data_str == '':
        return None
    
    try:
        # Try to parse as JSON
        data = json.loads(data_str)
        return data
    except json.JSONDecodeError:
        try:
            # Try to parse as literal
            data = ast.literal_eval(data_str)
            return data
        except (ValueError, SyntaxError):
            return None

def normalize_to_100_timesteps(sequence):
    """Normalize a sequence to 100 timesteps using interpolation."""
    if sequence is None or len(sequence) < 2:
        return None
    
    original_length = len(sequence)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, 100)
    
    # Use linear interpolation
    f = interp1d(original_indices, sequence, kind='linear')
    normalized_sequence = f(target_indices)
    
    return normalized_sequence.tolist()

# Joint names to analyze
joint_names = [
    'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee'
]

# Dictionary to store standard deviation values for each server
server_std_dict = {}
# Dictionary to store overall server standard deviation (averaged across all joints)
server_overall_std = {}

# Process each joint
for joint in joint_names:
    velocity_col = f'{joint}_velocity'
    
    if velocity_col not in df.columns:
        print(f"Column {velocity_col} not found, skipping...")
        continue
    
    print(f"\nProcessing {joint} velocity...")
    
    # Collect sequences by server
    server_sequences = {}
    
    for idx, row in df.iterrows():
        if pd.isna(row['server_name']):
            continue
            
        velocity_data = parse_joint_data(row[velocity_col])
        if velocity_data is None:
            continue
            
        normalized_seq = normalize_to_100_timesteps(velocity_data)
        if normalized_seq is None:
            continue
            
        server_name = row['server_name']
        if server_name not in server_sequences:
            server_sequences[server_name] = []
        server_sequences[server_name].append(normalized_seq)
    
    print(f"  Found {len(server_sequences)} servers with valid sequences")
    
    # Calculate standard deviation for each server
    joint_std_dict = {}
    for server_name, sequences in server_sequences.items():
        if len(sequences) > 1:  # Need at least 2 sequences to calculate std
            server_array = np.array(sequences)
            # Calculate standard deviation across sequences for each timestep
            server_std_per_timestep = np.std(server_array, axis=0)
            # Average standard deviation across all timesteps
            server_avg_std = np.mean(server_std_per_timestep)
            joint_std_dict[server_name] = server_avg_std
            print(f"    {server_name}: {len(sequences)} sequences, avg_std={server_avg_std:.4f}")
    
    # Store in main dictionary
    server_std_dict[joint] = joint_std_dict
    
    # Colors for male and female (for visualization)
    male_color = '#1f77b4'  # Blue
    female_color = '#ff7f0e'  # Orange
    
    # Collect normalized sequences by gender for plotting
    male_sequences = []
    female_sequences = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['server_gender']):
            continue
            
        velocity_data = parse_joint_data(row[velocity_col])
        if velocity_data is None:
            continue
            
        normalized_seq = normalize_to_100_timesteps(velocity_data)
        if normalized_seq is None:
            continue
            
        if row['server_gender'] == 'M':
            male_sequences.append(normalized_seq)
        elif row['server_gender'] == 'F':
            female_sequences.append(normalized_seq)
    
    # Convert to numpy arrays for statistics
    if len(male_sequences) > 0:
        male_array = np.array(male_sequences)
        male_mean = np.mean(male_array, axis=0)
        male_std = np.std(male_array, axis=0)
        male_avg_std = np.mean(male_std)
    else:
        male_mean = male_std = male_avg_std = None
    
    if len(female_sequences) > 0:
        female_array = np.array(female_sequences)
        female_mean = np.mean(female_array, axis=0)
        female_std = np.std(female_array, axis=0)
        female_avg_std = np.mean(female_std)
    else:
        female_mean = female_std = female_avg_std = None
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    timesteps = np.arange(100)
    
    # Plot male data
    if male_mean is not None:
        plt.plot(timesteps, male_mean, color=male_color, linewidth=2, 
                label=f'Male (n={len(male_sequences)}, avg_std={male_avg_std:.3f})')
        plt.fill_between(timesteps, 
                        male_mean - male_std, 
                        male_mean + male_std,
                        color=male_color, alpha=0.3)
    
    # Plot female data
    if female_mean is not None:
        plt.plot(timesteps, female_mean, color=female_color, linewidth=2, 
                label=f'Female (n={len(female_sequences)}, avg_std={female_avg_std:.3f})')
        plt.fill_between(timesteps, 
                        female_mean - female_std, 
                        female_mean + female_std,
                        color=female_color, alpha=0.3)
    
    plt.title(f'{joint.replace("_", " ").title()} Velocity: Male vs Female\n(Normalized to 100 timesteps)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Timestep', fontsize=12)
    plt.ylabel('Velocity (radians/frame)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Calculate overall standard deviation for each server (averaged across all joints)
print("\n" + "="*80)
print("Calculating overall standard deviation for each server...")
print("="*80)

# Get all unique server names that have data
all_servers = set()
for joint_dict in server_std_dict.values():
    all_servers.update(joint_dict.keys())

# Calculate average std across all joints for each server
for server_name in all_servers:
    server_joint_stds = []
    for joint, joint_dict in server_std_dict.items():
        if server_name in joint_dict:
            server_joint_stds.append(joint_dict[server_name])
    
    if len(server_joint_stds) > 0:
        overall_avg_std = np.mean(server_joint_stds)
        server_overall_std[server_name] = overall_avg_std
        print(f"{server_name}: {overall_avg_std:.4f} (based on {len(server_joint_stds)} joints)")

# Print summary of server standard deviations
print("\n" + "="*80)
print("SUMMARY: Standard Deviation by Server for Each Joint")
print("="*80)

for joint, server_dict in server_std_dict.items():
    if len(server_dict) > 0:
        print(f"\n{joint.replace('_', ' ').title()}:")
        for server_name, std_value in sorted(server_dict.items(), key=lambda x: x[1]):
            print(f"  {server_name}: {std_value:.4f}")

print("\n" + "="*80)
print("OVERALL SERVER STANDARD DEVIATION (Averaged Across All Joints)")
print("="*80)
print("Format: Server Name: Overall Std Dev")

for server_name, overall_std in sorted(server_overall_std.items(), key=lambda x: x[1]):
    print(f"{server_name}: {overall_std:.4f}")

# Create comprehensive visualization showing the data processing workflow
print("\n" + "="*80)
print("CREATING COMPREHENSIVE VISUALIZATION OF DATA PROCESSING WORKFLOW")
print("="*80)

# Create a figure with multiple subplots to show the process
fig = plt.figure(figsize=(20, 16))

# 1. Raw data distribution by gender
ax1 = plt.subplot(4, 3, 1)
gender_counts = df['server_gender'].value_counts()
colors = ['#1f77b4', '#ff7f0e']
plt.pie(gender_counts.values, labels=['Male', 'Female'], autopct='%1.1f%%', colors=colors)
plt.title('Data Distribution by Gender', fontsize=12, fontweight='bold')

# 2. Number of serves per player
ax2 = plt.subplot(4, 3, 2)
server_counts = df['server_name'].value_counts().head(10)
plt.barh(range(len(server_counts)), server_counts.values, color='skyblue')
plt.yticks(range(len(server_counts)), server_counts.index, fontsize=8)
plt.xlabel('Number of Serves')
plt.title('Top 10 Players by Number of Serves', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()

# 3. Distribution of sequence lengths before normalization
ax3 = plt.subplot(4, 3, 3)
sequence_lengths = []
for idx, row in df.iterrows():
    if not pd.isna(row['left_elbow_velocity']):
        velocity_data = parse_joint_data(row['left_elbow_velocity'])
        if velocity_data:
            sequence_lengths.append(len(velocity_data))

if sequence_lengths:
    plt.hist(sequence_lengths, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Original Sequence Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Original Sequence Lengths\n(Before Normalization)', fontsize=12, fontweight='bold')
    plt.axvline(np.mean(sequence_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(sequence_lengths):.1f}')
    plt.legend()

# 4-6. Show example velocity profiles for different joints (Male vs Female)
example_joints = ['left_elbow', 'right_shoulder', 'left_knee']
for i, joint in enumerate(example_joints):
    ax = plt.subplot(4, 3, 4+i)
    
    # Get example sequences
    male_examples = []
    female_examples = []
    
    velocity_col = f'{joint}_velocity'
    if velocity_col in df.columns:
        for idx, row in df.sample(min(50, len(df))).iterrows():
            if pd.isna(row['server_gender']):
                continue
                
            velocity_data = parse_joint_data(row[velocity_col])
            if velocity_data is None:
                continue
                
            normalized_seq = normalize_to_100_timesteps(velocity_data)
            if normalized_seq is None:
                continue
                
            if row['server_gender'] == 'M' and len(male_examples) < 5:
                male_examples.append(normalized_seq)
            elif row['server_gender'] == 'F' and len(female_examples) < 5:
                female_examples.append(normalized_seq)
    
    # Plot example sequences
    timesteps = np.arange(100)
    for seq in male_examples:
        plt.plot(timesteps, seq, color='#1f77b4', alpha=0.6, linewidth=1)
    for seq in female_examples:
        plt.plot(timesteps, seq, color='#ff7f0e', alpha=0.6, linewidth=1)
    
    # Add legend only to first plot
    if i == 0:
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='#1f77b4', lw=2, label='Male'),
                          Line2D([0], [0], color='#ff7f0e', lw=2, label='Female')]
        plt.legend(handles=legend_elements)
    
    plt.title(f'{joint.replace("_", " ").title()} Velocity\n(Example Sequences)', fontsize=10, fontweight='bold')
    plt.xlabel('Normalized Timestep')
    plt.ylabel('Velocity (rad/frame)')
    plt.grid(True, alpha=0.3)

# 7. Standard deviation by joint type
ax7 = plt.subplot(4, 3, 7)
joint_avg_stds = {}
for joint, server_dict in server_std_dict.items():
    if len(server_dict) > 0:
        joint_avg_stds[joint] = np.mean(list(server_dict.values()))

if joint_avg_stds:
    joints = list(joint_avg_stds.keys())
    stds = list(joint_avg_stds.values())
    
    plt.barh(range(len(joints)), stds, color='lightcoral')
    plt.yticks(range(len(joints)), [j.replace('_', ' ').title() for j in joints])
    plt.xlabel('Average Standard Deviation')
    plt.title('Movement Variability by Joint', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()

# 8. Distribution of server standard deviations
ax8 = plt.subplot(4, 3, 8)
if server_overall_std:
    std_values = list(server_overall_std.values())
    plt.hist(std_values, bins=15, alpha=0.7, color='mediumpurple', edgecolor='black')
    plt.xlabel('Overall Standard Deviation')
    plt.ylabel('Number of Players')
    plt.title('Distribution of Player Movement\nVariability Scores', fontsize=12, fontweight='bold')
    plt.axvline(np.mean(std_values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(std_values):.3f}')
    plt.legend()

# 9. Gender comparison of movement variability
ax9 = plt.subplot(4, 3, 9)
male_stds = []
female_stds = []

for server_name, std_val in server_overall_std.items():
    server_data = df[df['server_name'] == server_name]
    if len(server_data) > 0:
        gender = server_data['server_gender'].iloc[0]
        if gender == 'M':
            male_stds.append(std_val)
        elif gender == 'F':
            female_stds.append(std_val)

if male_stds and female_stds:
    plt.boxplot([male_stds, female_stds], labels=['Male', 'Female'])
    plt.ylabel('Movement Variability Score')
    plt.title('Movement Variability by Gender', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

# 10. Data processing pipeline visualization
ax10 = plt.subplot(4, 3, 10)
plt.text(0.5, 0.9, 'DATA PROCESSING PIPELINE', ha='center', va='center', 
         fontsize=14, fontweight='bold', transform=ax10.transAxes)

pipeline_steps = [
    '1. Raw velocity sequences\n   (variable length)',
    '2. Parse JSON/literal data\n   (clean & validate)',
    '3. Normalize to 100 timesteps\n   (interpolation)',
    '4. Calculate per-timestep std\n   (across serves)',
    '5. Average std across time\n   (single score per player)',
    '6. Compare male vs female\n   (statistical analysis)'
]

for i, step in enumerate(pipeline_steps):
    y_pos = 0.8 - (i * 0.12)
    plt.text(0.05, y_pos, step, ha='left', va='top', fontsize=9, 
             transform=ax10.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
             facecolor='lightblue', alpha=0.7))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# 11. Sample size information
ax11 = plt.subplot(4, 3, 11)
plt.text(0.5, 0.9, 'SAMPLE SIZE INFORMATION', ha='center', va='center', 
         fontsize=14, fontweight='bold', transform=ax11.transAxes)

total_serves = len(df)
valid_serves = len(df.dropna(subset=['server_name', 'server_gender']))
male_serves = len(df[df['server_gender'] == 'M'])
female_serves = len(df[df['server_gender'] == 'F'])
unique_players = len(server_overall_std)

info_text = f"""
Total serves in dataset: {total_serves:,}
Valid serves (with gender): {valid_serves:,}
Male serves: {male_serves:,}
Female serves: {female_serves:,}
Unique players analyzed: {unique_players}
Joints analyzed: {len(joint_names)}
"""

plt.text(0.05, 0.7, info_text, ha='left', va='top', fontsize=11, 
         transform=ax11.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='lightyellow', alpha=0.8))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

# 12. Key findings summary
ax12 = plt.subplot(4, 3, 12)
plt.text(0.5, 0.9, 'KEY FINDINGS', ha='center', va='center', 
         fontsize=14, fontweight='bold', transform=ax12.transAxes)

if male_stds and female_stds:
    male_mean = np.mean(male_stds)
    female_mean = np.mean(female_stds)
    findings_text = f"""
Male players analyzed: {len(male_stds)}
Female players analyzed: {len(female_stds)}

Average movement variability:
• Male: {male_mean:.4f}
• Female: {female_mean:.4f}

Difference: {abs(male_mean - female_mean):.4f}
Higher variability: {'Male' if male_mean > female_mean else 'Female'}
"""
else:
    findings_text = "Insufficient data for\ngender comparison"

plt.text(0.05, 0.7, findings_text, ha='left', va='top', fontsize=10, 
         transform=ax12.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='lightgreen', alpha=0.8))

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis('off')

plt.suptitle('Tennis Serve Biomechanics Analysis: Data Processing Workflow & Results', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()

# %%
# Calculate average serve speed for each server and correlate with overall standard deviation
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Server Speed vs Movement Variability")
print("="*80)

# Calculate average Speed_MPH for each server
server_avg_speed = {}
for server_name in server_overall_std.keys():
    server_data = df[df['server_name'] == server_name]
    if len(server_data) > 0 and 'Speed_MPH' in server_data.columns:
        # Remove NaN values and calculate mean
        speed_data = server_data['Speed_MPH'].dropna()
        if len(speed_data) > 0:
            avg_speed = speed_data.mean()
            server_avg_speed[server_name] = avg_speed
            print(f"{server_name}: {avg_speed:.2f} MPH (from {len(speed_data)} serves)")

print(f"\nFound speed data for {len(server_avg_speed)} servers")
print(f"Found movement variability data for {len(server_overall_std)} servers")

# Find common servers between both datasets
common_servers = set(server_avg_speed.keys()) & set(server_overall_std.keys())
print(f"Common servers with both speed and variability data: {len(common_servers)}")

if len(common_servers) >= 3:  # Need at least 3 points for meaningful correlation
    # Prepare data for correlation analysis
    speeds = []
    variabilities = []
    server_names = []
    
    for server_name in common_servers:
        speeds.append(server_avg_speed[server_name])
        variabilities.append(server_overall_std[server_name])
        server_names.append(server_name)
    
    # Convert to numpy arrays
    speeds = np.array(speeds)
    variabilities = np.array(variabilities)
    
    # Calculate correlation
    correlation_coef = np.corrcoef(speeds, variabilities)[0, 1]
    
    print(f"\n" + "="*60)
    print("CORRELATION RESULTS")
    print("="*60)
    print(f"Correlation coefficient: {correlation_coef:.4f}")
    
    if abs(correlation_coef) > 0.7:
        strength = "Strong"
    elif abs(correlation_coef) > 0.5:
        strength = "Moderate"
    elif abs(correlation_coef) > 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if correlation_coef > 0 else "negative"
    print(f"Correlation strength: {strength} {direction}")
    
    if correlation_coef > 0:
        print("Interpretation: Higher serve speeds tend to be associated with higher movement variability")
    else:
        print("Interpretation: Higher serve speeds tend to be associated with lower movement variability")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(speeds, variabilities, alpha=0.7, s=60, color='blue')
    
    # Add trend line
    z = np.polyfit(speeds, variabilities, 1)
    p = np.poly1d(z)
    plt.plot(speeds, p(speeds), "r--", alpha=0.8, linewidth=2)
    
    # Add labels for each point
    for i, server_name in enumerate(server_names):
        plt.annotate(server_name, (speeds[i], variabilities[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('Average Serve Speed (MPH)', fontsize=12)
    plt.ylabel('Overall Movement Variability (Standard Deviation)', fontsize=12)
    plt.title(f'Serve Speed vs Movement Variability\nCorrelation: {correlation_coef:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print detailed data table
    print(f"\n" + "="*80)
    print("DETAILED DATA TABLE")
    print("="*80)
    print(f"{'Server Name':<25} {'Avg Speed (MPH)':<15} {'Movement Variability':<20}")
    print("-" * 80)
    
    # Sort by speed for better readability
    sorted_data = sorted(zip(server_names, speeds, variabilities), key=lambda x: x[1], reverse=True)
    for server_name, speed, variability in sorted_data:
        print(f"{server_name:<25} {speed:<15.2f} {variability:<20.4f}")

else:
    print(f"\nInsufficient data for correlation analysis. Need at least 3 servers with both speed and variability data.")
    print(f"Currently have {len(common_servers)} common servers.")


# %%
# Calculate average serve speed for each server and correlate with overall standard deviation
print("\n" + "="*80)
print("CORRELATION ANALYSIS: Server Speed vs Movement Variability")
print("="*80)

# Calculate average Speed_MPH for each server
server_avg_speed = {}
for server_name in server_overall_std.keys():
    server_data = df[df['server_name'] == server_name]
    if len(server_data) > 0 and 'Speed_MPH' in server_data.columns:
        # Remove NaN values and calculate mean
        speed_data = server_data['Speed_MPH'].dropna()
        if len(speed_data) > 0:
            avg_speed = speed_data.mean()
            server_avg_speed[server_name] = avg_speed
            print(f"{server_name}: {avg_speed:.2f} MPH (from {len(speed_data)} serves)")

print(f"\nFound speed data for {len(server_avg_speed)} servers")
print(f"Found movement variability data for {len(server_overall_std)} servers")

# Find common servers between both datasets
common_servers = set(server_avg_speed.keys()) & set(server_overall_std.keys())
print(f"Common servers with both speed and variability data: {len(common_servers)}")

if len(common_servers) >= 3:  # Need at least 3 points for meaningful correlation
    # Prepare data for correlation analysis
    speeds = []
    variabilities = []
    server_names = []
    
    for server_name in common_servers:
        speeds.append(server_avg_speed[server_name])
        variabilities.append(server_overall_std[server_name])
        server_names.append(server_name)
    
    # Convert to numpy arrays
    speeds = np.array(speeds)
    variabilities = np.array(variabilities)
    
    # Calculate correlation
    correlation_coef = np.corrcoef(speeds, variabilities)[0, 1]
    
    print(f"\n" + "="*60)
    print("CORRELATION RESULTS")
    print("="*60)
    print(f"Correlation coefficient: {correlation_coef:.4f}")
    
    if abs(correlation_coef) > 0.7:
        strength = "Strong"
    elif abs(correlation_coef) > 0.5:
        strength = "Moderate"
    elif abs(correlation_coef) > 0.3:
        strength = "Weak"
    else:
        strength = "Very weak"
    
    direction = "positive" if correlation_coef > 0 else "negative"
    print(f"Correlation strength: {strength} {direction}")
    
    if correlation_coef > 0:
        print("Interpretation: Higher serve speeds tend to be associated with higher movement variability")
    else:
        print("Interpretation: Higher serve speeds tend to be associated with lower movement variability")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(speeds, variabilities, alpha=0.7, s=60, color='blue')
    
    # Add trend line
    z = np.polyfit(speeds, variabilities, 1)
    p = np.poly1d(z)
    plt.plot(speeds, p(speeds), "r--", alpha=0.8, linewidth=2)
    
    # Add labels for each point
    for i, server_name in enumerate(server_names):
        plt.annotate(server_name, (speeds[i], variabilities[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('Average Serve Speed (MPH)', fontsize=12)
    plt.ylabel('Overall Movement Variability (Standard Deviation)', fontsize=12)
    plt.title(f'Serve Speed vs Movement Variability\nCorrelation: {correlation_coef:.4f}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print detailed data table
    print(f"\n" + "="*80)
    print("DETAILED DATA TABLE")
    print("="*80)
    print(f"{'Server Name':<25} {'Avg Speed (MPH)':<15} {'Movement Variability':<20}")
    print("-" * 80)
    
    # Sort by speed for better readability
    sorted_data = sorted(zip(server_names, speeds, variabilities), key=lambda x: x[1], reverse=True)
    for server_name, speed, variability in sorted_data:
        print(f"{server_name:<25} {speed:<15.2f} {variability:<20.4f}")

else:
    print(f"\nInsufficient data for correlation analysis. Need at least 3 servers with both speed and variability data.")
    print(f"Currently have {len(common_servers)} common servers.")



