#!/usr/bin/env python3
"""
Add server_name column to CSV by mapping server numbers to player names.
"""

import pandas as pd

# File paths
CSV_PATH = "data/full/usopen_points_clean_keypoints_cleaned.csv"
OUTPUT_CSV_PATH = "data/full/usopen_points_clean_keypoints_cleaned_with_server.csv"

def add_server_name_column():
    """Add server_name column based on PointServer and player names."""
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Create server_name column
    print("Adding server_name column...")
    df['server_name'] = df.apply(
        lambda row: row['player1'] if row['PointServer'] == 1 else row['player2'], 
        axis=1
    )
    
    print(f"Updated CSV shape: {df.shape}")
    print(f"Updated columns: {len(df.columns)}")
    
    # Show some examples
    print(f"\nSample server mappings:")
    sample_data = df[['player1', 'player2', 'PointServer', 'server_name']].head(10)
    for idx, row in sample_data.iterrows():
        print(f"  Row {idx}: Server {row['PointServer']} -> {row['server_name']}")
    
    # Verify the mapping worked correctly
    print(f"\nVerification:")
    server1_count = (df['PointServer'] == 1).sum()
    server1_name_count = (df['server_name'] == df['player1']).sum()
    server2_count = (df['PointServer'] == 2).sum()
    server2_name_count = (df['server_name'] == df['player2']).sum()
    
    print(f"  Server 1 points: {server1_count}, mapped to player1: {server1_name_count}")
    print(f"  Server 2 points: {server2_count}, mapped to player2: {server2_name_count}")
    
    if server1_count == server1_name_count and server2_count == server2_name_count:
        print("  ✅ Mapping verification successful!")
    else:
        print("  ❌ Mapping verification failed!")
    
    # Save updated CSV
    print(f"\nSaving updated CSV to {OUTPUT_CSV_PATH}")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    return df

def show_server_distribution(df):
    """Show distribution of serves by player."""
    print(f"\n=== SERVE DISTRIBUTION ===")
    
    # Count serves by player
    serve_counts = df['server_name'].value_counts()
    total_serves = len(df)
    
    print(f"Total serves: {total_serves}")
    print(f"\nServes by player:")
    for player, count in serve_counts.items():
        percentage = (count / total_serves) * 100
        print(f"  {player}: {count} serves ({percentage:.1f}%)")

def main():
    """Main function to add server name column."""
    print("Starting server name column addition...")
    
    # Add server name column
    df_updated = add_server_name_column()
    
    # Show distribution
    show_server_distribution(df_updated)
    
    print(f"\n✅ Server name column addition complete!")
    print(f"Output file: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main() 