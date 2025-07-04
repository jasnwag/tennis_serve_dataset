#!/usr/bin/env python3
"""
Add server gender column by mapping player names to their genders.
"""

import pandas as pd

# File paths
CSV_PATH = "data/full/usopen_points_clean_keypoints_cleaned_with_server.csv"
OUTPUT_CSV_PATH = "data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv"

# Comprehensive player gender mapping
PLAYER_GENDER_MAP = {
    # Men's players
    'sinner': 'M', 'tiafoe': 'M', 'fritz': 'M', 'dimitrov': 'M', 'rublev': 'M', 
    'medvedev': 'M', 'zverev': 'M', 'minaur': 'M', 'zandschulp': 'M', 'djokovic': 'M',
    'ruud': 'M', 'alcaraz': 'M', 'paul': 'M', 'borges': 'M', 'monfils': 'M',
    'nakashima': 'M', 'popyrin': 'M', 'kokkinakis': 'M', 'evans': 'M', 'diallo': 'M',
    'goffin': 'M', 'lehecka': 'M', 'virtanen': 'M', 'sonego': 'M', 'etcheverry': 'M',
    'schwartzman': 'M', 'machac': 'M', 'mensik': 'M', "o'connell": 'M', 'djere': 'M',
    'musetti': 'M', 'albot': 'M', 'michelsen': 'M', 'cobolli': 'M', 'perricard': 'M',
    'berrettini': 'M', 'mcdonald': 'M', 'fils': 'M', 'kovacevic': 'M', 'draper': 'M',
    'purcell': 'M', 'marozsan': 'M', 'korda': 'M', 'thiem': 'M', 'tsitsipas': 'M',
    'rinderknech': 'M', 'rune': 'M', 'griekspoor': 'M', 'thompson': 'M', 'shelton': 'M',
    'hurkacz': 'M', 'hijikata': 'M', 'wawrinka': 'M', 'tabilo': 'M', 'shapovalov': 'M',
    'shang': 'M', 'lajovic': 'M', 'bellucci': 'M', 'mannarino': 'M', 'tu': 'M', 'navone': 'F',
    'bu': 'F',
    
    # Women's players
    'sabalenka': 'F', 'pegula': 'F', 'navarro': 'F', 'badosa': 'F', 'gauff': 'F',
    'muchova': 'F', 'zheng': 'F', 'swiatek': 'F', 'azarenka': 'F', 'mertens': 'F',
    'wozniacki': 'F', 'wang': 'F', 'maneiro': 'F', 'kenin': 'F', 'potapova': 'F',
    'burel': 'F', 'ruse': 'F', 'keys': 'F', 'svitolina': 'F',
    'vekic': 'F', 'osaka': 'F', 'raducanu': 'F', 'paolini': 'F', 'starodubtseva': 'F',
    'errani': 'F', 'boulter': 'F', 'stephens': 'F', 'alexandrova': 'F', 'andreescu': 'F',
    'pavlyuchenkova': 'F', 'rakhimova': 'F', 'krejcikova': 'F', 'samsonova': 'F',
    'krueger': 'F', 'ponchet': 'F', 'putintseva': 'F', 'kostyuk': 'F', 'gracheva': 'F',
    'zhang': 'F', 'hon': 'F', 'rybakina': 'F',  'niemeier': 'F',
    'maria': 'F', 'andreeva': 'F', 'kalinina': 'F', 'dolehide': 'F', 'tomljanovic': 'F',
    'maia': 'F', 'siniakova': 'F', 'ostapenko': 'F', 'zarazua': 'F', 'bronzetti': 'F',
    'aiava': 'F', 'kalinskaya': 'F', 'hibino': 'F', 'shnaider': 'F', 'fernandez': 'F',
    'golubic': 'F', 'townsend': 'F', 'anisimova': 'F', 'rus': 'F'
}

def add_server_gender_column():
    """Add server_gender column based on player names."""
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Create server_gender column
    print("Adding server_gender column...")
    df['server_gender'] = df['server_name'].map(PLAYER_GENDER_MAP)
    
    print(f"Updated CSV shape: {df.shape}")
    print(f"Updated columns: {len(df.columns)}")
    
    # Show some examples
    print(f"\nSample gender mappings:")
    sample_data = df[['server_name', 'server_gender']].head(15)
    for idx, row in sample_data.iterrows():
        print(f"  {row['server_name']} -> {row['server_gender']}")
    
    # Check for unmapped players
    unmapped = df[df['server_gender'].isna()]['server_name'].unique()
    if len(unmapped) > 0:
        print(f"\n⚠️  Unmapped players ({len(unmapped)}):")
        for player in sorted(unmapped):
            print(f"  - {player}")
    
    # Show gender distribution
    print(f"\n=== GENDER DISTRIBUTION ===")
    gender_counts = df['server_gender'].value_counts()
    total_serves = len(df)
    
    print(f"Total serves: {total_serves}")
    for gender, count in gender_counts.items():
        percentage = (count / total_serves) * 100
        print(f"  {gender}: {count} serves ({percentage:.1f}%)")
    
    # Show top players by gender
    print(f"\n=== TOP PLAYERS BY GENDER ===")
    for gender in ['M', 'F']:
        gender_df = df[df['server_gender'] == gender]
        top_players = gender_df['server_name'].value_counts().head(5)
        print(f"\nTop {gender} players:")
        for player, count in top_players.items():
            print(f"  {player}: {count} serves")
    
    # Save updated CSV
    print(f"\nSaving updated CSV to {OUTPUT_CSV_PATH}")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    return df

def main():
    """Main function to add server gender column."""
    print("Starting server gender column addition...")
    
    # Add server gender column
    df_updated = add_server_gender_column()
    
    print(f"\n✅ Server gender column addition complete!")
    print(f"Output file: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main() 