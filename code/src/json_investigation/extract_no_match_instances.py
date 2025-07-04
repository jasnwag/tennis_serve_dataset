#!/usr/bin/env python3
"""
Extract all no-match instances and organize them into two folders:
1. no_match_images/ - containing the scorebug images
2. no_match_jsons/ - containing the corresponding JSON predictions
"""

import pandas as pd
import json
import os
import shutil
from pathlib import Path
import glob

def get_no_match_instances():
    """Get the list of video names that had no matches."""
    print("🔍 Identifying no-match instances...")
    
    # Load US Open data
    usopen_path = "../../../data/scorebug/us_open_data/us_open.csv"
    usopen_df = pd.read_csv(usopen_path)
    
    # Normalize player names
    def get_lastname(name):
        if pd.isnull(name):
            return None
        s = str(name).strip()
        if not s:
            return None
        parts = s.split()
        if not parts:
            return None
        return parts[-1].lower()
    
    usopen_df['player1'] = usopen_df['player1'].apply(get_lastname)
    usopen_df['player2'] = usopen_df['player2'].apply(get_lastname)
    
    # Load scorebug data
    scorebug_path = "../../../data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
    
    scorebug_records = []
    with open(scorebug_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "response" in obj and "body" in obj["response"] and "choices" in obj["response"]["body"]:
                message_content = obj["response"]["body"]["choices"][0]["message"]["content"]
                try:
                    content_dict = json.loads(message_content)
                except Exception:
                    continue
                
                games = content_dict.get("games") or content_dict.get("current_set", {}).get("games", {})
                current_game = content_dict.get("current_game", {})
                
                record = {
                    "player1": content_dict.get("player1", content_dict.get("player1_name")),
                    "player2": content_dict.get("player2", content_dict.get("player2_name")),
                    "SetNo": content_dict.get("SetNo", content_dict.get("set_number")),
                    "P1GamesWon": games.get("P1GamesWon", games.get("player1")) if games else None,
                    "P2GamesWon": games.get("P2GamesWon", games.get("player2")) if games else None,
                    "P1Score": current_game.get("P1Score", current_game.get("player1_points")),
                    "P2Score": current_game.get("P2Score", current_game.get("player2_points")),
                    "video_name": obj.get("custom_id"),
                    "json_data": obj  # Store the full JSON for later
                }
                scorebug_records.append(record)
    
    scorebug_df = pd.DataFrame(scorebug_records)
    scorebug_df['player1'] = scorebug_df['player1'].apply(get_lastname)
    scorebug_df['player2'] = scorebug_df['player2'].apply(get_lastname)
    
    # Remove duplicates first
    group_cols = ["player1", "player2", "SetNo", "P1GamesWon", "P2GamesWon", "P1Score", "P2Score"]
    dupes = scorebug_df.duplicated(subset=group_cols, keep=False)
    unique_scorebug = scorebug_df[~dupes]
    
    # Find no-match cases
    merged = pd.merge(usopen_df, unique_scorebug, on=group_cols, how="right", indicator=True)
    no_match_df = merged[merged['_merge'] == 'right_only']
    
    print(f"Found {len(no_match_df)} no-match instances")
    
    return no_match_df

def find_image_file(video_name, chunks_dir):
    """Find the image file for a given video name across all chunks."""
    # Search for the image file across all chunk directories
    for chunk_dir in glob.glob(os.path.join(chunks_dir, "chunk_*")):
        image_path = os.path.join(chunk_dir, video_name)
        if os.path.exists(image_path):
            return image_path
    return None

def extract_no_match_instances():
    """Extract all no-match instances to the undergrads folder."""
    print("🎾 EXTRACTING NO-MATCH INSTANCES")
    print("=" * 60)
    
    # Get no-match instances
    no_match_df = get_no_match_instances()
    
    # Create output directories
    undergrads_dir = "../../../data/scorebug/undergrads"
    images_dir = os.path.join(undergrads_dir, "no_match_images")
    jsons_dir = os.path.join(undergrads_dir, "no_match_jsons")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)
    
    print(f"Created directories:")
    print(f"  Images: {images_dir}")
    print(f"  JSONs: {jsons_dir}")
    
    # Process each no-match instance
    chunks_dir = "../../../data/scorebug/positive_scorebug_chunks"
    copied_images = 0
    copied_jsons = 0
    
    for idx, row in no_match_df.iterrows():
        video_name = row['video_name']
        
        # Find and copy image
        image_path = find_image_file(video_name, chunks_dir)
        if image_path:
            dest_image_path = os.path.join(images_dir, video_name)
            shutil.copy2(image_path, dest_image_path)
            copied_images += 1
        
        # Copy JSON prediction
        json_data = row['json_data']
        json_filename = video_name.replace('.jpg', '.json')
        dest_json_path = os.path.join(jsons_dir, json_filename)
        
        with open(dest_json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        copied_jsons += 1
        
        # Progress update
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(no_match_df)} instances...")
    
    print(f"\n✅ EXTRACTION COMPLETE")
    print(f"  Images copied: {copied_images}")
    print(f"  JSONs copied: {copied_jsons}")
    print(f"  Total no-match instances: {len(no_match_df)}")
    
    # Create a summary file
    summary_path = os.path.join(undergrads_dir, "no_match_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("NO-MATCH INSTANCES SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total no-match instances: {len(no_match_df)}\n")
        f.write(f"Images copied: {copied_images}\n")
        f.write(f"JSONs copied: {copied_jsons}\n\n")
        
        f.write("Sample no-match instances:\n")
        for idx, row in no_match_df.head(20).iterrows():
            f.write(f"  {row['video_name']}\n")
            f.write(f"    Players: {row['player1']} vs {row['player2']}\n")
            f.write(f"    Set: {row['SetNo']}, Games: {row['P1GamesWon']}-{row['P2GamesWon']}\n")
            f.write(f"    Score: {row['P1Score']}-{row['P2Score']}\n\n")
    
    print(f"Summary saved to: {summary_path}")
    
    return {
        'total': len(no_match_df),
        'images_copied': copied_images,
        'jsons_copied': copied_jsons
    }

def analyze_no_match_patterns():
    """Analyze patterns in the no-match instances."""
    print(f"\n📊 NO-MATCH PATTERN ANALYSIS")
    print("=" * 60)
    
    no_match_df = get_no_match_instances()
    
    # Analyze by player
    print("Players with most no-match cases:")
    player_counts = no_match_df.groupby('player1')['video_name'].count().sort_values(ascending=False)
    for player, count in player_counts.head(10).items():
        print(f"  {player}: {count}")
    
    # Analyze by set number
    print(f"\nNo-match cases by set number:")
    set_counts = no_match_df.groupby('SetNo')['video_name'].count().sort_values(ascending=False)
    for set_no, count in set_counts.items():
        print(f"  Set {set_no}: {count}")
    
    # Analyze by score patterns
    print(f"\nMost common score patterns in no-matches:")
    score_patterns = no_match_df.groupby(['P1Score', 'P2Score'])['video_name'].count().sort_values(ascending=False)
    for (p1_score, p2_score), count in score_patterns.head(10).items():
        print(f"  {p1_score}-{p2_score}: {count} instances")

def main():
    """Main function."""
    print("🎾 NO-MATCH INSTANCE EXTRACTION")
    print("=" * 80)
    
    # Extract instances
    results = extract_no_match_instances()
    
    # Analyze patterns
    analyze_no_match_patterns()
    
    print(f"\n" + "=" * 80)
    print("📋 FINAL SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully extracted {results['total']} no-match instances")
    print(f"📁 Images saved to: data/scorebug/undergrads/no_match_images/")
    print(f"📄 JSONs saved to: data/scorebug/undergrads/no_match_jsons/")
    print(f"📊 Summary saved to: data/scorebug/undergrads/no_match_summary.txt")

if __name__ == "__main__":
    main() 