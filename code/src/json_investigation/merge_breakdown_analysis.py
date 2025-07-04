#!/usr/bin/env python3
"""
Detailed breakdown of merge results showing duplicates vs no matches.
"""

import pandas as pd
import json

def analyze_merge_breakdown():
    """Analyze the detailed breakdown of merge results."""
    print("🎾 DETAILED MERGE BREAKDOWN ANALYSIS")
    print("=" * 60)
    
    # Load the actual merge results from the script output
    total_scorebug = 10663
    unique_scorebug = 7925
    no_match = 1378
    successful_matches = 6694
    
    # Calculate duplicates
    duplicates = total_scorebug - unique_scorebug
    
    print("📊 MERGE RESULTS BREAKDOWN:")
    print(f"  Total OpenAI results: {total_scorebug:,}")
    print(f"  Unique combinations: {unique_scorebug:,}")
    print(f"  Duplicates removed: {duplicates:,}")
    print(f"  No matches found: {no_match:,}")
    print(f"  Successfully matched: {successful_matches:,}")
    
    print(f"\n📈 PERCENTAGES:")
    print(f"  Successfully matched: {successful_matches/total_scorebug*100:.1f}%")
    print(f"  No matches found: {no_match/total_scorebug*100:.1f}%")
    print(f"  Duplicates removed: {duplicates/total_scorebug*100:.1f}%")
    
    print(f"\n🔍 DETAILED ANALYSIS:")
    print(f"  Of {total_scorebug:,} total OpenAI results:")
    print(f"    - {duplicates:,} ({duplicates/total_scorebug*100:.1f}%) were duplicates and removed")
    print(f"    - {unique_scorebug:,} ({unique_scorebug/total_scorebug*100:.1f}%) were unique combinations")
    print(f"    - Of the unique combinations:")
    print(f"      * {successful_matches:,} ({successful_matches/unique_scorebug*100:.1f}%) successfully matched")
    print(f"      * {no_match:,} ({no_match/unique_scorebug*100:.1f}%) had no match in US Open data")
    
    return {
        'total': total_scorebug,
        'unique': unique_scorebug,
        'duplicates': duplicates,
        'no_match': no_match,
        'successful': successful_matches
    }

def analyze_duplicate_patterns():
    """Analyze patterns in the duplicates."""
    print(f"\n🔄 DUPLICATE ANALYSIS")
    print("=" * 60)
    
    # Load the scorebug data to analyze duplicates
    scorebug_path = "../../../data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
    
    try:
        # Parse scorebug data
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
                    
                    # Extract nested fields
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
                        "video_name": obj.get("custom_id")
                    }
                    scorebug_records.append(record)
        
        scorebug_df = pd.DataFrame(scorebug_records)
        
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
        
        scorebug_df['player1'] = scorebug_df['player1'].apply(get_lastname)
        scorebug_df['player2'] = scorebug_df['player2'].apply(get_lastname)
        
        # Find duplicates
        group_cols = ["player1", "player2", "SetNo", "P1GamesWon", "P2GamesWon", "P1Score", "P2Score"]
        dupes = scorebug_df.duplicated(subset=group_cols, keep=False)
        duplicate_df = scorebug_df[dupes]
        
        print(f"Total duplicates found: {len(duplicate_df)}")
        
        # Analyze duplicate patterns
        duplicate_counts = duplicate_df.groupby(group_cols).size().reset_index(name='count')
        duplicate_counts = duplicate_counts.sort_values('count', ascending=False)
        
        print(f"\nMost common duplicate combinations:")
        for idx, row in duplicate_counts.head(10).iterrows():
            print(f"  {row['player1']} vs {row['player2']}, Set {row['SetNo']}, "
                  f"Games {row['P1GamesWon']}-{row['P2GamesWon']}, "
                  f"Score {row['P1Score']}-{row['P2Score']}: {row['count']} times")
        
        # Analyze by player
        print(f"\nPlayers with most duplicates:")
        player_duplicates = duplicate_df.groupby('player1')['video_name'].count().sort_values(ascending=False)
        for player, count in player_duplicates.head(10).items():
            print(f"  {player}: {count} duplicate instances")
        
        return duplicate_df
        
    except FileNotFoundError:
        print(f"❌ Scorebug file not found: {scorebug_path}")
        return None
    except Exception as e:
        print(f"❌ Error analyzing duplicates: {e}")
        return None

def analyze_no_match_patterns():
    """Analyze patterns in the no-match cases."""
    print(f"\n❌ NO MATCH ANALYSIS")
    print("=" * 60)
    
    # Load US Open data for comparison
    usopen_path = "../../../data/scorebug/us_open_data/us_open.csv"
    
    try:
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
                        "video_name": obj.get("custom_id")
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
        
        print(f"Total no-match cases: {len(no_match_df)}")
        
        # Analyze no-match patterns
        print(f"\nPlayers with most no-match cases:")
        player_no_matches = no_match_df.groupby('player1')['video_name'].count().sort_values(ascending=False)
        for player, count in player_no_matches.head(10).items():
            print(f"  {player}: {count} no-match instances")
        
        # Analyze by set number
        print(f"\nNo-match cases by set number:")
        set_no_matches = no_match_df.groupby('SetNo')['video_name'].count().sort_values(ascending=False)
        for set_no, count in set_no_matches.items():
            print(f"  Set {set_no}: {count} no-match instances")
        
        return no_match_df
        
    except FileNotFoundError:
        print(f"❌ Data file not found")
        return None
    except Exception as e:
        print(f"❌ Error analyzing no-match patterns: {e}")
        return None

def main():
    """Main analysis function."""
    print("🎾 COMPREHENSIVE MERGE BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    # Get basic breakdown
    breakdown = analyze_merge_breakdown()
    
    # Analyze duplicates
    duplicate_df = analyze_duplicate_patterns()
    
    # Analyze no-match patterns
    no_match_df = analyze_no_match_patterns()
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("📋 FINAL SUMMARY")
    print("=" * 80)
    
    print(f"Merge Results Summary:")
    print(f"  📊 Total OpenAI results: {breakdown['total']:,}")
    print(f"  ✅ Successfully matched: {breakdown['successful']:,} ({breakdown['successful']/breakdown['total']*100:.1f}%)")
    print(f"  ❌ No matches found: {breakdown['no_match']:,} ({breakdown['no_match']/breakdown['total']*100:.1f}%)")
    print(f"  🔄 Duplicates removed: {breakdown['duplicates']:,} ({breakdown['duplicates']/breakdown['total']*100:.1f}%)")
    
    print(f"\nKey Insights:")
    print(f"  - {breakdown['duplicates']:,} instances were duplicates of the same scoreboard state")
    print(f"  - {breakdown['no_match']:,} unique combinations couldn't be found in US Open data")
    print(f"  - {breakdown['successful']:,} unique combinations successfully matched")
    print(f"  - Overall success rate: {breakdown['successful']/breakdown['total']*100:.1f}%")

if __name__ == "__main__":
    main() 