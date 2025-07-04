#!/usr/bin/env python3
"""
Analyze the merge between scoreboard data and OpenAI results.
"""

import pandas as pd
import json
import numpy as np
from collections import Counter

def analyze_merge_results():
    """Analyze the results of the scoreboard/OpenAI merge."""
    print("🎾 SCOREBOARD/OPENAI MERGE ANALYSIS")
    print("=" * 60)
    
    # Load merge summary
    summary_path = "../../src/logistics/data_integration/annotation_merge_summary.csv"
    summary_df = pd.read_csv(summary_path)
    
    print("📊 MERGE SUMMARY:")
    for _, row in summary_df.iterrows():
        print(f"  {row['type']}: {row['count']:,}")
    
    total_combined = summary_df[summary_df['type'] == 'total_combined_renamed']['count'].iloc[0]
    matched = summary_df[summary_df['type'] == 'matched']['count'].iloc[0]
    no_match = summary_df[summary_df['type'] == 'no_match']['count'].iloc[0]
    too_many = summary_df[summary_df['type'] == 'too_many_matches']['count'].iloc[0]
    
    print(f"\n📈 MERGE STATISTICS:")
    print(f"  Total OpenAI results: {total_combined:,}")
    print(f"  Successfully matched: {matched:,} ({matched/total_combined*100:.1f}%)")
    print(f"  No matches found: {no_match:,} ({no_match/total_combined*100:.1f}%)")
    print(f"  Too many matches: {too_many:,} ({too_many/total_combined*100:.1f}%)")
    
    return summary_df

def analyze_unmatched_reasons():
    """Analyze why matches failed."""
    print(f"\n🔍 UNMATCHED REASONS ANALYSIS")
    print("=" * 60)
    
    reasons_path = "../../src/logistics/data_integration/unmatched_annotation_merge_reasons.csv"
    reasons_df = pd.read_csv(reasons_path)
    
    print(f"Total unmatched rows: {len(reasons_df)}")
    
    # Analyze which fields are most problematic
    fields = ['player1', 'player2', 'SetNo', 'P1GamesWon', 'P2GamesWon', 'P1Score', 'P2Score']
    
    print(f"\nField matching analysis:")
    for field in fields:
        no_match_count = (reasons_df[field] == 'NO MATCH').sum()
        match_count = (reasons_df[field] == 'MATCH').sum()
        total = len(reasons_df)
        
        print(f"  {field}:")
        print(f"    Matches: {match_count:,} ({match_count/total*100:.1f}%)")
        print(f"    No matches: {no_match_count:,} ({no_match_count/total*100:.1f}%)")
    
    # Show examples of different failure types
    print(f"\n📋 EXAMPLES OF UNMATCHED ROWS:")
    
    # Rows where all fields failed to match
    all_no_match = reasons_df[(reasons_df[fields] == 'NO MATCH').all(axis=1)]
    if len(all_no_match) > 0:
        print(f"  All fields failed to match ({len(all_no_match)} rows):")
        for idx, row in all_no_match.head(3).iterrows():
            print(f"    {row['video_name']}")
    
    # Rows where only some fields failed
    partial_failures = reasons_df[~(reasons_df[fields] == 'NO MATCH').all(axis=1)]
    if len(partial_failures) > 0:
        print(f"  Partial failures ({len(partial_failures)} rows):")
        for idx, row in partial_failures.head(3).iterrows():
            failed_fields = [field for field in fields if row[field] == 'NO MATCH']
            print(f"    {row['video_name']} - Failed fields: {failed_fields}")
    
    return reasons_df

def analyze_final_merged_data():
    """Analyze the final merged dataset."""
    print(f"\n📁 FINAL MERGED DATA ANALYSIS")
    print("=" * 60)
    
    merged_path = "../../../data/scorebug/usopen_points_with_scorebug.csv"
    
    try:
        merged_df = pd.read_csv(merged_path)
        print(f"Final merged dataset shape: {merged_df.shape}")
        print(f"Columns: {len(merged_df.columns)}")
        
        # Check for video_name column
        if 'video_name' in merged_df.columns:
            unique_videos = merged_df['video_name'].nunique()
            print(f"Unique videos: {unique_videos}")
            
            # Sample video names
            print(f"Sample video names:")
            for video in merged_df['video_name'].unique()[:5]:
                print(f"  {video}")
        
        # Check for key columns
        key_columns = ['player1', 'player2', 'SetNo', 'P1Score', 'P2Score']
        missing_columns = [col for col in key_columns if col not in merged_df.columns]
        
        if missing_columns:
            print(f"Missing key columns: {missing_columns}")
        else:
            print(f"All key columns present")
        
        return merged_df
        
    except FileNotFoundError:
        print(f"❌ Merged file not found: {merged_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading merged file: {e}")
        return None

def analyze_openai_results_structure():
    """Analyze the structure of OpenAI results."""
    print(f"\n🤖 OPENAI RESULTS STRUCTURE ANALYSIS")
    print("=" * 60)
    
    scorebug_path = "../../../data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
    
    try:
        # Load a sample of OpenAI results
        sample_records = []
        with open(scorebug_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Only read first 10 lines
                    break
                obj = json.loads(line)
                sample_records.append(obj)
        
        print(f"Sample OpenAI results structure:")
        for i, record in enumerate(sample_records):
            print(f"  Record {i+1}:")
            if "response" in record and "body" in record["response"]:
                choices = record["response"]["body"]["choices"]
                if choices:
                    content = choices[0]["message"]["content"]
                    try:
                        content_dict = json.loads(content)
                        print(f"    Keys: {list(content_dict.keys())}")
                        print(f"    Video: {record.get('custom_id', 'N/A')}")
                    except:
                        print(f"    Content not valid JSON")
            else:
                print(f"    No valid response structure")
        
        return sample_records
        
    except FileNotFoundError:
        print(f"❌ OpenAI results file not found: {scorebug_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading OpenAI results: {e}")
        return None

def suggest_improvements():
    """Suggest improvements for the merge process."""
    print(f"\n💡 SUGGESTED IMPROVEMENTS")
    print("=" * 60)
    
    print("1. **Fuzzy Matching**:")
    print("   - Use fuzzy string matching for player names")
    print("   - Handle variations in name formats")
    
    print("2. **Score Normalization**:")
    print("   - Standardize score formats (15, 30, 40 vs 1, 2, 3)")
    print("   - Handle deuce and advantage scoring")
    
    print("3. **Set Number Handling**:")
    print("   - Normalize set number formats")
    print("   - Handle tiebreaks and special set cases")
    
    print("4. **Video Name Matching**:")
    print("   - Use video names as additional matching criteria")
    print("   - Extract player names from video filenames")
    
    print("5. **Data Quality Checks**:")
    print("   - Validate OpenAI results before merging")
    print("   - Add confidence scores for matches")
    
    print("6. **Incremental Matching**:")
    print("   - Start with exact matches")
    print("   - Then try partial matches")
    print("   - Finally use fuzzy matching for remaining cases")

def main():
    """Main analysis function."""
    print("🎾 COMPREHENSIVE SCOREBOARD/OPENAI MERGE ANALYSIS")
    print("=" * 80)
    
    # Analyze merge results
    summary_df = analyze_merge_results()
    
    # Analyze unmatched reasons
    reasons_df = analyze_unmatched_reasons()
    
    # Analyze final merged data
    merged_df = analyze_final_merged_data()
    
    # Analyze OpenAI results structure
    openai_samples = analyze_openai_results_structure()
    
    # Suggest improvements
    suggest_improvements()
    
    # Summary
    print(f"\n" + "=" * 80)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 80)
    
    total_combined = summary_df[summary_df['type'] == 'total_combined_renamed']['count'].iloc[0]
    matched = summary_df[summary_df['type'] == 'matched']['count'].iloc[0]
    
    print(f"Merge Success Rate: {matched/total_combined*100:.1f}%")
    print(f"Total OpenAI Results: {total_combined:,}")
    print(f"Successfully Matched: {matched:,}")
    print(f"Failed to Match: {total_combined - matched:,}")
    
    if merged_df is not None:
        print(f"Final Dataset Size: {len(merged_df):,} rows")
    
    print(f"\nKey Issues Identified:")
    print(f"  - Low match rate suggests data format inconsistencies")
    print(f"  - Player name variations need normalization")
    print(f"  - Score format differences need standardization")
    print(f"  - Video name matching could improve results")

if __name__ == "__main__":
    main() 