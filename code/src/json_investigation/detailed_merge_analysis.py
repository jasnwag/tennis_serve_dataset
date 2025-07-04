#!/usr/bin/env python3
"""
Detailed analysis of the scoreboard/OpenAI merge results.
"""

import pandas as pd
import json
from collections import Counter

def analyze_merge_breakdown():
    """Get detailed breakdown of merge results."""
    print("🎾 DETAILED MERGE BREAKDOWN ANALYSIS")
    print("=" * 60)
    
    # Load merge summary
    summary_path = "../../src/logistics/data_integration/annotation_merge_summary.csv"
    summary_df = pd.read_csv(summary_path)
    
    print("📊 MERGE SUMMARY FROM FILE:")
    for _, row in summary_df.iterrows():
        print(f"  {row['type']}: {row['count']:,}")
    
    # Calculate totals
    total_combined = summary_df[summary_df['type'] == 'total_combined_renamed']['count'].iloc[0]
    matched = summary_df[summary_df['type'] == 'matched']['count'].iloc[0]
    no_match = summary_df[summary_df['type'] == 'no_match']['count'].iloc[0]
    too_many = summary_df[summary_df['type'] == 'too_many_matches']['count'].iloc[0]
    
    print(f"\n📈 CALCULATED BREAKDOWN:")
    print(f"  Total OpenAI results: {total_combined:,}")
    print(f"  Successfully matched: {matched:,} ({matched/total_combined*100:.1f}%)")
    print(f"  No matches found: {no_match:,} ({no_match/total_combined*100:.1f}%)")
    print(f"  Too many matches (duplicates): {too_many:,} ({too_many/total_combined*100:.1f}%)")
    
    # Verify the math
    calculated_total = matched + no_match + too_many
    print(f"\n🔢 VERIFICATION:")
    print(f"  Sum of categories: {calculated_total:,}")
    print(f"  Original total: {total_combined:,}")
    print(f"  Match: {'✅' if calculated_total == total_combined else '❌'}")
    
    return {
        'total': total_combined,
        'matched': matched,
        'no_match': no_match,
        'too_many': too_many
    }

def analyze_unmatched_reasons():
    """Analyze the unmatched reasons file."""
    print(f"\n🔍 UNMATCHED REASONS ANALYSIS")
    print("=" * 60)
    
    reasons_path = "../../src/logistics/data_integration/unmatched_annotation_merge_reasons.csv"
    reasons_df = pd.read_csv(reasons_path)
    
    print(f"Total rows in unmatched reasons file: {len(reasons_df):,}")
    
    # Analyze field matching patterns
    fields = ['player1', 'player2', 'SetNo', 'P1GamesWon', 'P2GamesWon', 'P1Score', 'P2Score']
    
    print(f"\nField matching patterns:")
    for field in fields:
        no_match_count = (reasons_df[field] == 'NO MATCH').sum()
        match_count = (reasons_df[field] == 'MATCH').sum()
        total = len(reasons_df)
        
        print(f"  {field}:")
        print(f"    Matches: {match_count:,} ({match_count/total*100:.1f}%)")
        print(f"    No matches: {no_match_count:,} ({no_match_count/total*100:.1f}%)")
    
    # Count rows with different numbers of failed fields
    print(f"\nFailure pattern analysis:")
    for field in fields:
        failed_count = (reasons_df[field] == 'NO MATCH').sum()
        print(f"  Rows where {field} failed: {failed_count:,}")
    
    # Find rows where all fields failed
    all_failed = (reasons_df[fields] == 'NO MATCH').all(axis=1).sum()
    print(f"  Rows where ALL fields failed: {all_failed:,}")
    
    # Find rows where some fields failed
    some_failed = ((reasons_df[fields] == 'NO MATCH').any(axis=1) & 
                   ~(reasons_df[fields] == 'NO MATCH').all(axis=1)).sum()
    print(f"  Rows where SOME fields failed: {some_failed:,}")
    
    return reasons_df

def check_final_merged_file():
    """Check what's actually in the final merged file."""
    print(f"\n📁 FINAL MERGED FILE ANALYSIS")
    print("=" * 60)
    
    merged_path = "../../../data/scorebug/usopen_points_with_scorebug.csv"
    
    try:
        merged_df = pd.read_csv(merged_path)
        print(f"Final merged file shape: {merged_df.shape}")
        print(f"Rows: {len(merged_df):,}")
        print(f"Columns: {len(merged_df.columns)}")
        
        # Check if this is just the original US Open data
        if 'video_name' in merged_df.columns:
            unique_videos = merged_df['video_name'].nunique()
            print(f"Unique videos: {unique_videos:,}")
            
            # Check if these are the same as the original data
            original_path = "../../../data/scorebug/us_open_data/us_open.csv"
            try:
                original_df = pd.read_csv(original_path)
                print(f"Original US Open data shape: {original_df.shape}")
                print(f"Original rows: {len(original_df):,}")
                
                if len(merged_df) == len(original_df):
                    print("⚠️  Final merged file appears to be identical to original US Open data")
                    print("   This suggests the merge with OpenAI results failed completely")
                else:
                    print(f"✅ Final file has different number of rows than original")
                    
            except FileNotFoundError:
                print("❌ Original US Open data file not found for comparison")
        
        return merged_df
        
    except FileNotFoundError:
        print(f"❌ Merged file not found: {merged_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading merged file: {e}")
        return None

def analyze_openai_data_sample():
    """Analyze a sample of the actual OpenAI data."""
    print(f"\n🤖 OPENAI DATA SAMPLE ANALYSIS")
    print("=" * 60)
    
    scorebug_path = "../../../data/scorebug/positive_scorebug_chunks/results/combined_renamed.jsonl"
    
    try:
        # Count total lines
        with open(scorebug_path, "r") as f:
            total_lines = sum(1 for line in f)
        
        print(f"Total OpenAI results: {total_lines:,}")
        
        # Analyze structure of first few records
        sample_records = []
        with open(scorebug_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 5:  # Only read first 5 lines
                    break
                obj = json.loads(line)
                sample_records.append(obj)
        
        print(f"\nSample OpenAI result structure:")
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
                        
                        # Show actual values for key fields
                        for key in ['player1_name', 'player2_name', 'set_number']:
                            if key in content_dict:
                                print(f"    {key}: {content_dict[key]}")
                        
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

def main():
    """Main analysis function."""
    print("🎾 DETAILED SCOREBOARD/OPENAI MERGE ANALYSIS")
    print("=" * 80)
    
    # Get detailed breakdown
    breakdown = analyze_merge_breakdown()
    
    # Analyze unmatched reasons
    reasons_df = analyze_unmatched_reasons()
    
    # Check final merged file
    merged_df = check_final_merged_file()
    
    # Analyze OpenAI data sample
    openai_samples = analyze_openai_data_sample()
    
    # Final summary
    print(f"\n" + "=" * 80)
    print("📋 FINAL SUMMARY")
    print("=" * 80)
    
    print(f"Merge Results:")
    print(f"  ✅ Successfully matched: {breakdown['matched']:,} ({breakdown['matched']/breakdown['total']*100:.1f}%)")
    print(f"  ❌ No matches found: {breakdown['no_match']:,} ({breakdown['no_match']/breakdown['total']*100:.1f}%)")
    print(f"  ⚠️  Too many matches (duplicates): {breakdown['too_many']:,} ({breakdown['too_many']/breakdown['total']*100:.1f}%)")
    print(f"  📊 Total OpenAI results: {breakdown['total']:,}")
    
    print(f"\nKey Findings:")
    if breakdown['matched'] == 0:
        print(f"  🚨 CRITICAL: No successful merges - merge process failed completely")
    else:
        print(f"  ✅ Some successful merges: {breakdown['matched']:,}")
    
    print(f"  📈 {breakdown['no_match']:,} results couldn't be matched")
    print(f"  🔄 {breakdown['too_many']:,} results had duplicate matches")
    
    if merged_df is not None and len(merged_df) > 0:
        print(f"  📁 Final dataset has {len(merged_df):,} rows")
        if breakdown['matched'] == 0:
            print(f"  ⚠️  Final dataset appears to be original data only (no OpenAI annotations)")

if __name__ == "__main__":
    main() 