#!/usr/bin/env python3
"""
Clean CSV by removing completely empty columns.
"""

import pandas as pd
import numpy as np

# File paths
CSV_PATH = "data/full/usopen_points_clean_keypoints.csv"
OUTPUT_CSV_PATH = "data/full/usopen_points_clean_keypoints_cleaned.csv"

def clean_empty_columns():
    """Remove columns that are completely empty (all null/NaN values)."""
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Original CSV shape: {df.shape}")
    print(f"Original columns: {len(df.columns)}")
    
    # Find columns that are completely empty
    empty_columns = []
    for col in df.columns:
        # Check if column is completely empty (all null/NaN or empty strings)
        if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
            empty_columns.append(col)
    
    print(f"\nEmpty columns found: {len(empty_columns)}")
    if empty_columns:
        print("Empty columns to be removed:")
        for col in empty_columns:
            print(f"  - {col}")
    
    # Remove empty columns
    df_cleaned = df.drop(columns=empty_columns)
    
    print(f"\nCleaned CSV shape: {df_cleaned.shape}")
    print(f"Remaining columns: {len(df_cleaned.columns)}")
    
    # Save cleaned CSV
    print(f"\nSaving cleaned CSV to {OUTPUT_CSV_PATH}")
    df_cleaned.to_csv(OUTPUT_CSV_PATH, index=False)
    
    return df_cleaned, empty_columns

def show_column_summary(df):
    """Show summary of remaining columns."""
    print(f"\n=== COLUMN SUMMARY ===")
    print(f"Total columns: {len(df.columns)}")
    
    # Show data types and non-null counts
    print(f"\nColumn info:")
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        data_type = df[col].dtype
        print(f"  {col}: {non_null_count}/{len(df)} non-null ({data_type})")

def main():
    """Main function to clean the CSV."""
    print("Starting CSV column cleanup...")
    
    # Clean empty columns
    df_cleaned, empty_columns = clean_empty_columns()
    
    # Show summary
    show_column_summary(df_cleaned)
    
    print(f"\n✅ CSV cleanup complete!")
    print(f"Removed {len(empty_columns)} empty columns")
    print(f"Output file: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main() 