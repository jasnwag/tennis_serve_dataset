#!/usr/bin/env python3
"""
Process all MMPose JSON files in a directory to calculate joint angles.
"""

import os
import sys
import glob
import pandas as pd
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from calculate_angles import process_json_file, plot_angles

def process_single_file(json_path, output_dir, plots_dir):
    """Process a single JSON file and save results"""
    try:
        # Get the filename without extension for output naming
        filename = Path(json_path).stem
        output_csv = output_dir / f"{filename}_angles.csv"
        
        print(f"Processing {json_path}...")
        
        # Process the JSON file
        angles_df = process_json_file(json_path)
        
        # Save to CSV
        angles_df.to_csv(output_csv, index=False)
        
        # Generate plots
        if plots_dir:
            plot_dir = plots_dir / filename
            os.makedirs(plot_dir, exist_ok=True)
            plot_angles(angles_df, plot_dir)
        
        return {
            'filename': filename,
            'processed': True,
            'frame_count': len(angles_df),
            'success': True
        }
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")
        return {
            'filename': Path(json_path).stem,
            'processed': False,
            'error': str(e),
            'success': False
        }

def main():
    parser = argparse.ArgumentParser(description='Batch process MMPose JSON files to calculate joint angles')
    parser.add_argument('input_dir', help='Directory containing MMPose JSON files')
    parser.add_argument('--output_dir', '-o', help='Output directory for CSV files', default=None)
    parser.add_argument('--plots_dir', '-p', help='Directory to save angle plots', default=None)
    parser.add_argument('--parallel', '-j', type=int, help='Number of parallel processes', default=1)
    
    args = parser.parse_args()
    
    # Set up directories
    input_dir = Path(args.input_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "angle_analysis"
    
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = output_dir / "plots" if args.plots_dir is not None else None
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    if plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process files
    results = []
    
    if args.parallel > 1:
        # Parallel processing
        print(f"Processing files using {args.parallel} parallel processes")
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = [
                executor.submit(process_single_file, json_path, output_dir, plots_dir) 
                for json_path in json_files
            ]
            for future in futures:
                results.append(future.result())
    else:
        # Sequential processing
        for json_path in json_files:
            result = process_single_file(json_path, output_dir, plots_dir)
            results.append(result)
    
    # Save processing report
    report_df = pd.DataFrame(results)
    report_path = output_dir / "processing_report.csv"
    report_df.to_csv(report_path, index=False)
    
    # Print summary
    success_count = sum(1 for r in results if r['success'])
    print(f"\nProcessing complete: {success_count}/{len(results)} files processed successfully")
    print(f"Results saved to {output_dir}")
    if plots_dir:
        print(f"Plots saved to {plots_dir}")
    print(f"Processing report saved to {report_path}")

if __name__ == "__main__":
    main() 