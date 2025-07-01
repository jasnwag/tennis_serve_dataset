import argparse
import os
from mmpose.apis import MMPoseInferencer
import torch
from tqdm import tqdm

def run_inferencer(input_file, output_dir=None):
    """
    Runs the mmpose inferencer (alias: human3d) on the specified input file.
    Args:
        input_file (str): Path to the image or video file to be processed.
        output_dir (str, optional): Directory to save results. If None, results are not saved.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Initialize the inferencer with RTMO model (fast and accurate)
    inferencer = MMPoseInferencer(pose3d='human3d', device=device)
    
    # Process the video
    # The inferencer will automatically handle video processing and saving
    for _ in tqdm(inferencer(
        input_file,
        out_dir=output_dir,
    ), desc='Processing video'):
        pass
    
    print(f'Video processing complete. Output saved to {output_dir}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mmpose human3d inferencer on an input file.")
    parser.add_argument('--input', required=True, help='Path to the input image or video file')
    parser.add_argument('--output', required=False, help='Directory to save results (optional)')
    args = parser.parse_args()
    run_inferencer(args.input, args.output)
