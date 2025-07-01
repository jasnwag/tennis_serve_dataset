# Joint Angle Analysis for Tennis Coaching

This directory contains scripts for analyzing joint angles from MMPose predictions for tennis coaching purposes.

## Scripts Overview

1. `calculate_angles.py` - Core module for calculating joint angles from MMPose JSON output
2. `process_angles.py` - Example script for processing a single JSON file
3. `batch_process_angles.py` - Script for processing multiple JSON files in a directory
4. `visualize_angles.py` - Script for visualizing joint angles on frames and creating videos

## Installation

These scripts require the following Python packages:

```bash
pip install numpy pandas matplotlib opencv-python tqdm
```

## Usage

### 1. Process a Single JSON File

```bash
python calculate_angles.py /path/to/mmpose_output.json --output /path/to/output.csv --plot /path/to/plot/directory
```

### 2. Process a Single File Using the Example Script

```bash
python process_angles.py
```

This will process the example file and save the results to the default locations.

### 3. Process Multiple JSON Files

```bash
python batch_process_angles.py /path/to/json/directory --output_dir /path/to/output/directory --plots_dir /path/to/plots/directory --parallel 4
```

### 4. Visualize Angles

```bash
python visualize_angles.py /path/to/mmpose_output.json /path/to/angles.csv --output /path/to/output/directory --video --frames
```

Options:
- `--video` creates a video visualization
- `--frames` creates individual frame visualizations
- `--fps 30` sets the output video frame rate
- `--start_frame 0` sets the first frame to visualize
- `--end_frame 100` sets the last frame to visualize
- `--step 5` sets the step size for individual frame output (every 5th frame)

## Joint Angle Definitions

The scripts calculate the following joint angles:

- `left_elbow`: Angle between left shoulder, elbow, and wrist
- `right_elbow`: Angle between right shoulder, elbow, and wrist
- `left_shoulder`: Angle between left elbow, shoulder, and hip
- `right_shoulder`: Angle between right elbow, shoulder, and hip
- `left_hip`: Angle between left shoulder, hip, and knee
- `right_hip`: Angle between right shoulder, hip, and knee
- `left_knee`: Angle between left hip, knee, and ankle
- `right_knee`: Angle between right hip, knee, and ankle

## Keypoint Indices (COCO 17-keypoint format)

- 0: nose
- 1: left_eye
- 2: right_eye
- 3: left_ear
- 4: right_ear
- 5: left_shoulder
- 6: right_shoulder
- 7: left_elbow
- 8: right_elbow
- 9: left_wrist
- 10: right_wrist
- 11: left_hip
- 12: right_hip
- 13: left_knee
- 14: right_knee
- 15: left_ankle
- 16: right_ankle

## Example Workflow

1. Generate pose predictions using MMPose and save to JSON
2. Process the JSON file(s) to calculate joint angles:
   ```bash
   python batch_process_angles.py /path/to/predictions --output_dir /path/to/angles
   ```
3. Visualize the results:
   ```bash
   python visualize_angles.py /path/to/predictions/video.json /path/to/angles/video_angles.csv --output /path/to/visualizations --video --frames
   ```
4. Analyze the joint angle data in the CSV files for coaching insights 