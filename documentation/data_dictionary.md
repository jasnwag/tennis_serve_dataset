# Data Dictionary

This document provides detailed descriptions of all columns in the tennis serve analysis dataset.

## Core Identification Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `video_name` | String | Original video filename | "Jannik Sinner vs. Mackenzie McDonald Full Match ｜ 2024 US Open Round 1.f617_30.1_1975.jpg" |
| `json_file_path` | String | Path to corresponding JSON keypoint file | "data/full/all_jsons/..." |
| `json_file_found` | Boolean | Whether JSON file was successfully matched | True |

## Player Information

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `player1` | String | First player in the match | "sinner" |
| `player2` | String | Second player in the match | "mcdonald" |
| `server_name` | String | Name of the serving player | "sinner" |
| `server_gender` | String | Gender of the server (M/F) | "M" |
| `PointServer` | Integer | Server identifier (1 or 2) | 1 |

## Match Context

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `tournament` | String | Tournament name | "US Open" |
| `round` | String | Match round | "Round 1" |
| `court` | String | Court information | "Arthur Ashe Stadium" |
| `date` | String | Match date | "2024-08-26" |

## Serve Analysis Data

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `n_frames` | Integer | Number of frames in serve sequence | 91 |
| `keypoints_clean` | JSON Array | 3D keypoint coordinates (n_frames × 17 × 3) | `[[[x,y,z], ...], ...]` |
| `keypoint_scores_clean` | JSON Array | Confidence scores (n_frames × 17) | `[[0.95, 0.87, ...], ...]` |

## Point Details

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `PointNumber` | Integer | Point number in the match | 1 |
| `GameScore` | String | Game score when point occurred | "0-0" |
| `SetScore` | String | Set score when point occurred | "0-0" |
| `MatchScore` | String | Match score when point occurred | "0-0" |

## Serve Outcome

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `PointWinner` | Integer | Winner of the point (1 or 2) | 1 |
| `PointServer` | Integer | Server for the point (1 or 2) | 1 |
| `PointType` | String | Type of point outcome | "Ace" |

## Technical Metadata

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `frame_start` | Integer | Starting frame number | 1975 |
| `frame_end` | Integer | Ending frame number | 2066 |
| `video_fps` | Float | Video frames per second | 30.0 |

## Data Quality Indicators

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `keypoint_quality` | Float | Average confidence score | 0.85 |
| `frame_completeness` | Float | Percentage of frames with valid keypoints | 0.98 |

## Notes

- **Coordinate System**: All 3D coordinates are normalized to the video frame
- **Confidence Scores**: Range from 0.0 (low confidence) to 1.0 (high confidence)
- **Frame Counts**: Vary from 60-120 frames per serve
- **Player Names**: All lowercase, no special characters
- **Gender Mapping**: Based on known player information from tennis databases

## Missing Data

- Some columns may contain NaN values for serves where data was unavailable
- Empty columns have been removed during data cleaning
- Unmatched JSON files are marked with `json_file_found = False` 