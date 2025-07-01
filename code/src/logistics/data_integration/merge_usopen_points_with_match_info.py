import pandas as pd

# File paths
matches_path = "2024-usopen-matches.csv"
points_path = "2024-usopen-points.csv"
output_path = "2024-usopen-points-with-match-info.csv"

# Load data
matches = pd.read_csv(matches_path)
points = pd.read_csv(points_path)

# Merge: left join points with matches on match_id
merged = points.merge(matches, on="match_id", how="left", suffixes=("", "_match"))

# Save merged file
merged.to_csv(output_path, index=False)

print(f"Merged file saved as {output_path}")
