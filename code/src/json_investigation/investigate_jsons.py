import os
import json
from glob import glob
from collections import Counter
import matplotlib.pyplot as plt

# Directory containing the JSON files
JSON_DIR = os.path.expanduser("~/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/full/all_jsons")

# Sample a few files
json_files = sorted(glob(os.path.join(JSON_DIR, '*.json')))
print(f"Found {len(json_files)} JSON files.")

# How many files to sample
N_SAMPLE = 3

serve_lengths = []

for i, json_path in enumerate(json_files):
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
        serve_lengths.append(len(data))

print(f"\n--- Serve Length Statistics ---")
print(f"Min frames: {min(serve_lengths)}")
print(f"Max frames: {max(serve_lengths)}")
print(f"Mean frames: {sum(serve_lengths)/len(serve_lengths):.2f}")

# Print a histogram (text)
dist = Counter(serve_lengths)
print("\nFrame count distribution (frame_count: num_files):")
for k in sorted(dist):
    print(f"  {k}: {dist[k]}")

# Plot histogram
plt.figure(figsize=(10,5))
plt.hist(serve_lengths, bins=range(min(serve_lengths), max(serve_lengths)+2), edgecolor='black')
plt.xlabel('Number of Frames per Serve')
plt.ylabel('Number of Files')
plt.title('Distribution of Serve Lengths (Frames)')
plt.tight_layout()
plt.show()

for i, json_path in enumerate(json_files[:N_SAMPLE]):
    print(f"\n--- Sample {i+1}: {os.path.basename(json_path)} ---")
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
        # Print type and keys/structure
        print(f"Type: {type(data)}")
        if isinstance(data, list) and len(data) > 0:
            print(f"First element type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"Keys in first element: {list(data[0].keys())}")
                # Print a preview of the first element
                for k, v in data[0].items():
                    if isinstance(v, (list, dict)):
                        print(f"  {k}: <{type(v).__name__}> (length: {len(v) if hasattr(v, '__len__') else 'n/a'})")
                    else:
                        print(f"  {k}: {v}")
        else:
            print(f"Data preview: {str(data)[:500]}") 