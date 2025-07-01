import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the folder containing images and predictions
scorebug_dir = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_test_onethirdres"

# List all files in the directory
files = os.listdir(scorebug_dir)

# Get all image and json pairs
image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
json_files = [f for f in files if f.lower().endswith('.json')]

def match_json(image_file):
    # Replace '｜' with '_' and extension with .json to match naming
    base = os.path.splitext(image_file)[0]
    base = base.replace('｜', '_').replace('  ', ' ').replace(' .', '.').replace('  ', ' ')
    # Find the closest matching JSON file
    for jf in json_files:
        if base in jf:
            return jf
    return None

import math

# Only show up to 10 results
num_to_show = min(10, len(image_files))
cols = 5
rows = math.ceil(num_to_show / cols)

fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
axes = axes.flatten() if num_to_show > 1 else [axes]

for idx, img_file in enumerate(image_files[:num_to_show]):
    json_file = match_json(img_file)
    ax = axes[idx]
    if json_file is None:
        ax.axis('off')
        continue

    img_path = os.path.join(scorebug_dir, img_file)
    json_path = os.path.join(scorebug_dir, json_file)

    img = mpimg.imread(img_path)
    with open(json_path, 'r') as f:
        prediction = json.load(f)

    ax.imshow(img)
    ax.axis('off')
    # Show image filename and summary of prediction as the title
    # Title: just filename, small font
    ax.set_title(img_file, fontsize=8)
    # JSON summary below image
    summary = (
        f"set: {prediction.get('set_number')}\n"
        f"games: {prediction.get('current_set',{}).get('games')}\n"
        f"points: {prediction.get('current_game',{})}"
    )
    ax.text(0.5, -0.15, summary, fontsize=8, ha='center', va='top', transform=ax.transAxes, wrap=True)

# Hide any unused axes
for j in range(idx+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


