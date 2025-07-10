import os
import shutil

dir1 = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_part1_onethirdres"
dir2 = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_part2_onethirdres"
out_root = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_onethirdres_chunks"

os.makedirs(out_root, exist_ok=True)

# Collect all image files from both directories
image_files = []
for d in [dir1, dir2]:
    image_files += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

image_files.sort()  # sort for reproducibility

chunk_size = 1000
num_chunks = (len(image_files) + chunk_size - 1) // chunk_size

for i in range(num_chunks):
    chunk_dir = os.path.join(out_root, f"chunk_{i+1:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_imgs = image_files[i*chunk_size:(i+1)*chunk_size]
    for img_path in chunk_imgs:
        fname = os.path.basename(img_path)
        dest = os.path.join(chunk_dir, fname)
        shutil.copy2(img_path, dest)
    print(f"Created {chunk_dir} with {len(chunk_imgs)} images.")

print(f"All images split into {num_chunks} folders of up to 1000 images each in {out_root}")
