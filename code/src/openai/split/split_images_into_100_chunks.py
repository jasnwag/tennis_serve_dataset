import os
import shutil

# Gather all images from the 1000-chunk folders
chunks_root = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_onethirdres_chunks"
out_root = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_onethirdres_chunks_100"
os.makedirs(out_root, exist_ok=True)

# Find all images in all chunk folders
all_images = []
for chunk_dir in sorted(os.listdir(chunks_root)):
    chunk_path = os.path.join(chunks_root, chunk_dir)
    if not os.path.isdir(chunk_path):
        continue
    imgs = [os.path.join(chunk_path, f) for f in os.listdir(chunk_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    all_images.extend(imgs)

all_images.sort()  # for reproducibility

chunk_size = 100
num_chunks = (len(all_images) + chunk_size - 1) // chunk_size

for i in range(num_chunks):
    chunk_dir = os.path.join(out_root, f"chunk_{i+1:03d}")
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_imgs = all_images[i*chunk_size:(i+1)*chunk_size]
    for img_path in chunk_imgs:
        fname = os.path.basename(img_path)
        dest = os.path.join(chunk_dir, fname)
        shutil.copy2(img_path, dest)
    print(f"Created {chunk_dir} with {len(chunk_imgs)} images.")

print(f"All images split into {num_chunks} folders of up to 100 images each in {out_root}")
