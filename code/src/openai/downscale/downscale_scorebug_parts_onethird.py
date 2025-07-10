import os
from PIL import Image

folders = [
    "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_part1",
    "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_part2"
]

for src_dir in folders:
    dst_dir = src_dir + "_onethirdres"
    os.makedirs(dst_dir, exist_ok=True)
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for fname in image_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        with Image.open(src_path) as img:
            new_size = (max(1, int(img.width / 3)), max(1, int(img.height / 3)))
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_resized.save(dst_path)
        print(f"Saved {dst_path} ({new_size[0]}x{new_size[1]})")
    print(f"Finished downscaling {src_dir} to {dst_dir}")
