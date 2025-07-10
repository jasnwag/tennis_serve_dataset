import os
from PIL import Image

src_dir = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_test"
dst_dir = os.path.join(os.path.dirname(src_dir), "positive_scorebugs_test_onethirdres")
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

print(f"All images downscaled to one-third resolution and saved to {dst_dir}")
