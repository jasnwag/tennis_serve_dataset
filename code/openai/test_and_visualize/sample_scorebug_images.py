import os
import random
import shutil

src = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs'
dst = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs_test'
os.makedirs(dst, exist_ok=True)

imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
for f in random.sample(imgs, min(30, len(imgs))):
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
print(f"Copied {min(30, len(imgs))} images to {dst}") 