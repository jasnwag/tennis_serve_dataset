import os
import shutil

# Define absolute paths
data_dir = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug'
batch_1_dir = os.path.join(data_dir, 'batch_1')
batch_2_dir = os.path.join(data_dir, 'batch_2')
combined_dir = os.path.join(data_dir, 'combined_batches')

# Create combined_batches directory if it doesn't exist
os.makedirs(combined_dir, exist_ok=True)

def copy_images(src_dir, dst_dir):
    for filename in os.listdir(src_dir):
        src_file = os.path.join(src_dir, filename)
        dst_file = os.path.join(dst_dir, filename)
        # Only copy files (skip directories)
        if os.path.isfile(src_file):
            # If file with same name exists, add a prefix to avoid overwrite
            if os.path.exists(dst_file):
                name, ext = os.path.splitext(filename)
                count = 1
                while True:
                    new_filename = f"{name}_dup{count}{ext}"
                    new_dst_file = os.path.join(dst_dir, new_filename)
                    if not os.path.exists(new_dst_file):
                        dst_file = new_dst_file
                        break
                    count += 1
            shutil.move(src_file, dst_file)

# Copy images from both batches
copy_images(batch_1_dir, combined_dir)
copy_images(batch_2_dir, combined_dir)

print(f"All images from batch_1 and batch_2 have been moved to {combined_dir}")

# --- New logic: Move positive instances ---
positive_dir = os.path.join(data_dir, 'positive_instances')
os.makedirs(positive_dir, exist_ok=True)

# Files to read
filenames_files = [
    os.path.join(data_dir, 'filenames.txt'),
    os.path.join(data_dir, 'file_list.txt')
]

# Collect all unique filenames from both lists
all_filenames = set()
for txt_file in filenames_files:
    with open(txt_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                all_filenames.add(name)

# Move files if they exist in combined_batches
moved_count = 0
for filename in all_filenames:
    src = os.path.join(combined_dir, filename)
    dst = os.path.join(positive_dir, filename)
    if os.path.exists(src):
        # Handle duplicate in destination
        if os.path.exists(dst):
            name, ext = os.path.splitext(filename)
            count = 1
            while True:
                new_dst = os.path.join(positive_dir, f"{name}_dup{count}{ext}")
                if not os.path.exists(new_dst):
                    dst = new_dst
                    break
                count += 1
        shutil.move(src, dst)
        moved_count += 1

print(f"Moved {moved_count} files to {positive_dir}")
