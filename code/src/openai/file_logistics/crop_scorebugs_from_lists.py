import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, TimeoutError

def crop_scorebug(src_path, out_path, left, right, top, bottom):
    try:
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            return f'Failed to open video: {os.path.basename(src_path)}'
        # Go to last frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return f'Failed to read last frame for: {os.path.basename(src_path)}'
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return f'Crop failed for: {os.path.basename(src_path)}'
        cv2.imwrite(out_path, crop)
        return None
    except Exception as e:
        return f'Error processing {os.path.basename(src_path)}: {e}'

def process_one(args):
    fname, src_path, out_path, left, right, top, bottom = args
    return crop_scorebug(src_path, out_path, left, right, top, bottom)

if __name__ == '__main__':
    batch_dirs = [
        '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/previous/pose_youtube/data/videos/broadcast_tennis/usopen_2024/data_undergrads/batch_1',
        '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/previous/pose_youtube/data/videos/broadcast_tennis/usopen_2024/data_undergrads/batch_2',
    ]
    txt_files = [
        '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/file_list.txt',
        '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/filenames.txt',
    ]
    out_dir = '/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/scorebug/positive_scorebugs'
    os.makedirs(out_dir, exist_ok=True)
    left, right, top, bottom = 103, 800, 888, 980  # right reduced by 300

    all_filenames = set()
    for txt in txt_files:
        with open(txt, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    all_filenames.add(name)

    file_paths = {}
    for batch_dir in batch_dirs:
        for fname in os.listdir(batch_dir):
            if fname in all_filenames:
                file_paths[fname] = os.path.join(batch_dir, fname)

    tasks = []
    for fname in all_filenames:
        if fname not in file_paths:
            continue
        src_path = file_paths[fname]
        out_path = os.path.join(out_dir, fname.replace('.mp4', '.jpg'))
        if os.path.exists(out_path):
            continue
        tasks.append((fname, src_path, out_path, left, right, top, bottom))

    warnings = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_one, task): task[0] for task in tasks}
        for future in tqdm(futures, desc='Cropping scorebugs', total=len(futures)):
            try:
                result = future.result(timeout=2)
                if result:
                    warnings.append(result)
            except TimeoutError:
                warnings.append(f'Timeout for: {futures[future]}, skipping.')
            except Exception as e:
                warnings.append(f'Error for: {futures[future]}: {e}, skipping.')
    for w in warnings:
        print(w) 