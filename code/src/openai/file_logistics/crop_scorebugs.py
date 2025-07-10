import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_file(f):
    try:
        src_dir = 'tennis/data/scorebug/positive_instances'
        dst_dir = 'tennis/data/scorebug/positive_scorebugs'
        left, right, top, bottom = 103, 1100, 888, 980
        src_path = os.path.join(src_dir, f)
        cap = cv2.VideoCapture(src_path)
        ret, frame = cap.read()
        if not ret:
            return f'Failed to read frame for: {f}, skipping.'
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return f'Crop failed for: {f}, skipping.'
        out_path = os.path.join(dst_dir, f.replace('.mp4', '.jpg'))
        cv2.imwrite(out_path, crop)
        cap.release()
        return None
    except Exception as e:
        return f'Error processing {f}: {e}, skipping.'

if __name__ == '__main__':
    src_dir = 'tennis/data/scorebug/positive_instances'
    dst_dir = 'tennis/data/scorebug/positive_scorebugs'
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith('.mp4')]
    warnings = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in tqdm(as_completed(futures, timeout=None), total=len(files)):
            try:
                result = future.result(timeout=1)
                if result:
                    warnings.append(result)
            except Exception:
                warnings.append(f'Timeout or error for: {futures[future]}, skipping.')
    for w in warnings:
        print(w) 