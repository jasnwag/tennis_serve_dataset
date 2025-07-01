import cv2
import numpy as np
from PIL import Image
import imageio
import os

# --- Paths ---
VIDEO1 = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/server_comparison/Carlos Alcaraz vs. Li Tu Full Match ｜ 2024 US Open Round 1.f617_56.7_225224-2.mp4"
VIDEO2 = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/server_comparison/Taylor Townsend vs. Paula Badosa Full Match ｜ 2024 US Open Round 2.f617_50.6_272930.mp4"
GIF_PATH = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/server_comparison/left_shoulder_average.gif"

# --- Parameters ---
NUM_FRAMES = 90

# --- Load animated plot GIF ---
gif_frames = []
with Image.open(GIF_PATH) as im:
    for frame in range(NUM_FRAMES):
        im.seek(frame)
        gif_frames.append(np.array(im.convert('RGB')))
plot_h, plot_w = gif_frames[0].shape[:2]

# --- Open videos ---
cap1 = cv2.VideoCapture(VIDEO1)
cap2 = cv2.VideoCapture(VIDEO2)

# --- Get frame size ---
ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
if not (ret1 and ret2):
    raise RuntimeError("Could not read first frame from one or both videos.")

h1, w1 = frame1.shape[:2]
h2, w2 = frame2.shape[:2]
vid_h = max(h1, h2)
vid_w = w1 + w2

# --- Resize frames if needed to same height ---
def resize_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w*scale), height))

# --- Output GIF path ---
out_gif_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/server_comparison/comparison.gif"
combined_frames = []

for idx in range(NUM_FRAMES):
    if idx > 0:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            break
    # Convert BGR to RGB for natural colors
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    # Resize both video frames to same width if needed
    target_width = max(frame1.shape[1], frame2.shape[1])
    if frame1.shape[1] != target_width:
        frame1 = cv2.resize(frame1, (target_width, frame1.shape[0]), interpolation=cv2.INTER_AREA)
    if frame2.shape[1] != target_width:
        frame2 = cv2.resize(frame2, (target_width, frame2.shape[0]), interpolation=cv2.INTER_AREA)
    # Stack videos vertically
    video_col = np.vstack([frame1, frame2])
    # Resize plot frame to match combined height of videos
    plot_frame = gif_frames[idx]
    if plot_frame.shape[0] != video_col.shape[0]:
        plot_frame = cv2.resize(plot_frame, (plot_frame.shape[1], video_col.shape[0]), interpolation=cv2.INTER_AREA)
    # Resize plot width to match video_col height ratio (optional: or keep as-is for left-side bar)
    # To make the plot not too thin, set a minimum width (e.g., 1/2 of video_col width)
    min_plot_width = video_col.shape[1] // 2
    if plot_frame.shape[1] < min_plot_width:
        plot_frame = cv2.resize(plot_frame, (min_plot_width, plot_frame.shape[0]), interpolation=cv2.INTER_AREA)
    # --- Stack horizontally ---
    combined = np.hstack([plot_frame, video_col])
    combined_frames.append(combined)

cap1.release()
cap2.release()

# --- Save as GIF ---
imageio.mimsave(out_gif_path, combined_frames, duration=30/1000)
print(f"GIF saved to {out_gif_path}")

# --- Open videos ---
cap1 = cv2.VideoCapture(VIDEO1)
cap2 = cv2.VideoCapture(VIDEO2)

# --- Get frame size ---
ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
if not (ret1 and ret2):
    raise RuntimeError("Could not read first frame from one or both videos.")

h1, w1 = frame1.shape[:2]
h2, w2 = frame2.shape[:2]
vid_h = max(h1, h2)
vid_w = w1 + w2

# --- Resize frames if needed to same height ---
def resize_to_height(img, height):
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w*scale), height))

import imageio

# --- Output GIF path ---
gif_path = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/tennis/data/initial/visualizations/server_comparison/comparison.gif"
gif_frames = []

# --- Main loop ---
for idx in range(NUM_FRAMES):
    if idx > 0:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not (ret1 and ret2):
            break
    # Convert BGR to RGB for natural colors
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    # Resize if needed
    if h1 != vid_h:
        frame1 = resize_to_height(frame1, vid_h)
    if h2 != vid_h:
        frame2 = resize_to_height(frame2, vid_h)
    # Stack videos side by side
    video_row = np.hstack([frame1, frame2])
    # --- Plot with moving line ---
    plot_with_line = get_plot_with_line(idx)
    # Pad plot width if needed to match video width
    if plot_with_line.shape[1] < video_row.shape[1]:
        pad_w = video_row.shape[1] - plot_with_line.shape[1]
        plot_with_line = np.pad(plot_with_line, ((0,0),(0,pad_w),(0,0)), mode='constant', constant_values=255)
    elif plot_with_line.shape[1] > video_row.shape[1]:
        plot_with_line = plot_with_line[:,:video_row.shape[1],:]
    # --- Stack vertically ---
    combined = np.vstack([plot_with_line, video_row])
    gif_frames.append(combined)

cap1.release()
cap2.release()

# --- Save as GIF ---
imageio.mimsave(gif_path, gif_frames, duration=1/1000)
print(f"GIF saved to {gif_path}")
