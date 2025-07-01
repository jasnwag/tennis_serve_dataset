import os
import subprocess
import argparse

"""
Usage:
    python download_youtube_video.py --url <YOUTUBE_URL> --output <OUTPUT_DIRECTORY>

Example:
    python download_youtube_video.py --url https://www.youtube.com/shorts/nklnqDYgp-8 --output ./data/USTA
"""

def download_youtube_video(url, output_path):
    os.makedirs(output_path, exist_ok=True)
    print(f"Downloading from {url} to {output_path}")
    result = subprocess.run([
        "yt-dlp",
        "-o", os.path.join(output_path, '%(title)s.%(ext)s'),
        url
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading video:")
        print(result.stderr)
    else:
        print("Download completed successfully.")
        print(result.stdout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a YouTube video to a specified directory using yt-dlp.")
    parser.add_argument('--url', required=True, help='YouTube video URL')
    parser.add_argument('--output', required=True, help='Output directory')
    args = parser.parse_args()
    download_youtube_video(args.url, args.output)
