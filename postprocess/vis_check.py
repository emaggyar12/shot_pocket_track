#!/usr/bin/env python3
import cv2
import os

VIDEO_PATH = "output/output_edwards_make.mp4"
OUT_DIR = "postprocess/debug_frames"

# Replace these with the frame numbers you want to inspect
FIRST_TOUCH = 80
POCKET      = 96
RISE_START  = 105

# Number of frames before/after to grab for context
PAD = 3

def extract_frames(video_path, frame_nums, out_dir, pad=0):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for f in frame_nums:
        for offset in range(-pad, pad+1):
            frame_id = f + offset
            if frame_id < 0 or frame_id >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            out_name = os.path.join(out_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(out_name, frame)
            print(f"Saved {out_name}")

    cap.release()

if __name__ == "__main__":
    frames_of_interest = [FIRST_TOUCH, POCKET, RISE_START]
    extract_frames(VIDEO_PATH, frames_of_interest, OUT_DIR, pad=PAD)
    print("Done. Check the debug_frames/ folder for output images.")
