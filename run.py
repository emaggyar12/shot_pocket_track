import os
import argparse
import cv2
import csv

from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from team_assigner import TeamAssigner

from configs import (
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    OUTPUT_VIDEO_PATH,
    # STUBS_DEFAULT_PATH,  # only if you want to use stubs
)

def parse_args():
    p = argparse.ArgumentParser(description="Minimal Basketball Detector/Tracker")
    p.add_argument("input_video", type=str, help="Path to input video file")
    p.add_argument(
        "--output_video",
        type=str,
        default=OUTPUT_VIDEO_PATH,
        help="Path to output video file (e.g., output.mp4)",
    )
    # If you want stubs, uncomment these:
    # p.add_argument("--stub_path", type=str, default=STUBS_DEFAULT_PATH, help="Path to stub dir")
    # p.add_argument("--use_stubs", action="store_true", help="Read detections from stubs if available")
    return p.parse_args()

def save_tracks_csv(player_tracks, ball_tracks, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "type", "x1", "y1", "x2", "y2"])
        
        # Players
        for frame_idx, tracks in enumerate(player_tracks):
            if isinstance(tracks, dict):
                for pid, info in tracks.items():
                    x1, y1, x2, y2 = info["bbox"]
                    writer.writerow([frame_idx, pid, "player", x1, y1, x2, y2])

        # Ball
        for frame_idx, tracks in enumerate(ball_tracks):
            if isinstance(tracks, dict):
                for bid, info in tracks.items():
                    x1, y1, x2, y2 = info["bbox"]
                    writer.writerow([frame_idx, bid, "ball", x1, y1, x2, y2])

def draw_basic_boxes(frames, player_tracks, ball_tracks, player_color=(0, 255, 0), ball_color=(0, 0, 255)):
    """
    Draw minimal rectangles for players and ball.
    Expects each frame index to map to a dict of {track_id: {"bbox":[x1,y1,x2,y2], ...}}
    for players; and for ball a list-of-dicts or same shape (we normalize below).
    """
    out = []
    n = len(frames)

    # Normalize ball format: we accept either
    #   ball_tracks[f] -> {ball_id: {"bbox":[x1,y1,x2,y2]}}
    # or
    #   ball_tracks[f] -> {"bbox":[x1,y1,x2,y2]} (we’ll wrap it)
    norm_ball = []
    for f in range(n):
        bt = ball_tracks[f] if f < len(ball_tracks) else {}
        if isinstance(bt, dict) and "bbox" in bt:
            norm_ball.append({1: {"bbox": bt["bbox"]}})
        elif isinstance(bt, dict):
            norm_ball.append(bt)
        else:
            norm_ball.append({})

    for f in range(n):
        frame = frames[f].copy()

        # Players
        if f < len(player_tracks) and isinstance(player_tracks[f], dict):
            for pid, info in player_tracks[f].items():
                bbox = info.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), player_color, 2)
                    cv2.putText(frame, f"P{pid}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, player_color, 1, cv2.LINE_AA)

        # Ball
        if f < len(norm_ball) and isinstance(norm_ball[f], dict):
            for bid, info in norm_ball[f].items():
                bbox = info.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), ball_color, 2)
                    cv2.putText(frame, "Ball", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ball_color, 1, cv2.LINE_AA)

        out.append(frame)
    return out

def main():
    args = parse_args()

    # 1) Read frames
    frames = read_video(args.input_video)

    # 2) Init trackers
    player_tracker = PlayerTracker(PLAYER_DETECTOR_PATH)
    ball_tracker = BallTracker(BALL_DETECTOR_PATH)

    # 3) Run detection/tracking
    # If you want stub support, change read_from_stub and add stub_path.
    player_tracks = player_tracker.get_object_tracks(
        frames,
        read_from_stub=False,
        # stub_path=os.path.join(args.stub_path, "player_track_stubs.pkl")
    )
    ball_tracks = ball_tracker.get_object_tracks(
        frames,
        read_from_stub=False,
        # stub_path=os.path.join(args.stub_path, "ball_track_stubs.pkl")
    )

    # 4) Optional ball cleanup (safe no-ops if your class includes these)
    if hasattr(ball_tracker, "remove_wrong_detections"):
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
    if hasattr(ball_tracker, "interpolate_ball_positions"):
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # 5) Team assignment (minimal)
    team_assigner = TeamAssigner()
    _player_assignment = team_assigner.get_player_teams_across_frames(
        frames,
        player_tracks,
        read_from_stub=False,
        # stub_path=os.path.join(args.stub_path, "player_assignment_stub.pkl")
    )
    # (We don’t draw team colors here, but you now have the mapping if you need it.)

    # 6) Minimal overlay (rectangles only)
    output_frames = draw_basic_boxes(frames, player_tracks, ball_tracks)

    # 7) Save video
    save_video(output_frames, args.output_video)
    os.makedirs("output", exist_ok=True)   # make sure output folder exists
    save_tracks_csv(player_tracks, ball_tracks, "output/tracks.csv")

if __name__ == "__main__":
    main()
