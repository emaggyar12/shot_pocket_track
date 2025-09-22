#!/usr/bin/env python3
import argparse
import datetime as dt
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --------------------------
# DEFAULT CONFIG (can be overridden by CLI)
# --------------------------

GAME_NAME   = "bc_louisville_02052025"          # change this or pass --game
TABLE_PATH  = Path("all_videos.xls")  # master table with columns: game,start,end,name
VIDEO_DIR   = Path("../data/game_videos")              # where full-game videos live
OUT_ROOT    = Path("../data/game_clips")               # root folder for outputs
FRAME_ACCURATE = False                               # --no-frame-accurate to speed up

# --------------------------
# HELPERS
# --------------------------

def ensure_ext(name: str, ext: str = ".mp4") -> str:
    root, e = os.path.splitext(name)
    return name if e else root + ext

def normalize_name(name: str) -> str:
    """Trim, drop extension, make a clean stem (underscores for spaces)."""
    stem = Path(name).stem.strip()
    stem = re.sub(r"\s+", "_", stem)
    return stem or "clip"

def uniquify_name(base_name: str, used_counts: dict) -> str:
    """
    Always start numbering at 001. Example: 'hepburn_miss' -> 'hepburn_miss_001.mp4', then _002, ...
    """
    stem = normalize_name(base_name)
    ext = ".mp4"

    n = used_counts[stem] + 1  # start at 1
    candidate = f"{stem}_{n:03d}{ext}"

    # If file exists already (re-running), keep incrementing.
    while (OUTPUT_DIR / candidate).exists():
        n += 1
        candidate = f"{stem}_{n:03d}{ext}"

    used_counts[stem] = n
    return candidate

def excel_time_to_hhmmss(val: object) -> str:
    """
    Coerce an Excel/CSV cell (string, datetime/time, or Excel serial) to 'HH:MM:SS'.
    Keeps strings as-is (trimmed). Returns '' for NaN.
    """
    if pd.isna(val):
        return ""
    # Pandas Timestamp / Python datetime
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        return val.strftime("%H:%M:%S")
    # Python time
    if isinstance(val, dt.time):
        return val.strftime("%H:%M:%S")
    # Excel serial number (days since 1899-12-30); use time-of-day fraction
    if isinstance(val, (int, float)) and not (isinstance(val, bool)) and not math.isnan(float(val)):
        total_seconds = int(round((float(val) % 1.0) * 24 * 3600))
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    # String
    return str(val).strip()

def _parse_hhmmss_to_seconds(t: str) -> float:
    """
    Parse 'HH:MM:SS', 'MM:SS', raw seconds, or '+X.Y' (duration seconds) to seconds (float).
    """
    if not t:
        return float("nan")
    t = t.strip()
    if t.startswith("+"):  # duration format
        return float(t[1:])
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(t)

def cut_clip_ffmpeg(src_video: Path, start: str, end: str, out_path: Path, accurate: bool = True) -> int:
    """
    Use -t <duration> instead of -to to avoid '-to smaller than -ss' errors.
    If end == start (zero length) or invalid, force a 1.0s duration and warn.
    Supports end like '+2.5' to indicate a duration from start.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_s = _parse_hhmmss_to_seconds(start)
    if str(end).strip().startswith("+"):
        dur_s = _parse_hhmmss_to_seconds(end)
    else:
        end_s = _parse_hhmmss_to_seconds(end)
        dur_s = end_s - start_s

    if not math.isfinite(dur_s) or dur_s <= 0:
        print(f"[WARN] Non-positive duration for {out_path.name} ({start}–{end}); forcing 1.0s")
        dur_s = 1.0

    if accurate:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(src_video),
            "-ss", f"{start_s:.3f}",
            "-t",  f"{dur_s:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-an", "-movflags", "+faststart",
            str(out_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-t",  f"{dur_s:.3f}",
            "-i", str(src_video),
            "-map", "0:v:0",
            "-c:v", "copy",
            "-an",
            str(out_path)
        ]
    return subprocess.call(cmd)

def read_table(path: Path) -> pd.DataFrame:
    """
    Read master table supporting .csv, .tsv, .xls, .xlsx
    Must contain columns: game, start, end, name
    """
    if not path.exists():
        raise SystemExit(f"Table not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(path)
    elif suffix in {".tsv"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".xls", ".xlsx"}:
        # For .xls, make sure xlrd is installed. For .xlsx, openpyxl.
        try:
            df = pd.read_excel(path)
        except ImportError as e:
            raise SystemExit(
                f"Failed to read {suffix} file: {e}\n"
                f"Try: pip install {'xlrd' if suffix=='.xls' else 'openpyxl'}"
            )
    else:
        raise SystemExit(f"Unsupported table format: {suffix}")

    # Basic validation
    required = {"game", "start", "end", "name"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        # Try case-insensitive align
        cols_map = {c.lower(): c for c in df.columns}
        if not missing - set(cols_map.keys()):
            df = df.rename(columns={cols_map[k]: k for k in required if k in cols_map})
        else:
            raise SystemExit(f"Table is missing required columns: {sorted(missing)}")
    else:
        # Normalize all to lowercase expected names
        df = df.rename(columns={c: c.lower() for c in df.columns})

    return df

# --------------------------
# MAIN
# --------------------------

def main():
    parser = argparse.ArgumentParser(description="Cut clips for a single game from a master table.")
    parser.add_argument("--game", type=str, default=GAME_NAME, help="Game name to filter (matches 'game' column).")
    parser.add_argument("--table", type=Path, default=TABLE_PATH, help="Path to master table (csv/tsv/xls/xlsx).")
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR, help="Directory containing full-game .mp4 files.")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT, help="Root output directory for clips.")
    parser.add_argument("--frame-accurate", action="store_true", default=FRAME_ACCURATE, help="Re-encode for frame-accurate cutting.")
    parser.add_argument("--no-frame-accurate", dest="frame_accurate", action="store_false", help="Faster keyframe cuts (no re-encode).")
    args = parser.parse_args()

    game_name = args.game.strip()
    table_path = args.table
    video_dir  = args.video_dir
    out_root   = args.out_root
    accurate   = args.frame_accurate

    global OUTPUT_DIR
    OUTPUT_DIR = out_root / game_name

    # Load and filter rows for this game
    df = read_table(table_path)

    # Normalize time columns to HH:MM:SS strings
    df["start"] = df["start"].apply(excel_time_to_hhmmss)
    df["end"]   = df["end"].apply(excel_time_to_hhmmss)

    # Filter by game (exact match). If you want case-insensitive, uncomment the lower() logic:
    # rows = df[df["game"].str.lower() == game_name.lower()]
    rows = df[df["game"] == game_name].copy()

    if rows.empty:
        # Give a friendly hint with samples
        example_values = df["game"].dropna().astype(str).unique().tolist()[:10]
        raise SystemExit(
            f"No rows found for game='{game_name}' in {table_path}.\n"
            f"Examples available: {example_values}"
        )

    video_path = video_dir / f"{game_name}.mp4"
    if not video_path.exists():
        raise SystemExit(f"Video not found for game: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    used_counts = defaultdict(int)
    created = 0
    failed  = 0

    for _, row in rows.iterrows():
        start = str(row.get("start", "")).strip()
        end   = str(row.get("end", "")).strip()
        base_name = str(row.get("name", "")).strip()

        if not start or not end or not base_name:
            print(f"[SKIP] Missing field(s): start='{start}', end='{end}', name='{base_name}'")
            continue

        out_name = uniquify_name(base_name, used_counts)
        out_path = OUTPUT_DIR / out_name

        rc = cut_clip_ffmpeg(video_path, start, end, out_path, accurate=accurate)
        if rc == 0:
            print(f"Created: {out_name}")
            created += 1
        else:
            print(f"[WARN] ffmpeg failed for {out_name} ({start}–{end})")
            failed += 1

    print(f"\nDone. Created {created} clip(s). Failures: {failed}. Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
