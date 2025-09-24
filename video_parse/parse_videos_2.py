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
from edit_excel import normalize_times_fill_end

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

def uniquify_name(base_name: str, used_counts: dict, output_dir: Path) -> str:
    """
    Always start numbering at 001. Example: 'hepburn_miss' -> 'hepburn_miss_001.mp4', then _002, ...
    """
    stem = normalize_name(base_name)
    ext = ".mp4"

    n = used_counts[stem] + 1  # start at 1
    candidate = f"{stem}_{n:03d}{ext}"

    # If file exists already (re-running), keep incrementing.
    while (output_dir / candidate).exists():
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
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        return val.strftime("%H:%M:%S")
    if isinstance(val, dt.time):
        return val.strftime("%H:%M:%S")
    if isinstance(val, (int, float)) and not (isinstance(val, bool)) and not math.isnan(float(val)):
        total_seconds = int(round((float(val) % 1.0) * 24 * 3600))
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"
    return str(val).strip()

def normalize_time_string(val: object) -> str:
    """
    Normalize different time formats into 'HH:MM:SS'.

    Accepts:
    - 'HH:MM:SS' or 'MM:SS' (kept as-is)
    - Excel serials (floats/ints)
    - Pandas/Python datetime or time objects
    - Raw digit strings like '13456' (-> '01:34:56') or '945' (-> '00:09:45')
    """
    if pd.isna(val):
        return ""

    # If already a datetime or time
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        return val.strftime("%H:%M:%S")
    if isinstance(val, dt.time):
        return val.strftime("%H:%M:%S")

    # If numeric (Excel serial, or just seconds)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        # Excel-style serials (fraction of a day)
        if float(val) < 1.0:
            total_seconds = int(round((float(val) % 1.0) * 24 * 3600))
        else:
            total_seconds = int(val)
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # If string
    s = str(val).strip()
    if ":" in s:  # already formatted
        return s

    # Raw digits like 945 -> 00:09:45, 13456 -> 01:34:56
    if s.isdigit():
        # Pad left so length is always multiple of 2 or 6 digits
        s = s.zfill(6)
        h, m, sec = s[0:2], s[2:4], s[4:6]
        return f"{h}:{m}:{sec}"

    return s


def _parse_hhmmss_to_seconds(t: str) -> float:
    if not t:
        return float("nan")
    t = t.strip()
    if t.startswith("+"):
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
    if not Path(path).exists():
        raise SystemExit(f"Table not found: {path}")

    suffix = Path(path).suffix.lower()
    if suffix in {".csv"}:
        df = pd.read_csv(path)
    elif suffix in {".tsv"}:
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise SystemExit(f"Unsupported table format: {suffix}")

    required = {"game", "start", "end", "name"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        cols_map = {c.lower(): c for c in df.columns}
        if not missing - set(cols_map.keys()):
            df = df.rename(columns={cols_map[k]: k for k in required if k in cols_map})
        else:
            raise SystemExit(f"Table is missing required columns: {sorted(missing)}")
    else:
        df = df.rename(columns={c: c.lower() for c in df.columns})

    return df


# --------------------------
# MAIN FUNCTION (dynamic)
# --------------------------

def parse_videos(game_name, table_path, frame_accurate=False):
    # Your explicit paths for the full game and output directory
    game_full_video = f"data/game_videos/{game_name}.mp4"
    out_dir = f"data/game_clips/{game_name}/"   # added missing slash

    video_path = Path(game_full_video)
    output_dir = Path(out_dir)

    df = read_table(table_path)

    df[['start', 'end']] = df.apply(
        lambda r: pd.Series(normalize_times_fill_end(r['start'], r['end'])),
        axis=1
    )

    df["start"] = df["start"].apply(normalize_time_string)
    df["end"]   = df["end"].apply(normalize_time_string)

    rows = df[df["game"] == game_name].copy()
    if rows.empty:
        example_values = df["game"].dropna().astype(str).unique().tolist()[:10]
        raise SystemExit(
            f"No rows found for game='{game_name}' in {table_path}.\n"
            f"Examples available: {example_values}"
        )

    if not video_path.exists():
        raise SystemExit(f"Video not found for game: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

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

        out_name = uniquify_name(base_name, used_counts, output_dir)
        out_path = output_dir / out_name

        rc = cut_clip_ffmpeg(video_path, start, end, out_path, accurate=frame_accurate)
        if rc == 0:
            print(f"Created: {out_name}")
            created += 1
        else:
            print(f"[WARN] ffmpeg failed for {out_name} ({start}–{end})")
            failed += 1

    print(f"\nDone. Created {created} clip(s). Failures: {failed}. Output: {output_dir}")



# --------------------------
# CLI fallback
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and cut clips for a single game.")
    parser.add_argument("--game", dest="game", required=True, help="Game name key (e.g., syracuse_louisville_01142025)")
    parser.add_argument("--table", dest="table", required=True, help="Path to master table (xls/xlsx)")
    parser.add_argument("--frame-accurate", dest="frame_accurate", action="store_true", help="Enable frame-accurate cuts")
    args = parser.parse_args()
    parse_videos(args.game, args.table, frame_accurate=args.frame_accurate)
