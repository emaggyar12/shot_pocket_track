#!/usr/bin/env python3
"""
select_target_switch.py

Workflow:
1) Read tracks.csv
2) Find player with the largest *cumulative intersection area* vs the ball (global winner)
3) Create most_overlap.csv (do NOT force/replace anything yet)
4) In most_overlap.csv, find "semi-consecutive" occurrences of the target player as a gapped run
   (allow up to MAX_HOLE consecutive non-target frames)
5) The switch_frame is the FIRST frame of the best gapped run
6) From switch_frame onward, force that player in the table (using that player's actual boxes when present)
   and write most_overall_edited.csv

Notes:
- We choose the per-frame "most overlap" player by *IoU* by default (you can switch to raw intersection).
- “Semi-consecutive” is tuned by MAX_HOLE; default = 2 (allows short interruptions).
- Strongest run = most target hits; tiebreak = longest span; final tiebreak = earliest start.
"""

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Tunables (adjust on the CLI if you like) ----------------
DEFAULT_CSV_IN   = "output/tracks.csv"
DEFAULT_OUT_A    = "most_overlap.csv"
DEFAULT_OUT_B    = "most_overall_edited.csv"
ALLOW_CASE_TYPES = True  # treat type case-insensitively

# Gapped-run parameters for "semi-consecutive"
MAX_HOLE = 5          # allow up to this many consecutive non-target frames inside a run
MIN_HITS = 5          # optional: require at least this many target frames inside a run to accept
USE_IOU  = True       # if False, uses raw intersection area instead of IoU to choose per-frame winner

# --------------------------------------------------------------------------

REQUIRED = {"frame","id","type","x1","y1","x2","y2"}

def area(x1,y1,x2,y2):
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def intersect(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return (ix1, iy1, ix2, iy2), area(ix1, iy1, ix2, iy2)

def iou(a, b):
    _, inter = intersect(a, b)
    A = area(*a); B = area(*b)
    denom = A + B - inter
    return inter / denom if denom > 0 else 0.0

def modal_player(series: pd.Series) -> int:
    s2 = series[(series != -1) & (~pd.isna(series))]
    return int(s2.value_counts().idxmax()) if not s2.empty else -1

def build_players_balls(df: pd.DataFrame):
    """Split, standardize, and build per-frame maps for ball and players."""
    if ALLOW_CASE_TYPES:
        df["type"] = df["type"].astype(str).str.lower()

    balls = df[df["type"] == "ball"].copy()
    players = df[df["type"] != "ball"].copy()

    # Keep only the largest ball per frame
    balls["area"] = (balls["x2"]-balls["x1"]).clip(lower=0) * (balls["y2"]-balls["y1"]).clip(lower=0)
    balls = (balls.sort_values(["frame","area"], ascending=[True, False])
                  .drop_duplicates(subset=["frame"], keep="first"))

    balls_by_frame = {
        int(r.frame):(float(r.x1),float(r.y1),float(r.x2),float(r.y2))
        for _,r in balls.iterrows()
    }

    players_by_frame = defaultdict(list)   # frame -> list[(pid, box)]
    player_box_at = {}                     # (frame, pid) -> box
    for f, grp in players.groupby("frame"):
        f = int(f)
        for _, r in grp.iterrows():
            pid = int(r.id)
            box = (float(r.x1),float(r.y1),float(r.x2),float(r.y2))
            players_by_frame[f].append((pid, box))
            player_box_at[(f, pid)] = box

    max_frame = int(df["frame"].max())
    frames = list(range(0, max_frame+1))
    return frames, balls_by_frame, players_by_frame, player_box_at

def global_top_intersection_player(frames, balls_by_frame, players_by_frame):
    """Return (top_inter_pid, total_raw_intersection, top_iou_pid, total_iou)."""
    iou_sum = defaultdict(float)
    inter_sum = defaultdict(float)
    for f in frames:
        ball_box = balls_by_frame.get(f)
        if ball_box is None:
            continue
        for pid, pbox in players_by_frame.get(f, []):
            # IoU
            iou_val = iou(ball_box, pbox)
            iou_sum[pid] += float(iou_val)
            # Raw inter area
            _, inter_area = intersect(ball_box, pbox)
            inter_sum[pid] += float(inter_area)

    def top_key(d):
        return max(d.items(), key=lambda kv: kv[1])[0] if d else -1

    top_iou_pid = top_key(iou_sum)
    top_inter_pid = top_key(inter_sum)
    return top_inter_pid, inter_sum.get(top_inter_pid, 0.0), top_iou_pid, iou_sum.get(top_iou_pid, 0.0)

def build_most_overlap_table(frames, balls_by_frame, players_by_frame, use_iou=True) -> pd.DataFrame:
    """Per frame, pick the player whose box has highest IoU (or raw inter area) with the ball."""
    out_rows = []
    for f in frames:
        ball_box = balls_by_frame.get(f)
        if ball_box is None:
            out_rows.append({
                "frame": f,
                "ball_x1": np.nan, "ball_y1": np.nan, "ball_x2": np.nan, "ball_y2": np.nan,
                "player_most_overlap": -1,
                "player_x1": np.nan, "player_y1": np.nan, "player_x2": np.nan, "player_y2": np.nan
            })
            continue

        best_pid = -1
        best_box = (np.nan, np.nan, np.nan, np.nan)
        best_score = -1.0
        for pid, pbox in players_by_frame.get(f, []):
            if use_iou:
                score = iou(ball_box, pbox)
            else:
                _, score = intersect(ball_box, pbox)
            if score > best_score:
                best_score = score
                best_pid = pid
                best_box = pbox

        out_rows.append({
            "frame": f,
            "ball_x1": ball_box[0], "ball_y1": ball_box[1], "ball_x2": ball_box[2], "ball_y2": ball_box[3],
            "player_most_overlap": int(best_pid),
            "player_x1": best_box[0], "player_y1": best_box[1], "player_x2": best_box[2], "player_y2": best_box[3]
        })

    return pd.DataFrame(out_rows, columns=[
        "frame",
        "ball_x1","ball_y1","ball_x2","ball_y2",
        "player_most_overlap",
        "player_x1","player_y1","player_x2","player_y2"
    ])

def find_best_gapped_run(df: pd.DataFrame, target_pid: int, max_hole: int = MAX_HOLE, min_hits: int = MIN_HITS):
    """
    Find the strongest 'gapped run' of target_pid in df['player_most_overlap'] where
    we allow up to `max_hole` consecutive non-target frames inside the run.

    Returns a dict with:
        { 'start_frame', 'end_frame', 'hits', 'span', 'valid' }
    or None if nothing meets min_hits.
    """
    frames = df["frame"].to_numpy()
    ids    = df["player_most_overlap"].to_numpy()

    best = None
    in_run = False
    run_start_idx = None
    last_target_idx = None
    hits = 0

    for i, (f, pid) in enumerate(zip(frames, ids)):
        if pid == target_pid:
            if not in_run:
                # start a new run at this first target
                in_run = True
                run_start_idx = i
                hits = 0
            hits += 1
            last_target_idx = i
        else:
            if in_run:
                # check gap length from last target
                if last_target_idx is None:
                    # shouldn't happen if in_run, but guard anyway
                    in_run = False
                    continue
                gap = i - last_target_idx
                if gap > max_hole:
                    # close current run at last_target_idx
                    start_i = run_start_idx
                    end_i   = last_target_idx
                    span = end_i - start_i + 1
                    if hits >= min_hits:
                        cand = dict(start_frame=int(frames[start_i]),
                                    end_frame=int(frames[end_i]),
                                    hits=int(hits),
                                    span=int(span),
                                    valid=True)
                        # pick best by hits, then span, then earliest start
                        if (best is None or
                            cand["hits"] > best["hits"] or
                            (cand["hits"] == best["hits"] and cand["span"] > best["span"]) or
                            (cand["hits"] == best["hits"] and cand["span"] == best["span"] and cand["start_frame"] < best["start_frame"])):
                            best = cand
                    # reset: next run can start only when we see target again
                    in_run = False
                    run_start_idx = None
                    last_target_idx = None
                    hits = 0
                # else: gap is small; keep the run open

    # Close trailing run if still open
    if in_run and last_target_idx is not None:
        start_i = run_start_idx
        end_i   = last_target_idx
        span = end_i - start_i + 1
        if hits >= min_hits:
            cand = dict(start_frame=int(frames[start_i]),
                        end_frame=int(frames[end_i]),
                        hits=int(hits),
                        span=int(span),
                        valid=True)
            if (best is None or
                cand["hits"] > best["hits"] or
                (cand["hits"] == best["hits"] and cand["span"] > best["span"]) or
                (cand["hits"] == best["hits"] and cand["span"] == best["span"] and cand["start_frame"] < best["start_frame"])):
                best = cand

    return best

def force_after_switch(out_df: pd.DataFrame, player_box_at: dict, target_pid: int, switch_frame: int) -> pd.DataFrame:
    """Return a copy where from switch_frame onward, target_pid is forced; coordinates pulled from player_box_at if present."""
    out = out_df.copy()
    mask = out["frame"] >= switch_frame
    forced_ids = []
    new_boxes = {"player_x1": [], "player_y1": [], "player_x2": [], "player_y2": []}
    for f in out.loc[mask, "frame"]:
        box = player_box_at.get((int(f), target_pid))
        forced_ids.append(target_pid)
        if box is None:
            new_boxes["player_x1"].append(np.nan)
            new_boxes["player_y1"].append(np.nan)
            new_boxes["player_x2"].append(np.nan)
            new_boxes["player_y2"].append(np.nan)
        else:
            new_boxes["player_x1"].append(box[0])
            new_boxes["player_y1"].append(box[1])
            new_boxes["player_x2"].append(box[2])
            new_boxes["player_y2"].append(box[3])

    out.loc[mask, "player_most_overlap"] = forced_ids
    out.loc[mask, ["player_x1","player_y1","player_x2","player_y2"]] = pd.DataFrame(new_boxes, index=out.index[mask])
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_in", default=DEFAULT_CSV_IN, help="Input tracks.csv")
    parser.add_argument("--out_overlap", default=DEFAULT_OUT_A, help="Write most_overlap.csv here")
    parser.add_argument("--out_edited", default=DEFAULT_OUT_B, help="Write most_overall_edited.csv here")
    parser.add_argument("--max_hole", type=int, default=MAX_HOLE, help="Allowed consecutive non-target frames inside a run")
    parser.add_argument("--min_hits", type=int, default=MIN_HITS, help="Minimum target frames inside a run")
    parser.add_argument("--use_iou", action="store_true", help="Use IoU to pick per-frame winner (default)")
    parser.add_argument("--use_intersection", action="store_true", help="Use raw intersection area instead of IoU")
    args = parser.parse_args()

    use_iou = True
    if args.use_intersection:
        use_iou = False
    elif args.use_iou:
        use_iou = True

    # 1) Read CSV
    csv_path = Path(args.csv_in)
    if not csv_path.exists():
        raise SystemExit(f"Input not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if not REQUIRED.issubset(set(df.columns)):
        raise SystemExit(f"Input must have columns {sorted(REQUIRED)}; got {df.columns.tolist()}")

    # standardize types
    df["frame"] = df["frame"].astype(int)
    df["id"]    = df["id"].astype(int, errors="ignore")

    # 2) Build per-frame maps & find global top intersection player
    frames, balls_by_frame, players_by_frame, player_box_at = build_players_balls(df)
    top_inter_pid, inter_total, top_iou_pid, iou_total = global_top_intersection_player(frames, balls_by_frame, players_by_frame)

    print(f"[global] top_cumulative_iou: player={top_iou_pid}, total_iou={iou_total:.6f}")
    print(f"[global] top_cumulative_intersection_area: player={top_inter_pid}, total_area={inter_total:.2f}")

    target_pid = top_inter_pid
    if target_pid == -1:
        raise SystemExit("Could not determine a target player (no intersections found).")

    # 3) Build most_overlap table (no forcing)
    out_df = build_most_overlap_table(frames, balls_by_frame, players_by_frame, use_iou=use_iou)

    # save the raw overlap table
    out_df.to_csv(args.out_overlap, index=False)
    print(f"[OK] wrote {args.out_overlap}")

    # 4) Find gapped run for the target player
    best_run = find_best_gapped_run(out_df, target_pid=target_pid, max_hole=args.max_hole, min_hits=args.min_hits)
    if not best_run:
        # fallback: no acceptable run; just write edited identical to overlap
        out_df.to_csv(args.out_edited, index=False)
        print(f"[WARN] No semi-consecutive run found for player {target_pid} (max_hole={args.max_hole}, min_hits={args.min_hits}).")
        print(f"[OK] wrote {args.out_edited} (no forcing applied).")
        return

    switch_frame = best_run["start_frame"]
    print(f"[run] target={target_pid} start={best_run['start_frame']} end={best_run['end_frame']} "
          f"hits={best_run['hits']} span={best_run['span']}")
    print(f"[switch] switch_frame={switch_frame} (first target-frame of best gapped run)")

    # 6) Force from switch_frame onward and write edited file
    edited_df = force_after_switch(out_df, player_box_at, target_pid=target_pid, switch_frame=switch_frame)
    edited_df.to_csv(args.out_edited, index=False)
    print(f"[OK] wrote {args.out_edited} (forced player {target_pid} from frame {switch_frame}+).")

if __name__ == "__main__":
    main()
