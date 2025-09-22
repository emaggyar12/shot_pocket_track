#!/usr/bin/env python3
# Robust shot-pocket finder from most_overlap.csv (no CLI).
# Pocket = inside torso band, AFTER shooter's first sustained touch,
#          at/after global-min distance (with grace),
#          local min of BOTH distance-to-player-center AND speed,
#          AND within a short "stable hold" run (low speed + near-center) ≥ POCKET_HOLD_RUN.
#
# Outputs:
#   - pocket_result.json
#   - pocket_scores.csv

CSV_PATH       = "most_overlap.csv"
OUT_JSON       = "pocket_result.json"
OUT_SCORE_CSV  = "pocket_scores.csv"

import json
import numpy as np
import pandas as pd

# ===================== Tunables =====================
# Smoothing for ball center
ROLL_MED   = 5
ROLL_MEAN  = 5

# Guards / anchoring
EDGE_GUARD = 5
DEBOUNCE_AFTER_TOUCH = 3

# First-touch (sustained) detection
SUST_TOUCH_WIN = 5
SUST_TOUCH_MIN = 4

# Torso band (fraction of player height, measured from y1)
TORSO_BAND = (0.40, 0.65)

# Local minimum window (±win)
LOCAL_MIN_WIN = 2

# Distance percentile gating (post-touch)
DIST_PCTL_MAX = 0.40  # candidate must be within this dist percentile (closer is better)

# Require pocket to be AT/AFTER global-min distance (with small grace)
AFTER_GLOBAL_MIN_ALLOW = 12  # frames allowed after global min(dist) to search

# Stability (hold) requirement
POCKET_HOLD_RUN    = 3          # need ≥ this many consecutive "stable" frames
SPEED_HOLD_THRESH  = 6.0        # px/frame; low speed threshold for hold
DIST_HOLD_PCTL     = 0.35       # near-center threshold for hold (percentile on dist_norm)

# Scoring weights (favor speed stillness now)
W_DIST = 0.35
W_VEL  = 0.65

# Rise detection (upward vy)
MIN_RUN_UP = 3
VY_UP      = 0.8
# ====================================================

def roll_med(a, k):
    s = pd.Series(a)
    return s.rolling(k, center=True, min_periods=1).median().bfill().ffill().to_numpy()

def roll_mean(a, k):
    s = pd.Series(a)
    return s.rolling(k, center=True, min_periods=1).mean().bfill().ffill().to_numpy()

def velocity(x):
    v = np.zeros_like(x)
    v[1:] = x[1:] - x[:-1]
    return v

def robust_scale_abs(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    return np.abs(x - med) / (1.4826 * mad)

def modal_player(series):
    s = series[(series != -1) & (~pd.isna(series))]
    return int(s.value_counts().idxmax()) if not s.empty else -1

def first_sustained_touch(mask, win, min_hits, start=0):
    n = len(mask)
    for i in range(max(start, EDGE_GUARD), n - win):
        if np.count_nonzero(mask[i:i+win]) >= min_hits:
            return i
    return None

def is_local_min(arr, i, win):
    L = max(0, i - win); R = min(len(arr), i + win + 1)
    return np.all(arr[i] <= arr[L:R])

def first_rise_index(vy, start):
    n = len(vy)
    for r in range(max(start, EDGE_GUARD), n - MIN_RUN_UP):
        if np.all(vy[r+1:r+1+MIN_RUN_UP] < -VY_UP):
            return r
    return None

def best_index_from_mask(mask, score):
    idx = np.flatnonzero(mask)
    if idx.size:
        return int(idx[np.argmin(score[idx])])
    return None

def main():
    df = pd.read_csv(CSV_PATH).sort_values("frame").reset_index(drop=True)
    needed = {"frame","ball_x1","ball_y1","ball_x2","ball_y2",
              "player_most_overlap","player_x1","player_y1","player_x2","player_y2"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"CSV must contain: {sorted(list(needed))}")

    n = len(df)
    if n < 12:
        raise SystemExit("CSV too short.")

    frames = df["frame"].to_numpy()

    # --- Ball center (smoothed) ---
    cx = ((df["ball_x1"] + df["ball_x2"]) / 2.0).to_numpy()
    cy = ((df["ball_y1"] + df["ball_y2"]) / 2.0).to_numpy()
    cx = pd.Series(cx).interpolate(limit_direction="both").to_numpy()
    cy = pd.Series(cy).interpolate(limit_direction="both").to_numpy()
    cx_s = roll_mean(roll_med(cx, ROLL_MED), ROLL_MEAN)
    cy_s = roll_mean(roll_med(cy, ROLL_MED), ROLL_MEAN)

    vx = velocity(cx_s); vy = velocity(cy_s)
    speed = np.sqrt(vx*vx + vy*vy)

    # --- Player center / scale ---
    px = ((df["player_x1"] + df["player_x2"]) / 2.0).to_numpy()
    py = ((df["player_y1"] + df["player_y2"]) / 2.0).to_numpy()
    ph = (df["player_y2"] - df["player_y1"]).clip(lower=1.0).to_numpy()

    # --- Shooter & first-touch ---
    mid = n // 2
    shooter = modal_player(df.loc[mid:, "player_most_overlap"])
    if shooter == -1:
        shooter = modal_player(df["player_most_overlap"])
    smask = (df["player_most_overlap"].to_numpy() == shooter)

    ft_idx = first_sustained_touch(smask, SUST_TOUCH_WIN, SUST_TOUCH_MIN, start=0)
    if ft_idx is None:
        ap = np.flatnonzero(smask)
        ft_idx = int(ap[0]) if ap.size else EDGE_GUARD

    # --- Distances & scoring terms ---
    dist_px   = np.sqrt((cx_s - px)**2 + (cy_s - py)**2)
    dist_norm = dist_px / ph
    dist_scaled  = robust_scale_abs(dist_norm)
    speed_scaled = robust_scale_abs(speed)

    # --- Valid mask ---
    valid = np.ones(n, dtype=bool)
    valid[:EDGE_GUARD] = False
    valid[n-EDGE_GUARD:] = False
    valid &= (np.arange(n) >= (ft_idx + DEBOUNCE_AFTER_TOUCH))
    valid &= (df["player_most_overlap"].to_numpy() != -1)
    valid &= np.isfinite(px) & np.isfinite(py) & np.isfinite(ph)

    # Torso band
    y_ratio = (cy_s - df["player_y1"].to_numpy()) / ph
    torso_ok = (y_ratio >= TORSO_BAND[0]) & (y_ratio <= TORSO_BAND[1])
    valid &= torso_ok

    # Local minima (both)
    local_min_speed = np.array([is_local_min(speed,     i, LOCAL_MIN_WIN) for i in range(n)], dtype=bool)
    local_min_dist  = np.array([is_local_min(dist_norm, i, LOCAL_MIN_WIN) for i in range(n)], dtype=bool)

    # Post-touch percentile gates
    post_mask = (np.arange(n) >= (ft_idx + DEBOUNCE_AFTER_TOUCH))
    post_idx = np.flatnonzero(post_mask)
    if post_idx.size:
        d_post = dist_norm[post_idx]
        p_gate = np.nanpercentile(d_post, 100*DIST_PCTL_MAX)
        near_center = (dist_norm <= p_gate)
    else:
        near_center = np.ones(n, dtype=bool)

    # Enforce at/after global-min(dist) with small allow window
    if post_idx.size:
        i_min_d_global = int(post_idx[np.argmin(dist_norm[post_idx])])
    else:
        i_min_d_global = int(np.argmin(dist_norm))
    after_global_min = np.zeros(n, dtype=bool)
    Lg = i_min_d_global
    Rg = min(n, i_min_d_global + AFTER_GLOBAL_MIN_ALLOW + 1)
    after_global_min[Lg:Rg] = True

    # --- Stable hold run (speed low + near-center) ---
    # Use DIST_HOLD_PCTL on *all* frames to define "near-center" for the hold
    d_hold_gate = np.nanpercentile(dist_norm, 100*DIST_HOLD_PCTL)
    stable = (speed <= SPEED_HOLD_THRESH) & (dist_norm <= d_hold_gate)

    stable_run = np.zeros(n, dtype=bool)
    if POCKET_HOLD_RUN <= n:
        for i in range(0, n - POCKET_HOLD_RUN + 1):
            if np.all(stable[i:i+POCKET_HOLD_RUN]):
                stable_run[i:i+POCKET_HOLD_RUN] = True

    # --- Final candidate mask ---
    cand = (valid &
            near_center &
            after_global_min &
            local_min_speed &
            local_min_dist &
            stable_run)

    # --- Score & robust selection ---
    score = W_DIST * dist_scaled + W_VEL * speed_scaled

    def select(mask):
        idx = np.flatnonzero(mask)
        return int(idx[np.argmin(score[idx])]) if idx.size else None

    best_idx = select(cand)
    if best_idx is None:
        # relax 1: drop near_center (keep after_global_min + both locals + stable)
        best_idx = select(valid & after_global_min & local_min_speed & local_min_dist & stable_run)
    if best_idx is None:
        # relax 2: keep after_global_min & local_min_speed & stable
        best_idx = select(valid & after_global_min & local_min_speed & stable_run)
    if best_idx is None:
        # relax 3: valid & stable_run
        best_idx = select(valid & stable_run)
    if best_idx is None:
        # relax 4: valid only
        best_idx = select(valid)
    if best_idx is None:
        # edge-guarded global
        vg = np.ones(n, dtype=bool)
        vg[:EDGE_GUARD] = vg[n-EDGE_GUARD:] = False
        best_idx = select(vg)
    if best_idx is None:
        best_idx = int(np.argmin(score))

    pocket_idx    = int(best_idx)
    pocket_frame  = int(frames[pocket_idx])

    # --- Rise start after pocket ---
    rise_idx = first_rise_index(vy, start=pocket_idx)
    rise_frame = (int(frames[rise_idx]) if rise_idx is not None else None)

    # --- Diagnostics CSV ---
    out_scores = pd.DataFrame({
        "frame": frames,
        "valid": valid.astype(int),
        "cand": cand.astype(int),
        "local_min_speed": local_min_speed.astype(int),
        "local_min_dist": local_min_dist.astype(int),
        "near_center_pctl": near_center.astype(int),
        "after_global_min": after_global_min.astype(int),
        "stable": stable.astype(int),
        "stable_run": stable_run.astype(int),
        "ball_cx": cx_s, "ball_cy": cy_s,
        "player_cx": px, "player_cy": py,
        "player_h": ph,
        "y_ratio": y_ratio,
        "dist_px": dist_px, "dist_norm": dist_norm, "dist_scaled": dist_scaled,
        "speed": speed, "speed_scaled": speed_scaled,
        "score": score
    })
    out_scores.to_csv(OUT_SCORE_CSV, index=False)

    # --- Emit JSON ---
    # recompute/confirm first_touch_frame cleanly
    ft_confirm = first_sustained_touch(smask, SUST_TOUCH_WIN, SUST_TOUCH_MIN, start=0)
    first_touch_frame = int(frames[ft_confirm]) if ft_confirm is not None else int(frames[ft_idx])

    result = {
        "csv_path": CSV_PATH,
        "shooter_id": int(modal_player(df["player_most_overlap"])),
        "first_touch_frame": first_touch_frame,
        "pocket_frame": pocket_frame,
        "rise_start_frame": rise_frame,
        "pocket_idx_in_csv": pocket_idx,
        "pocket_center": {"cx": float(cx_s[pocket_idx]), "cy": float(cy_s[pocket_idx])},
        "params": {
            "ROLL_MED": ROLL_MED, "ROLL_MEAN": ROLL_MEAN,
            "EDGE_GUARD": EDGE_GUARD,
            "DEBOUNCE_AFTER_TOUCH": DEBOUNCE_AFTER_TOUCH,
            "SUST_TOUCH_WIN": SUST_TOUCH_WIN, "SUST_TOUCH_MIN": SUST_TOUCH_MIN,
            "TORSO_BAND": TORSO_BAND,
            "LOCAL_MIN_WIN": LOCAL_MIN_WIN,
            "DIST_PCTL_MAX": DIST_PCTL_MAX,
            "AFTER_GLOBAL_MIN_ALLOW": AFTER_GLOBAL_MIN_ALLOW,
            "POCKET_HOLD_RUN": POCKET_HOLD_RUN,
            "SPEED_HOLD_THRESH": SPEED_HOLD_THRESH,
            "DIST_HOLD_PCTL": DIST_HOLD_PCTL,
            "W_DIST": W_DIST, "W_VEL": W_VEL,
            "MIN_RUN_UP": MIN_RUN_UP, "VY_UP": VY_UP
        },
        "artifacts": {"score_csv": OUT_SCORE_CSV}
    }

    print("=== find_pocket result ===")
    print(json.dumps(result, indent=2))
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
