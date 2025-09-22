import pandas as pd
import numpy as np
from find_pocket_2 import find_pocket

CSV_PATH       = "most_overall_edited.csv"

import pandas as pd
import numpy as np

def find_highest_point(csv_path: str):
    """
    Return (frame_id, min_abs_delta, mode_player_id), where:
      abs_delta = abs(ball_y2 - player_y1)
    computed only on rows where:
      - player_most_overlap == modal player,
      - frame >= shot-pocket frame (from find_pocket),
      - Euclidean center distance(ball, player) < player_height (player_y2 - player_y1).
    """
    df = pd.read_csv(csv_path)

    # Required columns
    required = {
        "frame", "player_most_overlap",
        "ball_x1","ball_y1","ball_x2","ball_y2",
        "player_x1","player_y1","player_x2","player_y2"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # 0) Shot-pocket frame
    pocket_frame, _, _ = find_pocket(csv_path)  # must return (frame_id, dist, mode)

    # 1) Modal player
    s = df["player_most_overlap"].dropna()
    if s.empty:
        raise ValueError("No valid player_most_overlap values.")
    mode_val = int(s.mode()[0])

    # 2) Filter rows
    df_filt = df[
        (df["player_most_overlap"] == mode_val) &
        (df["frame"] >= int(pocket_frame))
    ].copy()

    # Drop NaNs we need
    df_filt = df_filt.dropna(subset=[
        "ball_x1","ball_y1","ball_x2","ball_y2",
        "player_x1","player_y1","player_x2","player_y2","frame"
    ])
    if df_filt.empty:
        raise ValueError(f"No rows for modal player {mode_val} at/after pocket frame {pocket_frame} with valid coords.")

    # 3) Compute centers and height
    ball_cx = (df_filt["ball_x1"] + df_filt["ball_x2"]) / 2.0
    ball_cy = (df_filt["ball_y1"] + df_filt["ball_y2"]) / 2.0
    player_cx = (df_filt["player_x1"] + df_filt["player_x2"]) / 2.0
    player_cy = (df_filt["player_y1"] + df_filt["player_y2"]) / 2.0
    player_h = (df_filt["player_y2"] - df_filt["player_y1"]).abs()

    # Euclidean distance between centers
    center_dist = np.hypot(ball_cx - player_cx, ball_cy - player_cy)

    # 4) Keep only frames where center distance < player height
    valid_mask = center_dist < player_h
    df_filt = df_filt.loc[valid_mask].copy()
    if df_filt.empty:
        raise ValueError(
            f"No frames meet center-distance constraint after pocket frame "
            f"(dist < player_height) for modal player {mode_val}."
        )

    # 5) Minimize vertical separation (ball highest relative to player)
    abs_delta = (df_filt["ball_y2"] - df_filt["player_y1"]).abs()

    min_val = float(abs_delta.min())
    best_rows = df_filt.loc[abs_delta == min_val]
    best_row = best_rows.sort_values("frame").iloc[0]
    frame_id = int(best_row["frame"])

    print(f"Shot-pocket frame: {int(pocket_frame)}")
    print(f"Mode player_most_overlap: {mode_val}")
    print(f"Constraint: center_dist < player_height satisfied.")
    print(f"Highest after pocket (min |ball_y2 - player_y1|): frame={frame_id}, value={min_val:.2f}")

    return frame_id, min_val, mode_val


if __name__ == '__main__':
    frame_id, min_value, model_value = find_highest_point(CSV_PATH)