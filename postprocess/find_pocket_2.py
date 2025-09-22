import pandas as pd
import numpy as np

CSV_PATH       = "most_overall_edited.csv"

"""
TODO
This is the best file for predicting the shot pocket of the player
next steps would be to find the release point of the player, this is the last frame where the ball has overlap with the shooter
use the first catch frame and the shot pocket frame to find the deviation in the catch and the shot pocket
maybe do something where you calculate the average shot pocket of a shooter
"""

def find_pocket(csv_path):
    df = pd.read_csv(csv_path)

    # 1. Find most common value in 'player_overlap'
    mode_val = df['player_most_overlap'].mode()[0]

    # 2. Filter df to only rows with that value
    df_filt = df[df['player_most_overlap'] == mode_val]

    # 3. Compute centers
    ball_cx = (df_filt['ball_x1'] + df_filt['ball_x2']) / 2.0
    ball_cy = (df_filt['ball_y1'] + df_filt['ball_y2']) / 2.0
    player_cx = (df_filt['player_x1'] + df_filt['player_x2']) / 2.0
    player_cy = (df_filt['player_y1'] + df_filt['player_y2']) / 2.0

    dist = np.hypot(player_cx - ball_cx, player_cy - ball_cy)

    # 4. Find frame with min distance
    min_idx = dist.idxmin()
    frame_id = df_filt.loc[min_idx, 'frame'] if 'frame' in df_filt.columns else min_idx

    print(f"Mode player_overlap: {mode_val}")
    print(f"Closest frame: {frame_id}, distance: {float(dist.loc[min_idx]):.2f}")

    return frame_id, float(dist.loc[min_idx]), mode_val



if __name__ == '__main__':
    id, float_id, mode = find_pocket(CSV_PATH)