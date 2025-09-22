CSV_IN  = "output/tracks.csv"
CSV_OUT = "most_overlap.csv"

import pandas as pd
import numpy as np

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
    (_, _, _, _), inter = intersect(a, b)
    A = area(*a); B = area(*b)
    denom = A + B - inter
    return inter / denom if denom > 0 else 0.0

def main():
    df = pd.read_csv(CSV_IN)
    if not REQUIRED.issubset(df.columns):
        raise SystemExit(f"Input must have columns {sorted(REQUIRED)}; got {df.columns.tolist()}")

    # Ensure types
    df["frame"] = df["frame"].astype(int)
    df["id"]    = df["id"].astype(int, errors="ignore")

    max_frame = int(df["frame"].max())
    frames = list(range(0, max_frame+1))

    # Split detections
    balls   = df[df["type"].astype(str).str.lower()=="ball"].copy()
    players = df[df["type"].astype(str).str.lower()!="ball"].copy()

    # Keep largest ball per frame
    balls["area"] = (balls["x2"]-balls["x1"]).clip(lower=0) * (balls["y2"]-balls["y1"]).clip(lower=0)
    balls = (balls.sort_values(["frame","area"], ascending=[True, False])
                  .drop_duplicates(subset=["frame"], keep="first"))

    balls_by_frame = {
        int(r.frame):(float(r.x1),float(r.y1),float(r.x2),float(r.y2))
        for _,r in balls.iterrows()
    }

    # Map: frame -> list[(pid, (x1,y1,x2,y2))]
    players_by_frame = {}
    for f, grp in players.groupby("frame"):
        items = []
        for _, r in grp.iterrows():
            items.append( (int(r.id), (float(r.x1),float(r.y1),float(r.x2),float(r.y2))) )
        players_by_frame[int(f)] = items

    # Also build quick lookup: (frame, pid) -> box  (for the overwrite stage)
    player_box_at = {}
    for f, items in players_by_frame.items():
        for pid, box in items:
            player_box_at[(f, pid)] = box

    # ---------- Build initial most-overlap table ----------
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
            score = iou(ball_box, pbox)  # or intersect(ball_box, pbox)[1] for raw intersection area
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

    from collections import defaultdict

    iou_sum_by_player = defaultdict(float)
    inter_sum_by_player = defaultdict(float)

    for f in frames:
        ball_box = balls_by_frame.get(f)
        if ball_box is None:
            continue  # no ball this frame, skip

        for pid, pbox in players_by_frame.get(f, []):
            # IoU
            iou_val = iou(ball_box, pbox)
            iou_sum_by_player[pid] += float(iou_val)

            # Raw intersection area
            _, inter_area = intersect(ball_box, pbox)
            inter_sum_by_player[pid] += float(inter_area)

    def top_key(d):
        return max(d.items(), key=lambda kv: kv[1])[0] if d else -1

    top_iou_pid = top_key(iou_sum_by_player)
    top_iou_val = iou_sum_by_player.get(top_iou_pid, 0.0)

    top_inter_pid = top_key(inter_sum_by_player)
    top_inter_val = inter_sum_by_player.get(top_inter_pid, 0.0)

    print(f"[global] top_cumulative_iou: player={top_iou_pid}, total_iou={top_iou_val:.6f}")
    print(f"[global] top_cumulative_intersection_area: player={top_inter_pid}, total_area={top_inter_val:.2f}")

    out_df = pd.DataFrame(out_rows, columns=[
        "frame",
        "ball_x1","ball_y1","ball_x2","ball_y2",
        "player_most_overlap",
        "player_x1","player_y1","player_x2","player_y2"
    ])

    # ---------- Determine halves & dominant players ----------
    # Halfway by index into `frames` to keep halves balanced even if frames don't start at 0.
    half_index = len(frames)//2
    split_frame = frames[half_index]  # frames < split_frame => first half; >= split_frame => second half

    first_half = out_df[out_df["frame"] < split_frame]
    second_half = out_df[out_df["frame"] >= split_frame]

    def modal_player(s):
        s2 = s[(s != -1) & (~pd.isna(s))]
        return int(s2.value_counts().idxmax()) if not s2.empty else -1

    first_modal  = modal_player(first_half["player_most_overlap"])
    second_modal = modal_player(second_half["player_most_overlap"])

    # If we couldn't find a valid second-half modal, just save and exit.
    if second_modal == -1:
        out_df.to_csv(CSV_OUT, index=False)
        print(f"[OK] wrote {CSV_OUT} (no second-half modal found; no overwrite applied)")
        return

    # Find the first frame where the second-half modal player appears at all
    first_touch_rows = out_df[out_df["player_most_overlap"] == second_modal]
    if first_touch_rows.empty:
        # Never appears — nothing to overwrite
        out_df.to_csv(CSV_OUT, index=False)
        print(f"[OK] wrote {CSV_OUT} (player {second_modal} never appears; no overwrite applied)")
        return

    switch_frame = int(first_touch_rows["frame"].iloc[0])

    # ---------- From switch_frame onward, force player to `second_modal` ----------
    # For each frame >= switch_frame, replace the player columns with that player's actual box
    mask = out_df["frame"] >= switch_frame
    forced_ids = []
    forced_boxes = {"player_x1": [], "player_y1": [], "player_x2": [], "player_y2": []}

    for f in out_df.loc[mask, "frame"]:
        box = player_box_at.get((int(f), second_modal))
        if box is None:
            # keep ID but write NaNs for coords if the player has no box on this frame
            forced_ids.append(second_modal)
            forced_boxes["player_x1"].append(np.nan)
            forced_boxes["player_y1"].append(np.nan)
            forced_boxes["player_x2"].append(np.nan)
            forced_boxes["player_y2"].append(np.nan)
        else:
            forced_ids.append(second_modal)
            forced_boxes["player_x1"].append(box[0])
            forced_boxes["player_y1"].append(box[1])
            forced_boxes["player_x2"].append(box[2])
            forced_boxes["player_y2"].append(box[3])

    out_df.loc[mask, "player_most_overlap"] = forced_ids
    out_df.loc[mask, ["player_x1","player_y1","player_x2","player_y2"]] = pd.DataFrame(forced_boxes, index=out_df.index[mask])

    # (Optional) if you also want to force the first half to the first modal, uncomment below:
    # mask_first = out_df["frame"] < switch_frame
    # for f in out_df.loc[mask_first, "frame"]:
    #     box = player_box_at.get((int(f), first_modal))
    #     if box is not None:
    #         out_df.loc[out_df["frame"] == f, ["player_most_overlap","player_x1","player_y1","player_x2","player_y2"]] = \
    #             [first_modal, box[0], box[1], box[2], box[3]]

    out_df.to_csv(CSV_OUT, index=False)
    print(f"[OK] wrote {CSV_OUT} (first_modal={first_modal}, second_modal={second_modal}, switch_frame={switch_frame})")

if __name__ == "__main__":
    main()
