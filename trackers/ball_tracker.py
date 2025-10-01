from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
# from utils import read_stub, save_stub
from utils.ema_smoothing import EMABoxSmoother
from utils import numpy_core_shim

class BallTracker:
    """
    A class that handles basketball detection and tracking using YOLO.

    This class provides methods to detect the ball in video frames, process detections
    in batches, and refine tracking results through filtering and interpolation.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # self.ball_smoother = EMABoxSmoother(alpha_pos=0.5, alpha_size=0.3)
        self.last_bbox_xyxy = None

        # === CHANGED: infer ball class ids from model names (supports 'Ball' OR 'Basketball') ===
        self.ball_class_ids = self._infer_ball_class_ids()
        print(f"[BallTracker] using ball class ids: {sorted(self.ball_class_ids)}")

    # === ADDED: helper to get class ids for ball/basketball ===
    def _infer_ball_class_ids(self):
        try:
            names = getattr(self.model, "names", None) or getattr(self.model.model, "names", None)
            # normalize to {int_id: 'lowername'}
            names = {int(k): str(v).lower() for k, v in dict(names).items()}
        except Exception:
            names = {}

        # match common variants
        ball_ids = [k for k, v in names.items() if v in ("ball", "basketball")]
        # single-class fallback → assume 0
        if not ball_ids and len(names) == 1:
            ball_ids = [list(names.keys())[0]]
        return set(ball_ids)

    def detect_frames(self, frames):
        """
        Detect the ball in a sequence of frames using batch processing.

        Args:
            frames (list): List of video frames to process.

        Returns:
            list: YOLO detection results for each frame.
        """
        batch_size = 20
        detections = []

        # === CHANGED: use inferred class ids instead of hard-coded classes=[0] ===
        classes_arg = list(self.ball_class_ids) if self.ball_class_ids else None

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.25,
                classes=classes_arg,   # CHANGED
                max_det=1,
                verbose=False
            )
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Get ball tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing ball tracking information for each frame.
        """
        # tracks = read_stub(read_from_stub, stub_path)
        # if tracks is not None and len(tracks) == len(frames):
        #     return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            # === CHANGED: avoid assuming 'Ball' key exists; we already filtered classes in detect ===
            det_sup: sv.Detections = sv.Detections.from_ultralytics(detection)

            tracks.append({})
            chosen_bbox = None
            max_confidence = -1.0

            # Each det_sup row: (xyxy, mask, confidence, class_id, tracker_id, data)
            for det in det_sup:
                bbox = det[0].tolist()          # xyxy
                conf = float(det[2])
                cls_id = int(det[3])

                # === CHANGED: filter by inferred class ids ===
                if (not self.ball_class_ids) or (cls_id in self.ball_class_ids):
                    if conf > max_confidence:
                        chosen_bbox = bbox
                        max_confidence = conf

            if chosen_bbox is not None:
                tracks[frame_num][1] = {"bbox": chosen_bbox}

        # save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_positions):
        """
        Filter out incorrect ball detections based on maximum allowed movement distance.

        Args:
            ball_positions (list): List of detected ball positions across frames.

        Returns:
            list: Filtered ball positions with incorrect detections removed.
        """
        maximum_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            current_box = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_box) == 0:
                continue

            if last_good_frame_index == -1:
                # First valid detection
                last_good_frame_index = i
                continue

            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
                ball_positions[i] = {}
            else:
                last_good_frame_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball positions to create smooth tracking results.

        Args:
            ball_positions (list): List of ball positions with potential gaps.

        Returns:
            list: List of ball positions with interpolated values filling the gaps.
        """
        # === CHANGED: guard empty detections to avoid pandas "0 columns" crash ===
        bboxes = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        any_non_empty = any(len(bb) == 4 for bb in bboxes)
        if not any_non_empty:
            print("[BallTracker] No ball detections found; skipping interpolation.")
            return ball_positions  # CHANGED: early return

        # Normalize: convert non-4 boxes to NaNs so DataFrame has 4 columns
        norm_bboxes = []
        for bb in bboxes:
            if len(bb) == 4:
                norm_bboxes.append(bb)
            else:
                norm_bboxes.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(norm_bboxes, columns=['x1', 'y1', 'x2', 'y2'])  # CHANGED: safe build

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions



# class BallTracker:
#     """
#     A class that handles basketball detection and tracking using YOLO.

#     This class provides methods to detect the ball in video frames, process detections
#     in batches, and refine tracking results through filtering and interpolation.
#     """
#     def __init__(self, model_path):
#         self.model = YOLO(model_path) 
#         # self.ball_smoother = EMABoxSmoother(alpha_pos=0.5, alpha_size=0.3)
#         self.last_bbox_xyxy = None

#     def detect_frames(self, frames):
#         """
#         Detect the ball in a sequence of frames using batch processing.

#         Args:
#             frames (list): List of video frames to process.

#         Returns:
#             list: YOLO detection results for each frame.
#         """
#         batch_size=20 
#         detections = [] 
#         for i in range(0,len(frames),batch_size):
#             detections_batch = self.model.predict(
#                 frames[i:i+batch_size],
#                 conf=.25,
#                 classes=[0],
#                 max_det=1,
#                 verbose=False
#             )
#             detections += detections_batch
#         return detections

#     def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
#         """
#         Get ball tracking results for a sequence of frames with optional caching.

#         Args:
#             frames (list): List of video frames to process.
#             read_from_stub (bool): Whether to attempt reading cached results.
#             stub_path (str): Path to the cache file.

#         Returns:
#             list: List of dictionaries containing ball tracking information for each frame.
#         """
#         # tracks = read_stub(read_from_stub,stub_path)
#         # if tracks is not None:
#         #     if len(tracks) == len(frames):
#         #         return tracks

#         detections = self.detect_frames(frames)
#         tracks=[]

#         for frame_num, detection in enumerate(detections):
#             cls_names = detection.names
#             cls_names_inv = {v:k for k,v in cls_names.items()}

#             # Covert to supervision Detection format
#             detection_supervision = sv.Detections.from_ultralytics(detection)

#             tracks.append({})
#             chosen_bbox =None
#             max_confidence = 0
            
#             for frame_detection in detection_supervision:
#                 bbox = frame_detection[0].tolist()
#                 cls_id = frame_detection[3]
#                 confidence = frame_detection[2]
                
#                 if cls_id == cls_names_inv['Ball']:
#                     if max_confidence<confidence:
#                         chosen_bbox = bbox
#                         max_confidence = confidence

#             if chosen_bbox is not None:
#                 tracks[frame_num][1] = {"bbox":chosen_bbox}

#         # save_stub(stub_path,tracks)
        
#         return tracks

#     # def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
#     #     """
#     #     Returns: list[dict], where each element is {1: {"bbox": [x1,y1,x2,y2]}} or {} if none.
#     #     Applies EMA smoothing frame-by-frame.
#     #     """
#     #     # tracks = read_stub(read_from_stub, stub_path)
#     #     # if tracks is not None and len(tracks) == len(frames):
#     #     #     return tracks

#     #     detections = self.detect_frames(frames)
#     #     tracks = []

#     #     for frame_num, detection in enumerate(detections):
#     #         # convert to supervision Detections
#     #         det_sup: sv.Detections = sv.Detections.from_ultralytics(detection)

#     #         # pick the best ball (we already set max_det=1, but keep this robust)
#     #         chosen_bbox = None
#     #         chosen_conf = -1.0

#     #         for det in det_sup:
#     #             bbox = det[0].tolist()     # xyxy
#     #             conf = float(det[2])
#     #             if conf > chosen_conf:
#     #                 chosen_bbox = bbox
#     #                 chosen_conf = conf

#     #         # Smooth / handle miss
#     #         if chosen_bbox is not None:
#     #             smoothed = self.ball_smoother.update(chosen_bbox)     # EMA here
#     #             self._last_box_xyxy = smoothed
#     #             tracks.append({1: {"bbox": smoothed}})
#     #         else:
#     #             # no detection this frame: carry forward last smoothed box (no decay)
#     #             self.ball_smoother.mark_missed(decay=0.0)
#     #             if self.ball_smoother.state:
#     #                 cx, cy, w, h = self.ball_smoother.state
#     #                 x1, y1 = int(cx - w * 0.5), int(cy - h * 0.5)
#     #                 x2, y2 = int(cx + w * 0.5), int(cy + h * 0.5)
#     #                 carry = [x1, y1, x2, y2]
#     #                 self._last_box_xyxy = carry
#     #                 tracks.append({1: {"bbox": carry}})
#     #             else:
#     #                 tracks.append({})  # truly nothing yet

#     #     # if stub_path:
#     #     #     save_stub(stub_path, tracks)
#     #     return tracks

#     def remove_wrong_detections(self,ball_positions):
#         """
#         Filter out incorrect ball detections based on maximum allowed movement distance.

#         Args:
#             ball_positions (list): List of detected ball positions across frames.

#         Returns:
#             list: Filtered ball positions with incorrect detections removed.
#         """
        
#         maximum_allowed_distance = 25
#         last_good_frame_index = -1

#         for i in range(len(ball_positions)):
#             current_box = ball_positions[i].get(1, {}).get('bbox', [])

#             if len(current_box) == 0:
#                 continue

#             if last_good_frame_index == -1:
#                 # First valid detection
#                 last_good_frame_index = i
#                 continue

#             last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
#             frame_gap = i - last_good_frame_index
#             adjusted_max_distance = maximum_allowed_distance * frame_gap

#             if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_box[:2])) > adjusted_max_distance:
#                 ball_positions[i] = {}
#             else:
#                 last_good_frame_index = i

#         return ball_positions

#     def interpolate_ball_positions(self,ball_positions):
#         """
#         Interpolate missing ball positions to create smooth tracking results.

#         Args:
#             ball_positions (list): List of ball positions with potential gaps.

#         Returns:
#             list: List of ball positions with interpolated values filling the gaps.
#         """
#         ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
#         df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

#         # Interpolate missing values
#         df_ball_positions = df_ball_positions.interpolate()
#         df_ball_positions = df_ball_positions.bfill()

#         ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]
#         return ball_positions