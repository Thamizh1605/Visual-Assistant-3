# src/navigation.py
"""
navigation.py
- analyze_frame_for_navigation(frame) returns a spoken guidance string like:
  "Clear path ahead. Chair on the left. Move slightly right."
- Implementation uses: object detection (YOLO if available), otherwise image heuristics.
"""

import numpy as np
import cv2
from typing import List, Tuple

# Try to use ultralytics YOLO if available
try:
    from ultralytics import YOLO
    _has_yolo = True
except Exception:
    _has_yolo = False

# Optional local model path (if you've downloaded a small yolo or custom)
YOLO_LOCAL = "models/yolov8n.pt"

_yolo_model = None
if _has_yolo:
    try:
        _yolo_model = YOLO(YOLO_LOCAL) if __import__("pathlib").Path(YOLO_LOCAL).exists() else None
    except Exception as e:
        print("[navigation] YOLO load failed:", e)
        _yolo_model = None

def analyze_frame_for_navigation(frame: np.ndarray, yolo_model=_yolo_model) -> str:
    """
    Given BGR frame (numpy), return a short navigation guidance sentence.
    Strategy:
      1. Run detector -> get bounding boxes and labels.
      2. Partition frame into left/center/right vertical slices.
      3. If big obstacle appears in center and close (large bbox), say 'Obstacle ahead'.
      4. If left side crowded, recommend move right, and vice versa.
    """
    h, w = frame.shape[:2]
    left_count = 0
    center_count = 0
    right_count = 0
    close_center = False
    found_labels = []

    # Run YOLO if available
    if yolo_model is not None:
        try:
            results = yolo_model(frame, imgsz=640, conf=0.25, verbose=False)
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(float, xyxy)
                    cx = (x1 + x2) / 2.0
                    area = (x2 - x1) * (y2 - y1)
                    if cx < w/3:
                        left_count += 1
                    elif cx < 2*w/3:
                        center_count += 1
                        if area / (w*h) > 0.10:  # big center object -> close
                            close_center = True
                    else:
                        right_count += 1
                    # label resolution (depends on model.names)
                    try:
                        cls_idx = int(b.cls[0].cpu().numpy())
                        label = yolo_model.names.get(cls_idx, str(cls_idx))
                    except Exception:
                        label = "object"
                    found_labels.append((label, (x1,y1,x2,y2)))
        except Exception as e:
            print("[navigation] YOLO detection error:", e)

    # If no detector, use simple intensity heuristics/saliency (rough)
    if not found_labels:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # compute thresholded regions
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # count nonzeros in left/center/right
        left_count = int(np.count_nonzero(th[:, :w//3]) / 1000)
        center_count = int(np.count_nonzero(th[:, w//3:2*w//3]) / 1000)
        right_count = int(np.count_nonzero(th[:, 2*w//3:]) / 1000)

    # Compose guidance
    guidance = []
    if center_count == 0 and left_count == 0 and right_count == 0:
        guidance.append("Clear path ahead.")
    else:
        if close_center:
            guidance.append("Obstacle ahead. Stop and check.")
        else:
            if center_count > max(left_count, right_count):
                guidance.append("Some objects ahead, but path may be passable.")
            # direction advice
            if left_count > right_count:
                guidance.append("Left side has obstacles. Keep to the right slightly.")
            elif right_count > left_count:
                guidance.append("Right side has obstacles. Keep to the left slightly.")
            else:
                guidance.append("Obstacles at both sides. Use caution and proceed slowly.")

    # Add short list of main objects detected (first few)
    if found_labels:
        top = ", ".join([lbl for lbl, _ in found_labels[:4]])
        guidance.append(f"Detected: {top}.")

    return " ".join(guidance)
