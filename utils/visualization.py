"""
Visualization Utility Module for OBJECT VISION.
Renders high-visibility bounding boxes, class labels, tracking IDs,
motion trajectory trails, and an unobtrusive in-frame HUD.
"""

from typing import Dict, List, Optional, Tuple
import colorsys
import cv2
import numpy as np


_COLOR_CACHE: Dict[int, Tuple[int, int, int]] = {}

# Carefully tuned high-contrast distinct BGR colors for multi-object tracking
DISTINCT_COLORS = [
    (248, 189, 56),   # Sky Blue #38BDF8 (BGR: 248, 189, 56)
    (74, 222, 128),   # Green #4ADE80
    (248, 140, 129),  # Indigo #818CF8
    (0, 165, 255),    # Amber/Orange
    (238, 130, 238),  # Violet
    (0, 230, 230),    # Yellow
    (204, 50, 153),   # Magenta
    (0, 215, 255),    # Gold
    (180, 105, 255),  # Hot Pink
    (128, 255, 0),    # Chartreuse
    (255, 191, 0),    # Deep Sky
    (0, 250, 154),    # Spring Green
    (255, 105, 180),  # Pink
    (60, 179, 113),   # Sea Green
    (0, 206, 209),    # Turquoise
    (255, 140, 0),    # Orange
    (147, 112, 219),  # Purple
    (220, 20, 60),    # Crimson
    (255, 215, 0),    # Amber
    (75, 0, 130)      # Indigo
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Returns a deterministic, vibrant, visually distinct BGR color for each track ID."""
    if track_id in _COLOR_CACHE:
        return _COLOR_CACHE[track_id]

    if 0 <= track_id < len(DISTINCT_COLORS):
        bgr = DISTINCT_COLORS[track_id]
    else:
        golden_ratio_conjugate = 0.618033988749895
        h = (track_id * golden_ratio_conjugate) % 1.0
        s = 0.90
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        bgr = (int(b * 255), int(g * 255), int(r * 255))

    _COLOR_CACHE[track_id] = bgr
    return bgr


def resize_frame(frame: np.ndarray, target_width: int = 0) -> np.ndarray:
    """
    Resizes a frame to target_width while preserving original aspect ratio.
    If target_width <= 0, returns the complete original frame unaltered.
    """
    if target_width <= 0 or frame is None:
        return frame

    h, w = frame.shape[:2]
    if w <= target_width:
        return frame

    scale = target_width / float(w)
    new_w = target_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_bounding_boxes(
    frame: np.ndarray,
    tracked_objects: List[any],
    show_trajectories: bool = True
) -> np.ndarray:
    """
    Draws professional, high-contrast bounding boxes, standardized labels,
    and motion trajectory trails.
    
    Label format:
    CLASS | ID: 01 | 94%
    """
    h_img, w_img = frame.shape[:2]

    for obj in tracked_objects:
        track_id = int(obj.track_id)
        x1, y1, x2, y2 = obj.bbox
        cls_name = str(obj.class_name).upper()
        conf = float(obj.confidence)
        conf_pct = int(round(max(0.0, min(1.0, conf)) * 100.0))
        color = get_track_color(track_id)

        # Coordinate safety clamping
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        # 1. Trajectory motion trail
        if show_trajectories and hasattr(obj, "trajectory") and len(obj.trajectory) > 1:
            pts = obj.trajectory
            num_pts = len(pts)
            for j in range(1, num_pts):
                thick = max(1, int((j / num_pts) * 3))
                cv2.line(frame, pts[j - 1], pts[j], color, thick, lineType=cv2.LINE_AA)
            cx, cy = pts[-1]
            cv2.circle(frame, (cx, cy), 3, color, -1, lineType=cv2.LINE_AA)

        # 2. Dual-layer Bounding Box (Dark outline + Color)
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (11, 16, 32), 3, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

        # 3. Corner Brackets
        corner_w = min(14, int((x2 - x1) * 0.20))
        corner_h = min(14, int((y2 - y1) * 0.20))
        if corner_w > 4 and corner_h > 4:
            thick = 3
            cv2.line(frame, (x1, y1), (x1 + corner_w, y1), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_h), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2 - corner_w, y1), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_h), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1 + corner_w, y2), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_h), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - corner_w, y2), color, thick, lineType=cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2 - corner_h, y2), color, thick, lineType=cv2.LINE_AA)

        # 4. Standardized Label: CLASS | ID: 01 | 94%
        label_text = f"{cls_name} | ID: {track_id:02d} | {conf_pct}%"
        font_scale = 0.44
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
        )

        badge_h = txt_h + baseline + 8
        badge_w = txt_w + 12

        if y1 - badge_h >= 0:
            by1 = y1 - badge_h
            by2 = y1
        else:
            by1 = y1
            by2 = y1 + badge_h

        bx1 = max(0, x1)
        bx2 = min(w_img - 1, bx1 + badge_w)

        # Dark solid pill with color accent border
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (16, 31, 50), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1, lineType=cv2.LINE_AA)

        text_y = by1 + txt_h + 4
        text_x = bx1 + 6
        cv2.putText(
            frame,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (248, 250, 252),
            font_thick,
            cv2.LINE_AA
        )

    return frame


def draw_tracking_hud(
    frame: np.ndarray,
    fps: float,
    active_tracks: int,
    total_unique: int,
    is_detection_frame: bool,
    inference_ms: float = 0.0
) -> np.ndarray:
    """Draws a sleek, semi-transparent top HUD banner matching the #0B1020 theme."""
    h, w = frame.shape[:2]
    hud_h = 32

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (11, 16, 32), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Accent underline
    cv2.line(frame, (0, hud_h), (w, hud_h), (39, 52, 73), 1, lineType=cv2.LINE_AA)

    mode_text = "YOLO DETECT" if is_detection_frame else "KALMAN TRACK"
    mode_color = (74, 222, 128) if is_detection_frame else (248, 189, 56)

    cv2.circle(frame, (16, int(hud_h / 2)), 4, mode_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        mode_text,
        (26, int(hud_h / 2) + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        mode_color,
        1,
        cv2.LINE_AA
    )

    hud_str = f"FPS: {fps:4.1f}   Tracks: {active_tracks}   Total: {total_unique}   Lat: {inference_ms:4.1f}ms"
    cv2.putText(
        frame,
        hud_str,
        (145, int(hud_h / 2) + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (248, 250, 252),
        1,
        cv2.LINE_AA
    )

    return frame
