"""
SORT: Simple Online and Realtime Tracking with Class-Aware Association
Based on Alex Bewley et al. (2016) - https://arxiv.org/abs/1602.00763

Enhanced with:
1. Linear Kalman Filter (7-state motion model: [u, v, s, r, u_dot, v_dot, s_dot]^T).
2. Hungarian Algorithm (scipy.optimize.linear_sum_assignment).
3. Class-Aware Bipartite Matching: Prevents identity hijacking across distinct classes.
4. Duplicate Track Suppression: Rejects redundant overlapping candidate tracks.
5. Persistent ID Life-Cycle: Preserves confirmed tracks smoothly across skipped frames.
6. Temporal Disappearance Handling (graceful max_age track retention).
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box [x1, y1, x2, y2] and returns observation vector z:
    [u, v, s, r]^T where:
    - u, v: center coordinate
    - s: scale/area (w * h)
    - r: aspect ratio (w / h)
    """
    b = np.asarray(bbox, dtype=np.float32).flatten()
    w = max(1.0, float(b[2] - b[0]))
    h = max(1.0, float(b[3] - b[1]))
    x = float(b[0] + w / 2.0)
    y = float(b[1] + h / 2.0)
    s = float(w * h)
    r = float(w / h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float = None) -> np.ndarray:
    """
    Takes state vector x [u, v, s, r, ...] and returns bounding box [x1, y1, x2, y2].
    Always returns shape (1, 4) or (1, 5) if score is provided.
    """
    x_flat = np.asarray(x, dtype=np.float32).flatten()
    u, v, s, r = float(x_flat[0]), float(x_flat[1]), float(x_flat[2]), float(x_flat[3])

    # Prevent negative values under square root
    w = np.sqrt(max(1.0, s * max(1e-4, r)))
    h = float(s / w) if w > 0.0 else 1.0
    x1 = u - w / 2.0
    y1 = v - h / 2.0
    x2 = u + w / 2.0
    y2 = v + h / 2.0

    if score is None:
        return np.array([[x1, y1, x2, y2]], dtype=np.float32)
    return np.array([[x1, y1, x2, y2, float(score)]], dtype=np.float32)


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    bb_test: (N, 4) array [[x1, y1, x2, y2], ...]
    bb_gt: (M, 4) array [[x1, y1, x2, y2], ...]
    Returns (N, M) array of IoU values.
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty((len(bb_test), len(bb_gt)), dtype=np.float32)

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = np.maximum(0.0, (bb_test[..., 2] - bb_test[..., 0])) * np.maximum(0.0, (bb_test[..., 3] - bb_test[..., 1]))
    area_gt = np.maximum(0.0, (bb_gt[..., 2] - bb_gt[..., 0])) * np.maximum(0.0, (bb_gt[..., 3] - bb_gt[..., 1]))
    union = area_test + area_gt - intersection

    iou = np.where(union > 0, intersection / union, 0.0)
    return iou.astype(np.float32)


class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    Uses a 7-dimensional state space: [u, v, s, r, u_dot, v_dot, s_dot]^T
    where (u, v) is center, s is scale (area), r is aspect ratio, and dots are velocities.
    """
    count = 0

    def __init__(self, bbox: np.ndarray, class_id: int = 0, class_name: str = "object", confidence: float = 1.0):
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix F (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement function H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)

        # Covariance matrix P
        self.P = np.eye(self.dim_x, dtype=np.float32)
        self.P[4:, 4:] *= 1000.0  # High initial velocity uncertainty
        self.P *= 10.0

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.Q[-1, -1] *= 0.01

        # Initialize state with observation
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.x[:4] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        # Metadata
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.confidence = float(confidence)

    def update(self, bbox: np.ndarray, class_id: int = None, class_name: str = None, confidence: float = None):
        """Updates the state vector with observed bounding box using Kalman measurement update."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        if class_id is not None:
            self.class_id = int(class_id)
        if class_name is not None:
            self.class_name = str(class_name)
        if confidence is not None:
            # Smooth confidence score with exponential moving average
            self.confidence = 0.7 * float(confidence) + 0.3 * self.confidence

        z = convert_bbox_to_z(bbox)
        # Kalman Update equations:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def predict(self) -> np.ndarray:
        """Advances the state vector and returns the predicted bounding box estimate."""
        if (self.x[2, 0] + self.x[6, 0]) <= 0:
            self.x[6, 0] = 0.0

        # Kalman Prediction equations:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """Returns the current bounding box estimate of shape (1, 4)."""
        return convert_x_to_bbox(self.x)


def associate_detections_to_trackers(
    detections: np.ndarray,
    trackers: np.ndarray,
    det_classes: np.ndarray,
    trk_classes: np.ndarray,
    iou_threshold: float = 0.3
):
    """
    Class-Aware Hungarian Bipartite Matching:
    Associates detections to trackers with class consistency enforcement.
    Cross-class matches are rejected by nullifying IoU overlap for different classes.
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    iou_matrix = iou_batch(detections, trackers)

    # Enforce Class Consistency: Disallow matching between different object classes
    if len(det_classes) > 0 and len(trk_classes) > 0:
        for d in range(len(det_classes)):
            for t in range(len(trk_classes)):
                if int(det_classes[d]) != int(trk_classes[t]):
                    iou_matrix[d, t] = 0.0

    if min(iou_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches below IoU threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)


class Sort:
    """
    Main SORT Tracker Manager with Class-Aware Matching and Duplicate Suppression.
    Supports periodic detection and smooth Kalman prediction across skipped frames.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray = np.empty((0, 6))) -> np.ndarray:
        """
        Process a new frame with detections.
        dets: [[x1, y1, x2, y2, score, class_id], ...]
        Returns:
        np.ndarray of active tracked objects:
        [[x1, y1, x2, y2, track_id, class_id, score], ...]
        """
        self.frame_count += 1

        # 1. Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        trk_classes = np.zeros((len(self.trackers),), dtype=int)
        to_del = []
        ret = []
        for t in range(len(self.trackers)):
            pos = self.trackers[t].predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]
            trk_classes[t] = self.trackers[t].class_id
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. Extract coordinates and classes
        det_boxes = dets[:, :4] if len(dets) > 0 else np.empty((0, 4), dtype=np.float32)
        det_classes = dets[:, 5].astype(int) if len(dets) > 0 and dets.shape[1] > 5 else np.zeros((len(det_boxes),), dtype=int)

        # 3. Class-Aware Bipartite Association
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            det_boxes, trks, det_classes, trk_classes, self.iou_threshold
        )

        # 4. Update matched trackers
        for m in matched:
            det_idx, trk_idx = int(m[0]), int(m[1])
            det_box = dets[det_idx, :4]
            score = float(dets[det_idx, 4]) if dets.shape[1] > 4 else 1.0
            cls_id = int(dets[det_idx, 5]) if dets.shape[1] > 5 else 0
            self.trackers[trk_idx].update(
                det_box,
                class_id=cls_id,
                confidence=score
            )

        # 5. Initialize new trackers for unmatched detections with duplicate suppression
        for i in unmatched_dets:
            det_box = dets[i, :4]
            score = float(dets[i, 4]) if dets.shape[1] > 4 else 1.0
            cls_id = int(dets[i, 5]) if dets.shape[1] > 5 else 0

            # Suppress candidate if it heavily overlaps an existing active tracker of the same class
            is_duplicate = False
            for trk in self.trackers:
                if trk.class_id == cls_id:
                    tb = trk.get_state()[0]
                    iou_overlap = float(iou_batch(det_box.reshape(1, 4), tb.reshape(1, 4))[0, 0])
                    if iou_overlap > 0.70:
                        is_duplicate = True
                        break

            if not is_duplicate:
                trk = KalmanBoxTracker(
                    bbox=det_box,
                    class_id=cls_id,
                    confidence=score
                )
                self.trackers.append(trk)

        # 6. Gather active tracking results
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            is_freshly_detected = (trk.time_since_update < 1)
            is_confirmed = (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)

            if is_freshly_detected or is_confirmed:
                ret.append(np.array([d[0], d[1], d[2], d[3], trk.id, trk.class_id, trk.confidence], dtype=np.float32))

            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.array(ret, dtype=np.float32)
        return np.empty((0, 7), dtype=np.float32)

    def predict_only(self) -> np.ndarray:
        """
        Advance all active trackers by Kalman prediction alone across skipped frames.
        Maintains bounding boxes smoothly without disappearing.
        """
        self.frame_count += 1
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            pos = trk.predict()[0]
            if np.any(np.isnan(pos)):
                self.trackers.pop(i)
                continue

            if trk.hits >= self.min_hits and trk.time_since_update <= self.max_age:
                ret.append(np.array([pos[0], pos[1], pos[2], pos[3], trk.id, trk.class_id, trk.confidence], dtype=np.float32))

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.array(ret, dtype=np.float32)
        return np.empty((0, 7), dtype=np.float32)

    def reset(self):
        """Reset all trackers and counters."""
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
