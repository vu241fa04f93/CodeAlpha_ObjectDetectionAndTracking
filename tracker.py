"""
Object Tracker Module.
Integrates SORT tracking algorithm with lifecycle management, trajectory history,
and class-level object counting.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import (
    COCO_CLASSES,
    SORT_IOU_THRESHOLD,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
)
from sort import Sort


@dataclass
class TrackedObject:
    """Represents a single tracked object in the current frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) as integers
    class_id: int
    class_name: str
    confidence: float
    trajectory: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[int, int]:
        """Calculates center point (cx, cy) of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


class ObjectTracker:
    """
    High-level tracking manager that orchestrates the SORT algorithm,
    maintains object trajectories, and tracks cumulative and per-frame statistics.
    """
    def __init__(
        self,
        max_age: int = SORT_MAX_AGE,
        min_hits: int = SORT_MIN_HITS,
        iou_threshold: float = SORT_IOU_THRESHOLD,
        trajectory_length: int = 25
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trajectory_length = trajectory_length

        self.sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.trajectories: Dict[int, deque] = {}
        self.all_seen_ids: Set[int] = set()
        self.class_counts: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    def process_frame(
        self,
        detections: np.ndarray,
        is_detection_frame: bool = True,
        class_names_map: Optional[Dict[int, str]] = None
    ) -> List[TrackedObject]:
        """
        Processes a video frame with SORT.

        Args:
            detections: (N, 6) numpy array [x1, y1, x2, y2, confidence, class_id]
            is_detection_frame: If True, runs full Hungarian matching + update.
                               If False (skipped frame), runs Kalman prediction only.
            class_names_map: Optional dictionary mapping class ID to class name.

        Returns:
            List of confirmed TrackedObject instances for the current frame.
        """
        if is_detection_frame:
            sort_output = self.sort.update(detections)
        else:
            sort_output = self.sort.predict_only()

        current_tracked_objects: List[TrackedObject] = []
        active_ids_in_frame: Set[int] = set()

        if len(sort_output) > 0:
            for item in sort_output:
                # Format: [x1, y1, x2, y2, track_id, class_id, score]
                x1 = int(round(item[0]))
                y1 = int(round(item[1]))
                x2 = int(round(item[2]))
                y2 = int(round(item[3]))

                # Ensure valid coordinates
                if x2 <= x1 or y2 <= y1:
                    continue

                track_id = int(item[4])
                cls_id = int(item[5]) if len(item) > 5 else 0
                conf = float(item[6]) if len(item) > 6 else 1.0

                # Determine class name
                if class_names_map and cls_id in class_names_map:
                    cls_name = class_names_map[cls_id]
                elif 0 <= cls_id < len(COCO_CLASSES):
                    cls_name = COCO_CLASSES[cls_id]
                else:
                    cls_name = f"object_{cls_id}"

                # Update trajectory history
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = deque(maxlen=self.trajectory_length)
                self.trajectories[track_id].append((cx, cy))

                # Track unique objects
                if track_id not in self.all_seen_ids:
                    self.all_seen_ids.add(track_id)
                    self.id_to_class[track_id] = cls_name
                    self.class_counts[cls_name] = self.class_counts.get(cls_name, 0) + 1

                active_ids_in_frame.add(track_id)

                obj = TrackedObject(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    trajectory=list(self.trajectories[track_id])
                )
                current_tracked_objects.append(obj)

        # Cleanup trajectories for expired tracks
        active_tracker_ids = {t.id for t in self.sort.trackers}
        dead_ids = [tid for tid in self.trajectories if tid not in active_tracker_ids]
        for dead_id in dead_ids:
            del self.trajectories[dead_id]

        return current_tracked_objects

    @property
    def total_unique_objects(self) -> int:
        """Returns total distinct object IDs tracked across the entire session."""
        return len(self.all_seen_ids)

    def get_frame_class_counts(self, tracked_objects: List[TrackedObject]) -> Dict[str, int]:
        """Returns count of each detected class currently visible in the active frame."""
        counts: Dict[str, int] = {}
        for obj in tracked_objects:
            cname = obj.class_name.upper()
            counts[cname] = counts.get(cname, 0) + 1
        return counts

    def get_summary_stats(self) -> Dict[str, any]:
        """Returns comprehensive tracking statistics for the UI."""
        return {
            "total_unique": len(self.all_seen_ids),
            "class_breakdown": dict(self.class_counts),
            "active_tracks_count": len(self.sort.trackers)
        }

    def reset(self):
        """Resets tracker state, trajectories, and statistics."""
        self.sort.reset()
        self.trajectories.clear()
        self.all_seen_ids.clear()
        self.class_counts.clear()
        self.id_to_class.clear()
