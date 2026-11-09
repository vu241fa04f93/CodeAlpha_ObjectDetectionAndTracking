"""
Test detection and tracking on real video containing people and a bus.
"""
from pathlib import Path
import cv2
import numpy as np
from detector import YOLOv8Detector
from tracker import ObjectTracker
from utils.visualization import draw_bounding_boxes, draw_tracking_hud
from utils.performance import PerformanceMonitor

def test_real_tracking():
    video_path = "sample_videos/pedestrian_bus_demo.mp4"
    out_path = "output/pedestrian_bus_tracked.mp4"

    detector = YOLOv8Detector()
    tracker = ObjectTracker(max_age=15, min_hits=2, iou_threshold=0.3)
    perf = PerformanceMonitor()

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Failed to open video"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, 25, (640, 480))

    frame_count = 0
    frame_skip = 3

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        perf.start_frame()

        is_det = (frame_count % frame_skip == 0) or (frame_count == 1)
        inf_ms = 0.0

        if is_det:
            dets, inf_ms = detector.detect(frame, conf_threshold=0.35, img_size=416)
            tracks = tracker.process_frame(dets, is_detection_frame=True, class_names_map=detector.names)
        else:
            tracks = tracker.process_frame(np.empty((0, 6)), is_detection_frame=False, class_names_map=detector.names)

        perf.end_frame(inference_time_ms=inf_ms, was_detected=is_det)

        annotated = draw_bounding_boxes(frame, tracked_objects=tracks, show_trajectories=True)
        annotated = draw_tracking_hud(
            annotated,
            fps=perf.current_fps,
            active_tracks=len(tracks),
            total_unique=tracker.total_unique_objects,
            is_detection_frame=is_det,
            inference_ms=inf_ms
        )
        out.write(annotated)

    cap.release()
    out.release()

    stats = tracker.get_summary_stats()
    print("=" * 60)
    print("🎯 REAL VIDEO TRACKING TEST RESULTS:")
    print(f"  ✓ Frames processed: {frame_count}")
    print(f"  ✓ Average Live FPS: {perf.current_fps:.1f}")
    print(f"  ✓ Total unique objects tracked: {stats['total_unique']}")
    print(f"  ✓ Tracked class breakdown: {stats['class_breakdown']}")
    print(f"  ✓ Annotated video saved: {out_path}")
    print("=" * 60)

    assert stats['total_unique'] > 0, "Expected at least 1 object to be tracked!"
    assert 'person' in stats['class_breakdown'] or 'bus' in stats['class_breakdown'], "Expected person or bus tracked!"


if __name__ == "__main__":
    test_real_tracking()

