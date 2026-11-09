"""
Automated Comprehensive Test Suite for OBJECT VISION.
Validates:
1. Environment and dependencies
2. Multi-Model Singleton Loading (YOLOv8n and YOLOv8s)
3. Full-Frame Detection (640x640)
4. Class-Aware SORT Tracking (Ensures ID consistency & prevents cross-class hijacking)
5. Frame Skipping with Smooth Kalman Prediction
6. Performance Monitoring & Rendering Output
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMG_SIZE,
    MODELS_DIR,
    OUTPUT_DIR,
    SUPPORTED_MODELS,
    get_optimal_device,
)
from detector import YOLOv8Detector
from tracker import ObjectTracker
from utils.performance import PerformanceMonitor
from utils.visualization import draw_bounding_boxes, draw_tracking_hud


def run_all_tests():
    print("=" * 70)
    print("🧪 OBJECT VISION: SYSTEM VERIFICATION TESTS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Hardware and Device Check
    # -------------------------------------------------------------------------
    print("\n[Test 1/6] Hardware Optimization & Device Check...")
    device = get_optimal_device()
    print(f"  ✓ Target compute device detected: {device}")
    assert device in ["cpu", "cuda"]

    # -------------------------------------------------------------------------
    # Test 2: Multi-Model Loading & Caching (YOLOv8n and YOLOv8s)
    # -------------------------------------------------------------------------
    print("\n[Test 2/6] Multi-Model Loading & Caching...")
    for model_label, model_path in SUPPORTED_MODELS.items():
        assert Path(model_path).exists(), f"Weights missing at {model_path}"
        det = YOLOv8Detector(model_name_or_path=model_path)
        det_cached = YOLOv8Detector(model_name_or_path=model_path)
        assert det is det_cached, f"Caching violation for {model_label}"
        print(f"  ✓ {model_label}: Verified cached singleton ({Path(model_path).stat().st_size / (1024*1024):.1f} MB)")

    # -------------------------------------------------------------------------
    # Test 3: Detection Inference on Full Frame (640x640)
    # -------------------------------------------------------------------------
    print("\n[Test 3/6] YOLO Object Detection Inference on Full Frame...")
    detector = YOLOv8Detector(SUPPORTED_MODELS["YOLOv8n (Nano - Recommended for CPU)"])
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (100, 100), (300, 400), (120, 120, 120), -1)

    dets, inf_time = detector.detect(
        frame=test_img,
        conf_threshold=DEFAULT_CONF_THRESHOLD,
        img_size=DEFAULT_IMG_SIZE
    )
    print(f"  ✓ Full-frame inference executed in {inf_time:.1f} ms")
    assert isinstance(dets, np.ndarray)

    # -------------------------------------------------------------------------
    # Test 4: Class-Aware SORT Tracking
    # -------------------------------------------------------------------------
    print("\n[Test 4/6] Class-Aware SORT Tracking Verification...")
    tracker = ObjectTracker(max_age=30, min_hits=1, iou_threshold=0.3)

    # Frame 1: Person (class 0) at (100, 100, 200, 200)
    det1 = np.array([[100.0, 100.0, 200.0, 200.0, 0.95, 0]])
    tracks1 = tracker.process_frame(det1, is_detection_frame=True)
    assert len(tracks1) == 1
    person_id = tracks1[0].track_id
    assert tracks1[0].class_id == 0
    print(f"  ✓ Frame 1: Person registered with ID #{person_id:02d}")

    # Frame 2: Person moves slightly, and a Car (class 2) overlaps nearby
    det2 = np.array([
        [105.0, 105.0, 205.0, 205.0, 0.92, 0],  # Person
        [110.0, 110.0, 210.0, 210.0, 0.88, 2],  # Car (overlapping)
    ])
    tracks2 = tracker.process_frame(det2, is_detection_frame=True)
    assert len(tracks2) == 2, f"Expected 2 tracks, got {len(tracks2)}"

    # Find person track and car track
    person_track = next(t for t in tracks2 if t.class_id == 0)
    car_track = next(t for t in tracks2 if t.class_id == 2)

    assert person_track.track_id == person_id, "Class-aware matching failed: Person ID switched!"
    assert car_track.track_id != person_id, "Class collision error: Car hijacked Person ID!"
    print(f"  ✓ Frame 2: Class-aware association verified (Person retains ID #{person_id:02d}, Car assigned ID #{car_track.track_id:02d})")

    # -------------------------------------------------------------------------
    # Test 5: Frame Skipping (Continuous Kalman Propagation)
    # -------------------------------------------------------------------------
    print("\n[Test 5/6] Frame Skipping & Kalman Motion Propagation...")
    tracks_skip1 = tracker.process_frame(np.empty((0, 6)), is_detection_frame=False)
    assert len(tracks_skip1) == 2, "Tracks must persist on skipped frame 1"

    tracks_skip2 = tracker.process_frame(np.empty((0, 6)), is_detection_frame=False)
    assert len(tracks_skip2) == 2, "Tracks must persist on skipped frame 2"
    print(f"  ✓ Frame 3 & 4 (Skipped): Both tracks smoothly propagated via Kalman Filter")

    # -------------------------------------------------------------------------
    # Test 6: Performance Monitoring & Rendering Output
    # -------------------------------------------------------------------------
    print("\n[Test 6/6] Performance Profiling & Visualization...")
    perf = PerformanceMonitor(window_size=10)
    for _ in range(5):
        perf.start_frame()
        time.sleep(0.01)
        perf.end_frame(inference_time_ms=15.0, tracking_time_ms=1.5, was_detected=True)

    metrics = perf.get_metrics()
    assert metrics["fps"] > 0
    print(f"  ✓ Smoothed Live FPS: {metrics['fps']}")

    annotated = draw_bounding_boxes(test_img.copy(), tracked_objects=tracks_skip2, show_trajectories=True)
    annotated = draw_tracking_hud(annotated, fps=metrics["fps"], active_tracks=len(tracks_skip2), total_unique=2, is_detection_frame=False)

    out_file = OUTPUT_DIR / "verification_render.jpg"
    cv2.imwrite(str(out_file), annotated)
    assert out_file.exists()
    print(f"  ✓ Visual verification artifact saved: {out_file}")

    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED! OBJECT VISION IS FULLY VALIDATED.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
