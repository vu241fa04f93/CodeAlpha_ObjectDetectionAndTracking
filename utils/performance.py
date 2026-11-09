"""
Performance Monitoring and FPS Calculation Utility.
Provides moving-average FPS calculation, inference/tracking latency tracking,
and resource metrics without introducing computational overhead.
"""

import time
from collections import deque
from typing import Dict, List, Tuple


class PerformanceMonitor:
    """
    Tracks frame rate (FPS), latency for detection/tracking, and frame statistics
    using a sliding window for smooth, jitter-free metrics.
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.inference_latencies = deque(maxlen=window_size)
        self.tracking_latencies = deque(maxlen=window_size)

        self.last_timestamp: float = time.perf_counter()
        self.total_frames_processed: int = 0
        self.detection_frames_count: int = 0
        self.skipped_frames_count: int = 0
        self.start_time: float = time.perf_counter()

    def start_frame(self) -> None:
        """Mark the beginning of a frame processing cycle."""
        self.last_timestamp = time.perf_counter()

    def end_frame(
        self,
        inference_time_ms: float = 0.0,
        tracking_time_ms: float = 0.0,
        was_detected: bool = True
    ) -> float:
        """
        Mark the completion of a frame and record latency.
        Returns the instantaneous FPS.
        """
        now = time.perf_counter()
        delta = now - self.last_timestamp
        if delta <= 0:
            delta = 1e-6

        self.frame_times.append(delta)
        self.inference_latencies.append(inference_time_ms)
        self.tracking_latencies.append(tracking_time_ms)

        self.total_frames_processed += 1
        if was_detected:
            self.detection_frames_count += 1
        else:
            self.skipped_frames_count += 1

        self.last_timestamp = now
        return 1.0 / delta

    @property
    def current_fps(self) -> float:
        """Computes smoothed moving average FPS."""
        if not self.frame_times:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Average YOLO inference time in milliseconds."""
        if not self.inference_latencies:
            return 0.0
        return sum(self.inference_latencies) / len(self.inference_latencies)

    @property
    def avg_tracking_ms(self) -> float:
        """Average SORT tracking time in milliseconds."""
        if not self.tracking_latencies:
            return 0.0
        return sum(self.tracking_latencies) / len(self.tracking_latencies)

    def get_fps_history(self) -> List[float]:
        """Returns recent FPS values for performance charting."""
        return [round(1.0 / t, 1) for t in self.frame_times if t > 0]

    def get_metrics(self) -> Dict[str, float]:
        """Returns snapshot of current performance metrics."""
        elapsed = time.perf_counter() - self.start_time
        return {
            "fps": round(self.current_fps, 1),
            "inference_ms": round(self.avg_inference_ms, 1),
            "tracking_ms": round(self.avg_tracking_ms, 1),
            "total_frames": self.total_frames_processed,
            "detection_frames": self.detection_frames_count,
            "skipped_frames": self.skipped_frames_count,
            "uptime_seconds": round(elapsed, 1),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.frame_times.clear()
        self.inference_latencies.clear()
        self.tracking_latencies.clear()
        self.total_frames_processed = 0
        self.detection_frames_count = 0
        self.skipped_frames_count = 0
        self.start_time = time.perf_counter()
        self.last_timestamp = time.perf_counter()
