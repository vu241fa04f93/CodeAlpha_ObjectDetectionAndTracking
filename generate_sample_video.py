"""
Generates a realistic sample synthetic video for testing object detection and tracking
when no webcam or external video files are present.
Draws moving objects across the screen to test bounding boxes, Kalman motion estimation,
and Hungarian bipartite association.
"""

import math
from pathlib import Path
import cv2
import numpy as np

OUTPUT_PATH = Path(__file__).resolve().parent / "sample_videos" / "sample_traffic_test.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_synthetic_sample(output_file: str = str(OUTPUT_PATH), num_frames: int = 180, fps: int = 30):
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Object 1: Moves left to right
    # Object 2: Moves right to left
    # Object 3: Bounces diagonally
    for t in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 35  # dark asphalt background

        # Draw road lanes
        cv2.line(frame, (0, 180), (width, 180), (200, 200, 200), 2)
        for x_dash in range(0, width, 40):
            cv2.line(frame, (x_dash, 180), (x_dash + 20, 180), (255, 255, 255), 2)

        # Object 1 (simulated car 1)
        x1 = int(20 + t * 3.0) % (width + 60) - 40
        y1 = 120
        cv2.rectangle(frame, (x1, y1), (x1 + 70, y1 + 35), (60, 120, 220), -1)
        cv2.rectangle(frame, (x1 + 10, y1 + 5), (x1 + 30, y1 + 30), (20, 20, 20), -1)
        cv2.putText(frame, "CAR", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        # Object 2 (simulated car 2)
        x2 = int(width - 20 - t * 2.5) % (width + 80) - 40
        y2 = 220
        cv2.rectangle(frame, (x2, y2), (x2 + 80, y2 + 40), (220, 100, 50), -1)
        cv2.putText(frame, "BUS", (x2 + 5, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        # Object 3 (simulated pedestrian)
        x3 = int(250 + 60 * math.sin(t * 0.05))
        y3 = int(60 + t * 0.8) % (height - 40)
        cv2.circle(frame, (x3, y3), 14, (80, 210, 100), -1)
        cv2.putText(frame, "PERSON", (x3 - 20, y3 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 120), 1)

        out.write(frame)

    out.release()
    print(f"[Sample Video] Successfully generated test video at: {output_file}")


if __name__ == "__main__":
    create_synthetic_sample()

