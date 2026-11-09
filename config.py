"""
Configuration module for Real-Time Object Detection and Tracking (OBJECT VISION).
Defines centralized hyperparameters, design system constants, and paths.
"""

import os
from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
MODELS_DIR = BASE_DIR / "models"
SAMPLE_VIDEOS_DIR = BASE_DIR / "sample_videos"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Supported Pretrained Models
SUPPORTED_MODELS = {
    "YOLOv8n (Nano - Recommended for CPU)": str(MODELS_DIR / "yolov8n.pt"),
    "YOLOv8s (Small - Higher Accuracy)": str(MODELS_DIR / "yolov8s.pt")
}
DEFAULT_MODEL_NAME = "YOLOv8n (Nano - Recommended for CPU)"
DEFAULT_MODEL_PATH = SUPPORTED_MODELS[DEFAULT_MODEL_NAME]

# Model Detection Defaults
SUPPORTED_IMG_SIZES = [640, 416, 320]
DEFAULT_IMG_SIZE = 640
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_NMS_IOU_THRESHOLD = 0.45

# Device detection & optimization
def get_optimal_device() -> str:
    """Detect if CUDA (GPU) is available, otherwise default to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def is_half_precision_supported(device_name: str) -> bool:
    """Check if FP16 (half-precision) is safely supported on the device."""
    return device_name.startswith("cuda") and torch.cuda.is_available()

# Frame skipping & video processing optimization
DEFAULT_FRAME_SKIP = 3  # Run YOLO detection every 3rd frame; SORT tracks in-between
MIN_FRAME_SKIP = 1
MAX_FRAME_SKIP = 8

# Processing resolution options (width in px, aspect ratio preserved)
SUPPORTED_RESOLUTIONS = {
    "Original (Native View)": 0,
    "High Quality (800p)": 800,
    "Balanced (640p)": 640,
    "Fast (480p)": 480
}
DEFAULT_RESOLUTION_WIDTH = 0

# SORT Tracker Hyperparameters
SORT_MAX_AGE = 30       # Consecutive frames to keep track alive without detection
SORT_MIN_HITS = 1       # Detections before track confirmation
SORT_IOU_THRESHOLD = 0.30  # IoU overlap threshold for Hungarian bipartite matching

# Professional UI Design System Colors (Part 1 Design Spec)
THEME_COLORS = {
    "bg": "#0B1020",
    "sidebar": "#111827",
    "card": "#161F32",
    "accent": "#38BDF8",
    "secondary_accent": "#818CF8",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "text": "#F8FAFC",
    "secondary_text": "#94A3B8",
    "border": "#273449"
}

# Standard COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
