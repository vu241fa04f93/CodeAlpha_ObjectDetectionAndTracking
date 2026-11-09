"""
YOLOv8 Object Detector Module for OBJECT VISION.
Supports multi-model singleton caching (YOLOv8n, YOLOv8s), GPU/CPU auto-detection,
and full-frame inference without spatial restrictions or center-cropping.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

from config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_IMG_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_NMS_IOU_THRESHOLD,
    MODELS_DIR,
    get_optimal_device,
    is_half_precision_supported,
)


class YOLOv8Detector:
    """
    Cached model wrapper for Ultralytics YOLOv8.
    Ensures each requested model (e.g. YOLOv8n, YOLOv8s) is loaded ONLY ONCE
    and reused across all subsequent frames.
    """
    _instances: Dict[str, "YOLOv8Detector"] = {}

    def __new__(cls, model_name_or_path: str = DEFAULT_MODEL_PATH, *args, **kwargs):
        path_str = str(Path(model_name_or_path).resolve())
        if path_str not in cls._instances:
            instance = super(YOLOv8Detector, cls).__new__(cls)
            cls._instances[path_str] = instance
        return cls._instances[path_str]

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        force_cpu: bool = False
    ):
        # Prevent re-initialization if already loaded
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_path = Path(model_name_or_path)
        self.device = "cpu" if force_cpu else (device or get_optimal_device())
        self.half = is_half_precision_supported(self.device)
        self.model: Optional[YOLO] = None
        self.names: Dict[int, str] = {}
        self.last_inference_time_ms: float = 0.0

        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """Loads the YOLOv8 model safely and verifies weights in models/."""
        try:
            print(f"[Detector] Initializing {self.model_path.name} on device: {self.device} (FP16: {self.half})")

            # Check if weights exist locally in models/
            target_file = MODELS_DIR / self.model_path.name
            if not target_file.exists():
                print(f"[Detector] Model file not found at {target_file}. Fetching pretrained {self.model_path.name}...")
                temp_model = YOLO(self.model_path.name)
                downloaded_file = Path(self.model_path.name)
                if downloaded_file.exists() and downloaded_file != target_file:
                    shutil.move(str(downloaded_file), str(target_file))
                self.model = YOLO(str(target_file))
            else:
                self.model = YOLO(str(target_file))

            self.names = self.model.names if hasattr(self.model, "names") else {}
            print(f"[Detector] {self.model_path.name} loaded successfully with {len(self.names)} classes.")

            # Warmup
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            warmup_kwargs = {
                "source": dummy,
                "device": self.device,
                "imgsz": 320,
                "verbose": False
            }
            if self.half:
                warmup_kwargs["half"] = True
            self.model.predict(**warmup_kwargs)
            print(f"[Detector] {self.model_path.name} warmup complete.")

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_NMS_IOU_THRESHOLD,
        img_size: int = DEFAULT_IMG_SIZE,
        selected_classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Runs full-frame inference using letterbox scaling (no center cropping or spatial gating).
        Returns bounding boxes, confidence scores, and class IDs.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized.")

        if frame is None or frame.size == 0:
            return np.empty((0, 6), dtype=np.float32), 0.0

        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        classes_filter = selected_classes if (selected_classes is not None and len(selected_classes) > 0) else None

        predict_kwargs = {
            "source": frame,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "imgsz": img_size,
            "classes": classes_filter,
            "device": self.device,
            "verbose": False
        }
        if self.half:
            predict_kwargs["half"] = True

        results = self.model.predict(**predict_kwargs)
        t1 = time.perf_counter()
        self.last_inference_time_ms = (t1 - t0) * 1000.0

        boxes_list = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for i in range(len(xyxy)):
                x1 = max(0.0, min(float(w - 1), float(xyxy[i][0])))
                y1 = max(0.0, min(float(h - 1), float(xyxy[i][1])))
                x2 = max(0.0, min(float(w - 1), float(xyxy[i][2])))
                y2 = max(0.0, min(float(h - 1), float(xyxy[i][3])))

                if x2 > x1 and y2 > y1:
                    boxes_list.append([
                        x1, y1, x2, y2,
                        float(conf[i]),
                        int(cls_ids[i])
                    ])

        if len(boxes_list) == 0:
            return np.empty((0, 6), dtype=np.float32), self.last_inference_time_ms

        return np.array(boxes_list, dtype=np.float32), self.last_inference_time_ms

    def get_class_name(self, class_id: int) -> str:
        """Returns the human-readable class name for a given class ID."""
        return self.names.get(int(class_id), f"Class {class_id}")
