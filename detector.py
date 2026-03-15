# src/detector.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO


# Must match the class order used in your attire_data.yaml during training
CLASS_NAMES = [
    "short_sleeve",
    "long_sleeve",
    "sleeveless",
    "shorts",
    "trousers",
    "slippers",
    "covered_shoes",
]


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    cls_id: int
    cls_name: str
    conf: float

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4, iou=0.5, imgsz=640, device=None):
        """
        Simple wrapper for YOLOv8.
        - model_path: path to trained model .pt
        - conf: global confidence threshold (pre-filter)
        - iou: NMS IoU threshold
        - imgsz: inference image size
        - device: "cpu" or "cuda"
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Runs YOLO on a single BGR image (OpenCV format).
        Returns the first result object.
        """
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False
        )
        return results[0]

    def detect_detections(self, image_bgr: np.ndarray) -> List[Detection]:
        """
        Runs YOLO and returns structured detections for downstream logic
        (tracking thread, ROI checks, logging).
        """
        r0 = self.detect(image_bgr)
        dets: List[Detection] = []

        if r0.boxes is None:
            return dets

        for b in r0.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)

            dets.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    cls_id=cls_id,
                    cls_name=cls_name,
                    conf=conf,
                )
            )

        return dets

    @staticmethod
    def get_class_names() -> List[str]:
        return CLASS_NAMES.copy()
