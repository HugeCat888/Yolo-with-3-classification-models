"""
detector.py
------------
ห่อ (wrap) YOLO11 inference ให้เรียกใช้ง่ายจาก app.py ตาม flow ใน Section 14
รวม logic การจัดการ threshold / เลือก box ที่ดีที่สุด ตาม Section 14.3 และ 14.4
"""

from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO

from config import DETECTION_CONFIDENCE_THRESHOLD


@dataclass
class Detection:
    box_xyxy: tuple  # (x1, y1, x2, y2)
    confidence: float


class LeafDetector:
    def __init__(self, weights_path: str, confidence_threshold: float = DETECTION_CONFIDENCE_THRESHOLD):
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image: Image.Image):
        """
        รันตรวจจับใบบนภาพเดียว คืน list[Detection] ที่ confidence >= threshold
        (Section 14.4: ถ้า confidence ต่ำกว่า threshold ไม่ถือว่าเป็นผลลัพธ์ที่ใช้ได้)
        """
        results = self.model.predict(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(Detection(box_xyxy=(x1, y1, x2, y2), confidence=conf))
        # เรียงจาก confidence สูง -> ต่ำ เพื่อให้เลือก "ใบที่มั่นใจที่สุด" ได้ง่าย (Section 14.3)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_best(self, image: Image.Image):
        """คืนเฉพาะ Detection ที่ confidence สูงสุด หรือ None ถ้าไม่เจอเลย (Section 14.3 กรณีต้องการ 1 ใบ)"""
        detections = self.detect(image)
        return detections[0] if detections else None
