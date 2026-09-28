"""
evaluation/evaluate_yolo.py
-----------------------------
รัน evaluation ของ YOLO model ที่เทรนแล้ว แล้ว print ผลลัพธ์ในรูปแบบที่กรอกลง
ตาราง Section 12.1 ได้ตรง ๆ (Precision, Recall, mAP50, mAP50-95, Inference Time, Parameters, Model Size)

ตัวอย่าง:
    python -m evaluation.evaluate_yolo --weights weights/yolo11n_best.pt --data data/yolo/data.yaml
"""

import argparse
import os
import time

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    metrics = model.val(data=args.data, imgsz=args.imgsz)

    # จับเวลา inference เฉลี่ยต่อภาพแบบง่าย ๆ จาก speed dict ที่ ultralytics คืนมา
    speed = metrics.speed  # dict: preprocess/inference/postprocess (ms)
    n_params = sum(p.numel() for p in model.model.parameters())
    model_size_mb = os.path.getsize(args.weights) / (1024 * 1024)

    print("\n=== YOLO Evaluation (Section 7 / 12.1) ===")
    print(f"Weights:          {args.weights}")
    print(f"Precision:        {metrics.box.mp:.4f}")
    print(f"Recall:           {metrics.box.mr:.4f}")
    print(f"mAP50:            {metrics.box.map50:.4f}")
    print(f"mAP50-95:         {metrics.box.map:.4f}")
    print(f"Inference (ms):   {speed.get('inference', float('nan')):.2f}")
    print(f"Parameters:       {n_params:,}")
    print(f"Model Size (MB):  {model_size_mb:.2f}")


if __name__ == "__main__":
    main()
