"""
evaluation/evaluate_classifier.py
-----------------------------------
ประเมินผล classifier ที่เทรนแล้วบน Test Set ตาม metric ที่กำหนดใน Section 11:
Accuracy, Precision, Recall, F1, Confusion Matrix, Per-class P/R/F1, Inference Time,
Parameters, Model Size — ครบสำหรับกรอกตาราง Section 12.2

ตัวอย่าง:
    python -m evaluation.evaluate_classifier --model resnet50 \
        --checkpoint weights/resnet50_v1_best.keras
"""

import argparse
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_NAMES, TEST_DIR
from datasets import build_dataset
from evaluation.plots import plot_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["efficientnet_b0", "mobilenetv3_large", "resnet50"],
                   help="ใช้เพื่อเลือก preprocessing/input size ให้ตรงกับตอนเทรน (Section 10)")
    p.add_argument("--checkpoint", required=True, help="path ของไฟล์ .keras ที่บันทึกไว้")
    p.add_argument("--test_dir", default=TEST_DIR)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", default="evaluation/reports")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = tf.keras.models.load_model(args.checkpoint)
    test_ds, n_test = build_dataset(args.test_dir, args.model, args.batch_size, shuffle=False)

    y_true, y_pred = [], []
    start = time.time()
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))
    elapsed = time.time() - start

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    n_params = model.count_params()
    model_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    avg_inference_ms = (elapsed / max(n_test, 1)) * 1000

    print("\n=== Classification Report (Section 11 Test) ===")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print(f"\nAvg inference time per image: {avg_inference_ms:.2f} ms")
    print(f"Parameters: {n_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")

    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    report_path = os.path.join(args.output_dir, f"{run_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\n\nAvg inference time per image: {avg_inference_ms:.2f} ms\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Model size: {model_size_mb:.2f} MB\n")

    cm_path = os.path.join(args.output_dir, f"{run_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, CLASS_NAMES, cm_path)

    print(f"\nSaved report to: {report_path}")
    print(f"Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    main()
