"""
evaluation/plots.py
---------------------
ฟังก์ชันวาดกราฟที่ Section 13 กำหนดไว้ (ฝั่ง Classification):
Training/Validation Loss, Training/Validation Accuracy, Confusion Matrix
YOLO ใช้กราฟที่ Ultralytics สร้างให้อัตโนมัติอยู่แล้วใน weights/<run>/ (results.png, confusion_matrix.png ฯลฯ)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curves(history_csv_path: str, output_path: str):
    """อ่าน CSVLogger output จาก train_classifier.py แล้ววาด Loss/Accuracy curve (Section 13)"""
    df = pd.read_csv(history_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(df["epoch"], df["loss"], label="Train Loss")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "accuracy" in df.columns:
        axes[1].plot(df["epoch"], df["accuracy"], label="Train Accuracy")
    if "val_accuracy" in df.columns:
        axes[1].plot(df["epoch"], df["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, output_path: str):
    """วาด Confusion Matrix heatmap (Section 13)"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
