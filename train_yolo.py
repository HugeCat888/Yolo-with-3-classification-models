"""
train_yolo.py
--------------
เทรน YOLO11 สำหรับ Localization (หา bounding box ของใบทุเรียน — class เดียว "leaf")
ตามขั้นตอนใน Section 6.3:
    เตรียม data.yaml -> โหลด pretrained YOLO11 -> Train -> Validate -> ดู metrics
    -> Confusion Matrix/F1/PR -> ตรวจ prediction -> Save best.pt

ตัวอย่างการใช้งาน:
    python train_yolo.py --variant n --epochs 100
    python train_yolo.py --variant s --epochs 100 --imgsz 640

ก่อนรันต้องเตรียม data/yolo/data.yaml โดย annotate bounding box ด้วย Roboflow
ตามที่ระบุใน Section 6.1 แล้ว export เป็น YOLO format
(images/ + labels/ ของแต่ละ train/valid/test)
"""

import argparse
import os
import shutil

from ultralytics import YOLO

from config import RANDOM_SEED, WEIGHTS_DIR, YOLO_DATA_YAML, YOLO_DEFAULTS, set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11 for leaf localization (Section 6).")
    p.add_argument("--variant", default="n", choices=["n", "s"],
                   help="YOLO11n (baseline) หรือ YOLO11s (comparison) — Section 6.2")
    p.add_argument("--data_yaml", default=YOLO_DATA_YAML)
    p.add_argument("--imgsz", type=int, default=YOLO_DEFAULTS["imgsz"])
    p.add_argument("--epochs", type=int, default=YOLO_DEFAULTS["epochs"])
    p.add_argument("--batch", type=int, default=YOLO_DEFAULTS["batch"])
    p.add_argument("--patience", type=int, default=YOLO_DEFAULTS["patience"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--device", default="", help="ระบุอุปกรณ์ในการเทรน เช่น 0 (GPU 0), 0,1 (Multi-GPU) หรือ cpu")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if not os.path.isfile(args.data_yaml):
        raise FileNotFoundError(
            f"ไม่พบ {args.data_yaml}\n"
            "ต้องสร้าง data/yolo/data.yaml ก่อน (annotate ด้วย Roboflow ตาม Section 6.1 "
            "แล้ว export เป็น YOLO format)"
        )

    pretrained_name = f"yolo11{args.variant}.pt"
    print(f"=== Training {pretrained_name} | epochs={args.epochs} imgsz={args.imgsz} seed={args.seed} ===")

    model = YOLO(pretrained_name)  # โหลด pretrained YOLO11 (Ultralytics จะดาวน์โหลดอัตโนมัติถ้ายังไม่มี)

    results = model.train(
        data=args.data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device=args.device if args.device else None,
        project=WEIGHTS_DIR,
        name=f"yolo11{args.variant}_leaf",
    )

    # Validate ทันทีหลัง train (ขั้นตอนที่ 4 ใน Section 6.3)
    metrics = model.val()
    print("\n=== Validation metrics (Section 7) ===")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    # คัดลอก best.pt ไปไว้ที่ weights/ ตามชื่อที่ Section 15 ระบุไว้
    run_dir = os.path.join(WEIGHTS_DIR, f"yolo11{args.variant}_leaf")
    best_src = os.path.join(run_dir, "weights", "best.pt")
    best_dst = os.path.join(WEIGHTS_DIR, f"yolo11{args.variant}_best.pt")
    if os.path.isfile(best_src):
        shutil.copy(best_src, best_dst)
        print(f"\nSaved best checkpoint to: {best_dst}")


if __name__ == "__main__":
    main()
