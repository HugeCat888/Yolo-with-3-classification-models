"""
prepare_classification_dataset.py
---------------------------------
สร้าง Classification Dataset จาก YOLO bounding boxes โดยรักษา Train/Valid/Test
ของ Original Dataset เดิมไว้แบบ 1:1 เพื่อป้องกัน data leakage

Pipeline:
    data/original/<split>/<class>/original image
                    +
    data/yolo/<split>/images + labels
                    |
                    v
              crop ตาม bbox
                    |
                    v
    data/classification/<split>/<class>/crop

สำคัญ:
- Original แบ่งไว้แล้วเป็น train=280, valid=60, test=60 ต่อ class
- ห้ามรวมแล้วสุ่ม split ใหม่
- YOLO ใช้ class เดียว: 0 = leaf
- พันธุ์ทุเรียนมาจากโฟลเดอร์ของ Original ไม่ได้มาจาก YOLO label
- Validation/Test ไม่ทำ augmentation ในสคริปต์นี้

Expected structure:

data/
├── original/
│   ├── train/<5 classes>     # 280/class
│   ├── valid/<5 classes>     # 60/class
│   └── test/<5 classes>      # 60/class
├── yolo/
│   ├── train/images + labels
│   ├── valid/images + labels
│   └── test/images + labels
└── classification/           # script สร้าง crop ให้
    ├── train/<5 classes>
    ├── validation/<5 classes>
    └── test/<5 classes>

Run:
    python prepare_classification_dataset.py

หมายเหตุเรื่องชื่อไฟล์:
Roboflow อาจเติม suffix เช่น .rf.<hash> ให้ชื่อไฟล์ YOLO
สคริปต์จะพยายามจับคู่ชื่อเต็มและ stem ก่อน .rf. แต่ต้องอยู่ใน split เดียวกัน
ถ้าชื่อภาพไม่สัมพันธ์กับ Original จริง ๆ ต้องทำ mapping ก่อน ไม่ควรเดาพันธุ์
"""

from pathlib import Path
from PIL import Image
from config import CLASS_NAMES
from preprocessing import crop_box

PROJECT_ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = PROJECT_ROOT / "data" / "original"
YOLO_DIR = PROJECT_ROOT / "data" / "yolo"
CLASSIFICATION_DIR = PROJECT_ROOT / "data" / "classification"

SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_LEAF_CLASS_ID = 0
CROP_PADDING_RATIO = 0.05


def normalize_filename(filename: str) -> str:
    path = Path(filename)
    stem = path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem.lower()


def build_original_index(split: str):
    """สร้าง index เฉพาะ Original split นี้: filename/stem -> class name"""
    index = {}
    split_dir = ORIGINAL_DIR / split

    if not split_dir.exists():
        print(f"[WARNING] ไม่พบ Original split: {split_dir}")
        return index

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"[WARNING] ไม่พบโฟลเดอร์ class: {class_dir}")
            continue

        for file_path in class_dir.iterdir():
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            keys = {file_path.name.lower(), normalize_filename(file_path.name)}
            for key in keys:
                if key in index and index[key] != class_name:
                    raise ValueError(
                        f"ชื่อไฟล์/ชื่อ stem ชนกันใน Original {split}: {key}"
                        f"พบทั้ง {index[key]} และ {class_name}"
                    )
                index[key] = class_name

    return index


def find_original_class(filename: str, original_index):
    """ค้นหาพันธุ์จาก Original ของ split เดียวกัน"""
    filename_key = filename.lower()
    stem_key = normalize_filename(filename)
    return original_index.get(filename_key) or original_index.get(stem_key)


def read_yolo_labels(label_path: Path):
    boxes = []
    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"[WARNING] label ไม่ถูกต้อง: {label_path}:{line_number}")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                print(f"[WARNING] อ่าน label ไม่ได้: {label_path}:{line_number}")
                continue

            if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
                print(f"[WARNING] center อยู่นอก 0-1: {label_path}:{line_number}")
                continue
            if not 0 < width <= 1 or not 0 < height <= 1:
                print(f"[WARNING] width/height ไม่ถูกต้อง: {label_path}:{line_number}")
                continue
            boxes.append((class_id, x_center, y_center, width, height))

    return boxes


def yolo_to_pixel_box(x_center, y_center, width, height, image_width, image_height):
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    box_width_pixel = width * image_width
    box_height_pixel = height * image_height
    return (
        x_center_pixel - box_width_pixel / 2,
        y_center_pixel - box_height_pixel / 2,
        x_center_pixel + box_width_pixel / 2,
        y_center_pixel + box_height_pixel / 2,
    )


def process_image(image_path: Path, label_path: Path, output_dir: Path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[WARNING] เปิดภาพไม่ได้: {image_path} -> {exc}")
        return 0

    boxes = read_yolo_labels(label_path)
    if not boxes:
        print(f"[WARNING] ไม่มี bounding box: {label_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    image_width, image_height = image.size
    saved_count = 0

    for box_index, (class_id, x_center, y_center, width, height) in enumerate(boxes):
        if class_id != YOLO_LEAF_CLASS_ID:
            print(f"[WARNING] พบ class_id={class_id}; คาดว่า 0 = leaf: {label_path}")
            continue

        box = yolo_to_pixel_box(
            x_center, y_center, width, height, image_width, image_height
        )
        cropped = crop_box(image, box, padding_ratio=CROP_PADDING_RATIO)
        if cropped.width <= 1 or cropped.height <= 1:
            print(f"[WARNING] Crop เล็กเกินไป: {image_path}")
            continue

        suffix = "_crop.jpg" if len(boxes) == 1 else f"_crop_{box_index + 1}.jpg"
        cropped.save(output_dir / f"{image_path.stem}{suffix}", quality=95)
        saved_count += 1

    return saved_count


def process_split(original_split: str, yolo_split: str, classification_split: str):
    """จับคู่ข้อมูลจาก split เดียวกันเท่านั้น"""
    original_index = build_original_index(original_split)
    images_dir = YOLO_DIR / yolo_split / "images"
    labels_dir = YOLO_DIR / yolo_split / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[WARNING] YOLO split ไม่ครบ: {YOLO_DIR / yolo_split}")
        return 0

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_crops = matched = unmatched = 0
    print(f"\n=== Original {original_split} -> YOLO {yolo_split} -> Classification {classification_split} ===")
    print(f"YOLO images: {len(image_paths)}")

    for image_path in image_paths:
        class_name = find_original_class(image_path.name, original_index)
        if class_name is None:
            print(f"[WARNING] จับคู่ Original {original_split} ไม่ได้: {image_path.name}")
            unmatched += 1
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARNING] ไม่พบ label คู่กับภาพ: {image_path.name}")
            unmatched += 1
            continue

        output_dir = CLASSIFICATION_DIR / classification_split / class_name
        saved = process_image(image_path, label_path, output_dir)
        if saved:
            matched += 1
            total_crops += saved

    print(f"Matched images : {matched}")
    print(f"Unmatched      : {unmatched}")
    print(f"Crops saved    : {total_crops}")
    return total_crops


def main():
    print("=" * 70)
    print("Prepare Classification Dataset - Preserve Original Splits")
    print("=" * 70)
    print(f"Original : {ORIGINAL_DIR}")
    print(f"YOLO     : {YOLO_DIR}")
    print(f"Output   : {CLASSIFICATION_DIR}")

    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(
            f"ไม่พบ {ORIGINAL_DIR}"
            "สร้าง data/original/train, valid, test ก่อนรัน"
        )
    if not YOLO_DIR.exists():
        raise FileNotFoundError(
            f"ไม่พบ {YOLO_DIR}"
            "เตรียม YOLO train/valid/test จาก Roboflow ก่อนรัน"
        )

    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for yolo_split, classification_split in SPLIT_MAP.items():
        total += process_split(yolo_split, yolo_split, classification_split)

    print("\n" + "=" * 70)
    print(f"DONE - Total crops saved: {total}")
    print("=" * 70)
    print("ห้ามสุ่ม split ใหม่หลังจากขั้นตอนนี้")


if __name__ == "__main__":
    main()
