"""
datasets.py
------------
สร้าง tf.data.Dataset จากโฟลเดอร์ data/classification/<split>/<ClassName>/
โดยใช้ preprocessing.resize_with_padding เดียวกับที่ app.py ใช้ตอน inference
เพื่อให้ Train / Validation / Test / Inference สอดคล้องกันตาม Section 10

หมายเหตุ: ไม่ได้อยู่ใน Section 15 เดิม แต่แยกออกมาจาก train_classifier.py
เพื่อให้ data-loading logic ทดสอบ/แก้ไขแยกจาก training loop ได้
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASS_NAMES, get_input_size
from preprocessing import resize_with_padding, get_preprocess_fn


def list_paths_and_labels(directory: str):
    """สแกนโฟลเดอร์ตามชื่อ class ใน CLASS_NAMES คืน (paths, labels) ที่ index ตรงกับ CLASS_NAMES"""
    paths, labels = [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return paths, labels


def _load_and_pad_numpy(path_bytes, target_size):
    path = path_bytes.numpy().decode("utf-8")
    img = Image.open(path).convert("RGB")
    img = resize_with_padding(img, target_size)
    return np.array(img).astype("float32")


def _make_load_fn(target_size):
    def load_fn(path, label):
        img = tf.py_function(
            func=lambda p: _load_and_pad_numpy(p, target_size),
            inp=[path],
            Tout=tf.float32,
        )
        img.set_shape([target_size[0], target_size[1], 3])
        return img, label

    return load_fn


def build_dataset(
    directory: str,
    model_name: str,
    batch_size: int,
    shuffle: bool = False,
    augmentation_layer=None,
    seed: int = 42,
):
    """
    สร้าง tf.data.Dataset สำหรับ train/validation/test
    - augmentation_layer: ส่งเข้ามาเฉพาะตอนสร้าง train dataset เท่านั้น (None สำหรับ val/test)
    """
    target_size = get_input_size(model_name)
    paths, labels = list_paths_and_labels(directory)

    if len(paths) == 0:
        raise ValueError(
            f"ไม่พบภาพในโฟลเดอร์ {directory}\n"
            f"ตรวจสอบว่าวางไฟล์ตามโครงสร้าง data/classification/<split>/<ClassName>/ "
            f"โดยชื่อ ClassName ต้องตรงกับ config.CLASS_NAMES = {CLASS_NAMES}"
        )

    one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_NAMES))
    ds = tf.data.Dataset.from_tensor_slices((paths, one_hot))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_make_load_fn(target_size), num_parallel_calls=tf.data.AUTOTUNE)

    if augmentation_layer is not None:
        ds = ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    preprocess_fn = get_preprocess_fn(model_name)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)
