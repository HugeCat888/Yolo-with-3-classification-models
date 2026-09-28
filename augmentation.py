"""
augmentation.py
----------------
สร้าง Augmentation Version 1 / 2 / 3 ตามที่นิยามไว้ใน Section 4 ของ final_md
โดยใช้ tf.keras preprocessing layers (ทำงานเฉพาะตอน training=True เท่านั้น
จึงไม่กระทบ Validation/Test — ตรงตามกฎ "Augmentation ให้ทำกับ Train เท่านั้น" ใน Section 3.2)

หมายเหตุ: ไฟล์นี้ไม่ได้ระบุไว้ใน Section 15 เดิม แต่แยกออกมาจาก train_classifier.py
เพื่อให้ config การทดลองแต่ละ version อ่านและปรับง่าย โดยไม่ปนกับ training loop
"""

from tensorflow.keras import layers, Sequential


def get_augmentation(version: str):
    """
    version: 'none' | 'v1' | 'v2' | 'v3'
    คืน tf.keras.Sequential ของ augmentation layers หรือ None ถ้า version == 'none'
    """
    version = version.lower()

    if version == "none":
        return None

    if version == "v1":
        # Version 1 — Mild Augmentation (Section 4)
        return Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.03),          # rotation เล็กน้อย
                layers.RandomZoom(0.05),               # zoom/scale เล็กน้อย
                layers.RandomTranslation(0.05, 0.05),  # translation เล็กน้อย
            ],
            name="augmentation_v1_mild",
        )

    if version == "v2":
        # Version 2 — Moderate Augmentation (Section 4)
        return Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.06),
                layers.RandomZoom(0.10),
                layers.RandomTranslation(0.08, 0.08),
                layers.RandomBrightness(0.10),
                layers.RandomContrast(0.10),
            ],
            name="augmentation_v2_moderate",
        )

    if version == "v3":
        # Version 3 — Strong Augmentation (Section 4)
        # หมายเหตุ (ข้อควรระวังใน Section 4): ไม่ควรบิดเบือนสี/รูปร่างจนผิดจากภาพจริง
        # เกินไป เพราะสี/ขอบ/เส้นใบเป็น feature สำคัญของการจำแนกพันธุ์
        return Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.10),
                layers.RandomZoom(0.15),
                layers.RandomTranslation(0.10, 0.10),
                layers.RandomBrightness(0.20),
                layers.RandomContrast(0.20),
            ],
            name="augmentation_v3_strong",
        )

    raise ValueError(f"Unknown augmentation version: {version}. Choices: none, v1, v2, v3")
