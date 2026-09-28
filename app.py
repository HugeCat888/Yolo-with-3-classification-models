"""
app.py
-------
Streamlit application ตาม Section 14 ของ final_md

User Flow:
    Open Streamlit -> Upload JPG/PNG -> YOLO11 detects leaf -> Draw Bounding Box
    -> Crop detected leaf -> Classification Model -> Predict Durian Variety -> Display Result

รันด้วย:
    streamlit run app.py
"""

import os

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw

from config import CLASS_NAMES, CLASS_NAMES_TH, DETECTION_CONFIDENCE_THRESHOLD, WEIGHTS_DIR
from detector import LeafDetector
from preprocessing import crop_box, load_and_preprocess_image

st.set_page_config(page_title="Durian Leaf Classifier", page_icon="🍃", layout="centered")


# ---------------------------------------------------------------------------
# Cached model loading (โหลดครั้งเดียวต่อ session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_detector(weights_path: str, confidence_threshold: float):
    return LeafDetector(weights_path, confidence_threshold=confidence_threshold)


@st.cache_resource
def load_classifier(checkpoint_path: str):
    return tf.keras.models.load_model(checkpoint_path)


def draw_boxes(image: Image.Image, detections, labels=None):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        tag = f"Leaf #{i + 1} ({det.confidence:.2f})"
        if labels and i < len(labels):
            tag += f" -> {labels[i]}"
        draw.text((x1 + 4, max(0, y1 - 18)), tag, fill="lime")
    return annotated


def main():
    st.title("🍃 Durian Leaf Classification")
    st.caption("YOLO11 Localization + Transfer Learning Classification (Section 14)")

    # --- Sidebar: config / model selection -------------------------------
    st.sidebar.header("Settings")

    yolo_weight_options = [
        f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pt")
    ] if os.path.isdir(WEIGHTS_DIR) else []
    classifier_weight_options = [
        f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".keras")
    ] if os.path.isdir(WEIGHTS_DIR) else []

    if not yolo_weight_options or not classifier_weight_options:
        st.warning(
            "ยังไม่พบไฟล์ weight ใน `weights/` — ต้องรัน `train_yolo.py` และ "
            "`train_classifier.py` ก่อน (หรือวางไฟล์ .pt / .keras ที่เทรนไว้แล้วในโฟลเดอร์นี้)"
        )
        st.stop()

    yolo_weight_name = st.sidebar.selectbox("YOLO weights (Section 6.2)", yolo_weight_options)
    clf_weight_name = st.sidebar.selectbox("Classifier weights (Section 8.1)", classifier_weight_options)
    confidence_threshold = st.sidebar.slider(
        "Detection confidence threshold (Section 14.4)",
        min_value=0.05, max_value=0.95, value=DETECTION_CONFIDENCE_THRESHOLD, step=0.05,
    )
    single_leaf_mode = st.sidebar.checkbox(
        "แสดงเฉพาะใบที่มั่นใจที่สุด (Section 14.3)", value=False
    )

    # ต้องระบุชื่อโมเดล classifier ให้ preprocessing เลือก input size/preprocess_fn ถูกต้อง (Section 10)
    model_name_for_preprocess = st.sidebar.selectbox(
        "Classifier architecture ของไฟล์ที่เลือก",
        ["efficientnet_b0", "mobilenetv3_large", "resnet50"],
        help="ต้องตรงกับโมเดลที่ใช้ตอนเทรน checkpoint ที่เลือกไว้ด้านบน",
    )

    detector = load_detector(os.path.join(WEIGHTS_DIR, yolo_weight_name), confidence_threshold)
    detector.confidence_threshold = confidence_threshold  # เผื่อผู้ใช้ปรับ slider ระหว่าง session
    classifier = load_classifier(os.path.join(WEIGHTS_DIR, clf_weight_name))

    # --- Upload -------------------------------------------------------------
    uploaded_file = st.file_uploader("อัปโหลดภาพใบทุเรียน (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("กรุณาอัปโหลดภาพเพื่อเริ่มการตรวจจับและจำแนกพันธุ์")
        return

    # --- Edge case: ไฟล์เสียหาย / ไม่ใช่ภาพจริง (Section 14.4) ------------------
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("ไม่สามารถเปิดไฟล์นี้เป็นภาพได้ กรุณาอัปโหลดไฟล์ JPG/PNG ที่ถูกต้อง")
        return

    st.image(image, caption="ภาพที่อัปโหลด", use_container_width=True)

    with st.spinner("กำลังตรวจจับใบด้วย YOLO11..."):
        detections = detector.detect(image)

    # --- Edge case: ไม่เจอใบเลย (Section 14.4) ---------------------------------
    if not detections:
        st.warning(
            f"ไม่พบใบทุเรียนในภาพ (หรือ confidence ต่ำกว่า {confidence_threshold:.2f}) "
            "กรุณาอัปโหลดภาพใหม่ หรือปรับ threshold ในแถบด้านซ้าย"
        )
        return

    if single_leaf_mode:
        detections = [detections[0]]  # เลือกเฉพาะ confidence สูงสุด (Section 14.3)

    # --- Classify each detected leaf ---------------------------------------
    labels_display = []
    results = []
    for det in detections:
        cropped = crop_box(image, det.box_xyxy, padding_ratio=0.05)
        input_tensor = load_and_preprocess_image(cropped, model_name_for_preprocess)
        preds = classifier.predict(input_tensor, verbose=0)[0]
        class_idx = int(preds.argmax())
        class_name = CLASS_NAMES[class_idx]
        class_conf = float(preds[class_idx])

        labels_display.append(f"{CLASS_NAMES_TH[class_name]} ({class_conf:.2f})")
        results.append(dict(
            crop=cropped,
            detection_confidence=det.confidence,
            class_name=class_name,
            class_name_th=CLASS_NAMES_TH[class_name],
            class_confidence=class_conf,
        ))

    annotated_image = draw_boxes(image, detections, labels=labels_display)
    st.image(annotated_image, caption="ผลการตรวจจับใบ (Bounding Box)", use_container_width=True)

    # --- Display results (Section 14: UI ควรแสดงอย่างน้อยนี้) -------------------
    st.subheader("ผลลัพธ์")
    for i, res in enumerate(results):
        with st.container(border=True):
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(res["crop"], caption=f"Leaf #{i + 1}", use_container_width=True)
            with cols[1]:
                st.markdown(f"**Detected Object:** Leaf #{i + 1}")
                st.markdown(f"**Detection Confidence:** {res['detection_confidence']:.2f}")
                st.markdown(f"**Predicted Class:** {res['class_name_th']} ({res['class_name']})")
                st.markdown(f"**Classification Confidence:** {res['class_confidence']:.2f}")


if __name__ == "__main__":
    main()
