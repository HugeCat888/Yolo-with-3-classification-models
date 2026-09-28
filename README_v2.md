# Durian Leaf Detection & Classification

This project implements a **Two-Stage Pipeline** for Durian Leaf Classification and Localization:
1. **Localization (YOLO11)**: Detects durian leaves in raw images.
2. **Classification (Transfer Learning)**: Crops the bounding boxes and classifies the leaf variant using deep learning models (`efficientnet_b0`, `mobilenetv3_large`, `resnet50`).
3. **Web App (Streamlit)**: Interactive interface to upload and test the pipeline end-to-end.

---

## 🚀 Getting Started

### 1. Installation
```bash
python -m venv venv
# Activate your virtual environment (e.g., `venv\Scripts\activate` on Windows)
pip install -r requirements.txt
```

### 2. Prepare Data
Organize your data properly to maintain consistent splits (Train/Validation/Test) and prevent data leakage:
* **Detection (YOLO)**: Place YOLO format dataset in `data/yolo/`
* **Classification**: Place cropped images in `data/classification/`. 
  * *Tip: Run `python prepare_classification_dataset.py` to auto-crop leaves from your YOLO ground-truth bounding boxes.*

### 3. Training

#### Train YOLO (Detection)
```bash
python train_yolo.py --variant n --epochs 100   # Nano model
python train_yolo.py --variant s --epochs 100   # Small model
```

#### Train Classifiers
```bash
python train_classifier.py --model resnet50 --augmentation none
```
*(Available models: `efficientnet_b0`, `mobilenetv3_large`, `resnet50`. Available augmentations: `none`, `v1`, `v2`, `v3`)*

### 4. Evaluation

Evaluate YOLO:
```bash
python -m evaluation.evaluate_yolo --weights weights/yolo11s_best.pt --data data/yolo/data.yaml
```

Evaluate Classifier:
```bash
python -m evaluation.evaluate_classifier --model resnet50 --checkpoint weights/resnet50_v1_best.keras
```

### 5. Launch the Web App
Run the Streamlit demo application to test the models in real-time:
```bash
streamlit run app.py
```
