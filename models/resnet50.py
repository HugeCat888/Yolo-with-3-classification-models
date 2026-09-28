"""
models/resnet50.py
--------------------
จุดประสงค์ (Section 8.1 Model 3): ใช้เป็น architecture แบบ residual network
สำหรับเปรียบเทียบกับ model ที่เน้น efficiency
"""

from tensorflow.keras.applications import ResNet50
from models.base import build_transfer_model, unfreeze_top_layers

MODEL_NAME = "resnet50"


def build_model(num_classes: int, input_shape=(224, 224, 3), dropout: float = 0.3, augmentation_layer=None):
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    return build_transfer_model(base_model, num_classes, dropout=dropout, augmentation_layer=augmentation_layer, input_shape=input_shape)


def unfreeze_for_finetune(model, num_layers: int = 30):
    return unfreeze_top_layers(model, "resnet50", num_layers=num_layers)
