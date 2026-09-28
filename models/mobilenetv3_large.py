"""
models/mobilenetv3_large.py
-----------------------------
จุดประสงค์ (Section 8.1 Model 2): ทดลอง model ที่เหมาะกับงานที่ต้องการ inference ที่เบาและรวดเร็ว
"""

from tensorflow.keras.applications import MobileNetV3Large
from models.base import build_transfer_model, unfreeze_top_layers

MODEL_NAME = "mobilenetv3_large"


def build_model(num_classes: int, input_shape=(224, 224, 3), dropout: float = 0.3, augmentation_layer=None):
    base_model = MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    return build_transfer_model(base_model, num_classes, dropout=dropout, augmentation_layer=augmentation_layer, input_shape=input_shape)


def unfreeze_for_finetune(model, num_layers: int = 30):
    return unfreeze_top_layers(model, "MobilenetV3large", num_layers=num_layers)
