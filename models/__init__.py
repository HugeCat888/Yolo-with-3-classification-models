"""
models package
---------------
รวม registry ของโมเดลทั้งหมด (Section 8.1) ให้ train_classifier.py และ
evaluation/evaluate_classifier.py เรียกใช้ผ่านชื่อเดียวกันได้
"""

from . import efficientnet_b0, mobilenetv3_large, resnet50

MODEL_REGISTRY = {
    "efficientnet_b0": efficientnet_b0,
    "mobilenetv3_large": mobilenetv3_large,
    "resnet50": resnet50,
}


def get_model_module(model_name: str):
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choices: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
