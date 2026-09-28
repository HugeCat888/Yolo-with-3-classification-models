"""
models/base.py
---------------
โครง Transfer Learning กลางตาม Section 9 ของ final_md:

Pretrained Model (ImageNet) -> Freeze Base -> Add New Head -> Train Classifier
-> Unfreeze Some Layers -> Fine-tune -> Save Best Model

โมเดลทั้ง 3 ตัว (efficientnet_b0, mobilenetv3_large, resnet50)
เรียกใช้ builder นี้ร่วมกัน เพื่อให้โครง head/การ freeze เหมือนกันทุกตัว
(fair comparison ตาม Section 18)
"""

from tensorflow.keras import layers, models


def build_transfer_model(base_model, num_classes: int, dropout: float = 0.3, augmentation_layer=None, input_shape=None):
    """
    รับ base_model (instantiated, include_top=False) ที่ freeze ไว้แล้ว
    แล้วต่อ classification head เดียวกันให้ทุกโมเดล
    """
    base_model.trainable = False  # Freeze Base (ขั้นตอน Train Head)

    if augmentation_layer is not None:
        inputs = layers.Input(shape=input_shape)
        x = augmentation_layer(inputs)
        x = base_model(x)
    else:
        inputs = base_model.input
        x = base_model.output

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"{base_model.name}_durian_classifier")
    model._custom_base_model = base_model
    return model


def unfreeze_top_layers(model, base_model_name: str, num_layers: int = 30):
    """
    Unfreeze บางส่วนของ base model (ชั้นบนสุด num_layers ชั้น) สำหรับ Fine-tuning
    ตาม Section 9 ขั้นตอนที่ 6-7 — BatchNormalization layers จะยังคง frozen ไว้
    เพื่อความเสถียรของสถิติที่เรียนมาจาก ImageNet (แนวทางมาตรฐานของ fine-tuning)
    """
    # หา base model submodel จากชื่อ (ตั้งชื่อไว้ตอน build)
    if hasattr(model, "_custom_base_model"):
        target_base = model._custom_base_model
    else:
        target_base = None
        for layer in model.layers:
            if base_model_name in layer.name or layer.name == base_model_name:
                target_base = layer
                break

    if target_base is None:
        # กรณี base ถูก flatten เข้ากับ functional graph โดยตรง ให้ unfreeze
        # จากปลาย model.layers แทน
        trainable_layers = model.layers[-num_layers:]
        for layer in trainable_layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        return model

    target_base.trainable = True
    for layer in target_base.layers[:-num_layers]:
        layer.trainable = False
    for layer in target_base.layers[-num_layers:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    return model
