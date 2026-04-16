"""Shared U-Net model definition."""

import tensorflow as tf
from tensorflow.keras import layers, models


def unet(input_shape: tuple = (256, 256, 3), num_classes: int = 1) -> models.Model:
    """Build the U-Net used for binary face segmentation."""
    IMG_SIZE = input_shape[:2]

    def conv_block(x, filters: int):
        """Double conv + BN + ReLU."""
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    inputs = layers.Input(shape=input_shape, name="image_input")

    c1 = conv_block(inputs,   16);  p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1,       32);  p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2,       64);  p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3, 64)
    b = layers.Dropout(0.3)(b)

    u1 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(b)
    c4 = conv_block(layers.Concatenate()([u1, c3]), 64)

    u2 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(c4)
    c5 = conv_block(layers.Concatenate()([u2, c2]), 32)

    u3 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(c5)
    c6 = conv_block(layers.Concatenate()([u3, c1]), 16)

    # Keep raw logits for PTQ conversion and thresholding at inference time.
    outputs = layers.Conv2D(num_classes, 1, activation=None, name="logits")(c6)

    return models.Model(inputs, outputs, name="unet_face_segmentation")
