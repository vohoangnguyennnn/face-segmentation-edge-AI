"""Train a PTQ-ready U-Net for face segmentation."""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from training.models.unet_model import unet
from training.models.config import (
    DATASET_SPLIT_DIR,
    MODEL_DIR,
    KERAS_MODEL,
    KERAS_BEST,
)

IMG_SIZE    = (256, 256)
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE  = 2
EPOCHS      = 20
SEED        = 42

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

tf.config.optimizer.set_jit(True)

import gc

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def load_pair(img_path, mask_path):
    """Load and preprocess image/mask pair. Images normalized to [0, 1]."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape(INPUT_SHAPE)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    mask.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))

    return img, mask


def augment(image: tf.Tensor, mask: tf.Tensor):
    """
    Apply light augmentation that stays safe for segmentation masks.
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)  

    return image, mask


def make_dataset(split: str) -> tf.data.Dataset:
    """
    Build a memory-efficient tf.data pipeline without caching full splits.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    img_dir  = os.path.join(DATASET_SPLIT_DIR, "images", split)
    mask_dir = os.path.join(DATASET_SPLIT_DIR, "masks",  split)

    imgs = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not imgs:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {img_dir}")

    img_paths  = [os.path.join(img_dir,  f) for f in imgs]
    mask_paths = [os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
                  for f in imgs]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)

    if split == "train":
        ds = ds.shuffle(
            buffer_size=100,
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """Compute Dice loss from raw logits."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.sigmoid(tf.cast(y_pred, tf.float32))
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter  = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    dice   = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


@tf.keras.utils.register_keras_serializable()
def combined_bce_dice_loss(y_true, y_pred):
    """Compute BCE(from_logits=True) plus Dice loss."""
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable()
def dice_metric(y_true, y_pred, smooth: float = 1e-6):
    """Compute Dice score from raw logits."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter  = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth: float = 1e-6):
    """Compute IoU from raw logits."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = (tf.reduce_sum(y_true_f, axis=1)
             + tf.reduce_sum(y_pred_f, axis=1) - inter)
    return tf.reduce_mean((inter + smooth) / (union + smooth))



train_ds = make_dataset("train")
val_ds   = make_dataset("val")

test_ds = None
if (os.path.isdir(os.path.join(DATASET_SPLIT_DIR, "images", "test")) and
        os.path.isdir(os.path.join(DATASET_SPLIT_DIR, "masks",  "test"))):
    test_ds = make_dataset("test")

tf.keras.backend.clear_session()
gc.collect()

model = unet()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=combined_bce_dice_loss,
    metrics=[
        # Logit 0.0 is equivalent to sigmoid 0.5 for binary accuracy.
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
        dice_metric,
        iou_metric,
    ],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        KERAS_BEST,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]

os.makedirs(MODEL_DIR, exist_ok=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


print("\n" + "=" * 60)
val_results = model.evaluate(val_ds, verbose=0)
print(
    f"[Validation] loss={val_results[0]:.4f} | "
    f"accuracy={val_results[1]:.4f} | "
    f"dice={val_results[2]:.4f} | "
    f"iou={val_results[3]:.4f}"
)

if test_ds is not None:
    test_results = model.evaluate(test_ds, verbose=0)
    print(
        f"[Test]       loss={test_results[0]:.4f} | "
        f"accuracy={test_results[1]:.4f} | "
        f"dice={test_results[2]:.4f} | "
        f"iou={test_results[3]:.4f}"
    )
print("=" * 60)

model.save(KERAS_MODEL)
print(f"\nModel đã lưu: {KERAS_MODEL}")
