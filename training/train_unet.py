"""
U-Net Face Segmentation — Training Script (PTQ-Ready)
======================================================
Lab  : Huấn luyện mô hình Face Segmentation bằng U-Net
File : 02_train_unet.py

Pipeline tuân theo yêu cầu lab:
  - Dataset : dataset_split/images/{train,val,test} + masks/{train,val,test}
  - Output  : unet_face_segmentation.keras  (native Keras format, PTQ-ready)
  - Metrics : IoU, Dice score như lab yêu cầu

Thiết kế hướng tới PTQ (Post-Training Quantization) ở bước tiếp theo:
  - Output layer : activation=None  → raw logits (tốt hơn sigmoid cho INT8 calibration)
  - Loss         : BCE(from_logits=True) + Dice loss  (nhất quán với logits)
  - Lưu .keras   : TFLiteConverter đọc trực tiếp, không cần tfmot
  - Không QAT    : PTQ là đủ cho binary segmentation task này
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ── GPU: tránh OOM trên các máy có VRAM hạn chế ──────────────────────────────
for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # Đã được khởi tạo — bỏ qua

# XLA JIT: biên dịch các op nhỏ thành một kernels fuzed
# → ít memory fragmentation, giảm peak VRAM khi inference
# → thường tăng throughput 5–15% khi training
# ⚠️ Chỉ hoạt động trên GPU; trên CPU fallback tự động
tf.config.optimizer.set_jit(True)

import gc  # garbage collector — giải phóng RAM sau các bước nặng

# ── Hằng số — khớp với cấu trúc lab ─────────────────────────────────────────
DATA_DIR       = "dataset_split"          # thư mục gốc theo lab
IMG_SIZE       = (256, 256)               # kích thước ảnh theo lab
INPUT_SHAPE    = (256, 256, 3)
BATCH_SIZE     = 2                        # giảm 8→4→2: tối thiểu cho 2GB RAM, tránh OOM
EPOCHS         = 20                       # theo lab (EPOCHS = 20)
SEED           = 42

# Đường dẫn output — tên file theo đúng yêu cầu nộp bài lab
MODEL_PATH      = "unet_face_segmentation.keras"   # file nộp lab
CHECKPOINT_PATH = "unet_face_segmentation_best.keras"

# Cố định random seed để kết quả tái lặp được
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def load_pair(img_path, mask_path):
    """Load and preprocess image/mask pair. Images normalized to [0, 1]."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0   # [0, 1]
    img.set_shape(INPUT_SHAPE)

    # --- Mask ---
    # Lab: mask nhị phân 0 = background, 255 = face
    # → chuẩn hóa về {0.0, 1.0} bằng ngưỡng 0.5
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)   # binary {0.0, 1.0}
    mask.set_shape((IMG_SIZE[0], IMG_SIZE[1], 1))

    return img, mask


def augment(image: tf.Tensor, mask: tf.Tensor):
    """
    Augmentation nhẹ — chỉ các phép biến đổi an toàn cho segmentation.
    Mask được áp đúng phép flip/crop như ảnh.
    Brightness/contrast chỉ áp cho ảnh, không áp cho mask.
    """
    # Flip ngang ngẫu nhiên (50%)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)  

    return image, mask


def make_dataset(split: str) -> tf.data.Dataset:
    """
    Memory-efficient tf.data pipeline — KHÔNG dùng .cache() trên bất kỳ split nào.
    Dataset ~1200 ảnh × 256×256×3 float32 ≈ ~900 MB khi cache → OOM trên 2 GB RAM.

    Train pipeline  : from_tensor_slices → map(load) → shuffle(buffer=100)
                      → map(augment) → batch → prefetch(AUTOTUNE)
    Val/Test pipeline: from_tensor_slices → map(load) → batch → prefetch(AUTOTUNE)

    AUTOTUNE: TensorFlow tự động chọn số parallel calls tối ưu theo CPU cores.
    Prefetch: overlap data-prefetching với computation, giảm I/O bottleneck.
    Shuffle buffer=100: đủ lớn để shuffle tốt, nhỏ để không tốn thêm RAM.
    """
    AUTOTUNE = tf.data.AUTOTUNE          # let TF pick the best parallelism

    img_dir  = os.path.join(DATA_DIR, "images", split)
    mask_dir = os.path.join(DATA_DIR, "masks",  split)

    # Lọc đúng định dạng ảnh theo lab
    imgs = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not imgs:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong: {img_dir}")

    img_paths  = [os.path.join(img_dir,  f) for f in imgs]
    mask_paths = [os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
                  for f in imgs]

    # from_tensor_slices: TF lưu trữ paths (string) — rất nhẹ RAM, ~few KB
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    # map(load_pair): đọc file từ disk ngay khi cần — không đưa toàn bộ ảnh vào RAM
    ds = ds.map(load_pair, num_parallel_calls=AUTOTUNE)

    if split == "train":
        # Shuffle buffer=100: đủ shuffle tốt mà không tốn thêm RAM đáng kể
        # reshuffle_each_iteration=True: shuffle lại mỗi epoch → reproducibility
        ds = ds.shuffle(
            buffer_size=100,           # chỉ shuffle 100 items tại mỗi thời điểm
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)            # batch TRƯỚC prefetch — đúng thứ tự
    ds = ds.prefetch(AUTOTUNE)          # overlap CPU preprocessing với GPU compute
    return ds


# ── 2. Custom loss & metrics ──────────────────────────────────────────────────
# Tất cả hàm đều nhận RAW LOGITS (không qua sigmoid) vì output layer activation=None
# → nhất quán, tránh double-sigmoid bug

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """
    Dice loss tính trên raw logits.
    Sigmoid được áp bên trong — không gọi từ ngoài.
    """
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
    """
    Loss = BCE(from_logits=True) + Dice loss.
    Kết hợp hai loss giúp hội tụ nhanh hơn và tránh trường hợp
    Dice bị unstable khi mask rất sparse.
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable()
def dice_metric(y_true, y_pred, smooth: float = 1e-6):
    """Dice score — metric quan sát khi training. Nhận raw logits."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter  = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom  = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth: float = 1e-6):
    """IoU score — metric yêu cầu bởi lab (mục 7). Nhận raw logits."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.sigmoid(y_pred) > 0.5, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    union = (tf.reduce_sum(y_true_f, axis=1)
             + tf.reduce_sum(y_pred_f, axis=1) - inter)
    return tf.reduce_mean((inter + smooth) / (union + smooth))


# ── 3. Model architecture ─────────────────────────────────────────────────────

def unet() -> models.Model:
    """
    U-Net encoder-decoder cho binary face segmentation.
    Memory-optimized: filters giảm 50% (32→16, 64→32, 128→64) để chạy trên 2GB RAM.

    Cấu trúc giữ nguyên:
      - Double conv block (2 Conv + BN + ReLU)
      - BatchNormalization — ổn định training
      - use_bias=False trước BN (BN đã có learnable beta)
      - Dropout(0.3) tại bottleneck
      - Bilinear upsampling — ít artifact hơn nearest-neighbor

    Output:
      activation=None → raw logits → phù hợp PTQ INT8 calibration
      Threshold quyết định: sigmoid(0) = 0.5 → logit threshold = 0.0
    """
    def conv_block(x, filters: int):
        """Double conv + BN + ReLU."""
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")

    # ── Encoder ──────────────────────────────────────────────
    # Filter giảm 50% (32→16, 64→32, 128→64) giảm ~75% params trong conv layers
    c1 = conv_block(inputs, 16);   p1 = layers.MaxPooling2D()(c1)  # 128×128
    c2 = conv_block(p1,     32);   p2 = layers.MaxPooling2D()(c2)  # 64×64
    c3 = conv_block(p2,     64);  p3 = layers.MaxPooling2D()(c3)  # 32×32

    # ── Bottleneck ────────────────────────────────────────────
    b = conv_block(p3, 64)
    b = layers.Dropout(0.3)(b)

    # ── Decoder (bilinear upsampling + skip connection) ───────
    u1 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(b)
    c4 = conv_block(layers.Concatenate()([u1, c3]), 64)            # 64×64

    u2 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(c4)
    c5 = conv_block(layers.Concatenate()([u2, c2]), 32)            # 128×128

    u3 = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(c5)
    c6 = conv_block(layers.Concatenate()([u3, c1]), 16)           # 256×256

    # ── Output ────────────────────────────────────────────────
    # activation=None → raw logits → PTQ calibration không bị giới hạn range
    outputs = layers.Conv2D(1, 1, activation=None, name="logits")(c6)

    return models.Model(inputs, outputs, name="unet_face_segmentation")


# ── 4. Xây dựng dataset ───────────────────────────────────────────────────────

train_ds = make_dataset("train")
val_ds   = make_dataset("val")

# Test split — tùy chọn, chỉ dùng nếu tồn tại (lab không bắt buộc)
test_ds = None
if (os.path.isdir(os.path.join(DATA_DIR, "images", "test")) and
        os.path.isdir(os.path.join(DATA_DIR, "masks",  "test"))):
    test_ds = make_dataset("test")


# ── 5. Compile model ──────────────────────────────────────────────────────────

# clear_session: giải phóng memory từ các session/layer trước đó
# gc.collect(): force reclaim Python objects trước khi tạo model mới
# quan trọng khi chạy lại script nhiều lần trên cùng process
tf.keras.backend.clear_session()
gc.collect()

model = unet()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=combined_bce_dice_loss,
    metrics=[
        # threshold=0.0: logit > 0 ↔ sigmoid(logit) > 0.5 ↔ "face"
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
        dice_metric,
        iou_metric,   # IoU — tiêu chí chấm điểm của lab (mục 9)
    ],
)


# ── 6. Callbacks ──────────────────────────────────────────────────────────────

callbacks = [
    # Dừng sớm nếu val_loss không cải thiện — lưu lại trọng số tốt nhất
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    # Giảm LR khi plateau — tránh stuck tại saddle point
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
    # Lưu checkpoint tốt nhất — dùng cho PTQ nếu training bị gián đoạn
    tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
]


# ── 7. Training ───────────────────────────────────────────────────────────────

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ── 8. Đánh giá — theo yêu cầu lab mục 7 ────────────────────────────────────

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


# ── 9. Lưu model — PTQ-ready ─────────────────────────────────────────────────
#
# Dùng native .keras format (không phải SavedModel / save_format="tf"):
#   - TFLiteConverter.from_keras_model(model)  đọc trực tiếp
#   - Không cần tfmot installed khi convert
#   - Custom loss/metric tự deserialize nhờ @register_keras_serializable
#
# Sau bước này, chạy script PTQ để tạo:
#   unet_fp32.tflite  — quantize weights về float16 (optional)
#   unet_int8.tflite  — full INT8 với representative dataset

model.save(MODEL_PATH)
print(f"\nModel đã lưu: {MODEL_PATH}")
