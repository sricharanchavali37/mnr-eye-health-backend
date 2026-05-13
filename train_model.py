"""
MNR Eye Health Platform — Model Training Script
Trains MobileNetV2 on ODIR-5K dataset
Run from backend folder: python train_model.py
Output: eye_model.h5
"""

import os
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 16
EPOCHS        = 20
LEARNING_RATE = 0.0001
NUM_CLASSES   = 8
DATA_DIR      = "data/odir/preprocessed_images"
CSV_PATH      = "data/odir/full_df.csv"
MODEL_OUTPUT  = "eye_model.h5"

CLASS_NAMES = [
    "Normal", "Diabetes", "Glaucoma", "Cataract",
    "AMD", "Hypertension", "Myopia", "Other"
]

# ── Load and validate data ────────────────────────────────────────────────────
logger.info("Loading dataset...")
df = pd.read_csv(CSV_PATH)
logger.info(f"Total records: {len(df)}")

def parse_target(t):
    if isinstance(t, str):
        return ast.literal_eval(t)
    return t

df["target"] = df["target"].apply(parse_target)

# Build FULL file paths in pandas (not inside tf.data)
df["full_path"] = df["filename"].apply(
    lambda f: os.path.join(DATA_DIR, f)
)

# Keep only rows where image actually exists
df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)
logger.info(f"Records with valid images: {len(df)}")

# ── Class distribution ────────────────────────────────────────────────────────
targets = np.array(df["target"].tolist())
logger.info("Class distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = int(targets[:, i].sum())
    logger.info(f"  {name}: {count}")

# ── Split dataset ─────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
logger.info(f"Training samples:   {len(train_df)}")
logger.info(f"Validation samples: {len(val_df)}")

# ── Image loading (uses full_path, no os.path inside tf.data) ─────────────────
def load_image_from_path(full_path, label):
    """
    full_path is already a complete path string built in pandas.
    No os.path.join needed inside tf graph — avoids SymbolicTensor error.
    """
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.85, 1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_dataset(dataframe, training=True):
    # Pass full paths directly — built in pandas, not in tf graph
    full_paths = dataframe["full_path"].tolist()
    labels     = np.array(dataframe["target"].tolist(), dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((full_paths, labels))
    ds = ds.map(load_image_from_path, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=500)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

logger.info("Building data pipelines...")
train_ds = make_dataset(train_df, training=True)
val_ds   = make_dataset(val_df,   training=False)
logger.info("Data pipelines ready.")

# ── Build model ───────────────────────────────────────────────────────────────
logger.info("Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

logger.info(f"Model parameters: {model.count_params():,}")

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_OUTPUT,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ── Phase 1 — Train top layers only ──────────────────────────────────────────
logger.info("Phase 1: Training top layers (base model frozen)...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# ── Phase 2 — Fine tune last 30 layers ───────────────────────────────────────
logger.info("Phase 2: Fine-tuning last 30 layers of base model...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ── Save final model ──────────────────────────────────────────────────────────
model.save(MODEL_OUTPUT)
logger.info(f"Model saved to: {MODEL_OUTPUT}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
logger.info("Evaluating on validation set...")
loss, acc = model.evaluate(val_ds, verbose=0)
logger.info(f"Validation accuracy: {acc * 100:.1f}%")
logger.info(f"Validation loss:     {loss:.4f}")

# ── Test single prediction ────────────────────────────────────────────────────
logger.info("Testing single prediction...")
sample     = val_df.iloc[0]
img, _     = load_image_from_path(sample["full_path"],
                                   np.zeros(NUM_CLASSES))
img_batch  = tf.expand_dims(img, 0)
pred       = model.predict(img_batch, verbose=0)[0]
pred_class = CLASS_NAMES[np.argmax(pred)]
confidence = float(np.max(pred)) * 100
true_labels = [CLASS_NAMES[i]
               for i, v in enumerate(sample["target"]) if v == 1]

logger.info(f"Sample:      {sample['filename']}")
logger.info(f"True labels: {true_labels}")
logger.info(f"Predicted:   {pred_class} ({confidence:.1f}%)")
logger.info("Training complete!")
logger.info(f"Model saved at: {MODEL_OUTPUT}")