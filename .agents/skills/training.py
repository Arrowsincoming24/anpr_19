"""
Skill: Model Training
Trains an InceptionResNetV2-based bounding box regressor for plate localisation.
Run from project root: python -m .agents.skills.training
"""
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4


def load_manifest(manifest_json: Path):
    with open(manifest_json) as f:
        return json.load(f)


def load_dataset(manifest: list[dict]):
    """Convert manifest into (X, y) numpy arrays suitable for training."""
    import cv2
    X, y = [], []
    for ann in manifest:
        img_path = ann.get("image_path")
        boxes = ann.get("boxes", [])
        if not img_path or not boxes:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_resized = cv2.resize(img, IMG_SIZE)
        img_norm = img_resized / 255.0

        # Use first box only (primary plate)
        box = boxes[0]
        # Normalize coords to [0, 1]
        xmin = box["xmin"] / w
        ymin = box["ymin"] / h
        xmax = box["xmax"] / w
        ymax = box["ymax"] / h

        X.append(img_norm)
        y.append([xmin, ymin, xmax, ymax])

    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def build_model():
    """Build InceptionResNetV2 + custom regression head."""
    import tensorflow as tf
    from tensorflow import keras

    base = keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False   # freeze during warm-up

    x = base.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(4, activation="sigmoid")(x)   # (xmin, ymin, xmax, ymax)

    model = keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train(manifest_paths: list[Path] = None):
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    if manifest_paths is None:
        manifest_paths = list(DATA_PROCESSED.glob("*_manifest.json"))

    if not manifest_paths:
        logger.error("No manifest files found. Run data_ingestion.py first.")
        return

    all_annotations = []
    for mp in manifest_paths:
        all_annotations.extend(load_manifest(mp))

    logger.info(f"Total annotations: {len(all_annotations)}")

    X, y = load_dataset(all_annotations)
    if len(X) == 0:
        logger.error("Dataset is empty — check image_path fields in manifests.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "best_model.keras"), save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.TensorBoard(log_dir=str(PROJECT_ROOT / "logs")),
    ]

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    model.save(str(MODELS_DIR / "anpr_model.keras"))
    logger.info("Model saved to models/anpr_model.keras")
    return history


if __name__ == "__main__":
    train()
