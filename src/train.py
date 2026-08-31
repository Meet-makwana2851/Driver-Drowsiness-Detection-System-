from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

try:
    from src.preprocess import build_generators, split_dataset
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.preprocess import build_generators, split_dataset


def get_class_labels(data_root: Path):
    return sorted([child.name for child in data_root.iterdir() if child.is_dir()])


def build_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    model = Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.45),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def save_training_history(history, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=200)
    plt.close()

    with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)


def prepare_dataset(data_root: Path, output_root: Path):
    if (output_root / "train").exists() and (output_root / "validation").exists() and (output_root / "test").exists():
        return output_root

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    class_dirs = [child for child in data_root.iterdir() if child.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"No class directories were found under {data_root}. "
            "Expected subfolders such as Open_Eyes, Closed_Eyes, Yawning, Non_Yawning."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    split_dataset(data_root, output_root, class_names=[p.name for p in class_dirs])
    return output_root


def parse_args():
    parser = argparse.ArgumentParser(description="Train the drowsiness classifier CNN.")
    parser.add_argument("--data-root", type=str, default="dataset", help="Dataset root or raw class folder.")
    parser.add_argument("--output-root", type=str, default="dataset", help="Directory for train/validation/test split.")
    parser.add_argument("--output-model", type=str, default="models/drowsiness_model.keras", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    model_path = Path(args.output_model)
    image_size = tuple(args.image_size)

    prepared_root = prepare_dataset(data_root, output_root)
    train_dir = prepared_root / "train"
    validation_dir = prepared_root / "validation"
    test_dir = prepared_root / "test"
    class_labels = get_class_labels(train_dir)

    train_generator, validation_generator, test_generator = build_generators(
        train_dir=train_dir,
        validation_dir=validation_dir,
        test_dir=test_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        augment_data=True,
    )

    model = build_cnn_model(input_shape=(image_size[0], image_size[1], 3), num_classes=len(class_labels))
    model.summary()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    save_training_history(history, report_dir)

    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    conf_matrix = confusion_matrix(y_true, y_pred)
    print("\nEvaluation summary:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(conf_matrix)

    np.savetxt(report_dir / "confusion_matrix.csv", conf_matrix, delimiter=",")
    summary = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "class_labels": class_labels,
    }
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Training metrics saved under {report_dir}")


if __name__ == "__main__":
    main()
