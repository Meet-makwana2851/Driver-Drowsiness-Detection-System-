from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the driver drowsiness CNN model.")
    parser.add_argument("--model", type=str, default="models/drowsiness_model.keras", help="Path to trained model.")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Directory containing test dataset.")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory where evaluation reports will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    test_dir = Path(args.test_dir)
    report_dir = Path(args.report_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    model = keras.models.load_model(str(model_path))
    train_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = train_generator.flow_from_directory(
        str(test_dir),
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    y_true = test_generator.classes
    y_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "evaluation_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }, handle, indent=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.savefig(report_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
