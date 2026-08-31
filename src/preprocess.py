from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
    )


def split_dataset(
    raw_dataset_dir: str | Path,
    output_dir: str | Path,
    class_names: Sequence[str] | None = None,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Split a raw dataset into train/validation/test folders."""
    raw_dir = Path(raw_dataset_dir)
    output_root = Path(output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {raw_dir}")

    if class_names is None:
        class_names = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())

    split_map: Dict[str, Dict[str, List[str]]] = {"train": {}, "validation": {}, "test": {}}

    for split in ("train", "validation", "test"):
        for class_name in class_names:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
            split_map.setdefault(split, {}).setdefault(class_name, [])

    for class_name in class_names:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue

        files = list_image_files(class_dir)
        if not files:
            continue

        train_files, temp_files = train_test_split(
            files, test_size=validation_ratio + test_ratio, random_state=seed, shuffle=True
        )

        validation_size = validation_ratio / (validation_ratio + test_ratio)
        validation_files, test_files = train_test_split(
            temp_files, test_size=1 - validation_size, random_state=seed, shuffle=True
        )

        for source_file in train_files:
            target = output_root / "train" / class_name / source_file.name
            shutil.copy2(source_file, target)
            split_map["train"].setdefault(class_name, []).append(str(target))

        for source_file in validation_files:
            target = output_root / "validation" / class_name / source_file.name
            shutil.copy2(source_file, target)
            split_map["validation"].setdefault(class_name, []).append(str(target))

        for source_file in test_files:
            target = output_root / "test" / class_name / source_file.name
            shutil.copy2(source_file, target)
            split_map["test"].setdefault(class_name, []).append(str(target))

    return split_map


def build_generators(
    train_dir: str | Path,
    validation_dir: str | Path,
    test_dir: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment_data: bool = True,
):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
    ) if augment_data else ImageDataGenerator(rescale=1.0 / 255.0)

    validation_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    validation_generator = validation_test_datagen.flow_from_directory(
        str(validation_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = validation_test_datagen.flow_from_directory(
        str(test_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, validation_generator, test_generator


def preprocess_image(image: np.ndarray | str | Path, target_size=(224, 224)) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image_array = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if image_array is None:
            raise FileNotFoundError(f"Unable to read image: {image}")
    else:
        image_array = image

    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[-1] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGRA2RGB)
    elif image_array.shape[-1] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(image_array, target_size)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)
