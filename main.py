import os
import kagglehub
import yaml
import multiprocessing
from ultralytics import YOLO


def main():
    print("Pobieranie datasetu...")
    dataset_path = kagglehub.dataset_download("lakshaytyagi01/fruit-detection")
    print(f"Ścieżka: {dataset_path}")

    base_dir = os.path.join(dataset_path, "Fruits-detection")

    data_yaml = {
        "path": base_dir,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 6,
        "names": ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]
    }

    yaml_path = "data_max_quality.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    model = YOLO("yolov8l.pt")

    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=24,
        device=0,
        workers=8,
        project="runs",
        name="rtx5070ti_large_model"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()