import os
import kagglehub
import yaml
from ultralytics import YOLO

def main():
    print("📦 Pobieranie datasetu Fruit Detection (YOLO format)...")
    dataset_path = kagglehub.dataset_download("lakshaytyagi01/fruit-detection")
    print(f"✅ Dataset pobrany do: {dataset_path}")

    base_dir = os.path.join(dataset_path, "Fruits-detection")
    if not os.path.exists(base_dir):
        raise RuntimeError(f"❌ Nie znaleziono folderu Fruits-detection w: {dataset_path}")
    print(f"📂 Folder bazowy datasetu: {base_dir}")

    train_path = os.path.join(base_dir, "train", "images").replace("\\", "/")
    val_path = os.path.join(base_dir, "valid", "images").replace("\\", "/")
    test_path = os.path.join(base_dir, "test", "images").replace("\\", "/")

    data_yaml = {
        "path": base_dir,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": 6,
        "names": ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]
    }

    yaml_path = os.path.join(base_dir, "data_quick.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    print(f"🧩 Utworzono plik konfiguracji YOLO: {yaml_path}")

    print("⚡ Uruchamiamy SZYBKI trening YOLOv8 (5 epok)...")
    model = YOLO("yolov8n.pt")  # mniejszy model = szybciej

    results = model.train(
        data=yaml_path,
        epochs=5,             # ⏱️ tylko 5 epok!
        imgsz=512,            # trochę mniejszy input, szybszy
        batch=4,              # dopasowany do GTX 1070
        device=0,             # GPU
        lr0=0.003,
        patience=3,
        optimizer="SGD",
        amp=True,
        cache="disk",
        mosaic=0.8,
        mixup=0.05,
        cos_lr=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5,
        fliplr=0.5,
        name="fruits-detection-quick",
        project="runs",
    )

    print("✅ Szybki trening zakończony!")
    print(f"📁 Wyniki: {results.save_dir}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
