import os
import kagglehub
import yaml
from ultralytics import YOLO

def main():
    # ============================================
    # 1️⃣ Pobranie datasetu Fruits Detection
    # ============================================
    print("📦 Pobieranie datasetu Fruit Detection (YOLO format)...")
    dataset_path = kagglehub.dataset_download("lakshaytyagi01/fruit-detection")
    print(f"✅ Dataset pobrany do: {dataset_path}")

    # ============================================
    # 2️⃣ Wykrycie folderu z danymi
    # ============================================
    base_dir = os.path.join(dataset_path, "Fruits-detection")
    if not os.path.exists(base_dir):
        raise RuntimeError(f"❌ Nie znaleziono folderu Fruits-detection w: {dataset_path}")

    print(f"📂 Folder bazowy datasetu: {base_dir}")

    # ============================================
    # 3️⃣ Tworzymy własny plik data.yaml (poprawny)
    # ============================================
    train_path = os.path.join(base_dir, "train", "images").replace("\\", "/")
    val_path = os.path.join(base_dir, "valid", "images").replace("\\", "/")
    test_path = os.path.join(base_dir, "test", "images").replace("\\", "/")

    data_yaml = {
        "path": base_dir,
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": 6,
        "names": ["Apple", "Grapes", "Pineapple", "Orange", "Banana", "Watermelon"]
    }

    yaml_path = os.path.join(base_dir, "data_fixed.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"🧩 Utworzono nowy plik konfiguracji YOLO: {yaml_path}")

    # ============================================
    # 4️⃣ Trening YOLOv8
    # ============================================
    print("🚀 Rozpoczynamy trening YOLOv8...")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=yaml_path,
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,
        lr0=0.005,
        patience=10,
        augment=True,
        close_mosaic=5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0,
        cache="disk",
        name="fruits-detection-yolo-tuned",
        project="runs",
    )

    print("✅ Trening zakończony!")
    print(f"📁 Wyniki zapisane w: {results.save_dir}")

    # ============================================
    # 5️⃣ Ewaluacja
    # ============================================
    print("🔍 Ewaluacja modelu...")
    metrics = model.val(data=yaml_path)
    print(metrics)

    # ============================================
    # 6️⃣ Detekcja testowa
    # ============================================
    print("🖼️ Detekcja na obrazach testowych...")

    output_dir = "runs/detect-fruits"
    os.makedirs(output_dir, exist_ok=True)

    model.predict(
        source=test_path,
        imgsz=640,
        conf=0.4,
        save=True,
        project="runs",
        name="detect-fruits",
    )

    print(f"✅ Wyniki detekcji zapisane w: {output_dir}")
    print("🍉 YOLOv8 działa – z ramkami na owocach!")

# ============================================
#  ✅ WAŻNE DLA WINDOWS
# ============================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
