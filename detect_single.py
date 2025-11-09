import os
from ultralytics import YOLO

# ============================================
# 1️⃣ Ścieżki
# ============================================
model_path = r"runs/fruits-detection-yolo-tuned/weights/best.pt"
image_path = r"owoce.jpg"
output_dir = r"runs/detect-fruits"

os.makedirs(output_dir, exist_ok=True)

# ============================================
# 2️⃣ Ładowanie modelu
# ============================================
print("🍏 Ładowanie wytrenowanego modelu YOLOv8...")
model = YOLO(model_path)

# ============================================
# 3️⃣ Detekcja na jednym obrazie
# ============================================
print(f"🖼️ Wykrywanie owoców na obrazie: {image_path}")

results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.4,
    device=0,        # ✅ wymusza GPU (GTX 1070)
    save=True,       # zapisze wynik
    project=output_dir,
    name="result",   # zapisze do runs/detect-fruits/result/
    show=False       # w PyCharm bez GUI – lepiej nie otwierać okna
)

# ============================================
# 4️⃣ Informacje o wynikach
# ============================================
# YOLO zapisze wynik jako np. runs/detect-fruits/result/owoce.jpg
result_folder = os.path.join(output_dir, "result")
result_path = os.path.join(result_folder, os.path.basename(image_path))
print(f"✅ Wynik zapisany w: {result_path}")

# Wypisz wykryte obiekty
if results and len(results) > 0:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"🔸 {label} ({conf:.2f})")
else:
    print("⚠️ Nie wykryto żadnych obiektów.")
