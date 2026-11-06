
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent

IMAGE = BASE_DIR / "external_test" / "test4.jpg"

MODELS = {
    "640": BASE_DIR / "best_v4_640_openvino_model",
    "512": BASE_DIR / "best_v4_512_openvino_model",
}

for size, model_path in MODELS.items():

    print("\n" + "=" * 40)
    print(f"TEST4 IMAGE - {size}x{size}")
    print("=" * 40)

    model = YOLO(str(model_path), task="detect")

    result = model.predict(
        source=str(IMAGE),
        imgsz=int(size),
        device="intel:cpu",
        conf=0.25,
        verbose=False
    )[0]

    counts = Counter()

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = result.names[class_id]

        counts[class_name] += 1

        print(
            f"Class: {class_name} | "
            f"Confidence: {confidence:.4f}"
        )

    print("\nFinal Detection:", dict(counts) if counts else "None")

print("\nTEST COMPLETED")
