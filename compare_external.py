
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "external_test"

MODEL_640 = BASE_DIR / "best_v4_640_openvino_model"
MODEL_512 = BASE_DIR / "best_v4_512_openvino_model"

OUTPUT_DIR = BASE_DIR / "external_results"

CONFIDENCE = 0.25

# =====================================
# CHECK FILES
# =====================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError("External test folder not found.")

if not MODEL_640.exists():
    raise FileNotFoundError("640 OpenVINO model not found.")

if not MODEL_512.exists():
    raise FileNotFoundError("512 OpenVINO model not found.")

images = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

if not images:
    raise FileNotFoundError("No test images found.")

print(f"Total Test Images: {len(images)}")

# =====================================
# BENCHMARK FUNCTION
# =====================================

def test_model(model_path, image_size):

    print("\n" + "=" * 45)
    print(f"TESTING YOLO11l - {image_size}x{image_size}")
    print("=" * 45)

    model = YOLO(str(model_path), task="detect")

    results = model.predict(
        source=str(IMAGE_DIR),
        imgsz=image_size,
        device="intel:cpu",
        conf=CONFIDENCE,
        save=True,
        project=str(OUTPUT_DIR),
        name=str(image_size),
        exist_ok=True,
        verbose=False
    )

    for result in results:

        counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                class_name = result.names[class_id]
                counts[class_name] += 1

                print(
                    f"  {class_name}: "
                    f"Confidence = {confidence:.4f}"
                )

        print(f"\nImage: {Path(result.path).name}")
        print("Detections:", dict(counts) if counts else "None")

    print(f"\nSaved results: {OUTPUT_DIR / str(image_size)}")


# =====================================
# RUN BOTH MODELS
# =====================================

test_model(MODEL_640, 640)

test_model(MODEL_512, 512)

print("\n" + "=" * 45)
print("COMPARISON COMPLETED")
print("=" * 45)
