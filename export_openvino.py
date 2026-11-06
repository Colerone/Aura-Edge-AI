
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_v4.pt"

print("Loading YOLO11l...")

model = YOLO(str(MODEL_PATH), task="detect")

print("Exporting OpenVINO FP32 at 640...")

export_path = model.export(
    format="openvino",
    imgsz=640,
    device="cpu"
)

print("\nExport completed!")
print("Model:", export_path)
