
import os
import io

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO


app = FastAPI(title="Aura Edge AI - Real-Time Backend")


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load INT8 OpenVINO Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_int8_openvino_model"
)

model = YOLO(MODEL_PATH, task="detect")

print("YOLO11 INT8 OpenVINO Model Loaded Successfully!")


@app.get("/")
def home():
    return {
        "message": "Aura Edge AI Real-Time Backend is running!"
    }


# Real-Time Prediction
@app.post("/realtime/predict")
def realtime_predict(file: UploadFile = File(...)):

    try:
        image_bytes = file.file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image file."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # INT8 OpenVINO Inference
    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        iou=0.1,
        device="intel:gpu",
        verbose=False
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[class_id],
            "confidence": round(confidence, 4),
            "box": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        })

    return {
        "detections": detections,
        "performance": {
            "preprocess_ms": round(
                result.speed.get("preprocess", 0), 2
            ),
            "inference_ms": round(
                result.speed.get("inference", 0), 2
            ),
            "postprocess_ms": round(
                result.speed.get("postprocess", 0), 2
            ),
        }
    }
