
import os
import io
from contextlib import asynccontextmanager

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from ultralytics import YOLO


# =========================================
# MODEL CONFIGURATION
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_int8_openvino_model"
)

# Load YOLO INT8 OpenVINO model
model = YOLO(MODEL_PATH, task="detect")

print("YOLO11 INT8 OpenVINO Model Initialized!")


# =========================================
# MODEL WARM-UP
# =========================================

def warmup_model():

    print("======================================")
    print("Starting OpenVINO GPU Warm-up...")
    print("======================================")

    # Create dummy image matching inference input
    warmup_image = Image.new(
        "RGB",
        (640, 640),
        color=(0, 0, 0)
    )

    try:

        # Perform first inference to trigger
        # OpenVINO loading and GPU compilation
        model.predict(
            source=warmup_image,
            imgsz=640,
            conf=0.25,
            iou=0.1,
            device="intel:gpu",
            verbose=False
        )

        print("======================================")
        print("OpenVINO GPU Warm-up Completed!")
        print("Model is Ready for Real-Time Detection.")
        print("======================================")

    except Exception as error:

        print("OpenVINO Warm-up Failed!")
        print(error)

        raise

    finally:

        warmup_image.close()


# =========================================
# APPLICATION LIFESPAN
# =========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Warm-up before accepting user requests
    await run_in_threadpool(warmup_model)

    yield

    print("Real-Time Backend Shutting Down...")


app = FastAPI(
    title="Aura Edge AI - Real-Time Backend",
    lifespan=lifespan
)


# =========================================
# CORS CONFIGURATION
# =========================================

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


# =========================================
# HOME ENDPOINT
# =========================================

@app.get("/")
def home():

    return {
        "message": "Aura Edge AI Real-Time Backend is running!",
        "model_status": "ready"
    }


# =========================================
# REAL-TIME PREDICTION
# =========================================

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

    try:

        # INT8 OpenVINO GPU Inference
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

                "confidence": round(
                    confidence, 4
                ),

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

    finally:

        image.close()
