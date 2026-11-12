
import os
import io
import threading
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

MODEL_DEVICE = "intel:gpu"

MAX_IMAGE_BYTES = 10 * 1024 * 1024

# Prevent simultaneous inference on the same YOLO model.
inference_lock = threading.Lock()

model = YOLO(
    MODEL_PATH,
    task="detect"
)

print("YOLO11 INT8 OpenVINO Model Initialized!")


# =========================================
# MODEL WARM-UP
# =========================================

def warmup_model():

    print("======================================")
    print("Starting OpenVINO GPU Warm-up...")
    print("======================================")

    warmup_image = Image.new(
        "RGB",
        (640, 640),
        color=(0, 0, 0)
    )

    try:

        with inference_lock:

            model.predict(
                source=warmup_image,
                imgsz=640,
                conf=0.25,
                iou=0.1,
                device=MODEL_DEVICE,
                verbose=False
            )

        print("======================================")
        print("OpenVINO GPU Warm-up Completed!")
        print("Model is Ready for Real-Time Detection.")
        print("======================================")

    except Exception as error:

        print("OpenVINO Warm-up Failed!")
        print(repr(error))

        raise

    finally:

        warmup_image.close()


# =========================================
# APPLICATION LIFESPAN
# =========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

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
        "model_status": "ready",
        "device": MODEL_DEVICE
    }


# =========================================
# REAL-TIME PREDICTION
# =========================================

@app.post("/realtime/predict")
def realtime_predict(file: UploadFile = File(...)):

    image = None

    try:

        # Read uploaded image with a size limit
        image_bytes = file.file.read(
            MAX_IMAGE_BYTES + 1
        )

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Empty image file."
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:

            raise HTTPException(
                status_code=413,
                detail="Image file is too large. Maximum size is 10 MB."
            )

        # Decode and convert to RGB
        try:

            with Image.open(
                io.BytesIO(image_bytes)
            ) as source_image:

                source_image.load()

                image = source_image.convert("RGB")

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):

            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )

        # =====================================
        # YOLO INT8 OPENVINO GPU INFERENCE
        # =====================================

        with inference_lock:

            results = model.predict(
                source=image,
                imgsz=640,
                conf=0.25,
                iou=0.1,
                device=MODEL_DEVICE,
                verbose=False
            )

        if not results:

            raise HTTPException(
                status_code=500,
                detail="AI model returned no prediction result."
            )

        result = results[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                detections.append({

                    "class": model.names[class_id],

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "box": {

                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),

                    }

                })

        speed = result.speed or {}

        return {

            "detections": detections,

            "performance": {

                "preprocess_ms": round(
                    speed.get("preprocess", 0),
                    2
                ),

                "inference_ms": round(
                    speed.get("inference", 0),
                    2
                ),

                "postprocess_ms": round(
                    speed.get("postprocess", 0),
                    2
                ),

            }

        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            "Real-Time Prediction Error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail="Real-time AI prediction failed."
        )

    finally:

        if image is not None:

            image.close()
