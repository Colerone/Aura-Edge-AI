import os
import shutil
import uuid
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI()

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


model = YOLO("best_v4.pt")

# Folder for uploaded and result images
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Aura Edge AI Backend is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):


    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"

    original_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save uploaded image
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    # results = model(original_path, conf=0.1, iou=0.1)
    results = model.predict(
        source=original_path,
        imgsz=640,
        conf=0.1,
        iou=0.1,
        verbose=False
    )

    result = results[0]

    result_filename = f"result_{unique_filename}"
    result_path = os.path.join(RESULT_DIR, result_filename)

    result.save(filename=result_path)

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "class": model.names[class_id],
                "confidence": round(confidence, 4),
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            }
        )

    # return {
    #     "detections": detections,
    #     "original_image": f"/image/original/{unique_filename}",
    #     "result_image": f"/image/result/{result_filename}",
    # }

    return {
        "detections": detections,

        "original_image": f"/image/original/{unique_filename}",
        "result_image": f"/image/result/{result_filename}",

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
        },
    }


@app.get("/image/original/{filename}")
def get_original_image(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.get("/image/result/{filename}")
def get_result_image(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)