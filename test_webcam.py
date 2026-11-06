import time
import cv2
from ultralytics import YOLO

# 1. Load the trained YOLO model
model = YOLO("best_v4.pt")

# 2. Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started... Press 'q' on your keyboard to stop.")

prev_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # 3. Perform prediction (imgsz=480 for balance between speed and clarity)
        results = model(frame, imgsz=480, conf=0.1, iou=0.1)
        annotated_frame = results[0].plot()

        # 4. Extract speed metrics (preprocess + inference + postprocess)
        speed_dict = results[0].speed
        total_latency = (
            speed_dict.get("preprocess", 0)
            + speed_dict.get("inference", 0)
            + speed_dict.get("postprocess", 0)
        )

        # 5. Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # 6. Overlay FPS and Latency on the frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated_frame,
            f"Latency: {total_latency:.1f}ms",
            (30, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # 7. Display the live feed
        cv2.imshow("AURA Edge - Real-Time Skin Disease Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Closing webcam...")
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Successfully closed.")