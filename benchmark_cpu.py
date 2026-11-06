
import time
import statistics
from pathlib import Path

from ultralytics import YOLO

# =====================================
# CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "best_v4.pt"
IMAGE_PATH = BASE_DIR / "test_img2.jpg"

IMAGE_SIZE = 640
WARMUP_RUNS = 5
TEST_RUNS = 20

# =====================================
# CHECK FILES
# =====================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# =====================================
# LOAD MODEL
# =====================================

print("Loading YOLO11 model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded successfully!")

# =====================================
# WARM-UP
# =====================================

print("\nWarming up CPU...")

for _ in range(WARMUP_RUNS):
    model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        device="cpu",
        conf=0.25,
        verbose=False
    )

print("Warm-up completed!")

# =====================================
# BENCHMARK
# =====================================

print("\nStarting CPU Benchmark...")

latencies = []

for i in range(TEST_RUNS):

    start_time = time.perf_counter()

    results = model.predict(
        source=str(IMAGE_PATH),
        imgsz=IMAGE_SIZE,
        device="cpu",
        conf=0.25,
        verbose=False
    )

    elapsed = (
        time.perf_counter() - start_time
    ) * 1000

    latencies.append(elapsed)

    print(
        f"Run {i + 1}/{TEST_RUNS}: "
        f"{elapsed:.2f} ms"
    )

# =====================================
# RESULTS
# =====================================

average_latency = statistics.mean(latencies)
median_latency = statistics.median(latencies)
p95_latency = sorted(latencies)[
    int(0.95 * (len(latencies) - 1))
]

effective_fps = 1000 / average_latency

print("\n" + "=" * 40)
print("YOLO11 CPU BENCHMARK RESULTS")
print("=" * 40)

print(f"Model: YOLO11m")
print(f"Device: CPU")
print(f"Image Size: {IMAGE_SIZE}")

print(f"\nAverage Latency: {average_latency:.2f} ms")
print(f"Median Latency:  {median_latency:.2f} ms")
print(f"P95 Latency:     {p95_latency:.2f} ms")
print(f"Effective FPS:   {effective_fps:.2f}")

# YOLO inference breakdown
print("\nInference Breakdown:")

for key, value in results[0].speed.items():
    print(f"{key}: {value:.2f} ms")

print("\nBenchmark completed!")
