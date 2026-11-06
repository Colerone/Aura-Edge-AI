
import sys
from pathlib import Path
import torch
import ultralytics

print("===== SYSTEM INFORMATION =====")

print("Python Version:", sys.version)
print("PyTorch Version:", torch.__version__)
print("Ultralytics Version:", ultralytics.__version__)

print("\n===== DEVICE INFORMATION =====")

print("CUDA Available:", torch.cuda.is_available())
print("CPU Available:", "cpu" in torch.device("cpu").type)

print("\n===== MODEL INFORMATION =====")

model_path = Path(__file__).parent / "best_v2.pt"

print("Model Path:", model_path)
print("Model Exists:", model_path.exists())

print("\n===== CHECK COMPLETED =====")
