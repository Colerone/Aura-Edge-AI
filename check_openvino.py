
import openvino as ov

print("===== OPENVINO INFORMATION =====")

print("OpenVINO Version:", ov.__version__)

core = ov.Core()

print("\n===== AVAILABLE DEVICES =====")

devices = core.available_devices

for device in devices:
    print("Device:", device)

print("\n===== CPU INFORMATION =====")

if "CPU" in devices:
    cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
    print("CPU Name:", cpu_name)
    print("CPU Plugin: Available")
else:
    print("CPU Plugin: Not Available")

print("\n===== CHECK COMPLETED =====")
