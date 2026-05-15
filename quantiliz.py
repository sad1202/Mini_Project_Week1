from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = "yolo26n.pt"


OUTPUT_DIR = "openvino_int8"


model = YOLO(MODEL_PATH)

model.export(
    format="openvino",
    int8=True,
    imgsz=320,
    dynamic=False,
    half=False,
    data="coco8.yaml",
    batch=4,
)

print("\nDone export OpenVINO INT8")

export_path = Path(MODEL_PATH).stem + "_openvino_model"

print(f"\nExport folder: {export_path}")