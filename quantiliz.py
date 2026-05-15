from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "yolo26n.pt"
OUTPUT_DIR = "openvino"
model = YOLO(MODEL_PATH)

model.export(
    format="openvino",
    int8=False,
    imgsz=320,
    dynamic=False,
    half=True,
    batch=4,
    data="coco8.yaml"
)



export_path = Path(MODEL_PATH).stem + "_openvino_model"

print(f"{export_path}")