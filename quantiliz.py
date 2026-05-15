from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.export(
    format="openvino",
    int8=True,
    imgsz=320
)