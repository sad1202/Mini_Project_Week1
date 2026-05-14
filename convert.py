from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Export ONNX FP16
model.export(
    format="onnx",
    batch=4,
    imgsz=320,
    opset=17,
    # half=True
)