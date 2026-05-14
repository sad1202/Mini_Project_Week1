import torch
import numpy as np
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.overrides["verbose"] = False

for bs in [1, 2, 4, 8, 16]:
    dummy = torch.rand(bs, 3, 320, 320)

    # warmup
    for _ in range(3):
        model.predictor = None
        _ = model.model(dummy)

    t = time.perf_counter()
    for _ in range(20):
        model.model(dummy)
    ms = (time.perf_counter() - t) / 20 * 1000
    print(f"batch={bs}: {ms:.1f}ms total | {ms/bs:.1f}ms per frame")