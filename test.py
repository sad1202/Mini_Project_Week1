import time
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "yolo26n.pt"  
IMG_SIZE = 320
NUM_FRAMES = 200
BATCH_SIZE = 4


print("Loading model...")
model = YOLO(MODEL_PATH)



frames = [
    np.random.randint(
        0,
        255,
        (IMG_SIZE, IMG_SIZE, 3),
        dtype=np.uint8
    )
    for _ in range(NUM_FRAMES)
]


print("Warmup...")
dummy_batch = [frames[0]] * BATCH_SIZE

for _ in range(5):
    model.predict(
        dummy_batch,
        imgsz=IMG_SIZE,
        device="cpu",
        verbose=False
    )


print("\n==============================")
print("Single")
print("==============================")

start = time.perf_counter()

for frame in frames:
    model.predict(
        frame,
        imgsz=IMG_SIZE,
        device="cpu",
        verbose=False
    )

single_time = time.perf_counter() - start
single_avg = single_time / NUM_FRAMES

print(f"Total time     : {single_time:.4f}s")
print(f"Avg per image  : {single_avg:.4f}s")

print("\n==============================")
print(f"BATCH = {BATCH_SIZE}")
print("==============================")

batches = [
    frames[i:i+BATCH_SIZE]
    for i in range(0, NUM_FRAMES, BATCH_SIZE)
]

start = time.perf_counter()

for batch in batches:
    model.predict(
        batch,
        imgsz=IMG_SIZE,
        device="cpu",
        verbose=False
    )

batch_time = time.perf_counter() - start
batch_avg = batch_time / NUM_FRAMES

print(f"Total time     : {batch_time:.4f}s")
print(f"Avg per image  : {batch_avg:.4f}s")

# =====================================================
# SPEEDUP
# =====================================================
speedup = single_time / batch_time


print(f"Speedup: {speedup:.2f}x")