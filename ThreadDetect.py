from ultralytics import YOLO
# import torch
import sys
import time
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from queue import Queue
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from ThreadCamera import CameraWorker
from PyQt5.QtGui import QPixmap, QImage
import cv2

model = YOLO("yolov8n.pt")

class DetectWorker(QObject):
    frame_ready = pyqtSignal(object)
    fps_update = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = False

    def run(self):
        prev_time = time.time()
        while self.running:
            if not self.queue.empty():
                frame = self.queue.get()
                model_results = model(frame)
                annotated_frame = model_results[0].plot()
                self.frame_ready.emit(annotated_frame)
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                self.fps_update.emit(fps)
        self.finished.emit()

    def stop(self):
        self.running = False


class TestDetectUI(QWidget):
    def __init__(self, queue):
        super().__init__()
        self.label_fps = QLabel("FPS: 0")
        self.label = QLabel("Frame")
        layout = QVBoxLayout()
        layout.addWidget(self.label_fps)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.worker_camera = CameraWorker(0, queue=queue)
        self.thread_camera = QThread()
        self.worker_camera.moveToThread(self.thread_camera)
        self.thread_camera.started.connect(self.worker_camera.run)
        # self.worker_camera.frame_ready.connect(self.display_frame)

        self.worker_detect = DetectWorker(queue)
        self.thread_detect = QThread()
        self.worker_detect.moveToThread(self.thread_detect)
        self.thread_detect.started.connect(self.worker_detect.run)
        self.worker_detect.frame_ready.connect(self.display_frame)
        self.worker_detect.fps_update.connect(self.update_fps)
        self.thread_camera.start()
        self.thread_detect.start()

    def update_fps(self, fps):
        self.label_fps.setText(f"FPS: {fps:.2f}")
        print(f"FPS: {fps:.2f}")

    def display_frame(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    queue = Queue(maxsize=4)
    ui = TestDetectUI(queue)
    ui.worker_camera.running = True
    ui.worker_detect.running = True
    ui.show()
    sys.exit(app.exec_())
