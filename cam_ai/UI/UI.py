import sys
import time
import cv2
from cam_ai.threads.ThreadCamera import CameraWorker
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
class TestUI(QWidget):
    def __init__(self):
        super().__init__()
        self.label_fps = QLabel("FPS: 0")
        self.label = QLabel("Frame")
        layout = QVBoxLayout()
        layout.addWidget(self.label_fps)
        layout.addWidget(self.label)
        self.setLayout(layout)
    
        self.worker = CameraWorker(0) 
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.display_frame)
        self.worker.fps_update.connect(self.update_fps)
        self.worker.disconnected.connect(self.handle_disconnection)
        self.worker.connect.connect(self.handle_connection)
        self.worker.finished.connect(self.thread.quit)
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
    def handle_disconnection(self):
        print("Camera disconnected")
    def handle_connection(self):
        print("Camera connected")