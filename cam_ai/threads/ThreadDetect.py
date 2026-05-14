from ultralytics import YOLO
import sys
import cv2
import time
import numpy as np
from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import os
# --- THREAD 1: CHỈ ĐỌC FRAME (CAPTURE) ---
class CaptureWorker(QThread):
    def __init__(self, source, queue):
        super().__init__()
        self.source = source
        self.queue = queue
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                cap.open(self.source)
                time.sleep(1)
                continue
            
            if self.queue.full():
                try: self.queue.get_nowait()
                except Empty: pass
            frame = cv2.resize(frame, (320, 320))
            self.queue.put(frame)
        cap.release()

    def stop(self):
        self.running = False

# --- THREAD 2: XỬ LÝ BATCH INFERENCE + TÍNH FPS ---
class InferenceWorker(QThread):
    # Trả về list frame và con số FPS
    batch_ready = pyqtSignal(list, float)

    def __init__(self, queues, model_path='yolo26n.onnx'):
        super().__init__()
        self.queues = queues
        self.model = YOLO(model_path, task='detect')
        self.running = True
        self.prev_time = 0
        self.fps_list = [] # Lưu lịch sử để tính trung bình

    def run(self):
        self.prev_time = time.perf_counter()
        while self.running:
            batch_frames = []
            for q in self.queues:
                try:
                    frame = q.get(timeout=0.1)
                    batch_frames.append(frame)
                except Empty:
                    break

            if len(batch_frames) == len(self.queues):
                # Dự đoán Batch 4
                results = self.model.predict(batch_frames, stream=False, imgsz=320, device='cpu')

                processed_frames = [r.plot() for r in results]
                
                # TÍNH FPS TRUNG BÌNH
                curr_time = time.perf_counter()
                delta_time = curr_time - self.prev_time
                self.prev_time = curr_time
                
                if delta_time > 0:
                    actual_fps = 1.0 / delta_time
                    self.fps_list.append(actual_fps)
                    if len(self.fps_list) > 20: self.fps_list.pop(0)
                    avg_fps = sum(self.fps_list) / len(self.fps_list)
                    
                    # Phát tín hiệu kèm FPS
                    self.batch_ready.emit(processed_frames, avg_fps)

    def stop(self):
        self.running = False

# --- GIAO DIỆN CHÍNH ---
class MultiCamApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Monitoring 4-Cam | Ryzen 5 6600H INT8 Demo")
        self.main_layout = QGridLayout(self)
        self.setStyleSheet("background-color: #121212; color: white;")
        
        self.sources = [
            "rtsp://localhost:8554/cam1",
            "rtsp://localhost:8554/cam2",
            "rtsp://localhost:8554/cam3",
            "rtsp://localhost:8554/cam4"
        ]
        
        self.labels = []
        self.fps_labels = []
        self.queues = []
        self.capture_workers = []

        for i in range(4):
            # Container cho mỗi Camera (để chứa Video + Text FPS đè lên)
            container = QWidget()
            v_layout = QVBoxLayout(container)
            v_layout.setContentsMargins(5, 5, 5, 5)

            # Label hiển thị Video
            video_lbl = QLabel(f"Camera {i+1}")
            video_lbl.setFixedSize(600, 340)
            video_lbl.setScaledContents(True)
            video_lbl.setStyleSheet("border: 2px solid #333; background-color: black;")
            
            # Label hiển thị FPS riêng cho mỗi cam
            fps_lbl = QLabel("FPS: --")
            fps_lbl.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 14px;")
            
            v_layout.addWidget(video_lbl)
            v_layout.addWidget(fps_lbl)
            
            self.main_layout.addWidget(container, i // 2, i % 2)
            
            self.labels.append(video_lbl)
            self.fps_labels.append(fps_lbl)
            
            q = Queue(maxsize=1)
            self.queues.append(q)
            
            worker = CaptureWorker(self.sources[i], q)
            self.capture_workers.append(worker)
            worker.start()

        self.inference_thread = InferenceWorker(self.queues)
        self.inference_thread.batch_ready.connect(self.update_ui)
        self.inference_thread.start()

    def update_ui(self, processed_frames, fps):
        for i, frame in enumerate(processed_frames):
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            # QUAN TRỌNG: Phải có .copy() ở cuối
            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.labels[i].setPixmap(QPixmap.fromImage(qt_img))
            self.fps_labels[i].setText(f"FPS: {fps:.2f}")

    def closeEvent(self, event):
        for w in self.capture_workers: w.stop()
        self.inference_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiCamApp()
    window.showMaximized() # Mở toàn màn hình cho nó "pro"
    sys.exit(app.exec_())