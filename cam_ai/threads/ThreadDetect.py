import sys
import cv2
import time
import os

from ultralytics import YOLO

from queue import Queue, Empty

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QGridLayout,
    QWidget,
    QVBoxLayout,
)
from PyQt5.QtGui import QPixmap, QImage


# =========================================================
# THREAD 1: CAPTURE RTSP
# =========================================================
class CaptureWorker(QThread):

    def __init__(self, source, queue):
        super().__init__()

        self.source = source
        self.queue = queue
        self.running = True

    def run(self):

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:

            ret, frame = cap.read()

            if not ret:
                print(f"Reconnect: {self.source}")

                cap.release()

                time.sleep(1)

                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

                continue

            frame = cv2.resize(frame, (320, 320))

            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass

            self.queue.put(frame)

        cap.release()

    def stop(self):
        self.running = False


class InferenceWorker(QThread):

    batch_ready = pyqtSignal(list, float, float)

    def __init__(self, queues, model_path="yolo26n.pt"):
        super().__init__()

        self.queues = queues

        self.model = YOLO(model_path, task="detect")

        self.running = True

        self.prev_time = 0

        self.fps_history = []

        self.latency_history = []

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

            # phải đủ 4 frame
            if len(batch_frames) != len(self.queues):
                continue

            # =================================================
            # LATENCY START
            # =================================================
            infer_start = time.perf_counter()

            results = self.model.predict(
                batch_frames,
                stream=False,
                imgsz=320,
                device="cpu",
                verbose=False,
                classes=[0],
            )

            infer_end = time.perf_counter()

            # =================================================
            # LATENCY
            # =================================================
            latency_ms = (infer_end - infer_start) * 1000

            self.latency_history.append(latency_ms)

            if len(self.latency_history) > 30:
                self.latency_history.pop(0)

            avg_latency = sum(self.latency_history) / len(self.latency_history)

            # =================================================
            # DRAW
            # =================================================
            processed_frames = [r.plot() for r in results]

            # =================================================
            # FPS
            # =================================================
            current_time = time.perf_counter()

            delta = current_time - self.prev_time

            self.prev_time = current_time

            if delta > 0:

                fps = 1.0 / delta

                self.fps_history.append(fps)

                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)

                avg_fps = sum(self.fps_history) / len(self.fps_history)

                print(f"FPS: {avg_fps:.2f} | " f"Latency: {avg_latency:.2f} ms")

                self.batch_ready.emit(processed_frames, avg_fps, avg_latency)

    def stop(self):
        self.running = False


# =========================================================
# UI
# =========================================================
class MultiCamApp(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("4 CAM AI MONITORING | OPENVINO INT8")

        self.setStyleSheet("background-color: #121212; color: white;")

        self.main_layout = QGridLayout(self)

        # =================================================
        # RTSP SOURCES
        # =================================================
        self.sources = [
            "rtsp://localhost:8554/cam1",
            "rtsp://localhost:8554/cam2",
            "rtsp://localhost:8554/cam3",
            "rtsp://localhost:8554/cam4",
        ]

        self.labels = []

        self.info_labels = []

        self.queues = []

        self.capture_workers = []

        # =================================================
        # CREATE UI
        # =================================================
        for i in range(4):

            container = QWidget()

            v_layout = QVBoxLayout(container)

            v_layout.setContentsMargins(5, 5, 5, 5)

            # VIDEO LABEL
            video_lbl = QLabel(f"Camera {i+1}")

            video_lbl.setFixedSize(640, 360)

            video_lbl.setScaledContents(True)

            video_lbl.setStyleSheet("""
                border: 2px solid #333;
                background-color: black;
                """)

            # INFO LABEL
            info_lbl = QLabel("FPS: -- | Latency: --")

            info_lbl.setStyleSheet("""
                color: #00FF00;
                font-size: 14px;
                font-weight: bold;
                """)

            v_layout.addWidget(video_lbl)

            v_layout.addWidget(info_lbl)

            self.main_layout.addWidget(container, i // 2, i % 2)

            self.labels.append(video_lbl)

            self.info_labels.append(info_lbl)

            # =================================================
            # QUEUE
            # =================================================
            q = Queue(maxsize=1)

            self.queues.append(q)

            # =================================================
            # CAPTURE THREAD
            # =================================================
            worker = CaptureWorker(self.sources[i], q)

            self.capture_workers.append(worker)

            worker.start()

        # =================================================
        # INFERENCE THREAD
        # =================================================
        self.inference_thread = InferenceWorker(self.queues, model_path="yolo26n_openvino_model")

        self.inference_thread.batch_ready.connect(self.update_ui)

        self.inference_thread.start()

    # =====================================================
    # UPDATE UI
    # =====================================================
    def update_ui(self, processed_frames, fps, latency):

        for i, frame in enumerate(processed_frames):

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb.shape

            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

            self.labels[i].setPixmap(QPixmap.fromImage(qt_img))

            self.info_labels[i].setText(
                f"FPS: {fps:.2f} | " f"Latency: {latency:.1f} ms"
            )

    # =====================================================
    # CLOSE EVENT
    # =====================================================
    def closeEvent(self, event):

        for worker in self.capture_workers:
            worker.stop()

        self.inference_thread.stop()

        event.accept()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = MultiCamApp()

    window.showMaximized()

    sys.exit(app.exec_())
