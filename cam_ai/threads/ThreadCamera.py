import sys
import time
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from queue import Queue


class CameraWorker(QObject):
    frame_ready = pyqtSignal(object)
    smooth_fps_update = pyqtSignal(float)
    disconnected = pyqtSignal()
    connect = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, source=0, num_attempts=5, queue=None, max_queue_size=4):
        super().__init__()
        self.source = source
        self.running = True
        self.num_attempts = num_attempts
        self.attempts = 0
        self.queue = queue
        self.queue.maxsize = max_queue_size
        self.queue_fps_history = []

    def run(self):
        while self.running:
            cap = cv2.VideoCapture(self.source)
            print(
                f"Attempting to connect to {self.source} (Attempt {self.attempts + 1}/{self.num_attempts})"
            )
            cnt_frame = 0
            if not cap.isOpened():
                self.attempts += 1
                if self.attempts >= self.num_attempts:
                    print(
                        f"Failed to connect after {self.attempts} attempts. Stopping."
                    )
                    self.running = False
                    break
                self.disconnected.emit()
                time.sleep(3)
                continue
            self.connect.emit()
            prev_time = time.time()
            while self.running and cap.isOpened():
                # print(f"Queue size: {self.queue.qsize()}")
                ret, frame = cap.read()
                # frame = cv2.flip(frame, 1)
                self.queue.put(frame)
                if self.queue.qsize() > self.queue.maxsize:
                    self.queue.get()
                if not ret:
                    print("Stream ended or error occurred")
                    self.disconnected.emit()
                    break
                cnt_frame += 1
                current_time = time.time()
                delta_time = current_time - prev_time
                if delta_time > 0:
                    fps = 1 / delta_time
                    self.queue_fps_history.append(fps)
                else:
                    fps = 0.0
                if len(self.queue_fps_history) > 30:
                    self.queue_fps_history.pop(0)
                prev_time = current_time
                smooth_fps = sum(self.queue_fps_history) / len(self.queue_fps_history)
                self.smooth_fps_update.emit(smooth_fps)
                frame = cv2.resize(frame, (320, 320))
                self.frame_ready.emit(frame)
            cap.release()
            time.sleep(3)
        self.finished.emit()

    def stop(self):
        self.running = False


class TestCameraUI(QWidget):
    def __init__(self, queue):
        super().__init__()
        self.label_fps = QLabel("FPS: 0")
        self.label = QLabel("Frame")
        layout = QVBoxLayout()
        layout.addWidget(self.label_fps)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.worker_camera = CameraWorker(
            source="rtsp://localhost:8554/live", queue=queue, max_queue_size=4
        )

        self.thread_camera = QThread()
        self.worker_camera.moveToThread(self.thread_camera)
        self.thread_camera.started.connect(self.worker_camera.run)
        self.worker_camera.frame_ready.connect(self.display_frame)
        self.worker_camera.smooth_fps_update.connect(self.update_fps)
        self.thread_camera.start()

    def update_fps(self, fps):
        self.label_fps.setText(f"FPS: {fps:.2f}")
        print(f"FPS: {fps:.2f}")

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    queue = Queue()
    test_ui = TestCameraUI(queue)
    test_ui.show()
    sys.exit(app.exec_())
