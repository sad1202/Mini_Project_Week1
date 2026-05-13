import sys
import time 
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from queue import Queue
class CameraWorker(QObject):
    frame_ready = pyqtSignal(object)
    fps_update = pyqtSignal(float)
    disconnected = pyqtSignal()
    connect = pyqtSignal()
    finished = pyqtSignal()
    
    
    def __init__(self, source, num_attempts=5, queue=None):
        super().__init__()
        self.source = source
        self.running = False
        self.num_attempts = num_attempts
        self.attempts = 0
        self.queue = queue
    def run(self):
        while self.running:
            cap = cv2.VideoCapture(self.source)
            cnt_frame = 0
            if not cap.isOpened():
                self.attempts += 1
                if self.attempts >= self.num_attempts:
                    print(f"Failed to connect after {self.attempts} attempts. Stopping.")
                    self.running = False    
                    break
                self.disconnected.emit()
                time.sleep(3)
                continue
            self.connect.emit()
            prev_time = time.time()
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                self.queue.put(frame)
                if self.queue.qsize() > 1:
                    self.queue.get()
                if not ret:
                    print("Stream ended or error occurred")
                    self.disconnected.emit()
                    break
                cnt_frame += 1
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                self.fps_update.emit(fps)
                
                self.frame_ready.emit(frame)
            cap.release()
            time.sleep(3)
        self.finished.emit()
        
        def stop(self):
            self.running = False
            