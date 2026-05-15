import os
import time
from pathlib import Path
from queue import Queue
from typing import Any

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from cam_ai.threads.pipeline_data import FramePacket, put_latest


class CaptureThread(QThread):
   
    fps_updated = pyqtSignal(float)
    source_fps_ready = pyqtSignal(float)

    def __init__(
        self,
        source: Any,
        output_queue: Queue,
        camera_id: str = "camera-1",
        frame_size: tuple[int, int] | None = (640, 360),
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 0,
    ):
        super().__init__()
        self.source = self._normalize_source(source)
        self.output_queue = output_queue
        self.camera_id = camera_id
        self.frame_size = frame_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.running = True
        self._fps_history: list[float] = []

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    @staticmethod
    def _normalize_source(source: Any) -> Any:
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source

    @staticmethod
    def _is_file_source(source: Any) -> bool:
        if not isinstance(source, str):
            return False
        lowered = source.lower()
        if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return False
        return Path(source).exists()

    @staticmethod
    def _frame_interval(cap: cv2.VideoCapture, source: Any) -> float | None:
        if not CaptureThread._is_file_source(source):
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            return None
        return 1.0 / fps

    def run(self) -> None:
        frame_id = 0
        reconnect_attempts = 0

        while self.running:
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                reconnect_attempts += 1
                if self.max_reconnect_attempts and reconnect_attempts >= self.max_reconnect_attempts:
                    break
                time.sleep(self.reconnect_delay)
                continue

            reconnect_attempts = 0
            frame_interval = self._frame_interval(cap, self.source)
            if frame_interval is not None:
                self.source_fps_ready.emit(1.0 / frame_interval)
            else:
                self.source_fps_ready.emit(0.0)

            self._fps_history.clear()
            last_frame_time = time.perf_counter()

            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                if self.frame_size:
                    frame = cv2.resize(frame, self.frame_size)

                now = time.perf_counter()
                elapsed = now - last_frame_time
                if frame_interval is not None and elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    now = time.perf_counter()
                delta = now - last_frame_time
                last_frame_time = now
                fps = self._smooth_fps(1.0 / delta if delta > 0 else 0.0)

                frame_id += 1
                packet = FramePacket(
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    frame=frame,
                    timestamp=time.time(),
                    capture_fps=fps,
                )
                put_latest(self.output_queue, packet)
                self.fps_updated.emit(fps)

            cap.release()

            if self.running:
                time.sleep(self.reconnect_delay)


    def _smooth_fps(self, fps: float) -> float:
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)

    def stop(self) -> None:
        self.running = False
        self.wait(1500)



CameraWorker = CaptureThread
