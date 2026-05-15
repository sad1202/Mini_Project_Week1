import time
from queue import Empty, Queue

import cv2
from PyQt5.QtCore import QMutex, QThread, pyqtSignal

from cam_ai.model.person_detector import PersonDetector, detect_frames_in_roi
from cam_ai.threads.pipeline_data import (
    Detection,
    FramePacket,
    ProcessedPacket,
    put_latest,
)


class ProcessThread(QThread):
    

    metrics_ready = pyqtSignal(object)

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        model_path: str | None = None,
        model_format: str | None = None,
        mode: str = "none",
        conf_threshold: float = 0.35,
        roi: tuple[float, float, float, float] | None = None,
        imgsz: int = 320,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detector = PersonDetector(
            model_path=model_path,
            model_format=model_format,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            strict_model=model_path is not None or model_format is not None,
        )
        self.running = True
        self.mode = mode
        self.roi = roi
        self._mode_mutex = QMutex()
        self._fps_history: list[float] = []

    def run(self) -> None:
        while self.running:
            try:
                packet: FramePacket = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            start = time.perf_counter()
            mode = self.get_mode()
            display_frame = self._apply_mode(packet.frame, mode)
            detections = self._detect_in_roi(packet.frame)
            process_fps = self._smooth_fps(1.0 / max(time.perf_counter() - start, 1e-6))

            processed = ProcessedPacket(
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                frame=display_frame,
                timestamp=packet.timestamp,
                capture_fps=packet.capture_fps,
                process_fps=process_fps,
                detections=detections,
                processing_mode=mode,
            )
            put_latest(self.output_queue, processed)
            self.metrics_ready.emit(
                {
                    "process_fps": process_fps,
                    "people_count": len(detections),
                    "processing_mode": mode,
                    "detector": self.detector.backend,
                }
            )


    def _detect_in_roi(self, frame) -> list[Detection]:
        return detect_frames_in_roi(self.detector, [frame], self.roi)[0]

    def _resolve_roi(self, frame_shape) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.roi
        if all(0.0 <= value <= 1.0 for value in self.roi):
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

        left = max(0, min(w - 1, int(x1)))
        top = max(0, min(h - 1, int(y1)))
        right = max(left + 1, min(w, int(x2)))
        bottom = max(top + 1, min(h, int(y2)))
        return left, top, right, bottom

    def _apply_mode(self, frame, mode: str):
        if mode == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if mode == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        return frame.copy()

    def _smooth_fps(self, fps: float) -> float:
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)

    def set_mode(self, mode: str) -> None:
        if mode not in {"none", "grayscale", "blur"}:
            return
        self._mode_mutex.lock()
        try:
            self.mode = mode
        finally:
            self._mode_mutex.unlock()

    def get_mode(self) -> str:
        self._mode_mutex.lock()
        try:
            return self.mode
        finally:
            self._mode_mutex.unlock()

    def stop(self) -> None:
        self.running = False
        self.wait(1500)


class BatchProcessThread(QThread):
    

    metrics_ready = pyqtSignal(object)

    def __init__(
        self,
        streams: list[tuple[str, Queue, Queue]],
        model_path: str | None = None,
        model_format: str | None = None,
        mode: str = "none",
        conf_threshold: float = 0.35,
        roi: tuple[float, float, float, float] | None = None,
        imgsz: int = 320,
    ):
        super().__init__()
        self.streams = streams
        self.detector = PersonDetector(
            model_path=model_path,
            model_format=model_format,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            strict_model=model_path is not None or model_format is not None,
        )
        self.running = True
        self.mode = mode
        self.roi = roi
        self._mode_mutex = QMutex()
        self._fps_history: list[float] = []

    def run(self) -> None:
        while self.running:
            packets = []
            output_queues = []
            for _camera_id, input_queue, output_queue in self.streams:
                try:
                    packet: FramePacket = input_queue.get(timeout=0.2)
                except Empty:
                    packets = []
                    break
                packets.append(packet)
                output_queues.append(output_queue)

            if not packets:
                continue

            start = time.perf_counter()
            mode = self.get_mode()
            display_frames = [self._apply_mode(packet.frame, mode) for packet in packets]
            detections_by_frame = detect_frames_in_roi(
                self.detector,
                [packet.frame for packet in packets],
                self.roi,
            )
            process_fps = self._smooth_fps(1.0 / max(time.perf_counter() - start, 1e-6))

            for packet, display_frame, detections, output_queue in zip(
                packets,
                display_frames,
                detections_by_frame,
                output_queues,
            ):
                processed = ProcessedPacket(
                    camera_id=packet.camera_id,
                    frame_id=packet.frame_id,
                    frame=display_frame,
                    timestamp=packet.timestamp,
                    capture_fps=packet.capture_fps,
                    process_fps=process_fps,
                    detections=detections,
                    processing_mode=mode,
                )
                put_latest(output_queue, processed)
                self.metrics_ready.emit(
                    {
                        "camera_id": packet.camera_id,
                        "process_fps": process_fps,
                        "people_count": len(detections),
                        "processing_mode": mode,
                        "detector": self.detector.backend,
                    }
                )


    def _apply_mode(self, frame, mode: str):
        if mode == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if mode == "blur":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        return frame.copy()

    def _smooth_fps(self, fps: float) -> float:
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)

    def set_mode(self, mode: str) -> None:
        if mode not in {"none", "grayscale", "blur"}:
            return
        self._mode_mutex.lock()
        try:
            self.mode = mode
        finally:
            self._mode_mutex.unlock()

    def get_mode(self) -> str:
        self._mode_mutex.lock()
        try:
            return self.mode
        finally:
            self._mode_mutex.unlock()

    def stop(self) -> None:
        self.running = False
        self.wait(1500)
