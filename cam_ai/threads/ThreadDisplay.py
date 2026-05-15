import time
from queue import Empty, Queue

from PyQt5.QtCore import QThread, pyqtSignal

from cam_ai.threads.pipeline_data import DisplayPacket


class DisplayThread(QThread):

    frame_ready = pyqtSignal(object)
    metrics_ready = pyqtSignal(object)

    def __init__(self, input_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.running = True
        self._fps_history: list[float] = []
        self._last_time = time.perf_counter()

    def run(self) -> None:
        while self.running:
            try:
                packet: DisplayPacket = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            now = time.perf_counter()
            delta = now - self._last_time
            self._last_time = now
            packet.display_fps = self._smooth_fps(1.0 / delta if delta > 0 else 0.0)
            packet.latency_ms = max(0.0, (time.time() - packet.timestamp) * 1000.0)

            self.frame_ready.emit(packet)
            self.metrics_ready.emit(
                {
                    "display_fps": packet.display_fps,
                    "latency_ms": packet.latency_ms,
                    "capture_fps": packet.capture_fps,
                    "process_fps": packet.process_fps,
                    "tracking_fps": packet.tracking_fps,
                    "people_count": packet.people_count,
                    "track_count": packet.track_count,
                    "processing_mode": packet.processing_mode,
                }
            )


    def _smooth_fps(self, fps: float) -> float:
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)

    def stop(self) -> None:
        self.running = False
        self.wait(1500)
