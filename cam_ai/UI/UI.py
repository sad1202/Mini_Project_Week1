from pathlib import Path
from queue import Queue

try:
    import onnxruntime as _onnxruntime  # noqa: F401
except Exception:
    _onnxruntime = None

try:
    import torch as _torch  # noqa: F401
except Exception:
    _torch = None

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cam_ai.threads.ThreadCamera import CaptureThread
from cam_ai.threads.ThreadDisplay import DisplayThread
from cam_ai.threads.ThreadEvent import EventThread
from cam_ai.threads.ThreadProcess import BatchProcessThread, ProcessThread
from cam_ai.threads.ThreadTracking import BatchTrackingThread, TrackingThread
from cam_ai.threads.pipeline_data import DisplayPacket, Event


class CameraPanel(QWidget):
    def __init__(
        self,
        index: int,
        source,
        model_path: str | None = None,
        model_format: str | None = None,
        mode: str = "none",
        roi: tuple[float, float, float, float] = (0.25, 0.2, 0.75, 0.9),
        loiter_seconds: float = 5.0,
        crowd_threshold: int = 3,
        imgsz: int = 320,
        build_process_thread: bool = True,
        build_tracking_thread: bool = True,
        process_mode_changed=None,
        parent=None,
    ):
        super().__init__(parent)
        self.index = index
        self.camera_id = f"camera-{index + 1}"
        self.source = source
        self.model_path = model_path
        self.model_format = model_format
        self.mode = mode
        self.roi = roi
        self.loiter_seconds = loiter_seconds
        self.crowd_threshold = crowd_threshold
        self.imgsz = imgsz
        self.build_process_thread = build_process_thread
        self.build_tracking_thread = build_tracking_thread
        self.process_mode_changed = process_mode_changed
        self.metric_values = {
            "source_fps": "--",
            "capture_fps": "--",
            "process_fps": "--",
            "tracking_fps": "--",
            "display_fps": "--",
            "latency_ms": "--",
        }

        self._build_ui()
        self._build_pipeline()
        self._connect_pipeline()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.title_label = QLabel(f"Camera {self.index + 1}")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("None", "none")
        self.mode_combo.addItem("Gray", "grayscale")
        self.mode_combo.addItem("Blur", "blur")
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(self.mode)))
        header.addWidget(self.title_label, stretch=1)
        header.addWidget(self.mode_combo)
        layout.addLayout(header)

        self.video_label = QLabel("Waiting for frame")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(420, 236)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label, stretch=1)

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)

        self.event_list = QListWidget()
        self.event_list.setMaximumHeight(76)
        layout.addWidget(self.event_list)

        self._render_metrics()

    def _build_pipeline(self) -> None:
        self.capture_queue = Queue(maxsize=2)
        self.process_queue = Queue(maxsize=2)
        self.tracking_queue = Queue(maxsize=2)
        self.event_queue = Queue(maxsize=2)

        self.capture_thread = CaptureThread(
            self.source,
            self.capture_queue,
            camera_id=self.camera_id,
            frame_size=(640, 360),
        )
        self.process_thread = None
        if self.build_process_thread:
            self.process_thread = ProcessThread(
                self.capture_queue,
                self.process_queue,
                model_path=self.model_path,
                model_format=self.model_format,
                mode=self.mode_combo.currentData(),
                roi=self.roi,
                imgsz=self.imgsz,
            )
        self.tracking_thread = None
        if self.build_tracking_thread:
            self.tracking_thread = TrackingThread(self.process_queue, self.tracking_queue)
        self.event_thread = EventThread(
            self.tracking_queue,
            self.event_queue,
            roi=self.roi,
            loiter_seconds=self.loiter_seconds,
            crowd_threshold=self.crowd_threshold,
        )
        self.display_thread = DisplayThread(self.event_queue)
        self.worker_threads = [
            self.capture_thread,
            self.event_thread,
            self.display_thread,
        ]
        if self.process_thread is not None:
            self.worker_threads.insert(1, self.process_thread)
        if self.tracking_thread is not None:
            insert_at = 2 if self.process_thread is not None else 1
            self.worker_threads.insert(insert_at, self.tracking_thread)

    def _connect_pipeline(self) -> None:
        self.capture_thread.source_fps_ready.connect(lambda fps: self.update_metrics({"source_fps": fps}))
        self.capture_thread.fps_updated.connect(lambda fps: self.update_metrics({"capture_fps": fps}))
        if self.process_thread is not None:
            self.process_thread.metrics_ready.connect(self.update_metrics)
        if self.tracking_thread is not None:
            self.tracking_thread.metrics_ready.connect(self.update_metrics)
        self.event_thread.metrics_ready.connect(self.update_metrics)
        self.event_thread.event_ready.connect(self.add_event)
        self.display_thread.frame_ready.connect(self.display_frame)
        self.display_thread.metrics_ready.connect(self.update_metrics)
        self.mode_combo.currentIndexChanged.connect(self.change_process_mode)

    def start_pipeline(self) -> None:
        for thread in self.worker_threads:
            if not thread.isRunning():
                thread.running = True
                thread.start()

    def stop_pipeline(self) -> None:
        for thread in self.worker_threads:
            if thread.isRunning():
                thread.stop()

    def change_process_mode(self) -> None:
        mode = self.mode_combo.currentData()
        if self.process_thread is not None:
            self.process_thread.set_mode(mode)
        elif self.process_mode_changed is not None:
            self.process_mode_changed(mode)

    def display_frame(self, packet: DisplayPacket) -> None:
        rgb_image = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def update_metrics(self, metrics: dict) -> None:
        for key, value in metrics.items():
            if isinstance(value, float):
                self.metric_values[key] = f"{value:.2f}"
            else:
                self.metric_values[key] = str(value)
        self._render_metrics()

    def add_event(self, event: Event) -> None:
        self.event_list.insertItem(0, f"[{event.level.upper()}] {event.message}")
        while self.event_list.count() > 8:
            self.event_list.takeItem(self.event_list.count() - 1)

    def _render_metrics(self) -> None:
        self.metrics_label.setText(
            " | ".join(
                [
                    f"Source FPS: {self.metric_values['source_fps']}",
                    f"Cam FPS: {self.metric_values['capture_fps']}",
                    f"Process: {self.metric_values['process_fps']}",
                    f"Track: {self.metric_values['tracking_fps']}",
                    f"Display: {self.metric_values['display_fps']}",
                    f"Latency: {self.metric_values['latency_ms']} ms",
                ]
            )
        )


class MonitoringWindow(QWidget):
    def __init__(
        self,
        source=None,
        sources: list | None = None,
        model_path: str | None = None,
        model_format: str | None = None,
        mode: str = "none",
        roi: tuple[float, float, float, float] = (0.25, 0.2, 0.75, 0.9),
        loiter_seconds: float = 5.0,
        crowd_threshold: int = 3,
        imgsz: int = 320,
    ):
        super().__init__()
        self.setWindowTitle("AI Camera Monitoring - QThread Queue Demo")
        self.resize(1280, 820)

        if sources is None:
            sources = [source if source is not None else self._default_source()]
        self.sources = sources
        self.panels: list[CameraPanel] = []
        self.panels_by_camera_id: dict[str, CameraPanel] = {}
        self.shared_process_thread = None
        self.shared_tracking_thread = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel(f"AI Camera Monitoring ({len(self.sources)} camera)")
        self.start_button = QPushButton("Start All")
        self.stop_button = QPushButton("Stop All")
        header.addWidget(title, stretch=1)
        header.addWidget(self.start_button)
        header.addWidget(self.stop_button)
        root_layout.addLayout(header)

        grid = QGridLayout()
        grid.setSpacing(10)
        root_layout.addLayout(grid, stretch=1)

        for index, camera_source in enumerate(self.sources):
            panel = CameraPanel(
                index=index,
                source=camera_source,
                model_path=model_path,
                model_format=model_format,
                mode=mode,
                roi=roi,
                loiter_seconds=loiter_seconds,
                crowd_threshold=crowd_threshold,
                imgsz=imgsz,
                build_process_thread=len(self.sources) == 1,
                build_tracking_thread=len(self.sources) == 1,
                process_mode_changed=self.set_shared_process_mode,
            )
            self.panels.append(panel)
            self.panels_by_camera_id[panel.camera_id] = panel
            columns = 2 if len(self.sources) > 1 else 1
            grid.addWidget(panel, index // columns, index % columns)

        if len(self.sources) > 1:
            streams = [
                (panel.camera_id, panel.capture_queue, panel.process_queue)
                for panel in self.panels
            ]
            self.shared_process_thread = BatchProcessThread(
                streams,
                model_path=model_path,
                model_format=model_format,
                mode=mode,
                roi=roi,
                imgsz=imgsz,
            )
            self.shared_process_thread.metrics_ready.connect(self.update_shared_process_metrics)
            tracking_streams = [
                (panel.camera_id, panel.process_queue, panel.tracking_queue)
                for panel in self.panels
            ]
            self.shared_tracking_thread = BatchTrackingThread(tracking_streams)
            self.shared_tracking_thread.metrics_ready.connect(self.update_shared_tracking_metrics)

        self.start_button.clicked.connect(self.start_pipeline)
        self.stop_button.clicked.connect(self.stop_pipeline)
        self.start_pipeline()

    def start_pipeline(self) -> None:
        for panel in self.panels:
            panel.start_pipeline()
        if self.shared_process_thread is not None and not self.shared_process_thread.isRunning():
            self.shared_process_thread.running = True
            self.shared_process_thread.start()
        if self.shared_tracking_thread is not None and not self.shared_tracking_thread.isRunning():
            self.shared_tracking_thread.running = True
            self.shared_tracking_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_pipeline(self) -> None:
        if self.shared_process_thread is not None and self.shared_process_thread.isRunning():
            self.shared_process_thread.stop()
        if self.shared_tracking_thread is not None and self.shared_tracking_thread.isRunning():
            self.shared_tracking_thread.stop()
        for panel in self.panels:
            panel.stop_pipeline()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def set_shared_process_mode(self, mode: str) -> None:
        if self.shared_process_thread is not None:
            self.shared_process_thread.set_mode(mode)
            for panel in self.panels:
                panel.metric_values["processing_mode"] = mode
                panel._render_metrics()

    def update_shared_process_metrics(self, metrics: dict) -> None:
        camera_id = metrics.get("camera_id")
        panel = self.panels_by_camera_id.get(camera_id)
        if panel is None:
            return
        filtered_metrics = {key: value for key, value in metrics.items() if key != "camera_id"}
        panel.update_metrics(filtered_metrics)

    def update_shared_tracking_metrics(self, metrics: dict) -> None:
        camera_id = metrics.get("camera_id")
        panel = self.panels_by_camera_id.get(camera_id)
        if panel is None:
            return
        filtered_metrics = {key: value for key, value in metrics.items() if key != "camera_id"}
        panel.update_metrics(filtered_metrics)

    def closeEvent(self, event) -> None:
        self.stop_pipeline()
        event.accept()

    @staticmethod
    def _default_source():
        root = Path(__file__).resolve().parents[2]
        video = root / "cam_ai" / "assets" / "videos" / "camera_1.mp4"
        if video.exists():
            return str(video)
        return 0

    @staticmethod
    def default_rtsp_sources() -> list[str]:
        return [
            "rtsp://localhost:8554/cam1",
            "rtsp://localhost:8554/cam2",
            "rtsp://localhost:8554/cam3",
            "rtsp://localhost:8554/cam4",
        ]


# Backward-compatible name for old examples.
TestUI = MonitoringWindow
