import time
from pathlib import Path
from queue import Empty, Queue

try:
    import onnxruntime as ort
except Exception as exc:
    ort = None
    ORT_IMPORT_ERROR = exc
else:
    ORT_IMPORT_ERROR = None

import cv2
import numpy as np
from PyQt5.QtCore import QMutex, QThread, pyqtSignal

from cam_ai.threads.pipeline_data import (
    Detection,
    FramePacket,
    ProcessedPacket,
    put_latest,
)


class PersonDetector:
    """YOLO person detector with an OpenCV HOG fallback for demo resilience."""

    def __init__(
        self,
        model_path: str | None = None,
        model_format: str | None = None,
        conf_threshold: float = 0.35,
        imgsz: int = 320,
        strict_model: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = 0.45
        self.imgsz = imgsz
        self.strict_model = strict_model
        self.backend = "OpenCV HOG"
        self.model = None
        self.session = None
        self.input_name = ""
        self.output_name = ""
        self.input_width = imgsz
        self.input_height = imgsz
        self.batch_size = 1
        self.hog = None

        resolved_model = self._resolve_model(model_path, model_format)
        if resolved_model is not None and resolved_model.suffix.lower() == ".onnx":
            try:
                self._load_onnxruntime(resolved_model)
                return
            except Exception as exc:
                if self.strict_model:
                    raise RuntimeError(f"Cannot load ONNXRuntime detector: {exc}") from exc
                print(f"Cannot load ONNXRuntime detector, trying ultralytics: {exc}")

        if resolved_model is not None:
            try:
                from ultralytics import YOLO

                self.model = YOLO(str(resolved_model), task="detect")
                self.backend = f"YOLO ({resolved_model.name})"
                return
            except Exception as exc:
                if self.strict_model:
                    raise RuntimeError(f"Cannot load YOLO model: {exc}") from exc
                print(f"Cannot load YOLO model, fallback to OpenCV HOG: {exc}")

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _load_onnxruntime(self, model_path: Path) -> None:
        if ort is None:
            raise RuntimeError(f"onnxruntime import failed: {ORT_IMPORT_ERROR}")

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        model_input = self.session.get_inputs()[0]
        model_output = self.session.get_outputs()[0]
        self.input_name = model_input.name
        self.output_name = model_output.name

        shape = model_input.shape
        if len(shape) != 4:
            raise ValueError(f"Unsupported YOLO input shape: {shape}")

        self.batch_size = shape[0] if isinstance(shape[0], int) and shape[0] > 0 else 1
        self.input_height = (
            shape[2] if isinstance(shape[2], int) and shape[2] > 0 else self.imgsz
        )
        self.input_width = (
            shape[3] if isinstance(shape[3], int) and shape[3] > 0 else self.imgsz
        )
        self.backend = f"YOLO ONNXRuntime ({model_path.name})"

    @staticmethod
    def _resolve_model(model_path: str | None, model_format: str | None) -> Path | None:
        root = Path(__file__).resolve().parents[2]

        # ===== User truyền model =====
        if model_path:
            requested = Path(model_path)

            if not requested.is_absolute():
                requested = root / requested

            # Nếu là folder OpenVINO
            if requested.is_dir():
                xml_files = list(requested.glob("*.xml"))

                if len(xml_files) == 0:
                    raise FileNotFoundError(
                        f"No .xml model found inside OpenVINO directory: {requested}"
                    )

                return xml_files[0]

            # Nếu là file
            if not requested.exists():
                raise FileNotFoundError(f"Model file not found: {requested}")

            return requested

        # ===== Auto detect =====
        if model_format == "onnx":
            candidates = [root / "yolo26n.onnx"]

        elif model_format == "pt":
            candidates = [root / "yolo26n.pt"]

        elif model_format == "openvino":
            candidates = list((root / "cam_ai" / "model").glob("**/*.xml"))

        else:
            candidates = [
                root / "yolo26n.onnx",
                root / "yolo26n.pt",
                *list((root / "cam_ai" / "model").glob("**/*.xml")),
                root / "cam_ai" / "assets" / "checkpoint" / "yolov8n.pt",
            ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # ===== Error message =====
        if model_format == "pt":
            raise FileNotFoundError(
                f"PT model requested but not found."
            )

        if model_format == "onnx":
            raise FileNotFoundError(
                f"ONNX model requested but not found."
            )

        if model_format == "openvino":
            raise FileNotFoundError(
                f"OpenVINO XML model not found."
            )

        return None

    def detect(self, frame) -> list[Detection]:
        if self.session is not None:
            return self._detect_onnxruntime(frame)
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_hog(frame)

    def detect_batch(self, frames) -> list[list[Detection]]:
        if not frames:
            return []
        if self.session is not None:
            return self._detect_batch_onnxruntime(frames)
        if self.model is not None:
            return self._detect_batch_yolo(frames)
        return [self._detect_hog(frame) for frame in frames]

    def _detect_onnxruntime(self, frame) -> list[Detection]:
        return self._detect_batch_onnxruntime([frame])[0]

    def _detect_batch_onnxruntime(self, frames) -> list[list[Detection]]:
        original_shapes = [frame.shape[:2] for frame in frames]
        blobs = []
        for frame in frames:
            resized = cv2.resize(frame, (self.input_width, self.input_height))
            image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blobs.append(np.transpose(image, (2, 0, 1)))

        feed = np.stack(blobs, axis=0)
        requested_count = len(frames)
        if self.batch_size > requested_count:
            pad = np.repeat(feed[-1:], self.batch_size - requested_count, axis=0)
            feed = np.concatenate([feed, pad], axis=0)

        output = self.session.run([self.output_name], {self.input_name: feed})[0]
        batch_predictions = output[:requested_count]

        all_detections: list[list[Detection]] = []
        for predictions, (original_h, original_w) in zip(batch_predictions, original_shapes):
            if predictions.ndim != 2:
                all_detections.append([])
                continue

            if predictions.shape[1] == 6:
                all_detections.append(
                    self._parse_xyxy_predictions(predictions, original_w, original_h)
                )
                continue

            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T
            all_detections.append(
                self._parse_yolov8_predictions(predictions, original_w, original_h)
            )
        return all_detections

    def _detect_yolo(self, frame) -> list[Detection]:
        return self._detect_batch_yolo([frame])[0]

    def _detect_batch_yolo(self, frames) -> list[list[Detection]]:
        results = self.model.predict(
            frames,
            classes=[0],
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=True,
            device="cpu",
        )
        all_detections: list[list[Detection]] = []
        for result in results:
            detections: list[Detection] = []
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    confidence = float(box.conf[0].cpu().item())
                    class_id = int(box.cls[0].cpu().item())
                    detections.append(Detection((x1, y1, x2, y2), confidence, class_id))
            all_detections.append(detections)
        return all_detections

    def _parse_xyxy_predictions(
        self, predictions, original_w: int, original_h: int
    ) -> list[Detection]:
        boxes = []
        scores = []
        class_ids = []
        scale_x = original_w / self.input_width
        scale_y = original_h / self.input_height

        for row in predictions:
            confidence = float(row[4])
            class_id = int(row[5])
            if class_id != 0 or confidence < self.conf_threshold:
                continue
            x1, y1, x2, y2 = row[:4]
            left = int(max(0, min(original_w - 1, x1 * scale_x)))
            top = int(max(0, min(original_h - 1, y1 * scale_y)))
            right = int(max(0, min(original_w - 1, x2 * scale_x)))
            bottom = int(max(0, min(original_h - 1, y2 * scale_y)))
            boxes.append([left, top, max(1, right - left), max(1, bottom - top)])
            scores.append(confidence)
            class_ids.append(class_id)

        return self._nms_to_detections(boxes, scores, class_ids)

    def _parse_yolov8_predictions(
        self, predictions, original_w: int, original_h: int
    ) -> list[Detection]:
        boxes = []
        scores = []
        class_ids = []
        scale_x = original_w / self.input_width
        scale_y = original_h / self.input_height

        for row in predictions:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id != 0 or confidence < self.conf_threshold:
                continue

            cx, cy, width, height = row[:4]
            left = int(max(0, min(original_w - 1, (cx - width / 2) * scale_x)))
            top = int(max(0, min(original_h - 1, (cy - height / 2) * scale_y)))
            box_width = int(max(1, min(original_w - left, width * scale_x)))
            box_height = int(max(1, min(original_h - top, height * scale_y)))
            boxes.append([left, top, box_width, box_height])
            scores.append(confidence)
            class_ids.append(class_id)

        return self._nms_to_detections(boxes, scores, class_ids)

    def _nms_to_detections(self, boxes, scores, class_ids) -> list[Detection]:
        if not boxes:
            return []

        indexes = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.nms_threshold
        )
        if len(indexes) == 0:
            return []

        detections: list[Detection] = []
        for index in np.array(indexes).flatten():
            left, top, width, height = boxes[int(index)]
            detections.append(
                Detection(
                    bbox=(left, top, left + width, top + height),
                    confidence=float(scores[int(index)]),
                    class_id=int(class_ids[int(index)]),
                )
            )
        return detections

    def _detect_hog(self, frame) -> list[Detection]:
        rects, weights = self.hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        detections: list[Detection] = []
        for (x, y, w, h), confidence in zip(rects, weights):
            if float(confidence) < self.conf_threshold:
                continue
            detections.append(
                Detection(
                    (int(x), int(y), int(x + w), int(y + h)), float(confidence), 0
                )
            )
        return detections


class ProcessThread(QThread):
    """Apply image preprocessing and person detection."""

    status_changed = pyqtSignal(str, str)
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
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.detector = PersonDetector(
            model_path=model_path,
            model_format=model_format,
            conf_threshold=conf_threshold,
            strict_model=model_path is not None or model_format is not None,
        )
        self.running = True
        self.mode = mode
        self.roi = roi
        self._mode_mutex = QMutex()
        self._fps_history: list[float] = []

    def run(self) -> None:
        self.status_changed.emit("Process", f"Running - {self.detector.backend}")
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

        self.status_changed.emit("Process", "Stopped")

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
        self.status_changed.emit("Process", f"Running - mode: {mode}")

    def get_mode(self) -> str:
        self._mode_mutex.lock()
        try:
            return self.mode
        finally:
            self._mode_mutex.unlock()

    def stop(self) -> None:
        self.running = False
        self.wait(1500)


def resolve_roi(frame_shape, roi: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    if all(0.0 <= value <= 1.0 for value in roi):
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

    left = max(0, min(w - 1, int(x1)))
    top = max(0, min(h - 1, int(y1)))
    right = max(left + 1, min(w, int(x2)))
    bottom = max(top + 1, min(h, int(y2)))
    return left, top, right, bottom


def detect_frames_in_roi(
    detector: PersonDetector,
    frames,
    roi: tuple[float, float, float, float] | None,
) -> list[list[Detection]]:
    if roi is None:
        return detector.detect_batch(frames)

    roi_frames = []
    roi_offsets = []
    for frame in frames:
        x1, y1, x2, y2 = resolve_roi(frame.shape, roi)
        roi_frame = frame[y1:y2, x1:x2]
        roi_frames.append(roi_frame)
        roi_offsets.append((x1, y1))

    roi_detections = detector.detect_batch(roi_frames)
    all_detections: list[list[Detection]] = []
    for detections, (x1, y1) in zip(roi_detections, roi_offsets):
        remapped: list[Detection] = []
        for detection in detections:
            bx1, by1, bx2, by2 = detection.bbox
            remapped.append(
                Detection(
                    bbox=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    label=detection.label,
                )
            )
        all_detections.append(remapped)
    return all_detections


class BatchProcessThread(QThread):
    """Run one shared detector over one frame from each camera as a batch."""

    status_changed = pyqtSignal(str, str)
    metrics_ready = pyqtSignal(object)

    def __init__(
        self,
        streams: list[tuple[str, Queue, Queue]],
        model_path: str | None = None,
        model_format: str | None = None,
        mode: str = "none",
        conf_threshold: float = 0.35,
        roi: tuple[float, float, float, float] | None = None,
    ):
        super().__init__()
        self.streams = streams
        self.detector = PersonDetector(
            model_path=model_path,
            model_format=model_format,
            conf_threshold=conf_threshold,
            strict_model=model_path is not None or model_format is not None,
        )
        self.running = True
        self.mode = mode
        self.roi = roi
        self._mode_mutex = QMutex()
        self._fps_history: list[float] = []

    def run(self) -> None:
        self.status_changed.emit("Process", f"Batch running - {self.detector.backend}")
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

        self.status_changed.emit("Process", "Stopped")

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
        self.status_changed.emit("Process", f"Batch running - mode: {mode}")

    def get_mode(self) -> str:
        self._mode_mutex.lock()
        try:
            return self.mode
        finally:
            self._mode_mutex.unlock()

    def stop(self) -> None:
        self.running = False
        self.wait(1500)
