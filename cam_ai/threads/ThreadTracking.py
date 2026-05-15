import time
from queue import Empty, Queue

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from cam_ai.model.centroid_tracker import CentroidTracker
from cam_ai.threads.pipeline_data import (
    ProcessedPacket,
    TrackObject,
    TrackingPacket,
    put_latest,
)


class TrackingThread(QThread):
    

    metrics_ready = pyqtSignal(object)

    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.trackers: dict[str, CentroidTracker] = {}
        self._fps_history: list[float] = []

    def run(self) -> None:
        while self.running:
            try:
                packet: ProcessedPacket = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            start = time.perf_counter()
            tracker = self.trackers.setdefault(packet.camera_id, CentroidTracker())
            tracks = tracker.update(packet.detections)
            annotated = self._draw_tracks(packet.frame, tracks)
            tracking_fps = self._smooth_fps(1.0 / max(time.perf_counter() - start, 1e-6))

            tracking_packet = TrackingPacket(
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                frame=annotated,
                timestamp=packet.timestamp,
                capture_fps=packet.capture_fps,
                process_fps=packet.process_fps,
                tracking_fps=tracking_fps,
                detections=packet.detections,
                tracks=tracks,
                processing_mode=packet.processing_mode,
            )
            put_latest(self.output_queue, tracking_packet)
            self.metrics_ready.emit({"tracking_fps": tracking_fps, "track_count": len(tracks)})


    def _draw_tracks(self, frame, tracks: list[TrackObject]):
        annotated = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cx, cy = track.centroid
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(
                annotated,
                f"ID {track.track_id}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for index in range(1, len(track.trajectory)):
                cv2.line(annotated, track.trajectory[index - 1], track.trajectory[index], (255, 160, 0), 2)
        return annotated

    def _smooth_fps(self, fps: float) -> float:
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)

    def stop(self) -> None:
        self.running = False
        self.wait(1500)


class BatchTrackingThread(QThread):
    

    metrics_ready = pyqtSignal(object)

    def __init__(self, streams: list[tuple[str, Queue, Queue]]):
        super().__init__()
        self.streams = streams
        self.running = True
        self.trackers: dict[str, CentroidTracker] = {}
        self._fps_history: list[float] = []

    def run(self) -> None:
        while self.running:
            packets = []
            output_queues = []
            for _camera_id, input_queue, output_queue in self.streams:
                try:
                    packet: ProcessedPacket = input_queue.get(timeout=0.2)
                except Empty:
                    packets = []
                    break
                packets.append(packet)
                output_queues.append(output_queue)

            if not packets:
                continue

            start = time.perf_counter()
            tracked_results = []
            for packet in packets:
                tracker = self.trackers.setdefault(packet.camera_id, CentroidTracker())
                tracks = tracker.update(packet.detections)
                annotated = self._draw_tracks(packet.frame, tracks)
                tracked_results.append((packet, tracks, annotated))

            tracking_fps = self._smooth_fps(1.0 / max(time.perf_counter() - start, 1e-6))

            for (packet, tracks, annotated), output_queue in zip(tracked_results, output_queues):
                tracking_packet = TrackingPacket(
                    camera_id=packet.camera_id,
                    frame_id=packet.frame_id,
                    frame=annotated,
                    timestamp=packet.timestamp,
                    capture_fps=packet.capture_fps,
                    process_fps=packet.process_fps,
                    tracking_fps=tracking_fps,
                    detections=packet.detections,
                    tracks=tracks,
                    processing_mode=packet.processing_mode,
                )
                put_latest(output_queue, tracking_packet)
                self.metrics_ready.emit(
                    {
                        "camera_id": packet.camera_id,
                        "tracking_fps": tracking_fps,
                        "track_count": len(tracks),
                    }
                )


    def _draw_tracks(self, frame, tracks: list[TrackObject]):
        annotated = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cx, cy = track.centroid
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 255, 255), -1)
            cv2.putText(
                annotated,
                f"ID {track.track_id}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for index in range(1, len(track.trajectory)):
                cv2.line(annotated, track.trajectory[index - 1], track.trajectory[index], (255, 160, 0), 2)
        return annotated

    def _smooth_fps(self, fps: float) -> float:
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        return sum(self._fps_history) / len(self._fps_history)

    def stop(self) -> None:
        self.running = False
        self.wait(1500)
