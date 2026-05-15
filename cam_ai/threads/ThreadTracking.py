import math
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from cam_ai.threads.pipeline_data import (
    Detection,
    ProcessedPacket,
    TrackObject,
    TrackingPacket,
    put_latest,
)


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[int, int]
    confidence: float
    missed: int = 0
    trajectory: list[tuple[int, int]] = field(default_factory=list)


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 12, trajectory_length: int = 40):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.trajectory_length = trajectory_length
        self.next_id = 1
        self.tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[TrackObject]:
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indexes = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        distances: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_index, detection in enumerate(detections):
                distances.append((self._distance(track.centroid, detection.centroid), track_id, det_index))
        distances.sort(key=lambda item: item[0])

        for distance, track_id, det_index in distances:
            if distance > self.max_distance:
                continue
            if track_id not in unmatched_track_ids or det_index not in unmatched_detection_indexes:
                continue
            matches.append((track_id, det_index))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(det_index)

        for track_id, det_index in matches:
            detection = detections[det_index]
            track = self.tracks[track_id]
            track.bbox = detection.bbox
            track.centroid = detection.centroid
            track.confidence = detection.confidence
            track.missed = 0
            track.trajectory.append(detection.centroid)
            track.trajectory = track.trajectory[-self.trajectory_length :]

        for det_index in unmatched_detection_indexes:
            detection = detections[det_index]
            self.tracks[self.next_id] = _TrackState(
                track_id=self.next_id,
                bbox=detection.bbox,
                centroid=detection.centroid,
                confidence=detection.confidence,
                trajectory=[detection.centroid],
            )
            self.next_id += 1

        expired_ids = []
        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            self.tracks.pop(track_id, None)

        return [
            TrackObject(
                track_id=track.track_id,
                bbox=track.bbox,
                centroid=track.centroid,
                confidence=track.confidence,
                trajectory=list(track.trajectory),
            )
            for track in self.tracks.values()
            if track.missed == 0
        ]

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


class TrackingThread(QThread):
    """Assign IDs to detected people and draw tracks/trajectories."""

    status_changed = pyqtSignal(str, str)
    metrics_ready = pyqtSignal(object)

    def __init__(self, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.trackers: dict[str, CentroidTracker] = {}
        self._fps_history: list[float] = []

    def run(self) -> None:
        self.status_changed.emit("Tracking", "Running")
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

        self.status_changed.emit("Tracking", "Stopped")

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
    """Track all cameras in one thread while keeping tracker state per camera."""

    status_changed = pyqtSignal(str, str)
    metrics_ready = pyqtSignal(object)

    def __init__(self, streams: list[tuple[str, Queue, Queue]]):
        super().__init__()
        self.streams = streams
        self.running = True
        self.trackers: dict[str, CentroidTracker] = {}
        self._fps_history: list[float] = []

    def run(self) -> None:
        self.status_changed.emit("Tracking", "Batch running")
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

        self.status_changed.emit("Tracking", "Stopped")

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
