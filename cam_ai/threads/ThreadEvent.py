import time
from queue import Empty, Queue

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from cam_ai.threads.pipeline_data import DisplayPacket, Event, TrackingPacket, TrackObject, put_latest


class EventThread(QThread):
    """Detect ROI loitering/crowding events and annotate the display frame."""

    status_changed = pyqtSignal(str, str)
    event_ready = pyqtSignal(object)
    metrics_ready = pyqtSignal(object)

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        roi: tuple[float, float, float, float] = (0.25, 0.2, 0.75, 0.9),
        loiter_seconds: float = 5.0,
        crowd_threshold: int = 3,
        event_cooldown: float = 2.0,
    ):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.roi = roi
        self.loiter_seconds = loiter_seconds
        self.crowd_threshold = crowd_threshold
        self.event_cooldown = event_cooldown
        self.running = True
        self._inside_since: dict[tuple[str, int], float] = {}
        self._last_event_at: dict[tuple[str, str, int | None], float] = {}

    def run(self) -> None:
        self.status_changed.emit("Event", "Running")
        while self.running:
            try:
                packet: TrackingPacket = self.input_queue.get(timeout=0.2)
            except Empty:
                continue

            roi_px = self._resolve_roi(packet.frame.shape)
            events = self._detect_events(packet, roi_px)
            annotated = self._draw_event_layer(packet.frame, roi_px, events)

            display_packet = DisplayPacket(
                camera_id=packet.camera_id,
                frame_id=packet.frame_id,
                frame=annotated,
                timestamp=packet.timestamp,
                capture_fps=packet.capture_fps,
                process_fps=packet.process_fps,
                tracking_fps=packet.tracking_fps,
                people_count=len(packet.detections),
                track_count=len(packet.tracks),
                events=events,
                processing_mode=packet.processing_mode,
            )
            put_latest(self.output_queue, display_packet)
            self.metrics_ready.emit(
                {
                    "people_count": display_packet.people_count,
                    "track_count": display_packet.track_count,
                    "event_count": len(events),
                }
            )

            for event in events:
                self.event_ready.emit(event)

        self.status_changed.emit("Event", "Stopped")

    def _detect_events(
        self,
        packet: TrackingPacket,
        roi_px: tuple[int, int, int, int],
    ) -> list[Event]:
        now = time.time()
        active_track_ids = {track.track_id for track in packet.tracks}
        for key in list(self._inside_since):
            camera_id, track_id = key
            if camera_id == packet.camera_id and track_id not in active_track_ids:
                self._inside_since.pop(key, None)

        tracks_in_roi = [track for track in packet.tracks if self._point_in_roi(track.centroid, roi_px)]
        track_ids_in_roi = {track.track_id for track in tracks_in_roi}
        events: list[Event] = []

        for track in packet.tracks:
            key = (packet.camera_id, track.track_id)
            if track.track_id in track_ids_in_roi:
                self._inside_since.setdefault(key, now)
                dwell = now - self._inside_since[key]
                if dwell >= self.loiter_seconds:
                    event = Event(
                        event_type="loitering",
                        message=f"Person ID {track.track_id} stayed in ROI for {dwell:.1f}s",
                        level="warning",
                        timestamp=now,
                        track_id=track.track_id,
                    )
                    if self._allow_event(packet.camera_id, event):
                        events.append(event)
            else:
                self._inside_since.pop(key, None)

        if len(tracks_in_roi) >= self.crowd_threshold:
            event = Event(
                event_type="crowding",
                message=f"{len(tracks_in_roi)} people in ROI",
                level="critical",
                timestamp=now,
            )
            if self._allow_event(packet.camera_id, event):
                events.append(event)

        return events

    def _allow_event(self, camera_id: str, event: Event) -> bool:
        key = (camera_id, event.event_type, event.track_id)
        now = time.time()
        last_at = self._last_event_at.get(key, 0.0)
        if now - last_at < self.event_cooldown:
            return False
        self._last_event_at[key] = now
        return True

    def _resolve_roi(self, frame_shape) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.roi
        if all(0.0 <= value <= 1.0 for value in self.roi):
            return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _point_in_roi(point: tuple[int, int], roi_px: tuple[int, int, int, int]) -> bool:
        x, y = point
        x1, y1, x2, y2 = roi_px
        return x1 <= x <= x2 and y1 <= y <= y2

    def _draw_event_layer(self, frame, roi_px: tuple[int, int, int, int], events: list[Event]):
        annotated = frame.copy()
        x1, y1, x2, y2 = roi_px
        color = (0, 165, 255) if not events else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            "ROI",
            (x1 + 6, max(22, y1 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        for index, event in enumerate(events[:3]):
            y = 28 + index * 28
            cv2.putText(
                annotated,
                event.message,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def stop(self) -> None:
        self.running = False
        self.wait(1500)
