from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import List, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class FramePacket:
    camera_id: str
    frame_id: int
    frame: np.ndarray
    timestamp: float
    capture_fps: float


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int = 0
    label: str = "person"

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class ProcessedPacket:
    camera_id: str
    frame_id: int
    frame: np.ndarray
    timestamp: float
    capture_fps: float
    process_fps: float
    detections: List[Detection] = field(default_factory=list)
    processing_mode: str = "none"


@dataclass
class TrackObject:
    track_id: int
    bbox: BBox
    centroid: Point
    confidence: float
    trajectory: List[Point] = field(default_factory=list)


@dataclass
class TrackingPacket:
    camera_id: str
    frame_id: int
    frame: np.ndarray
    timestamp: float
    capture_fps: float
    process_fps: float
    tracking_fps: float
    detections: List[Detection] = field(default_factory=list)
    tracks: List[TrackObject] = field(default_factory=list)
    processing_mode: str = "none"


@dataclass
class Event:
    event_type: str
    message: str
    level: str
    timestamp: float
    track_id: int | None = None


@dataclass
class DisplayPacket:
    camera_id: str
    frame_id: int
    frame: np.ndarray
    timestamp: float
    capture_fps: float
    process_fps: float
    tracking_fps: float
    display_fps: float = 0.0
    people_count: int = 0
    track_count: int = 0
    events: List[Event] = field(default_factory=list)
    processing_mode: str = "none"


def put_latest(queue: Queue, item) -> None:
    """Put an item into a bounded queue, dropping stale items first."""
    while queue.full():
        try:
            queue.get_nowait()
        except Empty:
            break

    try:
        queue.put_nowait(item)
    except Full:
        pass
