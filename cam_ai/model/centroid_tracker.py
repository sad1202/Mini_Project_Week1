import math
from dataclasses import dataclass, field

from cam_ai.threads.pipeline_data import Detection, TrackObject


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
