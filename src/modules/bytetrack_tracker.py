"""
Enhanced ByteTrack tracker with track lifecycle management,
status metadata, lost-track buffering, and zone counting.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from bytetracker import BYTETracker


class TrackInfo:
    """Metadata for a single tracked person."""

    __slots__ = (
        "track_id", "box", "first_seen", "last_seen",
        "age_frames", "lost_frames", "status", "history",
    )

    def __init__(self, track_id: int, box: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.box = box
        self.first_seen: float = time.time()
        self.last_seen: float = self.first_seen
        self.age_frames: int = 1
        self.lost_frames: int = 0
        self.status: str = "active"   # active | lost | removed
        self.history: List[Tuple[int, int, int, int]] = [box]

    def update(self, box: Tuple[int, int, int, int]) -> None:
        self.box = box
        self.last_seen = time.time()
        self.age_frames += 1
        self.lost_frames = 0
        self.status = "active"
        self.history.append(box)
        # Keep last 120 positions only
        if len(self.history) > 120:
            self.history = self.history[-120:]

    def mark_lost(self) -> None:
        self.lost_frames += 1
        self.status = "lost"

    @property
    def duration_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.track_id,
            "box": self.box,
            "status": self.status,
            "age_frames": self.age_frames,
            "lost_frames": self.lost_frames,
            "duration_s": round(self.duration_seconds, 2),
            "center": self.center,
        }


class ByteTrackTracker:
    """
    Enhanced ByteTrack wrapper with:
    - Per-track lifecycle metadata (age, duration, status)
    - Lost-track buffering (keep tracks visible for N frames after loss)
    - Direction/velocity estimation
    - Track statistics
    """

    def __init__(
        self,
        frame_rate: int = 12,
        track_buffer: int = 600,
        match_thresh: float = 0.8,
        lost_buffer_frames: int = 15,
        max_lost_frames: int = 60,
    ):
        """
        Args:
            frame_rate:         Expected FPS of the source.
            track_buffer:       Internal ByteTrack buffer length.
            match_thresh:       IoU match threshold.
            lost_buffer_frames: How many frames to keep a lost track visible.
            max_lost_frames:    After this many lost frames, remove track entirely.
        """
        self.tracker = BYTETracker(
            frame_rate=frame_rate,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self._lost_buffer = lost_buffer_frames
        self._max_lost = max_lost_frames

        # Persistent track registry  { track_id: TrackInfo }
        self._tracks: Dict[int, TrackInfo] = {}
        self._removed_ids: set = set()
        self._total_unique = 0
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        frame_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        detections:  list of (x1, y1, x2, y2, conf)
        frame_shape: (H, W) of the current frame

        Returns list of track dicts (see TrackInfo.to_dict).
        """
        self._frame_idx += 1

        # --- run ByteTrack ---
        active_ids: set = set()

        if len(detections) > 0:
            dets_np = np.array(
                [list(det) + [0.0] for det in detections], dtype=np.float32
            )
            dets = torch.from_numpy(dets_np)
            raw_tracks = self.tracker.update(dets, frame_shape)

            for t in raw_tracks:
                x1, y1, x2, y2 = map(int, t[0:4])
                track_id = int(t[4])
                box = (x1, y1, x2, y2)
                active_ids.add(track_id)

                if track_id in self._tracks:
                    self._tracks[track_id].update(box)
                else:
                    self._tracks[track_id] = TrackInfo(track_id, box)
                    self._total_unique += 1
        else:
            raw_tracks = []

        # --- manage lost tracks ---
        for tid, info in list(self._tracks.items()):
            if tid not in active_ids:
                info.mark_lost()
                if info.lost_frames > self._max_lost:
                    info.status = "removed"
                    self._removed_ids.add(tid)

        # Clean up fully removed tracks
        for tid in list(self._removed_ids):
            self._tracks.pop(tid, None)
        self._removed_ids.clear()

        # --- build output (active + recently-lost) ---
        output: List[Dict[str, Any]] = []
        for info in self._tracks.values():
            if info.status == "active":
                output.append(info.to_dict())
            elif info.status == "lost" and info.lost_frames <= self._lost_buffer:
                output.append(info.to_dict())

        return output

    def get_track(self, track_id: int) -> Optional[TrackInfo]:
        """Get metadata for a specific track."""
        return self._tracks.get(track_id)

    def get_stats(self) -> Dict[str, Any]:
        """Return tracker-wide statistics."""
        active = sum(1 for t in self._tracks.values() if t.status == "active")
        lost = sum(1 for t in self._tracks.values() if t.status == "lost")
        return {
            "frame_index": self._frame_idx,
            "total_unique_tracks": self._total_unique,
            "currently_active": active,
            "currently_lost": lost,
            "tracked_ids": sorted(self._tracks.keys()),
        }

    def get_velocity(self, track_id: int, n_frames: int = 5) -> Optional[Tuple[float, float]]:
        """
        Estimate velocity (dx, dy) per frame for a track.
        Returns None if track not found or too few history points.
        """
        info = self._tracks.get(track_id)
        if info is None or len(info.history) < 2:
            return None

        recent = info.history[-min(n_frames, len(info.history)):]
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in recent]
        dx = (centers[-1][0] - centers[0][0]) / max(len(centers) - 1, 1)
        dy = (centers[-1][1] - centers[0][1]) / max(len(centers) - 1, 1)
        return (round(dx, 2), round(dy, 2))
