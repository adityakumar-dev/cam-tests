"""
Unified frame-processing pipeline.

Ties together:
  - YOLOv8 person detection (with motion, adaptive conf, NMS, smoothing)
  - ByteTrack person tracking
  - YoloFace per-person face detection
  - HUD / overlay drawing
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from src.modules.yolo_persons import YOLOv8PersonDetector
from src.modules.bytetrack_tracker import ByteTrackTracker
from src.modules.yoloface_detector import YoloFaceDetector

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data classes for structured results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass
class PersonResult:
    track_id: int
    box: Tuple[int, int, int, int]
    status: str                           # "active" | "lost"
    age_frames: int
    duration_s: float
    faces: List[FaceResult] = field(default_factory=list)


@dataclass
class FrameResult:
    """Complete result for one processed frame."""
    frame: np.ndarray                     # annotated frame
    persons: List[PersonResult]
    motion_score: float
    motion_regions: List[Tuple[int, int, int, int]]
    frame_skipped: bool                   # True if YOLO was skipped
    frame_number: int
    process_time_ms: float
    detector_stats: Dict[str, Any]
    tracker_stats: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────

_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_BLUE = (255, 0, 0)
_RED = (0, 0, 255)
_CYAN = (255, 255, 0)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_person(
    frame: np.ndarray,
    person: PersonResult,
    show_faces: bool = True,
) -> None:
    x1, y1, x2, y2 = person.box
    color = _GREEN if person.status == "active" else _YELLOW

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"P{person.track_id}"
    if person.duration_s > 0:
        label += f"  {person.duration_s:.1f}s"

    cv2.putText(frame, label, (x1, max(0, y1 - 8)), _FONT, 0.55, color, 2)

    if show_faces:
        for f in person.faces:
            cv2.rectangle(frame, (f.x1, f.y1), (f.x2, f.y2), _BLUE, 2)
            cv2.putText(
                frame,
                f"face {f.confidence:.2f}",
                (f.x1, max(0, f.y1 - 5)),
                _FONT, 0.4, _BLUE, 1,
            )


def _draw_motion_regions(
    frame: np.ndarray,
    regions: List[Tuple[int, int, int, int]],
) -> None:
    for (x1, y1, x2, y2) in regions:
        cv2.rectangle(frame, (x1, y1), (x2, y2), _CYAN, 1)


def _draw_hud(
    frame: np.ndarray,
    result: "FrameResult",
    show_motion: bool = True,
    detect_res: str = "",
) -> None:
    """Draw a translucent heads-up display bar at the top."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    fps_val = 1000 / max(result.process_time_ms, 0.1)
    line1 = (
        f"Persons: {len(result.persons)}  |  "
        f"FPS: {fps_val:.0f}  |  "
        f"Frame: {result.frame_number}"
    )
    if detect_res:
        line1 += f"  |  Det: {detect_res}"
    cv2.putText(frame, line1, (10, 20), _FONT, 0.48, _WHITE, 1)

    if show_motion:
        motion_pct = result.motion_score * 100
        skip = "SKIP" if result.frame_skipped else "RUN"
        line2 = (
            f"Motion: {motion_pct:.1f}%  |  "
            f"YOLO: {skip}  |  "
            f"Tracks: {result.tracker_stats.get('total_unique_tracks', 0)}"
        )
        cv2.putText(frame, line2, (10, 42), _FONT, 0.44, _CYAN, 1)


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

class FramePipeline:
    """
    High-level pipeline: feed it frames, get annotated results back.

    Automatically downscales high-resolution input for detection speed,
    then maps all coordinates back to the original frame for drawing.

    Usage::

        pipe = FramePipeline()
        result = pipe.process(frame)
        cv2.imshow("out", result.frame)
    """

    # If any frame dimension exceeds this, it gets downscaled for detection
    MAX_DETECT_DIM = 1280

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        person_conf: float = 0.4,
        use_motion: bool = True,
        use_smoothing: bool = True,
        use_face_detection: bool = True,
        face_conf: float = 0.25,
        tracker_fps: int = 12,
        show_hud: bool = True,
        show_motion_boxes: bool = False,
        device: Optional[str] = None,
        max_detect_dim: int = MAX_DETECT_DIM,
        face_every_n_frames: int = 3,
        max_faces_per_frame: int = 5,
    ):
        """
        Args:
            max_detect_dim:      Downscale frames to this max dimension for
                                 YOLO + motion (then scale boxes back).
            face_every_n_frames: Only run face detection every N frames.
            max_faces_per_frame: Cap face-detection calls per frame.
        """
        self.detector = YOLOv8PersonDetector(
            model_path=model_path,
            conf=person_conf,
            use_motion=use_motion,
            use_smoothing=use_smoothing,
            device=device,
        )
        self.tracker = ByteTrackTracker(frame_rate=tracker_fps)
        self.face_detector = YoloFaceDetector(conf=face_conf) if use_face_detection else None

        self._show_hud = show_hud
        self._show_motion_boxes = show_motion_boxes
        self._max_detect_dim = max_detect_dim
        self._face_every_n = face_every_n_frames
        self._max_faces = max_faces_per_frame
        self._process_frame_idx = 0

        # Cache face results keyed by track_id
        self._face_cache: Dict[int, List[FaceResult]] = {}

    def _compute_scale(self, h: int, w: int) -> float:
        """Return the downscale factor (<=1.0) so max(h,w) fits MAX_DETECT_DIM."""
        if max(h, w) <= self._max_detect_dim:
            return 1.0
        return self._max_detect_dim / max(h, w)

    def process(self, frame: np.ndarray) -> FrameResult:
        """Run the full pipeline on one BGR frame."""
        t0 = time.perf_counter()
        self._process_frame_idx += 1

        orig_h, orig_w = frame.shape[:2]
        scale = self._compute_scale(orig_h, orig_w)

        # ---- downscale for detection ----------------------------------------
        if scale < 1.0:
            det_w = int(orig_w * scale)
            det_h = int(orig_h * scale)
            detect_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            detect_frame = frame
            det_h, det_w = orig_h, orig_w

        # 1) Detect persons on the smaller frame
        det_out = self.detector.detect_with_motion(detect_frame)
        persons_raw_small = det_out["persons"]
        motion_regions_small = det_out["motion_regions"]
        motion_score = det_out["motion_score"]
        frame_skipped = det_out["frame_skipped"]
        frame_number = det_out["frame_number"]

        # Scale detections back to original resolution
        inv_scale = 1.0 / scale if scale < 1.0 else 1.0

        persons_raw = []
        for (x1, y1, x2, y2, conf) in persons_raw_small:
            persons_raw.append((
                int(x1 * inv_scale), int(y1 * inv_scale),
                int(x2 * inv_scale), int(y2 * inv_scale),
                conf,
            ))

        motion_regions = []
        for (x1, y1, x2, y2) in motion_regions_small:
            motion_regions.append((
                int(x1 * inv_scale), int(y1 * inv_scale),
                int(x2 * inv_scale), int(y2 * inv_scale),
            ))

        # 2) Track (at original resolution coords)
        tracks = self.tracker.update(persons_raw, (orig_h, orig_w))

        # 3) Face detection — throttled
        do_faces = (
            self.face_detector is not None
            and self._process_frame_idx % self._face_every_n == 0
        )

        person_results: List[PersonResult] = []
        face_calls = 0
        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            pr = PersonResult(
                track_id=t["id"],
                box=t["box"],
                status=t["status"],
                age_frames=t["age_frames"],
                duration_s=t["duration_s"],
            )

            tid = t["id"]
            if do_faces and t["status"] == "active" and face_calls < self._max_faces:
                crop = frame[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)]
                if crop.size > 0:
                    faces = self.face_detector.detect(crop)
                    face_results = []
                    for (fx1, fy1, fx2, fy2, fc) in faces:
                        face_results.append(FaceResult(
                            x1=x1 + fx1, y1=y1 + fy1,
                            x2=x1 + fx2, y2=y1 + fy2,
                            confidence=fc,
                        ))
                    pr.faces = face_results
                    self._face_cache[tid] = face_results
                    face_calls += 1
            elif tid in self._face_cache:
                # Reuse cached face results from last detection
                pr.faces = self._face_cache[tid]

            person_results.append(pr)

        process_ms = (time.perf_counter() - t0) * 1000

        # Clean stale face cache entries
        active_tids = {pr.track_id for pr in person_results}
        for stale_tid in list(self._face_cache.keys()):
            if stale_tid not in active_tids:
                del self._face_cache[stale_tid]

        # 4) Draw annotations
        annotated = frame.copy()

        if self._show_motion_boxes:
            _draw_motion_regions(annotated, motion_regions)

        for pr in person_results:
            _draw_person(annotated, pr, show_faces=self.face_detector is not None)

        result = FrameResult(
            frame=annotated,
            persons=person_results,
            motion_score=motion_score,
            motion_regions=motion_regions,
            frame_skipped=frame_skipped,
            frame_number=frame_number,
            process_time_ms=round(process_ms, 2),
            detector_stats=self.detector.get_stats(),
            tracker_stats=self.tracker.get_stats(),
        )

        if self._show_hud:
            detect_res = f"{det_w}x{det_h}" if scale < 1.0 else f"{orig_w}x{orig_h}"
            _draw_hud(result.frame, result, detect_res=detect_res)

        return result