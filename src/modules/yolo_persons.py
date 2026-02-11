"""
Enhanced YOLOv8 person detector with motion-guided detection,
adaptive confidence, size filtering, NMS, and temporal smoothing.
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO


class MotionDetector:
    """Background-subtraction based motion detector to guide person detection."""

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 25.0,
        detect_shadows: bool = True,
        min_contour_area: int = 1500,
        dilation_kernel_size: int = 7,
        dilation_iterations: int = 3,
    ):
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._min_area = min_contour_area
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size)
        )
        self._dilation_iters = dilation_iterations
        self._motion_score: float = 0.0

    @property
    def motion_score(self) -> float:
        """Fraction of frame pixels with motion (0.0–1.0)."""
        return self._motion_score

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Return bounding boxes of motion regions.
        Each box: (x1, y1, x2, y2).
        """
        fg_mask = self._bg_sub.apply(frame)

        # Remove shadow pixels (value 127 in MOG2)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        thresh = cv2.dilate(thresh, self._kernel, iterations=self._dilation_iters)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self._kernel)

        self._motion_score = float(np.count_nonzero(thresh)) / max(thresh.size, 1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self._min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, x + w, y + h))

        return boxes

    def get_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return the raw foreground mask for visualization."""
        fg_mask = self._bg_sub.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, self._kernel, iterations=self._dilation_iters)
        return thresh


class DetectionSmoother:
    """Temporal smoothing of detections across frames to reduce flicker."""

    def __init__(self, buffer_size: int = 5, iou_threshold: float = 0.3):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._iou_thresh = iou_threshold

    @staticmethod
    def _iou(box_a: Tuple, box_b: Tuple) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    def smooth(
        self, detections: List[Tuple[int, int, int, int, float]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Add current detections to the buffer and return temporally
        smoothed detections (average of matched boxes across recent frames).
        """
        self._buffer.append(detections)

        if len(self._buffer) < 2:
            return detections

        # Use latest frame as reference
        current = list(detections)
        if not current:
            return current

        smoothed = []
        for det in current:
            matched_boxes = [det]
            for past in list(self._buffer)[:-1]:
                for p in past:
                    if self._iou(det[:4], p[:4]) > self._iou_thresh:
                        matched_boxes.append(p)
                        break

            # Average matched boxes for stability
            avg_x1 = int(np.mean([b[0] for b in matched_boxes]))
            avg_y1 = int(np.mean([b[1] for b in matched_boxes]))
            avg_x2 = int(np.mean([b[2] for b in matched_boxes]))
            avg_y2 = int(np.mean([b[3] for b in matched_boxes]))
            max_conf = max(b[4] for b in matched_boxes)
            smoothed.append((avg_x1, avg_y1, avg_x2, avg_y2, max_conf))

        return smoothed


class YOLOv8PersonDetector:
    """
    Robust YOLOv8 person detector with:
    - Motion-guided detection (skip static frames for speed)
    - Adaptive confidence based on scene motion
    - Size and aspect-ratio filtering
    - Non-maximum suppression
    - Temporal detection smoothing
    - Optional ROI (region-of-interest) masking
    """

    # Detection size limits (relative to frame dimensions)
    DEFAULT_MIN_HEIGHT_RATIO = 0.025   # person must be ≥ 2.5% of frame height
    DEFAULT_MAX_HEIGHT_RATIO = 0.98   # person must be ≤ 98% of frame height
    DEFAULT_MIN_ASPECT = 0.12         # min width/height
    DEFAULT_MAX_ASPECT = 1.3          # max width/height

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        iou_nms: float = 0.45,
        img_size: int = 960,
        use_motion: bool = True,
        motion_skip_threshold: float = 0.002,
        adaptive_conf: bool = True,
        adaptive_conf_range: Tuple[float, float] = (0.25, 0.6),
        use_smoothing: bool = True,
        smoothing_buffer: int = 4,
        min_height_ratio: float = DEFAULT_MIN_HEIGHT_RATIO,
        max_height_ratio: float = DEFAULT_MAX_HEIGHT_RATIO,
        min_aspect: float = DEFAULT_MIN_ASPECT,
        max_aspect: float = DEFAULT_MAX_ASPECT,
        roi: Optional[Tuple[int, int, int, int]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path:  Path to YOLOv8 weights (.pt).
            conf:        Base confidence threshold.
            iou_nms:     IoU threshold for non-maximum suppression.
            img_size:    Inference image size (px).
            use_motion:  Enable motion-guided detection.
            motion_skip_threshold: If scene motion < this, skip YOLO inference.
            adaptive_conf: Dynamically adjust conf based on motion level.
            adaptive_conf_range: (low_conf, high_conf) mapped to motion.
            use_smoothing: Enable temporal detection smoothing.
            smoothing_buffer: Number of frames for smoothing window.
            min_height_ratio: Min person height as fraction of frame.
            max_height_ratio: Max person height as fraction of frame.
            min_aspect: Min bbox width/height ratio.
            max_aspect: Max bbox width/height ratio.
            roi: Optional region-of-interest (x1, y1, x2, y2) in pixels.
            device: Force device ('cpu', 'cuda', 'cuda:0', etc.).
        """
        self.model = YOLO(model_path)
        self.base_conf = conf
        self.iou_nms = iou_nms
        self.img_size = img_size
        self.device = device

        # Motion detection
        self.use_motion = use_motion
        self.motion_skip_threshold = motion_skip_threshold
        self.motion_detector = MotionDetector() if use_motion else None

        # Adaptive confidence
        self.adaptive_conf = adaptive_conf
        self._conf_low, self._conf_high = adaptive_conf_range

        # Smoothing
        self.smoother = DetectionSmoother(buffer_size=smoothing_buffer) if use_smoothing else None

        # Size/aspect filtering
        self.min_height_ratio = min_height_ratio
        self.max_height_ratio = max_height_ratio
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        # ROI
        self.roi = roi

        # Cache last detections for when frames are skipped
        self._last_detections: List[Tuple[int, int, int, int, float]] = []
        self._last_motion_regions: List[Tuple[int, int, int, int]] = []
        self._last_motion_score: float = 0.0
        self._frame_count = 0
        self._skip_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in *frame*.

        Returns:
            List of (x1, y1, x2, y2, confidence) for each person.
        """
        if frame is None or frame.size == 0:
            return []

        self._frame_count += 1
        fh, fw = frame.shape[:2]

        # ---- motion gate ---------------------------------------------------
        motion_regions: List[Tuple[int, int, int, int]] = []
        motion_level = 1.0  # assume full motion if disabled

        if self.motion_detector is not None:
            # Run motion on a downscaled frame for speed (max 480p)
            motion_scale = 1.0
            motion_frame = frame
            if max(fh, fw) > 640:
                motion_scale = 640.0 / max(fh, fw)
                small_w, small_h = int(fw * motion_scale), int(fh * motion_scale)
                motion_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

            small_regions = self.motion_detector.detect(motion_frame)
            motion_level = self.motion_detector.motion_score

            # Scale motion regions back to original resolution
            if motion_scale != 1.0:
                inv = 1.0 / motion_scale
                motion_regions = [
                    (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
                    for (x1, y1, x2, y2) in small_regions
                ]
            else:
                motion_regions = small_regions

            # Cache motion results so detect_with_motion() doesn't re-run
            self._last_motion_regions = motion_regions
            self._last_motion_score = motion_level

            if motion_level < self.motion_skip_threshold and self._frame_count > 30:
                # Scene is static → reuse last detections
                self._skip_count += 1
                return list(self._last_detections)

        # ---- adaptive confidence -------------------------------------------
        conf = self.base_conf
        if self.adaptive_conf and self.motion_detector is not None:
            # More motion → lower threshold (catch partially visible people)
            conf = self._conf_high - (self._conf_high - self._conf_low) * min(motion_level * 10, 1.0)
            conf = max(self._conf_low, min(conf, self._conf_high))

        # ---- ROI crop ------------------------------------------------------
        detect_frame = frame
        roi_offset = (0, 0)
        if self.roi is not None:
            rx1, ry1, rx2, ry2 = self.roi
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(fw, rx2), min(fh, ry2)
            detect_frame = frame[ry1:ry2, rx1:rx2]
            roi_offset = (rx1, ry1)
            if detect_frame.size == 0:
                return []

        # ---- YOLO inference -------------------------------------------------
        predict_kwargs: Dict[str, Any] = dict(
            conf=conf,
            iou=self.iou_nms,
            imgsz=self.img_size,
            verbose=False,
            classes=[0],  # person class only
        )
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(detect_frame, **predict_kwargs)[0]

        raw_boxes: List[Tuple[int, int, int, int, float]] = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = float(box.conf[0])

            # Apply ROI offset
            x1 += roi_offset[0]
            y1 += roi_offset[1]
            x2 += roi_offset[0]
            y2 += roi_offset[1]

            raw_boxes.append((x1, y1, x2, y2, c))

        # ---- size / aspect filter -------------------------------------------
        filtered = self._filter_by_size(raw_boxes, fh, fw)

        # ---- NMS (secondary) -----------------------------------------------
        filtered = self._nms(filtered, self.iou_nms)

        # ---- temporal smoothing ---------------------------------------------
        if self.smoother is not None:
            filtered = self.smoother.smooth(filtered)

        self._last_detections = filtered
        return filtered

    def detect_with_motion(
        self, frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extended detection that also returns motion metadata.

        Returns dict with keys:
            persons:        list of (x1,y1,x2,y2,conf)
            motion_regions: list of (x1,y1,x2,y2) from motion detector
            motion_score:   float 0..1
            frame_skipped:  bool — whether YOLO was skipped this frame
            frame_number:   int
        """
        old_count = self._skip_count
        persons = self.detect(frame)
        was_skipped = self._skip_count > old_count

        # Reuse cached motion data from detect() — do NOT re-run motion
        motion_regions = self._last_motion_regions
        motion_score = self._last_motion_score

        return {
            "persons": persons,
            "motion_regions": motion_regions,
            "motion_score": motion_score,
            "frame_skipped": was_skipped,
            "frame_number": self._frame_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        total = max(self._frame_count, 1)
        return {
            "total_frames": self._frame_count,
            "skipped_frames": self._skip_count,
            "skip_ratio": round(self._skip_count / total, 3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_by_size(
        self,
        boxes: List[Tuple[int, int, int, int, float]],
        frame_h: int,
        frame_w: int,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Remove boxes that are too small, too large, or wrong aspect ratio."""
        kept = []
        for (x1, y1, x2, y2, c) in boxes:
            bw = x2 - x1
            bh = y2 - y1
            if bh <= 0 or bw <= 0:
                continue

            height_ratio = bh / frame_h
            aspect = bw / bh

            if height_ratio < self.min_height_ratio:
                continue
            if height_ratio > self.max_height_ratio:
                continue
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            # Clamp to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_w, x2)
            y2 = min(frame_h, y2)

            kept.append((x1, y1, x2, y2, c))
        return kept

    @staticmethod
    def _nms(
        boxes: List[Tuple[int, int, int, int, float]], iou_thresh: float
    ) -> List[Tuple[int, int, int, int, float]]:
        """Simple greedy non-maximum suppression."""
        if not boxes:
            return []

        sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []

        while sorted_boxes:
            best = sorted_boxes.pop(0)
            keep.append(best)
            remaining = []
            for other in sorted_boxes:
                if DetectionSmoother._iou(best[:4], other[:4]) < iou_thresh:
                    remaining.append(other)
            sorted_boxes = remaining

        return keep
