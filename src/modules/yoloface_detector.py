"""YoloFace detector with upscaling for small crops and robust error handling."""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
from yoloface import face_analysis

logger = logging.getLogger(__name__)


class YoloFaceDetector:
    """
    Face detector backed by yoloface-tiny.

    Improvements over the basic version:
    - Upscales tiny crops so the model can still find faces
    - Clamps output boxes to crop boundaries
    - Graceful error handling (never crashes the pipeline)
    """

    MIN_INPUT_DIM = 48          # absolute minimum px to even attempt
    UPSCALE_TARGET_DIM = 160    # if crop is smaller, resize up to this

    def __init__(self, conf: float = 0.25, model_variant: str = "tiny"):
        """
        Args:
            conf:            Face confidence threshold.
            model_variant:   'tiny' (fast) or 'full'.
        """
        self.conf = conf
        self._variant = model_variant
        try:
            self.model = face_analysis()
        except Exception as exc:
            logger.error("Failed to load yoloface model: %s", exc)
            self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a person crop.

        Args:
            frame:  BGR image (usually a person bounding-box crop).

        Returns:
            List of (x1, y1, x2, y2, confidence) in *original* crop coords.
        """
        if self.model is None:
            return []

        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if h < self.MIN_INPUT_DIM or w < self.MIN_INPUT_DIM:
            return []

        # Upscale small crops so the face model has enough pixels
        scale = 1.0
        work_frame = frame
        if h < self.UPSCALE_TARGET_DIM or w < self.UPSCALE_TARGET_DIM:
            scale = self.UPSCALE_TARGET_DIM / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            work_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        try:
            _, boxes, confs = self.model.face_detection(
                frame_arr=work_frame, frame_status=True, model=self._variant
            )
        except Exception as exc:
            logger.debug("Face detection error: %s", exc)
            return []

        if not boxes or not confs:
            return []

        outputs: List[Tuple[int, int, int, int, float]] = []
        for i, box in enumerate(boxes):
            confidence = float(confs[i])
            if confidence < self.conf:
                continue

            # yoloface returns [x, y, h, w]
            bx, by, bh, bw = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh

            # Undo upscaling
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)

            # Clamp to original crop dimensions
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            outputs.append((x1, y1, x2, y2, confidence))

        return outputs
