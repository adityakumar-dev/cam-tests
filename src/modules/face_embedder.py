"""
Face embedding extractor using InsightFace (ArcFace, 512-d).

Workflow:
  1. Detect face locations in a BGR person crop.
  2. Extract 512-dim embedding per face (normalised → cosine distance).
  3. Score image clarity via Laplacian variance so the caller can keep
     the sharpest capture.

InsightFace bundles a face-detector + ArcFace recognition model.
The 512-d embeddings are L2-normalised, so Euclidean distance on them
equals  2·(1 - cosine_similarity).  Threshold ≈ 1.2 works well.
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from insightface.app import FaceAnalysis
    IF_AVAILABLE = True
except Exception as _exc:
    IF_AVAILABLE = False
    logger.warning("insightface import failed: %s", _exc, exc_info=True)


class FaceEmbedder:
    """
    Detects faces and produces 512-d ArcFace embeddings via ``insightface``.
    Also exposes a *clarity score* so the database can always keep the
    sharpest capture.
    """

    EMBED_DIM = 512
    MIN_FACE_SIZE = 30  # px — skip tiny crops

    def __init__(self, det_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            det_size: detection input size (w, h).  Smaller = faster.
        """
        self._app: Optional["FaceAnalysis"] = None
        if IF_AVAILABLE:
            try:
                self._app = FaceAnalysis(
                    name="buffalo_l",
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                self._app.prepare(ctx_id=-1, det_size=det_size)
                logger.info("InsightFace ready (det_size=%s)", det_size)
            except Exception as exc:
                logger.error("InsightFace init failed: %s", exc)
                self._app = None

    # ── helpers ──────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._app is not None

    @staticmethod
    def clarity_score(bgr_image: np.ndarray) -> float:
        """
        Compute image sharpness using the variance of the Laplacian.
        Higher value → sharper / more in-focus face.
        """
        if bgr_image is None or bgr_image.size == 0:
            return 0.0
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def face_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Euclidean distance between two 512-d embeddings (lower = same).
        For L2-normalised vectors this ranges from 0 (identical) to 2."""
        return float(np.linalg.norm(np.asarray(emb_a) - np.asarray(emb_b)))

    @staticmethod
    def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity (higher = same person)."""
        a = np.asarray(emb_a, dtype=np.float64)
        b = np.asarray(emb_b, dtype=np.float64)
        dot = np.dot(a, b)
        norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(dot / norm)

    # ── main API ─────────────────────────────────────────────────────

    def get_embeddings(
        self, bgr_frame: np.ndarray
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float, float]]:
        """
        Detect faces and extract embeddings from a BGR image.

        Returns
        -------
        list of (embedding_512d, (x1,y1,x2,y2), clarity, det_area_ratio)
            - embedding:  np.float64 shape (512,)
            - bbox:       face location in input image coords
            - clarity:    Laplacian variance of the face crop (higher=better)
            - det_area_ratio: face area / image area (for size filtering)
        """
        if self._app is None or bgr_frame is None or bgr_frame.size == 0:
            return []

        h, w = bgr_frame.shape[:2]
        if h < self.MIN_FACE_SIZE or w < self.MIN_FACE_SIZE:
            return []

        try:
            faces = self._app.get(bgr_frame)
            if not faces:
                return []
        except Exception as exc:
            logger.debug("InsightFace error: %s", exc)
            return []

        img_area = h * w
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Clamp to image bounds
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(w, x2)
            y2c = min(h, y2)

            # Compute clarity of this specific face crop
            face_crop = bgr_frame[y1c:y2c, x1c:x2c]
            if face_crop.size == 0:
                continue
            clarity = self.clarity_score(face_crop)

            face_area = max(1, (x2c - x1c) * (y2c - y1c))
            area_ratio = face_area / max(img_area, 1)

            embedding = face.normed_embedding  # 512-d, L2-normalised
            if embedding is None:
                continue

            results.append((
                np.asarray(embedding, dtype=np.float64),
                (x1, y1, x2, y2),
                clarity,
                area_ratio,
            ))

        return results

    def get_embedding_for_crop(
        self, face_crop: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Extract a single embedding + clarity from a tight face crop.

        Returns (embedding_512d, clarity) or None.
        """
        results = self.get_embeddings(face_crop)
        if not results:
            return None
        best = max(results, key=lambda r: r[2])  # best clarity
        return best[0], best[2]
