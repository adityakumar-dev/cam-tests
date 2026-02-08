import cv2
from typing import Optional, Tuple
import time


class VideoFileSource:
    def __init__(self, path: str, target_fps: Optional[int] = None) -> None:
        self.path = path
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.path}")

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[Tuple[bool, "cv2.Mat"]]:
        if self._cap is None:
            raise RuntimeError("Video not opened. Call _open() first.")

        if self.target_fps is not None:
            now = time.time()
            min_interval = 1.0 / self.target_fps
            if now - self._last_frame_time < min_interval:
                return None
            self._last_frame_time = now

        ret, frame = self._cap.read()
        if not ret:
            return None

        return True, frame

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close()
