import cv2
import time
from typing import Optional, Tuple

class WebcamSource:
    def __init__(self, camera_index: int = 0, target_fps: Optional[int] = None) -> None:
        """
        :param camera_index: webcam index (0 = default)
        :param target_fps: if set, limits frame rate (e.g., 10 or 15)
        """
        self.camera_index = camera_index
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0

    def _open(self) -> None:
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required to use WebcamSource. "
                "Please install it with: pip install opencv-python"
            )

        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open webcam at index {self.camera_index}")

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Optional[Tuple[bool, "cv2.Mat"]]:
        """
        Reads one frame from webcam.
        Returns (success, frame) or None if not opened.
        """
        if self._cap is None:
            raise RuntimeError("Webcam not opened. Call _open() first.")

        # FPS control (optional)
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

    # Optional: make it usable in "with" statement
    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._close()
