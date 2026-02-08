import numpy as np
import torch
from bytetracker import BYTETracker

class ByteTrackTracker:
    def __init__(self, frame_rate: int = 12):
        # Default parameters (work well for most cases)
        self.tracker = BYTETracker(
            frame_rate=frame_rate,
            track_buffer=600,
            match_thresh=0.8,
        )

    def update(self, detections, frame_shape):
        """
        detections: list of (x1, y1, x2, y2, conf)s
        frame_shape: (H, W) of current frame

        Returns: list of tracks with:
        - track_id
        - box (x1,y1,x2,y2)
        """
        if len(detections) == 0:
            return []

        # Convert to ByteTrack format: [x1,y1,x2,y2,score,class_id]
        dets_np = np.array([list(det) + [0.] for det in detections], dtype=np.float32)
        dets = torch.from_numpy(dets_np) # Convert to PyTorch tensor

        # Update tracker
        tracks = self.tracker.update(dets, frame_shape)

        output = []
        for t in tracks:
            # Assuming t is a numpy array with format [x1, y1, x2, y2, track_id, ...]
            # We skip the is_activated check since numpy arrays don't have this attribute
            
            x1, y1, x2, y2 = map(int, t[0:4])
            track_id = int(t[4]) # Assuming track_id is at index 4

            output.append({
                "id": track_id,
                "box": (x1, y1, x2, y2)
            })

        return output
