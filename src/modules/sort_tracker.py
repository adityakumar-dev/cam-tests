import numpy as np
from sort import Sort

class SORTTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2,conf)
        returns list of dicts: {"id": id, "box": (x1,y1,x2,y2)}
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets)

        output = []
        for t in tracks:
            x1, y1, x2, y2, track_id = map(int, t)
            output.append({
                "id": track_id,
                "box": (x1, y1, x2, y2)
            })

        return output
