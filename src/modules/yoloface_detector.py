import cv2
from yoloface import face_analysis

class YoloFaceDetector:
    def __init__(self, conf: float = 0.25):
        """
        Build the model once.
        conf: face confidence threshold
        """
        self.model = face_analysis()
        self.conf = conf

    def detect(self, frame):
        """
        frame: cropped person image (numpy array)
        Returns list of faces in format:
        (x1, y1, x2, y2, confidence)
        """

        # Run detection
        _, boxes, confs = self.model.face_detection(frame_arr=frame, frame_status=True, model='tiny')

        outputs = []

        # boxes from face_detection are [x, y, h, w]
        # we need to convert to (x1, y1, x2, y2, confidence)
        for i in range(len(boxes)):
            box = boxes[i]
            x, y, h, w = box[0], box[1], box[2], box[3]
            x1, y1, x2, y2 = x, y, x + w, y + h
            confidence = confs[i]
            if confidence >= self.conf:
                outputs.append((x1, y1, x2, y2, confidence))

        return outputs
