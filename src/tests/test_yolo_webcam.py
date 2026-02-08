import cv2
from src.modules.yolo_persons import YOLOv8PersonDetector
from src.sources.webcam import WebcamSource
from src.modules.bytetrack_tracker import ByteTrackTracker
from src.modules.yoloface_detector import YoloFaceDetector

detector = YOLOv8PersonDetector(model_path="yolov8n.pt", conf=0.25)
tracker = ByteTrackTracker(frame_rate=12)
face_detector = YoloFaceDetector()

with WebcamSource(camera_index=0, target_fps=12) as cam:
    while True:
        result = cam.read()
        if result is None:
            continue

        success, frame = result
        if not success:
            break

        h, w = frame.shape[:2]

        # Step 1: Detect persons
        persons = detector.detect(frame)

        # Step 2: Track persons
        tracks = tracker.update(persons, (h, w))

        for t in tracks:
            x1, y1, x2, y2 = t["box"]
            pid = t["id"]

            # Draw person box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"P_{pid}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # === NEW PART: PER-PERSON YoloFace ===
            person_crop = frame[y1:y2, x1:x2]

            # Run YoloFace only inside this crop
            faces = face_detector.detect(person_crop)

            for (fx1, fy1, fx2, fy2, fconf) in faces:
                # Convert face coords back to original frame
                abs_x1 = x1 + fx1
                abs_y1 = y1 + fy1
                abs_x2 = x1 + fx2
                abs_y2 = y1 + fy2

                # Draw face box (blue)
                cv2.rectangle(
                    frame,
                    (abs_x1, abs_y1),
                    (abs_x2, abs_y2),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"face {fconf:.2f}",
                    (abs_x1, max(0, abs_y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

        cv2.imshow("YOLOv8 + ByteTrack + YoloFace", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()
