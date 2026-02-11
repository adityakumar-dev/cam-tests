"""Test: YOLOv8 + ByteTrack + YoloFace on webcam via FramePipeline."""

import sys
import argparse
import cv2

sys.path.insert(0, ".")
from frame import FramePipeline
from src.sources.webcam import WebcamSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pipeline on webcam")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.35, help="Person confidence")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS")
    parser.add_argument("--no-face", action="store_true", help="Skip face detection")
    parser.add_argument("--show-motion", action="store_true", help="Show motion boxes")
    args = parser.parse_args()

    pipeline = FramePipeline(
        model_path=args.model,
        person_conf=args.conf,
        use_motion=True,
        use_smoothing=True,
        use_face_detection=not args.no_face,
        show_hud=True,
        show_motion_boxes=args.show_motion,
    )

    with WebcamSource(camera_index=args.camera, target_fps=args.fps) as src:
        print(f"Webcam {args.camera}: res={src.get_resolution()}  fps={src.actual_fps}")
        while True:
            result = src.read()
            if result is None:
                continue
            ok, frame = result
            if not ok:
                break

            out = pipeline.process(frame)
            cv2.imshow("Webcam Pipeline Test", out.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    det = pipeline.detector.get_stats()
    trk = pipeline.tracker.get_stats()
    print(f"\nDetector stats: {det}")
    print(f"Tracker  stats: {trk}")
    print(f"Source   stats: {src.get_stats()}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
