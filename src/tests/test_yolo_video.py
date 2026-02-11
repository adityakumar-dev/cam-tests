"""Test: YOLOv8 + ByteTrack + YoloFace on a video file via FramePipeline."""

import sys
import argparse
import cv2

sys.path.insert(0, ".")
from frame import FramePipeline
from src.sources.video_file import VideoFileSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pipeline on video file")
    parser.add_argument("--video", type=str, default="src/tests/test5.mp4", help="Video path")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Person confidence")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS")
    parser.add_argument("--loop", action="store_true", help="Loop video")
    parser.add_argument("--no-face", action="store_true", help="Skip face detection")
    parser.add_argument("--show-motion", action="store_true", help="Show motion boxes")
    parser.add_argument("--max-dim", type=int, default=1280,
                        help="Max detection resolution (downscale large frames)")
    args = parser.parse_args()

    pipeline = FramePipeline(
        model_path=args.model,
        person_conf=args.conf,
        use_motion=True,
        use_smoothing=True,
        use_face_detection=not args.no_face,
        show_hud=True,
        show_motion_boxes=args.show_motion,
        max_detect_dim=args.max_dim,
        face_every_n_frames=3,
        max_faces_per_frame=5,
    )

    with VideoFileSource(args.video, target_fps=args.fps, loop=args.loop) as src:
        w, h = src.get_resolution()
        print(f"Video: {args.video}  res={w}x{h}  fps={src.actual_fps:.1f}")
        if max(w, h) > args.max_dim:
            s = args.max_dim / max(w, h)
            print(f"  → Detection downscaled to {int(w*s)}x{int(h*s)} for speed")
        while True:
            result = src.read()
            if result is None:
                continue
            ok, frame = result
            if not ok:
                break

            out = pipeline.process(frame)
            cv2.imshow("Video Pipeline Test", out.frame)

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
