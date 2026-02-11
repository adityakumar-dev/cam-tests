#!/usr/bin/env python3
"""
main.py — CLI entry point for the person-detection pipeline.

Usage examples:
    python main.py --source webcam
    python main.py --source video --video-path sample.mp4 --model yolov8s.pt
    python main.py --source video --video-path demo.mp4 --no-face --show-motion
"""

import argparse
import logging
import sys
import cv2

from frame import FramePipeline
from src.sources.webcam import WebcamSource
from src.sources.video_file import VideoFileSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Person detection + tracking + face detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- source ---
    p.add_argument(
        "--source", choices=["webcam", "video"], default="webcam",
        help="Input source type.",
    )
    p.add_argument("--camera-index", type=int, default=0, help="Webcam device index.")
    p.add_argument("--video-path", type=str, default="src/tests/test.mp4", help="Path to video file.")
    p.add_argument("--loop", action="store_true", help="Loop video file playback.")

    # --- model ---
    p.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 weights file.")
    p.add_argument("--conf", type=float, default=0.4, help="Base person-detection confidence.")
    p.add_argument("--device", type=str, default=None, help="Force device (cpu, cuda, cuda:0).")

    # --- features ---
    p.add_argument("--no-motion", action="store_true", help="Disable motion-guided detection.")
    p.add_argument("--no-smoothing", action="store_true", help="Disable temporal smoothing.")
    p.add_argument("--no-face", action="store_true", help="Disable face detection.")
    p.add_argument("--face-conf", type=float, default=0.25, help="Face-detection confidence.")

    # --- display ---
    p.add_argument("--fps", type=int, default=None, help="Target playback FPS (None = native).")
    p.add_argument("--no-hud", action="store_true", help="Hide the heads-up display overlay.")
    p.add_argument("--show-motion", action="store_true", help="Draw motion-region boxes.")
    p.add_argument("--window", type=str, default="Person Detection Pipeline", help="Window title.")

    # --- performance ---
    p.add_argument("--max-dim", type=int, default=1280,
                   help="Downscale frames to this max dimension for detection.")

    # --- misc ---
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")

    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    pipeline = FramePipeline(
        model_path=args.model,
        person_conf=args.conf,
        use_motion=not args.no_motion,
        use_smoothing=not args.no_smoothing,
        use_face_detection=not args.no_face,
        face_conf=args.face_conf,
        show_hud=not args.no_hud,
        show_motion_boxes=args.show_motion,
        device=args.device,
        max_detect_dim=args.max_dim,
    )

    # Build the video source
    if args.source == "webcam":
        source = WebcamSource(camera_index=args.camera_index, target_fps=args.fps)
    else:
        source = VideoFileSource(args.video_path, target_fps=args.fps, loop=args.loop)

    with source:
        print(f"[INFO] Source opened — resolution {source.get_resolution()}")
        print("[INFO] Press ESC or 'q' to quit.\n")

        while True:
            result = source.read()
            if result is None:
                continue

            success, frame = result
            if not success:
                break

            out = pipeline.process(frame)

            cv2.imshow(args.window, out.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        # Print final stats
        det_stats = pipeline.detector.get_stats()
        trk_stats = pipeline.tracker.get_stats()
        src_stats = source.get_stats()

        print("\n── Final Statistics ──")
        print(f"  Source      : {src_stats}")
        print(f"  Detector    : {det_stats}")
        print(f"  Tracker     : {trk_stats}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())