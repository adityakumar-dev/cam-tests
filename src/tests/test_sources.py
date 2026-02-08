import cv2
import argparse
from src.sources.webcam import WebcamSource
from src.sources.video_file import VideoFileSource


def test_webcam(camera_index=0, fps=15):
    """Test webcam source"""
    print(f"Starting webcam test with camera index {camera_index} at {fps} FPS...")
    with WebcamSource(camera_index=camera_index, target_fps=fps) as cam:
        while True:
            result = cam.read()
            if result is None:
                continue

            success, frame = result
            if not success:
                break

            cv2.imshow("Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break

    cv2.destroyAllWindows()
    print("Webcam test completed")


def test_video_file(video_path, fps=30):
    """Test video file source"""
    print(f"Starting video file test: {video_path} at {fps} FPS...")
    with VideoFileSource(video_path, target_fps=fps) as cam:
        while True:
            result = cam.read()
            if result is None:
                continue

            success, frame = result
            if not success:
                break

            cv2.imshow("Video Test", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    print("Video file test completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test video sources")
    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "video"],
        default="video",
        help="Which source to test: 'webcam' or 'video'"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for webcam source (default: 0)"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="src/tests/test.mp4",
        help="Path to video file (default: src/tests/test.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30)"
    )

    args = parser.parse_args()

    if args.source == "webcam":
        test_webcam(camera_index=args.camera_index, fps=args.fps)
    elif args.source == "video":
        test_video_file(args.video_path, fps=args.fps)
