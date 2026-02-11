# Person Detection & Tracking Pipeline

Real-time person detection, tracking, and face recognition pipeline built with **YOLOv8**, **ByteTrack**, and **YoloFace**.

## Features

| Feature | Description |
|---|---|
| **Motion-guided detection** | Background subtraction (MOG2) detects motion regions; static frames skip YOLO inference for speed |
| **Adaptive confidence** | Detection threshold adjusts automatically based on scene motion level |
| **Size & aspect filtering** | Eliminates false positives that are too small, too large, or wrong shape |
| **Non-maximum suppression** | Secondary NMS removes duplicate overlapping boxes |
| **Temporal smoothing** | Averages detections across recent frames to reduce flicker |
| **ROI support** | Restrict detection to a specific region of interest |
| **ByteTrack tracking** | Robust multi-object tracking with track lifecycle (active → lost → removed) |
| **Lost-track buffering** | Keeps recently-lost tracks visible for configurable frames |
| **Per-track metadata** | Age, duration, velocity, direction, and center position per person |
| **Per-person face detection** | YoloFace runs on each person crop with automatic upscaling for small crops |
| **HUD overlay** | Live stats: person count, FPS, motion level, YOLO skip status, total tracks |
| **Auto-reconnect** | Webcam source automatically retries on read failures |
| **Video looping** | Video file source supports seamless loop playback |

## Project Structure

```
├── main.py                          # CLI entry point
├── frame.py                         # Unified frame-processing pipeline
├── yolov8n.pt / yolov8s.pt          # YOLO model weights
├── src/
│   ├── modules/
│   │   ├── yolo_persons.py          # Enhanced person detector + motion
│   │   ├── bytetrack_tracker.py     # ByteTrack with track lifecycle
│   │   ├── sort_tracker.py          # SORT tracker (alternative)
│   │   └── yoloface_detector.py     # Face detector with upscaling
│   ├── sources/
│   │   ├── webcam.py                # Webcam source + auto-reconnect
│   │   ├── video_file.py            # Video file source + looping
│   │   └── rtsp.py                  # RTSP source (planned)
│   └── tests/
│       ├── test_sources.py          # Source integration tests
│       ├── test_yolo_video.py       # Video pipeline test
│       └── test_yolo_webcam.py      # Webcam pipeline test
```

## Installation

```bash
pip install ultralytics opencv-python numpy torch bytetracker yoloface sort-tracker
```

## Quick Start

### Webcam (default)
```bash
python main.py --source webcam
```

### Video file
```bash
python main.py --source video --video-path path/to/video.mp4
```

### With more accurate model
```bash
python main.py --source webcam --model yolov8s.pt --conf 0.3
```

### Disable face detection (faster)
```bash
python main.py --source webcam --no-face
```

### Show motion regions
```bash
python main.py --source video --video-path demo.mp4 --show-motion
```

### Loop a video
```bash
python main.py --source video --video-path demo.mp4 --loop
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--source` | `webcam` | `webcam` or `video` |
| `--camera-index` | `0` | Webcam device index |
| `--video-path` | `src/tests/test.mp4` | Video file path |
| `--loop` | off | Loop video playback |
| `--model` | `yolov8n.pt` | YOLO weights (`yolov8n.pt` fast, `yolov8s.pt` accurate) |
| `--conf` | `0.4` | Base person-detection confidence |
| `--device` | auto | Force `cpu`, `cuda`, or `cuda:0` |
| `--no-motion` | off | Disable motion-guided detection |
| `--no-smoothing` | off | Disable temporal smoothing |
| `--no-face` | off | Disable face detection |
| `--face-conf` | `0.25` | Face-detection confidence |
| `--fps` | native | Target playback FPS |
| `--no-hud` | off | Hide the HUD overlay |
| `--show-motion` | off | Draw motion-region rectangles |
| `--verbose` | off | Debug logging |

## How It Works

1. **Frame read** → webcam / video file source with FPS throttling
2. **Motion detection** → MOG2 background subtraction computes motion score
3. **Motion gate** → if motion < threshold and warm-up done, YOLO is skipped (reuse last detections)
4. **Adaptive confidence** → conf threshold lowered when scene is busy, raised when calm
5. **YOLO inference** → `classes=[0]` (person only), with optional ROI crop
6. **Size / aspect filter** → removes boxes that are too small, too tall, or wrong ratio
7. **NMS** → greedy non-maximum suppression on remaining boxes
8. **Temporal smoothing** → averages matched boxes across last N frames
9. **ByteTrack** → assigns persistent IDs; manages active / lost / removed lifecycle
10. **Face detection** → YoloFace runs on each person crop (with upscaling for small crops)
11. **HUD overlay** → draws boxes, IDs, face boxes, and live stats

## License

MIT
