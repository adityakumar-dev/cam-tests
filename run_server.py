#!/usr/bin/env python3
"""
run_server.py — Launch the FastAPI web dashboard for person detection + face re-ID.

Usage:
    python run_server.py                           # webcam, default settings
    python run_server.py --source video --video-path demo.mp4
    python run_server.py --port 8080 --host 0.0.0.0
    python run_server.py --model yolov8s.pt --camera 1
"""

# ── Fix conda/pyenv GLIBCXX mismatch ─────────────────────────────────
# Miniconda ships an old libstdc++ (GLIBCXX ≤3.4.29) but native
# extensions (insightface mesh_core_cython, dlib) need ≥3.4.32.
# If we detect the system lib is newer, re-exec with LD_PRELOAD.
import os, sys

_PRELOAD_MARKER = "_LIBSTDCXX_PRELOADED"
if not os.environ.get(_PRELOAD_MARKER):
    _sys_lib = None
    for _p in ("/usr/lib/libstdc++.so.6", "/usr/lib64/libstdc++.so.6"):
        if os.path.isfile(_p):
            _sys_lib = _p
            break
    if _sys_lib:
        os.environ[_PRELOAD_MARKER] = "1"
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if _sys_lib not in ld_preload:
            os.environ["LD_PRELOAD"] = (
                f"{_sys_lib}:{ld_preload}" if ld_preload else _sys_lib
            )
        os.execv(sys.executable, [sys.executable] + sys.argv)
# ──────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch the person detection + face re-ID web dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server
    p.add_argument("--host", type=str, default="127.0.0.1", help="Server bind host")
    p.add_argument("--port", type=int, default=8000, help="Server port")

    # Source
    p.add_argument("--source", choices=["webcam", "video"], default="webcam",
                   help="Input source type")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index")
    p.add_argument("--video-path", type=str, default="src/tests/test.mp4",
                   help="Path to video file (when --source video)")

    # Model
    p.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model weights")

    # Misc
    p.add_argument("--db-dir", type=str, default="face_db", help="Face database directory")
    p.add_argument("--verbose", action="store_true", help="Debug logging")

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    # Configure the server state before importing/starting
    import server
    server.state.source_type = args.source
    server.state.video_path = args.video_path
    server.state.camera_index = args.camera
    server.state.model_path = args.model

    # Update DB dir
    os.makedirs(args.db_dir, exist_ok=True)

    import uvicorn

    print("=" * 60)
    print("  Person Detection & Face Re-ID Dashboard")
    print("=" * 60)
    print(f"  Source    : {args.source} (camera={args.camera})")
    print(f"  Model     : {args.model}")
    print(f"  Face DB   : {args.db_dir}/")
    print(f"  Dashboard : http://{args.host}:{args.port}")
    print("=" * 60)
    print()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        log_level="info" if not args.verbose else "debug",
    )


if __name__ == "__main__":
    main()
