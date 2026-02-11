"""
FastAPI web server for the person-detection + face re-identification pipeline.

Provides:
- Live MJPEG video stream in the browser
- WebSocket for real-time face tracking events
- REST API for person gallery, stats, thumbnails
- Full dashboard UI
"""

import asyncio
import json
import time
import logging
import threading
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from frame import FramePipeline, FrameResult, PersonResult
from src.sources.webcam import WebcamSource
from src.sources.video_file import VideoFileSource
from src.modules.face_embedder import FaceEmbedder
from src.modules.face_database import FaceDatabase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────

class AppState:
    """Shared application state for the pipeline and streaming."""

    def __init__(self):
        self.pipeline: Optional[FramePipeline] = None
        self.embedder: Optional[FaceEmbedder] = None
        self.face_db: Optional[FaceDatabase] = None
        self.source = None
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_result: Optional[FrameResult] = None
        self.running = False
        self.lock = threading.Lock()
        self.ws_clients: list[WebSocket] = []
        self.frame_count = 0
        self.start_time = 0.0
        self._pipeline_thread: Optional[threading.Thread] = None
        # config
        self.source_type = "webcam"
        self.video_path = "src/tests/test.mp4"
        self.camera_index = 0
        self.model_path = "yolov8n.pt"


state = AppState()


# ──────────────────────────────────────────────────────────────────────
# Pipeline worker thread
# ──────────────────────────────────────────────────────────────────────

def _run_pipeline_loop():
    """Background thread: reads frames, runs pipeline, extracts embeddings."""
    logger.info("Pipeline thread started")

    if state.source_type == "webcam":
        source = WebcamSource(camera_index=state.camera_index)
    else:
        source = VideoFileSource(state.video_path, loop=True)

    state.source = source

    with source:
        logger.info("Source opened: %s", source.get_resolution())
        state.start_time = time.time()

        while state.running:
            result = source.read()
            if result is None:
                continue
            success, frame = result
            if not success:
                if state.source_type == "video":
                    continue  # loop handles restart
                break

            # Run the detection pipeline
            out = state.pipeline.process(frame)
            state.frame_count += 1

            # Face embedding + re-identification
            _process_face_embeddings(frame, out)

            with state.lock:
                state.latest_frame = out.frame
                state.latest_result = out

    logger.info("Pipeline thread stopped")


def _process_face_embeddings(raw_frame: np.ndarray, result: FrameResult):
    """
    Extract face embeddings and match/register persons.

    Logic per tracked person:
      1. First time face detected → register_or_match (creates new person
         or matches existing via embedding distance).
      2. Every subsequent frame where a face is found → try_upgrade_capture
         compares clarity (Laplacian sharpness) and replaces the best capture
         if the new one is sharper.
      3. Raw face crop is *always* saved so we have a full history.
    """
    if state.embedder is None or not state.embedder.available:
        return
    if state.face_db is None:
        return

    active_track_ids = set()

    for person in result.persons:
        if person.status != "active":
            continue

        active_track_ids.add(person.track_id)

        # Update session duration for every active person
        state.face_db.update_session_duration(person.track_id)

        # Extract person crop from the raw (un-annotated) frame
        x1, y1, x2, y2 = person.box
        h, w = raw_frame.shape[:2]
        px1, py1 = max(0, x1), max(0, y1)
        px2, py2 = min(w, x2), min(h, y2)
        person_crop = raw_frame[py1:py2, px1:px2]
        if person_crop.size == 0:
            continue

        # Run face detection + embedding on the person crop
        face_results = state.embedder.get_embeddings(person_crop)
        if not face_results:
            continue

        # Pick the face with the best clarity
        embedding, face_bbox, clarity, area_ratio = max(
            face_results, key=lambda r: r[2]
        )

        # Cut out the face sub-crop for the thumbnail
        fx1, fy1, fx2, fy2 = face_bbox
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2 = min(person_crop.shape[1], fx2)
        fy2 = min(person_crop.shape[0], fy2)
        face_crop = person_crop[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            face_crop = person_crop

        # Resize for storage (consistent thumbnail size)
        face_thumb = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Check if this track already has a session (person already identified)
        session = state.face_db.get_session_for_track(person.track_id)

        if session and session.face_captured:
            # ── SUBSEQUENT DETECTION ──
            # Person already identified for this track.
            # Try to upgrade the best capture if this frame is sharper.
            upgraded = state.face_db.try_upgrade_capture(
                person_id=session.person_id,
                face_crop=face_thumb,
                raw_crop=person_crop,
                clarity=clarity,
                embedding=embedding,
            )
            if upgraded:
                _draw_label(raw_frame, x1, y2, session.person_id, clarity, upgraded=True)
        else:
            # ── FIRST DETECTION for this track ──
            person_id, is_new, dist, cap_upgraded = state.face_db.register_or_match(
                embedding=embedding,
                face_crop=face_thumb,
                raw_person_crop=person_crop,
                clarity=clarity,
                track_id=person.track_id,
            )
            _draw_label(raw_frame, x1, y2, person_id, clarity, is_new=is_new)

    # Clean up sessions for tracks that left the frame
    for track_id in list(state.face_db._active_sessions.keys()):
        if track_id not in active_track_ids:
            state.face_db.end_session(track_id)


def _draw_label(
    raw_frame: np.ndarray,
    x1: int, y2: int,
    person_id: str,
    clarity: float,
    is_new: bool = False,
    upgraded: bool = False,
):
    """Draw the re-ID label on the annotated frame."""
    label_info = state.face_db.get_person(person_id)
    if not label_info:
        return

    name = label_info.get("name", "")
    visits = label_info.get("total_visits", 1)
    id_text = f"{name} v{visits} c:{clarity:.0f}"
    if upgraded:
        id_text += " ⬆"

    color = (0, 200, 255) if is_new else (0, 255, 100)
    if upgraded:
        color = (255, 200, 0)

    with state.lock:
        if state.latest_frame is not None:
            cv2.putText(
                state.latest_frame, id_text,
                (x1, max(0, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )


# ──────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the pipeline on app startup, stop on shutdown."""
    # Initialize components
    state.pipeline = FramePipeline(
        model_path=state.model_path,
        person_conf=0.4,
        use_motion=True,
        use_smoothing=True,
        use_face_detection=True,
        face_conf=0.25,
        show_hud=True,
        show_motion_boxes=False,
    )
    state.embedder = FaceEmbedder()
    state.face_db = FaceDatabase(db_dir="face_db")
    state.running = True

    # Start pipeline in background thread
    state._pipeline_thread = threading.Thread(target=_run_pipeline_loop, daemon=True)
    state._pipeline_thread.start()

    # Start WebSocket broadcaster
    broadcast_task = asyncio.create_task(_ws_broadcaster())

    logger.info("Application started")
    yield

    # Shutdown
    state.running = False
    broadcast_task.cancel()
    if state._pipeline_thread:
        state._pipeline_thread.join(timeout=5.0)
    logger.info("Application stopped")


app = FastAPI(
    title="Person Detection & Face Re-ID Dashboard",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve face thumbnails and raw captures
import os
os.makedirs("face_db/faces", exist_ok=True)
os.makedirs("face_db/raw_captures", exist_ok=True)
app.mount("/faces", StaticFiles(directory="face_db/faces"), name="faces")
app.mount("/raw", StaticFiles(directory="face_db/raw_captures"), name="raw_captures")


# ──────────────────────────────────────────────────────────────────────
# MJPEG video stream
# ──────────────────────────────────────────────────────────────────────

def _generate_mjpeg():
    """Yield MJPEG frames for the video stream."""
    while state.running:
        with state.lock:
            frame = state.latest_frame

        if frame is None:
            time.sleep(0.03)
            continue

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(0.033)  # ~30fps


@app.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint."""
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ──────────────────────────────────────────────────────────────────────
# WebSocket for real-time events
# ──────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time tracking events."""
    await websocket.accept()
    state.ws_clients.append(websocket)
    logger.info("WebSocket client connected (%d total)", len(state.ws_clients))
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Handle rename commands from UI
            try:
                msg = json.loads(data)
                if msg.get("action") == "rename":
                    pid = msg.get("person_id")
                    new_name = msg.get("name", "")
                    if pid and new_name and state.face_db:
                        state.face_db.rename_person(pid, new_name)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        state.ws_clients.remove(websocket)
        logger.info("WebSocket client disconnected (%d remain)", len(state.ws_clients))


async def _ws_broadcaster():
    """Periodically broadcast state updates to all WebSocket clients."""
    last_event_count = 0
    while True:
        await asyncio.sleep(0.5)  # broadcast every 500ms

        if not state.ws_clients or state.face_db is None:
            continue

        try:
            result = state.latest_result
            persons_data = []
            if result:
                for p in result.persons:
                    pd = {
                        "track_id": p.track_id,
                        "box": list(p.box),
                        "status": p.status,
                        "duration_s": p.duration_s,
                        "faces_detected": len(p.faces),
                    }
                    # Add person identity info if available
                    session = state.face_db.get_session_for_track(p.track_id)
                    if session:
                        person_info = state.face_db.get_person(session.person_id)
                        if person_info:
                            pd["person_id"] = session.person_id
                            pd["person_name"] = person_info.get("name", "")
                            pd["total_visits"] = person_info.get("total_visits", 0)
                    persons_data.append(pd)

            db_stats = state.face_db.get_stats()
            events = state.face_db.get_recent_events(limit=20)
            new_events = events[last_event_count:] if len(events) > last_event_count else []
            last_event_count = len(events)

            elapsed = time.time() - state.start_time if state.start_time else 0
            fps = state.frame_count / max(elapsed, 0.01)

            payload = json.dumps({
                "type": "update",
                "fps": round(fps, 1),
                "frame_count": state.frame_count,
                "persons_in_frame": persons_data,
                "total_persons_db": db_stats["total_persons"],
                "active_sessions": db_stats["active_sessions"],
                "gallery": state.face_db.get_all_persons()[:20],
                "new_events": new_events,
            })

            disconnected = []
            for ws in state.ws_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.append(ws)
            for ws in disconnected:
                state.ws_clients.remove(ws)

        except Exception as e:
            logger.debug("Broadcast error: %s", e)


# ──────────────────────────────────────────────────────────────────────
# REST API endpoints
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/persons")
async def get_persons():
    """Get all tracked persons."""
    if state.face_db is None:
        return JSONResponse({"persons": []})
    return JSONResponse({"persons": state.face_db.get_all_persons()})


@app.get("/api/persons/{person_id}")
async def get_person(person_id: str):
    """Get a specific person."""
    if state.face_db is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    person = state.face_db.get_person(person_id)
    if not person:
        return JSONResponse({"error": "not found"}, status_code=404)
    # Include thumbnail filenames and best capture
    thumbs = state.face_db.get_person_thumbnails(person_id)
    thumb_urls = [f"/faces/{os.path.basename(t)}" for t in thumbs]
    best = state.face_db.get_best_capture_path(person_id)
    person["thumbnails"] = thumb_urls
    person["best_capture"] = f"/faces/{os.path.basename(best)}" if best else None
    return JSONResponse(person)


@app.get("/api/persons/{person_id}/thumbnails")
async def get_person_thumbnails(person_id: str):
    """Get thumbnail URLs for a person."""
    if state.face_db is None:
        return JSONResponse({"thumbnails": []})
    thumbs = state.face_db.get_person_thumbnails(person_id)
    urls = [f"/faces/{os.path.basename(t)}" for t in thumbs]
    best = state.face_db.get_best_capture_path(person_id)
    best_url = f"/faces/{os.path.basename(best)}" if best else None
    return JSONResponse({
        "person_id": person_id,
        "thumbnails": urls,
        "best_capture": best_url,
    })


@app.post("/api/persons/{person_id}/rename")
async def rename_person(person_id: str, name: str = Query(...)):
    """Rename a tracked person."""
    if state.face_db is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    success = state.face_db.rename_person(person_id, name)
    if success:
        return JSONResponse({"ok": True, "person_id": person_id, "name": name})
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/api/events")
async def get_events(limit: int = 50):
    """Get recent tracking events."""
    if state.face_db is None:
        return JSONResponse({"events": []})
    return JSONResponse({"events": state.face_db.get_recent_events(limit)})


@app.get("/api/stats")
async def get_stats():
    """Get pipeline and database stats."""
    result = state.latest_result
    elapsed = time.time() - state.start_time if state.start_time else 0
    fps = state.frame_count / max(elapsed, 0.01)

    data = {
        "fps": round(fps, 1),
        "frame_count": state.frame_count,
        "uptime_s": round(elapsed, 1),
        "persons_in_frame": len(result.persons) if result else 0,
        "motion_score": round(result.motion_score * 100, 1) if result else 0,
        "embedder_available": state.embedder.available if state.embedder else False,
    }
    if state.face_db:
        data.update(state.face_db.get_stats())

    if result:
        data["detector_stats"] = result.detector_stats
        data["tracker_stats"] = result.tracker_stats

    return JSONResponse(data)


@app.get("/api/active")
async def get_active_sessions():
    """Get currently active tracking sessions."""
    if state.face_db is None:
        return JSONResponse({"sessions": []})
    return JSONResponse({"sessions": state.face_db.get_active_sessions()})


# ──────────────────────────────────────────────────────────────────────
# Main HTML page
# ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main dashboard page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())
