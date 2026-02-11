"""
Persistent face database for person re-identification.

Design
------
For every person we store **two kinds of face data**:

1. **Raw face captures** (JPEG on disk) — every time a face is detected
   we save the raw crop.  We keep a *best* capture: the one with the
   highest **clarity score** (Laplacian variance).  Each new detection
   is compared against the current best; if the new frame is sharper
   the best-capture file is replaced.

2. **128-d face embeddings** (BLOBs in SQLite) — used for matching.
   We store up to N embeddings per person so the average embedding
   adapts as we see the person from different angles.

Flow
~~~~
face detected → capture raw + compute embedding + clarity score
   → is this a known person?  (match via embedding distance)
       YES → update last_seen, add embedding, compare clarity
              → new capture clearer?  replace best capture
       NO  → register new person, save as initial best capture
   → next frame, face detected again → repeat, keep upgrading
"""

import os
import json
import time
import uuid
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PersonRecord:
    """A unique person in the database."""
    person_id: str
    name: str
    first_seen: float
    last_seen: float
    total_visits: int = 1
    total_duration_s: float = 0.0
    best_clarity: float = 0.0             # clarity of the best raw capture
    best_capture_path: str = ""           # path to the sharpest face JPEG
    embeddings: List[np.ndarray] = field(default_factory=list, repr=False)
    all_capture_paths: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def avg_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        avg = np.mean(self.embeddings, axis=0)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-8)

    def to_api_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "first_seen": self.first_seen,
            "first_seen_str": datetime.fromtimestamp(self.first_seen).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "last_seen": self.last_seen,
            "last_seen_str": datetime.fromtimestamp(self.last_seen).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "total_visits": self.total_visits,
            "total_duration_s": round(self.total_duration_s, 1),
            "num_embeddings": len(self.embeddings),
            "best_clarity": round(self.best_clarity, 1),
            "capture_count": len(self.all_capture_paths),
            "has_best_capture": bool(self.best_capture_path),
        }


@dataclass
class TrackSession:
    """One continuous tracking session of a person."""
    session_id: str
    person_id: str
    track_id: int
    started: float
    last_update: float
    face_captured: bool = False
    best_clarity_this_session: float = 0.0  # best clarity seen in this track


# ──────────────────────────────────────────────────────────────────────
# FaceDatabase
# ──────────────────────────────────────────────────────────────────────

class FaceDatabase:
    """
    Persistent face identity database with **clarity-based best-capture**.

    • Embeddings stored as BLOBs in SQLite (no separate .npy files).
    • Raw face images stored on disk; the sharpest one is marked as
      ``best_capture``.
    • Thread-safe for use with FastAPI.
    """

    DISTANCE_THRESHOLD = 1.20   # InsightFace 512-d L2-normalised Euclidean
    MAX_EMBEDDINGS_PER_PERSON = 15
    CAPTURE_COOLDOWN_S = 0.8    # don't re-embed the same track faster than this
    CLARITY_UPGRADE_MARGIN = 5.0  # new capture must be this much sharper to replace

    def __init__(self, db_dir: str = "face_db"):
        self._db_dir = os.path.abspath(db_dir)
        self._images_dir = os.path.join(self._db_dir, "faces")
        self._raw_dir = os.path.join(self._db_dir, "raw_captures")
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._raw_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._persons: Dict[str, PersonRecord] = {}
        self._active_sessions: Dict[int, TrackSession] = {}
        self._last_capture_time: Dict[int, float] = {}
        self._person_counter = 0
        self._events: List[dict] = []
        self._max_events = 200

        self._db_path = os.path.join(self._db_dir, "faces.db")
        self._init_db()
        self._load_from_db()

    # ── SQLite schema ────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                person_id       TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                first_seen      REAL NOT NULL,
                last_seen       REAL NOT NULL,
                total_visits    INTEGER DEFAULT 1,
                total_duration_s REAL DEFAULT 0.0,
                best_clarity    REAL DEFAULT 0.0,
                best_capture    TEXT DEFAULT '',
                metadata        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                clarity     REAL DEFAULT 0.0,
                created_at  REAL NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            );

            CREATE TABLE IF NOT EXISTS captures (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT NOT NULL,
                path        TEXT NOT NULL,
                clarity     REAL DEFAULT 0.0,
                is_best     INTEGER DEFAULT 0,
                created_at  REAL NOT NULL,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            );

            CREATE TABLE IF NOT EXISTS sightings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                duration_s  REAL DEFAULT 0.0,
                FOREIGN KEY (person_id) REFERENCES persons(person_id)
            );
        """)
        conn.commit()
        conn.close()

    # ── load ─────────────────────────────────────────────────────────

    def _load_from_db(self):
        conn = self._get_conn()

        for row in conn.execute("SELECT * FROM persons").fetchall():
            pid, name, first_seen, last_seen, visits, dur, best_cl, best_cap, meta = row

            # Load embeddings
            emb_rows = conn.execute(
                "SELECT embedding FROM embeddings WHERE person_id=? ORDER BY id",
                (pid,),
            ).fetchall()
            embeddings = []
            for (blob,) in emb_rows:
                embeddings.append(np.frombuffer(blob, dtype=np.float64).copy())

            # Load captures
            cap_rows = conn.execute(
                "SELECT path FROM captures WHERE person_id=? ORDER BY id",
                (pid,),
            ).fetchall()
            all_caps = [r[0] for r in cap_rows]

            self._persons[pid] = PersonRecord(
                person_id=pid,
                name=name,
                first_seen=first_seen,
                last_seen=last_seen,
                total_visits=visits,
                total_duration_s=dur,
                best_clarity=best_cl,
                best_capture_path=best_cap or "",
                embeddings=embeddings,
                all_capture_paths=all_caps,
                metadata=json.loads(meta) if meta else {},
            )

        conn.close()
        self._person_counter = len(self._persons)
        logger.info("Loaded %d persons from face database", len(self._persons))

    # ── persistence helpers ──────────────────────────────────────────

    def _save_person_meta(self, person: PersonRecord):
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO persons
            (person_id, name, first_seen, last_seen, total_visits,
             total_duration_s, best_clarity, best_capture, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            person.person_id, person.name, person.first_seen,
            person.last_seen, person.total_visits, person.total_duration_s,
            person.best_clarity, person.best_capture_path,
            json.dumps(person.metadata),
        ))
        conn.commit()
        conn.close()

    def _store_embedding(self, person_id: str, embedding: np.ndarray, clarity: float):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO embeddings (person_id, embedding, clarity, created_at) VALUES (?,?,?,?)",
            (person_id, embedding.tobytes(), clarity, time.time()),
        )
        conn.commit()
        conn.close()

    def _store_capture(self, person_id: str, path: str, clarity: float, is_best: bool):
        conn = self._get_conn()
        if is_best:
            conn.execute(
                "UPDATE captures SET is_best=0 WHERE person_id=?", (person_id,)
            )
        conn.execute(
            "INSERT INTO captures (person_id, path, clarity, is_best, created_at) VALUES (?,?,?,?,?)",
            (person_id, path, clarity, int(is_best), time.time()),
        )
        conn.commit()
        conn.close()

    def _save_face_image(
        self, person_id: str, face_img: np.ndarray, tag: str = ""
    ) -> str:
        """Save a face crop to disk.  Returns the file path."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"_{tag}" if tag else ""
        filename = f"{person_id}{suffix}_{ts}.jpg"
        path = os.path.join(self._images_dir, filename)
        cv2.imwrite(path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    def _save_raw_capture(
        self, person_id: str, raw_crop: np.ndarray
    ) -> str:
        """Save the raw (un-resized) face crop for archival."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{person_id}_raw_{ts}.jpg"
        path = os.path.join(self._raw_dir, filename)
        cv2.imwrite(path, raw_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    def _add_event(self, event_type: str, data: dict):
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "timestamp_str": datetime.now().strftime("%H:%M:%S"),
            **data,
        }
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    # ── matching ─────────────────────────────────────────────────────

    def match_embedding(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Find the closest person by Euclidean distance.

        Returns (person_id, distance) if under threshold, else (None, best_dist).
        """
        best_pid = None
        best_dist = float("inf")

        for pid, person in self._persons.items():
            avg = person.avg_embedding
            if avg is None:
                continue
            dist = float(np.linalg.norm(embedding - avg))
            if dist < best_dist:
                best_dist = dist
                best_pid = pid

        if best_dist <= self.DISTANCE_THRESHOLD:
            return best_pid, best_dist
        return None, best_dist

    # ── main public API ──────────────────────────────────────────────

    def register_or_match(
        self,
        embedding: np.ndarray,
        face_crop: np.ndarray,
        raw_person_crop: np.ndarray,
        clarity: float,
        track_id: int,
    ) -> Tuple[str, bool, float, bool]:
        """
        Register a new person **or** match + update an existing one.

        The clarity-upgrade logic:
          - first detection  → store as best capture
          - later detections → compare clarity with current best
          - if new clarity > best + MARGIN → upgrade best capture

        Args:
            embedding:        128-d face encoding.
            face_crop:        The face-only crop (resized thumbnail).
            raw_person_crop:  The raw person crop before any resizing.
            clarity:          Laplacian-variance sharpness score.
            track_id:         ByteTrack track id for this person.

        Returns:
            (person_id, is_new, distance_or_sim, capture_upgraded)
        """
        with self._lock:
            now = time.time()

            # Cooldown — don't spam the DB for the same track
            last_cap = self._last_capture_time.get(track_id, 0)
            if now - last_cap < self.CAPTURE_COOLDOWN_S:
                session = self._active_sessions.get(track_id)
                if session:
                    return session.person_id, False, 0.0, False

            matched_pid, distance = self.match_embedding(embedding)
            capture_upgraded = False

            if matched_pid is not None:
                # ── KNOWN PERSON ──
                person = self._persons[matched_pid]
                person.last_seen = now

                # Always store the raw capture
                raw_path = self._save_raw_capture(matched_pid, raw_person_crop)

                # Add embedding (cap at MAX)
                if len(person.embeddings) < self.MAX_EMBEDDINGS_PER_PERSON:
                    person.embeddings.append(embedding.copy())
                    self._store_embedding(matched_pid, embedding, clarity)

                # ── Clarity comparison: is this capture better? ──
                if clarity > person.best_clarity + self.CLARITY_UPGRADE_MARGIN:
                    # New capture is sharper → upgrade best
                    thumb_path = self._save_face_image(matched_pid, face_crop, "best")
                    person.best_capture_path = thumb_path
                    person.best_clarity = clarity
                    capture_upgraded = True
                    self._store_capture(matched_pid, thumb_path, clarity, is_best=True)
                    logger.info(
                        "⬆ Upgraded best capture for %s — clarity %.1f → %.1f",
                        person.name, person.best_clarity - self.CLARITY_UPGRADE_MARGIN,
                        clarity,
                    )
                else:
                    # Still save as a regular capture (not best)
                    thumb_path = self._save_face_image(matched_pid, face_crop)
                    self._store_capture(matched_pid, thumb_path, clarity, is_best=False)

                person.all_capture_paths.append(thumb_path)

                # Session management
                session = self._active_sessions.get(track_id)
                if session is None or session.person_id != matched_pid:
                    person.total_visits += 1
                    self._active_sessions[track_id] = TrackSession(
                        session_id=str(uuid.uuid4())[:8],
                        person_id=matched_pid,
                        track_id=track_id,
                        started=now,
                        last_update=now,
                        face_captured=True,
                        best_clarity_this_session=clarity,
                    )
                    self._add_event("person_returned", {
                        "person_id": matched_pid,
                        "name": person.name,
                        "distance": round(distance, 3),
                        "total_visits": person.total_visits,
                        "track_id": track_id,
                        "clarity": round(clarity, 1),
                        "upgraded": capture_upgraded,
                    })
                    logger.info(
                        "Recognized %s (dist=%.3f, visits=%d, clarity=%.1f)",
                        person.name, distance, person.total_visits, clarity,
                    )
                else:
                    session.last_update = now
                    if clarity > session.best_clarity_this_session:
                        session.best_clarity_this_session = clarity

                self._save_person_meta(person)
                self._last_capture_time[track_id] = now
                return matched_pid, False, distance, capture_upgraded

            else:
                # ── NEW PERSON ──
                self._person_counter += 1
                pid = str(uuid.uuid4())[:12]
                name = f"Person-{self._person_counter:04d}"

                # Save both raw and thumbnail
                thumb_path = self._save_face_image(pid, face_crop, "best")
                raw_path = self._save_raw_capture(pid, raw_person_crop)

                person = PersonRecord(
                    person_id=pid,
                    name=name,
                    first_seen=now,
                    last_seen=now,
                    total_visits=1,
                    total_duration_s=0.0,
                    best_clarity=clarity,
                    best_capture_path=thumb_path,
                    embeddings=[embedding.copy()],
                    all_capture_paths=[thumb_path],
                )
                self._persons[pid] = person
                self._save_person_meta(person)
                self._store_embedding(pid, embedding, clarity)
                self._store_capture(pid, thumb_path, clarity, is_best=True)

                self._active_sessions[track_id] = TrackSession(
                    session_id=str(uuid.uuid4())[:8],
                    person_id=pid,
                    track_id=track_id,
                    started=now,
                    last_update=now,
                    face_captured=True,
                    best_clarity_this_session=clarity,
                )
                self._last_capture_time[track_id] = now

                self._add_event("new_person", {
                    "person_id": pid,
                    "name": name,
                    "track_id": track_id,
                    "clarity": round(clarity, 1),
                })
                logger.info(
                    "New person registered: %s (track %d, clarity=%.1f)",
                    name, track_id, clarity,
                )
                return pid, True, 0.0, False

    def try_upgrade_capture(
        self,
        person_id: str,
        face_crop: np.ndarray,
        raw_crop: np.ndarray,
        clarity: float,
        embedding: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Called on *every* subsequent face detection for a known track.
        Compares clarity and upgrades the best capture if sharper.

        This is the core of "keep the best face" logic — the pipeline
        calls this every frame where a face is found, even after the
        initial register_or_match.

        Returns True if the best capture was upgraded.
        """
        with self._lock:
            person = self._persons.get(person_id)
            if person is None:
                return False

            person.last_seen = time.time()

            # Always save the raw capture
            self._save_raw_capture(person_id, raw_crop)

            # Add embedding if provided and we haven't maxed out
            if embedding is not None and len(person.embeddings) < self.MAX_EMBEDDINGS_PER_PERSON:
                person.embeddings.append(embedding.copy())
                self._store_embedding(person_id, embedding, clarity)

            if clarity > person.best_clarity + self.CLARITY_UPGRADE_MARGIN:
                # Sharper! Upgrade.
                thumb_path = self._save_face_image(person_id, face_crop, "best")
                person.best_capture_path = thumb_path
                person.best_clarity = clarity
                person.all_capture_paths.append(thumb_path)
                self._store_capture(person_id, thumb_path, clarity, is_best=True)
                self._save_person_meta(person)

                self._add_event("capture_upgraded", {
                    "person_id": person_id,
                    "name": person.name,
                    "clarity": round(clarity, 1),
                })
                logger.info(
                    "⬆ Best capture upgraded for %s — clarity now %.1f",
                    person.name, clarity,
                )
                return True

            return False

    # ── session management ───────────────────────────────────────────

    def update_session_duration(self, track_id: int):
        session = self._active_sessions.get(track_id)
        if session is None:
            return
        now = time.time()
        session.last_update = now
        person = self._persons.get(session.person_id)
        if person:
            person.total_duration_s += 0.033
            person.last_seen = now

    def end_session(self, track_id: int):
        with self._lock:
            session = self._active_sessions.pop(track_id, None)
            if session:
                person = self._persons.get(session.person_id)
                if person:
                    self._save_person_meta(person)
                    self._add_event("person_left", {
                        "person_id": session.person_id,
                        "name": person.name,
                        "duration_s": round(time.time() - session.started, 1),
                        "track_id": track_id,
                        "best_clarity": round(session.best_clarity_this_session, 1),
                    })

    def get_session_for_track(self, track_id: int) -> Optional[TrackSession]:
        return self._active_sessions.get(track_id)

    # ── query API ────────────────────────────────────────────────────

    def get_all_persons(self) -> List[dict]:
        with self._lock:
            return [
                p.to_api_dict()
                for p in sorted(
                    self._persons.values(),
                    key=lambda p: p.last_seen,
                    reverse=True,
                )
            ]

    def get_person(self, person_id: str) -> Optional[dict]:
        with self._lock:
            person = self._persons.get(person_id)
            return person.to_api_dict() if person else None

    def get_person_thumbnails(self, person_id: str) -> List[str]:
        with self._lock:
            person = self._persons.get(person_id)
            return list(person.all_capture_paths) if person else []

    def get_best_capture_path(self, person_id: str) -> Optional[str]:
        with self._lock:
            person = self._persons.get(person_id)
            return person.best_capture_path if person else None

    def get_active_sessions(self) -> List[dict]:
        with self._lock:
            results = []
            for tid, session in self._active_sessions.items():
                person = self._persons.get(session.person_id)
                results.append({
                    "track_id": tid,
                    "person_id": session.person_id,
                    "person_name": person.name if person else "Unknown",
                    "session_started": session.started,
                    "duration_s": round(time.time() - session.started, 1),
                    "best_clarity": round(session.best_clarity_this_session, 1),
                })
            return results

    def get_recent_events(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(self._events[-limit:])

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_persons": len(self._persons),
                "active_sessions": len(self._active_sessions),
                "total_events": len(self._events),
            }

    def rename_person(self, person_id: str, new_name: str) -> bool:
        with self._lock:
            person = self._persons.get(person_id)
            if person:
                person.name = new_name
                self._save_person_meta(person)
                return True
            return False
