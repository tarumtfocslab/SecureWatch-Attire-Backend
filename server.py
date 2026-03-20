# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, PlainTextResponse, JSONResponse
from collections import deque
from pathlib import Path
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional, Set, Tuple, List
import numpy as np
import threading, time, queue
import cv2, json
cv2.setNumThreads(1)
import os, uuid, shutil, sys, traceback, csv, io, asyncio
import base64, hashlib, hmac, secrets
from fisheye_multiview_dewarp import (
    is_fisheye,
    build_fisheye_remap,
    CURRENT_VIEW_CONFIGS,
    FisheyeMultiViewDewarper,
    OUTPUT_SHAPE,
    INPUT_FOV_DEG,
    VIEW_CONFIGS,
)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://qy248.github.io",
]

app = FastAPI()

HERE = Path(__file__).resolve().parent
UPLOAD_DIR = str(HERE / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def force_cors_headers(request: Request, call_next):
    origin = request.headers.get("origin", "")
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Vary"] = "Origin"

    return response


@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    origin = request.headers.get("origin", "")
    headers = {}

    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "*"
        headers["Access-Control-Allow-Headers"] = "*"
        headers["Vary"] = "Origin"

    return Response(status_code=204, headers=headers)

@app.get("/health")
def render_health():
    return {"status": "ok"}

ENABLE_HEAVY_PIPELINE = os.getenv("ENABLE_HEAVY_PIPELINE", "false").lower() == "true"

# ----------------------------
# Constants / Defaults
# ----------------------------
MAX_LIVE_SESSIONS = 4
STREAM_FPS = 0.0 # 0 = AUTO/native fps
DETECT_FPS = 2.0

LIVE_VIOLATION_CLASSES = {"sleeveless", "shorts", "slippers"}
LIVE_VIOLATION_CONF = 0.35
LIVE_PERSIST_FRAMES = 3
LIVE_COOLDOWN_SEC = 10.0

EVIDENCE_PAD_RATIO = 0.20
EVIDENCE_PAD_MIN = 20
EVIDENCE_VIEW_SHAPE = (1080, 1920)

GLOBAL_PERSON_DETECTOR = None
# ----------------------------
# Lightweight event-time tracking (person_id)
# ----------------------------
TRACK_TTL_SEC = 12.0          # track expires if not seen for 12s
TRACK_MATCH_IOU = 0.25        # bbox IoU required to reuse a track_id
MAX_TRACKS_PER_VIEW = 60

TRACKS_LOCK = threading.Lock()
TRACKS_BY_VIEW = {}  # (source_id, view_name) -> {track_id: {"bbox":[x1,y1,x2,y2], "ts": float}}
# ----------------------------
# Persistent Stores (JSON)
# ----------------------------
# --- RTSP persistent store ---
RTSP_PATH = str(HERE / "attire_rtsp.json")
RTSP_LOCK = threading.Lock()
# rtsp_id -> {"name": str, "url": str}
RTSP_BY_ID = {
  "rtsp-1": {"name": "Gate A - Main", "url": "rtsp://..."},
  "rtsp-2": {"name": "Corridor B", "url": "rtsp://..."}
}

def _load_rtsp_file() -> dict:
    try:
        if os.path.exists(RTSP_PATH):
            with open(RTSP_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_rtsp_file(data: dict) -> None:
    try:
        tmp = RTSP_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, RTSP_PATH)
    except Exception:
        pass

with RTSP_LOCK:
    RTSP_BY_ID = _load_rtsp_file()

def _get_rtsp(rtsp_id: str) -> dict:
    with RTSP_LOCK:
        cfg = RTSP_BY_ID.get(rtsp_id) or {}
    if not isinstance(cfg, dict):
        return {}
    return cfg

# --- Dewarp persistent store ---
DEWARP_PATH = str(HERE / "attire_dewarp.json")
DEWARP_LOCK = threading.Lock()
DEWARP_BY_VIDEO = {}  # video_id -> {"views":[{name, roll_deg, pitch_deg, fov_deg},...], "ver": int}
DEWARP_PREVIEW_BY_VIDEO = {} # unsaved live preview (used while tuning)

DEFAULT_DEWARP_VIEWS = [
    {"name": "entrance",    "label": "entrance",    "roll_deg": -105, "pitch_deg": -70, "fov_deg": 40},
    {"name": "corridor",    "label": "corridor",    "roll_deg": -100, "pitch_deg": -55, "fov_deg": 70},
    {"name": "left_seats",  "label": "left_seats",  "roll_deg":  180, "pitch_deg": -55, "fov_deg": 80},
    {"name": "right_seats", "label": "right_seats", "roll_deg":  160, "pitch_deg":  45, "fov_deg": 80},
]

def _load_dewarp_file() -> dict:
    try:
        if os.path.exists(DEWARP_PATH):
            with open(DEWARP_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_dewarp_file(data: dict) -> None:
    try:
        tmp = DEWARP_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, DEWARP_PATH)
    except Exception:
        pass

with DEWARP_LOCK:
    DEWARP_BY_VIDEO = _load_dewarp_file()

# --- FPS persistent store ---
FPS_PATH = str(HERE / "attire_fps.json")
FPS_LOCK = threading.Lock()
FPS_BY_VIDEO = {}  # video_id -> {"stream_fps": float, "detect_fps": float}

def _load_fps_file() -> dict:
    try:
        if os.path.exists(FPS_PATH):
            with open(FPS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_fps_file(data: dict) -> None:
    try:
        tmp = FPS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, FPS_PATH)
    except Exception:
        pass

with FPS_LOCK:
    FPS_BY_VIDEO = _load_fps_file()

def _clamp_stream_fps(v: float) -> float:
    # allow 0 = AUTO/native
    try:
        v = float(v)
    except Exception:
        return 0.0
    return max(0.0, min(30.0, v))

def _clamp_detect_fps(v: float) -> float:
    try:
        v = float(v)
    except Exception:
        return DETECT_FPS
    return max(0.5, min(10.0, v))


def _get_fps_for_video(video_id: str) -> Tuple[float, float]:
    # defaults if never saved
    stream_fps = STREAM_FPS
    detect_fps = DETECT_FPS
    with FPS_LOCK:
        cfg = FPS_BY_VIDEO.get(video_id) or {}
    if isinstance(cfg, dict):
        if cfg.get("stream_fps") is not None:
            stream_fps = _clamp_stream_fps(cfg["stream_fps"])
        if cfg.get("detect_fps") is not None:
            detect_fps = _clamp_detect_fps(cfg["detect_fps"])
    return stream_fps, detect_fps

# --- ROI persistent store ---
ROI_PATH = str(HERE / "attire_rois.json")
ROI_LOCK = threading.Lock()
ROI_BY_VIDEO = {}

def _load_roi_file():
    global ROI_BY_VIDEO
    if not os.path.exists(ROI_PATH):
        ROI_BY_VIDEO = {}
        return
    try:
        with open(ROI_PATH, "r", encoding="utf-8") as f:
            ROI_BY_VIDEO = json.load(f) or {}
    except Exception:
        ROI_BY_VIDEO = {}

def _save_roi_file():
    os.makedirs(os.path.dirname(ROI_PATH), exist_ok=True)
    with open(ROI_PATH, "w", encoding="utf-8") as f:
        json.dump(ROI_BY_VIDEO, f, indent=2)

_load_roi_file()

# --- Schedule persistent store ---
SCHEDULE_PATH = str(HERE / "attire_schedule.json")
SCHEDULE_LOCK = threading.Lock()
SCHEDULE_BY_VIDEO = {}  # video_id -> {"enabled": bool, "schedules": [ ... ]}

def _load_schedule_file() -> dict:
    try:
        if os.path.exists(SCHEDULE_PATH):
            with open(SCHEDULE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_schedule_file(data: dict) -> None:
    try:
        tmp = SCHEDULE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, SCHEDULE_PATH)
    except Exception:
        pass

with SCHEDULE_LOCK:
    SCHEDULE_BY_VIDEO = _load_schedule_file()

_DAY_TO_IDX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

def _parse_hhmm(s: str):
    # "08:00" -> (8,0)
    try:
        hh, mm = (s or "").split(":")
        return int(hh), int(mm)
    except Exception:
        return None

def _is_now_in_schedule(cfg: dict, now_dt: datetime) -> bool:
    """
    Returns True if:
      - schedule-based disabled => True (always detect)
      - enabled and now matches any enabled schedule (day + time window)
    Handles overnight windows: e.g. 22:00 -> 06:00.
    """
    if not isinstance(cfg, dict):
        return True

    if not cfg.get("enabled", False):
        return True

    schedules = cfg.get("schedules", [])
    if not isinstance(schedules, list) or not schedules:
        return True  # enabled but empty => treat as always on

    now_day = now_dt.weekday()  # Mon=0..Sun=6
    now_min = now_dt.hour * 60 + now_dt.minute

    for sch in schedules:
        if not isinstance(sch, dict):
            continue
        if not sch.get("enabled", True):
            continue

        days = sch.get("days", [])
        if not isinstance(days, list) or not days:
            continue

        day_idxs = [ _DAY_TO_IDX.get(d) for d in days ]
        day_idxs = [ d for d in day_idxs if d is not None ]
        if not day_idxs:
            continue

        st = _parse_hhmm(sch.get("startTime", ""))
        et = _parse_hhmm(sch.get("endTime", ""))
        if st is None or et is None:
            continue

        st_min = st[0] * 60 + st[1]
        et_min = et[0] * 60 + et[1]

        if st_min == et_min:
            continue  # treat as invalid window

        if st_min < et_min:
            # same-day window
            if (now_day in day_idxs) and (st_min <= now_min < et_min):
                return True
        else:
            # overnight window (e.g. 22:00 -> 06:00)
            # active if:
            #  - on day D after st_min
            #  - OR on next day after midnight before et_min
            prev_day = (now_day - 1) % 7
            if (now_day in day_idxs) and (now_min >= st_min):
                return True
            if (prev_day in day_idxs) and (now_min < et_min):
                return True

    return False

def _get_schedule_for_video(video_id: str) -> dict:
    with SCHEDULE_LOCK:
        cfg = SCHEDULE_BY_VIDEO.get(video_id)
    if isinstance(cfg, dict):
        return cfg
    return {"enabled": False, "schedules": []}

# --- Sources (Camera Feed Status) persistent store ---
SOURCES_PATH = str(HERE / "attire_sources.json")
SOURCES_LOCK = threading.Lock()
SOURCES_BY_VIDEO = {}  # video_id -> {"enabled": bool}

def _load_sources_file() -> dict:
    try:
        if os.path.exists(SOURCES_PATH):
            with open(SOURCES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_sources_file(data: dict) -> None:
    try:
        tmp = SOURCES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, SOURCES_PATH)
    except Exception:
        pass

with SOURCES_LOCK:
    SOURCES_BY_VIDEO = _load_sources_file()

def _get_enabled_for_video(video_id: str) -> bool:
    # default ON if never saved
    with SOURCES_LOCK:
        cfg = SOURCES_BY_VIDEO.get(video_id) or {}
    if isinstance(cfg, dict) and cfg.get("enabled") is not None:
        return bool(cfg.get("enabled"))
    return True

# --- Violation Types persistent store (global) ---
VIOLATION_TYPES_PATH = str(HERE / "attire_violation_types.json")
VIOLATION_TYPES_LOCK = threading.Lock()
VIOLATION_TYPES_CFG = {}  # {"enabled": {"sleeveless": true, "shorts": true, "slippers": true}}

def _load_violation_types_file() -> dict:
    try:
        if os.path.exists(VIOLATION_TYPES_PATH):
            with open(VIOLATION_TYPES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_violation_types_file(data: dict) -> None:
    try:
        tmp = VIOLATION_TYPES_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, VIOLATION_TYPES_PATH)
    except Exception:
        pass

with VIOLATION_TYPES_LOCK:
    VIOLATION_TYPES_CFG = _load_violation_types_file()

def _get_enabled_violation_map() -> dict:
    # default ON for all known classes
    with VIOLATION_TYPES_LOCK:
        enabled = (VIOLATION_TYPES_CFG.get("enabled") or {})
    out = {k: True for k in LIVE_VIOLATION_CLASSES}
    if isinstance(enabled, dict):
        for k in list(out.keys()):
            if k in enabled:
                out[k] = bool(enabled[k])
    return out

def _set_enabled_violation_map(enabled: dict) -> dict:
    if not isinstance(enabled, dict):
        raise HTTPException(status_code=400, detail="enabled must be an object")

    cleaned = {k: True for k in LIVE_VIOLATION_CLASSES}
    for k in cleaned.keys():
        if k in enabled:
            cleaned[k] = bool(enabled[k])

    with VIOLATION_TYPES_LOCK:
        VIOLATION_TYPES_CFG["enabled"] = cleaned
        _save_violation_types_file(VIOLATION_TYPES_CFG)

    return cleaned

# ----------------------------
# Persistent Store: Attire Events (JSON) 
# ----------------------------
ATTIRE_EVENTS_PATH = str(HERE / "attire_events.json")
ATTIRE_EVENTS_LOCK = threading.Lock()

def _load_attire_events_file() -> list:
    try:
        if os.path.exists(ATTIRE_EVENTS_PATH):
            with open(ATTIRE_EVENTS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []

def _save_attire_events_file(items: list) -> None:
    try:
        tmp = ATTIRE_EVENTS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp, ATTIRE_EVENTS_PATH)
    except Exception:
        pass

# load once at startup (keeps same ATTIRE_EVENTS variable you already use)
with ATTIRE_EVENTS_LOCK:
    ATTIRE_EVENTS = _load_attire_events_file()

# ----------------------------
# Universal Live Event Writer
# ----------------------------
LIVE_EVENT_STATE_LOCK = threading.Lock()
LIVE_EVENT_STATE = {}  # key -> {"count": int, "last_ts": int}

def _live_event_key(source_id: str, view: str, label: str, track_id=None):
    # RTSP future: if you have tracker, include track_id for best dedupe
    if track_id is not None:
        return f"{source_id}|{view}|{label}|trk:{track_id}"
    return f"{source_id}|{view}|{label}"

def _write_attire_event_common(
    *,
    source_id: str,         # "webcam" / "vid-xxxx" / "rtsp-xxx"
    source_name: str,       # display name for UI
    view_name: str,         # "normal"/"entrance"/...
    label: str,             # "sleeveless"/"shorts"/"slippers"
    conf: float,
    frame_bgr,
    bbox_xyxy,              # [x1,y1,x2,y2] in PIXELS (relative to frame_bgr)
    track_id=None,
    source_type: str,       # "Live Detection" or "Uploaded Video"
    evidence_kind: str,     # "live" or "offline" or "rtsp"
    id_prefix: str,         # "live"/"offline"/"rtsp"
):
    now_s = int(time.time())
    key = _live_event_key(source_id, view_name or "normal", label, track_id)

    with LIVE_EVENT_STATE_LOCK:
        st = LIVE_EVENT_STATE.get(key) or {"count": 0, "last_ts": 0}

        # cooldown
        if (now_s - int(st["last_ts"])) < int(LIVE_COOLDOWN_SEC):
            st["count"] = 0
            LIVE_EVENT_STATE[key] = st
            return None

        # persistence frames
        st["count"] = int(st["count"]) + 1
        if st["count"] < int(LIVE_PERSIST_FRAMES):
            LIVE_EVENT_STATE[key] = st
            return None

        # trigger
        st["count"] = 0
        st["last_ts"] = now_s
        LIVE_EVENT_STATE[key] = st

    filename = f"{now_s}_{uuid.uuid4().hex[:8]}.jpg"
    out_path = os.path.join(VIOLATIONS_DIR, evidence_kind, source_id, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        _save_crop_evidence_whole_person(frame_bgr, bbox_xyxy, label, out_path)
    except Exception:
        cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    evidence_url = f"/violations/{evidence_kind}/{source_id}/{filename}"

    new_event = {
        "id": f"{id_prefix}-{source_id}-{uuid.uuid4().hex[:8]}",
        "video_id": source_id,
        "video_name": source_name,
        "label": label,
        "view": view_name or "normal",
        "ts": now_s,
        "conf": float(conf) if conf is not None else None,
        "severity": "High",
        "evidence_url": evidence_url,
        "status": "Pending",
        "resolved_ts": None,
        "location": view_name or "normal",
        "notes": "",
        "source": source_type,
        "person_id": track_id,
    }

    with ATTIRE_EVENTS_LOCK:
        ATTIRE_EVENTS.append(new_event)
        if len(ATTIRE_EVENTS) > 5000:
            ATTIRE_EVENTS[:] = ATTIRE_EVENTS[-5000:]
        _save_attire_events_file(ATTIRE_EVENTS)
        # Publish notification (rate-limited)
        try:
            sid = new_event.get("video_id") or "unknown"
            vtype = new_event.get("label") or "unknown"
            print("[NOTIF] event created:",
                "id=", new_event.get("id"),
                "video_id=", new_event.get("video_id"),
                "label=", new_event.get("label"),
                "status=", new_event.get("status"))

            ok_notif = _should_publish_notif(sid, vtype)

            print("[NOTIF] should_publish =",
                ok_notif,
                "sid=", sid,
                "type=", vtype)

            if ok_notif:
                with ATTIRE_NOTIF_SUBS_LOCK:
                    sub_count = len(ATTIRE_NOTIF_SUBS)

                print("[NOTIF] publishing to subscribers:", sub_count)

                payload = {
                    "id": new_event["id"],
                    "source_id": sid,
                    "source_name": source_name,
                    "violation_type": vtype,
                    "timestamp": new_event["ts"],
                    "event_id": new_event["id"],
                }

                print("[NOTIF] payload:", payload)

                _publish_attire_notification(payload)
            else:
                print("[NOTIF] notification suppressed",
                    "sid=", sid,
                    "type=", vtype)
        except Exception:
            pass

    return new_event

def _write_attire_event_live(
    *,
    source_id: str,
    source_name: str,
    view_name: str,
    label: str,
    conf: float,
    frame_bgr,
    bbox_xyxy,
    track_id=None,
):
    return _write_attire_event_common(
        source_id=source_id,
        source_name=source_name,
        view_name=view_name,
        label=label,
        conf=conf,
        frame_bgr=frame_bgr,
        bbox_xyxy=bbox_xyxy,
        track_id=track_id,
        source_type="Live Detection",
        evidence_kind="live",
        id_prefix="live",
    )

def _write_attire_event_offline(
    *,
    video_id: str,
    video_name: str,
    view_name: str,
    label: str,
    conf: float,
    frame_bgr,
    bbox_xyxy,
    track_id=None,
):
    return _write_attire_event_common(
        source_id=video_id,
        source_name=video_name,
        view_name=view_name,
        label=label,
        conf=conf,
        frame_bgr=frame_bgr,
        bbox_xyxy=bbox_xyxy,
        track_id=track_id,
        source_type="Uploaded Video",
        evidence_kind="offline",
        id_prefix="offline",
    )

# ----------------------------
# Persistent Store: Video Display Names
# ----------------------------
LABELS_PATH = str(HERE / "attire_video_labels.json")
LABELS_LOCK = threading.Lock()
LABELS_BY_VIDEO = {}  # video_id -> {"name": "B001G - Class 1"}

def _load_labels_file() -> dict:
    try:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_labels_file(data: dict) -> None:
    try:
        tmp = LABELS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, LABELS_PATH)
    except Exception:
        pass

with LABELS_LOCK:
    LABELS_BY_VIDEO = _load_labels_file()

def _get_video_display_name(video_id: str, original_name: str = "", fallback: str = "") -> str:
    # default name from filename / fallback
    name = original_name or fallback or ""

    # 1) RTSP name (attire_rtsp.json) should override everything
    with RTSP_LOCK:
        r = (RTSP_BY_ID or {}).get(video_id) or {}
    if isinstance(r, dict):
        rn = (r.get("name") or "").strip()
        if rn:
            return rn

    # 2) Manual labels (attire_video_labels.json)
    with LABELS_LOCK:
        cfg = LABELS_BY_VIDEO.get(video_id) or {}
    if isinstance(cfg, dict):
        alias = (cfg.get("name") or "").strip()
        if alias:
            return alias

    return name or str(video_id)

def _get_view_display_label(video_id: str, view_name: str) -> str:
    v = (view_name or "normal").strip()

    if not video_id:
        return v

    try:
        with DEWARP_LOCK:
            cfg = DEWARP_BY_VIDEO.get(str(video_id)) or {}
        views = cfg.get("views") if isinstance(cfg, dict) else None

        if isinstance(views, list):
            for it in views:
                if not isinstance(it, dict):
                    continue
                if str(it.get("name") or "") == v:
                    lab = str(it.get("label") or "").strip()
                    return lab or v
    except Exception:
        pass

    return v

def _decorate_attire_event(e: dict) -> dict:
    """Return a copy of event with location formatted as: '<video_name>, <view>'."""
    out = dict(e)

    view_raw = (out.get("view") or out.get("location") or "normal")
    view_raw = str(view_raw)

    vid = out.get("video_id") or ""
    fallback_name = out.get("video_name") or ""

    # NEW: map to display label (from dewarp config)
    view_label = _get_view_display_label(str(vid), view_raw)

    # optional: keep both fields (useful for debugging)
    out["view_name"] = view_raw
    out["view"] = view_label

    vid = out.get("video_id") or ""
    fallback_name = out.get("video_name") or ""

    if vid:
        src_name = _get_video_display_name(str(vid), fallback_name)
    else:
        src_name = fallback_name or "Unknown Source"

    out["video_name"] = src_name
    out["location"] = f"{src_name}, {view_label}"
    return out

# ----------------------------
# Persistent Store: Users + Sessions (JSON)
# ----------------------------
USERS_PATH = str(HERE / "users.json")
USERS_LOCK = threading.Lock()
USERS = []  # list[dict]

SESSIONS_PATH = str(HERE / "sessions.json")
SESSIONS_LOCK = threading.Lock()
SESSIONS = {}  # token -> {"user_id": "...", "exp": unix_ts}

SESSION_TTL_SEC = 60 * 60 * 12  # 12 hours (adjust)
PASSWORD_ITERS = 120_000

def _load_json_file(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                x = json.load(f)
            return x if x is not None else default
    except Exception:
        pass
    return default

def _save_json_file(path: str, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass

def _ensure_usernames():
    changed = False

    taken = set()
    for u in USERS:
        un = (u.get("username") or "").strip().lower()
        if un:
            taken.add(un)

    for u in USERS:
        if (u.get("username") or "").strip():
            continue

        email = (u.get("email") or "").strip().lower()
        if email and "@" in email:
            base = email.split("@", 1)[0]
        else:
            base = "user"

        candidate = base
        i = 2
        while candidate.lower() in taken:
            candidate = f"{base}{i}"
            i += 1

        u["username"] = candidate
        taken.add(candidate.lower())
        changed = True

    if changed:
        _save_json_file(USERS_PATH, USERS)

def _now() -> int:
    return int(time.time())

def _uid(prefix="u-") -> str:
    return prefix + uuid.uuid4().hex[:8]

def _pbkdf2_hash(password: str, salt_b64: str = "") -> dict:
    salt = base64.b64decode(salt_b64) if salt_b64 else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERS)
    return {
        "salt": base64.b64encode(salt).decode("utf-8"),
        "hash": base64.b64encode(dk).decode("utf-8"),
        "iters": PASSWORD_ITERS,
    }

def _verify_password(password: str, salt_b64: str, hash_b64: str, iters: int) -> bool:
    try:
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False

def _sanitize_user(u: dict) -> dict:
    return {
        "id": u.get("id"),
        "username": u.get("username", ""), 
        "name": u.get("name", ""),
        "email": u.get("email", ""),
        "role": u.get("role", "Viewer"),
        "status": u.get("status", "Active"),
        "createdAt": u.get("createdAt"),
    }

def _ensure_default_users():
    if USERS:
        return

    defaults = [
        {
            "username": "admin",
            "name": "Admin User",
            "email": "admin@securewatch.local",
            "role": "Admin",
            "status": "Active",
            "password": "user1234",
        },
        {
            "username": "security",
            "name": "Security Team",
            "email": "security@securewatch.local",
            "role": "Security",
            "status": "Active",
            "password": "user1234",
        },
        {
            "username": "staff",
            "name": "Lab Staff",
            "email": "staff@securewatch.local",
            "role": "Staff",
            "status": "Disabled",
            "password": "user1234",
        },
    ]

    for d in defaults:
        pw = _pbkdf2_hash(d["password"])
        USERS.append({
            "id": _uid("u-"),
            "username": d["username"], 
            "name": d["name"],
            "email": d["email"],
            "role": d["role"],
            "status": d["status"],
            "createdAt": datetime.now().isoformat(),
            "pw_salt": pw["salt"],
            "pw_hash": pw["hash"],
            "pw_iters": pw["iters"],
        })

    _save_json_file(USERS_PATH, USERS)

with USERS_LOCK:
    USERS = _load_json_file(USERS_PATH, [])
    _ensure_usernames()
    _ensure_default_users()
with SESSIONS_LOCK:
    SESSIONS = _load_json_file(SESSIONS_PATH, {})

def _get_user_by_email(email: str):
    email = (email or "").strip().lower()
    for u in USERS:
        if (u.get("email") or "").lower() == email:
            return u
    return None

def _get_user_by_username(username: str):
    username = (username or "").strip().lower()
    for u in USERS:
        if (u.get("username") or "").strip().lower() == username:
            return u
    return None

def _get_user_by_id(uid_: str):
    for u in USERS:
        if str(u.get("id")) == str(uid_):
            return u
    return None

def _admin_count() -> int:
    return sum(1 for u in USERS if (u.get("role") == "Admin"))

def _cleanup_sessions():
    now = _now()
    dead = []
    for tok, s in (SESSIONS or {}).items():
        if int((s or {}).get("exp", 0)) <= now:
            dead.append(tok)
    for tok in dead:
        SESSIONS.pop(tok, None)
    if dead:
        _save_json_file(SESSIONS_PATH, SESSIONS)

def _issue_token(user_id: str) -> str:
    token = "t-" + uuid.uuid4().hex
    exp = _now() + int(SESSION_TTL_SEC)
    with SESSIONS_LOCK:
        _cleanup_sessions()
        SESSIONS[token] = {"user_id": user_id, "exp": exp}
        _save_json_file(SESSIONS_PATH, SESSIONS)
    return token

def _get_token_from_auth(authorization: str) -> str:
    # "Bearer <token>"
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    if len(parts) != 2:
        return ""
    if parts[0].lower() != "bearer":
        return ""
    return parts[1].strip()

def get_current_user(authorization: str = Header(default="")) -> dict:
    token = _get_token_from_auth(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    with SESSIONS_LOCK:
        _cleanup_sessions()
        sess = SESSIONS.get(token)

    if not sess:
        raise HTTPException(status_code=401, detail="Invalid token")

    if int(sess.get("exp", 0)) <= _now():
        raise HTTPException(status_code=401, detail="Token expired")

    uid_ = sess.get("user_id")
    with USERS_LOCK:
        u = _get_user_by_id(uid_)
        if not u:
            raise HTTPException(status_code=401, detail="User not found")
        if u.get("status") != "Active":
            raise HTTPException(status_code=403, detail="User is disabled")
        return u

def require_admin(u: dict = Depends(get_current_user)) -> dict:
    if u.get("role") != "Admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return u

# ----------------------------
# Persistent Store: Notifications
# ----------------------------
NOTIF_PATH = str(HERE / "attire_notifications.json")
NOTIF_LOCK = threading.Lock()

DEFAULT_NOTIF_CFG = {
    "enabled": True,
    "cooldown_sec": 30,   # prevent spam
    "toast_sec": 6,       # frontend can use this
    "play_sound": False,  # frontend can use this
}

ATTIRE_NOTIF_CFG = DEFAULT_NOTIF_CFG.copy()

def _load_notif_file() -> dict:
    try:
        if os.path.exists(NOTIF_PATH):
            with open(NOTIF_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _save_notif_file(data: dict) -> None:
    try:
        tmp = NOTIF_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, NOTIF_PATH)
    except Exception:
        pass

with NOTIF_LOCK:
    loaded = _load_notif_file()
    ATTIRE_NOTIF_CFG = {**DEFAULT_NOTIF_CFG, **loaded}

# --- Notification publish / SSE ---
ATTIRE_NOTIF_SUBS_LOCK = threading.Lock()
ATTIRE_NOTIF_SUBS = set()  # set of (loop, asyncio.Queue)

# per (source_id, violation_type) cooldown
ATTIRE_NOTIF_LAST_TS_LOCK = threading.Lock()
ATTIRE_NOTIF_LAST_TS = {}  # (source_id, violation_type) -> unix_ts

def _should_publish_notif(source_id: str, violation_type: str) -> bool:
    with NOTIF_LOCK:
        cfg = dict(ATTIRE_NOTIF_CFG)

    print("[NOTIF] _should_publish_notif called",
          "source_id=", source_id,
          "violation_type=", violation_type,
          "cfg=", cfg)

    if not cfg.get("enabled", True):
        print("[NOTIF] blocked: notifications disabled")
        return False

    cd = float(cfg.get("cooldown_sec", 30) or 0)
    key = (source_id or "unknown", violation_type or "unknown")
    now = time.time()

    with ATTIRE_NOTIF_LAST_TS_LOCK:
        last = ATTIRE_NOTIF_LAST_TS.get(key, 0.0)
        diff = now - last

        print("[NOTIF] cooldown check",
              "key=", key,
              "last=", last,
              "now=", now,
              "diff=", diff,
              "cooldown=", cd)

        if cd > 0 and diff < cd:
            print("[NOTIF] blocked by cooldown")
            return False

        ATTIRE_NOTIF_LAST_TS[key] = now

    print("[NOTIF] allowed")
    return True

def _publish_attire_notification(payload: dict) -> None:
    # Called from worker threads, so must notify asyncio loop thread-safely
    with ATTIRE_NOTIF_SUBS_LOCK:
        subs = list(ATTIRE_NOTIF_SUBS)

    for loop, q in subs:
        try:
            loop.call_soon_threadsafe(_safe_notif_put, q, payload)
        except Exception:
            pass

def _safe_notif_put(q: asyncio.Queue, payload: dict):
    try:
        q.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            _ = q.get_nowait()   # drop oldest one
        except Exception:
            pass
        try:
            q.put_nowait(payload)
        except Exception:
            pass
    except Exception:
        pass

# ----------------------------
# In-memory runtime stores
# ----------------------------
VIDEOS = {}    # video_id -> {"path": "...", "name": "..."}
LIVE_SESSIONS = {}  # video_id -> LiveVideoSession
LIVE_LOCK = threading.Lock()

THUMB_CACHE_LOCK = threading.Lock()
THUMB_CACHE = {}  # key -> {"ts": float, "jpg": bytes}
THUMB_TTL_SEC = 8.0

# ----------------------------
# Live session janitor (idle cleanup)
# ----------------------------
IDLE_TIMEOUT_SEC = 60          # stop sessions if no viewer for 60s
JANITOR_INTERVAL_SEC = 10      # check every 10s

def _janitor():
    while True:
        time.sleep(JANITOR_INTERVAL_SEC)
        now = time.time()
        dead = []

        with LIVE_LOCK:
            for vid, sess in list(LIVE_SESSIONS.items()):
                # If no viewer recently, stop and remove session
                if (now - float(getattr(sess, "last_access", 0))) > IDLE_TIMEOUT_SEC:
                    dead.append(vid)

            for vid in dead:
                s = LIVE_SESSIONS.pop(vid, None)
                if s:
                    try:
                        s.stop()
                    except Exception:
                        pass

# Start janitor once at import/startup
threading.Thread(target=_janitor, daemon=True).start()

GLOBAL_DETECTOR = None
# NOTE: ATTIRE_EVENTS is loaded from attire_events.json above.

VIOLATIONS_DIR = str(HERE / "violations")
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
# Serve saved evidence images (front-end can load via http://localhost:8000/violations/...)
app.mount("/violations", StaticFiles(directory=VIOLATIONS_DIR), name="violations")

# ----------------------------
# Helpers 
# ----------------------------
# --- Generic helpers ---
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter + 1e-6
    return inter / union

def _assign_light_track_id(source_id: str, view_name: str, person_bbox: List[float]) -> str:
    """
    Given a person bbox in pixels, return a stable-ish track_id within a short TTL window.
    This is NOT full tracking; it's short-term association to dedupe events.
    """
    now = time.time()
    key = (source_id or "unknown", view_name or "normal")

    with TRACKS_LOCK:
        tracks = TRACKS_BY_VIEW.get(key)
        if tracks is None:
            tracks = {}
            TRACKS_BY_VIEW[key] = tracks

        # cleanup old tracks
        dead = []
        for tid, info in tracks.items():
            if (now - float(info.get("ts", 0.0))) > float(TRACK_TTL_SEC):
                dead.append(tid)
        for tid in dead:
            tracks.pop(tid, None)

        # match existing track by IoU
        best_tid = None
        best_iou = 0.0
        for tid, info in tracks.items():
            iou = _box_iou(info["bbox"], person_bbox)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        if best_tid is not None and best_iou >= float(TRACK_MATCH_IOU):
            tracks[best_tid] = {"bbox": person_bbox, "ts": now}
            return best_tid

        # create new
        new_tid = uuid.uuid4().hex[:10]
        tracks[new_tid] = {"bbox": person_bbox, "ts": now}

        # keep bounded
        if len(tracks) > int(MAX_TRACKS_PER_VIEW):
            oldest = sorted(tracks.items(), key=lambda kv: float(kv[1].get("ts", 0.0)))[0][0]
            tracks.pop(oldest, None)

        return new_tid

def _xyxy_to_percent(x1, y1, x2, y2, w, h):
    w = max(1, int(w))
    h = max(1, int(h))
    x = (x1 / w) * 100.0
    y = (y1 / h) * 100.0
    ww = ((x2 - x1) / w) * 100.0
    hh = ((y2 - y1) / h) * 100.0
    return x, y, ww, hh

def _iter_boxes_from_raw(raw):
        """
        Yields (bbox_xyxy, label, conf) where bbox is [x1,y1,x2,y2] floats in PIXELS.
        Supports:
        1) Ultralytics Results or [Results]
        2) list[dict] style outputs from custom detectors
        3) dict with 'detections' list
        """
        if raw is None:
            return

        # -------------------------
        # Case A: Ultralytics Results or [Results]
        # -------------------------
        r0 = raw
        if isinstance(r0, list) and r0 and hasattr(r0[0], "boxes"):
            r0 = r0[0]

        if hasattr(r0, "boxes") and r0.boxes is not None and len(r0.boxes) > 0:
            names = getattr(r0, "names", {}) or {}
            xyxy = r0.boxes.xyxy.cpu().numpy()
            cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
            confs = r0.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs):
                label = names.get(int(cid), str(cid))
                yield [float(x1), float(y1), float(x2), float(y2)], str(label), float(cf)
            return

        # -------------------------
        # Case B: dict wrapper
        # -------------------------
        if isinstance(r0, dict):
            dets = r0.get("detections") or r0.get("dets") or r0.get("boxes") or []
            if isinstance(dets, list):
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    bbox = d.get("bbox") or d.get("xyxy")
                    if not bbox or len(bbox) != 4:
                        continue
                    label = d.get("label") or d.get("name") or d.get("class") or ""
                    conf = d.get("conf") if d.get("conf") is not None else d.get("confidence", 0.0)
                    yield [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], str(label), float(conf)
            return

        # -------------------------
        # Case C: list[dict]
        # -------------------------
        if isinstance(r0, list):
            for d in r0:
                if not isinstance(d, dict):
                    continue
                bbox = d.get("bbox") or d.get("xyxy")
                if not bbox or len(bbox) != 4:
                    continue
                label = d.get("label") or d.get("name") or d.get("class") or ""
                conf = d.get("conf") if d.get("conf") is not None else d.get("confidence", 0.0)
                yield [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])], str(label), float(conf)
            return
        
def _thumb_cache_get(key: str):
    now = time.time()
    with THUMB_CACHE_LOCK:
        item = THUMB_CACHE.get(key)
        if not item:
            return None
        if (now - float(item.get("ts", 0.0))) > THUMB_TTL_SEC:
            THUMB_CACHE.pop(key, None)
            return None
        return item.get("jpg")

def _thumb_cache_set(key: str, jpg: bytes):
    with THUMB_CACHE_LOCK:
        THUMB_CACHE[key] = {"ts": time.time(), "jpg": jpg}

# --- ROI helpers ---
def _bbox_inside_any_roi_percent(bbox_xyxy, roi_polys_percent, img_w, img_h, mode="center"):
    """
    bbox_xyxy: [x1,y1,x2,y2] in PIXELS (for the image that was detected on)
    roi_polys_percent: list of polygons; polygon = [[x%,y%],...]
    """
    if not roi_polys_percent:
        return True

    x1, y1, x2, y2 = bbox_xyxy
    if mode == "center":
        px = float((x1 + x2) / 2.0)
        py = float((y1 + y2) / 2.0)
    else:
        px = float((x1 + x2) / 2.0)
        py = float(y2)

    for poly in roi_polys_percent:
        if not poly or len(poly) < 3:
            continue
        # convert percent -> pixel poly
        pts = []
        for (xp, yp) in poly:
            pts.append([ (float(xp) / 100.0) * img_w, (float(yp) / 100.0) * img_h ])
        poly_np = np.array(pts, dtype=np.int32)
        if cv2.pointPolygonTest(poly_np, (px, py), False) >= 0:
            return True
    return False

# --- Mosaic / fisheye view helpers ---
def _center_crop_square(frame):
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    return frame[:, x0:x0+side], side

def _resize_keep(tile, target_wh):
    """Resize tile to exact (w,h)."""
    tw, th = target_wh
    return cv2.resize(tile, (tw, th), interpolation=cv2.INTER_LINEAR)

def _pick_4_views(views: dict):
    """
    Pick 4 views in a stable order using CURRENT_VIEW_CONFIGS first.
    This guarantees tile positions:
      0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right
    """
    if not isinstance(views, dict) or not views:
        return []

    ordered_names = [c.get("name") for c in CURRENT_VIEW_CONFIGS if c.get("name")]
    chosen = []

    for name in ordered_names:
        img = views.get(name)
        if img is not None:
            chosen.append((name, img))
        if len(chosen) == 4:
            break

    # fallback: any remaining keys
    if len(chosen) < 4:
        for k in sorted(views.keys()):
            if any(k == ck for ck, _ in chosen):
                continue
            img = views.get(k)
            if img is None:
                continue
            chosen.append((k, img))
            if len(chosen) == 4:
                break

    return chosen[:4]

def _make_2x2_mosaic(view_items, tile_size=(640, 360), labels_by_name=None):
    """
    view_items: list of (name, img) length 1..4
    Returns: mosaic_img, tile_meta
      tile_meta: list of dict {name, x0,y0,w,h} for mapping boxes
    """
    if not view_items:
        return None, []

    # ensure 4 tiles (pad with black if fewer)
    tiles = []
    for name, img in view_items:
        if img is None:
            continue
        tiles.append((name, img))

    # pad to 4
    while len(tiles) < 4:
        tiles.append((f"view_{len(tiles)}", None))

    tw, th = tile_size
    blank = (np.zeros((th, tw, 3), dtype=np.uint8))

    norm_tiles = []
    for name, img in tiles[:4]:
        if img is None:
            t = blank.copy()
        else:
            t = _resize_keep(img, (tw, th))
        # label of fisheye views
        label = name
        if isinstance(labels_by_name, dict):
            label = str(labels_by_name.get(name) or name)

        cv2.putText(t, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        norm_tiles.append((name, t))

    # arrange:
    # [0 1]
    # [2 3]
    top = cv2.hconcat([norm_tiles[0][1], norm_tiles[1][1]])
    bot = cv2.hconcat([norm_tiles[2][1], norm_tiles[3][1]])
    mosaic = cv2.vconcat([top, bot])

    tile_meta = [
        {"name": norm_tiles[0][0], "x0": 0,     "y0": 0,     "w": tw, "h": th},
        {"name": norm_tiles[1][0], "x0": tw,    "y0": 0,     "w": tw, "h": th},
        {"name": norm_tiles[2][0], "x0": 0,     "y0": th,    "w": tw, "h": th},
        {"name": norm_tiles[3][0], "x0": tw,    "y0": th,    "w": tw, "h": th},
    ]

    return mosaic, tile_meta

def _build_highres_planar_from_fisheye(full_frame, view_name, out_shape=EVIDENCE_VIEW_SHAPE):
    cfg = next((c for c in CURRENT_VIEW_CONFIGS if c.get("name") == view_name), None)
    if cfg is None:
        return None

    cropped, side = _center_crop_square(full_frame)
    out_h, out_w = out_shape

    map_x, map_y = build_fisheye_remap(
        input_shape=(side, side),
        output_shape=(out_h, out_w),
        input_fov_deg=INPUT_FOV_DEG,
        output_fov_deg=float(cfg.get("fov_deg", cfg.get("fov", 90.0))),
        yaw_deg=float(cfg.get("yaw_deg", cfg.get("yaw", 0.0))),
        pitch_deg=float(cfg.get("pitch_deg", cfg.get("pitch", 0.0))),
        roll_deg=float(cfg.get("roll_deg", cfg.get("roll", 0.0))),
    )
    planar = cv2.remap(cropped, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    if view_name == "right_seats":
        planar = cv2.rotate(planar, cv2.ROTATE_180)
    return planar

# --- Label + detection extraction helpers ---
def _label_to_violation(label: str):
    s = (label or "").lower()
    if "sleeveless" in s:
        return "sleeveless"
    if "shorts" in s:
        return "shorts"
    if s == "slippers" or s == "sandal" or s == "sandals":
        return "slippers"
    return None

def _extract_violation_boxes(raw):
    out = []
    for bbox, label, cf in _iter_boxes_from_raw(raw) or []:
        vio = _label_to_violation(label)
        if (vio in LIVE_VIOLATION_CLASSES) and (float(cf) >= LIVE_VIOLATION_CONF):
            out.append({
                "label": vio,
                "raw_label": label,
                "conf": float(cf),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            })
    return out

# --- Session helpers ---
def _touch_session(sess):
    sess.last_access = time.time()

def _ensure_live_slot(video_id: str):
    """
    Ensure we never run more than MAX_LIVE_SESSIONS sessions concurrently.
    If limit is reached, reject with 429 (no eviction).
    """
    if video_id in LIVE_SESSIONS:
        return

    if len(LIVE_SESSIONS) >= MAX_LIVE_SESSIONS:
        raise HTTPException(
            status_code=429,
            detail=f"Max live streams reached ({MAX_LIVE_SESSIONS}). Close a video first."
        )

def _get_uploaded_video_path(video_id: str) -> str:
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")
    return str(matches[0])

def _get_effective_dewarp_views(video_id: str):
    with DEWARP_LOCK:
        if video_id in DEWARP_PREVIEW_BY_VIDEO:
            return DEWARP_PREVIEW_BY_VIDEO[video_id]["views"]
        if video_id in DEWARP_BY_VIDEO:
            return DEWARP_BY_VIDEO[video_id]["views"]
    return [c.copy() for c in VIEW_CONFIGS]

def _validate_dewarp_views(views):
    if not isinstance(views, list) or len(views) != 4:
        raise HTTPException(status_code=400, detail="views must be a list of 4 view configs")

    allowed_names = {"entrance", "corridor", "left_seats", "right_seats"}
    seen = set()

    for v in views:
        if not isinstance(v, dict):
            raise HTTPException(status_code=400, detail="each view must be an object")

        name = v.get("name")
        if name not in allowed_names:
            raise HTTPException(status_code=400, detail=f"invalid view name: {name}")
        if name in seen:
            raise HTTPException(status_code=400, detail=f"duplicate view name: {name}")
        seen.add(name)

        if "label" in v and v["label"] is not None:
            if not isinstance(v["label"], (str, int, float)):
                raise HTTPException(status_code=400, detail="label must be a string")

        # ensure numeric
        float(v.get("roll_deg", 0))
        float(v.get("pitch_deg", 0))
        float(v.get("fov_deg", 90))

# --- Evidence helpers ---
def _crop_person_evidence(hi_view_bgr, vio_bbox_hi, scale=0.5, min_area=10_000, pad_ratio=0.08):
    """
    hi_view_bgr: high-res view image
    vio_bbox_hi: [x1,y1,x2,y2] in hi_view coords
    Returns: cropped image (person), or None if not found
    """
    if hi_view_bgr is None or hi_view_bgr.size == 0 or vio_bbox_hi is None:
        return None

    person_det = _get_person_detector()

    small = cv2.resize(hi_view_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    r = person_det.detect(small)
    if isinstance(r, list) and r and hasattr(r[0], "boxes"):
        r = r[0]

    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return None

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    best = None
    best_score = 0.0

    vx1, vy1, vx2, vy2 = vio_bbox_hi

    for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls_ids, confs):
        if cid != 0:  # COCO person
            continue

        # scale back to hi_view coords
        px1 = x1 / scale; py1 = y1 / scale
        px2 = x2 / scale; py2 = y2 / scale

        score = _box_iou([px1, py1, px2, py2], [vx1, vy1, vx2, vy2])
        if score > best_score:
            best_score = score
            best = [px1, py1, px2, py2]

    if best is None or best_score < 0.05:
        return None

    h, w = hi_view_bgr.shape[:2]
    px1, py1, px2, py2 = best
    bw = max(1, px2 - px1)
    bh = max(1, py2 - py1)
    pad = int(max(bw, bh) * pad_ratio)

    X1 = _clamp_int(px1 - pad, 0, w - 1)
    Y1 = _clamp_int(py1 - pad, 0, h - 1)
    X2 = _clamp_int(px2 + pad, 0, w - 1)
    Y2 = _clamp_int(py2 + pad, 0, h - 1)

    if X2 <= X1 or Y2 <= Y1:
        return None

    crop = hi_view_bgr[Y1:Y2, X1:X2]
    if crop.size == 0:
        return None
    if crop.shape[0] * crop.shape[1] < min_area:
        return None

    return crop

def _match_person_bbox_for_violation(hi_view_bgr, vio_bbox_hi, scale=0.5) -> Optional[List[float]]:
    """
    Return best matching PERSON bbox [x1,y1,x2,y2] in hi_view coords
    by IoU against vio_bbox_hi.
    Runs ONLY when an event is being written (or about to be written).
    """
    if hi_view_bgr is None or hi_view_bgr.size == 0 or vio_bbox_hi is None:
        return None

    person_det = _get_person_detector()

    small = cv2.resize(hi_view_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    r = person_det.detect(small)
    if isinstance(r, list) and r and hasattr(r[0], "boxes"):
        r = r[0]

    if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
        return None

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    vx1, vy1, vx2, vy2 = map(float, vio_bbox_hi)

    best = None
    best_iou = 0.0

    for (x1, y1, x2, y2), cid in zip(xyxy, cls_ids):
        if cid != 0:  # COCO person
            continue

        # scale back to hi_view coords
        px1 = float(x1 / scale); py1 = float(y1 / scale)
        px2 = float(x2 / scale); py2 = float(y2 / scale)

        iou = _box_iou([px1, py1, px2, py2], [vx1, vy1, vx2, vy2])
        if iou > best_iou:
            best_iou = iou
            best = [px1, py1, px2, py2]

    # weak match -> no track id
    if best is None or best_iou < 0.05:
        return None

    return best

def _save_crop_evidence_whole_person(img_bgr, bbox, label, out_path):
    """
    Save a clearer evidence crop that tries to include the whole person,
    based on where the violation usually appears.

    This runs ONLY when an event is fired, so it won't affect live performance much.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)

    # clamp bbox
    x1 = float(_clamp(x1, 0, w - 1)); x2 = float(_clamp(x2, 0, w - 1))
    y1 = float(_clamp(y1, 0, h - 1)); y2 = float(_clamp(y2, 0, h - 1))

    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)

    lab = (label or "").lower()

    # ---- label-aware expansion ----
    # slippers: bbox is near feet -> extend A LOT upward
    # shorts: bbox is mid-lower body -> extend upward a lot, slightly down
    # sleeveless: bbox is upper body -> extend down more
    if lab == "slippers":
        up_mul, down_mul, side_mul = 10.0, 1.5, 3.0
        anchor_bottom = True
    elif lab == "shorts":
        up_mul, down_mul, side_mul = 6.0, 2.0, 3.0
        anchor_bottom = False
    elif lab == "sleeveless":
        up_mul, down_mul, side_mul = 2.0, 6.0, 3.0
        anchor_bottom = False
    else:
        up_mul, down_mul, side_mul = 4.0, 3.0, 3.0
        anchor_bottom = False

    # build expanded crop
    cx = (x1 + x2) * 0.5
    if anchor_bottom:
        # keep feet near bottom of crop
        Y2 = y2 + bh * down_mul
        target_h = bh * (up_mul + down_mul)
        Y1 = Y2 - target_h
    else:
        Y1 = y1 - bh * up_mul
        Y2 = y2 + bh * down_mul

    X1 = cx - (bw * side_mul * 0.5)
    X2 = cx + (bw * side_mul * 0.5)

    # clamp crop
    X1 = int(_clamp(X1, 0, w - 1)); X2 = int(_clamp(X2, 0, w - 1))
    Y1 = int(_clamp(Y1, 0, h - 1)); Y2 = int(_clamp(Y2, 0, h - 1))

    if X2 <= X1 + 2 or Y2 <= Y1 + 2:
        crop = img_bgr
    else:
        crop = img_bgr[Y1:Y2, X1:X2]
        if crop is None or crop.size == 0:
            crop = img_bgr

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# --- Notifications helper (SSE) ---
def get_current_user_from_token(token: str):
    token = (token or "").strip()
    if not token:
        return None

    with SESSIONS_LOCK:
        _cleanup_sessions()
        sess = SESSIONS.get(token)

    if not sess or int((sess or {}).get("exp", 0)) <= _now():
        return None

    uid = (sess or {}).get("user_id")
    if not uid:
        return None

    with USERS_LOCK:
        u = _get_user_by_id(uid)
        if not u:
            return None
        if u.get("status") != "Active":
            return None
        return u

# ----------------------------
# Models / Detectors
# ----------------------------
def _find_model_path() -> str:
    env = os.getenv("ATTIRE_MODEL_PATH", "").strip()
    if env and Path(env).exists():
        return env

    candidates = [
        Path(HERE) / "models" / "attire_coco_best.pt",
        Path(HERE) / "models" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise FileNotFoundError(
        "Cannot find attire YOLO weights. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
        + "\nOr set ATTIRE_MODEL_PATH"
    )

def _get_detector():
    global GLOBAL_DETECTOR
    if GLOBAL_DETECTOR is not None:
        return GLOBAL_DETECTOR

    from detector import YoloDetector

    model_path = _find_model_path()

    GLOBAL_DETECTOR = YoloDetector(
        model_path=model_path,
        conf=0.30,
        iou=0.50,
    )
    return GLOBAL_DETECTOR

def _get_person_detector():
    global GLOBAL_PERSON_DETECTOR
    if GLOBAL_PERSON_DETECTOR is not None:
        return GLOBAL_PERSON_DETECTOR

    from detector import YoloDetector

    # NEW: load from attire_backend/models/yolov8n.pt
    person_model = Path(HERE) / "models" / "yolov8n.pt"
    if not person_model.exists():
        raise FileNotFoundError(f"Person model not found: {person_model}")

    GLOBAL_PERSON_DETECTOR = YoloDetector(
        model_path=str(person_model),
        conf=0.40,
        iou=0.50,
        imgsz=640,
    )
    return GLOBAL_PERSON_DETECTOR

def run_attire_inference_on_frame(frame_bgr):
    """
      ui_dets: list of Detection dicts for frontend overlay (PERCENT coords)
      vio_boxes: list of violation boxes for event logging (PIXEL coords)
        [{"label":"sleeveless","conf":0.88,"bbox":[x1,y1,x2,y2]}, ...]
    """
    detector = _get_detector()

    try:
        raw = detector.detect(frame_bgr)
    except Exception as e:
        print("[webcam detect] error:", e)
        return [], []

    h, w = frame_bgr.shape[:2]

    # 1) UI overlay dets (percent coords)
    ui_dets = []
    for i, (bbox, label, cf) in enumerate(_iter_boxes_from_raw(raw) or []):
        vx, vy, vw, vh = _xyxy_to_percent(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
        vio = _label_to_violation(label)

        ui_dets.append({
            "id": f"webcam-det-{i}",
            "x": vx, "y": vy, "width": vw, "height": vh,
            "label": label,
            "violation": vio,
            "conf": float(cf),
        })

    # 2) Event logging boxes (pixel coords, violations only)
    vio_boxes = _extract_violation_boxes(raw)

    # Respect your global violation toggles (sleeveless/shorts/slippers enabled)
    enabled_vios = _get_enabled_violation_map()
    vio_boxes = [vb for vb in vio_boxes if enabled_vios.get(vb["label"], True)]

    # keep only ONE box per label per frame (highest conf)
    best = {}
    for vb in vio_boxes:
        lab = vb["label"]
        if (lab not in best) or (vb["conf"] > best[lab]["conf"]):
            best[lab] = vb
    vio_boxes = list(best.values())

    return ui_dets, vio_boxes

def _webcam_detector_loop():
    global webcam_det_cache, _webcam_detect_running

    delay = 1.0 / max(0.5, WEBCAM_DETECT_FPS)

    while _webcam_detect_running:
        try:
            frame = webcam.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            t0 = time.time()
            ui_dets, vio_boxes = run_attire_inference_on_frame(frame)
            t1 = time.time()

            h, w = frame.shape[:2]
            fps = 1.0 / max(1e-6, (t1 - t0))

            with webcam_det_lock:
                webcam_det_cache = {
                    "ts": int(time.time() * 1000),
                    "fps": float(fps),
                    "resolution": [int(w), int(h)],
                    "detections": ui_dets or [],
                    "error": None,
                }
            source_id = "webcam"
            source_name = "Webcam"
            view_name = "normal"  # later RTSP can be entrance/corridor/etc

            for vb in (vio_boxes or []):
                try:
                    # 1) find person bbox that matches this violation bbox
                    person_bbox = _match_person_bbox_for_violation(frame, vb["bbox"], scale=0.5)

                    # 2) assign lightweight track id 
                    track_id = None
                    if person_bbox is not None:
                        track_id = _assign_light_track_id(source_id, view_name, person_bbox)

                    _write_attire_event_live(
                        source_id=source_id,
                        source_name=source_name,
                        view_name=view_name,
                        label=vb["label"],
                        conf=float(vb.get("conf", 0.0)),
                        frame_bgr=frame,
                        bbox_xyxy=vb["bbox"],
                        track_id=track_id,     # ✅ NOW USED
                    )
                except Exception as e:
                    print("[webcam event] failed:", repr(e))

            time.sleep(delay)

        except Exception as e:
            with webcam_det_lock:
                webcam_det_cache = {
                    "ts": int(time.time() * 1000),
                    "fps": 0.0,
                    "resolution": [0, 0],
                    "detections": [],
                    "error": repr(e),
                }
            time.sleep(0.5)

# ----------------------------
# --- Reports 
# ----------------------------
def _parse_yyyy_mm_dd(s: str):
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

def _label_title(label: str) -> str:
    m = {
        "sleeveless": "Sleeveless",
        "shorts": "Shorts",
        "slippers": "Slippers",
    }
    return m.get((label or "").lower(), (label or "").title() or "Unknown")

# ----------------------------
# ---  WebCamera 
# ----------------------------
# --- (globals)
WEBCAM_STREAM_FPS = 12.0   # streaming loop fps
WEBCAM_DETECT_FPS = 2.0    # detection loop fps

webcam_det_lock = threading.Lock()
webcam_det_cache = {
    "ts": 0,
    "fps": 0.0,
    "resolution": [0, 0],
    "detections": [],
}

_webcam_detect_thread = None
_webcam_detect_running = False

class WebcamStream:
    def __init__(self, cam_index: int = 0, width: int = 1280, height: int = 720, fps: int = 15):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.last_jpeg = None
        self.last_frame = None
        self.thread = None

    def start(self):
        if self.running:
            return

        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)  # Windows-friendly
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0).")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.cap = cap
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

    def _loop(self):
        delay = 1.0 / max(1, self.fps)
        while self.running and self.cap:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            # ✅ cache raw frame for YOLO
            with self.lock:
                self.last_frame = frame

            frame_to_send = frame
            max_w = 1280  # or 960
            h, w = frame_to_send.shape[:2]
            if w > max_w:
                nh = int(h * (max_w / w))
                frame_to_send = cv2.resize(frame_to_send, (max_w, nh), interpolation=cv2.INTER_LINEAR)

            ok2, buf = cv2.imencode(".jpg", frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            if ok2:
                with self.lock:
                    self.last_jpeg = buf.tobytes()

            time.sleep(delay)

    def get_jpeg(self):
        with self.lock:
            return self.last_jpeg
        
    def get_frame(self):
        with self.lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

webcam = WebcamStream(cam_index=0, fps=15)

# ------- Live RTSP Stream ---------
# cv2.VideoCapture(rtsp_url)

# ----------------------------------
class LiveVideoSession:
    def __init__(self, video_id: str, video_path: str, stream_fps: float = STREAM_FPS, detect_fps: float = DETECT_FPS):
        self._lock = threading.Lock()
        self.video_id = video_id
        self.video_path = video_path
        # nice display name for events page
        try:
            fname = Path(video_path).name
            original_name = fname.split("__", 1)[1] if "__" in fname else fname
        except Exception:
            original_name = Path(video_path).name
        self.video_name = _get_video_display_name(video_id, original_name, fallback=video_id)

        self.stream_fps = float(stream_fps)
        self.detect_fps = max(0.1, float(detect_fps))  # avoid div by 0

        self.stop_event = threading.Event()
        self.thread = None

        self.latest_jpeg = None
        self.latest_detections = []
        self.latest_ts = int(time.time())
        self.latest_fps = int(self.stream_fps)
        self.resolution = (0, 0)
        self._fps_times = deque(maxlen=30)  # timestamps for "actual fps" estimate

        self._frame_idx = 0
        self._mosaic_every_n = 12
        self._tile_size = (640, 360)
        self._last_mosaic = None
        self._last_tile_meta = None
        self.last_access = time.time()

        self.native_fps = None
        self.start_wall = None
        self.start_pos_msec = 0.0

        self._vio_counter = {}
        self._vio_last_ts = {}

        self._recent = deque(maxlen=50)
        self._next_detect_ts = 0.0
        self._is_fisheye = None

        self._dewarper = None
        self._dewarp_ver = -1
        self.preview_mode = False

    @property
    def lock(self):
        return self._lock

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def snapshot(self):
        with self._lock:
            return {
                "ts": self.latest_ts,
                "fps": self.latest_fps,
                "resolution": [int(self.resolution[0]), int(self.resolution[1])],
                "detections": list(self.latest_detections),
            }

    def iter_mjpeg(self):
        # PUSH_INTERVAL: how often the latest cached JPEG is sent (no re-encoding).
        # IDLE_INTERVAL: delay used when no JPEG is available.
        PUSH_INTERVAL = 0.06   # ~33 FPS maximum
        IDLE_INTERVAL = 0.05   # Wait time when idle

        while not self.stop_event.is_set():
            self.last_access = time.time()

            with self._lock:
                jpg = self.latest_jpeg

            if jpg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n" + jpg + b"\r\n"
                )
                time.sleep(PUSH_INTERVAL)
            else:
                time.sleep(IDLE_INTERVAL)

    def force_recompute_mosaic(self):
        self._last_mosaic = None
        self._last_tile_meta = None
        self._frame_idx = 0

    def _tile_bbox_local(self, tile_meta, bbox):
        x1,y1,x2,y2 = bbox
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        for t in (tile_meta or []):
            x0,y0,w,h = t["x0"], t["y0"], t["w"], t["h"]
            if x0 <= cx < x0+w and y0 <= cy < y0+h:
                return t["name"], [x1-x0, y1-y0, x2-x0, y2-y0], (w,h)
        return "normal", [x1,y1,x2,y2], None
    
    def _get_view_configs_for_video(self):
        with DEWARP_LOCK:
            cfg = None
            if self.preview_mode:
                cfg = DEWARP_PREVIEW_BY_VIDEO.get(self.video_id)
            if cfg is None:
                cfg = DEWARP_BY_VIDEO.get(self.video_id)

        if isinstance(cfg, dict) and isinstance(cfg.get("views"), list) and len(cfg["views"]) == 4:
            return cfg["views"], int(cfg.get("ver", 0))

        return [c.copy() for c in VIEW_CONFIGS], 0
    
    def _iou(self, a, b):
        return _box_iou(a, b)

    def _run(self):
        cap = cv2.VideoCapture(self.video_path)
        src = str(self.video_path or "").strip().lower()
        is_rtsp = src.startswith(("rtsp://", "rtsps://")) #startwith may need modify
        reconnect_wait = 0.5

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            return

        # --- read native FPS once (used when stream_fps is not explicitly set) ---
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps != src_fps or src_fps < 1.0:
            if is_rtsp:
                src_fps = 25.0   # default live camera
            else:
                src_fps = 25.0   # fallback for broken file
        self.native_fps = float(src_fps)

        # --- "real-time" anchor (video timeline -> wall clock) ---
        self.start_wall = time.time()
        self.start_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0

        detector = _get_detector()

        try:
            while not self.stop_event.is_set():
                try:
                    # if user sets stream_fps explicitly, respect it; otherwise use native FPS
                    sfps = float(self.stream_fps)
                    fps_target = sfps if sfps >= 1.0 else float(self.native_fps or 25.0)
                    fps_target = max(1.0, fps_target)
                    frame_period = 1.0 / fps_target

                    # --- read frame ---
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if is_rtsp:
                            # RTSP: reconnect (do NOT reset everything like looping a file)
                            try:
                                cap.release()
                            except Exception:
                                pass

                            time.sleep(min(5.0, reconnect_wait))
                            reconnect_wait = min(5.0, reconnect_wait * 1.6)

                            cap = cv2.VideoCapture(self.video_path)
                            try:
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            except Exception:
                                pass

                            if not cap.isOpened():
                                continue  # keep trying until stop_event set

                            # if reconnect success, clear jpeg/dets to avoid stale UI
                            with self._lock:
                                self.latest_detections = []
                                self.latest_jpeg = None

                            self._is_fisheye = None
                            self._fps_times.clear()
                            self.start_wall = time.time()
                            self.start_pos_msec = 0.0
                            self._frame_idx = 0
                            self._last_mosaic = None
                            self._last_tile_meta = None
                            self._dewarper = None
                            self._dewarp_ver = -1

                            continue
                        else:
                            # FILE: old behavior (loop)
                            cap.release()
                            cap = cv2.VideoCapture(self.video_path)
                            try:
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            except Exception:
                                pass
                            if not cap.isOpened():
                                return
                            # reset timeline anchor
                            self.start_wall = time.time()
                            self.start_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
                            self._last_mosaic = None
                            self._last_tile_meta = None
                            self._dewarper = None
                            self._dewarp_ver = -1
                            self._frame_idx = 0
                            self._is_fisheye = None
                            self._fps_times.clear()
                            continue
                    
                    if is_rtsp:
                        reconnect_wait = 0.5

                    # Detect fisheye ONCE, only when we have a REAL frame
                    if self._is_fisheye is None:
                        try:
                            self._is_fisheye = bool(is_fisheye(frame))
                        except Exception:
                            self._is_fisheye = False

                    self._frame_idx += 1
                    # ----------------------------
                    # "real-time" pacing using video timestamps
                    # (DISABLED for RTSP: POS_MSEC is unreliable)
                    # ----------------------------
                    due = None

                    if not is_rtsp:
                        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0

                        # Fallback if POS_MSEC is broken (some codecs return 0 always)
                        if pos_msec <= 0.0:
                            pos_msec = self.start_pos_msec + (self._frame_idx * frame_period * 1000.0)

                        video_elapsed = max(0.0, (pos_msec - float(self.start_pos_msec)) / 1000.0)
                        due = float(self.start_wall) + video_elapsed

                        # If we're behind, skip frames to catch up (low latency)
                        tolerance = 0.08  # 80ms
                        now = time.time()
                        if now > due + tolerance:
                            behind = now - due
                            drop = int(behind / frame_period)
                            drop = min(drop, 60)
                            for _ in range(drop):
                                if not cap.grab():
                                    break

                    # ----------------------------
                    # bookkeeping / timestamps
                    # ----------------------------
                    now = time.time()
                    self.latest_ts = int(now)

                    # compute "actual fps" based on recent frame timestamps
                    self._fps_times.append(now)
                    if len(self._fps_times) >= 2:
                        span = self._fps_times[-1] - self._fps_times[0]
                        if span > 1e-6:
                            fps_est = (len(self._fps_times) - 1) / span
                            with self._lock:
                                self.latest_fps = int(round(fps_est))

                    # ----------------------------
                    # Mosaic / fisheye dewarp (cached)
                    # ----------------------------
                    frame_to_show = frame
                    tile_meta = None

                    try:
                        fisheye = bool(self._is_fisheye)

                        if fisheye:
                            view_cfgs, ver = self._get_view_configs_for_video()
                            cfg_changed = (ver != self._dewarp_ver)

                            need_recompute = (
                                self.preview_mode
                                or cfg_changed
                                or (self._last_mosaic is None)
                                or (self._frame_idx % self._mosaic_every_n == 0)
                            )

                            if need_recompute:
                                if (self._dewarper is None) or (ver != self._dewarp_ver):
                                    self._dewarper = FisheyeMultiViewDewarper(
                                        frame.shape,
                                        view_configs=view_cfgs,
                                        output_shape=OUTPUT_SHAPE,
                                        input_fov=INPUT_FOV_DEG,
                                    )
                                    self._dewarp_ver = ver

                                views = self._dewarper.generate_views(frame)
                                view_items = _pick_4_views(views)

                                labels_by_name = {}
                                if isinstance(view_cfgs, list):
                                    for c in view_cfgs:
                                        if isinstance(c, dict) and c.get("name"):
                                            labels_by_name[c["name"]] = c.get("label") or c["name"]

                                mosaic, tm = _make_2x2_mosaic(view_items, tile_size=self._tile_size, labels_by_name=labels_by_name)

                                if mosaic is not None:
                                    self._last_mosaic = mosaic
                                    self._last_tile_meta = tm

                            if self._last_mosaic is not None:
                                frame_to_show = self._last_mosaic
                                tile_meta = self._last_tile_meta
                            else:
                                frame_to_show = frame
                                tile_meta = None
                        else:
                            frame_to_show = frame
                            tile_meta = None

                    except Exception as e:
                        print("[fisheye] mosaic failed:", e)
                        frame_to_show = frame
                        tile_meta = None

                    # ----------------------------
                    # Encode JPEG for streaming (with downscale cap)
                    # ----------------------------
                    hs, ws = frame_to_show.shape[:2]
                    self.resolution = (ws, hs)

                    has_viewer = (time.time() - self.last_access) <= 1.0
                    if has_viewer or (self.latest_jpeg is None):
                        frame_to_send = frame_to_show

                        # cap streaming resolution to reduce encode cost (big win)
                        max_w = 1280  # try 960 if still heavy
                        h, w = frame_to_send.shape[:2]
                        if w > max_w:
                            nh = int(h * (max_w / w))
                            frame_to_send = cv2.resize(frame_to_send, (max_w, nh), interpolation=cv2.INTER_LINEAR)

                        ok2, jpg = cv2.imencode(".jpg", frame_to_send, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
                        if ok2:
                            with self._lock:
                                self.latest_jpeg = jpg.tobytes()

                    # ----------------------------
                    # Detection scheduling (unchanged from your code)
                    # ----------------------------
                    now_ts = time.time()
                    detect_fps_now = max(0.1, float(self.detect_fps))

                    if now_ts >= self._next_detect_ts:
                        self._next_detect_ts = now_ts + (1.0 / detect_fps_now)

                        if not _get_enabled_for_video(self.video_id):
                            with self._lock:
                                self.latest_detections = []
                            raw = None
                            try:
                                self._vio_counter.clear()
                            except Exception:
                                pass
                            try:
                                self._vio_last_ts.clear()
                            except Exception:
                                pass
                            try:
                                self._recent.clear()
                            except Exception:
                                pass
                        else:
                            sched_cfg = _get_schedule_for_video(self.video_id)
                            if not _is_now_in_schedule(sched_cfg, datetime.now()):
                                with self._lock:
                                    self.latest_detections = []
                                raw = None
                            else:
                                h_show, w_show = frame_to_show.shape[:2]
                                detect_frame = frame_to_show
                                dw, dh = w_show, h_show

                                TARGET_DET_W = 960  # try 832/960/640 depending on speed vs accuracy
                                if w_show > TARGET_DET_W:
                                    dh = int(h_show * (TARGET_DET_W / w_show))
                                    dw = TARGET_DET_W
                                    detect_frame = cv2.resize(frame_to_show, (dw, dh), interpolation=cv2.INTER_LINEAR)

                                sx = w_show / float(dw)  # scale detect->show
                                sy = h_show / float(dh)

                                try:
                                    raw = detector.detect(detect_frame)
                                except Exception as e:
                                    print("[detect] error:", repr(e))
                                    with self._lock:
                                        self.latest_detections = []
                                    raw = None

                                if raw is not None:
                                    enabled_vios = _get_enabled_violation_map()
                                    dets_out = []

                                    with ROI_LOCK:
                                        roi_cfg = ROI_BY_VIDEO.get(self.video_id, {}) or {}

                                    best_vio = {}  # (view_name, vio_label) -> {conf,bbox,local_bbox,tile}
                                    for i, (bbox_det, label, cf) in enumerate(_iter_boxes_from_raw(raw) or []):
                                        # bbox_det is in detect_frame coords -> scale back to frame_to_show coords
                                        x1, y1, x2, y2 = bbox_det
                                        bbox = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]

                                        # tile mapping uses frame_to_show coords
                                        view_name, local_bbox, tile_wh = self._tile_bbox_local(tile_meta, bbox)

                                        vio = _label_to_violation(label)
                                        if (vio in LIVE_VIOLATION_CLASSES) and (not enabled_vios.get(vio, True)):
                                            continue
                                        # event candidates (violations only)
                                        if (vio in LIVE_VIOLATION_CLASSES) and (float(cf) >= LIVE_VIOLATION_CONF):
                                            k = (view_name or "normal", vio)
                                            prev = best_vio.get(k)
                                            if (prev is None) or (float(cf) > float(prev["conf"])):
                                                # find tile meta so we can crop evidence from the tile (not the whole mosaic)
                                                tile = None
                                                if tile_meta is not None:
                                                    for t in (tile_meta or []):
                                                        if t.get("name") == view_name:
                                                            tile = t
                                                            break

                                                best_vio[k] = {
                                                    "conf": float(cf),
                                                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],        # mosaic/full coords
                                                    "local_bbox": [float(local_bbox[0]), float(local_bbox[1]), float(local_bbox[2]), float(local_bbox[3])],
                                                    "tile": tile,
                                                }

                                        # ROI filtering (must use correct tile size!)
                                        if tile_wh is not None:
                                            tw, th = tile_wh
                                            roi_polys = roi_cfg.get(view_name, [])
                                            if roi_polys and (not _bbox_inside_any_roi_percent(
                                                local_bbox, roi_polys, tw, th, mode="center"
                                            )):
                                                continue
                                        else:
                                            roi_polys = roi_cfg.get("normal", [])
                                            if roi_polys and (not _bbox_inside_any_roi_percent(
                                                bbox, roi_polys, w_show, h_show, mode="center"
                                            )):
                                                continue

                                        # percent conversion must use frame_to_show dims
                                        vx, vy, vw, vh = _xyxy_to_percent(bbox[0], bbox[1], bbox[2], bbox[3], w_show, h_show)

                                        dets_out.append({
                                            "id": f"{self.video_id}-det-{i}",
                                            "x": vx, "y": vy, "width": vw, "height": vh,
                                            "label": label,
                                            "violation": vio,
                                            "conf": float(cf),
                                        })
                                    # write offline events (one per (view,label) per detect tick, best conf only)
                                    for (vname, vio_label), info in (best_vio or {}).items():
                                        try:
                                            tile = info.get("tile")

                                            # If fisheye mosaic: crop the tile first so evidence is clean
                                            if tile is not None:
                                                x0, y0, tw, th = int(tile["x0"]), int(tile["y0"]), int(tile["w"]), int(tile["h"])
                                                tile_frame = frame_to_show[y0:y0+th, x0:x0+tw].copy()
                                                bbox_use = info["local_bbox"]  # local coords within tile
                                                frame_use = tile_frame
                                            else:
                                                frame_use = frame_to_show
                                                bbox_use = info["bbox"]

                                             # 1) match person bbox for this violation bbox
                                            person_bbox = _match_person_bbox_for_violation(frame_use, bbox_use, scale=0.5)
                                            # 2) assign lightweight track id
                                            track_id = None
                                            if person_bbox is not None:
                                                track_id = _assign_light_track_id(self.video_id, vname, person_bbox)

                                            _write_attire_event_offline(
                                                video_id=self.video_id,
                                                video_name=self.video_name,
                                                view_name=vname,
                                                label=vio_label,
                                                conf=float(info["conf"]),
                                                frame_bgr=frame_use,
                                                bbox_xyxy=bbox_use,
                                                track_id=track_id,
                                            )
                                        except Exception as e:
                                            print("[offline event] failed:", repr(e))

                                    with self._lock:
                                        self.latest_detections = dets_out

                    if (not is_rtsp) and (due is not None):
                        now = time.time()
                        sleep_s = due - now
                        if sleep_s > 0:
                            time.sleep(min(0.05, sleep_s))
                    else:
                        # RTSP: simple throttle to target fps (prevents 100% CPU)
                        time.sleep(min(0.01, frame_period))

                except Exception as e:
                    print("[LiveVideoSession] _run error:", repr(e))
                    try:
                        traceback.print_exc()
                    except Exception:
                        pass
                    time.sleep(0.1)
                    continue

        finally:
            cap.release()

# ----------------------------
# API: Events
# ----------------------------
@app.get("/api/attire/events")
def get_attire_events(video_id: str = "", limit: int = 200):
    with ATTIRE_EVENTS_LOCK:
        items = list(ATTIRE_EVENTS)

    if video_id:
        items = [e for e in items if e.get("video_id") == video_id]

    items = sorted(items, key=lambda x: int(x.get("ts", 0)), reverse=True)
    events = items[:limit]
    events = [_decorate_attire_event(e) for e in events]
    return {"events": events}

@app.delete("/api/attire/events")
def clear_attire_events(video_id: str = ""):
    with ATTIRE_EVENTS_LOCK:
        if video_id:
            kept = [e for e in ATTIRE_EVENTS if e.get("video_id") != video_id]
            ATTIRE_EVENTS[:] = kept
        else:
            ATTIRE_EVENTS[:] = []
        _save_attire_events_file(ATTIRE_EVENTS)
    return {"ok": True}

@app.patch("/api/attire/events/{event_id}")
def patch_attire_event(event_id: str, body: dict = Body(...)):
    """Update a single event (status/location/notes). Persists to attire_events.json."""
    allowed = {"status", "location", "notes", "label", "severity", "source"}
    updates = {k: body.get(k) for k in allowed if k in body}

    # normalize status if present
    if "status" in updates:
        s = str(updates["status"] or "").strip().title()
        if s not in {"Pending", "Resolved"}:
            raise HTTPException(status_code=400, detail="status must be Pending or Resolved")
        updates["status"] = s

    with ATTIRE_EVENTS_LOCK:
        idx = next((i for i, e in enumerate(ATTIRE_EVENTS) if str(e.get("id")) == event_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="event not found")

        ev = dict(ATTIRE_EVENTS[idx])
        ev.update({k: v for k, v in updates.items()})
        ATTIRE_EVENTS[idx] = ev
        _save_attire_events_file(ATTIRE_EVENTS)

    return {"ok": True, "event": ev}

@app.delete("/api/attire/events/{event_id}")
def delete_attire_event(event_id: str):
    """Delete a single event by id. Persists to attire_events.json."""
    with ATTIRE_EVENTS_LOCK:
        before = len(ATTIRE_EVENTS)
        ATTIRE_EVENTS[:] = [e for e in ATTIRE_EVENTS if str(e.get("id")) != event_id]
        after = len(ATTIRE_EVENTS)
        if after == before:
            raise HTTPException(status_code=404, detail="event not found")
        _save_attire_events_file(ATTIRE_EVENTS)
    return {"ok": True, "deleted": event_id}

# ----------------------------
# API: RTSP Sources
# ----------------------------
@app.get("/api/rtsp/sources")
def list_rtsp_sources():
    with RTSP_LOCK:
        items = [
            {"id": rid, "name": (cfg or {}).get("name", rid), "url": (cfg or {}).get("url", "")}
            for rid, cfg in (RTSP_BY_ID or {}).items()
        ]
    # also include enabled state (reuses your existing SOURCES store)
    for it in items:
        it["enabled"] = _get_enabled_for_video(it["id"])
    items.sort(key=lambda x: x["id"])
    return {"sources": items}

@app.post("/api/rtsp/sources/{rtsp_id}")
def upsert_rtsp_source(rtsp_id: str, body: dict = Body(...)):
    name = (body.get("name") or rtsp_id).strip() or rtsp_id
    url = (body.get("url") or "").strip()
    if not url.lower().startswith(("rtsp://", "rtsps://")):
        raise HTTPException(status_code=400, detail="url must start with rtsp:// or rtsps://")

    with RTSP_LOCK:
        RTSP_BY_ID[rtsp_id] = {"name": name, "url": url}
        _save_rtsp_file(RTSP_BY_ID)

    # ensure it appears in Sources toggle store (default enabled=True)
    with SOURCES_LOCK:
        if rtsp_id not in SOURCES_BY_VIDEO:
            SOURCES_BY_VIDEO[rtsp_id] = {"enabled": True}
            _save_sources_file(SOURCES_BY_VIDEO)

    return {"ok": True, "id": rtsp_id, "name": name, "url": url}

@app.delete("/api/rtsp/sources/{rtsp_id}")
def delete_rtsp_source(rtsp_id: str):
    with LIVE_LOCK:
        sess = LIVE_SESSIONS.pop(rtsp_id, None)
        if sess:
            sess.stop()

    with RTSP_LOCK:
        existed = rtsp_id in RTSP_BY_ID
        RTSP_BY_ID.pop(rtsp_id, None)
        _save_rtsp_file(RTSP_BY_ID)

    return {"ok": True, "deleted": rtsp_id, "existed": existed}

# ----------------------------
# API: Offline Upload / Videos
# ----------------------------
@app.post("/api/offline/upload")
async def upload_video(file: UploadFile = File(...)):
    vid = f"vid-{uuid.uuid4().hex[:8]}"
    safe_name = file.filename.replace("\\", "_").replace("/", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{vid}__{safe_name}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    VIDEOS[vid] = {"path": save_path, "name": safe_name}
    # default enabled for live view
    with SOURCES_LOCK:
        if vid not in SOURCES_BY_VIDEO:
            SOURCES_BY_VIDEO[vid] = {"enabled": False}
            _save_sources_file(SOURCES_BY_VIDEO)

    return {"id": vid, "name": safe_name, "path": save_path, "status": "ready"}

@app.get("/api/offline/videos")
def list_videos():
    items = []
    for fp in Path(UPLOAD_DIR).glob("*"):
        if not fp.is_file():
            continue
        name = fp.name
        if "__" not in name:
            continue

        vid, original_name = name.split("__", 1)
        st = fp.stat()
        size_mb = f"{st.st_size / (1024*1024):.0f} MB"
        upload_iso = datetime.fromtimestamp(st.st_mtime).isoformat()
        display_name = _get_video_display_name(vid, original_name)

        items.append({
            "id": vid,
            "name": display_name,            # show alias if set
            "original_name": original_name,  # keep original for reference 
            "path": str(fp),
            "size": size_mb,
            "duration": "00:00:00",
            "uploadDate": upload_iso,
            "status": "ready",
        })

    items.sort(key=lambda x: x["uploadDate"], reverse=True)
    return items


@app.delete("/api/offline/videos/{video_id}")
def delete_video(video_id: str):
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found")

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.pop(video_id, None)
        if sess:
            sess.stop()

    for fp in matches:
        fp.unlink(missing_ok=True)

    VIDEOS.pop(video_id, None)
    # remove saved alias
    with LABELS_LOCK:
        if video_id in LABELS_BY_VIDEO:
            LABELS_BY_VIDEO.pop(video_id, None)
            _save_labels_file(LABELS_BY_VIDEO)

    return {"ok": True, "deleted_id": video_id, "deleted_files": len(matches)}

@app.get("/api/offline/snapshot/{video_id}")
def offline_snapshot(video_id: str):
    video_path = _get_uploaded_video_path(video_id)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Cannot read frame")

    # Optional: resize for faster Settings page (recommended)
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

    ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok2:
        raise HTTPException(status_code=500, detail="JPEG encode failed")

    return Response(content=jpg.tobytes(), media_type="image/jpeg")

@app.get("/api/offline/meta/{video_id}")
def offline_video_meta(video_id: str):
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    video_path = str(matches[0])
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return {"video_id": video_id, "is_fisheye": False}

    try:
        fisheye = bool(is_fisheye(frame))
    except Exception:
        fisheye = False

    return {"video_id": video_id, "is_fisheye": fisheye}

@app.get("/api/offline/labels")
def get_video_labels():
    with LABELS_LOCK:
        out = {k: (v or {}).get("name", "") for k, v in LABELS_BY_VIDEO.items()}
    return {"labels": out}

@app.post("/api/offline/labels/{video_id}")
def set_video_label(video_id: str, body: dict = Body(...)):
    """
    Body: {"name": "B001G - Class 1"}
    If name is empty/blank -> remove alias (fallback to original filename)
    """
    new_name = (body.get("name") or "").strip()

    with LABELS_LOCK:
        if new_name:
            LABELS_BY_VIDEO[video_id] = {"name": new_name}
        else:
            LABELS_BY_VIDEO.pop(video_id, None)
        _save_labels_file(LABELS_BY_VIDEO)

    return {"ok": True, "video_id": video_id, "name": new_name}

# RTSP snapshot + meta endpoints (Settings)
@app.get("/api/rtsp/snapshot/{rtsp_id}")
def rtsp_snapshot(rtsp_id: str):
    cfg = _get_rtsp(rtsp_id)
    url = (cfg.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="RTSP source not found")

    cap = cv2.VideoCapture(url)
    try:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open RTSP stream")

        for _ in range(3):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="Cannot read RTSP frame")

        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            raise HTTPException(status_code=500, detail="JPEG encode failed")

        return Response(content=jpg.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()

@app.get("/api/rtsp/meta/{rtsp_id}")
def rtsp_meta(rtsp_id: str):
    cfg = _get_rtsp(rtsp_id)
    url = (cfg.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="RTSP source not found")

    cap = cv2.VideoCapture(url)
    try:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        for _ in range(3):
            if not cap.grab():
                break
        ok, frame = cap.read()
        if not ok or frame is None:
            return {"video_id": rtsp_id, "is_fisheye": False}

        try:
            fisheye = bool(is_fisheye(frame))
        except Exception:
            fisheye = False

        return {"video_id": rtsp_id, "is_fisheye": fisheye}
    finally:
        cap.release()

# RTSP stream + live detections endpoints
@app.get("/api/rtsp/stream/{rtsp_id}")
def rtsp_stream(
    rtsp_id: str,
    stream_fps: Optional[float] = None,
    detect_fps: Optional[float] = None,
    preview: int = 0
):
    cfg = _get_rtsp(rtsp_id)
    url = (cfg.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="RTSP source not found")
    # BLOCK streaming if disabled
    if not _get_enabled_for_video(rtsp_id):
        raise HTTPException(status_code=403, detail="Source disabled")

    # saved fps defaults
    saved_stream_fps, saved_detect_fps = _get_fps_for_video(rtsp_id)
    stream_fps = saved_stream_fps if stream_fps is None else _clamp_stream_fps(stream_fps)
    detect_fps = saved_detect_fps if detect_fps is None else _clamp_detect_fps(detect_fps)

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(rtsp_id)
        if not sess:
            return {
                "ts": int(time.time()),
                "fps": 0,
                "resolution": [0, 0],
                "detections": [],
                "detail": "No active RTSP session"
            }
        else:
            fps_changed = (float(sess.stream_fps) != float(stream_fps)) or (float(sess.detect_fps) != float(detect_fps))
            sess.stream_fps = stream_fps
            sess.detect_fps = detect_fps
            if fps_changed:
                sess._next_detect_ts = 0.0

        sess.preview_mode = (int(preview) == 1)
        if sess.preview_mode:
            sess.force_recompute_mosaic()

        _touch_session(sess)
        sess.start()
        sess.last_access = time.time()

    return StreamingResponse(
        sess.iter_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.get("/api/rtsp/live/{rtsp_id}/detections")
def rtsp_live_detections(rtsp_id: str):
    cfg = _get_rtsp(rtsp_id)
    url = (cfg.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="RTSP source not found")

    saved_stream_fps, saved_detect_fps = _get_fps_for_video(rtsp_id)

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(rtsp_id)
        if not sess:
            _ensure_live_slot(rtsp_id)
            sess = LiveVideoSession(rtsp_id, url, stream_fps=saved_stream_fps, detect_fps=saved_detect_fps)
            LIVE_SESSIONS[rtsp_id] = sess
        else:
            fps_changed = (float(sess.stream_fps) != float(saved_stream_fps)) or (float(sess.detect_fps) != float(saved_detect_fps))
            sess.stream_fps = saved_stream_fps
            sess.detect_fps = saved_detect_fps
            if fps_changed:
                sess._next_detect_ts = 0.0

        _touch_session(sess)
        sess.start()
        return sess.snapshot()
    
@app.post("/api/rtsp/close/{rtsp_id}")
def rtsp_close(rtsp_id: str):
    with LIVE_LOCK:
        sess = LIVE_SESSIONS.pop(rtsp_id, None)
        if sess:
            sess.stop()
    return {"ok": True}

@app.get("/api/rtsp/snapshot_dewarp/{rtsp_id}")
def rtsp_snapshot_dewarp(rtsp_id: str):
    cfg = _get_rtsp(rtsp_id)
    url = (cfg.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="RTSP source not found")

    cap = cv2.VideoCapture(url)
    try:
        # Try to reduce latency (may be ignored depending on backend)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open RTSP stream")

        # Warm up a few frames so we don't get an old/black frame
        for _ in range(3):
            if not cap.grab():
                break

        ok, frame = cap.read()
        if not ok or frame is None:
            raise HTTPException(status_code=500, detail="Cannot read RTSP frame")

        # Detect if fisheye
        try:
            fisheye = bool(is_fisheye(frame))
        except Exception:
            fisheye = False

        # If not fisheye, just return a normal snapshot
        if not fisheye:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok2:
                raise HTTPException(status_code=500, detail="JPEG encode failed")
            return Response(
                content=jpg.tobytes(),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        # Use preview config if exists, else saved config, else defaults
        view_cfgs = _get_effective_dewarp_views(rtsp_id)

        labels_by_name = {}
        if isinstance(view_cfgs, list):
            for c in view_cfgs:
                if isinstance(c, dict) and c.get("name"):
                    labels_by_name[c["name"]] = c.get("label") or c["name"]

        # Build mosaic from dewarped fisheye views
        try:
            dewarper = FisheyeMultiViewDewarper(
                frame.shape,
                view_configs=view_cfgs,
                output_shape=OUTPUT_SHAPE,
                input_fov=INPUT_FOV_DEG,
            )
            views = dewarper.generate_views(frame)
            view_items = _pick_4_views(views)

            # tile_size (640x360) => mosaic 1280x720
            mosaic, _tm = _make_2x2_mosaic(
                view_items,
                tile_size=(640, 360),
                labels_by_name=labels_by_name,
            )
            if mosaic is None:
                raise RuntimeError("mosaic is None")
        except Exception:
            # Fallback so UI still shows something
            mosaic = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        ok2, jpg = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            raise HTTPException(status_code=500, detail="JPEG encode failed")

        return Response(
            content=jpg.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    finally:
        try:
            cap.release()
        except Exception:
            pass

# ----------------------------
# API: Offline Stream / Live Detections
# ----------------------------
@app.get("/api/offline/stream/{video_id}")
def offline_stream(
        video_id: str,
        stream_fps: Optional[float] = None,
        detect_fps: Optional[float] = None,
        preview: int = 0
    ):
    # If query params are missing, use saved per-video config
    saved_stream_fps, saved_detect_fps = _get_fps_for_video(video_id)
    stream_fps = saved_stream_fps if stream_fps is None else _clamp_stream_fps(stream_fps)
    detect_fps = saved_detect_fps if detect_fps is None else _clamp_detect_fps(detect_fps)
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    video_path = str(matches[0])

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(video_id)

        if not sess:
            _ensure_live_slot(video_id)
            sess = LiveVideoSession(video_id, video_path, stream_fps=stream_fps, detect_fps=detect_fps)
            LIVE_SESSIONS[video_id] = sess
        else:
            fps_changed = (float(sess.stream_fps) != float(stream_fps)) or (float(sess.detect_fps) != float(detect_fps))

            sess.stream_fps = stream_fps
            sess.detect_fps = detect_fps

            # Only reset the detection scheduler if fps values changed
            if fps_changed:
                sess._next_detect_ts = 0.0

        sess.preview_mode = (int(preview) == 1)
        if sess.preview_mode:
            sess.force_recompute_mosaic()

        _touch_session(sess)
        sess.start()
        sess.last_access = time.time()

    return StreamingResponse(
        sess.iter_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

@app.post("/api/offline/close/{video_id}")
def offline_close(video_id: str):
    with LIVE_LOCK:
        sess = LIVE_SESSIONS.pop(video_id, None)
        if sess:
            sess.stop()
    return {"ok": True}

@app.get("/api/offline/live/{video_id}/detections")
def offline_live_detections(video_id: str):
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    video_path = str(matches[0])
    saved_stream_fps, saved_detect_fps = _get_fps_for_video(video_id)

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(video_id)

        if not sess:
            return {
                "ts": int(time.time()),
                "fps": 0,
                "resolution": [0, 0],
                "detections": [],
                "detail": "No active offline session"
            }
        else:
            fps_changed = (float(sess.stream_fps) != float(saved_stream_fps)) or (float(sess.detect_fps) != float(saved_detect_fps))

            sess.stream_fps = saved_stream_fps
            sess.detect_fps = saved_detect_fps

            if fps_changed:
                sess._next_detect_ts = 0.0

        _touch_session(sess)
        sess.start()

        return sess.snapshot()
    
@app.get("/api/rtsp/thumb/{rtsp_id}")
def rtsp_thumb(rtsp_id: str):
    cfg = _get_rtsp(rtsp_id)
    url = (cfg.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="RTSP source not found")

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(rtsp_id)
        if sess:
            _touch_session(sess)
            with sess.lock:
                jpg = sess.latest_jpeg
            if jpg:
                _thumb_cache_set(f"rtsp:{rtsp_id}", jpg)
                return Response(
                    content=jpg,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, max-age=0"},
                )

    cached = _thumb_cache_get(f"rtsp:{rtsp_id}")
    if cached:
        return Response(
            content=cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    resp = rtsp_snapshot_dewarp(rtsp_id)
    try:
        body = resp.body
        if body:
            _thumb_cache_set(f"rtsp:{rtsp_id}", body)
    except Exception:
        pass
    return resp

@app.get("/api/offline/thumb/{video_id}")
def offline_thumb(video_id: str):
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(video_id)
        if sess:
            _touch_session(sess)
            with sess.lock:
                jpg = sess.latest_jpeg
            if jpg:
                _thumb_cache_set(f"offline:{video_id}", jpg)
                return Response(
                    content=jpg,
                    media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, max-age=0"},
                )

    cached = _thumb_cache_get(f"offline:{video_id}")
    if cached:
        return Response(
            content=cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    resp = snapshot_dewarp(video_id)
    try:
        body = resp.body
        if body:
            _thumb_cache_set(f"offline:{video_id}", body)
    except Exception:
        pass
    return resp
    
# ----------------------------
# API: ROI
# ----------------------------
@app.get("/api/attire/roi/{video_id}")
def get_attire_roi(video_id: str):
    with ROI_LOCK:
        return {"video_id": video_id, "rois": ROI_BY_VIDEO.get(video_id, {})}

@app.post("/api/attire/roi/{video_id}")
def set_attire_roi(video_id: str, payload: dict = Body(...)):
    rois = payload.get("rois", {})
    if not isinstance(rois, dict):
        raise HTTPException(status_code=400, detail="rois must be an object")

    # basic validation
    for k, polys in rois.items():
        if not isinstance(polys, list):
            raise HTTPException(status_code=400, detail=f"rois[{k}] must be a list")
    with ROI_LOCK:
        ROI_BY_VIDEO[video_id] = rois
        _save_roi_file()
    return {"ok": True, "video_id": video_id}

# ----------------------------
# API: Dewarp
# ----------------------------
@app.get("/api/offline/snapshot_dewarp/{video_id}")
def snapshot_dewarp(video_id: str):
    matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded video not found")

    cap = cv2.VideoCapture(str(matches[0]))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise HTTPException(status_code=500, detail="Failed to read frame")

    # IMPORTANT: detect fisheye first
    try:
        fisheye = bool(is_fisheye(frame))
    except Exception:
        fisheye = False

    # Normal video -> return normal snapshot, no dewarp
    if not fisheye:
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        ok2, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            raise HTTPException(status_code=500, detail="JPEG encode failed")

        return Response(
            content=jpg.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    # Fisheye video -> use preview config if exists, else saved config, else defaults
    view_cfgs = _get_effective_dewarp_views(video_id)

    labels_by_name = {}
    if isinstance(view_cfgs, list):
        for c in view_cfgs:
            if isinstance(c, dict) and c.get("name"):
                labels_by_name[c["name"]] = c.get("label") or c["name"]

    try:
        dewarper = FisheyeMultiViewDewarper(
            frame.shape,
            view_configs=view_cfgs,
            output_shape=OUTPUT_SHAPE,
            input_fov=INPUT_FOV_DEG,
        )
        views = dewarper.generate_views(frame)

        view_items = _pick_4_views(views)
        mosaic, _tm = _make_2x2_mosaic(
            view_items,
            tile_size=(640, 360),
            labels_by_name=labels_by_name,
        )
        if mosaic is None:
            raise RuntimeError("mosaic is None")

    except Exception:
        # fallback so UI still shows something
        mosaic = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

    ok2, jpg = cv2.imencode(".jpg", mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok2:
        raise HTTPException(status_code=500, detail="JPEG encode failed")

    return Response(
        content=jpg.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )

@app.get("/api/attire/dewarp/{video_id}")
def get_attire_dewarp(video_id: str):
    with DEWARP_LOCK:
        cfg = DEWARP_BY_VIDEO.get(video_id)

    if cfg and isinstance(cfg, dict) and isinstance(cfg.get("views"), list):
        out = []
        for v in cfg["views"]:
            if not isinstance(v, dict): 
                continue
            name = v.get("name")
            if not name:
                continue
            out.append({
                "name": name,
                "label": v.get("label") or name,
                "roll_deg": float(v.get("roll_deg", 0.0)),
                "pitch_deg": float(v.get("pitch_deg", 0.0)),
                "fov_deg": float(v.get("fov_deg", 90.0)),
            })
        return {"video_id": video_id, "views": out}

    # default
    out = []
    for c in VIEW_CONFIGS:
        name = c.get("name")
        out.append({**c, "label": c.get("label") or name})
    return {"video_id": video_id, "views": out}

@app.post("/api/attire/dewarp/{video_id}")
def set_attire_dewarp(video_id: str, payload: dict = Body(...)):
    views = payload.get("views", None)
    if not isinstance(views, list) or len(views) != 4:
        raise HTTPException(status_code=400, detail="views must be a list of 4 view configs")

    # 1) Validate source exists + fisheye check (OFFLINE or RTSP)
    is_rtsp_source = False
    rtsp_url = ""

    # RTSP?
    cfg = _get_rtsp(video_id)
    rtsp_url = (cfg.get("url") or "").strip()
    if rtsp_url:
        is_rtsp_source = True

    frame = None

    if is_rtsp_source:
        cap = cv2.VideoCapture(rtsp_url)
        try:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Cannot open RTSP stream")

            # warmup a bit
            for _ in range(3):
                if not cap.grab():
                    break

            ok, frame = cap.read()
            if not ok or frame is None:
                raise HTTPException(status_code=500, detail="Cannot read RTSP frame for fisheye check")
        finally:
            cap.release()

    else:
        # OFFLINE uploaded file
        matches = list(Path(UPLOAD_DIR).glob(f"{video_id}__*"))
        if not matches:
            raise HTTPException(status_code=404, detail="Uploaded video not found")

        cap = cv2.VideoCapture(str(matches[0]))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()

        if (not ok) or frame is None:
            raise HTTPException(status_code=500, detail="Failed to read video frame for fisheye check")

    # fisheye required
    try:
        if not bool(is_fisheye(frame)):
            raise HTTPException(status_code=400, detail="Dewarp is only available for fisheye sources")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Dewarp is only available for fisheye sources")

    # 2) Validate & clean view configs
    allowed_names = {"entrance", "corridor", "left_seats", "right_seats"}
    cleaned = []
    for v in views:
        if not isinstance(v, dict):
            raise HTTPException(status_code=400, detail="Each view must be an object")

        name = v.get("name")
        if name not in allowed_names:
            raise HTTPException(status_code=400, detail=f"Invalid view name: {name}")

        label = v.get("label")
        if label is None:
            label = name
        label = str(label).strip() or name

        cleaned.append({
            "name": name,
            "label": label,
            "roll_deg": float(v.get("roll_deg", 0.0)),
            "pitch_deg": float(v.get("pitch_deg", 0.0)),
            "fov_deg": float(v.get("fov_deg", 90.0)),
        })

    # 3) Save for BOTH offline + RTSP (keyed by video_id)
    with DEWARP_LOCK:
        old = DEWARP_BY_VIDEO.get(video_id, {}) if isinstance(DEWARP_BY_VIDEO.get(video_id), dict) else {}
        ver = int(old.get("ver", 0)) + 1
        DEWARP_BY_VIDEO[video_id] = {"views": cleaned, "ver": ver}
        DEWARP_PREVIEW_BY_VIDEO.pop(video_id, None)
        _save_dewarp_file(DEWARP_BY_VIDEO)

    return {"ok": True, "video_id": video_id, "is_rtsp": is_rtsp_source}

@app.post("/api/attire/dewarp_preview/{video_id}")
def set_dewarp_preview(video_id: str, payload: dict = Body(...)):
    views = payload.get("views")
    _validate_dewarp_views(views)

    ver = int(time.time() * 1000)

    with DEWARP_LOCK:
        DEWARP_PREVIEW_BY_VIDEO[video_id] = {"views": views, "ver": ver}

    return {"ok": True, "video_id": video_id, "ver": ver}

# ----------------------------
# API: FPS
# ----------------------------
@app.get("/api/attire/fps/{video_id}")
def get_attire_fps(video_id: str):
    stream_fps, detect_fps = _get_fps_for_video(video_id)
    return {"video_id": video_id, "stream_fps": stream_fps, "detect_fps": detect_fps}

@app.post("/api/attire/fps/{video_id}")
def set_attire_fps(video_id: str, body: dict = Body(...)):
    # Expected body: {"stream_fps": 12, "detect_fps": 2}
    stream_fps = body.get("stream_fps", STREAM_FPS)
    detect_fps = body.get("detect_fps", DETECT_FPS)

    stream_fps = _clamp_stream_fps(stream_fps)
    detect_fps = _clamp_detect_fps(detect_fps)

    with FPS_LOCK:
        FPS_BY_VIDEO[video_id] = {"stream_fps": stream_fps, "detect_fps": detect_fps}
        _save_fps_file(FPS_BY_VIDEO)

    # If a live session is already running, apply immediately
    with LIVE_LOCK:
        sess = LIVE_SESSIONS.get(video_id)
        if sess:
            if float(sess.stream_fps) != float(stream_fps) or float(sess.detect_fps) != float(detect_fps):
                sess.stream_fps = stream_fps
                sess.detect_fps = detect_fps
                sess._next_detect_ts = 0.0

    return {"ok": True, "video_id": video_id, "stream_fps": stream_fps, "detect_fps": detect_fps}

# ----------------------------
# API: Schedule
# ----------------------------
@app.get("/api/attire/schedule/{video_id}")
def get_attire_schedule(video_id: str):
    cfg = _get_schedule_for_video(video_id)
    return {"video_id": video_id, **cfg}

@app.post("/api/attire/schedule/{video_id}")
def set_attire_schedule(video_id: str, body: dict = Body(...)):
    enabled = bool(body.get("enabled", False))
    schedules = body.get("schedules", [])

    if not isinstance(schedules, list):
        raise HTTPException(status_code=400, detail="schedules must be a list")

    # basic sanitize (keep only expected keys)
    cleaned = []
    for s in schedules:
        if not isinstance(s, dict):
            continue
        cleaned.append({
            "id": str(s.get("id", "")),
            "startTime": str(s.get("startTime", "08:00"))[:5],
            "endTime": str(s.get("endTime", "18:00"))[:5],
            "enabled": bool(s.get("enabled", True)),
            "days": list(s.get("days", [])) if isinstance(s.get("days", []), list) else [],
        })

    with SCHEDULE_LOCK:
        SCHEDULE_BY_VIDEO[video_id] = {"enabled": enabled, "schedules": cleaned}
        _save_schedule_file(SCHEDULE_BY_VIDEO)

    return {"ok": True, "video_id": video_id, "enabled": enabled, "schedules": cleaned}

# ----------------------------
# API: Sources
# ----------------------------
@app.get("/api/attire/sources")
def get_attire_sources():
    video_ids = set()

    # 1) uploaded/offline videos
    for fp in Path(UPLOAD_DIR).glob("*__*"):
        vid = fp.name.split("__", 1)[0]
        if vid:
            video_ids.add(vid)

    # 2) rtsp sources from persistent store
    with RTSP_LOCK:
        for rid in (RTSP_BY_ID or {}).keys():
            video_ids.add(rid)

    out = {}
    for vid in sorted(video_ids):
        out[vid] = _get_enabled_for_video(vid)

    return {"sources": out}

@app.get("/api/attire/sources/{video_id}")
def get_attire_source(video_id: str):
    return {"video_id": video_id, "enabled": _get_enabled_for_video(video_id)}

@app.post("/api/attire/sources/{video_id}")
def set_attire_source(video_id: str, body: dict = Body(...)):
    enabled = bool(body.get("enabled", True))

    with SOURCES_LOCK:
        SOURCES_BY_VIDEO[video_id] = {"enabled": enabled}
        _save_sources_file(SOURCES_BY_VIDEO)

    return {"ok": True, "video_id": video_id, "enabled": enabled}

# ----------------------------
# API: Violation Types (global)
# ----------------------------
@app.get("/api/attire/violations")
def get_attire_violation_types():
    return {"enabled": _get_enabled_violation_map()}

@app.post("/api/attire/violations")
def set_attire_violation_types(body: dict = Body(...)):
    enabled = body.get("enabled", {})
    cleaned = _set_enabled_violation_map(enabled)
    return {"ok": True, "enabled": cleaned}

# ----------------------------
# API: Dashboard
# ----------------------------
@app.get("/api/attire/dashboard")
def attire_dashboard():
    now = int(time.time())
    since_24h = now - 24 * 3600
    since_7d  = now - 7 * 24 * 3600

    # today start (local server time)
    dt_now = datetime.fromtimestamp(now)
    dt_mid = dt_now.replace(hour=0, minute=0, second=0, microsecond=0)
    since_today = int(dt_mid.timestamp())

    with ATTIRE_EVENTS_LOCK:
        items = list(ATTIRE_EVENTS)

    def _get_vid(e: dict) -> str:
        # ✅ robust: supports older JSON schemas
        return str(
            e.get("video_id")
            or e.get("videoId")
            or e.get("source_id")
            or e.get("sourceId")
            or "unknown"
        )

    def _get_name(e: dict, vid: str) -> str:
        nm = str(e.get("video_name") or e.get("videoName") or "").strip()
        return _get_video_display_name(vid, original_name=nm, fallback=vid)

    def in_range(e, since_ts: int):
        try:
            return int(e.get("ts", 0) or 0) >= since_ts
        except Exception:
            return False

    events_24h = [e for e in items if isinstance(e, dict) and in_range(e, since_24h)]
    events_7d  = [e for e in items if isinstance(e, dict) and in_range(e, since_7d)]
    events_today = [e for e in items if isinstance(e, dict) and in_range(e, since_today)]

    def agg_by_type(evts):
        d = {}
        for e in evts:
            t = _label_title(e.get("label", ""))
            d[t] = d.get(t, 0) + 1
        return d

    type_24h = agg_by_type(events_24h)
    most_common_24h = sorted(type_24h.items(), key=lambda x: x[1], reverse=True)[0][0] if type_24h else None

    # ✅ robust hotspot aggregation (this is what fixes "no data")
    def top_rows_from_events(evts, topn: int = 5):
        cam_counts = {}
        cam_name = {}
        cam_type_counts = {}

        for e in evts:
            vid = _get_vid(e)
            cam_counts[vid] = cam_counts.get(vid, 0) + 1
            cam_name.setdefault(vid, _get_name(e, vid))

            t = _label_title(e.get("label", ""))
            cam_type_counts.setdefault(vid, {})
            cam_type_counts[vid][t] = cam_type_counts[vid].get(t, 0) + 1

        rows = []
        for vid, cnt in cam_counts.items():
            tc = cam_type_counts.get(vid, {}) or {}
            top_type = sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0] if tc else "N/A"

            risk = "low"
            if cnt >= 20: risk = "high"
            elif cnt >= 8: risk = "medium"

            rows.append({
                "video_id": vid,
                "name": cam_name.get(vid, vid),
                "count": int(cnt),
                "top_type": top_type,
                "risk": risk,
            })

        rows.sort(key=lambda r: r["count"], reverse=True)
        return rows[:topn]

    hotspot_24h = top_rows_from_events(events_24h, 5)
    hotspot_7d  = top_rows_from_events(events_7d,  5)
    worst_24h = hotspot_24h[0] if hotspot_24h else None

    # breakdown bars
    breakdown = [{"type": k, "count": int(v)} for k, v in sorted(type_24h.items(), key=lambda x: x[1], reverse=True)]

    # ✅ 24h hourly trend (for line chart)
    # returns last 24 buckets ending "now"
    hour_buckets = []
    base = (now // 3600) * 3600  # hour floor
    counts_by_hour = {base - i * 3600: 0 for i in range(23, -1, -1)}  # 24 hours

    for e in events_24h:
        ts = int(e.get("ts", 0) or 0)
        hkey = (ts // 3600) * 3600
        if hkey in counts_by_hour:
            counts_by_hour[hkey] += 1

    for hkey in sorted(counts_by_hour.keys()):
        dt = datetime.fromtimestamp(hkey)
        hour_buckets.append({
            "hour": dt.strftime("%H:00"),
            "count": int(counts_by_hour[hkey]),
        })

    # recent events (all cams)
    recent = sorted(items, key=lambda x: int((x or {}).get("ts", 0) or 0), reverse=True)[:20]
    recent = [_decorate_attire_event(e) for e in recent if isinstance(e, dict)]

    return {
        "generated_ts": now,
        "overview": {
            "violations_today": int(len(events_today)),
            "violations_24h": int(len(events_24h)),
            "most_common_24h": most_common_24h or "N/A",
            "worst_camera_24h": worst_24h,
        },
        "hotspot_24h": hotspot_24h,
        "hotspot_7d": hotspot_7d,
        "breakdown_24h": breakdown,
        "trend_24h_hourly": hour_buckets,   # ✅ NEW
        "recent_events": recent,
    }

# ----------------------------
# API: Reports
# ----------------------------
@app.get("/api/attire/reports")
def attire_reports(
    start: str = "",   # YYYY-MM-DD
    end: str = "",     # YYYY-MM-DD
    vtype: str = "All",
    status: str = "All",
    video_id: str = "",
    limit: int = 1000,
):
    start_dt = _parse_yyyy_mm_dd(start) if start else None
    end_dt = _parse_yyyy_mm_dd(end) if end else None
    if end_dt:
        end_dt = end_dt.replace(hour=23, minute=59, second=59)

    with ATTIRE_EVENTS_LOCK:
        items = list(ATTIRE_EVENTS)

    # filter
    out = []
    for e in items:
        if video_id and e.get("video_id") != video_id:
            continue

        ev_status = e.get("status", "Pending")
        if status != "All" and ev_status != status:
            continue

        ev_type = _label_title(e.get("label", ""))
        if vtype != "All" and ev_type != vtype:
            continue

        ts = int(e.get("ts", 0) or 0)
        dt = datetime.fromtimestamp(ts)
        if start_dt and dt < start_dt:
            continue
        if end_dt and dt > end_dt:
            continue

        out.append(e)

    # sort newest first + cap
    out = sorted(out, key=lambda x: int(x.get("ts", 0)), reverse=True)[:limit]
    out = [_decorate_attire_event(e) for e in out]

    # stats
    total = len(out)
    resolved = sum(1 for e in out if e.get("status") == "Resolved")
    pending = total - resolved
    rate = (resolved / total * 100.0) if total else 0.0

    type_counts = {}
    month_counts = {}  # "2024-12" -> count
    for e in out:
        t = _label_title(e.get("label", ""))
        type_counts[t] = type_counts.get(t, 0) + 1
        dt = datetime.fromtimestamp(int(e.get("ts", 0) or 0))
        key = f"{dt.year:04d}-{dt.month:02d}"
        month_counts[key] = month_counts.get(key, 0) + 1

    most_freq = None
    if type_counts:
        most_freq = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[0][0]

    # build chart arrays
    freq_chart = [{"name": k, "count": v} for k, v in sorted(type_counts.items(), key=lambda x: x[0])]
    trend_chart = [{"month": k, "violations": v} for k, v in sorted(month_counts.items(), key=lambda x: x[0])]

    return {
        "summary": {
            "total": total,
            "resolved": resolved,
            "pending": pending,
            "resolved_rate": round(rate, 1),
            "most_frequent": most_freq or "N/A",
        },
        "charts": {
            "type_frequency": freq_chart,
            "monthly_trend": trend_chart,
            "status": [
                {"name": "Resolved", "value": resolved},
                {"name": "Pending", "value": pending},
            ],
        },
        "events": out,
    }

@app.get("/api/attire/reports/export.csv")
def export_attire_csv(start: str = "", end: str = "", vtype: str = "All", status: str = "All", video_id: str = ""):
    r = attire_reports(start=start, end=end, vtype=vtype, status=status, video_id=video_id, limit=100000)
    events = r["events"]

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "violation_type", "status", "location", "detection_ts", "resolved_ts", "evidence_url", "video_id", "video_name"])

    for e in events:
        w.writerow([
            e.get("id", ""),
            _label_title(e.get("label", "")),
            e.get("status", ""),
            e.get("location", ""),
            int(e.get("ts", 0) or 0),
            e.get("resolved_ts") or "",
            e.get("evidence_url") or "",
            e.get("video_id") or "",
            e.get("video_name") or "",
        ])

    return PlainTextResponse(
        buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="attire_report.csv"'}
    )

@app.post("/api/attire/reports/export.pdf")
def export_attire_pdf(
    payload: dict = Body(default={})
):
    import io
    import base64
    from datetime import datetime

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as RLImage,
        )
    except Exception:
        raise HTTPException(status_code=501, detail="reportlab not installed")

    start = str(payload.get("start", "") or "")
    end = str(payload.get("end", "") or "")
    vtype = str(payload.get("vtype", "All") or "All")
    status = str(payload.get("status", "All") or "All")
    video_id = str(payload.get("video_id", "") or "")
    chart_image = payload.get("chart_image")

    r = attire_reports(
        start=start,
        end=end,
        vtype=vtype,
        status=status,
        video_id=video_id,
        limit=2000,
    )
    s = r["summary"]
    events = r["events"]

    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.4 * cm,
        rightMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="Attire Compliance Report",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleX",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "MetaX",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "H2X",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    cell_style = ParagraphStyle(
        "CellX",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.2,
        textColor=colors.HexColor("#0F172A"),
    )
    cell_center = ParagraphStyle(
        "CellCenterX",
        parent=cell_style,
        alignment=1,
    )
    header_left = ParagraphStyle(
        "HeaderLeftX",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=0,
    )
    cell_muted = ParagraphStyle(
        "CellMutedX",
        parent=cell_style,
        textColor=colors.HexColor("#334155"),
    )

    def P(txt: str, st=cell_style):
        return Paragraph(str(txt or ""), st)

    def _fmt_dt(ts: int) -> str:
        try:
            return datetime.fromtimestamp(int(ts or 0)).strftime("%m/%d/%Y, %I:%M:%S %p")
        except Exception:
            return "-"

    def _label_title(label: str) -> str:
        s0 = str(label or "").lower()
        if "sleeveless" in s0:
            return "Sleeveless"
        if "shorts" in s0:
            return "Shorts"
        if s0 == "slippers" or "sandal" in s0:
            return "Slippers"
        return str(label or "Unknown").replace("_", " ").title()

    def _format_location(e: dict) -> str:
        src_name = (
            e.get("video_name")
            or e.get("video_id")
            or "Unknown"
        )

        view_name = (
            e.get("view")
            or e.get("location")
            or "normal"
        )

        return f"{src_name}, {view_name}"

    story = []

    # Title
    story.append(Paragraph("Attire Compliance Report", title_style))
    story.append(
        Paragraph(
            f"Range: <b>{start or '-'}</b> → <b>{end or '-'}</b>"
            f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Type: <b>{vtype}</b>"
            f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Status: <b>{status}</b>"
            + (
                f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Source: <b>{video_id}</b>"
                if video_id else ""
            ),
            meta_style,
        )
    )

    # Summary cards
    card_label = ParagraphStyle(
        "CardLabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=1,
    )
    card_value = ParagraphStyle(
        "CardValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        alignment=1,
    )

    total = int(s.get("total", 0) or 0)
    pending = int(s.get("pending", 0) or 0)
    resolved = int(s.get("resolved", 0) or 0)
    rate = float(s.get("resolved_rate", 0) or 0)
    most_frequent = str(s.get("most_frequent", "N/A") or "N/A")

    cards = Table(
        [
            [
                P("Total Violations", card_label),
                P("Resolved Cases", card_label),
                P("Pending Cases", card_label),
                P("Resolution Rate", card_label),
            ],
            [
                P(str(total), card_value),
                P(str(resolved), card_value),
                P(str(pending), card_value),
                P(f"{rate:.1f}%", card_value),
            ],
        ],
        colWidths=[doc.width / 4.0] * 4,
        rowHeights=[16, 24],
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(cards)
    story.append(Spacer(1, 10))

    # Most frequent violation
    story.append(Paragraph("Most Frequent Violation", h2_style))
    badge = Table(
        [[P(most_frequent, ParagraphStyle(
            "BadgeVal",
            parent=cell_style,
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=colors.HexColor("#9A3412"),
            alignment=0,
        ))]],
        colWidths=[doc.width],
    )
    badge.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFEDD5")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#FDBA74")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(badge)
    story.append(Spacer(1, 12))

    # Insert ONLY charts screenshot here
    if chart_image:
        try:
            if isinstance(chart_image, str) and "," in chart_image:
                chart_image = chart_image.split(",", 1)[1]

            img_bytes = base64.b64decode(chart_image)
            img_buf = io.BytesIO(img_bytes)

            chart_pdf_img = RLImage(img_buf)
            chart_pdf_img._restrictSize(doc.width, 11.0 * cm)
            story.append(chart_pdf_img)
            story.append(Spacer(1, 12))
        except Exception as e:
            print(f"[ATTIRE PDF] failed to embed chart image: {e}")

    # Historical Violations table
    story.append(Paragraph("Historical Violations", h2_style))

    header = [
        "Violation Type",
        "Location",
        "Status",
        "Detection Date",
        "Resolved Date",
    ]
    rows = [header]

    for e in events[:200]:
        rows.append([
            _label_title(e.get("label", "")),
            _format_location(e),
            str(e.get("status", "Pending") or "Pending"),
            _fmt_dt(e.get("ts", 0)),
            _fmt_dt(e.get("resolved_ts")) if e.get("resolved_ts") else "-",
        ])

    col_fracs = [0.16, 0.38, 0.12, 0.17, 0.17]
    col_widths = [doc.width * f for f in col_fracs]

    rows[0] = [P(f"<b>{h}</b>", header_left) for h in rows[0]]

    for i in range(1, len(rows)):
        rows[i][0] = P(rows[i][0], cell_style)
        rows[i][1] = P(rows[i][1], cell_style)
        rows[i][2] = P(rows[i][2], cell_center)
        rows[i][3] = P(rows[i][3], cell_muted)
        rows[i][4] = P(rows[i][4], cell_muted)

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )

    for i in range(1, len(rows)):
        st_txt = str(events[i - 1].get("status", "Pending") or "Pending").lower()
        if "resolved" in st_txt:
            tbl_style.add("TEXTCOLOR", (2, i), (2, i), colors.HexColor("#16A34A"))
        else:
            tbl_style.add("TEXTCOLOR", (2, i), (2, i), colors.HexColor("#DC2626"))

    tbl.setStyle(tbl_style)
    story.append(tbl)

    def on_page(c, doc_):
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#64748B"))
        c.drawString(doc.leftMargin, 0.9 * cm, "SecureWatch • Attire Compliance Report")
        c.drawRightString(A4[0] - doc.rightMargin, 0.9 * cm, f"Page {doc_.page}")
        c.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

    pdf = buf.getvalue()
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="attire_report.pdf"'},
    )

# --- Webcam ---
@app.post("/api/live/webcam/start")
def start_webcam():
    global _webcam_detect_thread, _webcam_detect_running

    try:
        webcam.start()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ✅ start detector thread if not running
    if not _webcam_detect_running:
        _webcam_detect_running = True
        _webcam_detect_thread = threading.Thread(target=_webcam_detector_loop, daemon=True)
        _webcam_detect_thread.start()

    return {"ok": True}

@app.post("/api/live/webcam/stop")
def stop_webcam():
    global _webcam_detect_running, _webcam_detect_thread

    _webcam_detect_running = False
    if _webcam_detect_thread:
        try:
            _webcam_detect_thread.join(timeout=1.0)
        except:
            pass
        _webcam_detect_thread = None

    webcam.stop()

    # optional: clear cache
    with webcam_det_lock:
        webcam_det_cache.update({
            "ts": 0,
            "fps": 0.0,
            "resolution": [0, 0],
            "detections": [],
        })

    return {"ok": True}

@app.get("/api/live/webcam/stream")
def stream_webcam():
    def gen():
        while True:
            frame = webcam.get_jpeg()
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/live/webcam/detections")
def webcam_detections():
    with webcam_det_lock:
        return webcam_det_cache

@app.get("/api/live/webcam/snapshot")
def webcam_snapshot():
    try:
        jpg = webcam.get_jpeg()
        if jpg is None:
            raise HTTPException(status_code=503, detail="Webcam not ready")
        return Response(content=jpg, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webcam snapshot failed: {e}")

# ----------------------------
# API: Login and User 
# ----------------------------
@app.post("/api/auth/login")
def auth_login(body: dict = Body(...)):
    identifier = (body.get("email") or body.get("username") or "").strip()
    password = (body.get("password") or "")

    if not identifier or not password:
        raise HTTPException(status_code=400, detail="Missing username/email or password")

    ident = identifier.strip().lower()

    with USERS_LOCK:
        # try username first, then email
        u = _get_user_by_username(ident) or _get_user_by_email(ident)
        if not u:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if u.get("status") != "Active":
            raise HTTPException(status_code=403, detail="Account disabled")

        ok = _verify_password(
            password=password,
            salt_b64=str(u.get("pw_salt") or ""),
            hash_b64=str(u.get("pw_hash") or ""),
            iters=int(u.get("pw_iters") or PASSWORD_ITERS),
        )
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = _issue_token(str(u.get("id")))
        return {"token": token, "user": _sanitize_user(u)}

@app.post("/api/auth/logout")
def auth_logout(user: dict = Depends(get_current_user), authorization: str = Header(default="")):
    token = _get_token_from_auth(authorization)
    if not token:
        return {"ok": True}
    with SESSIONS_LOCK:
        SESSIONS.pop(token, None)
        _save_json_file(SESSIONS_PATH, SESSIONS)
    return {"ok": True}

@app.get("/api/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    return {"user": _sanitize_user(user)}

@app.get("/api/users")
def list_users(admin: dict = Depends(require_admin)):
    with USERS_LOCK:
        items = [_sanitize_user(u) for u in USERS]
    # newest first
    items.sort(key=lambda x: x.get("createdAt") or "", reverse=True)
    return {"users": items}

@app.post("/api/users")
def create_user(body: dict = Body(...), admin: dict = Depends(require_admin)):
    name = (body.get("name") or "").strip()
    username = (body.get("username") or "").strip().lower()
    email = (body.get("email") or "").strip().lower()
    role = (body.get("role") or "Viewer").strip()
    status = (body.get("status") or "Active").strip()
    password = (body.get("password") or "")

    if not name or not email:
        raise HTTPException(status_code=400, detail="Missing name/email")
    if not username:
        raise HTTPException(status_code=400, detail="Missing username")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    allowed_roles = {"Admin", "Security", "Staff", "Viewer"}
    allowed_status = {"Active", "Disabled"}
    if role not in allowed_roles:
        raise HTTPException(status_code=400, detail="Invalid role")
    if status not in allowed_status:
        raise HTTPException(status_code=400, detail="Invalid status")

    with USERS_LOCK:
        if _get_user_by_email(email):
            raise HTTPException(status_code=409, detail="Email already exists")
        if _get_user_by_username(username):
            raise HTTPException(status_code=409, detail="Username already exists")
    
        pw = _pbkdf2_hash(password)
        u = {
            "id": _uid("u-"),
            "username": username,
            "name": name,
            "email": email,
            "role": role,
            "status": status,
            "createdAt": datetime.now().isoformat(),
            "pw_salt": pw["salt"],
            "pw_hash": pw["hash"],
            "pw_iters": pw["iters"],
        }
        USERS.append(u)
        _save_json_file(USERS_PATH, USERS)

    return {"ok": True, "user": _sanitize_user(u)}

@app.put("/api/users/{user_id}")
def update_user(user_id: str, body: dict = Body(...), admin: dict = Depends(require_admin)):
    with USERS_LOCK:
        u = _get_user_by_id(user_id)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")

        # fields
        if "name" in body:
            u["name"] = (body.get("name") or "").strip()
        if "username" in body:
            new_un = (body.get("username") or "").strip().lower()
            if not new_un:
                raise HTTPException(status_code=400, detail="Username cannot be empty")
            if new_un != (u.get("username") or "").lower():
                if _get_user_by_username(new_un):
                    raise HTTPException(status_code=409, detail="Username already exists")
                u["username"] = new_un
        if "email" in body:
            new_email = (body.get("email") or "").strip().lower()
            if new_email and new_email != (u.get("email") or "").lower():
                if _get_user_by_email(new_email):
                    raise HTTPException(status_code=409, detail="Email already exists")
                u["email"] = new_email
        if "role" in body:
            role = (body.get("role") or "Viewer").strip()
            if role not in {"Admin", "Security", "Staff", "Viewer"}:
                raise HTTPException(status_code=400, detail="Invalid role")
            # prevent removing last admin
            if u.get("role") == "Admin" and role != "Admin" and _admin_count() <= 1:
                raise HTTPException(status_code=400, detail="Cannot remove last Admin")
            u["role"] = role

        if "status" in body:
            status = (body.get("status") or "Active").strip()
            if status not in {"Active", "Disabled"}:
                raise HTTPException(status_code=400, detail="Invalid status")
            u["status"] = status

        # optional password reset
        pw_new = (body.get("password") or "").strip()
        if pw_new:
            if len(pw_new) < 6:
                raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
            pw = _pbkdf2_hash(pw_new)
            u["pw_salt"] = pw["salt"]
            u["pw_hash"] = pw["hash"]
            u["pw_iters"] = pw["iters"]

        _save_json_file(USERS_PATH, USERS)

        return {"ok": True, "user": _sanitize_user(u)}

@app.delete("/api/users/{user_id}")
def delete_user(user_id: str, admin: dict = Depends(require_admin)):
    with USERS_LOCK:
        u = _get_user_by_id(user_id)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")

        if u.get("role") == "Admin" and _admin_count() <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete last Admin")

        USERS[:] = [x for x in USERS if str(x.get("id")) != str(user_id)]
        _save_json_file(USERS_PATH, USERS)

    # also revoke sessions for that user
    with SESSIONS_LOCK:
        dead = [tok for tok, s in (SESSIONS or {}).items() if (s or {}).get("user_id") == user_id]
        for tok in dead:
            SESSIONS.pop(tok, None)
        if dead:
            _save_json_file(SESSIONS_PATH, SESSIONS)

    return {"ok": True}

# ----------------------------
# API: Notifications
# ----------------------------
@app.get("/api/attire/notifications")
def get_attire_notifications_cfg(user=Depends(get_current_user)):
    with NOTIF_LOCK:
        return JSONResponse(
            content={"config": ATTIRE_NOTIF_CFG},
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

@app.post("/api/attire/notifications")
def set_attire_notifications_cfg(payload: dict = Body(...), user=Depends(get_current_user)):
    # basic sanitize
    enabled = bool(payload.get("enabled", True))
    cooldown = int(payload.get("cooldown_sec", 30))
    toast_sec = int(payload.get("toast_sec", 6))
    play_sound = bool(payload.get("play_sound", False))

    cooldown = max(0, min(3600, cooldown))
    toast_sec = max(1, min(30, toast_sec))

    with NOTIF_LOCK:
        ATTIRE_NOTIF_CFG.update({
            "enabled": enabled,
            "cooldown_sec": cooldown,
            "toast_sec": toast_sec,
            "play_sound": play_sound,
        })
        _save_notif_file(ATTIRE_NOTIF_CFG)

    return {"ok": True, "config": ATTIRE_NOTIF_CFG}

# --- SSE ---
@app.get("/api/attire/notifications/stream")
async def attire_notifications_stream(request: Request, token: str = ""):
    user = get_current_user_from_token(token)
    if not user:
        print("[NOTIF] SSE rejected: invalid token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    print("[NOTIF] SSE connect attempt by user:", user.get("username") if isinstance(user, dict) else user)

    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    loop = asyncio.get_running_loop()
    sub = (loop, q)

    with ATTIRE_NOTIF_SUBS_LOCK:
        ATTIRE_NOTIF_SUBS.add(sub)
        print("[NOTIF] SSE subscriber added. total =", len(ATTIRE_NOTIF_SUBS))

    async def gen():
        try:
            with NOTIF_LOCK:
                cfg = dict(ATTIRE_NOTIF_CFG)

            # send initial config event
            yield f"event: config\ndata: {json.dumps(cfg)}\n\n"

            while True:
                if await request.is_disconnected():
                    print("[NOTIF] SSE client disconnected")
                    break

                try:
                    item = await asyncio.wait_for(q.get(), timeout=10.0)
                    print("[NOTIF] SSE sending plain message:", item)
                    yield f"data: {json.dumps(item)}\n\n"

                except asyncio.TimeoutError:
                    print("[NOTIF] SSE sending ping")
                    yield ": ping\n\n"

        finally:
            with ATTIRE_NOTIF_SUBS_LOCK:
                ATTIRE_NOTIF_SUBS.discard(sub)
                print("[NOTIF] SSE subscriber removed. total =", len(ATTIRE_NOTIF_SUBS))

    origin = request.headers.get("origin", "")
    sse_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    if origin in ALLOWED_ORIGINS:
        sse_headers["Access-Control-Allow-Origin"] = origin
        sse_headers["Access-Control-Allow-Credentials"] = "true"
        sse_headers["Access-Control-Allow-Methods"] = "*"
        sse_headers["Access-Control-Allow-Headers"] = "*"
        sse_headers["Vary"] = "Origin"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers=sse_headers,
    )