"""
FastAPI Backend — ANPR Application v2
Endpoints:
  GET  /health              — liveness + model status
  GET  /info                — detector component info
  POST /detect              — upload image → plate detection
  POST /detect/url          — detect from image URL
  GET  /history             — recent detection log (last N)
  DELETE /history           — clear history
  WS   /ws/detect           — WebSocket stream for webcam frames
"""
import io
import sys
import time
import uuid
import logging
import asyncio
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator

# ── Dynamic skill import ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import backend.database as db

import importlib.util as _ilu
_skill_path = _PROJECT_ROOT / ".agents" / "skills" / "plate_detector.py"
_spec = _ilu.spec_from_file_location("plate_detector", _skill_path)
_mod  = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

detect_plates     = _mod.detect_plates
annotate_image    = _mod.annotate_image
ndarray_to_base64 = _mod.ndarray_to_base64
preprocess_image  = _mod.preprocess_image
get_detector_info = _mod.get_detector_info

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── In-memory history ─────────────────────────────────────────────────────────
_MAX_HISTORY = 50
_history: deque = deque(maxlen=_MAX_HISTORY)

app = FastAPI(
    title      = "ANPR Vision API",
    description= "Automatic Number Plate Recognition — Indian vehicles (v2)",
    version    = "2.0.0",
    docs_url   = "/docs",
    redoc_url  = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
    expose_headers = ["*"],
)

@app.on_event("startup")
async def startup():
    await db.init_db()
    
    # Inject mock data
    from backend.mock_data import inject_mock_data_if_empty
    await inject_mock_data_if_empty()

@app.on_event("shutdown")
async def shutdown():
    await db.close_db()


# ── Pydantic models ───────────────────────────────────────────────────────────

class PlateResult(BaseModel):
    text:        str
    confidence:  float
    bbox:        List[int]   # [x, y, w, h]
    crop_image:  str         # base64 data-URI
    state:       Optional[str] = None
    city:        Optional[str] = None
    series_age:  Optional[str] = None
    owner:       Optional[str] = None
    make:        Optional[str] = None
    speed:       Optional[float] = 0.0
    environment: Optional[str] = "Daylight"
    camera_id:   str = "CAM-01"
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    method:      str = "unknown"
    process_ms:  float = 0.0


class DetectResponse(BaseModel):
    id:              str
    timestamp:       str
    plates:          List[PlateResult]
    annotated_image: str         # base64 data-URI
    total_found:     int
    source:          str = "upload"
    total_ms:        float = 0.0


class UrlDetectRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class HistoryEntry(BaseModel):
    id:          str
    timestamp:   str
    plates:      List[PlateResult]
    total_found: int
    source:      str
    total_ms:    float


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
    count:   int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes — unsupported format?")
    return img


def _process(img: np.ndarray, source: str = "upload") -> DetectResponse:
    """Core processing: detection → annotate → build response."""
    t0         = time.perf_counter()
    detections = detect_plates(img)
    annotated  = annotate_image(img, detections)
    total_ms   = round((time.perf_counter() - t0) * 1000, 1)

    plates = [
        PlateResult(
            text          = d["text"],
            confidence    = round(d["confidence"], 4),
            bbox          = list(d["bbox"]),
            crop_image    = ndarray_to_base64(d["crop"]),
            state         = d.get("state"),
            city          = d.get("city"),
            series_age    = d.get("series_age"),
            owner         = d.get("telemetry", {}).get("owner"),
            make          = d.get("telemetry", {}).get("make"),
            speed         = d.get("telemetry", {}).get("speed_est", 0.0),
            environment   = d.get("environment", "Daylight"),
            camera_id     = "CAM-01", # could be passed from headers
            vehicle_type  = d.get("vehicle_type"),
            vehicle_color = d.get("vehicle_color"),
            method        = d.get("method", "unknown"),
            process_ms    = d.get("process_ms", 0.0),
        )
        for d in detections
    ]

    rid = str(uuid.uuid4())[:8]
    ts  = datetime.now(timezone.utc).isoformat()

    resp = DetectResponse(
        id              = rid,
        timestamp       = ts,
        plates          = plates,
        annotated_image = ndarray_to_base64(annotated),
        total_found     = len(plates),
        source          = source,
        total_ms        = total_ms,
    )

    # Persist to database (async background task would be better, but doing it here for simplicity)
    import asyncio
    for p in plates:
        asyncio.create_task(db.save_detection(
            uuid=rid,
            plate_text=p.text,
            confidence=p.confidence,
            source=source,
            total_ms=total_ms,
            crop_image=p.crop_image,
            state=p.state,
            city=p.city,
            series_age=p.series_age,
            owner=p.owner,
            make=p.make,
            speed=p.speed,
            environment=p.environment,
            camera_id=p.camera_id,
            vehicle_type=p.vehicle_type,
            vehicle_color=p.vehicle_color
        ))

    return resp


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    info = {}
    try:
        info = get_detector_info()
    except Exception:
        pass
    return {
        "status":   "ok",
        "service":  "ANPR Vision API v2",
        "detector": info,
    }


@app.get("/info", tags=["System"])
def model_info():
    return get_detector_info()


@app.post("/detect", response_model=DetectResponse, tags=["Detection"])
async def detect_upload(file: UploadFile = File(...)):
    """Upload a car image → get detected plate texts + annotated image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files accepted")
    data = await file.read()
    try:
        img = _bytes_to_bgr(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _process(img, source=f"upload:{file.filename}")


@app.post("/detect/url", response_model=DetectResponse, tags=["Detection"])
async def detect_url(body: UrlDetectRequest):
    """Detect plates from a publicly accessible image URL."""
    try:
        req = urllib.request.Request(
            body.url,
            headers={"User-Agent": "ANPR-Vision/2.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")
    try:
        img = _bytes_to_bgr(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _process(img, source=f"url:{body.url[:60]}")


@app.get("/history", response_model=HistoryResponse, tags=["History"])
async def get_history(limit: int = Query(20, ge=1, le=100), state: Optional[str] = Query(None)):
    """Return the last N detection results from DB."""
    records = await db.get_recent_detections(limit, state)
    # Group by UUID if needed, but for now just return flat list or adapt to frontend
    # Frontend expects HistoryEntry which has a list of plates. 
    # Since DB stores 1 row per plate, we might need to group them.
    
    entries = []
    # Simple grouping by UUID
    groups = {}
    for r in records:
        uid = r["uuid"]
        if uid not in groups:
            groups[uid] = {
                "id": uid,
                "timestamp": r["timestamp"].isoformat(),
                "plates": [],
                "total_found": 0,
                "source": r["source"],
                "total_ms": r["total_ms"]
            }
        groups[uid]["plates"].append(PlateResult(
            text=r["plate_text"],
            confidence=r["confidence"],
            bbox=[0,0,0,0], 
            crop_image=r["crop_image"],
            state=r["state"],
            city=r["city"],
            series_age=r["series_age"],
            owner=r["owner"],
            make=r["make"],
            speed=r["speed"],
            environment=r["environment"],
            camera_id=r["camera_id"],
            vehicle_type=r["vehicle_type"],
            vehicle_color=r["vehicle_color"]
        ))
        groups[uid]["total_found"] += 1
    
    return HistoryResponse(entries=list(groups.values()), count=len(groups))


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get aggregate statistics for the dashboard."""
    return await db.get_stats()


@app.get("/history/export", tags=["History"])
async def export_history():
    """Export detection logs as CSV."""
    import csv
    from fastapi.responses import StreamingResponse
    from io import StringIO
    
    records = await db.get_recent_detections(limit=1000)
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Plate Text", "Confidence", "State", "City", "Source", "Process Time (ms)"])
    
    for r in records:
        writer.writerow([
            r["uuid"], 
            r["timestamp"].isoformat(), 
            r["plate_text"], 
            f"{r['confidence']:.4f}", 
            r["state"] or "Unknown", 
            r["city"] or "Unknown",
            r["source"], 
            r["total_ms"]
        ])
    
    output.seek(0)
    return StreamingResponse(
        output, 
        media_type="text/csv", 
        headers={"Content-Disposition": "attachment; filename=anpr_audit_log.csv"}
    )
@app.delete("/history", tags=["History"])
async def clear_history():
    """Clear all detection history."""
    query = db.detections.delete()
    await db.database.execute(query)
    return {"status": "cleared"}


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """Real-time webcam plate detection via WebSocket (send JPEG bytes → receive JSON)."""
    await websocket.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                img    = _bytes_to_bgr(data)
                result = _process(img, source="webcam")
                await websocket.send_text(result.model_dump_json())
            except Exception as e:
                logger.error(f"WS frame error: {e}")
                await websocket.send_json({
                    "error": str(e), "plates": [], "total_found": 0,
                    "id": "", "timestamp": "", "annotated_image": "",
                    "source": "webcam", "total_ms": 0
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
