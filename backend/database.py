import databases
import sqlalchemy
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel

DATABASE_URL = "sqlite:///./anpr.db"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# ── Tables ────────────────────────────────────────────────────────────────────

detections = sqlalchemy.Table(
    "detections",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("uuid", sqlalchemy.String(36), index=True),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, default=lambda: datetime.now(timezone.utc)),
    sqlalchemy.Column("plate_text", sqlalchemy.String(20)),
    sqlalchemy.Column("confidence", sqlalchemy.Float),
    sqlalchemy.Column("state", sqlalchemy.String(50), nullable=True),
    sqlalchemy.Column("city", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("series_age", sqlalchemy.String(50), nullable=True),
    sqlalchemy.Column("owner", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("make", sqlalchemy.String(100), nullable=True),
    sqlalchemy.Column("speed", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("environment", sqlalchemy.String(30), nullable=True),
    sqlalchemy.Column("camera_id", sqlalchemy.String(20), default="CAM-01"),
    sqlalchemy.Column("vehicle_type", sqlalchemy.String(30), nullable=True),
    sqlalchemy.Column("vehicle_color", sqlalchemy.String(30), nullable=True),
    sqlalchemy.Column("source", sqlalchemy.String(100)),
    sqlalchemy.Column("total_ms", sqlalchemy.Float),
    sqlalchemy.Column("crop_image", sqlalchemy.Text), # Storing as base64 for simplicity in this project
)

# ── Pydantic Models ───────────────────────────────────────────────────────────

class DetectionRecord(BaseModel):
    id: int
    uuid: str
    timestamp: datetime
    plate_text: str
    confidence: float
    state: Optional[str] = None
    city: Optional[str] = None
    series_age: Optional[str] = None
    owner: Optional[str] = None
    make: Optional[str] = None
    speed: Optional[float] = None
    environment: Optional[str] = None
    camera_id: str = "CAM-01"
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    source: str
    total_ms: float
    crop_image: str

# ── DB Helpers ────────────────────────────────────────────────────────────────

engine = sqlalchemy.create_engine(DATABASE_URL)

async def init_db():
    await database.connect()
    # Create tables if they don't exist
    engine = sqlalchemy.create_engine(DATABASE_URL)
    metadata.create_all(engine)

async def close_db():
    await database.disconnect()

async def save_detection(
    uuid: str, 
    plate_text: str, 
    confidence: float, 
    source: str, 
    total_ms: float, 
    crop_image: str,
    state: Optional[str] = None, 
    city: Optional[str] = None,
    series_age: Optional[str] = None,
    owner: Optional[str] = None,
    make: Optional[str] = None,
    speed: Optional[float] = None,
    environment: Optional[str] = None,
    camera_id: str = "CAM-01",
    vehicle_type: Optional[str] = None,
    vehicle_color: Optional[str] = None,
):
    query = detections.insert().values(
        uuid=uuid,
        plate_text=plate_text,
        confidence=confidence,
        state=state,
        city=city,
        series_age=series_age,
        owner=owner,
        make=make,
        speed=speed,
        environment=environment,
        camera_id=camera_id,
        vehicle_type=vehicle_type,
        vehicle_color=vehicle_color,
        source=source,
        total_ms=total_ms,
        crop_image=crop_image
    )
    return await database.execute(query)

async def get_recent_detections(limit: int = 20, state: Optional[str] = None):
    query = detections.select()
    if state:
        query = query.where(detections.c.state.ilike(f"%{state}%"))
    query = query.order_by(detections.c.timestamp.desc()).limit(limit)
    return await database.fetch_all(query)

async def get_stats():
    # Total detections
    total_query = "SELECT COUNT(*) FROM detections"
    total = await database.fetch_val(total_query)
    
    # State distribution
    state_query = "SELECT state, COUNT(*) as count FROM detections WHERE state IS NOT NULL GROUP BY state ORDER BY count DESC LIMIT 5"
    states = await database.fetch_all(state_query)
    
    # Detections today
    today_query = "SELECT COUNT(*) FROM detections WHERE date(timestamp) = date('now')"
    today = await database.fetch_val(today_query)
    
    return {
        "total": total,
        "today": today,
        "states": [{"state": r["state"], "count": r["count"]} for r in states]
    }
