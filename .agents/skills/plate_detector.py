"""
Skill: ANPR Plate Detector  (v2 — Enhanced)
Multi-stage detection pipeline:
  1. OpenCV Haar Cascade (fast, broad detection)
  2. Contour/morphology-based fallback (handles non-standard plates)
  3. Canny edge + region candidate fallback (last resort)

OCR pipeline:
  - EasyOCR with adaptive pre-processing
  - CLAHE → Otsu binarisation → denoising
  - Indian state-code lookup from plate prefix
  - Confidence-weighted text selection

Compatible with:
  - alihassanml/car-number-plate  (Pascal-VOC XML, generic plates)
  - suprabhosaha/indian-vehicle-number-plate-detection-dataset
  - Keras/TF InceptionResNetV2 model output (bounding box regression)
"""
import cv2
import numpy as np
import easyocr
import re
import base64
import time
import logging
from pathlib import Path
import json
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

# ── Indian state / UT codes mapping ─────────────────────────────────────────
INDIAN_STATE_CODES: Dict[str, str] = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh",  "AR": "Arunachal Pradesh",
    "AS": "Assam",             "BR": "Bihar",            "CH": "Chandigarh",
    "CG": "Chhattisgarh",      "DD": "Daman & Diu",      "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli", "GA": "Goa",           "GJ": "Gujarat",
    "HP": "Himachal Pradesh",  "HR": "Haryana",          "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",   "KA": "Karnataka",        "KL": "Kerala",
    "LA": "Ladakh",            "LD": "Lakshadweep",      "MH": "Maharashtra",
    "ML": "Meghalaya",         "MN": "Manipur",          "MP": "Madhya Pradesh",
    "MZ": "Mizoram",           "NL": "Nagaland",         "OD": "Odisha",
    "PB": "Punjab",            "PY": "Puducherry",       "RJ": "Rajasthan",
    "SK": "Sikkim",            "TN": "Tamil Nadu",       "TR": "Tripura",
    "TS": "Telangana",         "UK": "Uttarakhand",      "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# ── Indian plate regex patterns (strict → relaxed order) ────────────────────
INDIAN_PLATE_PATTERNS = [
    (r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}",  "standard"),        # MH12AB1234
    (r"[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{4}",   "single-letter"),   # KA05E1234
    (r"[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]{4}",   "three-letter"),    # DL01CDF1234 (rare)
    (r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}",       "bh-series"),       # 22BH1234AB
    (r"[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{3,4}", "relaxed"),       # fallback
]

# ── Model / cascade paths ────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent   # project root
CASCADE_PATH      = _ROOT / "models" / "haarcascade_russian_plate_number.xml"
KERAS_MODEL_PATH  = _ROOT / "models" / "anpr_model.keras"
H5_MODEL_PATH     = _ROOT / "models" / "anpr_model.h5"
RTO_DATA_PATH     = _ROOT / "data" / "rto_codes.json"
_OPENCV_CASCADE   = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

# ── Singletons ───────────────────────────────────────────────────────────────
_cascade:  Optional[cv2.CascadeClassifier] = None
_reader:   Optional[easyocr.Reader]        = None
_keras_model = None   # lazy-loaded TF/Keras model
_yolo_model  = None   # lazy-loaded Ultralytics / YOLO model
_rto_db:   Optional[Dict]                  = None


# ── Loaders ──────────────────────────────────────────────────────────────────

def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is not None:
        return _cascade
    path = str(CASCADE_PATH) if CASCADE_PATH.exists() else _OPENCV_CASCADE
    _cascade = cv2.CascadeClassifier(path)
    if _cascade.empty():
        raise RuntimeError(f"Could not load Haar Cascade from: {path}")
    logger.info(f"Cascade loaded: {path}")
    return _cascade


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        logger.info("Initialising EasyOCR (first call may take ~10 s)…")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR ready.")
    return _reader


def _get_keras_model():
    """Lazy-load the Keras bounding-box regressor if available."""
    global _keras_model
    if _keras_model is not None:
        return _keras_model
    for p in [KERAS_MODEL_PATH, H5_MODEL_PATH]:
        if p.exists():
            try:
                import tensorflow as tf
                _keras_model = tf.keras.models.load_model(str(p))
                logger.info(f"Keras model loaded: {p}")
                return _keras_model
            except Exception as e:
                logger.warning(f"Could not load Keras model from {p}: {e}")
    return None


def _get_yolo_model():
    """Lazy-load YOLO model for high-speed modern detection."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        _yolo_model = YOLO('yolov8n.pt') 
        logger.info("YOLO Engine Attached.")
        return _yolo_model
    except Exception:
        return None


def _get_rto_db() -> Dict:
    """Lazy-load the RTO codes JSON database."""
    global _rto_db
    if _rto_db is not None:
        return _rto_db
    if RTO_DATA_PATH.exists():
        try:
            with open(RTO_DATA_PATH, "r", encoding="utf-8") as f:
                _rto_db = json.load(f)
            logger.info(f"RTO Database loaded: {len(_rto_db)} states.")
        except Exception as e:
            logger.warning(f"Failed to load RTO Database: {e}")
            _rto_db = {}
    else:
        _rto_db = {}
    return _rto_db


# ── Image utilities ───────────────────────────────────────────────────────────

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Normalise channels to BGR."""
    if img is None:
        raise ValueError("preprocess_image received None")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _deskew_plate(img: np.ndarray) -> np.ndarray:
    """Detect rotation angle and deskew the plate image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < ang < 45:
                angles.append(ang)
        if angles:
            angle = np.median(angles)
    
    if abs(angle) > 0.5:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img


def _enhance_plate(img: np.ndarray) -> np.ndarray:
    """
    Multi-step plate crop enhancement for OCR:
      deskew → upscale → grayscale → sharpening → CLAHE → Otsu threshold
    """
    # 1. Deskew
    img = _deskew_plate(img)
    
    h, w = img.shape[:2]
    # 2. Upscale small crops
    target_h = 100
    scale = target_h / h
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Sharpening
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel_sharp)

    # 4. Global Denoising / Smoothing
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 5. Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    
    # 6. Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ── Detection methods ─────────────────────────────────────────────────────────

def _detect_with_cascade(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Return list of (x,y,w,h) from Haar Cascade."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cascade = _get_cascade()
    plates = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=3,
        minSize=(50, 15),
        maxSize=(500, 200),
    )
    if len(plates) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in plates]


def _detect_with_contours(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Morphology-based contour detection.
    Works well for white/yellow plates on dark backgrounds.
    """
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    edged     = cv2.Canny(blurred, 30, 200)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed    = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    ih, iw = img.shape[:2]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        area_frac = (w * h) / (iw * ih)
        # Indian plates are roughly 4:1 to 6:1 aspect ratio
        if 2.5 < aspect < 8.0 and 0.005 < area_frac < 0.35:
            candidates.append((x, y, w, h))
    return candidates


def _detect_with_keras(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Legacy InceptionResNetV2 bounding-box regressor."""
    model = _get_keras_model()
    if model is None: return []
    try:
        ih, iw = img.shape[:2]
        resized = cv2.resize(img, (224, 224)) / 255.0
        inp     = np.expand_dims(resized.astype("float32"), 0)
        pred    = model.predict(inp, verbose=0)[0]
        return [(int(pred[0]*iw), int(pred[1]*ih), int((pred[2]-pred[0])*iw), int((pred[3]-pred[1])*ih))]
    except Exception: return []


def _detect_with_yolo(img: np.ndarray) -> List[Dict[str, Any]]:
    """YOLOv8 modern detection layer: returns [bbox, class, confidence]."""
    model = _get_yolo_model()
    if model is None: return []
    try:
        results = model(img, classes=[2, 3, 5, 7], verbose=False)
        boxes = []
        class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                boxes.append({
                    "bbox": (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                    "type": class_map.get(cls, "vehicle"),
                    "v_conf": conf
                })
        return boxes
    except Exception: return []


def _detect_vehicle_color(img: np.ndarray, bbox: Tuple[int,int,int,int]) -> str:
    """Sample pixels inside bbox to guess vehicle color."""
    x, y, w, h = bbox
    # Sample a small crop from the center-top of the vehicle bbox
    cx, cy = x + w//2, y + h//4
    sample_size = 10
    crop = img[max(0, cy-sample_size):min(img.shape[0], cy+sample_size), 
               max(0, cx-sample_size):min(img.shape[1], cx+sample_size)]
    if crop.size == 0: return "unknown"
    
    avg_color = np.mean(crop, axis=(0, 1)) # BGR
    # Simple color heuristic
    b, g, r = avg_color
    if r > 200 and g > 200 and b > 200: return "white"
    if r < 50 and g < 50 and b < 50: return "black"
    if r > 150 and g < 100 and b < 100: return "red"
    if b > 150 and r < 100 and g < 100: return "blue"
    if r > 150 and g > 150 and b < 100: return "yellow"
    if abs(r-g) < 20 and abs(g-b) < 20: return "gray"
    return "other"


def _nms_boxes(boxes: List[Tuple], threshold: float = 0.4) -> List[Tuple]:
    """Simple IoU-based NMS to remove duplicate detections."""
    if len(boxes) <= 1:
        return boxes
    arr = np.array([[x, y, x+w, y+h] for x,y,w,h in boxes], dtype=float)
    areas = (arr[:,2]-arr[:,0]) * (arr[:,3]-arr[:,1])
    order = areas.argsort()[::-1]
    keep  = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(arr[i,0], arr[order[1:],0])
        yy1 = np.maximum(arr[i,1], arr[order[1:],1])
        xx2 = np.minimum(arr[i,2], arr[order[1:],2])
        yy2 = np.minimum(arr[i,3], arr[order[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < threshold]
    return [boxes[k] for k in keep]


# ── OCR ───────────────────────────────────────────────────────────────────────

def _ocr_region(region: np.ndarray) -> Tuple[str, float, List[dict]]:
    """
    Run EasyOCR on *region* (BGR).
    Returns (cleaned_text, confidence, raw_fragments).
    """
    reader   = _get_reader()
    enhanced = _enhance_plate(region)

    # Try both enhanced and colour input; pick the better result
    results_e = reader.readtext(enhanced,  detail=1, paragraph=False)
    results_c = reader.readtext(region,    detail=1, paragraph=False)

    def _score(r): return sum(x[2] for x in r) / max(len(r), 1) if r else 0
    raw = results_e if _score(results_e) >= _score(results_c) else results_c

    if not raw:
        return "", 0.0, []

    fragments = [{"text": r[1], "confidence": float(r[2])} for r in raw]
    combined  = "".join(r[1] for r in raw).upper().replace(" ", "")
    cleaned   = _clean_plate_text(combined)
    avg_conf  = float(np.mean([r[2] for r in raw]))
    return cleaned, avg_conf, fragments


def _clean_plate_text(text: str) -> str:
    """
    Intelligent Indian plate parsing.
    Format: [State:2][City:2][Letters:1-3][Digits:4]
    """
    text = re.sub(r"[^A-Z0-9]", "", text)
    if not text: return ""

    # Common character confusions based on position
    # Indian plates: XX 00 XX 0000
    # Pos 0,1: Alphabetic
    # Pos 2,3: Digits
    # Pos 4,5: Alphabetic
    # Pos 6,7,8,9: Digits
    
    chars = list(text)
    
    def to_digit(c):
        mapping = {"O": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
        return mapping.get(c, c) if c.isalpha() else c
        
    def to_alpha(c):
        mapping = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
        return mapping.get(c, c) if c.isdigit() else c

    # Only apply if it looks like a standard length plate
    if 8 <= len(chars) <= 10:
        # State Code
        chars[0] = to_alpha(chars[0])
        chars[1] = to_alpha(chars[1])
        # District Code
        chars[2] = to_digit(chars[2])
        chars[3] = to_digit(chars[3])
        # Last 4 digits
        for i in range(len(chars)-4, len(chars)):
            chars[i] = to_digit(chars[i])
            
    text = "".join(chars)

    for pattern, _ in INDIAN_PLATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(0)
    return text


def _city_from_plate(text: str) -> Optional[str]:
    """Return the RTO city/district name from the first 4 characters (e.g., MH12)."""
    if len(text) < 4:
        return None
    
    state_code = text[:2]
    rto_code   = text[2:4]
    
    db = _get_rto_db()
    state_data = db.get(state_code)
    if state_data:
        return state_data.get(rto_code)
    return None


def _estimate_series_age(text: str) -> str:
    """Heuristic to guess if vehicle is new or old based on series letters."""
    # BH series is 2021+
    if "BH" in text:
        return "Brand New (BH-Series)"
    
    # Standard format: XX 00 [AA] 0000
    # Higher letters usually mean newer series
    m = re.search(r"[A-Z]{2}[0-9]{2}([A-Z]{1,2})", text)
    if m:
        series = m.group(1)
        if len(series) == 1:
            return "Legacy (Pre-2010)"
        if series[0] < 'M':
            return "Standard (2010-2018)"
        return "Modern (2019+)"
    
    return "Unknown"


def _detect_environment(img: np.ndarray) -> str:
    """Guess environment (Day/Night) based on average brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return "Daylight" if avg_brightness > 80 else "Night/Low Light"


def _mock_registration_data(text: str) -> Dict[str, Any]:
    """Generate fake registration details for UI demo."""
    # Deterministic-ish fake data based on plate text hash
    h = hash(text)
    owners = ["Amit Sharma", "Priya Patel", "Rahul Verma", "Sneha Gupta", "Vikram Singh", "Anjali Rao"]
    makes = ["Maruti Suzuki", "Hyundai", "Tata Motors", "Mahindra", "Kia", "Toyota"]
    fuels = ["Petrol", "Diesel", "CNG", "Electric"]
    return {
        "owner": owners[h % len(owners)],
        "make": makes[(h >> 2) % len(makes)],
        "fuel": fuels[(h >> 3) % len(fuels)],
        "cc": (h % 1500) + 800,
        "year": (h % 15) + 2010,
        "insurance_valid": (h % 10) > 2,
        "puc_valid": (h % 10) > 1,
        "stolen_record": (h % 100) == 7,
        "speed_est": (h % 40) + 20 # 20-60 km/h
    }


def _state_from_plate(text: str) -> Optional[str]:
    """Return the Indian state name from the 2-letter plate prefix, or None."""
    if len(text) >= 2:
        code = text[:2]
        return INDIAN_STATE_CODES.get(code)
    return None


# ── Main public API ───────────────────────────────────────────────────────────

def detect_plates(img: np.ndarray) -> List[Dict[str, Any]]:
    """
    Multi-stage plate detection + OCR pipeline.

    Returns a list of result dicts:
        bbox        – (x, y, w, h) pixels
        crop        – BGR ndarray of plate crop
        text        – cleaned plate text
        confidence  – float 0–1
        state       – Indian state name or None
        city        – RTO city/district name or None
        series_age  – Estimated age category
        environment – Daylight/Night
        telemetry   – Mock registration data
        method      – detection method used
        fragments   – raw OCR fragments
        process_ms  – time taken for this detection
    """
    t0  = time.perf_counter()
    img = preprocess_image(img)

    # ── 0. YOLO (Modern Attachment) ──────────────────────────────────────────    # Stage 0: YOLO (Premium AI Observer)
    yolo_results = _detect_with_yolo(img)
    boxes = [y["bbox"] for y in yolo_results]
    method = "yolo_v8_ai"

    # Fallback to Haar for plate-specific crops if YOLO only found vehicles
    if not boxes:
        boxes = _detect_with_cascade(img)
        yolo_results = [{"bbox": b, "type": "vehicle", "v_conf": 0.5} for b in boxes]
        method = "haar_cascade"

    # ── 3. Contour-based fallback ─────────────────────────────────────────────
    if not boxes:
        boxes  = _detect_with_contours(img)
        yolo_results = [{"bbox": b, "type": "vehicle", "v_conf": 0.5} for b in boxes]
        method = "contour"

    # NMS to remove overlapping detections
    boxes = _nms_boxes(boxes)

    results   = []
    ih, iw    = img.shape[:2]

    if not boxes:
        # Full-image OCR fallback
        logger.info("All detection methods failed — running full-image OCR")
        text, conf, frags = _ocr_region(img)
        if text:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            results.append({
                "bbox": (0, 0, iw, ih),
                "crop": img.copy(),
                "text": text,
                "confidence": conf,
                "state": _state_from_plate(text),
                "city": _city_from_plate(text),
                "series_age": _estimate_series_age(text),
                "environment": _detect_environment(img),
                "telemetry": _mock_registration_data(text),
                "method": "fullimage_ocr",
                "fragments": frags,
                "process_ms": ms,
            })
        return results

    for i, res in enumerate(yolo_results):
        x, y, w, h = res["bbox"]
        # Generous padding
        px = max(int(w * 0.05), 2); py = max(int(h * 0.05), 2)
        x1 = max(0, x-px); y1 = max(0, y-py); x2 = min(iw, x+w+px); y2 = min(ih, y+h+py)

        crop = img[y1:y2, x1:x2]
        text, conf, frags = _ocr_region(crop)
        v_color = _detect_vehicle_color(img, (x, y, w, h))
        v_type  = res.get("type", "car")
        ms = round((time.perf_counter() - t0) * 1000, 1)

        results.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "crop": crop,
            "text": text,
            "confidence": conf,
            "state": _state_from_plate(text),
            "city": _city_from_plate(text),
            "series_age": _estimate_series_age(text),
            "environment": _detect_environment(img),
            "telemetry": _mock_registration_data(text),
            "vehicle_type": v_type,
            "vehicle_color": v_color,
            "method": method,
            "fragments": frags,
            "process_ms": ms,
        })

    return results


def annotate_image(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw styled bounding boxes + labels onto *img*. Returns annotated copy."""
    out = img.copy()
    for i, det in enumerate(detections):
        x, y, w, h = det["bbox"]
        conf  = det.get("confidence", 0)
        text  = det.get("text", "")
        state = det.get("state")
        city  = det.get("city")

        # Colour by confidence: green > 0.6, amber 0.3–0.6, red < 0.3
        if conf >= 0.6:
            colour = (0, 230, 118)   # green
        elif conf >= 0.3:
            colour = (0, 180, 255)   # amber
        else:
            colour = (68, 68, 255)   # red-ish blue

        # Bounding box (thick)
        cv2.rectangle(out, (x, y), (x+w, y+h), colour, 3)
        # Corner accents
        clen = min(w, h) // 4
        for (cx, cy, dx, dy) in [
            (x, y, clen, clen), (x+w, y, -clen, clen),
            (x, y+h, clen, -clen), (x+w, y+h, -clen, -clen)
        ]:
            cv2.line(out, (cx, cy), (cx+dx, cy),   colour, 4)
            cv2.line(out, (cx, cy), (cx,    cy+dy), colour, 4)

        # Label
        label = text if text else "?"
        if city:
            label += f" ({city})"
        elif state:
            label += f" [{state}]"
        label += f"  {conf:.0%}"

        fscale = 0.55
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)
        by = max(y - 8, th + 8)
        cv2.rectangle(out, (x, by - th - 8), (x + tw + 8, by + baseline), colour, -1)
        cv2.putText(out, label, (x + 4, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fscale, (10, 10, 10), 1, cv2.LINE_AA)

    return out


def ndarray_to_base64(img: np.ndarray, fmt: str = ".jpg", quality: int = 88) -> str:
    """Encode a BGR ndarray → base64 data-URI string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, img, params)
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64  = base64.b64encode(buf).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def get_detector_info() -> Dict[str, Any]:
    """Return info about which detection components are loaded."""
    return {
        "cascade_path":  str(CASCADE_PATH) if CASCADE_PATH.exists() else _OPENCV_CASCADE,
        "keras_model":   str(KERAS_MODEL_PATH) if KERAS_MODEL_PATH.exists() else (
                         str(H5_MODEL_PATH) if H5_MODEL_PATH.exists() else None),
        "easyocr_ready": _reader is not None,
        "keras_loaded":  _keras_model is not None,
        "yolo_loaded":   _yolo_model is not None,
        "yolo_ready":    _yolo_model is not None,
        "supported_states": len(INDIAN_STATE_CODES),
        "rto_db_size": sum(len(v) for v in _get_rto_db().values()),
        "detection_methods": ["yolo_v8_ai", "keras_resnet", "haar_cascade", "contour"],
    }
