"""
Skill: Dataset Ingestion
Downloads and parses both Kaggle ANPR datasets.
Dataset 1: alihassanml/car-number-plate  (Pascal VOC XML annotations)
Dataset 2: suprabhosaha/indian-vehicle-number-plate-detection-dataset
"""
import os
import json
import zipfile
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


# ── Kaggle API helpers ───────────────────────────────────────────────────────

def _ensure_kaggle_credentials():
    """Raise helpful error if kaggle.json is missing."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            "kaggle.json not found. Please download it from "
            "https://www.kaggle.com/account and place it at ~/.kaggle/kaggle.json"
        )


def download_dataset(dataset_slug: str, dest_dir: Optional[Path] = None) -> Path:
    """
    Download a Kaggle dataset zip into *dest_dir* and unzip it.

    Args:
        dataset_slug: e.g. "alihassanml/car-number-plate"
        dest_dir: defaults to data/raw/<dataset_name>

    Returns:
        Path to the extracted dataset directory.
    """
    try:
        import kaggle   # noqa: F401
    except ImportError:
        raise ImportError("Install kaggle: pip install kaggle")

    _ensure_kaggle_credentials()

    name = dataset_slug.split("/")[-1]
    dest = dest_dir or (DATA_RAW / name)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset '{dataset_slug}' → {dest}")

    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

    logger.info(f"Dataset ready at: {dest}")
    return dest


# ── Pascal-VOC XML annotation parser ────────────────────────────────────────

def parse_voc_xml(xml_path: Path) -> dict:
    """
    Parse a Pascal-VOC XML file. Returns:
        {
            "filename": str,
            "width": int, "height": int,
            "boxes": [{"xmin": int, "ymin": int, "xmax": int, "ymax": int}]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    filename = root.findtext("filename", xml_path.stem)

    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        boxes.append(
            {
                "xmin": int(float(bndbox.findtext("xmin", "0"))),
                "ymin": int(float(bndbox.findtext("ymin", "0"))),
                "xmax": int(float(bndbox.findtext("xmax", "0"))),
                "ymax": int(float(bndbox.findtext("ymax", "0"))),
            }
        )
    return {"filename": filename, "width": w, "height": h, "boxes": boxes}


def build_manifest(dataset_dir: Path, output_json: Optional[Path] = None) -> list[dict]:
    """
    Walk *dataset_dir* for image+XML pairs, parse annotations, and
    optionally write a JSON manifest.

    Returns a list of annotation dicts.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    manifest = []

    xml_files = list(dataset_dir.rglob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML annotation files in {dataset_dir}")

    for xml_path in xml_files:
        ann = parse_voc_xml(xml_path)
        # Try to locate the corresponding image
        for ext in image_exts:
            img_path = xml_path.with_suffix(ext)
            if img_path.exists():
                ann["image_path"] = str(img_path)
                break
        manifest.append(ann)

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written to {output_json}")

    return manifest


# ── Convenience entry points ─────────────────────────────────────────────────

def ingest_all():
    """Download both datasets and build manifests."""
    datasets = [
        "alihassanml/car-number-plate",
        "suprabhosaha/indian-vehicle-number-plate-detection-dataset",
    ]
    for slug in datasets:
        name = slug.split("/")[-1]
        try:
            dest = download_dataset(slug)
            manifest_path = DATA_PROCESSED / f"{name}_manifest.json"
            build_manifest(dest, manifest_path)
        except Exception as exc:
            logger.error(f"Failed to ingest '{slug}': {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_all()
