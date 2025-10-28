"""
Storage utilities for reading/writing files and organizing outputs.

Changes:
- Do NOT create top-level `metrics/` directories.
- `save_metrics` now writes metrics under `logs/metrics/{metrics_type}/...`.
- Directory creation list no longer includes any `metrics/*` paths.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

def ensure_directories():
    """Create necessary directories."""
    directories = [
        "ocr_outputs/mistral",
        "ocr_outputs/mistral_enhanced",
        # metrics directories removed per request
        "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def save_ocr_output(engine: str, filename: str, data: Dict[str, Any]):
    """Save OCR output to JSON file."""
    output_dir = f"ocr_outputs/{engine}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / f"{filename}.json"

    # Create a clean version for storage (remove any binary data)
    clean_data = {}
    for key, value in data.items():
        if key not in ['image_data', 'binary_content']:  # Skip large binary data
            clean_data[key] = value

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {engine} OCR output: {output_path}")

def save_raw_ocr_text(engine: str, filename: str, ocr_text: str, metadata: Dict[str, Any] = None):
    """Save raw OCR text to a readable text file."""
    output_dir = f"ocr_outputs/{engine}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / f"{filename}.txt"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {engine.upper()} OCR OUTPUT ===\n")
        f.write(f"Filename: {filename}\n")
        if metadata:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        f.write("\n" + "="*50 + "\n\n")
        f.write(ocr_text)

    logger.info(f"Saved raw {engine} OCR text: {output_path}")

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save verification results to JSON file."""
    output_dir = Path(output_path).parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved results to: {output_path}")

def save_metrics(metrics_type: str, filename: str, data: Dict[str, Any]):
    """
    Save metrics to JSON file.

    NOTE: To avoid creating top-level `metrics/` folders, metrics are saved under `logs/metrics/{metrics_type}/`.
    """
    # Put metrics under logs to avoid top-level metrics/ directory
    output_dir = Path("logs") / "metrics" / metrics_type
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.debug(f"Saved metrics: {output_path}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_ocr_output_path(engine: str, filename: str) -> str:
    """Get the path where OCR output would be saved."""
    return f"ocr_outputs/{engine}/{filename}.json"
