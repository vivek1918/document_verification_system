"""
Google Cloud Vision OCR implementation (API key only, local images).
"""

import base64
import time
import requests
from typing import Dict, Any
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

class GoogleVisionOCR:
    def __init__(self, api_key: str):
        """Initialize Google Vision OCR with API key (no service account)."""
        self.api_key = api_key
        logger.info("Google Cloud Vision OCR initialized with API key")

    def run_ocr(self, image_path: str) -> Dict[str, Any]:
        """
        Run Google Vision OCR on a local image using API key.

        Args:
            image_path: Path to the local image file

        Returns:
            Dict: OCR result containing raw text, lines, timing, success flag
        """
        start_time = time.time()

        try:
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()

            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
            payload = {
                "requests": [
                    {
                        "image": {"content": img_base64},
                        "features": [{"type": "TEXT_DETECTION"}]
                    }
                ]
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            processing_time = (time.time() - start_time) * 1000

            text = ""
            lines = []

            if "responses" in data and "textAnnotations" in data["responses"][0]:
                text_annotations = data["responses"][0]["textAnnotations"]
                if text_annotations:
                    text = text_annotations[0].get("description", "")
                    # Skip first entry (full text) and process bounding boxes for lines
                    for t in text_annotations[1:]:
                        vertices = [{"x": v.get("x", 0), "y": v.get("y", 0)} 
                                    for v in t.get("boundingPoly", {}).get("vertices", [])]
                        lines.append({
                            "text": t.get("description", ""),
                            "confidence": 95.0,  # Google Vision doesn't provide per-word confidence
                            "bbox": vertices
                        })

            success = bool(text)

            logger.info(f"Google Vision OCR completed in {processing_time:.2f}ms, extracted {len(lines)} text regions, {len(text)} chars")

            return {
                "raw_text": text,
                "lines": lines,
                "time_ms": processing_time,
                "word_count": len(text.split()),
                "success": success,
                "engine": "google_vision"
            }

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Google Vision OCR failed: {e}")
            return {
                "raw_text": "",
                "lines": [],
                "time_ms": processing_time,
                "success": False,
                "error": str(e),
                "engine": "google_vision"
            }


# Global instance
_google_vision_ocr = None

def get_google_vision_ocr(api_key: str) -> GoogleVisionOCR:
    """Get or create Google Vision OCR instance (singleton)."""
    global _google_vision_ocr
    if _google_vision_ocr is None:
        _google_vision_ocr = GoogleVisionOCR(api_key)
    return _google_vision_ocr

def run_google_vision(image_path: str, api_key: str) -> Dict[str, Any]:
    """
    Run Google Vision OCR on image using API key.

    Args:
        image_path: Local image path
        api_key: Google Cloud Vision API key

    Returns:
        Dict: OCR result
    """
    ocr_engine = get_google_vision_ocr(api_key)
    return ocr_engine.run_ocr(image_path)
