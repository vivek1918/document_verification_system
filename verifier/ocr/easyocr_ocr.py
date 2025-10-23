"""
EasyOCR implementation with GPU support.
"""

import time
from typing import Dict, List, Any
import numpy as np
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy loading of EasyOCR to avoid GPU memory issues
_easyocr_reader = None

def get_easyocr_reader(device: str = 'cpu'):
    """Get or create EasyOCR reader with lazy loading."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            # Initialize with English language
            _easyocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
            logger.info(f"EasyOCR initialized with device: {device}")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    return _easyocr_reader

def run_easyocr(image: Any, device: str = 'cpu') -> Dict[str, Any]:
    """
    Run EasyOCR on image.
    
    Args:
        image: Preprocessed image (numpy array)
        device: 'cpu' or 'cuda'
        
    Returns:
        Dict with raw_text, lines, and timing info
    """
    start_time = time.time()
    
    try:
        reader = get_easyocr_reader(device)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            import cv2
            image = cv2.imread(image) if isinstance(image, str) else np.array(image)
        
        # Run OCR
        results = reader.readtext(image)
        
        # Process results
        lines = []
        for (bbox, text, confidence) in results:
            lines.append({
                'text': text.strip(),
                'confidence': confidence * 100,  # Convert to percentage
                'bbox': bbox
            })
        
        # Combine all text
        raw_text = "\n".join([line['text'] for line in lines])
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "raw_text": raw_text,
            "lines": lines,
            "time_ms": processing_time,
            "detection_count": len(results),
            "success": True,
            "device_used": device
        }
        
        logger.info(f"EasyOCR ({device}) completed in {processing_time:.2f}ms, detected {len(results)} text regions")
        return result
        
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
        processing_time = (time.time() - start_time) * 1000
        return {
            "raw_text": "",
            "lines": [],
            "time_ms": processing_time,
            "error": str(e),
            "success": False,
            "device_used": device
        }