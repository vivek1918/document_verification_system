"""
Tesseract OCR implementation.
"""

import pytesseract
from PIL import Image
import time
import json
from typing import Dict, List, Any
import numpy as np

from verifier.utils.logger import get_logger
from verifier.io.storage import save_ocr_output

logger = get_logger(__name__)

def run_tesseract(image: Any) -> Dict[str, Any]:
    """
    Run Tesseract OCR on image.
    
    Args:
        image: Preprocessed image (numpy array or PIL Image)
        
    Returns:
        Dict with raw_text, lines, and timing info
    """
    start_time = time.time()
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.open(image) if isinstance(image, str) else image
        
        # Get OCR data with confidence and bounding boxes
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Extract lines with confidence - IMPROVED VERSION
        lines = []
        current_line_text = ""
        current_line_confidence = []
        current_line_bbox = None
        
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            confidence = ocr_data['conf'][i]
            
            if int(ocr_data['level'][i]) == 5:  # Word level
                if text:  # Non-empty text
                    if current_line_text:
                        current_line_text += " " + text
                    else:
                        current_line_text = text
                    
                    current_line_confidence.append(confidence)
                    
                    # Update bbox to encompass all words in line
                    if current_line_bbox is None:
                        current_line_bbox = [
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['width'][i],
                            ocr_data['height'][i]
                        ]
                    else:
                        # Expand bbox
                        current_line_bbox[2] = max(current_line_bbox[2], ocr_data['left'][i] + ocr_data['width'][i] - current_line_bbox[0])
                        current_line_bbox[3] = max(current_line_bbox[3], ocr_data['top'][i] + ocr_data['height'][i] - current_line_bbox[1])
            
            # Check if we're at a new line
            if i < n_boxes - 1 and ocr_data['line_num'][i] != ocr_data['line_num'][i + 1]:
                if current_line_text.strip():
                    avg_confidence = sum(current_line_confidence) / len(current_line_confidence) if current_line_confidence else 0
                    lines.append({
                        'text': current_line_text.strip(),
                        'confidence': avg_confidence,
                        'bbox': current_line_bbox or [0, 0, 0, 0]
                    })
                
                # Reset for new line
                current_line_text = ""
                current_line_confidence = []
                current_line_bbox = None
        
        # Add last line if exists
        if current_line_text.strip():
            avg_confidence = sum(current_line_confidence) / len(current_line_confidence) if current_line_confidence else 0
            lines.append({
                'text': current_line_text.strip(),
                'confidence': avg_confidence,
                'bbox': current_line_bbox or [0, 0, 0, 0]
            })
        
        # Combine all text
        raw_text = "\n".join([line['text'] for line in lines])
        
        processing_time = (time.time() - start_time) * 1000
        
        result = {
            "raw_text": raw_text,
            "lines": lines,
            "time_ms": processing_time,
            "word_count": len([t for t in ocr_data['text'] if t.strip()]),
            "success": True
        }
        
        logger.info(f"Tesseract OCR completed in {processing_time:.2f}ms, extracted {len(lines)} lines, {result['word_count']} words")
        return result
        
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        processing_time = (time.time() - start_time) * 1000
        return {
            "raw_text": "",
            "lines": [],
            "time_ms": processing_time,
            "error": str(e),
            "success": False
        }