"""
Image preprocessing for OCR improvement.
"""

import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
from typing import Tuple, Any
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_image(img_path: str) -> Tuple[Any, float]:
    """
    Preprocess image for better OCR results.
    
    Args:
        img_path: Path to input image
        
    Returns:
        Tuple of (preprocessed_image, processing_time_ms)
    """
    start_time = time.time()
    
    try:
        # Read image - FIXED: Always read from file path
        if isinstance(img_path, str):
            # Convert to absolute path to avoid issues
            abs_path = Path(img_path).absolute()
            img = cv2.imread(str(abs_path))
            if img is None:
                raise ValueError(f"Could not load image: {abs_path}")
        else:
            # Assume it's already a numpy array (from debug)
            img = img_path
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply noise removal
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        processing_time = (time.time() - start_time) * 1000
        logger.debug(f"Image preprocessing completed in {processing_time:.2f}ms")
        
        return processed, processing_time
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Return original image if preprocessing fails
        processing_time = (time.time() - start_time) * 1000
        if isinstance(img_path, str):
            # Try to return the original image
            try:
                img = cv2.imread(img_path)
                return img, processing_time
            except:
                pass
        return img_path, processing_time