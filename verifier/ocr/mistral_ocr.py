"""
Official Mistral AI OCR implementation using their API.
"""

import base64
import time
from typing import Dict, Any, List
from pathlib import Path  # ADD THIS IMPORT
from mistralai import Mistral
from verifier.utils.logger import get_logger
from verifier.io.storage import save_ocr_output, save_raw_ocr_text

logger = get_logger(__name__)

class MistralOCR:
    def __init__(self, api_key: str):
        """Initialize Mistral AI client."""
        self.client = Mistral(api_key=api_key)
        logger.info("Mistral AI OCR initialized")
    
    def run_ocr(self, image_path: str, save_output: bool = True) -> Dict[str, Any]:
        """
        Run Mistral AI OCR on image.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save OCR output to file
            
        Returns:
            Dict with raw_text, lines, and timing info
        """
        start_time = time.time()
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Convert to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare messages for OCR
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this document image. Return the text exactly as it appears, preserving line breaks and formatting. Do not interpret or modify the text."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            # Call Mistral API for OCR
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                max_tokens=4000
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Extract OCR text from response
            if response and response.choices:
                full_text = response.choices[0].message.content
                
                # Process into lines
                lines = []
                for i, line_text in enumerate(full_text.split('\n')):
                    if line_text.strip():
                        lines.append({
                            'text': line_text.strip(),
                            'confidence': 95.0,
                            'bbox': [0, i * 20, 100, 20],
                            'line_number': i
                        })
                
                result = {
                    "raw_text": full_text,
                    "lines": lines,
                    "time_ms": processing_time,
                    "word_count": len(full_text.split()),
                    "success": True,
                    "engine": "mistral_api",
                    "image_path": image_path,
                    "timestamp": time.time()
                }
                
                # Save OCR output to file
                if save_output:
                    filename = Path(image_path).stem
                    save_ocr_output("mistral", filename, result)
                    save_raw_ocr_text("mistral", filename, full_text, {
                        "image_path": image_path,
                        "processing_time_ms": processing_time,
                        "lines_extracted": len(lines),
                        "word_count": len(full_text.split())
                    })
                
                logger.info(f"Mistral OCR completed in {processing_time:.2f}ms, extracted {len(lines)} lines, {len(full_text)} chars")
                return result
            else:
                logger.warning("No text detected by Mistral OCR")
                result = {
                    "raw_text": "",
                    "lines": [],
                    "time_ms": processing_time,
                    "success": False,
                    "error": "No text detected",
                    "engine": "mistral_api"
                }
                return result
                
        except Exception as e:
            logger.error(f"Mistral OCR failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            result = {
                "raw_text": "",
                "lines": [],
                "time_ms": processing_time,
                "success": False,
                "error": str(e),
                "engine": "mistral_api"
            }
            return result

# Global instance
_mistral_ocr = None

def get_mistral_ocr(api_key: str) -> MistralOCR:
    """Get or create Mistral OCR instance."""
    global _mistral_ocr
    if _mistral_ocr is None:
        _mistral_ocr = MistralOCR(api_key)
    return _mistral_ocr

def run_mistral_ocr(image_path: str, api_key: str, save_output: bool = True) -> Dict[str, Any]:
    """
    Run Mistral OCR on image.
    
    Args:
        image_path: Path to input image
        api_key: Mistral API key
        save_output: Whether to save OCR output to file
        
    Returns:
        Dict with OCR results
    """
    ocr_engine = get_mistral_ocr(api_key)
    return ocr_engine.run_ocr(image_path, save_output)