"""
Enhanced Mistral AI OCR with specialized document prompts.
"""

import base64
import time
from typing import Dict, Any
from pathlib import Path  # ADD THIS IMPORT
from mistralai import Mistral
from verifier.utils.logger import get_logger
from verifier.io.storage import save_ocr_output, save_raw_ocr_text

logger = get_logger(__name__)

class EnhancedMistralOCR:
    def __init__(self, api_key: str):
        """Initialize enhanced Mistral AI client."""
        self.client = Mistral(api_key=api_key)
        logger.info("Enhanced Mistral AI OCR initialized")
    
    def run_ocr(self, image_path: str, doc_type: str = "document", save_output: bool = True) -> Dict[str, Any]:
        """
        Run enhanced Mistral AI OCR with document-specific prompts.
        
        Args:
            image_path: Path to input image
            doc_type: Type of document for better prompting
            save_output: Whether to save OCR output to file
            
        Returns:
            Dict with raw_text, lines, and timing info
        """
        start_time = time.time()
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Document-specific prompts for better OCR
            doc_prompts = {
                "government_id": "Extract all text from this government ID document. Preserve exact formatting, including numbers, dates, and addresses. Return the text exactly as it appears.",
                "bank_statement": "Extract all text from this bank statement. Preserve exact account numbers, amounts, dates, and personal information. Return the text exactly as it appears.",
                "employment_letter": "Extract all text from this employment letter. Preserve exact names, dates, addresses, and employment details. Return the text exactly as it appears.",
                "document": "Extract all text from this document image. Return the text exactly as it appears, preserving line breaks, numbers, and special characters."
            }
            
            prompt = doc_prompts.get(doc_type, doc_prompts["document"])
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]
            
            # Call Mistral API
            response = self.client.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                max_tokens=4000
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            if response and response.choices:
                full_text = response.choices[0].message.content
                
                # Process into structured lines
                lines = []
                for i, line_text in enumerate(full_text.split('\n')):
                    clean_text = line_text.strip()
                    if clean_text:
                        lines.append({
                            'text': clean_text,
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
                    "engine": "mistral_api_enhanced",
                    "doc_type": doc_type,
                    "image_path": image_path,
                    "timestamp": time.time()
                }
                
                # Save OCR output to file
                if save_output:
                    filename = f"{Path(image_path).stem}_{doc_type}"
                    save_ocr_output("mistral_enhanced", filename, result)
                    save_raw_ocr_text("mistral_enhanced", filename, full_text, {
                        "image_path": image_path,
                        "doc_type": doc_type,
                        "processing_time_ms": processing_time,
                        "lines_extracted": len(lines),
                        "word_count": len(full_text.split())
                    })
                
                logger.info(f"Enhanced Mistral OCR completed in {processing_time:.2f}ms for {doc_type}, extracted {len(lines)} lines")
                return result
            else:
                logger.warning(f"No text detected by Enhanced Mistral OCR for {doc_type}")
                result = {
                    "raw_text": "",
                    "lines": [],
                    "time_ms": processing_time,
                    "success": False,
                    "error": "No text detected",
                    "engine": "mistral_api_enhanced"
                }
                return result
                
        except Exception as e:
            logger.error(f"Enhanced Mistral OCR failed: {e}")
            processing_time = (time.time() - start_time) * 1000
            result = {
                "raw_text": "",
                "lines": [],
                "time_ms": processing_time,
                "success": False,
                "error": str(e),
                "engine": "mistral_api_enhanced"
            }
            return result

def run_enhanced_mistral_ocr(image_path: str, api_key: str, doc_type: str = "document", save_output: bool = True) -> Dict[str, Any]:
    """
    Run enhanced Mistral OCR with document-specific prompting.
    """
    ocr_engine = EnhancedMistralOCR(api_key)
    return ocr_engine.run_ocr(image_path, doc_type, save_output)