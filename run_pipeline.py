#!/usr/bin/env python3
"""
Main CLI entry point for document verification pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import zipfile
import tempfile
from typing import Dict, List, Any
import torch
import time

# Add verifier package to path
sys.path.insert(0, str(Path(__file__).parent))

from verifier.utils.logger import get_logger
from verifier.io.storage import ensure_directories, save_results
from verifier.ocr.tesseract_ocr import run_tesseract
from verifier.ocr.easyocr_ocr import run_easyocr
from verifier.ocr.preproc import preprocess_image
from verifier.extract.regex_extractors import extract_entities
from verifier.verify.rules import verify_person
from verifier.normalize.cleaners import apply_confusion_corrections

logger = get_logger(__name__)

class DocumentVerificationPipeline:
    def __init__(self, config_path: str = "config.yml", use_llm: bool = False):
        self.config = self._load_config(config_path)
        self.use_llm = use_llm
        ensure_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {}
    
    def process_person(self, person_id: str, document_paths: Dict[str, str]) -> Dict[str, Any]:
        """Process all documents for a single person."""
        logger.info(f"Processing person: {person_id}")
        
        extracted_data = {}
        ocr_metrics = {}
        
        # Get API keys from config
        mistral_key = self.config.get('ocr', {}).get('mistral_api_key')
        groq_key = self.config.get('llm', {}).get('groq_api_key')
        groq_model = self.config.get('llm', {}).get('groq_model', 'llama2-70b-4096')
        
        for doc_type, doc_path in document_paths.items():
            logger.info(f"Processing {doc_type} for {person_id} from {doc_path}")
            
            if not os.path.exists(doc_path):
                logger.error(f"Document file not found: {doc_path}")
                continue
                
            # Preprocess image
            preprocessed_img, preproc_time = preprocess_image(doc_path)
            
            # Run OCR engines
            ocr_results = {}
            mistral_success = False
            
            # Use Official Mistral API OCR as primary (if we have key)
            if self.config.get('ocr', {}).get('enable_mistral_ocr', True) and mistral_key:
                try:
                    from verifier.ocr.mistral_ocr import run_mistral_ocr
                    mistral_result = run_mistral_ocr(doc_path, mistral_key, save_output=True)
                    ocr_results['mistral'] = mistral_result
                    mistral_success = mistral_result.get('success', False)
                    
                    if mistral_success:
                        logger.info(f"Mistral API OCR for {doc_type}: {len(mistral_result.get('raw_text', ''))} chars")
                    else:
                        logger.warning(f"Mistral API OCR failed for {doc_type}: {mistral_result.get('error', 'Unknown error')}")
                        
                    # Try enhanced version only if basic version fails
                    if not mistral_success:
                        try:
                            from verifier.ocr.mistral_ocr_enhanced import run_enhanced_mistral_ocr
                            enhanced_result = run_enhanced_mistral_ocr(doc_path, mistral_key, doc_type, save_output=True)
                            ocr_results['mistral_enhanced'] = enhanced_result
                            mistral_success = enhanced_result.get('success', False)
                            
                            if mistral_success:
                                logger.info(f"Enhanced Mistral OCR for {doc_type}: {len(enhanced_result.get('raw_text', ''))} chars")
                            else:
                                logger.warning(f"Enhanced Mistral OCR failed for {doc_type}")
                        except Exception as e:
                            logger.warning(f"Enhanced Mistral OCR not available: {e}")
                        
                except Exception as e:
                    logger.error(f"Mistral OCR failed: {e}")
            
            # Use fallback OCR engines if Mistral fails or is not available
            if not mistral_success:
                logger.info(f"Using fallback OCR engines for {doc_type}")
                
                # Try Tesseract
                if self.config.get('ocr', {}).get('enable_tesseract', True):
                    try:
                        from verifier.ocr.tesseract_ocr import run_tesseract
                        tesseract_result = run_tesseract(preprocessed_img)
                        ocr_results['tesseract'] = tesseract_result
                        
                        # Save Tesseract output
                        if tesseract_result.get('success'):
                            from verifier.io.storage import save_ocr_output, save_raw_ocr_text
                            from pathlib import Path
                            filename = f"{Path(doc_path).stem}"
                            save_ocr_output("tesseract", filename, tesseract_result)
                            if tesseract_result.get('raw_text'):
                                save_raw_ocr_text("tesseract", filename, tesseract_result['raw_text'], {
                                    "image_path": doc_path,
                                    "processing_time_ms": tesseract_result.get('time_ms', 0),
                                    "lines_extracted": len(tesseract_result.get('lines', [])),
                                    "word_count": len(tesseract_result['raw_text'].split())
                                })
                        
                        logger.info(f"Tesseract for {doc_type}: {len(tesseract_result.get('raw_text', ''))} chars")
                    except Exception as e:
                        logger.warning(f"Tesseract OCR failed: {e}")
                
                # Try EasyOCR
                if self.config.get('ocr', {}).get('enable_easyocr', True):
                    try:
                        from verifier.ocr.easyocr_ocr import run_easyocr
                        device = "cuda" if (self.config.get('gpu', {}).get('use_gpu', True) and torch and torch.cuda.is_available()) else "cpu"
                        easyocr_result = run_easyocr(preprocessed_img, device=device)
                        ocr_results['easyocr'] = easyocr_result
                        
                        # Save EasyOCR output
                        if easyocr_result.get('success'):
                            from verifier.io.storage import save_ocr_output, save_raw_ocr_text
                            from pathlib import Path
                            filename = f"{Path(doc_path).stem}"
                            save_ocr_output("easyocr", filename, easyocr_result)
                            if easyocr_result.get('raw_text'):
                                save_raw_ocr_text("easyocr", filename, easyocr_result['raw_text'], {
                                    "image_path": doc_path,
                                    "processing_time_ms": easyocr_result.get('time_ms', 0),
                                    "lines_extracted": len(easyocr_result.get('lines', [])),
                                    "word_count": len(easyocr_result['raw_text'].split())
                                })
                        
                        logger.info(f"EasyOCR for {doc_type}: {len(easyocr_result.get('raw_text', ''))} chars")
                    except Exception as e:
                        logger.warning(f"EasyOCR failed: {e}")
            
            # Extract entities using Groq (primary) or regex (fallback)
            use_groq = self.config.get('llm', {}).get('provider') == 'groq' and groq_key
            try:
                from verifier.extract.regex_extractors import extract_entities
                doc_extracted = extract_entities(ocr_results, doc_type, use_groq, groq_key, groq_model)
                extracted_data[doc_type] = doc_extracted
                
                # Log extraction results
                extracted_count = sum(1 for field in doc_extracted.values() if field.get('value'))
                logger.info(f"Extracted {extracted_count} fields for {doc_type}")
                
            except Exception as e:
                logger.error(f"Entity extraction failed: {e}")
                # Create empty extraction as fallback
                extracted_data[doc_type] = _get_empty_extraction()
            
            # Store OCR metrics
            ocr_metrics[doc_type] = {
                'preprocessing_time_ms': preproc_time,
                'mistral_time_ms': ocr_results.get('mistral', {}).get('time_ms', 0),
                'mistral_enhanced_time_ms': ocr_results.get('mistral_enhanced', {}).get('time_ms', 0),
                'tesseract_time_ms': ocr_results.get('tesseract', {}).get('time_ms', 0),
                'easyocr_time_ms': ocr_results.get('easyocr', {}).get('time_ms', 0)
            }
        
        # Run verification rules
        try:
            from verifier.verify.rules import verify_person
            verification_results = verify_person(extracted_data)
            
            # Determine overall status
            overall_status = "VERIFIED"
            failed_rules = []
            
            for rule_name, rule_result in verification_results.items():
                if rule_result.get('status') == 'FAIL':
                    overall_status = "FAILED"
                    failed_rules.append(rule_name)
            
            if overall_status == "VERIFIED":
                logger.info(f"Person {person_id} VERIFIED - All rules passed")
            else:
                logger.warning(f"Person {person_id} FAILED - Failed rules: {failed_rules}")
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification_results = {}
            overall_status = "FAILED"
        
        # Save person-specific OCR outputs summary
        try:
            from verifier.io.storage import save_ocr_output
            from pathlib import Path
            person_summary = {
                "person_id": person_id,
                "documents_processed": list(document_paths.keys()),
                "extraction_summary": {
                    doc_type: {
                        "fields_extracted": sum(1 for field in data.values() if field.get('value')),
                        "total_fields": len(data)
                    } for doc_type, data in extracted_data.items()
                },
                "verification_status": overall_status,
                "timestamp": time.time()
            }
            save_ocr_output("summary", f"person_{person_id}", person_summary)
        except Exception as e:
            logger.warning(f"Could not save person summary: {e}")
        
        return {
            "person_id": person_id,
            "extracted_data": extracted_data,
            "verification_results": verification_results,
            "overall_status": overall_status,
            "ocr_metrics": ocr_metrics,
            "logs": f"logs/{person_id}.log"
        }

    def _get_empty_extraction() -> Dict[str, Any]:
        """Return empty extraction structure."""
        return {
            'full_name': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'father_name': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'date_of_birth': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'address': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'phone_number': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'email_address': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'aadhaar_number': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'pan_number': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'employee_id': {"value": None, "raw_context": None, "confidence": "low", "source": "error"},
            'account_number': {"value": None, "raw_context": None, "confidence": "low", "source": "error"}
        }

    def process_dataset(self, input_path: str) -> List[Dict[str, Any]]:
        """Process entire dataset."""
        results = []
        
        # Handle zip file input
        if input_path.endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(input_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                results = self._process_folder(temp_dir)
        else:
            # Handle folder input
            results = self._process_folder(input_path)
        
        return results
    
    def _process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process folder structure with person documents."""
        results = []
        folder = Path(folder_path)

        # Expected document types
        expected_docs = ['government_id', 'bank_statement', 'employment_letter']

        for person_dir in folder.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name
                document_paths = {}

                # Iterate over files in folder
                for file in person_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff']:
                        for doc_type in expected_docs:
                            # Check if filename ends with the expected doc_type
                            if file.stem.endswith(doc_type):
                                document_paths[doc_type] = str(file)
                                break

                if len(document_paths) == len(expected_docs):  # All documents found
                    result = self.process_person(person_id, document_paths)
                    results.append(result)
                else:
                    logger.warning(f"Missing documents for {person_id}. Found: {list(document_paths.keys())}")

        return results

def main():
    parser = argparse.ArgumentParser(description='Document Verification Pipeline')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input path (zip file or folder)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--use-llm', type=bool, default=False,
                       help='Use LLM for extraction (default: False)')
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentVerificationPipeline(args.config, args.use_llm)
    
    # Process dataset
    logger.info(f"Starting pipeline with input: {args.input}")
    results = pipeline.process_dataset(args.input)
    
    # Save results
    save_results(results, args.output)
    logger.info(f"Pipeline completed. Results saved to: {args.output}")
    
    # Print summary
    verified_count = sum(1 for r in results if r['overall_status'] == 'VERIFIED')
    print(f"Processed {len(results)} persons. Verified: {verified_count}, Failed: {len(results) - verified_count}")

if __name__ == "__main__":
    main()