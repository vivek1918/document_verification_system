#!/usr/bin/env python3
"""
Document Verification Pipeline with both Mistral OCRs (Original + Enhanced)
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
from PIL import Image

# Load .env automatically
from dotenv import load_dotenv
load_dotenv()

# Add verifier package to path
sys.path.insert(0, str(Path(__file__).parent))

from verifier.utils.logger import get_logger
from verifier.io.storage import ensure_directories, save_results
from verifier.ocr.preproc import preprocess_image
from verifier.extract.regex_extractors import extract_entities
from verifier.normalize.cleaners import apply_confusion_corrections
from verifier.ocr.mistral_ocr import run_mistral_ocr
from verifier.ocr.mistral_ocr_enhanced import run_enhanced_mistral_ocr

logger = get_logger(__name__)


class DocumentVerificationPipeline:
    def __init__(self, config_path: str = "config.yml", use_llm: bool = False):
        self.config = self._load_config(config_path)
        self.use_llm = use_llm
        ensure_directories()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML config and replace ${ENV_VAR} placeholders."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)

            def replace_env(obj):
                if isinstance(obj, dict):
                    return {k: replace_env(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [replace_env(i) for i in obj]
                elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                    env_var = obj[2:-1]
                    return os.getenv(env_var, "")
                else:
                    return obj

            return replace_env(cfg)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return {}

    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Return size, format, and dimensions of an image file."""
        try:
            file = Path(file_path)
            size_bytes = file.stat().st_size
            with Image.open(file_path) as img:
                width, height = img.size
                fmt = img.format
            return {"size_bytes": size_bytes, "size_kb": size_bytes/1024,
                    "dimensions": f"{width}x{height}", "format": fmt}
        except Exception as e:
            return {"size_bytes": 0, "size_kb": 0, "dimensions": "Unknown", "format": "Unknown", "error": str(e)}

    def process_person(self, person_id: str, document_paths: Dict[str, str]) -> Dict[str, Any]:
        """Process all documents for a single person."""
        logger.info(f"📁 Processing person: {person_id}")
        for doc_type, doc_path in document_paths.items():
            info = self._get_file_info(doc_path)
            logger.info(f"   └─ {doc_type}: {doc_path} ({info['size_kb']:.2f} KB, {info['dimensions']}, {info['format']})")

        ocr_results = {}
        active_ocr_engines = []
        all_extracted_data = {}

        mistral_key = self.config.get("ocr", {}).get("mistral_api_key")
        groq_key = self.config.get("llm", {}).get("groq_api_key")
        groq_model = self.config.get("llm", {}).get("groq_model", "openai/gpt-oss-20b")

        # Process each document type separately
        for doc_type, doc_path in document_paths.items():
            if not os.path.exists(doc_path):
                logger.error(f"❌ Document file not found: {doc_path}")
                continue

            # Preprocess image
            logger.info(f"   ⚙️  Preprocessing {doc_type} image...")
            preprocessed_img, preproc_time = preprocess_image(doc_path)

            # Initialize OCR results for this document type
            doc_ocr_results = {}

            # --- Original Mistral OCR ---
            if self.config.get("ocr", {}).get("enable_mistral_ocr", True) and mistral_key:
                try:
                    if "Mistral API" not in active_ocr_engines:
                        active_ocr_engines.append("Mistral API")
                    logger.info(f"   🤖 Using Original Mistral API OCR for {doc_type}...")
                    mistral_result = run_mistral_ocr(doc_path, mistral_key, save_output=True)
                    # Ensure dict format
                    if isinstance(mistral_result, str):
                        mistral_result = {"raw_text": mistral_result, "success": True}
                    doc_ocr_results['mistral'] = mistral_result
                    logger.info(f"   ✅ Original Mistral OCR for {doc_type}: {len(mistral_result.get('raw_text',''))} chars")
                except Exception as e:
                    logger.error(f"   💥 Original Mistral OCR error for {doc_type}: {e}")

            # --- Enhanced Mistral OCR ---
            if self.config.get("ocr", {}).get("enable_mistral_ocr_enhanced", True) and mistral_key:
                try:
                    if "Mistral API Enhanced" not in active_ocr_engines:
                        active_ocr_engines.append("Mistral API Enhanced")
                    logger.info(f"   🤖 Using Mistral API Enhanced OCR for {doc_type}...")
                    mistral_result_enh = run_enhanced_mistral_ocr(doc_path, mistral_key, doc_type=doc_type, save_output=True)
                    # Ensure dict format
                    if isinstance(mistral_result_enh, str):
                        mistral_result_enh = {"raw_text": mistral_result_enh, "success": True}
                    doc_ocr_results['mistral_enhanced'] = mistral_result_enh
                    logger.info(f"   ✅ Mistral Enhanced OCR for {doc_type}: {len(mistral_result_enh.get('raw_text',''))} chars")
                except Exception as e:
                    logger.error(f"   💥 Mistral Enhanced OCR error for {doc_type}: {e}")

            # Store OCR results for this document type
            ocr_results[doc_type] = doc_ocr_results

            # --- Extract entities for this document ---
            try:
                # Pass the OCR results for THIS document only
                extracted_data = extract_entities(doc_ocr_results, doc_type, use_groq=True, groq_api_key=groq_key, groq_model=groq_model)
                extracted_data = apply_confusion_corrections(extracted_data)
                
                # Merge extracted data (prioritize non-empty values)
                for field, value in extracted_data.items():
                    if value and value.get('value'):
                        all_extracted_data[field] = value
            except Exception as e:
                logger.error(f"   💥 Extraction error for {doc_type}: {e}")

        # Determine overall status based on extracted data
        has_essential_data = any(all_extracted_data.get(field, {}).get('value') 
                            for field in ['full_name', 'aadhaar_number', 'pan_number'])
        
        overall_status = "VERIFIED" if has_essential_data else "FAILED"

        logger.info(f"📊 FINAL STRUCTURE for {person_id}:")
        logger.info(f"   - Document types found: {list(ocr_results.keys())}")
        logger.info(f"   - Documents count: {len(ocr_results)}")
        logger.info(f"   - OCR engines used: {active_ocr_engines}")
        logger.info(f"   - OCR engines count: {len(active_ocr_engines)}")
        
        for doc_type, doc_data in ocr_results.items():
            logger.info(f"   - {doc_type}: {len(doc_data)} OCR engines")
            for engine, engine_data in doc_data.items():
                success = engine_data.get('success', False)
                text_len = len(engine_data.get('raw_text', ''))
                logger.info(f"     * {engine}: success={success}, text_len={text_len}")

        return {
            "person_id": person_id,
            "ocr_results": ocr_results,
            "extracted_data": all_extracted_data,
            "overall_status": overall_status,
            "ocr_engines_used": active_ocr_engines
        }

    def process_dataset(self, input_path: str) -> List[Dict[str, Any]]:
        results = []

        if input_path.endswith(".zip"):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(input_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)
                results = self._process_folder(temp_dir)
        else:
            results = self._process_folder(input_path)

        return results

    def _process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        results = []
        folder = Path(folder_path)
        expected_docs = ["government_id", "bank_statement", "employment_letter"]

        for person_dir in folder.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name
                document_paths = {}
                for file in person_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg", ".png", ".tiff"]:
                        for doc_type in expected_docs:
                            if file.stem.endswith(doc_type):
                                document_paths[doc_type] = str(file)
                                break
                if len(document_paths) == len(expected_docs):
                    result = self.process_person(person_id, document_paths)
                    results.append(result)
                else:
                    logger.warning(f"Missing documents for {person_id}. Found: {list(document_paths.keys())}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Document Verification Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input path (zip or folder)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--use-llm", type=bool, default=False, help="Use LLM for extraction")
    parser.add_argument("--config", type=str, default="config.yml", help="Config file path")
    args = parser.parse_args()

    pipeline = DocumentVerificationPipeline(args.config, args.use_llm)
    logger.info(f"Starting pipeline with input: {args.input}")
    results = pipeline.process_dataset(args.input)

    save_results(results, args.output)
    logger.info(f"Pipeline completed. Results saved to: {args.output}")

    verified_count = sum(1 for r in results if r["overall_status"] == "VERIFIED")
    print(f"Processed {len(results)} persons. Verified: {verified_count}, Failed: {len(results) - verified_count}")


if __name__ == "__main__":
    main()