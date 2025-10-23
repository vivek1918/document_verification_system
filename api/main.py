#!/usr/bin/env python3
"""
FastAPI server for document verification (without Tesseract / EasyOCR).
"""

import os
import tempfile
import zipfile
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# Add parent directory to path so we can import run_pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifier.utils.logger import get_logger
from run_pipeline import DocumentVerificationPipeline

logger = get_logger(__name__)

app = FastAPI(
    title="Document Verification API",
    description="KYC Document Verification with OCR and Cross-Document Validation (No local OCR libs)",
    version="1.0.0"
)

# Response models are permissive: pipeline returns dicts; keep them flexible.
class VerificationResponse(BaseModel):
    person_id: str
    extracted_data: Optional[Dict[str, Any]] = None
    verification_results: Optional[Dict[str, Any]] = None
    overall_status: Optional[str] = None
    logs: Optional[str] = None
    ocr_results: Optional[Dict[str, Any]] = None
    ocr_engines_used: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    tesseract_available: bool
    easyocr_available: bool
    gpu_available: bool
    mistral_api_key_present: bool

MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024))  # default 50 MB

@app.post("/verify", response_model=List[VerificationResponse])
async def verify_documents(file: UploadFile = File(...), use_llm: bool = False):
    """
    Verify documents from uploaded zip file.
    Expects a ZIP where each folder is a person_id and contains expected images.
    """
    # Basic file type check
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only ZIP files are supported")

    # Protect against huge uploads
    file_size = 0
    try:
        content = await file.read()
        file_size = len(content)
        if file_size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                detail=f"Uploaded file is too large ({file_size} bytes). Max allowed is {MAX_UPLOAD_SIZE_BYTES} bytes.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed reading upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(content)
            temp_path = tmp.name

        pipeline = DocumentVerificationPipeline(use_llm=use_llm)
        results = pipeline.process_dataset(temp_path)

        return results

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error processing upload: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # cleanup temp file
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            logger.warning("Failed to delete temporary upload file", exc_info=True)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Since Tesseract and EasyOCR were removed, those availability flags will be False.
    We still check for GPU availability and presence of Mistral API key (if you use Mistral).
    """
    # local OCR libs removed intentionally
    tesseract_available = False
    easyocr_available = False

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except Exception:
        gpu_available = False

    # If you use Mistral or similar hosted OCR, check for API key presence
    mistral_api_key_present = bool(os.getenv("MISTRAL_API_KEY") or os.getenv("OCR_MISTRAL_API_KEY"))

    return HealthResponse(
        status="healthy",
        tesseract_available=tesseract_available,
        easyocr_available=easyocr_available,
        gpu_available=gpu_available,
        mistral_api_key_present=mistral_api_key_present
    )

@app.get("/metrics")
async def get_metrics():
    """
    Get current system metrics (if produced by the pipeline).
    """
    metrics_path = "metrics/ocr/comparison_report.json"
    if os.path.exists(metrics_path):
        import json
        try:
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read metrics file: {e}")
            raise HTTPException(status_code=500, detail="Failed to read metrics file")
    else:
        return {"message": "No metrics available yet"}

if __name__ == "__main__":
    # uvicorn.run() is used only for direct script execution (dev). In prod use 'uvicorn app:app --host 0.0.0.0 --port 8000'
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
