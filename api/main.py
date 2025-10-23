"""
FastAPI server for document verification.
"""

import os
import tempfile
import zipfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifier.utils.logger import get_logger
from run_pipeline import DocumentVerificationPipeline

logger = get_logger(__name__)

app = FastAPI(
    title="Document Verification API",
    description="KYC Document Verification with OCR and Cross-Document Validation",
    version="1.0.0"
)

class VerificationResponse(BaseModel):
    person_id: str
    extracted_data: Dict[str, Any]
    verification_results: Dict[str, Any]
    overall_status: str
    logs: str

class HealthResponse(BaseModel):
    status: str
    tesseract_available: bool
    easyocr_available: bool
    gpu_available: bool

@app.post("/verify", response_model=List[VerificationResponse])
async def verify_documents(file: UploadFile = File(...), use_llm: bool = False):
    """
    Verify documents from uploaded zip file.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process with pipeline
        pipeline = DocumentVerificationPipeline(use_llm=use_llm)
        results = pipeline.process_dataset(temp_path)
        
        # Cleanup
        os.unlink(temp_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check Tesseract
        import pytesseract
        pytesseract.get_tesseract_version()
        tesseract_available = True
    except:
        tesseract_available = False
    
    try:
        # Check EasyOCR
        import easyocr
        easyocr_available = True
    except:
        easyocr_available = False
    
    try:
        # Check GPU
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        gpu_available = False
    
    return HealthResponse(
        status="healthy",
        tesseract_available=tesseract_available,
        easyocr_available=easyocr_available,
        gpu_available=gpu_available
    )

@app.get("/metrics")
async def get_metrics():
    """
    Get current system metrics.
    """
    metrics_path = "metrics/ocr/comparison_report.json"
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            return json.load(f)
    else:
        return {"message": "No metrics available yet"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)