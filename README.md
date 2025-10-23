# Document Verification Pipeline

A production-grade document processing and KYC verification system with OCR, structured extraction, and cross-document verification.

## Features

- **Dual OCR Engine**: Tesseract (primary) + EasyOCR (fallback/ensemble)
- **Structured Entity Extraction**: Name, DOB, Address, Phone, Email, Aadhaar, PAN, etc.
- **Cross-Document Verification**: 7 validation rules across 3 document types
- **GPU Support**: CUDA acceleration for EasyOCR and LLMs
- **Optional LLM Integration**: Gemini API or Hugging Face models
- **Comprehensive Logging & Metrics**: Per-document processing and evaluation
- **REST API**: FastAPI endpoints for document verification
- **CLI Interface**: Command-line pipeline execution

## Quick Start

### 1. Prerequisites

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y tesseract-ocr libtesseract-dev

# macOS
brew install tesseract

# Windows: Download from https://github.com/tesseract-ocr/tesseract