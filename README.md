# DOCUMENT_VERIFICATION_SYSTEM

_Transforming Documents Into Trustworthy Digital Identities_

![Last Commit](https://img.shields.io/github/last-commit/vivekvasani99/document_verification_system?style=flat-square) 
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

_Built with the tools and technologies:_

![Markdown](https://img.shields.io/badge/-Markdown-000000?logo=markdown&logoColor=white&style=flat)
![LIN](https://img.shields.io/badge/-LIN-007ACC?style=flat)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white&style=flat)
![Pytorch](https://img.shields.io/badge/-Pytorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)
![OpenAI](https://img.shields.io/badge/-OpenAI-412991?logo=openai&logoColor=white&style=flat)
![Prettier](https://img.shields.io/badge/-Prettier-F7B93E?logo=prettier&logoColor=white&style=flat)

---

<img width="672" height="516" alt="_- visual selection (3)" src="https://github.com/user-attachments/assets/667c1263-62e8-4c2b-bbd9-f7f78ec101fa" />

---

## Table of Contents
- [Overview](#overview)
- [Project Demo](#project-demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Technology Stack](#technology-stack)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)

---

## Overview

The document verification system is a powerful developer tool that automates and enhances document validation workflows using advanced OCR, data extraction, and cross-document verification techniques. Designed for scalability and accuracy, it integrates multiple OCR engines, GPU acceleration, and optional large language model (LLM) support to handle diverse document types efficiently.

### Why document_verification_system?

This project aims to simplify complex identity and document verification processes. The core features include:

- 🖼️ **Image Preprocessing:** Enhances OCR accuracy through noise reduction and normalization.  
- 🧠 **Dual OCR Engines:** Supports Mistral and Groq OCR for flexible, high-precision text extraction.  
- 🧩 **Entity Extraction & Normalization:** Extracts structured data from unstructured text, ensuring data consistency.  
- 🔄 **Cross-Document Validation:** Implements rules to verify data integrity across multiple documents.  
- ⚙️ **API Access & Scalability:** Provides accessible APIs for seamless integration into enterprise systems.  
- 🚀 **Performance Evaluation:** Includes tools for monitoring accuracy and system performance.  

---

## Project Demo

Watch the demo video on [Loom](https://www.loom.com/share/your-video-id)

Explore how the Document Verification System performs OCR, entity extraction, and structured validation in real time.

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python  
- **Package Manager:** Pip  

---

### Installation

Build `document_verification_system` from the source and install dependencies:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vivekvasani99/document_verification_system
2. **Navigate to the project directory:**
   ```bash
   cd document_verification_system
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the backend API (FastAPI)
   ```bash
   python run_pipeline.py --input sample_dataset_placeholder --output results.json
5. Run the Streamlit frontend
   ```bash
   streamlit run app.py

---

## Technical System Overview

The system follows a modular architecture with multiple stages of document processing and validation.

<img width="1044" height="930" alt="visual selection" src="https://github.com/user-attachments/assets/8f008053-7142-4dfe-9fcd-d43e6c29a4ce" />

---

### 🖥️ Frontend (Streamlit)
The project includes a user-friendly **Streamlit** interface to upload and verify documents seamlessly.  
Users can:
- Upload images or zip files.
- View extracted text and structured data instantly.
- See visual feedback for OCR accuracy.
  
---
## 🧾 App Preview
<img width="1912" height="740" alt="image" src="https://github.com/user-attachments/assets/8fc3aaec-8048-4c48-8415-3ff3d0928269" />
<img width="1917" height="858" alt="image" src="https://github.com/user-attachments/assets/5f149f26-0d6b-4f27-a7cd-75f522b6d7db" />
<img width="1915" height="868" alt="image" src="https://github.com/user-attachments/assets/a582a2e5-fa26-4ce0-9d78-ecda60eb6e6d" />
<img width="1837" height="862" alt="image" src="https://github.com/user-attachments/assets/789cb67a-2ede-4c18-bdc4-c67fb2c64921" />
<img width="1919" height="840" alt="image" src="https://github.com/user-attachments/assets/9c656d6b-816e-4b4c-adae-bc6a08dc9371" />
<img width="1919" height="864" alt="image" src="https://github.com/user-attachments/assets/710574ac-b67a-4732-b74e-85a9bf8af78d" />
<img width="1919" height="859" alt="image" src="https://github.com/user-attachments/assets/7eae213a-e492-4943-83ee-0d6eb2d239fe" />
<img width="1635" height="1158" alt="localhost_8501_" src="https://github.com/user-attachments/assets/142437c3-039d-4928-92cd-43168159a61d" />

---

## Technology Stack

  **Backend:** FastAPI, Python 3.10+, Uvicorn  
  **OCR Engines:** Mistral  
  **LLM Integration:** openai/gpt-oss-20b (for structured JSON)  
  **Data Processing:** Pandas, Regex, NumPy  
  **Testing:** Pytest  

---

## Key Features

- Multi-engine OCR pipeline (Tesseract + EasyOCR + Mistral)
- Intelligent text normalization and error correction
- Entity extraction for ID, financial, and employment documents
- Cross-document consistency checks
- REST API for easy integration
- JSON and text-based structured outputs
- Logging and performance metrics

---

## Project Structure
```bash
doc-verification_system/
├─ README.md
├─ app.py
├─ requirements.txt
├─ config.yml
├─ run_pipeline.py
├─ api/
│  └─ main.py
├─ verifier/
│  ├─ __init__.py
│  ├─ ocr/
│  │  ├─ __init__.py
│  │  ├─ mistral_ocr.py
│  │  ├─ mistral_ocr_enhanced.py
│  │  └─ preproc.py
│  ├─ normalize/
│  │  ├─ __init__.py
│  │  ├─ cleaners.py
│  │  └─ normalizers.py
│  ├─ extract/
│  │  ├─ __init__.py
│  │  ├─ regex_extractors.py
│  │  └─ groq_extractors.py
│  ├─ verify/
│  │  ├─ __init__.py
│  │  └─ rules.py
│  ├─ io/
│  │  ├─ __init__.py
│  │  └─ storage.py
│  └─ utils/
│     ├─ __init__.py
│     └─ logger.py
├─ tests/
│  ├─ test_normalizers.py
│  ├─ test_extractors.py
│  └─ test_verification_rules.py
├─ sample_output.json
├─ metrics/
│  ├─ ocr/
│  └─ evaluation/
└─ sample_dataset_placeholder/
   └─ README.md
```
---

## API Endpoints

| Endpoint       | Method | Request                                              | Description                                                                                         | Response |
|----------------|--------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------|----------|
| `/verify`      | POST   | `file` (ZIP upload), `use_llm` (optional, bool)     | Upload a ZIP containing folders for each `person_id` with expected document images. Runs the document verification pipeline. | JSON list of `VerificationResponse` objects per person:<br>- `person_id`<br>- `extracted_data`<br>- `verification_results`<br>- `overall_status`<br>- `logs`<br>- `ocr_results`<br>- `ocr_engines_used` |
| `/health`      | GET    | None                                                 | Checks system health and availability of GPU, Mistral API key, and local OCR libraries (Tesseract/EasyOCR are removed). | `HealthResponse` object:<br>- `status` (`"healthy"`)<br>- `tesseract_available` (False)<br>- `easyocr_available` (False)<br>- `gpu_available` (True/False)<br>- `mistral_api_key_present` (True/False) |
| `/metrics`     | GET    | None                                                 | Returns current OCR/system metrics if available (`metrics/ocr/comparison_report.json`).              | JSON metrics or message if no metrics available. |

**Notes:**

- Only ZIP uploads are supported for `/verify`. Each folder inside the ZIP should correspond to a `person_id` and contain the expected documents (e.g., government ID, bank statement, employment letter).
- Large file uploads are restricted by `MAX_UPLOAD_SIZE_BYTES` (default 50 MB).  
- `/health` always returns `tesseract_available = False` and `easyocr_available = False` since these libraries were removed.  
- Use `use_llm=True` if you want the pipeline to leverage LLM-based extraction (requires configured Groq/OpenAI API key).  

