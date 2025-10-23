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

## Table of Contents
- [Overview](#overview)
- [Project Demo](#project-demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)

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

## 🎥 Project Demo

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
4. Run the project with:
   ```bash
   python run_pipeline.py --input sample_dataset_placeholder --output results.json
