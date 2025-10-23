"""
Groq API for structured entity extraction using LLama2/Mixtral models.
"""

import json
import re
import time
from typing import Dict, Any
from groq import Groq
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

class GroqExtractor:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        """Initialize Groq API client."""
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Groq extractor initialized with model: {model}")
    
    def extract_entities(self, ocr_text: Any, doc_type: str) -> Dict[str, Any]:
        """Extract structured entities using Groq API."""
        start_time = time.time()

        # 🔒 Ensure OCR text is a string
        if not isinstance(ocr_text, str):
            try:
                ocr_text = json.dumps(ocr_text, ensure_ascii=False)
                logger.debug(f"Converted non-string OCR text to JSON string for {doc_type}")
            except Exception:
                ocr_text = str(ocr_text)

        try:
            messages = self._create_extraction_messages(ocr_text, doc_type)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content

            extracted_data = self._parse_groq_response(response_text)
            processing_time = (time.time() - start_time) * 1000

            for field_name in extracted_data:
                if extracted_data[field_name].get('value'):
                    extracted_data[field_name].update({
                        'source': 'groq',
                        'confidence': 'high'
                    })

            logger.info(f"Groq extraction completed in {processing_time:.2f}ms for {doc_type}")
            return extracted_data

        except Exception as e:
            logger.error(f"Groq extraction failed: {e}")
            return self._get_empty_extraction()
    
    def _create_extraction_messages(self, ocr_text: str, doc_type: str) -> list:
        """Create messages for entity extraction."""
        system_prompt = (
            "You are an expert at extracting structured information from document OCR text. "
            "Extract the requested entities and return ONLY valid JSON format."
        )

        user_prompt = f"""
        DOCUMENT TYPE: {doc_type}

        OCR TEXT:
        {ocr_text[:3500]}

        EXTRACTION TASK:
        Extract the following entities and return ONLY valid JSON.

        REQUIRED FIELDS:
        1. full_name
        2. father_name
        3. date_of_birth (YYYY-MM-DD)
        4. address
        5. phone_number
        6. email_address
        7. aadhaar_number
        8. pan_number
        9. employee_id
        10. account_number

        Return JSON exactly like:
        {{
            "full_name": {{"value": "John Doe", "raw_context": "line with name"}},
            "father_name": {{"value": "Richard Doe", "raw_context": "line"}},
            "date_of_birth": {{"value": "1990-01-01", "raw_context": "line"}},
            "address": {{"value": {{"city": "Bangalore"}}, "raw_context": "line"}},
            "phone_number": {{"value": "+919876543210", "raw_context": "line"}},
            "email_address": {{"value": "john@example.com", "raw_context": "line"}},
            "aadhaar_number": {{"value": "123456789012", "raw_context": "line"}},
            "pan_number": {{"value": "ABCDE1234F", "raw_context": "line"}},
            "employee_id": {{"value": "EMP001", "raw_context": "line"}},
            "account_number": {{"value": "9876543210", "raw_context": "line"}}
        }}

        Return ONLY the JSON, no explanations.
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _parse_groq_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Groq response and extract JSON."""
        try:
            cleaned_text = response_text.strip()
            cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)

            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group()

            extracted_data = json.loads(cleaned_text)
            return self._validate_extraction(extracted_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response as JSON: {e}")
            return self._extract_fields_fallback(response_text)
        except Exception as e:
            logger.error(f"Error processing Groq response: {e}")
            return self._get_empty_extraction()
    
    def _extract_fields_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback extraction if JSON parsing fails."""
        extracted = self._get_empty_extraction()
        patterns = {
            'full_name': r'"full_name"[^}]*"value"\s*:\s*"([^"]*)"',
            'date_of_birth': r'"date_of_birth"[^}]*"value"\s*:\s*"([^"]*)"',
            'phone_number': r'"phone_number"[^}]*"value"\s*:\s*"([^"]*)"',
            'email_address': r'"email_address"[^}]*"value"\s*:\s*"([^"]*)"',
            'aadhaar_number': r'"aadhaar_number"[^}]*"value"\s*:\s*"([^"]*)"',
            'pan_number': r'"pan_number"[^}]*"value"\s*:\s*"([^"]*)"',
        }
        for field, pattern in patterns.items():
            match = re.search(pattern, response_text)
            if match:
                extracted[field]['value'] = match.group(1)
                extracted[field]['source'] = 'groq_fallback'
                extracted[field]['confidence'] = 'medium'
        return extracted
    
    def _validate_extraction(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize extracted data."""
        expected_fields = [
            'full_name', 'father_name', 'date_of_birth', 'address',
            'phone_number', 'email_address', 'aadhaar_number',
            'pan_number', 'employee_id', 'account_number'
        ]
        validated = {}
        for f in expected_fields:
            if f in extracted_data and isinstance(extracted_data[f], dict):
                validated[f] = extracted_data[f]
            else:
                validated[f] = {"value": None, "raw_context": None}
        return validated
    
    def _get_empty_extraction(self) -> Dict[str, Any]:
        """Return empty extraction structure."""
        fields = [
            'full_name', 'father_name', 'date_of_birth', 'address',
            'phone_number', 'email_address', 'aadhaar_number',
            'pan_number', 'employee_id', 'account_number'
        ]
        return {f: {"value": None, "raw_context": None} for f in fields}


# --- Global helper methods ---

_groq_extractor = None

def get_groq_extractor(api_key: str, model: str = "openai/gpt-oss-20b") -> GroqExtractor:
    """Get or create Groq extractor instance."""
    global _groq_extractor
    if _groq_extractor is None:
        _groq_extractor = GroqExtractor(api_key, model)
    return _groq_extractor


def extract_with_groq(ocr_text: Any, doc_type: str, api_key: str, model: str = "openai/gpt-oss-20b") -> Dict[str, Any]:
    """Extract entities using Groq API."""
    # 🔒 Ensure string before passing to GroqExtractor
    if not isinstance(ocr_text, str):
        try:
            ocr_text = json.dumps(ocr_text, ensure_ascii=False)
        except Exception:
            ocr_text = str(ocr_text)

    extractor = get_groq_extractor(api_key, model)
    return extractor.extract_entities(ocr_text, doc_type)