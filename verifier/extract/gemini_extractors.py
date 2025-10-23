"""
Gemini API for structured entity extraction.
"""

import json
import re
import time
from typing import Dict, Any, Optional
import google.generativeai as genai
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

class GeminiExtractor:
    def __init__(self, api_key: str):
        """Initialize Gemini API client."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        logger.info("Gemini extractor initialized")
    
    def extract_entities(self, ocr_text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured entities using Gemini API.
        
        Args:
            ocr_text: Raw OCR text from Google Vision
            doc_type: Type of document
            
        Returns:
            Dict of extracted fields with metadata
        """
        start_time = time.time()
        
        try:
            # Create detailed prompt for Gemini
            prompt = self._create_extraction_prompt(ocr_text, doc_type)
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Extract JSON from response
            extracted_data = self._parse_gemini_response(response_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Add Gemini source and confidence
            for field_name in extracted_data:
                if extracted_data[field_name].get('value'):
                    extracted_data[field_name].update({
                        'source': 'gemini',
                        'confidence': 'high'  # Gemini is generally reliable
                    })
            
            logger.info(f"Gemini extraction completed in {processing_time:.2f}ms for {doc_type}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            return self._get_empty_extraction()
    
    def _create_extraction_prompt(self, ocr_text: str, doc_type: str) -> str:
        """Create detailed prompt for entity extraction."""
        return f"""
        You are an expert at extracting structured information from document OCR text.
        
        DOCUMENT TYPE: {doc_type}
        OCR TEXT:
        {ocr_text[:4000]}  # Limit text length
        
        EXTRACTION TASK:
        Extract the following entities from the OCR text. Return ONLY valid JSON, no other text.
        
        REQUIRED FIELDS:
        1. full_name: Person's full name (string)
        2. father_name: Father's name (string or null)
        3. date_of_birth: Date in YYYY-MM-DD format (string or null)
        4. address: Structured address with house, street, city, state, pincode (object or null)
        5. phone_number: Phone number with country code (string or null)
        6. email_address: Email address (string or null)
        7. aadhaar_number: 12-digit Aadhaar number (string or null)
        8. pan_number: PAN number in ABCDE1234F format (string or null)
        9. employee_id: Employee ID (string or null)
        10. account_number: Bank account number (string or null)
        
        INSTRUCTIONS:
        - Return null for fields that are not found
        - Normalize dates to YYYY-MM-DD format
        - For addresses, extract and structure the components
        - For Indian phone numbers, ensure they start with +91
        - Validate Aadhaar (12 digits) and PAN (5 letters + 4 digits + 1 letter) formats
        - Be robust to OCR errors and variations in text formatting
        
        OUTPUT FORMAT (JSON only):
        {{
            "full_name": {{"value": "Extracted Name", "raw_context": "line where name was found"}},
            "father_name": {{"value": "Father's Name", "raw_context": "line where father name was found"}},
            "date_of_birth": {{"value": "1990-08-15", "raw_context": "line where DOB was found"}},
            "address": {{"value": {{"house": "123", "street": "Main St", "city": "Bangalore", "state": "Karnataka", "pincode": "560001"}}, "raw_context": "address line"}},
            "phone_number": {{"value": "+919876543210", "raw_context": "line where phone was found"}},
            "email_address": {{"value": "email@example.com", "raw_context": "line where email was found"}},
            "aadhaar_number": {{"value": "123456789012", "raw_context": "line where Aadhaar was found"}},
            "pan_number": {{"value": "ABCDE1234F", "raw_context": "line where PAN was found"}},
            "employee_id": {{"value": "EMP001", "raw_context": "line where employee ID was found"}},
            "account_number": {{"value": "12345678901234", "raw_context": "line where account number was found"}}
        }}
        
        IMPORTANT: Return ONLY the JSON object, no other text or explanations.
        """
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response and extract JSON."""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            
            # Parse JSON
            extracted_data = json.loads(cleaned_text)
            
            # Ensure all fields are present
            return self._validate_extraction(extracted_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            return self._get_empty_extraction()
        except Exception as e:
            logger.error(f"Error processing Gemini response: {e}")
            return self._get_empty_extraction()
    
    def _validate_extraction(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize extracted data."""
        validated = {}
        
        expected_fields = [
            'full_name', 'father_name', 'date_of_birth', 'address',
            'phone_number', 'email_address', 'aadhaar_number', 'pan_number',
            'employee_id', 'account_number'
        ]
        
        for field in expected_fields:
            if field in extracted_data and isinstance(extracted_data[field], dict):
                validated[field] = extracted_data[field]
            else:
                validated[field] = {"value": None, "raw_context": None}
        
        return validated
    
    def _get_empty_extraction(self) -> Dict[str, Any]:
        """Return empty extraction structure."""
        return {
            'full_name': {"value": None, "raw_context": None},
            'father_name': {"value": None, "raw_context": None},
            'date_of_birth': {"value": None, "raw_context": None},
            'address': {"value": None, "raw_context": None},
            'phone_number': {"value": None, "raw_context": None},
            'email_address': {"value": None, "raw_context": None},
            'aadhaar_number': {"value": None, "raw_context": None},
            'pan_number': {"value": None, "raw_context": None},
            'employee_id': {"value": None, "raw_context": None},
            'account_number': {"value": None, "raw_context": None}
        }

# Global instance
_gemini_extractor = None

def get_gemini_extractor(api_key: str) -> GeminiExtractor:
    """Get or create Gemini extractor instance."""
    global _gemini_extractor
    if _gemini_extractor is None:
        _gemini_extractor = GeminiExtractor(api_key)
    return _gemini_extractor

def extract_with_gemini(ocr_text: str, doc_type: str, api_key: str) -> Dict[str, Any]:
    """
    Extract entities using Gemini API.
    
    Args:
        ocr_text: Raw OCR text
        doc_type: Document type
        api_key: Gemini API key
        
    Returns:
        Dict of extracted entities
    """
    extractor = get_gemini_extractor(api_key)
    return extractor.extract_entities(ocr_text, doc_type)