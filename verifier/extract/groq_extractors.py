"""
Groq API for structured entity extraction using LLama2/Mixtral models.
"""

import json
import re
import time
from typing import Dict, Any, Optional
from groq import Groq
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

class GroqExtractor:
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        """Initialize Groq API client."""
        self.client = Groq(api_key=api_key)
        self.model = model
        logger.info(f"Groq extractor initialized with model: {model}")
    
    def extract_entities(self, ocr_text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract structured entities using Groq API.
        
        Args:
            ocr_text: Raw OCR text from Mistral OCR
            doc_type: Type of document
            
        Returns:
            Dict of extracted fields with metadata
        """
        start_time = time.time()
        
        try:
            # Create detailed prompt for Groq
            messages = self._create_extraction_messages(ocr_text, doc_type)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content
            
            # Extract JSON from response
            extracted_data = self._parse_groq_response(response_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Add Groq source and confidence
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
        system_prompt = """You are an expert at extracting structured information from document OCR text. 
        Extract the requested entities and return ONLY valid JSON format, no other text or explanations."""
        
        user_prompt = f"""
        DOCUMENT TYPE: {doc_type}
        
        OCR TEXT:
        {ocr_text[:3500]}  # Limit text length for Groq
        
        EXTRACTION TASK:
        Extract the following entities from the OCR text. Return ONLY valid JSON.
        
        REQUIRED FIELDS:
        1. full_name: Person's full name (string or null)
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
        - Include raw_context showing the original text where each field was found
        
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
        
        IMPORTANT: Return ONLY the JSON object, no other text, no code blocks, no explanations.
        """
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _parse_groq_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Groq response and extract JSON."""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Remove any markdown code blocks if present
            cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'```\s*', '', cleaned_text)
            
            # Find JSON in the response (sometimes Groq adds explanations)
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                cleaned_text = json_match.group()
            
            # Parse JSON
            extracted_data = json.loads(cleaned_text)
            
            # Ensure all fields are present
            return self._validate_extraction(extracted_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Groq response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            # Try to extract any valid JSON parts
            try:
                # Look for individual field patterns as fallback
                return self._extract_fields_fallback(response_text)
            except:
                return self._get_empty_extraction()
        except Exception as e:
            logger.error(f"Error processing Groq response: {e}")
            return self._get_empty_extraction()
    
    def _extract_fields_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback extraction if JSON parsing fails."""
        extracted = self._get_empty_extraction()
        
        # Simple pattern matching as fallback
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
_groq_extractor = None

def get_groq_extractor(api_key: str, model: str = "llama2-70b-4096") -> GroqExtractor:
    """Get or create Groq extractor instance."""
    global _groq_extractor
    if _groq_extractor is None:
        _groq_extractor = GroqExtractor(api_key, model)
    return _groq_extractor

def extract_with_groq(ocr_text: str, doc_type: str, api_key: str, model: str = "llama2-70b-4096") -> Dict[str, Any]:
    """
    Extract entities using Groq API.
    
    Args:
        ocr_text: Raw OCR text
        doc_type: Document type
        api_key: Groq API key
        model: Groq model to use
        
    Returns:
        Dict of extracted entities
    """
    extractor = get_groq_extractor(api_key, model)
    return extractor.extract_entities(ocr_text, doc_type)