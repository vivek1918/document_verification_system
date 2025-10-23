"""
LLM-based entity extraction (optional).
"""

import re
import json
from typing import Dict, Any, Optional
from verifier.utils.logger import get_logger
import requests
import time
import os

logger = get_logger(__name__)

def enhance_extraction_with_llm(extracted: Dict[str, Any], text: str, doc_type: str) -> Dict[str, Any]:
    """
    Enhance extraction results using LLM.
    
    Args:
        extracted: Current extraction results
        text: Raw OCR text
        doc_type: Document type
        
    Returns:
        Enhanced extraction results
    """
    try:
        # Try Gemini first if configured
        gemini_result = call_gemini_api(text, doc_type)
        if gemini_result:
            return merge_extractions(extracted, gemini_result, "gemini")
        
        # Fall back to Hugging Face
        hf_result = call_huggingface_model(text, doc_type)
        if hf_result:
            return merge_extractions(extracted, hf_result, "huggingface")
            
    except Exception as e:
        logger.warning(f"LLM enhancement failed: {e}")
    
    return extracted

def call_gemini_api(text: str, doc_type: str) -> Optional[Dict[str, Any]]:
    """
    Call Gemini API for entity extraction.
    
    Args:
        text: OCR text to analyze
        doc_type: Document type
        
    Returns:
        Extracted entities or None
    """
    try:
        import google.generativeai as genai
        from config import load_config
        
        config = load_config()
        api_key = config.get('llm', {}).get('gemini_api_key')
        
        if not api_key:
            logger.warning("Gemini API key not configured")
            return None
        
        genai.configure(api_key=api_key)
        
        # Create prompt
        prompt = f"""
        Extract the following entities from this {doc_type} document text:
        - full_name
        - father_name  
        - date_of_birth (format: YYYY-MM-DD)
        - address (as JSON: {{house, street, city, state, pincode}})
        - phone_number
        - email_address
        - aadhaar_number (12 digits)
        - pan_number (format: ABCDE1234F)
        - employee_id
        - account_number
        
        Document text:
        {text[:4000]}  # Limit text length
        
        Return ONLY valid JSON with the structure:
        {{
            "full_name": {{"value": "...", "confidence": "low/med/high", "source": "gemini"}},
            "father_name": {{...}},
            ...
        }}
        
        If a field is not found, set value to null.
        """
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_text = extract_json_from_text(response.text)
        if json_text:
            result = json.loads(json_text)
            # Validate all fields
            return validate_llm_output(result, text)
            
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
    
    return None

def call_huggingface_model(text: str, doc_type: str) -> Optional[Dict[str, Any]]:
    """
    Call Hugging Face model for entity extraction.
    
    Args:
        text: OCR text to analyze
        doc_type: Document type
        
    Returns:
        Extracted entities or None
    """
    try:
        from transformers import pipeline
        from config import load_config
        
        config = load_config()
        model_name = config.get('llm', {}).get('hf_model', 'microsoft/DialoGPT-medium')
        
        # This is a placeholder - you'd need a proper NER or extraction model
        # For demonstration, we'll use a text generation approach
        classifier = pipeline("text2text-generation", model=model_name)
        
        prompt = f"Extract personal details from this {doc_type}: {text[:1000]}"
        
        response = classifier(prompt, max_length=500)
        extracted_text = response[0]['generated_text']
        
        # Parse the response (this would need more sophisticated parsing)
        # For now, return None as this is just a placeholder
        logger.info(f"HF model response: {extracted_text[:200]}...")
        
    except Exception as e:
        logger.error(f"Hugging Face model call failed: {e}")
    
    return None

def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from LLM response text."""
    try:
        # Look for JSON pattern
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            # Validate it's parseable JSON
            json.loads(json_str)
            return json_str
    except:
        pass
    return None

def validate_llm_output(llm_output: Dict[str, Any], original_text: str) -> Dict[str, Any]:
    """
    Validate LLM output against regex patterns and assign confidence.
    
    Args:
        llm_output: Raw LLM output
        original_text: Original OCR text for context
        
    Returns:
        Validated and confidence-assigned output
    """
    validated = {}
    
    # Import normalizers for validation
    from verifier.normalize import normalizers
    
    for field_name, field_data in llm_output.items():
        if not isinstance(field_data, dict):
            continue
            
        value = field_data.get('value')
        source = field_data.get('source', 'llm')
        
        # Validate based on field type
        confidence = "low"  # Default for LLM outputs
        
        if value is not None:
            if field_name == 'date_of_birth':
                normalized = normalizers.normalize_date(str(value))
                if normalized:
                    value = normalized
                    confidence = "high"
            elif field_name == 'pan_number':
                normalized = normalizers.normalize_pan(str(value))
                if normalized:
                    value = normalized
                    confidence = "high"
            elif field_name == 'aadhaar_number':
                normalized = normalizers.normalize_aadhaar(str(value))
                if normalized:
                    value = normalized
                    confidence = "high"
            elif field_name == 'phone_number':
                normalized = normalizers.normalize_phone(str(value))
                if normalized:
                    value = normalized
                    confidence = "high"
            elif field_name == 'email_address':
                if re.match(r'^[^@]+@[^@]+\.[^@]+$', str(value)):
                    confidence = "high"
        
        validated[field_name] = {
            "value": value,
            "confidence": confidence,
            "source": source,
            "raw_context": original_text[:100] + "..." if original_text else None
        }
    
    return validated

def merge_extractions(original: Dict[str, Any], llm_enhanced: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Merge original regex extractions with LLM enhancements.
    
    Args:
        original: Original regex extractions
        llm_enhanced: LLM-enhanced extractions
        source: LLM source ('gemini' or 'huggingface')
        
    Returns:
        Merged extraction results
    """
    merged = original.copy()
    
    for field_name, llm_data in llm_enhanced.items():
        original_data = original.get(field_name, {})
        original_value = original_data.get('value')
        llm_value = llm_data.get('value')
        
        # Prefer LLM value if original is None or LLM has higher confidence
        if original_value is None and llm_value is not None:
            merged[field_name] = llm_data
        elif original_value is not None and llm_value is not None:
            # If both have values, prefer the one with higher confidence
            original_conf = original_data.get('confidence', 'low')
            llm_conf = llm_data.get('confidence', 'low')
            
            confidence_order = {'low': 0, 'medium': 1, 'high': 2}
            if confidence_order.get(llm_conf, 0) > confidence_order.get(original_conf, 0):
                merged[field_name] = llm_data
    
    return merged