"""
Main entity extraction orchestrator - uses Mistral API OCR + Groq API.
"""

from typing import Dict, Any
from verifier.utils.logger import get_logger
from verifier.extract.groq_extractors import extract_with_groq

logger = get_logger(__name__)

def extract_entities(ocr_results: Dict[str, Any], doc_type: str, use_groq: bool = True, groq_api_key: str = None, groq_model: str = "llama2-70b-4096") -> Dict[str, Any]:
    """
    Extract structured entities from OCR results using Groq API.
    
    Args:
        ocr_results: Dict with OCR results (primarily from Mistral API)
        doc_type: Type of document
        use_groq: Whether to use Groq for extraction
        groq_api_key: Groq API key
        groq_model: Groq model to use
        
    Returns:
        Dict of extracted fields with metadata
    """
    # Get OCR text from Mistral API (primary) or other engines
    ocr_text = ""
    
    # Prefer Mistral API if available
    if 'mistral' in ocr_results and ocr_results['mistral'].get('success'):
        ocr_text = ocr_results['mistral'].get('raw_text', '')
        logger.info(f"Using Mistral API OCR text: {len(ocr_text)} chars")
    
    # Fallback to enhanced Mistral if available
    elif 'mistral_enhanced' in ocr_results and ocr_results['mistral_enhanced'].get('success'):
        ocr_text = ocr_results['mistral_enhanced'].get('raw_text', '')
        logger.info(f"Using Enhanced Mistral OCR text: {len(ocr_text)} chars")
    
    # Fallback to other OCR engines
    if not ocr_text:
        for engine, result in ocr_results.items():
            if result.get('success') and result.get('raw_text'):
                ocr_text = result.get('raw_text', '')
                logger.info(f"Using {engine} OCR text: {len(ocr_text)} chars")
                break
    
    if not ocr_text:
        logger.warning(f"No OCR text available for {doc_type}")
        return _get_empty_extraction()
    
    # Use Groq for extraction if enabled and API key provided
    if use_groq and groq_api_key:
        try:
            logger.info(f"Using Groq ({groq_model}) for entity extraction on {doc_type}")
            extracted = extract_with_groq(ocr_text, doc_type, groq_api_key, groq_model)
            
            # Validate that we got some meaningful extraction
            extracted_count = sum(1 for field in extracted.values() if field.get('value'))
            if extracted_count > 0:
                logger.info(f"Groq extracted {extracted_count} fields for {doc_type}")
                return extracted
            else:
                logger.warning(f"Groq extraction failed for {doc_type}, falling back to regex")
                
        except Exception as e:
            logger.error(f"Groq extraction failed: {e}, falling back to regex")
    
    # Fallback to regex extraction
    return _extract_with_regex_fallback(ocr_text, doc_type)

def _get_empty_extraction() -> Dict[str, Any]:
    """Return empty extraction structure."""
    return {
        'full_name': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'father_name': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'date_of_birth': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'address': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'phone_number': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'email_address': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'aadhaar_number': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'pan_number': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'employee_id': {"value": None, "raw_context": None, "confidence": "low", "source": "none"},
        'account_number': {"value": None, "raw_context": None, "confidence": "low", "source": "none"}
    }

def _extract_with_regex_fallback(ocr_text: str, doc_type: str) -> Dict[str, Any]:
    """Fallback regex extraction."""
    logger.warning("Using regex fallback extraction")
    # Import and use your existing regex functions here
    # For now, return empty extraction
    return _get_empty_extraction()