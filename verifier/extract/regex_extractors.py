# regex_extractors.py
"""
Main entity extraction orchestrator - uses Mistral OCR + Groq API.
"""

from typing import Dict, Any
from verifier.utils.logger import get_logger
from verifier.extract.groq_extractors import extract_with_groq
import json

logger = get_logger(__name__)


def extract_entities(
    ocr_results: Dict[str, Any],
    doc_type: str,
    use_groq: bool = True,
    groq_api_key: str = None,
    groq_model: str = "openai/gpt-oss-20b"
) -> Dict[str, Any]:
    """
    Extract structured entities from OCR results using Groq API or fallback regex.

    Args:
        ocr_results: Dict with OCR results (mistral, mistral_enhanced, tesseract, easyocr)
        doc_type: Type of document
        use_groq: Whether to use Groq for extraction
        groq_api_key: Groq API key
        groq_model: Groq model to use

    Returns:
        Dict of extracted fields with metadata
    """
    ocr_text = ""

    # Prefer Mistral OCR first
    if 'mistral' in ocr_results and isinstance(ocr_results['mistral'], dict) and ocr_results['mistral'].get('success'):
        ocr_text = _extract_raw_text(ocr_results['mistral'])

    # Then Enhanced Mistral OCR
    elif 'mistral_enhanced' in ocr_results and isinstance(ocr_results['mistral_enhanced'], dict) and ocr_results['mistral_enhanced'].get('success'):
        ocr_text = _extract_raw_text(ocr_results['mistral_enhanced'])

    # Fallback: first successful OCR engine
    if not ocr_text:
        for engine, result in ocr_results.items():
            if isinstance(result, dict) and result.get('success') and result.get('raw_text'):
                ocr_text = _extract_raw_text(result)
                logger.info(f"Using {engine} OCR text: {len(ocr_text)} chars")
                break

    if not ocr_text:
        logger.warning(f"No OCR text available for {doc_type}")
        return _get_empty_extraction()

    # --- Use Groq if enabled and API key available ---
    if use_groq and groq_api_key:
        try:
            logger.info(f"Using Groq ({groq_model}) for entity extraction on {doc_type}")
            extracted = extract_with_groq(str(ocr_text), doc_type, groq_api_key, groq_model)

            extracted_count = sum(1 for field in extracted.values() if field.get('value'))
            if extracted_count > 0:
                logger.info(f"Groq extracted {extracted_count} fields for {doc_type}")
                return extracted
            else:
                logger.warning(f"Groq extraction returned no values for {doc_type}, falling back to regex")
        except Exception as e:
            logger.error(f"Groq extraction failed: {e}, falling back to regex")

    # --- Fallback to regex extraction ---
    return _extract_with_regex_fallback(str(ocr_text), doc_type)


def _extract_raw_text(ocr_result: Dict[str, Any]) -> str:
    """Ensure we always return a clean string from OCR result dict."""
    if not ocr_result:
        return ""

    raw = None
    if isinstance(ocr_result, dict):
        raw = ocr_result.get("raw_text", "")

    # Force conversion to a valid string
    if isinstance(raw, str):
        logger.info(f"Using OCR text: {len(raw)} chars")
        return raw
    else:
        try:
            import json
            return json.dumps(raw, ensure_ascii=False)
        except Exception:
            return str(raw)


def _get_empty_extraction() -> Dict[str, Any]:
    """Return empty extraction structure."""
    fields = [
        'full_name', 'father_name', 'date_of_birth', 'address', 'phone_number',
        'email_address', 'aadhaar_number', 'pan_number', 'employee_id', 'account_number'
    ]
    return {f: {"value": None, "raw_context": None, "confidence": "low", "source": "none"} for f in fields}


def _extract_with_regex_fallback(ocr_text: str, doc_type: str) -> Dict[str, Any]:
    """Fallback regex extraction. Use actual regex rules if implemented."""
    logger.warning(f"Using regex fallback extraction for {doc_type}")
    # TODO: Replace with real regex logic
    return _get_empty_extraction()