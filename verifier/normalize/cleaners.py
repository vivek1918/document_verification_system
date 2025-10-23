"""
Text cleaning and confusion correction utilities.
"""

import re
from typing import Optional, Dict, Any
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

# Character confusion mappings
CONFUSION_MAP = {
    'O': '0', 'o': '0',  # Letter O to zero
    'S': '5', 's': '5',  # Letter S to five
    'I': '1', 'l': '1',  # Letter I/l to one
    'B': '8', 'Z': '2',  # Letter B to eight, Z to two
    ' ': '',  # Remove spaces for certain fields
}

# Reverse mapping for alpha contexts
REVERSE_CONFUSION_MAP = {
    '0': 'O', '5': 'S', '1': 'I', '8': 'B', '2': 'Z'
}


def apply_confusion_corrections(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply character confusion corrections to extracted entity dictionary.
    
    Args:
        extracted_data: Dictionary of extracted entities with structure:
                       {field_name: {"value": ..., "raw_context": ..., ...}}
        
    Returns:
        Dictionary with corrected values
    """
    if not isinstance(extracted_data, dict):
        logger.warning(f"apply_confusion_corrections received non-dict: {type(extracted_data)}")
        return extracted_data
    
    corrected_data = {}
    
    # Field-specific correction hints
    field_hints = {
        'aadhaar_number': 'numeric',
        'pan_number': 'alphanumeric',
        'phone_number': 'numeric',
        'account_number': 'numeric',
        'employee_id': 'alphanumeric',
        'full_name': 'alpha',
        'father_name': 'alpha',
        'date_of_birth': 'numeric',
    }
    
    for field_name, field_data in extracted_data.items():
        if not isinstance(field_data, dict):
            corrected_data[field_name] = field_data
            continue
        
        # Copy the field data
        corrected_field = field_data.copy()
        
        # Apply corrections to the value if it's a string
        if 'value' in field_data and field_data['value']:
            value = field_data['value']
            
            # Only correct string values
            if isinstance(value, str):
                hint = field_hints.get(field_name, None)
                corrected_value = correct_text(value, hint)
                
                if corrected_value != value:
                    logger.debug(f"Corrected {field_name}: '{value}' -> '{corrected_value}'")
                    corrected_field['value'] = corrected_value
        
        corrected_data[field_name] = corrected_field
    
    return corrected_data


def correct_text(text: str, field_hint: Optional[str] = None) -> str:
    """
    Apply character confusion corrections based on field context.
    
    Args:
        text: Input text to correct
        field_hint: Field type hint ('numeric', 'alpha', 'alphanumeric', None)
        
    Returns:
        Corrected text
    """
    if not text or not isinstance(text, str):
        return text
    
    original_text = text
    
    if field_hint == 'numeric':
        # Apply corrections for numeric fields
        for wrong, correct in CONFUSION_MAP.items():
            text = text.replace(wrong, correct)
    
    elif field_hint == 'alpha':
        # Apply reverse corrections for alpha fields
        for wrong, correct in REVERSE_CONFUSION_MAP.items():
            text = text.replace(wrong, correct)
    
    else:  # alphanumeric or unknown
        # Be more conservative, only apply obvious corrections
        corrections = {
            'O': '0',  # Capital O to zero in mixed contexts
            'l': '1',  # Lowercase L to one
            'I': '1',  # Capital I to one
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
    
    return text


def clean_whitespace(text: str) -> str:
    """Clean and normalize whitespace."""
    if not text:
        return text
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_characters(text: str, keep_chars: str = "") -> str:
    """
    Remove special characters, keeping specified ones.
    
    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., "@.-" for emails)
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Escape special regex characters in keep_chars
    escaped_keep = re.escape(keep_chars)
    pattern = f'[^a-zA-Z0-9{escaped_keep}]'
    
    cleaned = re.sub(pattern, '', text)
    return cleaned