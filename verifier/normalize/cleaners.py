"""
Text cleaning and confusion correction utilities.
"""

import re
from typing import Optional
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

def apply_confusion_corrections(text: str, field_hint: Optional[str] = None) -> str:
    """
    Apply character confusion corrections based on field context.
    
    Args:
        text: Input text to correct
        field_hint: Field type hint ('numeric', 'alpha', 'alphanumeric', None)
        
    Returns:
        Corrected text
    """
    if not text:
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
    
    if text != original_text:
        logger.debug(f"Applied confusion correction: '{original_text}' -> '{text}'")
    
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