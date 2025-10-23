"""
Field-specific normalization functions.
"""

import re
from datetime import datetime
from typing import Optional, Dict, Any
from verifier.utils.logger import get_logger
from verifier.normalize.cleaners import apply_confusion_corrections, clean_whitespace

logger = get_logger(__name__)

# add near the top of your file
MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

# Expand date patterns to catch month-name variants (hyphens, commas, etc.)
DATE_PATTERNS = [
    r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})',                # DD-MM-YYYY, DD/MM/YYYY
    r'(\d{2,4})[-/](\d{1,2})[-/](\d{1,2})',                # YYYY-MM-DD, YYYY/MM/DD
    r'(\d{1,2})\s+(\w+)\s+(\d{2,4})',                      # DD Month YYYY
    r'(\d{1,2})[-\s](\w+)[-/\s](\d{2,4})',                 # 12-Jan-2020 or 12 Jan 20
    r'(\w+)\s+(\d{1,2}),?\s+(\d{2,4})',                    # Month DD, YYYY  (e.g., January 12, 2020)
]

PAN_PATTERN = r'[A-Z]{5}[0-9]{4}[A-Z]'
AADHAAR_PATTERN = r'\b[0-9]{4}\s?[0-9]{4}\s?[0-9]{4}\b'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

def normalize_phone(raw_phone: str) -> Optional[str]:
    """
    Normalize phone number to standard format.
    
    Args:
        raw_phone: Raw phone string
        
    Returns:
        Normalized phone number or None
    """
    if not raw_phone:
        return None
    
    # Clean and extract digits
    cleaned = apply_confusion_corrections(raw_phone, 'numeric')
    digits = re.sub(r'\D', '', cleaned)
    
    # Handle Indian phone numbers specifically
    if len(digits) == 10:
        return f"+91{digits}"  # Assume India number
    elif len(digits) == 11 and digits.startswith('0'):
        return f"+91{digits[1:]}"
    elif len(digits) == 12 and digits.startswith('91'):
        return f"+{digits}"
    elif len(digits) == 13 and digits.startswith('+91'):
        return f"+{digits[1:]}"
    elif len(digits) == 13 and not digits.startswith('91'):
        # This might be a malformed number, try to extract last 10 digits
        last_10 = digits[-10:]
        if last_10.isdigit():
            return f"+91{last_10}"
    elif 10 <= len(digits) <= 15:
        return f"+{digits}"
    else:
        logger.debug(f"Invalid phone number length: {len(digits)} for {raw_phone}")
        return None

def normalize_pan(raw_pan: str) -> Optional[str]:
    """
    Normalize PAN number.
    
    Args:
        raw_pan: Raw PAN string
        
    Returns:
        Normalized PAN or None
    """
    if not raw_pan:
        return None
    
    # Clean and uppercase
    cleaned = apply_confusion_corrections(raw_pan.upper(), 'alphanumeric')
    cleaned = re.sub(r'[^A-Z0-9]', '', cleaned)
    
    # Validate format
    if re.match(PAN_PATTERN, cleaned):
        return cleaned
    else:
        logger.debug(f"Invalid PAN format: {raw_pan}")
        return None

def normalize_aadhaar(raw_aadhaar: str) -> Optional[str]:
    """
    Normalize Aadhaar number (12 digits).
    
    Args:
        raw_aadhaar: Raw Aadhaar string
        
    Returns:
        Normalized Aadhaar or None
    """
    if not raw_aadhaar:
        return None
    
    # Clean and extract digits
    cleaned = apply_confusion_corrections(raw_aadhaar, 'numeric')
    digits = re.sub(r'\D', '', cleaned)
    
    if len(digits) == 12:
        return digits
    else:
        logger.debug(f"Invalid Aadhaar length: {len(digits)} for {raw_aadhaar}")
        return None

def canonicalize_address(raw_address: str) -> Dict[str, Optional[str]]:
    """
    Parse address into structured components.
    
    Args:
        raw_address: Raw address string
        
    Returns:
        Dict with house, street, city, state, pincode
    """
    if not raw_address:
        return {
            "house": None,
            "street": None, 
            "city": None,
            "state": None,
            "pincode": None
        }
    
    cleaned = clean_whitespace(raw_address)
    
    # Extract pincode (6 digits)
    pincode_match = re.search(r'\b[1-9][0-9]{5}\b', cleaned)
    pincode = pincode_match.group() if pincode_match else None
    
    # Remove pincode from address for better parsing
    address_without_pincode = re.sub(r'\b[1-9][0-9]{5}\b', '', cleaned).strip()
    
    # Simple heuristic parsing
    parts = [part.strip() for part in address_without_pincode.split(',')]
    
    result = {
        "house": None,
        "street": None,
        "city": None,
        "state": None,
        "pincode": pincode
    }
    
    if len(parts) >= 2:
        # Try to extract house number from first part
        first_part = parts[0]
        house_match = re.search(r'^(\d+[A-Za-z]?)\b', first_part)
        if house_match:
            result["house"] = house_match.group(1)
            result["street"] = first_part.replace(house_match.group(1), '').strip()
        else:
            result["street"] = first_part
        
        # City is usually the part before state or the last meaningful part
        if len(parts) >= 3:
            result["city"] = parts[1]
            result["state"] = parts[2] if len(parts) > 2 else None
        else:
            result["city"] = parts[-1]
    else:
        # Single part address
        result["street"] = address_without_pincode
    
    # Common Indian cities and states for better identification
    indian_cities = ['bangalore', 'mumbai', 'delhi', 'chennai', 'kolkata', 'hyderabad', 'pune']
    indian_states = ['karnataka', 'maharashtra', 'tamil nadu', 'west bengal', 'andhra pradesh', 'delhi']
    
    # Improve city/state detection
    if result["city"]:
        city_lower = result["city"].lower()
        for city in indian_cities:
            if city in city_lower:
                result["city"] = city.title()
                break
    
    if result["state"]:
        state_lower = result["state"].lower()
        for state in indian_states:
            if state in state_lower:
                result["state"] = state.title()
                break
    elif result["city"] and not result["state"]:
        # Infer state from city
        city_state_map = {
            'bangalore': 'Karnataka',
            'mumbai': 'Maharashtra', 
            'delhi': 'Delhi',
            'chennai': 'Tamil Nadu',
            'kolkata': 'West Bengal',
            'hyderabad': 'Telangana',
            'pune': 'Maharashtra'
        }
        for city, state in city_state_map.items():
            if city in result["city"].lower():
                result["state"] = state
                break
    
    return result

def normalize_employee_id(raw_emp_id: str) -> Optional[str]:
    """
    Normalize employee ID.
    
    Args:
        raw_emp_id: Raw employee ID string
        
    Returns:
        Normalized employee ID or None
    """
    if not raw_emp_id:
        return None
    
    # Clean whitespace and convert to uppercase
    cleaned = clean_whitespace(raw_emp_id.upper())
    
    # Remove common prefixes if they exist
    cleaned = re.sub(r'^(EMP|ID|STAFF|EMPLOYEE)[\s\-_]*', '', cleaned, flags=re.IGNORECASE)
    
    # Fix common OCR errors
    corrections = {
        'O': '0',  # Letter O to zero
        'I': '1',  # Letter I to one
        'L': '1',  # Letter L to one
        'S': '5',  # Letter S to five
        'B': '8',  # Letter B to eight
    }
    
    for wrong, correct in corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    
    # Remove any remaining special characters except hyphens (for IDs like EMP-001)
    cleaned = re.sub(r'[^A-Z0-9\-]', '', cleaned)
    
    # Validate length and format
    if len(cleaned) >= 2:  # Reasonable minimum length for employee ID
        return cleaned
    else:
        logger.debug(f"Employee ID too short: {cleaned} (original: {raw_emp_id})")
        return None
    
def normalize_email(raw_email: str) -> Optional[str]:
    """
    Normalize email address.
    
    Args:
        raw_email: Raw email string
        
    Returns:
        Normalized email or None
    """
    if not raw_email:
        return None
    
    # Clean whitespace and convert to lowercase
    cleaned = clean_whitespace(raw_email.lower())
    
    # Fix common OCR errors in emails
    corrections = {
        ' ': '.',  # Spaces often represent dots in names
        '..': '.',  # Remove double dots
        ' .': '.',  # Fix space before dot
        '. ': '.',  # Fix space after dot
    }
    
    for wrong, correct in corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    
    # Fix common domain OCR errors
    domain_corrections = {
        'gma1l.': 'gmail.',
        'gmai1.': 'gmail.', 
        'yah0o.': 'yahoo.',
        'yaho0.': 'yahoo.',
        'hotma1l.': 'hotmail.',
        'out1ook.': 'outlook.',
        'ema1l.': 'email.',
    }
    
    for wrong, correct in domain_corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    
    # Ensure proper email format
    if '@' not in cleaned:
        return None
    
    # Basic email validation - more lenient version
    local_part, domain_part = cleaned.split('@', 1)
    
    # Validate local part (before @)
    if not re.match(r'^[a-zA-Z0-9._%+-]+$', local_part):
        return None
    
    # Validate domain part (after @)
    if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain_part):
        # Try to fix missing TLD
        if '.' not in domain_part:
            domain_part += '.com'
            cleaned = f"{local_part}@{domain_part}"
    
    # Final validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, cleaned):
        return cleaned
    else:
        logger.debug(f"Invalid email format after normalization: {cleaned}")
        return None

def _normalize_two_digit_year(y: str) -> str:
    """Convert 2-digit year to 4-digit (assume 2000s)."""
    y = y.strip()
    if len(y) == 2:
        return f"20{y}"
    return y.zfill(4)

def _strip_ordinal(day_str: str) -> str:
    """Remove ordinal suffixes like 1st, 2nd, 3rd, 4th."""
    return re.sub(r'(st|nd|rd|th)$', '', day_str, flags=re.IGNORECASE)

def normalize_name(raw_name: str) -> str:
    """
    Normalize person name (title case, clean whitespace).
    
    Args:
        raw_name: Raw name string
        
    Returns:
        Normalized name
    """
    if not raw_name:
        return ""
    
    cleaned = clean_whitespace(raw_name)
    # Convert to title case, but be smart about initials
    parts = []
    for part in cleaned.split():
        if len(part) == 1 and part.isalpha():
            parts.append(part.upper() + '.')  # Initial with period
        else:
            parts.append(part.title())
    
    return ' '.join(parts)

def normalize_date(raw_date: str) -> Optional[str]:
    """
    Normalize date to YYYY-MM-DD format.
    Supports numeric dates and month-name dates (Jan, January, etc.).
    """
    if not raw_date:
        return None

    cleaned = clean_whitespace(raw_date)
    cleaned = cleaned.replace(',', ' ')  # allow "January 12, 2020" forms

    for pattern in DATE_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue

        try:
            parts = [p.strip() for p in match.groups()]

            # Normalize parts: remove ordinals from day
            parts = [ _strip_ordinal(p) for p in parts ]

            # Detect which part is year / month / day by content
            # If any part contains letters -> it's a month name
            month_idx = None
            for i, p in enumerate(parts):
                if re.search(r'[A-Za-z]', p):
                    month_idx = i
                    break

            if month_idx is not None:
                # We have a month-name style date. Map month name to number.
                # Identify day and year among remaining parts
                # Common layouts we covered:
                #  - [DD, Month, YYYY]  -> parts[0]=day, parts[1]=month, parts[2]=year
                #  - [Month, DD, YYYY]  -> parts[0]=month, parts[1]=day, parts[2]=year
                #  - [DD, Month, YY]    -> same with 2-digit year

                # month string -> month number
                month_str = parts[month_idx].lower()
                month_num = MONTH_MAP.get(month_str[:3], None) or MONTH_MAP.get(month_str, None)
                # try full lookup fallback
                if not month_num:
                    month_num = MONTH_MAP.get(month_str, None)

                if not month_num:
                    # try to match common abbreviation first 3 letters
                    month_num = MONTH_MAP.get(month_str[:3], None)

                if not month_num:
                    # unknown month name — skip this pattern
                    continue

                # Determine day and year indexes (the other two)
                idxs = [0,1,2]
                idxs.remove(month_idx)
                # heuristics: year is the one with length 4 or numeric > 31
                year_idx = None
                day_idx = None
                for i in idxs:
                    p = parts[i]
                    if re.fullmatch(r'\d{4}', p):
                        year_idx = i
                    elif re.fullmatch(r'\d{2}', p) and int(p) > 31:
                        # two-digit but looks like year
                        year_idx = i
                if year_idx is None:
                    # fallback: last part often year
                    year_idx = idxs[-1]
                idxs.remove(year_idx)
                day_idx = idxs[0]

                day = parts[day_idx].zfill(2)
                year = _normalize_two_digit_year(parts[year_idx])
                month = month_num

                # validate and return
                datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                return f"{year}-{month}-{day}"

            else:
                # Numeric-style date (no month names)
                # Determine if format is YYYY-MM-DD or DD-MM-YYYY by length
                a, b, c = parts
                # If first part length == 4 -> assume YYYY-MM-DD
                if len(a) == 4:
                    year, month, day = a, b, c
                elif len(c) == 4:
                    day, month, year = a, b, c
                else:
                    # two-digit year handling
                    day, month, year = a, b, c
                    year = _normalize_two_digit_year(year)

                # zfill and validate
                month = month.zfill(2)
                day = day.zfill(2)
                year = year.zfill(4)

                datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                return f"{year}-{month}-{day}"

        except (ValueError, IndexError):
            # try next pattern if conversion/validation fails
            continue

    logger.debug(f"Could not normalize date: {raw_date}")
    return None