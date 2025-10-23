"""
Unit tests for verification rules.
"""

import pytest
from verifier.verify.rules import verify_name_match, verify_dob_match, verify_pan_format

def test_name_match_rule():
    """Test name matching rule."""
    extracted_data = {
        "government_id": {
            "full_name": {"value": "John Doe", "confidence": "high", "source": "regex"}
        },
        "bank_statement": {
            "full_name": {"value": "John Doe", "confidence": "high", "source": "regex"}
        }
    }
    
    result = verify_name_match(extracted_data)
    assert result["status"] == "PASS"

def test_dob_match_rule():
    """Test DOB matching rule."""
    extracted_data = {
        "government_id": {
            "date_of_birth": {"value": "1990-08-15", "confidence": "high", "source": "regex"}
        },
        "employment_letter": {
            "date_of_birth": {"value": "1990-08-15", "confidence": "medium", "source": "regex"}
        }
    }
    
    result = verify_dob_match(extracted_data)
    assert result["status"] == "PASS"

def test_pan_format_rule():
    """Test PAN format validation rule."""
    extracted_data = {
        "government_id": {
            "pan_number": {"value": "ABCDE1234F", "confidence": "high", "source": "regex"}
        }
    }
    
    result = verify_pan_format(extracted_data)
    assert result["status"] == "PASS"