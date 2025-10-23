"""
Unit tests for entity extraction.
"""

import pytest
from verifier.extract.regex_extractors import extract_name, extract_dob, extract_pan

def test_extract_name():
    """Test name extraction from text."""
    text = "Name: John Doe\nAddress: Somewhere"
    result = extract_name(text, "government_id")
    assert result["value"] == "John Doe"
    assert result["confidence"] == "high"

def test_extract_dob():
    """Test date of birth extraction."""
    text = "Date of Birth: 15-08-1990\nName: John"
    result = extract_dob(text, "government_id")
    assert result["value"] == "1990-08-15"
    assert result["confidence"] == "high"

def test_extract_pan():
    """Test PAN number extraction."""
    text = "PAN: ABCDE1234F\nSome other text"
    result = extract_pan(text, "government_id")
    assert result["value"] == "ABCDE1234F"
    assert result["confidence"] == "high"