"""
Unit tests for normalization functions.
"""

import pytest
from verifier.normalize import normalizers

def test_normalize_date():
    """Test date normalization."""
    # Test various date formats
    assert normalizers.normalize_date("15-08-1947") == "1947-08-15"
    assert normalizers.normalize_date("15/08/1947") == "1947-08-15"
    assert normalizers.normalize_date("1947-08-15") == "1947-08-15"
    assert normalizers.normalize_date("15-Aug-1947") is None  # Text months not supported
    
    # Test invalid dates
    assert normalizers.normalize_date("32-13-2020") is None
    assert normalizers.normalize_date("invalid") is None

def test_normalize_phone():
    """Test phone number normalization."""
    # Test Indian numbers
    assert normalizers.normalize_phone("9876543210") == "+919876543210"
    assert normalizers.normalize_phone("+91-9876543210") == "+919876543210"
    assert normalizers.normalize_phone("09876543210") == "+9109876543210"
    
    # Test invalid numbers
    assert normalizers.normalize_phone("123") is None
    assert normalizers.normalize_phone("abcdef") is None

def test_normalize_pan():
    """Test PAN number normalization."""
    assert normalizers.normalize_pan("ABCDE1234F") == "ABCDE1234F"
    assert normalizers.normalize_pan("ab de 1234 f") == "ABDE1234F"  # Space removal
    assert normalizers.normalize_pan("ABCD12345F") is None  # Wrong format
    
def test_normalize_aadhaar():
    """Test Aadhaar number normalization."""
    assert normalizers.normalize_aadhaar("1234 5678 9012") == "123456789012"
    assert normalizers.normalize_aadhaar("123456789012") == "123456789012"
    assert normalizers.normalize_aadhaar("1234") is None  # Too short

def test_normalize_name():
    """Test name normalization."""
    assert normalizers.normalize_name("john doe") == "John Doe"
    assert normalizers.normalize_name("JOHN DOE") == "John Doe"
    assert normalizers.normalize_name("  john   doe  ") == "John Doe"