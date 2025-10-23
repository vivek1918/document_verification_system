"""
Cross-document verification rules implementation.
"""

from typing import Dict, Any, List, Set, Tuple
from verifier.utils.logger import get_logger

logger = get_logger(__name__)

def verify_person(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all verification rules on extracted data.
    
    Args:
        extracted_data: Extracted data from all documents
        
    Returns:
        Dict with verification results for all rules
    """
    results = {}
    
    # Rule 1: Name matching
    results['rule_1_name_match'] = verify_name_match(extracted_data)
    
    # Rule 2: DOB matching
    results['rule_2_dob_match'] = verify_dob_match(extracted_data)
    
    # Rule 3: Address matching
    results['rule_3_address_match'] = verify_address_match(extracted_data)
    
    # Rule 4: Phone matching
    results['rule_4_phone_match'] = verify_phone_match(extracted_data)
    
    # Rule 5: Father's name matching
    results['rule_5_father_name_match'] = verify_father_name_match(extracted_data)
    
    # Rule 6: PAN format validation
    results['rule_6_pan_format'] = verify_pan_format(extracted_data)
    
    # Rule 7: Aadhaar format validation
    results['rule_7_aadhaar_format'] = verify_aadhaar_format(extracted_data)
    
    # Calculate overall status - MORE LENIENT VERSION
    # Require only key rules to pass for overall verification
    key_rules = ['rule_1_name_match', 'rule_2_dob_match']
    
    passed_key_rules = sum(1 for rule in key_rules 
                          if results.get(rule, {}).get('status') == 'PASS')
    
    total_extracted_fields = sum(
        1 for doc_data in extracted_data.values() 
        for field_data in doc_data.values() 
        if field_data.get('value') is not None
    )
    
    # If we have very little data, be more conservative
    if total_extracted_fields < 5:
        overall_status = "FAILED"
        logger.warning(f"Insufficient data for verification: only {total_extracted_fields} fields extracted")
    elif passed_key_rules >= len(key_rules):
        overall_status = "VERIFIED"
    else:
        overall_status = "FAILED"
    
    logger.info(f"Verification completed: {overall_status}. Key rules passed: {passed_key_rules}/{len(key_rules)}")
    
    return results

def verify_name_match(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 1: Verify name consistency across documents."""
    names = {}
    for doc_type, data in extracted_data.items():
        name_field = data.get('full_name', {})
        if name_field.get('value'):
            names[doc_type] = name_field['value']
    
    if len(names) < 2:
        return {
            "status": "FAIL",
            "reason": "Insufficient name data for comparison",
            "values": names
        }
    
    # Normalize and compare names
    normalized_names = {}
    for doc_type, name in names.items():
        # Convert to lowercase, split into tokens, sort, and join
        tokens = set(name.lower().split())
        normalized_names[doc_type] = ' '.join(sorted(tokens))
    
    # Check if all normalized names are equal
    unique_names = set(normalized_names.values())
    
    if len(unique_names) == 1:
        return {
            "status": "PASS",
            "reason": "All document names match after normalization",
            "values": names
        }
    else:
        return {
            "status": "FAIL",
            "reason": f"Name mismatch across documents. Found {len(unique_names)} different normalized names",
            "values": names
        }

def verify_dob_match(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 2: Verify date of birth consistency."""
    dobs = {}
    for doc_type, data in extracted_data.items():
        dob_field = data.get('date_of_birth', {})
        if dob_field.get('value'):
            dobs[doc_type] = dob_field['value']
    
    if len(dobs) < 2:
        return {
            "status": "FAIL",
            "reason": "Insufficient DOB data for comparison",
            "values": dobs
        }
    
    # Check if all DOBs are equal
    unique_dobs = set(dobs.values())
    
    if len(unique_dobs) == 1:
        return {
            "status": "PASS",
            "reason": "All document dates of birth match",
            "values": dobs
        }
    else:
        return {
            "status": "FAIL",
            "reason": f"DOB mismatch across documents. Found {len(unique_dobs)} different dates",
            "values": dobs
        }

def verify_address_match(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 3: Verify address consistency (city, state, pincode must match)."""
    addresses = {}
    for doc_type, data in extracted_data.items():
        address_field = data.get('address', {})
        if address_field.get('value'):
            addresses[doc_type] = address_field['value']
    
    if len(addresses) < 2:
        return {
            "status": "FAIL",
            "reason": "Insufficient address data for comparison",
            "values": addresses
        }
    
    # Extract key components for comparison
    key_components = {}
    for doc_type, address in addresses.items():
        if isinstance(address, dict):
            city = address.get('city', '').lower().strip() if address.get('city') else None
            state = address.get('state', '').lower().strip() if address.get('state') else None
            pincode = address.get('pincode', '').strip() if address.get('pincode') else None
            
            key_components[doc_type] = (city, state, pincode)
    
    # Check if key components match
    unique_components = set([comp for comp in key_components.values() if all(comp)])
    
    if len(unique_components) == 1:
        return {
            "status": "PASS",
            "reason": "Address key components (city, state, pincode) match across documents",
            "values": addresses
        }
    else:
        mismatches = []
        for doc_type, components in key_components.items():
            mismatches.append(f"{doc_type}: city={components[0]}, state={components[1]}, pincode={components[2]}")
        
        return {
            "status": "FAIL",
            "reason": f"Address key components mismatch. Details: {'; '.join(mismatches)}",
            "values": addresses
        }

def verify_phone_match(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 4: Verify phone number consistency."""
    phones = {}
    for doc_type, data in extracted_data.items():
        phone_field = data.get('phone_number', {})
        if phone_field.get('value'):
            # Normalize by taking last 10 digits for Indian numbers
            phone = phone_field['value']
            if phone.startswith('+91') and len(phone) == 13:
                phones[doc_type] = phone[-10:]
            else:
                phones[doc_type] = phone
    
    if len(phones) < 2:
        return {
            "status": "FAIL",
            "reason": "Insufficient phone data for comparison",
            "values": phones
        }
    
    # Check if all phones are equal
    unique_phones = set(phones.values())
    
    if len(unique_phones) == 1:
        return {
            "status": "PASS",
            "reason": "All document phone numbers match",
            "values": phones
        }
    else:
        return {
            "status": "FAIL",
            "reason": f"Phone number mismatch across documents. Found {len(unique_phones)} different numbers",
            "values": phones
        }

def verify_father_name_match(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 5: Verify father's name consistency where present."""
    father_names = {}
    for doc_type, data in extracted_data.items():
        father_field = data.get('father_name', {})
        if father_field.get('value'):
            father_names[doc_type] = father_field['value']
    
    if len(father_names) < 2:
        return {
            "status": "FAIL",
            "reason": "Insufficient father's name data for comparison",
            "values": father_names
        }
    
    # Normalize and compare (similar to name matching)
    normalized_names = {}
    for doc_type, name in father_names.items():
        tokens = set(name.lower().split())
        normalized_names[doc_type] = ' '.join(sorted(tokens))
    
    unique_names = set(normalized_names.values())
    
    if len(unique_names) == 1:
        return {
            "status": "PASS",
            "reason": "All document father's names match after normalization",
            "values": father_names
        }
    else:
        return {
            "status": "FAIL",
            "reason": f"Father's name mismatch across documents. Found {len(unique_names)} different normalized names",
            "values": father_names
        }

def verify_pan_format(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 6: Validate PAN number format."""
    pans = {}
    for doc_type, data in extracted_data.items():
        pan_field = data.get('pan_number', {})
        if pan_field.get('value'):
            pans[doc_type] = pan_field['value']
    
    if not pans:
        return {
            "status": "FAIL",
            "reason": "No PAN numbers found in any document",
            "values": pans
        }
    
    # Validate each PAN format
    invalid_pans = {}
    for doc_type, pan in pans.items():
        if not (len(pan) == 10 and 
                pan[:5].isalpha() and 
                pan[5:9].isdigit() and 
                pan[9].isalpha()):
            invalid_pans[doc_type] = pan
    
    if not invalid_pans:
        return {
            "status": "PASS",
            "reason": "All PAN numbers have valid format",
            "values": pans
        }
    else:
        return {
            "status": "FAIL",
            "reason": f"Invalid PAN format in documents: {', '.join(invalid_pans.keys())}",
            "values": pans
        }

def verify_aadhaar_format(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 7: Validate Aadhaar number format."""
    aadhaars = {}
    for doc_type, data in extracted_data.items():
        aadhaar_field = data.get('aadhaar_number', {})
        if aadhaar_field.get('value'):
            aadhaars[doc_type] = aadhaar_field['value']
    
    if not aadhaars:
        return {
            "status": "FAIL",
            "reason": "No Aadhaar numbers found in any document",
            "values": aadhaars
        }
    
    # Validate each Aadhaar format (12 digits)
    invalid_aadhaars = {}
    for doc_type, aadhaar in aadhaars.items():
        if not (len(aadhaar) == 12 and aadhaar.isdigit()):
            invalid_aadhaars[doc_type] = aadhaar
    
    if not invalid_aadhaars:
        return {
            "status": "PASS",
            "reason": "All Aadhaar numbers have valid format (12 digits)",
            "values": aadhaars
        }
    else:
        return {
            "status": "FAIL",
            "reason": f"Invalid Aadhaar format in documents: {', '.join(invalid_aadhaars.keys())}",
            "values": aadhaars
        }