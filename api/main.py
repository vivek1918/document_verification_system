#!/usr/bin/env python3
"""
Document Verification System - UI Matching WhatsApp Image Design
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Document Verification",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the exact design from the image
st.markdown("""
<style>
    /* Main styling to match the image */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    .header {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .document-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #007bff;
    }
    
    .verification-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    
    .verification-table th {
        background-color: #007bff;
        color: white;
        padding: 10px;
        text-align: left;
    }
    
    .verification-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    
    .status-pass {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-fail {
        color: #dc3545;
        font-weight: bold;
    }
    
    .person-tabs {
        display: flex;
        gap: 5px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    
    .person-tab {
        padding: 8px 16px;
        background: #e9ecef;
        border-radius: 5px;
        cursor: pointer;
        border: 1px solid #dee2e6;
    }
    
    .person-tab.active {
        background: #007bff;
        color: white;
        border-color: #007bff;
    }
    
    .document-section {
        margin: 15px 0;
    }
    
    .field-row {
        display: flex;
        margin: 5px 0;
    }
    
    .field-label {
        width: 120px;
        font-weight: bold;
        color: #495057;
    }
    
    .field-value {
        flex: 1;
        color: #6c757d;
    }
    
    .ocr-preview {
        background: #e9ecef;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
        margin: 10px 0;
    }
    
    .controls-section {
        background: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.title("📋 Document Verification")
    st.markdown("Process P001-P010 and review OCR/LLM outputs.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Controls section
    st.markdown('<div class="controls-section">', unsafe_allow_html=True)
    st.subheader("Controls")
    
    # Person tabs
    persons = ["P010", "P001", "P002", "P003", "P004", "P005", "P006", "P007", "P008", "P009", "P010"]
    cols = st.columns(len(persons))
    
    for i, person in enumerate(persons):
        with cols[i]:
            if st.button(person, use_container_width=True):
                st.session_state.selected_person = person
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_document_section("Government ID", "government_id", {
            "Full name": "Thomas Sura",
            "DOB": "2000-09-16",
            "Phone": "+919901627204",
            "Aadhaar": "52991241",
            "Employee ID": "",
            "Father's name": "Theodore Sura",
            "Address": "H.No. 67, Ramachandran ...",
            "Email": ""
        })
    
    with col2:
        render_document_section("Bank Statement", "bank_statement", {
            "Full name": "Thomas Sura",
            "DOB": "2000-09-16",
            "Phone": "+919901627204",
            "Aadhaar": "",
            "Father's name": "",
            "Address": "H.No. 67 Ramachandran ...",
            "Email": ""
        })
    
    with col3:
        render_document_section("Employment Letter", "employment_letter", {
            "Full name": "Thomas Sura",
            "DOB": "2000-09-16",
            "Phone": "+918012345678",
            "Aadhaar": "",
            "Father's name": "Theodore Sura",
            "Address": "House No. H.No. 67, Ram...",
            "Email": "thomas.sura@yahoo.com",
            "PAN": "PEXPC8053W",
            "Account No.": ""
        })
    
    # Verification Rules
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Verification Rules")
    
    verification_rules = [
        {"RULE": "Name match", "STATUS": "(PASS)"},
        {"RULE": "DOB match", "STATUS": "(PASS)"},
        {"RULE": "Address match (pincode)", "STATUS": "(PASS)"},
        {"RULE": "Phone match", "STATUS": "(FAIL)"},
        {"RULE": "Father name match", "STATUS": "(PASS)"},
        {"RULE": "PAN format present", "STATUS": "(PASS)"},
        {"RULE": "Aadhaar format present", "STATUS": "(FAIL)"}
    ]
    
    # Create verification table
    html_table = """
    <table class="verification-table">
        <thead>
            <tr>
                <th>RULE</th>
                <th>STATUS</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for rule in verification_rules:
        status_class = "status-pass" if "(PASS)" in rule["STATUS"] else "status-fail"
        html_table += f"""
            <tr>
                <td>{rule['RULE']}</td>
                <td class="{status_class}">{rule['STATUS']}</td>
            </tr>
        """
    
    html_table += """
        </tbody>
    </table>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # OCR Preview
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("OCR Preview")
    
    ocr_content = """
NIB                                    NOTIONAL INDIA BANK
       Banking with Trust                   BANK STATEMENT
September 2025                           Account Holder:    
Account Number:                          Date of Birth:     
Phone Number:                            Address:           
Thomas Sura                              9232719374         
16-09-2009                               +91-9981627284     
H.No. 67 Ramachandran Nagar, Machillpatnam, Telangana, PTN: 729806
    """
    
    st.markdown(f'<div class="ocr-preview">{ocr_content}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_document_section(title, doc_type, fields):
    """Render a document section with fields"""
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader(title)
    
    # Document type badge
    st.markdown(f"**{doc_type}**")
    
    # Render fields
    for field, value in fields.items():
        st.markdown(f"""
        <div class="field-row">
            <div class="field-label">{field}</div>
            <div class="field-value">{value if value else "-"}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sample data for multiple persons (you can expand this)
def get_person_data(person_id):
    """Get data for a specific person - this would come from your pipeline"""
    sample_data = {
        "P001": {
            "government_id": {
                "Full name": "Thomas Sura",
                "DOB": "2000-09-16",
                "Phone": "+919901627204",
                "Aadhaar": "52991241",
                "Employee ID": "",
                "Father's name": "Theodore Sura",
                "Address": "H.No. 67, Ramachandran ...",
                "Email": ""
            },
            "bank_statement": {
                "Full name": "Thomas Sura",
                "DOB": "2000-09-16",
                "Phone": "+919901627204",
                "Aadhaar": "",
                "Father's name": "",
                "Address": "H.No. 67 Ramachandran ...",
                "Email": ""
            },
            "employment_letter": {
                "Full name": "Thomas Sura",
                "DOB": "2000-09-16",
                "Phone": "+918012345678",
                "Aadhaar": "",
                "Father's name": "Theodore Sura",
                "Address": "House No. H.No. 67, Ram...",
                "Email": "thomas.sura@yahoo.com",
                "PAN": "PEXPC8053W",
                "Account No.": ""
            }
        }
        # Add more persons as needed
    }
    return sample_data.get(person_id, sample_data["P001"])

if __name__ == "__main__":
    main()