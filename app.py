#!/usr/bin/env python3
"""
Streamlit Web App for Document Verification System
"""

import streamlit as st
import tempfile
import os
import zipfile
import json
import time
from pathlib import Path
from datetime import datetime
from run_pipeline import DocumentVerificationPipeline
from PIL import Image

# Page config
st.set_page_config(
    page_title="Document Verification",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .log-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        max-height: 400px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
tabs = st.sidebar.radio("Go to", ["Upload & Run", "Results", "Metrics / Summary", "Instructions / Tips"])

# App state
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = []
if 'processed_at' not in st.session_state:
    st.session_state.processed_at = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# ----------------------------
# Upload & Run Tab - SIMPLIFIED VERSION
# ----------------------------
if tabs == "Upload & Run":
    st.title("📤 Upload Documents")
    st.markdown("Upload a ZIP file containing folders for each `person_id`. Each folder should contain expected documents: government ID, bank statement, employment letter.")

    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip", accept_multiple_files=False)
    use_llm = st.checkbox("Use LLM for extraction (if configured)", value=False)

    if uploaded_file:
        # Show uploaded file info
        st.info(f"📁 Uploaded file: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")

        if st.button("🚀 Run Verification Pipeline", type="primary"):
            # Initialize processing state
            st.session_state.processing_complete = False
            
            # Create a simple progress area
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            with progress_placeholder.container():
                st.info("🔄 Starting document processing... This may take a few minutes.")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            try:
                # Save uploaded zip temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # Initialize pipeline
                pipeline = DocumentVerificationPipeline(use_llm=use_llm)
                
                # Update status
                status_text.text("📦 Extracting ZIP file...")
                progress_bar.progress(10)
                
                # Process dataset - WITHOUT real-time callbacks to avoid rendering issues
                status_text.text("🔍 Processing documents...")
                progress_bar.progress(30)
                
                results = pipeline.process_dataset(tmp_path)
                
                # Update progress
                status_text.text("✅ Processing complete! Finalizing results...")
                progress_bar.progress(90)
                
                # Store results
                st.session_state.pipeline_results = results
                st.session_state.processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.processing_complete = True
                
                progress_bar.progress(100)
                
                # Clear progress area and show results
                progress_placeholder.empty()
                status_placeholder.empty()
                
                # Show success message
                st.success("🎉 Pipeline completed successfully!")
                
                # Calculate metrics
                total_persons = len(results)
                total_docs = sum(len(p.get("ocr_results", {})) for p in results)
                verified_count = sum(1 for r in results if r.get("overall_status") == "VERIFIED")
                failed_count = total_persons - verified_count
                
                # Display metrics
                st.subheader("📊 Processing Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("👥 Total Persons", total_persons)
                col2.metric("📄 Total Documents", total_docs)
                col3.metric("✅ Verified", verified_count)
                col4.metric("❌ Failed", failed_count)
                
                # Show quick summary
                st.subheader("📋 Processed Persons")
                for person in results:
                    status_icon = "✅" if person.get("overall_status") == "VERIFIED" else "❌"
                    docs_count = len(person.get("ocr_results", {}))
                    st.write(f"{status_icon} **{person.get('person_id', 'Unknown')}**: {person.get('overall_status')} - {docs_count} document(s)")

            except Exception as e:
                # Clear progress area on error
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error(f"❌ Pipeline failed: {str(e)}")
                
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

# ----------------------------
# Results Tab - FIXED VERSION
# ----------------------------
elif tabs == "Results":
    st.title("📄 Verification Results")

    if not st.session_state.pipeline_results:
        st.warning("⚠️ No results to display. Please run the pipeline first.")
    else:
        st.info(f"📅 Results processed at: {st.session_state.processed_at}")
        
        # DEBUG: Show raw data structure
        st.subheader("🔍 DEBUG - Raw Data Structure")
        for i, person in enumerate(st.session_state.pipeline_results):
            st.write(f"**Person {i}:** {person.get('person_id')}")
            
            ocr_results = person.get("ocr_results", {})
            st.write(f"OCR Results type: {type(ocr_results)}")
            st.write(f"OCR Results keys: {list(ocr_results.keys())}")
            st.write(f"OCR Results count: {len(ocr_results)}")
            
            # Show what's inside each document type
            for doc_type, doc_data in ocr_results.items():
                st.write(f"  - {doc_type}: {type(doc_data)}")
                if isinstance(doc_data, dict):
                    st.write(f"    Sub-keys: {list(doc_data.keys())}")
                    for engine, engine_data in doc_data.items():
                        st.write(f"      * {engine}: {type(engine_data)}")
                        if isinstance(engine_data, dict):
                            st.write(f"        Success: {engine_data.get('success')}")
                            st.write(f"        Text length: {len(engine_data.get('raw_text', ''))}")
            
            engines = person.get("ocr_engines_used", [])
            st.write(f"OCR Engines: {engines}")
            st.write(f"OCR Engines count: {len(engines)}")
            st.write("---")
        
        # Continue with normal display...
        # Overall statistics
        total_persons = len(st.session_state.pipeline_results)
        total_docs = sum(len(p.get("ocr_results", {})) for p in st.session_state.pipeline_results)
        verified_count = sum(1 for p in st.session_state.pipeline_results if p.get("overall_status") == "VERIFIED")
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Overall Statistics:</strong><br>
            • 👥 Persons Processed: {total_persons}<br>
            • 📄 Documents Processed: {total_docs}<br>
            • ✅ Verified: {verified_count}<br>
            • ❌ Failed: {total_persons - verified_count}
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        for person in st.session_state.pipeline_results:
            person_id = person.get('person_id', 'Unknown')
            overall_status = person.get('overall_status', 'UNKNOWN')
            ocr_results = person.get('ocr_results', {})
            extracted_data = person.get('extracted_data', {})
            
            # Determine status color and icon
            status_icon = "✅" if overall_status == "VERIFIED" else "❌"
            
            with st.expander(f"{status_icon} Person ID: {person_id} | Status: {overall_status} | Documents: {len(ocr_results)}", expanded=False):
                
                # Document processing summary
                st.subheader("📋 Document Summary")
                doc_cols = st.columns(3)
                doc_cols[0].metric("Documents Processed", len(ocr_results))
                doc_cols[1].metric("OCR Engines Used", len(person.get("ocr_engines_used", [])))
                doc_cols[2].metric("Overall Status", overall_status)
                
                # Show document types processed
                if ocr_results:
                    st.write("**Document Types Processed:**")
                    for doc_type in ocr_results.keys():
                        st.write(f"• {doc_type.replace('_', ' ').title()}")
                
                # Display extracted key fields in a nice layout
                if extracted_data:
                    st.subheader("📝 Extracted Information")
                    
                    # Create columns for better layout
                    info_cols = st.columns(2)
                    
                    with info_cols[0]:
                        st.write("**Personal Information**")
                        personal_fields = ['full_name', 'dob', 'address', 'phone', 'email']
                        for field in personal_fields:
                            if field in extracted_data and extracted_data[field]:
                                st.write(f"**{field.replace('_', ' ').title()}:** {extracted_data[field]}")
                    
                    with info_cols[1]:
                        st.write("**Document Information**")
                        document_fields = [key for key in extracted_data.keys() if key not in personal_fields]
                        for field in document_fields:
                            if extracted_data[field]:
                                st.write(f"**{field.replace('_', ' ').title()}:** {extracted_data[field]}")
                else:
                    st.warning("No extracted data found for this person.")
                
                # Detailed OCR Results (collapsible)
                if ocr_results:
                    st.subheader("🔍 Detailed OCR Results")
                    for doc_type, ocr_data in ocr_results.items():
                        with st.expander(f"📄 {doc_type.replace('_', ' ').title()} - OCR Data", expanded=False):
                            if isinstance(ocr_data, dict):
                                if ocr_data:
                                    st.json(ocr_data)
                                else:
                                    st.warning("Empty OCR result")
                            elif isinstance(ocr_data, str):
                                if ocr_data.strip():
                                    st.text_area("OCR Text", ocr_data, height=200, key=f"{person_id}_{doc_type}_text")
                                else:
                                    st.warning("Empty text result")
                            else:
                                st.write(f"Data type: {type(ocr_data)}")
                                st.write(ocr_data)
                
                # Download buttons
                st.subheader("💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download Extracted Data (JSON)",
                        data=json.dumps(extracted_data, indent=2, ensure_ascii=False),
                        file_name=f"{person_id}_extracted_data.json",
                        mime="application/json",
                        key=f"extracted_{person_id}"
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download OCR Results (JSON)",
                        data=json.dumps(ocr_results, indent=2, ensure_ascii=False),
                        file_name=f"{person_id}_ocr_results.json",
                        mime="application/json",
                        key=f"ocr_{person_id}"
                    )

# ----------------------------
# Metrics / Summary Tab
# ----------------------------
elif tabs == "Metrics / Summary":
    st.title("📊 Metrics & Summary")
    
    if not st.session_state.pipeline_results:
        st.warning("⚠️ No results available. Please run the pipeline first.")
    else:
        # Calculate comprehensive metrics
        total_persons = len(st.session_state.pipeline_results)
        total_docs = sum(len(p.get("ocr_results", {})) for p in st.session_state.pipeline_results)
        verified_count = sum(1 for p in st.session_state.pipeline_results if p.get("overall_status") == "VERIFIED")
        failed_count = total_persons - verified_count
        
        # Calculate average documents per person
        avg_docs_per_person = total_docs / total_persons if total_persons > 0 else 0
        
        # Success rate
        success_rate = (verified_count / total_persons * 100) if total_persons > 0 else 0
        
        # Display key metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Total Persons", total_persons)
        col2.metric("📄 Total Documents", total_docs)
        col3.metric("✅ Verified", f"{verified_count} ({success_rate:.1f}%)")
        col4.metric("📊 Avg Docs/Person", f"{avg_docs_per_person:.1f}")
        
        # Simple visualization without Plotly (to avoid dependencies)
        st.subheader("📊 Status Distribution")
        
        status_counts = {
            "VERIFIED": verified_count,
            "FAILED": failed_count
        }
        
        # Create a simple bar chart using native Streamlit
        st.bar_chart(status_counts)
        
        # Detailed breakdown
        st.subheader("📋 Detailed Breakdown")
        for person in st.session_state.pipeline_results:
            status_icon = "✅" if person.get("overall_status") == "VERIFIED" else "❌"
            docs_count = len(person.get("ocr_results", {}))
            st.write(f"{status_icon} **{person.get('person_id')}**: {docs_count} documents | {person.get('overall_status')}")

# ----------------------------
# Instructions / Tips Tab
# ----------------------------
elif tabs == "Instructions / Tips":
    st.title("💡 Instructions & Tips")
    
    st.markdown("""
    ## 📋 How to Use This System
    
    ### 1. **Prepare Your Data**
    - Create a ZIP file with the following structure:
    ```
    your_dataset.zip
    ├── P001/
    │   ├── P001_government_id.png (or .jpg)
    │   ├── P001_bank_statement.png
    │   └── P001_employment_letter.png
    ├── P002/
    │   ├── P002_government_id.png
    │   ├── P002_bank_statement.png
    │   └── P002_employment_letter.png
    └── ...
    ```
    
    ### 2. **Upload & Process**
    - Go to **Upload & Run** tab
    - Upload your ZIP file
    - Click **"Run Verification Pipeline"**
    - Wait for processing to complete
    
    ### 3. **View Results**
    - **Results Tab**: Detailed view of each person's verification
    - **Metrics Tab**: Overall statistics and visualizations
    
    ### 4. **Download Data**
    - Download extracted data as JSON
    - Download OCR results for analysis
    
    ## 🚀 Best Practices
    
    - **Image Quality**: Use clear, high-resolution images (minimum 300 DPI)
    - **File Naming**: Follow the naming convention strictly
    - **File Formats**: PNG or JPG formats are supported
    - **File Size**: Keep individual images under 5MB for optimal performance
    - **Document Types**: Ensure you have all three required document types per person
    
    ## ⚙️ Configuration
    
    - **LLM Extraction**: Enable if you have API keys configured in `config.yml`
    - **OCR Engines**: The system automatically uses the best available OCR engine
    
    ## 🆘 Troubleshooting
    
    - **Processing Failed**: Check your ZIP file structure and image formats
    - **No Results**: Ensure all documents are present for each person
    - **Poor Extraction**: Use better quality images with clear text
    - **API Errors**: Verify your API keys and internet connection
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Document Verification System** v1.0")