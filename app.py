"""
AI Resume Screener Application
A Streamlit app that analyzes resumes against job descriptions using OpenAI's GPT-4o-mini.
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import PyPDF2
from io import BytesIO

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# =============================================================================
# PDF EXTRACTION FUNCTION
# =============================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extracts text content from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text from the PDF, or error message if extraction fails
    """
    try:
        # Create a BytesIO object from the uploaded file
        pdf_bytes = BytesIO(uploaded_file.read())
        
        # Initialize PDF reader
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            return "[ERROR] PDF is encrypted and cannot be read."
        
        # Extract text from all pages
        text_content = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
        
        # Join all pages with newlines
        full_text = "\n".join(text_content)
        
        # Check if any text was extracted
        if not full_text.strip():
            return "[ERROR] No readable text found in PDF (might be image-based)."
            
        return full_text
        
    except PyPDF2.errors.PdfReadError as e:
        return f"[ERROR] Failed to read PDF: {str(e)}"
    except Exception as e:
        return f"[ERROR] Unexpected error processing PDF: {str(e)}"

# =============================================================================
# AI ANALYSIS FUNCTION
# =============================================================================

def analyze_resume(resume_text: str, job_description: str) -> dict:
    """
    Analyzes a resume against a job description using OpenAI's GPT-4o-mini.
    
    Args:
        resume_text: The extracted text from the resume PDF
        job_description: The job description to match against
        
    Returns:
        dict: Analysis results with name, match_score, strengths, weaknesses, summary
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # System prompt defining the AI's role and output format
    system_prompt = """You are a skilled HR Assistant. You must analyze the resume against the job description.

You MUST respond with a valid JSON object containing exactly these keys:
- "name": string (extract the candidate's full name from the resume)
- "match_score": integer from 0-100 (how well the candidate matches the job)
- "key_strengths": array of strings (3-5 relevant strengths for this role)
- "weaknesses": array of strings (2-4 gaps or areas of concern)
- "summary": string (1-2 sentence overall assessment)

Respond ONLY with the JSON object, no additional text or markdown."""

    # User prompt with the actual content to analyze
    user_prompt = f"""Please analyze this resume against the job description.

=== JOB DESCRIPTION ===
{job_description}

=== RESUME ===
{resume_text}

Provide your analysis as a JSON object."""

    try:
        # Make API request to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent output
            response_format={"type": "json_object"}  # Enforce JSON response
        )
        
        # Extract the response content
        response_text = response.choices[0].message.content
        
        # Parse JSON response
        result = json.loads(response_text)
        
        # Validate required keys exist
        required_keys = ["name", "match_score", "key_strengths", "weaknesses", "summary"]
        for key in required_keys:
            if key not in result:
                result[key] = "N/A" if key in ["name", "summary"] else (0 if key == "match_score" else [])
        
        return result
        
    except json.JSONDecodeError as e:
        return {
            "name": "Parse Error",
            "match_score": 0,
            "key_strengths": [],
            "weaknesses": ["Failed to parse AI response"],
            "summary": f"JSON parsing error: {str(e)}"
        }
    except Exception as e:
        return {
            "name": "Error",
            "match_score": 0,
            "key_strengths": [],
            "weaknesses": ["API request failed"],
            "summary": f"Error: {str(e)}"
        }

# =============================================================================
# STREAMLIT USER INTERFACE
# =============================================================================

def main():
    """Main function to run the Streamlit application."""
    
    # App header
    st.title("📄 AI Resume Screener")
    st.markdown("Upload resumes and match them against your job description using AI.")
    st.divider()
    
    # Check for API key
    if not OPENAI_API_KEY:
        st.warning("⚠️ OpenAI API key not found! Please create a `.env` file with `OPENAI_API_KEY=your_key_here`")
        st.stop()
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=300,
            placeholder="Enter the full job description including required skills, experience, and responsibilities..."
        )
    
    with col2:
        st.subheader("📎 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF resumes (multiple allowed):",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
    
    st.divider()
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Resumes", type="primary", use_container_width=True)
    
    # Process when button is clicked
    if analyze_button:
        # Validation
        if not job_description.strip():
            st.error("❌ Please enter a job description.")
            return
            
        if not uploaded_files:
            st.error("❌ Please upload at least one resume PDF.")
            return
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        # Store results for summary table
        all_results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each uploaded file
        for idx, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (idx + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing: {uploaded_file.name}...")
            
            # Extract text from PDF
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # Check for extraction errors
            if resume_text.startswith("[ERROR]"):
                all_results.append({
                    "File": uploaded_file.name,
                    "Name": "N/A",
                    "Score": 0,
                    "Status": resume_text
                })
                continue
            
            # Analyze the resume
            analysis = analyze_resume(resume_text, job_description)
            
            # Store result for table
            all_results.append({
                "File": uploaded_file.name,
                "Name": analysis.get("name", "Unknown"),
                "Score": analysis.get("match_score", 0),
                "Summary": analysis.get("summary", "N/A")
            })
            
            # Display detailed results in expander
            with st.expander(f"📄 {uploaded_file.name} - {analysis.get('name', 'Unknown')} ({analysis.get('match_score', 0)}% match)", expanded=False):
                
                # Score with color coding
                score = analysis.get("match_score", 0)
                if score >= 70:
                    score_color = "green"
                    score_label = "Strong Match"
                elif score >= 50:
                    score_color = "orange"
                    score_label = "Moderate Match"
                else:
                    score_color = "red"
                    score_label = "Weak Match"
                
                st.markdown(f"### Match Score: :{score_color}[{score}%] - {score_label}")
                
                # Summary
                st.markdown(f"**Summary:** {analysis.get('summary', 'N/A')}")
                
                # Strengths and weaknesses in columns
                str_col, weak_col = st.columns(2)
                
                with str_col:
                    st.markdown("**✅ Key Strengths:**")
                    for strength in analysis.get("key_strengths", []):
                        st.markdown(f"- {strength}")
                
                with weak_col:
                    st.markdown("**⚠️ Weaknesses/Gaps:**")
                    for weakness in analysis.get("weaknesses", []):
                        st.markdown(f"- {weakness}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display summary table
        if all_results:
            st.divider()
            st.subheader("📈 Summary Overview")
            
            # Sort by score descending
            sorted_results = sorted(all_results, key=lambda x: x.get("Score", 0), reverse=True)
            
            # Create dataframe for display
            import pandas as pd
            df = pd.DataFrame(sorted_results)
            
            # Style the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Match Score",
                        min_value=0,
                        max_value=100,
                        format="%d%%"
                    )
                }
            )
            
            st.success(f"✅ Analysis complete! Processed {len(uploaded_files)} resume(s).")

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
