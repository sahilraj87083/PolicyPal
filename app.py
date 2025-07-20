# In app.py

import streamlit as st
import os
from src.pipeline import run_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="PolicyPal",
    page_icon="⚖️",
    layout="centered"
)

# --- App Title and Description ---
st.title("⚖️ PolicyPal: AI Claims Adjudicator")
st.markdown("Upload a claim document (PDF, DOCX, EML, TXT, HTML) to receive an AI-powered adjudication decision.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a claim file",
    type=['pdf', 'docx', 'eml', 'txt', 'html']
)

# --- Analysis Button and Output ---
if uploaded_file is not None:
    if st.button("Analyze Claim"):
        # Display a spinner while processing
        with st.spinner("Reading document, retrieving clauses, and making a decision... Please wait."):
            # Save the uploaded file to a temporary location
            temp_dir = "temp_queries"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Run the pipeline
            result = run_pipeline(file_path)

            # Clean up the temporary file
            os.remove(file_path)

        # Display the results
        st.subheader("Final Adjudication Decision")
        if "error" in result:
            st.error(result["error"])
        else:
            st.json(result)