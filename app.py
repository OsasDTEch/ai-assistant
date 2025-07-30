import streamlit as st
import requests
import uuid
import time

# Configuration
BACKEND_URL = "https://ai-assistant-37ym.onrender.com"  # Change if needed
MAX_RETRIES = 3  # For cold start issues
RETRY_DELAY = 5  # Seconds between retries

st.title("📄 Chat with Your Document:")
st.write('IntelDocs')

# Generate unique user_id per session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.file_uploaded = False

user_id = st.session_state.user_id

# Upload PDF
with st.form("upload-form"):
    file = st.file_uploader("Upload your PDF", type=["pdf"])
    submitted = st.form_submit_button("Upload")
    
    if submitted and file:
        with st.spinner("Processing PDF..."):
            for attempt in range(MAX_RETRIES):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/upload",
                        files={"file": (file.name, file, file.type)},
                        data={"user_id": user_id},
                        timeout=30  # Increase timeout
                    )
                    
                    if res.status_code == 200:
                        st.session_state.file_uploaded = True
                        st.success("✅ File uploaded and processed!")
                        break
                    else:
                        st.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying...")
                        time.sleep(RETRY_DELAY)
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    time.sleep(RETRY_DELAY)
                    
            else:  # If all retries failed
                st.error(f"❌ Failed after {MAX_RETRIES} attempts. Status: {res.status_code}")
                st.json(res.json())  # Show full error response

# Ask question
if st.session_state.get('file_uploaded', False):
    question = st.text_input("Ask something from the doc:")
    if question:
        with st.spinner("Generating answer..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/ask",
                    data={"question": question, "user_id": user_id},
                    timeout=20
                )
                
                if res.status_code == 200:
                    st.write("### Answer")
                    st.write(res.json()["answer"])
                else:
                    st.error(f"❌ Failed to get answer (Status: {res.status_code})")
                    st.json(res.json())  # Show error details
                    
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
else:
    st.warning("ℹ️ Please upload a PDF first")
