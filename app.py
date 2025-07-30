# streamlit_app.py
import streamlit as st
import requests
import uuid

st.title("📄 Chat with Your Document:")
st.write('IntelDocs')

# Generate unique user_id per session
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id

# Upload PDF
with st.form("upload-form"):
    file = st.file_uploader("Upload your PDF", type=["pdf"])
    submitted = st.form_submit_button("Upload")
    if submitted and file:
        print(user_id)  # Debug print (optional)
        res = requests.post(
            "https://ai-assistant-37ym.onrender.com/upload",
            files={"file": (file.name, file, file.type)},
            data={"user_id": user_id}
        )
        if res.status_code == 200:
            st.success("✅ File uploaded and processed!")
        else:
            st.error(f"❌ Failed to upload/process file. Status code: {res.status_code}")

# Ask question
question = st.text_input("Ask something from the doc:")
if question:
    res = requests.post("https://ai-assistant-37ym.onrender.com/ask", data={
        "question": question,
        "user_id": user_id
    })
    if res.status_code == 200:
        st.write("### ✅ Answer")
        st.write(res.json()["answer"])
    else:
        st.error(f"❌ Failed to get answer. Status code: {res.status_code}")
