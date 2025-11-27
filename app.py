# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from rag2 import MBA_RAG_Engine

import sys
import streamlit as st

st.write("Python executable:", sys.executable)

# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------
st.title("📘 MBA Case Debrief Bot (RAG)")
st.write("Upload a case study PDF and ask questions.")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

UPLOAD_FOLDER = "CaseUploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize engine
engine = MBA_RAG_Engine(db_folder="chroma_db")

uploaded_file = st.file_uploader("Upload PDF Case Study", type=["pdf"])

if uploaded_file:
    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.success("PDF uploaded successfully.")

    # Load
    with st.spinner("Loading PDF..."):
        docs = engine.load_pdf(file_path)

    # Build vectorstore
    with st.spinner("Building vector store..."):
        vectorstore = engine.build_vectorstore(docs)

    # RAG chain
    qa_chain = engine.create_qa_chain(vectorstore)

    # Ask question
    query = st.text_input("Ask a question about the case study:")

    if st.button("Get Answer"):
        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing case..."):
                answer = engine.answer_question(qa_chain, query)

            st.subheader("📌 Answer")
            st.write(answer)
