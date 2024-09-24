import os
import streamlit as st
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
import uuid

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient(path='vectorstore')

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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"
)

# Improved text extraction and cleaning
def load_resume(resume_file):
    try:
        reader = PdfReader(resume_file)
        resume_text = " ".join(page.extract_text() or "" for page in reader.pages).replace('\n', ' ')
        return resume_text.strip()
    except Exception as e:
        st.error(f"Error reading resume file: {e}")
        return ""

def clean_text(text):
    return ' '.join(text.split())

# Store resume and job description data
def store_data_in_chroma(collection_name, document, metadata):
    try:
        collection = client.get_or_create_collection(name=collection_name)
        document_id = str(uuid.uuid4())
        collection.add(documents=[document], metadatas=metadata, ids=[document_id])
    except Exception as e:
        st.error(f"Failed to store data in ChromaDB: {e}")

# Scrape job description
def scrape_job_description(url):
    try:
        loader = WebBaseLoader([url])
        page_data = loader.load()
        if page_data:
            return clean_text(page_data[0].page_content)
        return "No description found."
    except Exception as e:
        st.error(f"Failed to load the job description: {e}")
        return ""

# Generate cold email
def generate_cold_email(job_description, resume_text):
    try:
        email_content = llm.invoke(
            "Write a detailed cold email for a job application using the provided resume and job description.",
            {
                "job_description": job_description,
                "resume": resume_text
            }
        )
        return email_content.content
    except Exception as e:
        st.error(f"Failed to generate cold email: {e}")
        return ""

# Streamlit app
def create_streamlit_app():
    st.title("📧 Cold Email Generator")

    url_input = st.text_input("Enter a job URL:")
    resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")
    submit_button = st.button("Generate Cold Email")

    if submit_button and resume_file:
        job_description = scrape_job_description(url_input)
        resume_text = load_resume(resume_file)
        email = generate_cold_email(job_description, resume_text)
        st.text_area("Generated Email", email, height=300)

if __name__ == "__main__":
    create_streamlit_app()
