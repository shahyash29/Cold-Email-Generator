import os
import re
import uuid
import csv
from PyPDF2 import PdfReader
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient(path='vectorstore')

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    return text.strip()

def load_resume(resume_file):
    try:
        reader = PdfReader(resume_file)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()
        return clean_text(resume_text)
    except Exception as e:
        st.error(f"Error reading resume file: {e}")
        return ""

def store_resume_in_chroma(resume_text):
    try:
        resume_collection = client.get_or_create_collection(name="resume_information")
        resume_id = str(uuid.uuid4())
        resume_collection.add(documents=[resume_text], metadatas={"name": "Yash Shah"}, ids=[resume_id])
    except Exception as e:
        st.error(f"Failed to store resume in ChromaDB: {e}")

def store_job_description_in_chroma(job_description, url):
    try:
        job_collection = client.get_or_create_collection(name="job_descriptions")
        job_id = str(uuid.uuid4())
        job_collection.add(documents=[job_description], metadatas={"url": url}, ids=[job_id])
    except Exception as e:
        st.error(f"Failed to store job description in ChromaDB: {e}")

# Load specific project links from the uploaded CSV file
def load_project_links():
    project_links = {}
    try:
        with open("Projects - Sheet1.csv", mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                project_name = row['Project Name']
                project_url = row['Project Link']
                project_links[project_name] = project_url
    except Exception as e:
        st.error(f"Error loading project links: {e}")
    return project_links

# Match projects to their specific links, but exclude Research Assistant project
def query_portfolio_links_with_projects(resume_text, project_links):
    relevant_projects = []
    exclude_project = "Research Assistant"  # Exclude Research Assistant-related project
    for project in project_links:
        if project in resume_text and exclude_project not in project:
            relevant_projects.append(f"{project}: {project_links[project]}")
    return relevant_projects

def scrape_job_description(url):
    try:
        loader = WebBaseLoader([url])
        page_data = loader.load()
        if page_data and isinstance(page_data, list) and len(page_data) > 0:
            return clean_text(page_data[0].page_content)
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load the job description: {e}")
        return None

prompt_email_template = PromptTemplate.from_template(
    """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### RESUME:
        {resume}
        
        ### INSTRUCTION:
        You are applying for the job described above. Write a personalized cold email highlighting your skills 
        and experiences from your resume that match the job description. Be concise, professional, and persuasive.
        
        Also, showcase the relevant projects from these links: {links}.
        
        The email should include:
        - A professional greeting addressing the hiring manager (if a name is available).
        - A brief introduction of yourself, including your degree, years of experience, and how you found the job.
        - A clear connection between your skills and the job description (mention specific experiences).
        - A brief mention of relevant projects work that showcases your expertise (GitHub and LinkedIn links should be included only within this section and not at the end).
        - A request for a meeting or further discussion about the opportunity.
        - Avoid any redundant addition of LinkedIn or GitHub URLs at the end
    """
)

def generate_cold_email(job_description, resume_text, project_links):
    try:
        # Map the projects to their corresponding links, excluding Research Assistant projects
        project_links_in_email = query_portfolio_links_with_projects(resume_text, project_links)
        
        # Integrate project-specific links into the email template
        chain_email = prompt_email_template | llm
        email_content = chain_email.invoke({
            "job_description": job_description,
            "resume": resume_text,
            "links": project_links_in_email,  # Inject the specific project links
        })
        
        email_content_with_links = email_content.content
        return email_content_with_links
    except Exception as e:
        st.error(f"Failed to generate cold email: {e}")
        return ""

# Update Streamlit function to integrate project links from CSV
def create_streamlit_app():
    st.title("📧 Cold Email Generator")

    url_input = st.text_input("Enter a Job URL:", value="https://jobs.nike.com/job/R-38583")
    resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    submit_button = st.button("Generate Cold Email")

    if submit_button and resume_file:
        job_description = scrape_job_description(url_input)
        if not job_description:
            st.error("Failed to scrape job description. Please check the URL.")
            return

        resume_text = load_resume(resume_file)
        store_resume_in_chroma(resume_text)
        store_job_description_in_chroma(job_description, url_input)

        # Load project links from the uploaded CSV
        project_links = load_project_links()

        email = generate_cold_email(job_description, resume_text, project_links)
        st.code(email, language='markdown')

if __name__ == "__main__":
    create_streamlit_app()
