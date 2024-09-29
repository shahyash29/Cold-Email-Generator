import os
import re
import uuid
import fitz  # PyMuPDF for PDF parsing
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the page title and layout using Streamlit's page configuration
st.set_page_config(
    page_title="Cold Email Generator",  # Set the title of the browser tab
    page_icon="📧",  # Optional: Set a favicon for the page
    layout="centered",  # Page layout (can be 'wide' or 'centered')
)

# Initialize ChromaDB Persistent Client
client = chromadb.PersistentClient(path='vectorstore')

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

def clean_text(text):
    # Remove HTML tags and special characters
    text = re.sub(r'<[^>]*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_urls_from_text(text):
    # Regex pattern to find URLs
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, text)
    return urls

def extract_links_from_pdf(pdf_path):
    """ Extract links from the PDF using PyMuPDF (fitz). """
    doc = fitz.open(pdf_path)
    links = []

    # Iterate through all the pages and extract links
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load the page
        for link in page.get_links():
            if link.get("uri"):  # If the link contains a URL
                links.append(link.get("uri"))

    doc.close()
    return links

def extract_linkedin_github(urls):
    # More specific regex patterns for LinkedIn and GitHub
    linkedin_url = next((url for url in urls if re.match(r'https?://(www\.)?linkedin\.com/.*', url)), None)
    github_url = next((url for url in urls if re.match(r'https?://(www\.)?github\.com/.*', url)), None)
    return linkedin_url, github_url

def extract_projects_section(resume_text):
    # Extract the "Projects" section by finding the section labeled "Projects"
    projects_section = ""
    project_section_pattern = r'(Projects)([\s\S]*?)(\n\n|\Z)'  # Find "Projects" section
    match = re.search(project_section_pattern, resume_text)
    if match:
        projects_section = match.group(2)
    
    # Split by line to get individual project entries
    project_lines = [line.strip() for line in projects_section.split('\n') if line.strip()]
    return project_lines

def load_resume_and_extract_links(resume_file_path):
    try:
        # Open the PDF and extract links using PyMuPDF (fitz)
        links = extract_links_from_pdf(resume_file_path)
        
        # Read the text from the resume using PyMuPDF
        doc = fitz.open(resume_file_path)
        resume_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            resume_text += page.get_text()

        doc.close()

        resume_text = clean_text(resume_text)
        extracted_urls = extract_urls_from_text(resume_text)  # Extract URLs from text
        linkedin_url, github_url = extract_linkedin_github(links)  # Extract LinkedIn and GitHub URLs
        projects = extract_projects_section(resume_text)  # Extract "Projects" section
        return resume_text, links, projects, linkedin_url, github_url
    except Exception as e:
        st.error(f"Error reading resume file: {e}")
        return "", [], [], None, None

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

# Match projects to their specific links using URLs from resume
def query_portfolio_links_with_projects(projects, urls):
    relevant_projects = []
    for project in projects:
        # Assign URL if available, else show 'No URL found'
        if urls:
            relevant_projects.append(f"{project}: {urls.pop(0)}")  # Use the first available URL for each project
        else:
            relevant_projects.append(f"{project}: No URL found")
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

# Restore original email prompt template
prompt_email_template = PromptTemplate.from_template(
     """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### RESUME:
        {resume}
        
        ### INSTRUCTION:
        You are applying for the job described above. Write a short, concise, and professional cold email by selecting 
        only the most relevant experience and projects from your resume that match the job description. Avoid unnecessary 
        details, focus on key skills and achievements, and keep the email brief. 
        
        The email should include:
        - A professional greeting addressing the hiring manager (if a name is available).
        - A brief introduction of yourself, including your degree, years of experience, and how you found the job.
        - Mention key experiences from experience section(without using subtitles) that align with the job requirements.
        - Mention key projects from project section(without using subtitles) that align with the job requirements.
        - End the email professionally with "Best regards" and your name.
        

        If present, include your LinkedIn: {linkedin_link} and GitHub: {github_link} links at the end.
    """
)

def generate_cold_email(job_description, resume_text, project_links, linkedin_link, github_link):
    try:
        # Adjust the template based on whether LinkedIn and GitHub are available
        linkedin_info = f"LinkedIn: {linkedin_link}" if linkedin_link else ""
        github_info = f"GitHub: {github_link}" if github_link else ""

        chain_email = prompt_email_template | llm
        email_content = chain_email.invoke({
            "job_description": job_description,
            "resume": resume_text,
            "links": project_links,  # Inject the specific project links
            "linkedin_link": linkedin_info,  # Add LinkedIn link dynamically
            "github_link": github_info  # Add GitHub link dynamically
        })
        return email_content.content
    except Exception as e:
        st.error(f"Failed to generate cold email: {e}")
        return ""

# Update Streamlit function to process resume and detect links
def create_streamlit_app():
    st.title("📧 Cold Email Generator")

    st.markdown("""
        Welcome to the **Cold Email Generator** app. This app helps you create personalized cold emails 
        based on your resume and the job description. Just upload your resume, provide a job URL, and generate a tailored email!
    """)

    url_input = st.text_input("Enter a Job URL:", value="https://jobs.nike.com/job/R-38583")
    resume_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

    submit_button = st.button("Generate Cold Email")

    if submit_button and resume_file:
        # Save the uploaded file to a temporary directory
        with open("/tmp/resume.pdf", "wb") as f:
            f.write(resume_file.read())

        job_description = scrape_job_description(url_input)
        if not job_description:
            st.error("Failed to scrape job description. Please check the URL.")
            return

        # Extract resume text, URLs, and projects
        resume_text, urls, projects, linkedin_url, github_url = load_resume_and_extract_links("/tmp/resume.pdf")
        store_resume_in_chroma(resume_text)
        store_job_description_in_chroma(job_description, url_input)

        # Automatically detect project names and assign URLs if available
        relevant_projects = query_portfolio_links_with_projects(projects, urls)
        
        # Generate the cold email
        email = generate_cold_email(job_description, resume_text, relevant_projects, linkedin_url, github_url)
        st.code(email, language='markdown')

if __name__ == "__main__":
    create_streamlit_app()
