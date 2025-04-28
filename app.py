
# import sys
# sys.modules["torch.classes"] = None
# import base64
# import streamlit as st
# from PyPDF2 import PdfReader
# import pandas as pd
# import re
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from io import BytesIO

# # Load spaCy English model
# nlp = spacy.load("en_core_web_sm")

# def show_pdf(file_bytes):
#     base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
#     pdf_display = f"""
#     <iframe src="data:application/pdf;base64,{base64_pdf}" 
#             width="100%" 
#             height="800px" 
#             style="border: none">
#     </iframe>
#     """
#     st.markdown(pdf_display, unsafe_allow_html=True)

# def extract_text_from_pdf(file_bytes):
#     pdf = PdfReader(BytesIO(file_bytes))
#     text = ""
#     for page in pdf.pages:
#         page_text = page.extract_text()
#         if page_text:
#             text += page_text + " "
#     return text.strip()

# def extract_candidate_details(text):
#     doc = nlp(text)
#     name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
#     email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
#     email = email_match.group(0) if email_match else "Unknown"
#     phone_match = re.search(r'\+?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4,7}', text)
#     phone = phone_match.group(0) if phone_match else "Unknown"
#     return name, email, phone

# def rank_resumes(job_description, resumes):
#     documents = [job_description] + resumes
#     vectorizer = TfidfVectorizer().fit_transform(documents)
#     vectors = vectorizer.toarray()
#     job_description_vector = vectors[0]
#     resume_vectors = vectors[1:]
#     return cosine_similarity([job_description_vector], resume_vectors).flatten()

# # Streamlit app configuration
# st.set_page_config(page_title="Resume Ranking System", layout="wide")
# st.title("📄 Resume Ranking System")

# # ========== SIDEBAR CONTENT ==========
# with st.sidebar:
#     # Styling
#     st.markdown("""
#     <style>
#     .sidebar-section {
#         background-color: #f0f2f6;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#     }
#     .sidebar-header {
#         color: #2c3e50;
#         font-size: 1.2em;
#         font-weight: bold;
#         margin-bottom: 10px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # About Section
#     st.markdown('<div class="sidebar-section"><div class="sidebar-header">📚 About</div>', unsafe_allow_html=True)
#     st.caption("""
#     **Resume Ranker Pro** helps HR professionals:
#     - 🔍 Analyze PDF resumes
#     - 🎯 Match candidates to job descriptions
#     - 🏆 Rank applicants automatically
#     """)
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Help Section
#     with st.expander("❓ Step-by-Step Guide", expanded=False):
#         st.markdown("""
#         1. **Upload Resumes**  
#            📤 Click 'Browse files' to upload candidate PDFs
        
#         2. **Enter Job Description**  
#            📝 Paste the job requirements in the text area
        
#         3. **View Results**  
#            🏆 System will automatically:
#            - Extract candidate info
#            - Calculate match scores
#            - Display ranked results
        
#         4. **Preview Documents**  
#            🔍 Select any resume from the list to view details
#         """)

#     # Contact Section
#     st.markdown('<div class="sidebar-section"><div class="sidebar-header">📧 Contact</div>', unsafe_allow_html=True)
#     st.markdown("""
#     **Need Help?**  
#     ✉️ Email: [support@resumeranker.com](mailto:support@resumeranker.com)  
#     💼 Contacts:0704217816 /+256787092035

#     """)
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Footer
#     st.markdown("<br>", unsafe_allow_html=True)
#     st.caption("Made with ❤️ by ML-D7")

# # ========== MAIN CONTENT ==========
# st.header("📤 Upload Resume PDFs")
# uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# st.header("📌 Job Description")
# job_description = st.text_area("Enter the job description here...")

# if uploaded_files and job_description:
#     resume_texts_dict = {}
#     resume_files_dict = {}
#     resume_texts_list = []
#     candidate_info = []

#     for file in uploaded_files:
#         file_bytes = file.getvalue()
#         resume_files_dict[file.name] = file_bytes
#         text = extract_text_from_pdf(file_bytes)
#         resume_texts_dict[file.name] = text
#         resume_texts_list.append(text)
        
#         name, email, phone = extract_candidate_details(text)
#         candidate_info.append({
#             "File Name": file.name,
#             "Name": name,
#             "Email": email,
#             "Contact": phone
#         })

#     # Process rankings
#     scores = rank_resumes(job_description, resume_texts_list)
#     for i in range(len(candidate_info)):
#         candidate_info[i]["Score"] = round(scores[i], 4)
    
#     results_df = pd.DataFrame(candidate_info)
#     results_df = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
#     st.dataframe(results_df, use_container_width=True)

#     # Add preview section to sidebar after processing
#     with st.sidebar:
#         st.markdown('<div class="sidebar-section"><div class="sidebar-header">🔍 Resume Preview</div>', unsafe_allow_html=True)
#         sorted_file_names = results_df["File Name"].tolist()
#         selected_file = st.selectbox("Select resume to view:", sorted_file_names, key="preview_selector")
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Show selected PDF preview
#     st.subheader(f"📑 Preview of {selected_file}")
#     pdf_bytes = resume_files_dict.get(selected_file)
#     if pdf_bytes:
#         try:
#             show_pdf(pdf_bytes)
#         except Exception as e:
#             st.error(f"Error displaying PDF: {str(e)}")
#     else:
#         st.warning("No PDF content available")
import sys
sys.modules["torch.classes"] = None
import base64
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from streamlit_pdf_viewer import pdf_viewer  # New package

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

@st.cache_data
def extract_text_from_pdf(file_bytes):
    pdf = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()

@st.cache_data
def extract_candidate_details(text):
    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "Unknown"
    phone_match = re.search(r'\+?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4,7}', text)
    phone = phone_match.group(0) if phone_match else "Unknown"
    return name, email, phone

@st.cache_data
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_description_vector], resume_vectors).flatten()

# Streamlit app configuration
st.set_page_config(page_title="Resume Ranking System", layout="wide")
st.title("📄 Resume Ranking System")

# ========== SIDEBAR CONTENT ==========
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .sidebar-header {
        color: #2c3e50;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # About Section
    st.markdown('<div class="sidebar-section"><div class="sidebar-header">📚 About</div>', unsafe_allow_html=True)
    st.caption("""
    **Resume Ranker Pro** helps HR professionals:
    - 🔍 Analyze PDF resumes
    - 🎯 Match candidates to job descriptions
    - 🏆 Rank applicants automatically
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Help Section
    with st.expander("❓ Step-by-Step Guide", expanded=False):
        st.markdown("""
        1. **Upload Resumes**  
           📤 Click 'Browse files' to upload candidate PDFs
        
        2. **Enter Job Description**  
           📝 Paste the job requirements in the text area
        
        3. **View Results**  
           🏆 System will automatically:
           - Extract candidate info
           - Calculate match scores
           - Display ranked results
        
        4. **Preview Documents**  
           🔍 Select any resume from the list to view details
        """)

    # Contact Section
    st.markdown('<div class="sidebar-section"><div class="sidebar-header">📧 Contact</div>', unsafe_allow_html=True)
    st.markdown("""
    **Need Help?**  
    ✉️ Email: [support@resumeranker.com](mailto:support@resumeranker.com)  
    📞 Phone: +256-787-092-035  
    🌐 GitHub: [ML-D7](https://github.com/ML-D7)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Made with ❤️ by ML-D7")

# ========== MAIN CONTENT ==========
st.header("📤 Upload Resume PDFs")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

st.header("📌 Job Description")
job_description = st.text_area("Enter the job description here...")

if uploaded_files and job_description:
    resume_files_dict = {}
    resume_texts_list = []
    candidate_info = []

    for file in uploaded_files:
        file_bytes = file.getvalue()
        resume_files_dict[file.name] = file_bytes
        text = extract_text_from_pdf(file_bytes)
        resume_texts_list.append(text)
        
        name, email, phone = extract_candidate_details(text)
        candidate_info.append({
            "File Name": file.name,
            "Name": name,
            "Email": email,
            "Contact": phone
        })

    # Process rankings
    scores = rank_resumes(job_description, resume_texts_list)
    for i in range(len(candidate_info)):
        candidate_info[i]["Score"] = round(scores[i], 4)
    
    results_df = pd.DataFrame(candidate_info)
    results_df = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    st.dataframe(results_df, use_container_width=True)

    # Preview section
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><div class="sidebar-header">🔍 Resume Preview</div>', unsafe_allow_html=True)
        selected_file = st.selectbox("Select resume:", results_df["File Name"].tolist())
        st.markdown('</div>', unsafe_allow_html=True)

    # PDF Viewer
    st.subheader(f"📑 Preview of {selected_file}")
    pdf_bytes = resume_files_dict.get(selected_file)
    
    if pdf_bytes:
        try:
            # Using streamlit-pdf-viewer component
            pdf_viewer(input=pdf_bytes, width=700)
            
            # Fallback text preview
            with st.expander("View Text Content"):
                text = extract_text_from_pdf(pdf_bytes)
                st.write(text[:2000] + "...")
                
            # Download button
            st.download_button(
                label="Download Full PDF",
                data=pdf_bytes,
                file_name=selected_file,
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")
    else:
        st.warning("No PDF content found")