import sys
sys.modules["torch.classes"] = None
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from pdf2image import convert_from_bytes
import PIL

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def show_pdf(file_bytes):
    try:
        # Convert PDF to images
        images = convert_from_bytes(file_bytes)
        for i, img in enumerate(images, 1):
            st.image(img, caption=f"PDF Page {i}", use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying PDF as image: {str(e)}")

def extract_text_from_pdf(file_bytes):
    pdf = PdfReader(BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()

def extract_candidate_details(text):
    doc = nlp(text)
    name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "Unknown"
    phone_match = re.search(r'\+?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4,7}', text)
    phone = phone_match.group(0) if phone_match else "Unknown"
    return name, email, phone

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
    # Styling
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
    💼 Contacts:0704217816 /+256787092035
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Made with ❤️ by ML-D7")

# ========== MAIN CONTENT ==========
st.header("📤 Upload Resume PDFs")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

st.header("📌 Job Description")
job_description = st.text_area("Enter the job description here...")

if uploaded_files and job_description:
    resume_texts_dict = {}
    resume_files_dict = {}
    resume_texts_list = []
    candidate_info = []

    for file in uploaded_files:
        file_bytes = file.getvalue()
        resume_files_dict[file.name] = file_bytes
        text = extract_text_from_pdf(file_bytes)
        resume_texts_dict[file.name] = text
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

    # Add preview section to sidebar after processing
    with st.sidebar:
        st.markdown('<div class="sidebar-section"><div class="sidebar-header">🔍 Resume Preview</div>', unsafe_allow_html=True)
        sorted_file_names = results_df["File Name"].tolist()
        selected_file = st.selectbox("Select resume to view:", sorted_file_names, key="preview_selector")
        st.markdown('</div>', unsafe_allow_html=True)

    # Show selected PDF preview as images
    st.subheader(f"📑 Preview of {selected_file}")
    pdf_bytes = resume_files_dict.get(selected_file)
    if pdf_bytes:
        show_pdf(pdf_bytes)
    else:
        st.warning("No PDF content available")
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

# # import streamlit as st
# # import pandas as pd
# # from PyPDF2 import PdfReader
# # import re
# # import spacy
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# # import pdfplumber  # Added for PDF preview
# # from PIL import Image  # Added for image handling
# # import base64
# # from io import BytesIO

# # # Fix torch.classes error
# # import sys
# # sys.modules["torch.classes"] = None

# # # Load spaCy English model
# # nlp = spacy.load("en_core_web_sm")

# # def show_pdf_preview(file_bytes):
# #     """Convert first page of PDF to image preview"""
# #     try:
# #         with pdfplumber.open(BytesIO(file_bytes)) as pdf:
# #             if len(pdf.pages) > 0:
# #                 first_page = pdf.pages[0]
# #                 img = first_page.to_image(resolution=150)
# #                 img_bytes = BytesIO()
# #                 img.save(img_bytes, format="PNG")
# #                 img_bytes.seek(0)
# #                 st.image(img_bytes, caption="First Page Preview", use_column_width=True)
# #     except Exception as e:
# #         st.error(f"Preview generation failed: {str(e)}")

# # def extract_text_from_pdf(file_bytes):
# #     """Extract text from PDF using PyPDF2"""
# #     pdf = PdfReader(BytesIO(file_bytes))
# #     text = ""
# #     for page in pdf.pages:
# #         page_text = page.extract_text()
# #         if page_text:
# #             text += page_text + " "
# #     return text.strip()

# # def extract_candidate_details(text):
# #     """Extract candidate details using spaCy and regex"""
# #     doc = nlp(text)
# #     name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Unknown")
# #     email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
# #     email = email_match.group(0) if email_match else "Unknown"
# #     phone_match = re.search(r'\+?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4,7}', text)
# #     phone = phone_match.group(0) if phone_match else "Unknown"
# #     return name, email, phone

# # def rank_resumes(job_description, resumes):
# #     """Rank resumes based on cosine similarity"""
# #     documents = [job_description] + resumes
# #     vectorizer = TfidfVectorizer().fit_transform(documents)
# #     vectors = vectorizer.toarray()
# #     job_description_vector = vectors[0]
# #     resume_vectors = vectors[1:]
# #     return cosine_similarity([job_description_vector], resume_vectors).flatten()

# # # Streamlit app configuration
# # st.set_page_config(page_title="Resume Ranking System", layout="wide")
# # st.title("📄 AI-Powered Resume Ranking System")

# # # ========== SIDEBAR CONTENT ==========
# # with st.sidebar:
# #     st.markdown("""
# #     <style>
# #     .sidebar-section {
# #         background-color: #f0f2f6;
# #         border-radius: 10px;
# #         padding: 15px;
# #         margin: 10px 0;
# #     }
# #     .sidebar-header {
# #         color: #2c3e50;
# #         font-size: 1.2em;
# #         font-weight: bold;
# #         margin-bottom: 10px;
# #     }
# #     </style>
# #     """, unsafe_allow_html=True)

# #     # About Section
# #     st.markdown('<div class="sidebar-section"><div class="sidebar-header">📚 About</div>', unsafe_allow_html=True)
# #     st.caption("""
# #     **Resume Ranker Pro** helps HR professionals:
# #     - 🔍 Analyze PDF resumes
# #     - 🎯 Match candidates to job descriptions
# #     - 🏆 Rank applicants automatically
# #     """)
# #     st.markdown('</div>', unsafe_allow_html=True)

# #     # Help Section
# #     with st.expander("❓ Step-by-Step Guide", expanded=False):
# #         st.markdown("""
# #         1. **Upload Resumes**  
# #            📤 Click 'Browse files' to upload candidate PDFs
        
# #         2. **Enter Job Description**  
# #            📝 Paste the job requirements in the text area
        
# #         3. **View Results**  
# #            🏆 System will automatically:
# #            - Extract candidate info
# #            - Calculate match scores
# #            - Display ranked results
        
# #         4. **Preview Documents**  
# #            🔍 Select any resume from the list to view details
# #         """)

# #     # Contact Section
# #     st.markdown('<div class="sidebar-section"><div class="sidebar-header">📧 Contact</div>', unsafe_allow_html=True)
# #     st.markdown("""
# #     **Need Help?**  
# #     ✉️ Email: support@resumeranker.com  
# #     📞 Phone: +256-787-092-035  
# #     🌐 GitHub: [ML-D7](https://github.com/ML-D7)
# #     """)
# #     st.markdown('</div>', unsafe_allow_html=True)

# #     st.markdown("<br>", unsafe_allow_html=True)
# #     st.caption("Made with ❤️ by ML-D7")

# # # ========== MAIN CONTENT ==========
# # st.header("📤 Upload Resume PDFs")
# # uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# # st.header("📌 Job Description")
# # job_description = st.text_area("Enter the job description here...", height=150)

# # if uploaded_files and job_description:
# #     resume_files_dict = {}
# #     resume_texts_list = []
# #     candidate_info = []

# #     for file in uploaded_files:
# #         file_bytes = file.getvalue()
# #         resume_files_dict[file.name] = file_bytes
        
# #         text = extract_text_from_pdf(file_bytes)
# #         resume_texts_list.append(text)
        
# #         name, email, phone = extract_candidate_details(text)
# #         candidate_info.append({
# #             "File Name": file.name,
# #             "Name": name,
# #             "Email": email,
# #             "Contact": phone
# #         })

# #     # Process rankings
# #     scores = rank_resumes(job_description, resume_texts_list)
# #     for i in range(len(candidate_info)):
# #         candidate_info[i]["Score"] = round(scores[i], 4)
    
# #     results_df = pd.DataFrame(candidate_info)
# #     results_df = results_df.sort_values(by="Score", ascending=False).reset_index(drop=True)
# #     st.dataframe(results_df, use_column_width=True)

# #     # Preview Section
# #     with st.sidebar:
# #         st.markdown('<div class="sidebar-section"><div class="sidebar-header">🔍 Resume Preview</div>', unsafe_allow_html=True)
# #         selected_file = st.selectbox("Select resume:", results_df["File Name"].tolist())
# #         st.markdown('</div>', unsafe_allow_html=True)

# #     # Show selected preview
# #     st.subheader(f"📑 Preview of {selected_file}")
# #     pdf_bytes = resume_files_dict.get(selected_file)
    
# #     if pdf_bytes:
# #         col1, col2 = st.columns([2, 1])
# #         with col1:
# #             show_pdf_preview(pdf_bytes)
# #         with col2:
# #             with st.expander("📄 View Extracted Text"):
# #                 text = extract_text_from_pdf(pdf_bytes)
# #                 st.write(text[:2000] + "...")  # Show first 2000 characters
            
# #             st.download_button(
# #                 label="📥 Download Full Resume",
# #                 data=pdf_bytes,
# #                 file_name=selected_file,
# #                 mime="application/pdf"
# #             )
# #     else:
# #         st.warning("No PDF content available")