import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.set_page_config(page_title="AI Resume Ranker", page_icon="📄")
st.title("AI Resume Ranking System")

# Inputs
jd_text = st.text_area("Paste the Job Description here:", height=200)
uploaded_files = st.file_uploader("Upload Resumes (PDF only)", type="pdf", accept_multiple_files=True)

if st.button("Rank Resumes") and jd_text and uploaded_files:
    resumes_text = []
    file_names = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes_text.append(text)
        file_names.append(file.name)

    # Vectorization
    all_content = [jd_text] + resumes_text
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_content)

    # Calculate Similarity (JD is index 0)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Results
    results = sorted(zip(file_names, similarities), key=lambda x: x[1], reverse=True)

    st.subheader("Ranking Results")
    for name, score in results:
        # Converting similarity to a percentage
        st.write(f"**{name}**: Match Score: {round(score * 100, 2)}%")
        st.progress(float(score))
elif not jd_text or not uploaded_files:
    st.info("Please provide both a job description and at least one resume.")
