import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords

# Initializing NLTK Stopwords
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text


def get_cleaned_keywords(text):
    """
    Cleans text by removing punctuation and stopwords to extract raw keywords.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = set(text.split())
    keywords = words - STOPWORDS
    return keywords


# Page Configuration
st.set_page_config(page_title="AI Resume Ranker", page_icon="📄", layout="wide")

st.title("🎯 AI Resume Ranking & Gap Analysis")

st.markdown("""
Upload resumes and paste a job description to see how well they match.

This system uses **TF-IDF Vectorization** for scoring and **Natural Language Processing** for gap analysis.
""")


# Layout
col_a, col_b = st.columns([1, 1])

with col_a:
    jd_text = st.text_area("📋 Paste the Job Description here:", height=300)

with col_b:
    uploaded_files = st.file_uploader(
        "📤 Upload Resumes (PDF only)",
        type="pdf",
        accept_multiple_files=True
    )


if st.button("🚀 Rank Resumes"):

    if jd_text and uploaded_files:

        resumes_text = []
        file_names = []

        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resumes_text.append(text)
            file_names.append(file.name)

        # TF-IDF Similarity
        all_content = [jd_text] + resumes_text
        vectorizer = TfidfVectorizer(stop_words='english')

        tfidf_matrix = vectorizer.fit_transform(all_content)

        similarities = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:]
        ).flatten()

        # JD Keywords
        jd_keywords = get_cleaned_keywords(jd_text)

        # Sort Results
        results = sorted(
            zip(file_names, similarities, resumes_text),
            key=lambda x: x[1],
            reverse=True
        )

        st.divider()
        st.subheader("📊 Ranking Results")

        # Loop through resumes
        for name, score, raw_text in results:

            resume_keywords = get_cleaned_keywords(raw_text)

            matches = jd_keywords.intersection(resume_keywords)

            missing = jd_keywords - resume_keywords

            # Resume Strength
            strengths = []

            if len(matches) > 30:
                strengths.append("Strong keyword match with job description")

            if "python" in raw_text.lower():
                strengths.append("Python programming skill present")

            if "git" in raw_text.lower():
                strengths.append("Version control knowledge (Git)")

            if "database" in raw_text.lower() or "mysql" in raw_text.lower():
                strengths.append("Database knowledge")

            # Resume Weakness
            weaknesses = []

            if len(missing) > 30:
                weaknesses.append("Many required job keywords missing")

            if "react" not in raw_text.lower():
                weaknesses.append("No React or frontend framework mentioned")

            if "api" not in raw_text.lower():
                weaknesses.append("API development experience not mentioned")

            if "project" not in raw_text.lower():
                weaknesses.append("Projects section missing")

            # Display Result
            with st.expander(f"**{name}** — Match Score: {round(score * 100, 2)}%"):

                st.progress(float(score))

                c1, c2 = st.columns(2)

                with c1:
                    st.success(f"✅ **Matching Keywords ({len(matches)})**")

                    st.write(
                        ", ".join(list(matches)[:15])
                        if matches else "No significant matches."
                    )

                with c2:
                    st.error(f"❌ **Missing from Resume ({len(missing)})**")

                    st.write(
                        ", ".join(list(missing)[:15])
                        if missing else "No missing keywords found!"
                    )

                st.markdown("### 💪 Resume Strengths")

                if strengths:
                    for s in strengths:
                        st.write("- ", s)
                else:
                    st.write("No major strengths detected.")

                st.markdown("### ⚠ Resume Weaknesses")

                if weaknesses:
                    for w in weaknesses:
                        st.write("- ", w)
                else:
                    st.write("No major weaknesses detected.")

    else:
        st.warning(
            "Please provide both a job description and at least one resume."
        )


st.sidebar.markdown("---")

st.sidebar.info(
    "Tip: The 'Missing Keywords' section helps you identify what skills or certifications should be added to a resume to better align with the job."
)
