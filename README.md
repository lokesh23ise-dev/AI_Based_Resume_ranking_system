# 📄 AI Resume Ranking System

An AI-powered application that ranks multiple resumes against a specific Job Description (JD) using Natural Language Processing (NLP). This tool is designed to streamline the screening process by providing objective similarity scores.

## 🚀 Features
* **Multi-PDF Support:** Upload and process multiple resumes simultaneously.
* **Text Extraction:** Automated text parsing from PDF files.
* **Similarity Scoring:** Uses **TF-IDF Vectorization** and **Cosine Similarity** to calculate match percentages.
* **Interactive Dashboard:** A clean, user-friendly interface built with Streamlit.

## 🛠️ Technology Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Libraries:** Scikit-learn, PyPDF2

## 📊 How it Works
1. **Input:** User provides a Job Description and uploads resumes.
2. **Vectorization:** The system converts text into numerical vectors using TF-IDF.
3. **Calculation:** It computes the Cosine Similarity between the JD and each resume:
   $$\text{Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
4. **Ranking:** Results are sorted from highest to lowest match percentage.

## 📦 Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/lokesh23ise-dev/AI_Based_Resume_ranking_system](https://github.com/lokesh23ise-dev/AI_Based_Resume_ranking_system)


**#Install Dependinces**
pip install -r requirements.txt

**##Run the Application**
streamlit run streamlit_app.py
