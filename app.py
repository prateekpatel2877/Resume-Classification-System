import streamlit as st
import joblib
import pdfplumber
import docx

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="centered"
)

# ---------------- SAFE, SCOPED CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #1e3a8a, #0f172a);
}

/* Main card */
.card {
    background-color: #ffffff;
    padding: 40px;
    border-radius: 18px;
    max-width: 760px;
    margin: 60px auto;
    box-shadow: 0 25px 60px rgba(0,0,0,0.35);
}

/* Title */
.card .title {
    font-size: 36px;
    font-weight: 700;
    color: #0f172a;
    text-align: center;
}

/* Subtitle */
.card .subtitle {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section headers */
.card .section {
    font-size: 17px;
    font-weight: 600;
    color: #1e293b;
    margin-top: 25px;
    margin-bottom: 10px;
}

/* Result box */
.card .result {
    background-color: #e0f2fe;
    color: #0369a1;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    margin-top: 25px;
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5f5;
    margin-top: 25px;
    font-size: 14px;
}

/* IMPORTANT: scoped widget styling only */
.card textarea,
.card input {
    background-color: #f8fafc;
}

/* Hide Streamlit default footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    svm_model = joblib.load("svm_resume_classifier.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return svm_model, tfidf

svm_model, tfidf = load_model()

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    document = docx.Document(file)
    return "\n".join([para.text for para in document.paragraphs])

# ---------------- UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>📄 Resume Classification System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Machine Learning & NLP based automatic resume classification</div>",
    unsafe_allow_html=True
)

# -------- FILE UPLOAD (TOP) --------
st.markdown("<div class='section'>Upload Resume (PDF / DOCX)</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload resume",
    type=["pdf", "docx"],
    label_visibility="collapsed"
)

# -------- TEXT INPUT (BOTTOM) --------
st.markdown("<div class='section'>Paste Resume Text (Optional)</div>", unsafe_allow_html=True)

resume_text_input = st.text_area(
    label="Paste resume text",
    height=150,
    placeholder="Paste resume content here...",
    label_visibility="collapsed"
)

# ---------------- PREDICTION ----------------
resume_text = ""

if uploaded_file:
    if uploaded_file.name.lower().endswith(".pdf"):
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

elif resume_text_input.strip():
    resume_text = resume_text_input

if resume_text.strip():
    resume_vector = tfidf.transform([resume_text])
    prediction = svm_model.predict(resume_vector)[0]

    st.markdown(
        f"<div class='result'>✅ Predicted Role: {prediction}</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>Powered by Linear SVM & TF-IDF | NLP Resume Classification Deployment</div>",
    unsafe_allow_html=True
)
