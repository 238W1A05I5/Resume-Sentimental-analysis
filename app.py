import streamlit as st
import joblib
from io import BytesIO
from docx import Document
import PyPDF2

# Load model once (cache)
@st.cache_resource
def load_model():
    return joblib.load("resume_sentiment_model.pkl")

model = load_model()

def extract_text_from_txt(file):
    return file.read().decode('utf-8', errors='ignore')

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    fullText = [para.text for para in doc.paragraphs]
    return "\n".join(fullText)

def extract_text(file, file_type):
    if file_type == "text/plain":
        return extract_text_from_txt(file)
    elif file_type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    else:
        raise ValueError("Unsupported file type")

def is_resume_text(text):
    # Add more keywords if you want
    keywords = [
        'experience', 'education', 'skills', 'projects', 'certification',
        'summary', 'objective', 'internship', 'award', 'contact', 'profile'
    ]
    text_lower = text.lower()
    # Require at least 2 keywords to reduce false positives
    count = sum(keyword in text_lower for keyword in keywords)
    return count >= 2

st.title("📄 Resume Sentiment Analysis")

uploaded_file = st.file_uploader("Upload your resume file (.txt, .pdf, .docx)", type=['txt', 'pdf', 'docx'])

if uploaded_file is not None:
    try:
        # Convert to BytesIO for PyPDF2 and docx compatibility
        file_bytes = BytesIO(uploaded_file.getvalue())

        text = extract_text(file_bytes, uploaded_file.type)

        if not text.strip():
            st.error("❌ Could not extract any text from the uploaded file. Please try another file.")
        elif not is_resume_text(text):
            st.error("❌ The uploaded file does not appear to be a valid resume. Please upload a proper resume file.")
        else:
            st.subheader("Extracted Text Preview:")
            st.write(text[:1000] + ("..." if len(text) > 1000 else ""))

            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            confidence = max(probabilities) * 100

            st.success(f"Predicted sentiment: **{prediction.capitalize()}** with confidence {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a resume file to analyze.")
