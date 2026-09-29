
import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re

st.set_page_config(page_title="Resume Screening App", page_icon="📄")

#lodading models
@st.cache_resource
def load_models():
    with open('clf.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return clf, tfidf

clf, tfidf = load_models()

def cleanResume(txt):
    # remove urls
    cleanText = re.sub(r'http\S+', ' ', txt)
    # remove RT/cc
    cleanText = re.sub(r'\bRT\b|\bcc\b', ' ', cleanText)
    # remove hashtags
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    # remove email addresses
    cleanText = re.sub(r'\S+@\S+', ' ', cleanText)
    # remove mentions (@something but not inside words)
    cleanText = re.sub(r'@\w+', ' ', cleanText)
    # remove special characters/punctuations
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    # remove non-ASCII characters
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    # remove extra spaces
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()

    return cleanText

def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(uploaded_file)
        return ' '.join(page.extract_text() or '' for page in reader.pages)
    if name.endswith('.docx'):
        document = docx.Document(uploaded_file)
        return '\n'.join(p.text for p in document.paragraphs)
    data = uploaded_file.read()
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('latin-1')

# Map category ID to category name
category_mapping = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing'
}

#web app
def main():
    st.title("📄 Resume Screening App")
    st.write("Upload a resume and the NLP model (TF-IDF + scikit-learn) predicts which of 25 job categories it fits best.")
    uploaded_file = st.file_uploader("Upload Resume", type=['pdf', 'docx', 'txt'])

    if uploaded_file is not None:
        try:
            resume_text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
            return

        cleaned_resume = cleanResume(resume_text)
        if not cleaned_resume:
            st.warning("No readable text found in this file. Scanned/image-only PDFs are not supported.")
            return

        prediction_id = clf.predict(tfidf.transform([cleaned_resume]))[0]
        category_name = category_mapping.get(prediction_id, 'Unknown')

        st.success(f"Predicted Category: **{category_name}**")
        with st.expander("Show extracted text"):
            st.write(resume_text[:3000])

    st.caption("Built by Mohd Anas · [GitHub](https://github.com/anaschouhan001/Resume-Screening-App)")


#python main
if __name__ == "__main__":
    main()
