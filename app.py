
import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#lodading models

clf= pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

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

#web app
def main():
    st.title("Resume Screening App")
    uploaded_file=st.file_uploader("Upload Resume ", type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')


        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

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

        category_name= category_mapping.get(prediction_id,'Unknown')

        st.write("Predicted Category: ",  category_name)


    

#python main
if __name__ == "__main__":
    main()