import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SecureMail - Email Classifier",
    page_icon="📧",
    layout="centered"
)

# -------------------- PROFESSIONAL CSS --------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #1f2937, #111827);
}

.main {
    background-color: transparent;
}

.company-title {
    text-align: center;
    font-size: 45px;
    font-weight: 700;
    color: white;
}

.tagline {
    text-align: center;
    font-size: 18px;
    color: #d1d5db;
    margin-bottom: 40px;
}

.card {
    background-color: white;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}

.stTextArea textarea {
    border-radius: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    border: none;
}

.footer {
    text-align: center;
    font-size: 14px;
    color: #9ca3af;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<p class="company-title">SecureMail</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">AI Powered Email Spam Detection System</p>', unsafe_allow_html=True)

# -------------------- NLTK --------------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# -------------------- TEXT CLEANING --------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------------------- LOAD MODEL --------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------- MAIN CARD --------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    input_text = st.text_area("Enter Email Content:", height=150)

    if st.button("Analyze Email"):
        if input_text.strip() == "":
            st.warning("Please enter email content.")
        else:
            transformed_sms = transform_text(input_text)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            st.write("")
            st.write("### Analysis Result")

            if result == 1:
                st.error("⚠️ This email is classified as SPAM.")
            else:
                st.success("✔ This email is SAFE.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown('<p class="footer">© 2026 SecureMail Technologies | All Rights Reserved</p>', unsafe_allow_html=True)
