import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ---------------- TEXT PREPROCESS FUNCTION ----------------
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

# ---------------- LOAD MODEL & VECTORIZER ----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Email Spam Filter", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #4b0082, #8a2be2);
}
.main {
    background-color: rgba(255,255,255,0.05);
}
.big-title {
    font-size: 48px;
    font-weight: bold;
    color: white;
    text-align: center;
}
.subtitle {
    font-size: 20px;
    color: #e0d7ff;
    text-align: center;
    margin-bottom: 40px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
.spam {
    background-color: #ff4b5c;
    color: white;
}
.not-spam {
    background-color: #00c897;
    color: white;
}
.stButton>button {
    background-color: #8a2be2;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 25px;
}
.stTextArea textarea {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER SECTION ----------------
st.markdown('<div class="big-title">Email Spam Filter</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">This system intelligently analyzes email content and automatically classifies messages as spam or legitimate, similar to the spam filtering feature available in Gmail.</div>', unsafe_allow_html=True)

st.write("")

# ---------------- INPUT SECTION ----------------
input_sms = st.text_area("Enter Email Content Below:", height=200)

# ---------------- PREDICT BUTTON ----------------
if st.button("Analyze Email"):

    if input_sms.strip() == "":
        st.warning("Please enter email content to analyze.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.write("")

        if result == 1:
            st.markdown('<div class="result-box spam">⚠ Spam Email Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">✔ Legitimate Email</div>', unsafe_allow_html=True)
